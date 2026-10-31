"""
runtime.py — shared device, AMP and DataLoader policy.

One source of truth for the CUDA target, mixed-precision policy and host/device
transfer settings used by every training and inference entry point.  The
defaults are tuned for a 6 GB laptop GPU (RTX 3050) with a 16 GB host:

    * CUDA is selected explicitly, never inferred from the presence of a
      tensor on some device.
    * FP16 autocast + GradScaler are enabled on CUDA and disabled on CPU.
    * Host workers are capped so image decoding cannot exhaust 16 GB of RAM.
    * Pinned memory and non-blocking copies are enabled only for CUDA.

Environment overrides (all optional):
    IP_TORCH_DEVICE=auto|cuda|cpu
    IP_NUM_WORKERS=<int>
    IP_AMP=0|1
"""

from __future__ import annotations

import gc
import os
from typing import Any

import torch

__all__ = [
    "amp_dtype",
    "autocast",
    "cuda_memory_mb",
    "dataloader_kwargs",
    "describe_device",
    "empty_cache",
    "grad_scaler",
    "is_cuda",
    "optimize_cuda",
    "resolve_device",
    "seed_everything",
    "suggest_workers",
    "yolo_device",
]


def resolve_device(
    requested: str | torch.device | None = None,
) -> torch.device:
    """Return the explicit compute device.

    ``auto`` (the default) resolves to the literal expression
    ``torch.device("cuda" if torch.cuda.is_available() else "cpu")``.
    An explicit ``cuda`` falls back to CPU when CUDA is unavailable so a
    copy of a saved command does not crash on a CPU-only machine.
    """
    if isinstance(requested, torch.device):
        return requested
    value = str(requested or os.environ.get("IP_TORCH_DEVICE", "auto")).strip()
    if value.lower() in {"", "auto", "gpu", "cuda"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def is_cuda(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_available()


def amp_dtype(device: torch.device) -> torch.dtype:
    """FP16 is the fast path on Ampere/Turing tensor cores."""
    return torch.float16 if is_cuda(device) else torch.float32


def amp_enabled(device: torch.device, requested: bool | None = None) -> bool:
    if requested is None:
        requested = os.environ.get("IP_AMP", "1") not in {"0", "false", "False"}
    return bool(requested) and is_cuda(device)


def autocast(device: torch.device, enabled: bool | None = None):
    """Autocast context for CUDA; a no-op context on CPU.

    Usage::

        with autocast(device, enabled=use_amp):
            logits = model(x)
    """
    use_amp = amp_enabled(device, enabled)
    return torch.amp.autocast(
        device_type=device.type,
        dtype=amp_dtype(device),
        enabled=use_amp,
    )


def grad_scaler(device: torch.device, enabled: bool | None = None):
    """Return a GradScaler that is safely disabled on CPU."""
    use_amp = amp_enabled(device, enabled)
    try:
        return torch.amp.GradScaler(device.type, enabled=use_amp)
    except (AttributeError, TypeError):  # torch < 2.3 fallback
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def optimize_cuda(device: torch.device) -> None:
    """Enable the safe global CUDA fast paths once per process."""
    if not is_cuda(device):
        return
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except (AttributeError, RuntimeError):
        pass
    # Ampere-class GPUs execute these multiplications faster in TF32 with no
    # meaningful accuracy loss for detection/sequence models.
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except AttributeError:
        pass


def seed_everything(seed: int = 0) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN costs throughput and is not needed for this project;
    # benchmark=True in optimize_cuda() picks the fastest kernel instead.


def suggest_workers(explicit: int | None = None) -> int:
    """Pick a DataLoader worker count that fits a 16 GB host.

    Windows spawns (rather than forks) workers and each image worker holds its
    own decoder state, so the cap is lower there.  Six workers is enough to
    keep a 6 GB GPU fed without competing with the UI/desktop process.
    """
    if explicit is not None:
        return max(0, int(explicit))
    env = os.environ.get("IP_NUM_WORKERS")
    if env:
        try:
            return max(0, int(env))
        except ValueError:
            pass
    cpu = os.cpu_count() or 4
    ceiling = 6 if os.name == "nt" else 8
    return max(0, min(ceiling, max(2, cpu // 2)))


def dataloader_kwargs(
    device: torch.device,
    num_workers: int | None = None,
    *,
    persistent_workers: bool | None = None,
    prefetch_factor: int = 4,
) -> dict[str, Any]:
    """Return the loader kwargs shared by every custom PyTorch loop."""
    workers = suggest_workers(num_workers)
    kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": is_cuda(device),
    }
    if workers > 0:
        kwargs["persistent_workers"] = (
            True if persistent_workers is None else bool(persistent_workers)
        )
        kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    return kwargs


def yolo_device(device: torch.device) -> str:
    """Ultralytics expects ``"0"``, not ``"cuda"``, to select the first GPU."""
    if is_cuda(device):
        index = device.index if device.index is not None else 0
        return str(index)
    return "cpu"


def describe_device(device: torch.device) -> str:
    if is_cuda(device):
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        return f"cuda:{index} ({props.name}, {props.total_memory / 1024**3:.1f} GB)"
    return "cpu"


def empty_cache(device: torch.device | None = None) -> None:
    """Release cached CUDA blocks and Python garbage after OOM / shutdown."""
    gc.collect()
    if torch.cuda.is_available() and (
        device is None or device.type == "cuda"
    ):
        torch.cuda.empty_cache()


def cuda_memory_mb(device: torch.device) -> dict[str, float]:
    """Current and peak VRAM usage in MiB (zeros on CPU)."""
    if not is_cuda(device):
        return {"allocated": 0.0, "reserved": 0.0, "peak": 0.0}
    return {
        "allocated": round(torch.cuda.memory_allocated(device) / 1024**2, 1),
        "reserved": round(torch.cuda.memory_reserved(device) / 1024**2, 1),
        "peak": round(torch.cuda.max_memory_allocated(device) / 1024**2, 1),
    }
