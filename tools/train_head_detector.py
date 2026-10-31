"""
train_head_detector.py — fine-tune a one-class YOLO11 head detector on the
SCUT-HEAD export produced by ``tools/scut_head_export.py``.

The resulting weights are written to ``models/head/scut_head_yolo11n.pt`` and
are what the desktop app's "Head occupancy (SCUT-HEAD)" mode loads.

GPU defaults (RTX 3050 6 GB)
----------------------------
``--preset auto`` selects the ``6gb`` profile when CUDA is available:
640 px, batch 16, 6 workers, RAM label/image cache, AMP on, and nominal batch
64 (Ultralytics accumulates gradients when the physical batch is smaller).
If the GPU still runs out of memory, the script halves the batch and retries,
so a 6 GB card degrades instead of failing.  Use ``--preset 4gb`` for heavier
concurrent desktop use, or ``--preset cpu`` for the old CPU path.

Examples:
    python tools/train_head_detector.py --preset 6gb --epochs 50
    python tools/train_head_detector.py --preset 4gb --epochs 50 --batch 8
    python tools/train_head_detector.py --preset cpu --fraction 0.2 --epochs 8
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "datasets" / "SCUT_HEAD" / "yolo" / "data.yaml"
DEFAULT_MODEL = ROOT / "backend" / "yolo11n.pt"
DEFAULT_OUTPUT = ROOT / "models" / "head" / "scut_head_yolo11n.pt"

try:  # package import (python -m tools.train_head_detector)
    from backend.runtime import (
        cuda_memory_mb,
        describe_device,
        empty_cache,
        optimize_cuda,
        resolve_device,
        suggest_workers,
        yolo_device,
    )
except ImportError:  # direct script import (python tools/train_head_detector.py)
    import sys

    sys.path.insert(0, str(ROOT))
    from backend.runtime import (  # type: ignore[no-redef]
        cuda_memory_mb,
        describe_device,
        empty_cache,
        optimize_cuda,
        resolve_device,
        suggest_workers,
        yolo_device,
    )


@dataclass(frozen=True)
class TrainProfile:
    imgsz: int
    batch: int
    workers: int
    cache: str | bool
    amp: bool


PROFILES: dict[str, TrainProfile] = {
    # 640 px is native for YOLO11n.  Batch 16 with AMP fits comfortably in
    # 6 GB; nbs=64 makes Ultralytics accumulate 4 micro-batches.
    "6gb": TrainProfile(imgsz=640, batch=16, workers=6, cache="ram", amp=True),
    # Lower resolution and a smaller batch for concurrent desktop/UI use.
    "4gb": TrainProfile(imgsz=512, batch=8, workers=4, cache="ram", amp=True),
    # Historical CPU path: no AMP, no workers (Windows spawn cost dominates
    # when the model itself is the bottleneck), disk cache to save RAM.
    "cpu": TrainProfile(imgsz=640, batch=16, workers=0, cache="disk", amp=False),
}


def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _pick_profile(name: str, device) -> TrainProfile:
    if name == "auto":
        name = "6gb" if device.type == "cuda" else "cpu"
    return PROFILES[name]


def _parse_cache(value: str | None, default: str | bool) -> str | bool:
    if value is None:
        return default
    if value in {"none", "false", "0"}:
        return False
    if value in {"ram", "disk"}:
        return value
    if value in {"true", "1", "auto"}:
        return "ram"
    raise SystemExit(f"unknown --cache value: {value!r}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a YOLO head detector on SCUT-HEAD."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--preset", choices=["auto", *PROFILES], default="auto",
        help="hardware profile; explicit flags override its values",
    )
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument(
        "--nbs", type=int, default=64,
        help="nominal batch for gradient accumulation (default: 64)",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, ...")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--cache", choices=["auto", "ram", "disk", "none"], default=None,
        help="dataset cache; 'ram' avoids repeated JPEG decode",
    )
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=None,
        help="automatic mixed precision (default: on for CUDA profiles)",
    )
    parser.add_argument(
        "--fraction", type=float, default=1.0,
        help="fraction of the training split to use (default: 1.0)",
    )
    parser.add_argument(
        "--freeze", type=int, default=0,
        help="freeze the first N layers (10 is a useful CPU shortcut)",
    )
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument(
        "--cos-lr", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument("--project", type=Path, default=ROOT / "models" / "head")
    parser.add_argument("--name", default="scut_head_yolo11n")
    parser.add_argument("--output-model", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--summary", type=Path, default=None,
        help="training-summary JSON path (default: output model directory, "
             "or the run directory when --no-copy is used)",
    )
    parser.add_argument(
        "--no-val", action="store_true",
        help="skip per-epoch validation (faster; final val still runs)",
    )
    parser.add_argument(
        "--no-copy", action="store_true",
        help="do not copy best.pt to --output-model",
    )
    parser.add_argument(
        "--no-oom-retry", action="store_true",
        help="fail instead of halving --batch after a CUDA OOM",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data_path = args.data.resolve()
    if not data_path.is_file():
        raise SystemExit(
            f"dataset YAML not found: {data_path}\n"
            "Run: python tools/scut_head_export.py"
        )
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise SystemExit(f"starting weights not found: {model_path}")

    from ultralytics import YOLO
    import torch

    device = resolve_device(args.device)
    optimize_cuda(device)
    profile = _pick_profile(args.preset, device)

    imgsz = args.imgsz if args.imgsz is not None else profile.imgsz
    batch = args.batch if args.batch is not None else profile.batch
    workers = (
        args.workers if args.workers is not None else suggest_workers(profile.workers)
    )
    cache = _parse_cache(args.cache, profile.cache)
    amp = profile.amp if args.amp is None else bool(args.amp)

    started = time.time()
    print(f"[head-train] data={data_path}")
    print(f"[head-train] model={model_path}")
    print(f"[head-train] device={describe_device(device)}")
    print(
        f"[head-train] preset={args.preset} epochs={args.epochs} "
        f"imgsz={imgsz} batch={batch} nbs={args.nbs} workers={workers} "
        f"cache={cache} amp={amp} fraction={args.fraction} "
        f"freeze={args.freeze}"
    )

    train_kwargs = dict(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=imgsz,
        batch=batch,
        device=yolo_device(device),
        workers=workers,
        nbs=args.nbs,
        amp=amp,
        cache=cache,
        fraction=args.fraction,
        freeze=args.freeze,
        patience=args.patience,
        optimizer=args.optimizer,
        cos_lr=args.cos_lr,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
        val=not args.no_val,
        plots=False,
        seed=args.seed,
        deterministic=True,
        single_cls=True,
        verbose=True,
    )
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0

    # --- Train with an OOM-safe batch fallback ---------------------------
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    attempt = 0
    current_batch = batch
    while True:
        attempt += 1
        model = YOLO(str(model_path))
        try:
            model.train(**{**train_kwargs, "batch": current_batch})
            break
        except RuntimeError as exc:
            if (
                not _is_oom(exc)
                or args.no_oom_retry
                or attempt >= 3
                or current_batch <= 1
            ):
                raise
            new_batch = max(1, current_batch // 2)
            print(
                f"[head-train] CUDA OOM at batch={current_batch}; retrying at "
                f"batch={new_batch} (attempt {attempt + 1}/3)"
            )
            del model
            gc.collect()
            empty_cache(device)
            time.sleep(2.0)
            current_batch = new_batch

    trainer = getattr(model, "trainer", None)
    best = Path(getattr(trainer, "best", "") or "")
    if not best.is_file() and trainer is not None:
        best = Path(trainer.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit("training finished but best.pt could not be located")

    copied = ""
    if not args.no_copy:
        args.output_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, args.output_model)
        copied = str(args.output_model.resolve())

    elapsed = time.time() - started
    memory = cuda_memory_mb(device)
    summary = {
        "best_checkpoint": str(best.resolve()),
        "output_model": copied,
        "elapsed_seconds": round(elapsed, 1),
        "epochs": args.epochs,
        "preset": args.preset,
        "imgsz": imgsz,
        "batch": current_batch,
        "requested_batch": batch,
        "gradient_accumulation_nbs": args.nbs,
        "workers": workers,
        "cache": cache,
        "amp": amp,
        "fraction": args.fraction,
        "freeze": args.freeze,
        "device": describe_device(device),
        "vram_peak_mb": memory["peak"],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "data": str(data_path),
    }
    summary_path = args.summary
    if summary_path is None:
        if copied:
            summary_path = args.output_model.parent / "train.summary.json"
        else:
            summary_path = Path(trainer.save_dir) / "train.summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[head-train] complete")
    print(f"  best checkpoint : {best}")
    if copied:
        print(f"  app weights     : {copied}")
    print(f"  elapsed         : {elapsed / 60:.1f} min")
    if memory["peak"]:
        print(f"  peak VRAM       : {memory['peak']:.0f} MiB")
    print(f"  summary         : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
