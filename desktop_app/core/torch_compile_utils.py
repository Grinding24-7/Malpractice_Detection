"""Optional, failure-safe ``torch.compile`` for Ultralytics YOLO models.

``torch.compile`` is not free: the first call pays graph capture/compilation,
and on platforms without a supported backend (notably Windows without
Triton) it can fail outright.  A proctoring session must never fail to start
because an optimisation was unavailable.

The helper therefore replaces only the module's ``forward`` with a callable
that compiles lazily on first use, catches both compile-time and run-time
failures, and permanently falls back to the original eager forward for that
model.  Attributes such as ``names``, ``stride`` and ``fuse`` remain on the
untouched Ultralytics module, which is why this is safer than assigning a
compiled ``OptimizedModule`` over ``model.model``.

Disabled by default.  Enable with ``IP_TORCH_COMPILE=1``; ``auto`` enables it
on non-Windows platforms.  ``IP_TORCH_COMPILE_MODE`` selects the mode
(default ``reduce-overhead``).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import torch

logger = logging.getLogger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}


def compile_enabled() -> bool:
    raw = os.environ.get("IP_TORCH_COMPILE", "0").strip().lower()
    if raw == "auto":
        return sys.platform != "win32"
    return raw in _TRUE_VALUES


def compile_mode() -> str:
    return os.environ.get("IP_TORCH_COMPILE_MODE", "reduce-overhead")


class _SafeCompiledForward:
    """Callable installed as ``module.forward``; falls back on any failure."""

    __slots__ = (
        "_module",
        "_original",
        "_label",
        "_compiled",
        "_failed",
        "_mode",
    )

    def __init__(self, module: Any, original: Any, label: str) -> None:
        self._module = module
        self._original = original
        self._label = label
        self._compiled: Any = None
        self._failed = False
        self._mode = compile_mode()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._failed:
            return self._original(*args, **kwargs)

        if self._compiled is None:
            try:
                # Compile the original bound forward, not the module: calling
                # the module would re-enter this wrapper and recurse.
                self._compiled = torch.compile(
                    self._original, mode=self._mode
                )
                logger.info(
                    "torch.compile enabled for %s (mode=%s)",
                    self._label,
                    self._mode,
                )
            except Exception as exc:  # pragma: no cover - platform dependent
                self._failed = True
                logger.warning(
                    "torch.compile unavailable for %s (%s); using eager",
                    self._label,
                    exc,
                )
                return self._original(*args, **kwargs)

        try:
            return self._compiled(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - backend dependent
            self._failed = True
            self._compiled = None
            logger.warning(
                "torch.compile forward failed for %s (%s); "
                "falling back to eager for this model",
                self._label,
                exc,
            )
            return self._original(*args, **kwargs)


def install_safe_compile(yolo: Any, label: str) -> bool:
    """Install lazy compile on *yolo*'s underlying module.

    Returns ``True`` when the wrapper was installed.  Calling this twice on
    the same model is a no-op.
    """
    if yolo is None or not compile_enabled():
        return False
    if not hasattr(torch, "compile"):
        logger.info("torch.compile not present in this torch build")
        return False

    module = getattr(yolo, "model", None)
    if module is None:
        logger.warning("Cannot torch.compile %s: no .model attribute", label)
        return False
    if getattr(module, "_ip_safe_compile_installed", False):
        return False

    original = getattr(module, "forward", None)
    if not callable(original):
        logger.warning("Cannot torch.compile %s: forward is not callable", label)
        return False

    try:
        module.forward = _SafeCompiledForward(module, original, label)
        module._ip_safe_compile_installed = True
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Could not install torch.compile wrapper for %s: %s", label, exc
        )
        return False


__all__ = ["compile_enabled", "compile_mode", "install_safe_compile"]
