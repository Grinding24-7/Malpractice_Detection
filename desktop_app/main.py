"""
main.py — Entry point for the PyQt6 desktop application.

Launch:
    python desktop_app/main.py

Architecture:
    The HubWidget is shown first (page 0 of QStackedWidget).  Thread
    lifecycle is managed by MainWindow._on_launch_session(), not here.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so backend/ imports work
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from desktop_app.core.console_redirect import StreamRedirect
from desktop_app.ui.main_window import MainWindow

logger = logging.getLogger("desktop_app")


def main() -> int:
    # ── Logging setup ────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ── Torch runtime fast paths ────────────────────────────────────────
    # Apply these once, before any model is constructed.  optimize_cuda()
    # is a no-op on CPU and both switches have environment opt-outs, so a
    # deployment that sees a regression can disable them without a patch.
    try:
        import torch

        from backend.runtime import optimize_cuda, resolve_device
        from desktop_app.core import perf_config
        from desktop_app.core.vision_utils import configure_opencv_threads

        cv_threads = configure_opencv_threads()
        runtime_device = resolve_device(perf_config.INFERENCE_DEVICE)
        optimize_cuda(runtime_device)
        logger.info(
            "Torch runtime: device=%s cudnn_benchmark=%s tf32_matmul=%s "
            "cv_threads=%s",
            runtime_device,
            getattr(torch.backends.cudnn, "benchmark", None),
            getattr(torch.backends.cuda.matmul, "allow_tf32", None),
            cv_threads,
        )
    except Exception as exc:  # never let a fast-path flag block startup
        logger.warning("Torch runtime optimization skipped: %s", exc)

    logger.info("Starting Exam Malpractice Detection — P01-P40 HUD")

    # ── Qt application ───────────────────────────────────────────────────
    app = QApplication(sys.argv)
    app.setApplicationName("IntelliProctor")
    app.setStyle("Fusion")

    # Use the same branded icon that the PyInstaller executable embeds, so
    # the title bar, taskbar and shortcut all agree.
    icon_path = _PROJECT_ROOT / "assets" / "IntelliProctor.ico"
    if icon_path.is_file():
        app.setWindowIcon(QIcon(str(icon_path)))

    # ── Console redirect (stdout/stderr/logging → GUI) ───────────────────
    redirect = StreamRedirect()
    redirect.install()

    # ── Main window (shows Hub page by default) ──────────────────────────
    window = MainWindow()
    redirect.message.connect(window.console.append_message)
    window.show()

    logger.info(
        "Application ready — Launch command: python desktop_app/main.py"
    )

    # ── Event loop ───────────────────────────────────────────────────────
    exit_code = app.exec()

    # ── Clean shutdown ───────────────────────────────────────────────────
    logger.info("Shutting down...")
    window.close()
    redirect.uninstall()
    logger.info("Goodbye.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
