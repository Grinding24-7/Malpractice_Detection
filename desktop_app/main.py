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

    logger.info("Starting Exam Malpractice Detection — P01-P40 HUD")

    # ── Qt application ───────────────────────────────────────────────────
    app = QApplication(sys.argv)
    app.setApplicationName("Malpractice Detection")
    app.setStyle("Fusion")

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
