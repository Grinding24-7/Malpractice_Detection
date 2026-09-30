"""
console_redirect.py — Redirects stdout, stderr, and Python logging to Qt signals.

Usage:
    redirect = StreamRedirect()
    redirect.install()
    redirect.message.connect(my_widget.append_message)
    # ... later on shutdown ...
    redirect.uninstall()
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime

from PyQt6.QtCore import QObject, pyqtSignal


class _WriteStream:
    """Proxy that wraps a file-like object and emits a Qt signal on write."""

    def __init__(self, signal: pyqtSignal, level: str) -> None:  # type: ignore[valid-type]
        self._signal = signal
        self._level = level
        self._original: object | None = None

    def write(self, text: str) -> None:
        if text and text.strip():
            self._signal.emit(self._level, text.strip())
        if self._original is not None and hasattr(self._original, "write"):
            self._original.write(text)  # type: ignore[union-attr]

    def flush(self) -> None:
        if self._original is not None and hasattr(self._original, "flush"):
            self._original.flush()  # type: ignore[union-attr]

    def isatty(self) -> bool:
        return False


class _QtLogHandler(logging.Handler):
    """logging.Handler that emits records to a Qt signal."""

    def __init__(self, signal: pyqtSignal) -> None:  # type: ignore[valid-type]
        super().__init__()
        self._signal = signal

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = record.levelname
            msg = self.format(record)
            self._signal.emit(level, msg)
        except Exception:
            self.handleError(record)


class StreamRedirect(QObject):
    """Captures stdout, stderr, and root-logger output and forwards them as Qt signals."""

    message = pyqtSignal(str, str)  # (level, text)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._stdout = _WriteStream(self.message, "INFO")
        self._stderr = _WriteStream(self.message, "ERROR")
        self._handler = _QtLogHandler(self.message)
        self._handler.setFormatter(
            logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        )
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        self._stdout._original = sys.stdout
        self._stderr._original = sys.stderr
        sys.stdout = self._stdout  # type: ignore[assignment]
        sys.stderr = self._stderr  # type: ignore[assignment]
        logging.root.addHandler(self._handler)
        if logging.root.level == logging.NOTSET:
            logging.root.setLevel(logging.INFO)
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        sys.stdout = self._stdout._original  # type: ignore[assignment]
        sys.stderr = self._stderr._original  # type: ignore[assignment]
        logging.root.removeHandler(self._handler)
        self._installed = False
