"""
console_widget.py — Embedded GUI log drawer.

Captures stdout, stderr, and Python logging messages.
Color-coded: INFO=white, WARNING=yellow, ERROR=bright red.
"""

from __future__ import annotations

from datetime import datetime

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QTextEdit, QVBoxLayout, QWidget


class ConsoleWidget(QWidget):
    """Collapsible log drawer with color-coded messages."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Consolas", 9))
        self.log_view.setStyleSheet(
            "QTextEdit { background-color: #0d1117; color: #c9d1d9; "
            "border: 1px solid #30363d; padding: 4px; }"
        )
        layout.addWidget(self.log_view)

        self._COLORS = {
            "INFO": "#c9d1d9",
            "WARNING": "#d29922",
            "ERROR": "#ff6b6b",
            "CRITICAL": "#ff0000",
            "DEBUG": "#8b949e",
        }

    def append_message(self, level: str, text: str) -> None:
        """Slot: receives (level, text) from StreamRedirect."""
        color = self._COLORS.get(level, "#c9d1d9")
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        html = (
            f'<span style="color:#6e7681;">[{ts}]</span> '
            f'<span style="color:{color};"><b>[{level}]</b> '
            f"{text}</span>"
        )
        self.log_view.append(html)
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear(self) -> None:
        self.log_view.clear()
