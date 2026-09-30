"""
video_widget.py — QLabel-based renderer that displays OpenCV BGR frames
converted to QImage → QPixmap with aspect-ratio-preserving scaling.
"""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel


class VideoWidget(QLabel):
    """Displays annotated video frames at native resolution."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(480, 360)
        self.setStyleSheet("background-color: #1a1a2e;")
        self._pixmap: QPixmap | None = None

    def set_frame(self, frame: np.ndarray) -> None:
        """Slot: receives a BGR numpy array and displays it."""
        if frame is None or frame.size == 0:
            return
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        if pixmap.isNull():
            return
        self._pixmap = pixmap
        self._fit_pixmap()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_pixmap()

    def _fit_pixmap(self) -> None:
        if self._pixmap is not None and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)

    def cleanup(self) -> None:
        self._pixmap = None
        self.clear()
