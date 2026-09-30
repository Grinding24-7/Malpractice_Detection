"""
hub_widget.py — Intermediate Hub screen displayed before starting a
proctoring inspection.

Provides session setup (candidate name, exam code, camera device selector,
RTSP/CCTV stream URL input), camera toggle controls with live preview,
and a "Launch Proctoring Session" button that transitions the stacked
widget to the live monitoring dashboard.
"""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from desktop_app.core.camera_thread import CameraThread


class _RtspTestWorker(QThread):
    """Background thread to test RTSP connectivity without freezing UI."""

    finished = pyqtSignal(bool, str)  # (success, message)

    def __init__(self, url: str, parent=None) -> None:
        super().__init__(parent)
        self._url = url

    def run(self) -> None:
        ok = CameraThread.probe_rtsp(self._url, timeout=5)
        if ok:
            self.finished.emit(True, "Stream reachable")
        else:
            self.finished.emit(False, "Stream unreachable — check URL and credentials")


class HubWidget(QWidget):
    """Session setup hub with camera controls and live preview."""

    camera_started = pyqtSignal()
    camera_stopped = pyqtSignal()
    launch_session = pyqtSignal(str, str)  # (candidate_name, exam_code)
    recording_toggled = pyqtSignal(bool)  # (enabled)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._camera: CameraThread | None = None
        self._camera_running = False
        self._rtsp_test_worker: _RtspTestWorker | None = None
        self._build_ui()
        self._refresh_devices()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 30, 40, 30)
        root.setSpacing(16)

        # ── Title ────────────────────────────────────────────────────────
        title = QLabel("IntelliProctor — Session Setup")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "color: #58a6ff; font-size: 22px; font-weight: bold; "
            "font-family: Segoe UI, sans-serif; margin-bottom: 8px;"
        )
        root.addWidget(title)

        subtitle = QLabel("Configure your proctoring session before launch")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(
            "color: #8b949e; font-size: 12px; margin-bottom: 16px;"
        )
        root.addWidget(subtitle)

        # ── Form fields ──────────────────────────────────────────────────
        form = QHBoxLayout()
        form.setSpacing(20)

        # Candidate Name
        name_col = QVBoxLayout()
        name_col.setSpacing(4)
        name_lbl = QLabel("Candidate Name")
        name_lbl.setStyleSheet(self._LABEL_STYLE)
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g. John Doe")
        self.name_input.setStyleSheet(self._INPUT_STYLE)
        name_col.addWidget(name_lbl)
        name_col.addWidget(self.name_input)
        form.addLayout(name_col)

        # Exam Code
        code_col = QVBoxLayout()
        code_col.setSpacing(4)
        code_lbl = QLabel("Exam Code")
        code_lbl.setStyleSheet(self._LABEL_STYLE)
        self.code_input = QLineEdit()
        self.code_input.setPlaceholderText("e.g. CS301-MID")
        self.code_input.setStyleSheet(self._INPUT_STYLE)
        code_col.addWidget(code_lbl)
        code_col.addWidget(self.code_input)
        form.addLayout(code_col)

        # Camera Device (webcam dropdown)
        cam_col = QVBoxLayout()
        cam_col.setSpacing(4)
        cam_lbl = QLabel("Camera Device")
        cam_lbl.setStyleSheet(self._LABEL_STYLE)
        self.device_combo = QComboBox()
        self.device_combo.setStyleSheet(self._INPUT_STYLE)
        self.device_combo.setMinimumWidth(180)
        cam_col.addWidget(cam_lbl)
        cam_col.addWidget(self.device_combo)
        form.addLayout(cam_col)

        root.addLayout(form)

        # ── RTSP / CCTV Stream URL ───────────────────────────────────────
        rtsp_row = QHBoxLayout()
        rtsp_row.setSpacing(12)

        rtsp_col = QVBoxLayout()
        rtsp_col.setSpacing(4)
        rtsp_lbl = QLabel("RTSP / CCTV Stream URL")
        rtsp_lbl.setStyleSheet(self._LABEL_STYLE)
        self.rtsp_input = QLineEdit()
        self.rtsp_input.setPlaceholderText("rtsp://user:password@192.168.1.100:554/stream")
        self.rtsp_input.setStyleSheet(self._INPUT_STYLE)
        self.rtsp_input.setMinimumWidth(320)
        rtsp_col.addWidget(rtsp_lbl)
        rtsp_col.addWidget(self.rtsp_input)
        rtsp_row.addLayout(rtsp_col, stretch=1)

        self.test_rtsp_btn = QPushButton("Test Connection")
        self.test_rtsp_btn.setStyleSheet(self._SECONDARY_BTN_STYLE)
        self.test_rtsp_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.test_rtsp_btn.clicked.connect(self._on_test_rtsp)
        rtsp_row.addWidget(self.test_rtsp_btn, alignment=Qt.AlignmentFlag.AlignBottom)

        root.addLayout(rtsp_row)

        # ── Camera toggle + Refresh ──────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        btn_row.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.toggle_btn = QPushButton("  START CAMERA  ")
        self.toggle_btn.setStyleSheet(self._TOGGLE_BTN_STYLE)
        self.toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_btn.clicked.connect(self._on_toggle_camera)
        btn_row.addWidget(self.toggle_btn)

        refresh_btn = QPushButton("Refresh Devices")
        refresh_btn.setStyleSheet(self._SECONDARY_BTN_STYLE)
        refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        refresh_btn.clicked.connect(self._refresh_devices)
        btn_row.addWidget(refresh_btn)

        root.addLayout(btn_row)

        # ── Recording toggle ─────────────────────────────────────────────
        rec_row = QHBoxLayout()
        rec_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.record_checkbox = QCheckBox("Record Training Samples")
        self.record_checkbox.setStyleSheet(
            "QCheckBox { color: #8b949e; font-size: 12px; spacing: 8px; }"
            "QCheckBox::indicator { width: 16px; height: 16px; }"
        )
        self.record_checkbox.toggled.connect(self.recording_toggled.emit)
        rec_row.addWidget(self.record_checkbox)
        root.addLayout(rec_row)

        # ── Camera preview ───────────────────────────────────────────────
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(480, 300)
        self.preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.preview_label.setStyleSheet(
            "background-color: #0d1117; border: 2px solid #30363d; "
            "border-radius: 8px; color: #484f58; font-size: 14px;"
        )
        self.preview_label.setText("Camera preview will appear here")
        root.addWidget(self.preview_label, stretch=1)

        # ── Launch button ────────────────────────────────────────────────
        self.launch_btn = QPushButton("Launch Proctoring Session")
        self.launch_btn.setStyleSheet(self._LAUNCH_BTN_STYLE)
        self.launch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.launch_btn.setEnabled(False)
        self.launch_btn.clicked.connect(self._on_launch)
        root.addWidget(self.launch_btn)

        # ── Status ───────────────────────────────────────────────────────
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(
            "color: #8b949e; font-size: 11px; margin-top: 4px;"
        )
        root.addWidget(self.status_label)

        # ── Connect RTSP input ↔ webcam dropdown exclusivity ─────────────
        self.rtsp_input.textChanged.connect(self._on_rtsp_text_changed)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)

    # ------------------------------------------------------------------
    # RTSP ↔ Webcam mutual exclusion
    # ------------------------------------------------------------------

    def _on_rtsp_text_changed(self, text: str) -> None:
        """When RTSP URL is typed, disable webcam dropdown."""
        has_rtsp = bool(text.strip())
        self.device_combo.setEnabled(not has_rtsp)

    def _on_device_changed(self, index: int) -> None:
        """When a webcam is selected, clear RTSP field."""
        if index >= 0:
            self.rtsp_input.clear()

    # ------------------------------------------------------------------
    # RTSP connectivity test
    # ------------------------------------------------------------------

    def _on_test_rtsp(self) -> None:
        url = self.rtsp_input.text().strip()
        if not url:
            self.status_label.setText("Enter an RTSP URL first")
            return
        self.test_rtsp_btn.setEnabled(False)
        self.test_rtsp_btn.setText("Testing...")
        self.status_label.setText("Testing stream connectivity...")
        self._rtsp_test_worker = _RtspTestWorker(url)
        self._rtsp_test_worker.finished.connect(self._on_rtsp_test_result)
        self._rtsp_test_worker.start()

    def _on_rtsp_test_result(self, ok: bool, message: str) -> None:
        self.test_rtsp_btn.setEnabled(True)
        self.test_rtsp_btn.setText("Test Connection")
        self.status_label.setText(message)
        self.status_label.setStyleSheet(
            "color: #3fb950; font-size: 11px; margin-top: 4px;"
            if ok else
            "color: #f85149; font-size: 11px; margin-top: 4px;"
        )

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def _refresh_devices(self) -> None:
        self.device_combo.clear()
        devices = CameraThread.probe_devices()
        if not devices:
            self.device_combo.addItem("No cameras detected", -1)
            self.status_label.setText("No cameras found — use RTSP URL or mock feed")
            self.status_label.setStyleSheet(
                "color: #8b949e; font-size: 11px; margin-top: 4px;"
            )
        else:
            for dev in devices:
                self.device_combo.addItem(dev["name"], dev["index"])
            self.status_label.setText(f"{len(devices)} camera(s) detected")
            self.status_label.setStyleSheet(
                "color: #8b949e; font-size: 11px; margin-top: 4px;"
            )

    # ------------------------------------------------------------------
    # Camera toggle
    # ------------------------------------------------------------------

    def _on_toggle_camera(self) -> None:
        if self._camera_running:
            self._stop_camera()
        else:
            self._start_camera()

    def _get_camera_source(self) -> str | int:
        """Determine camera source from RTSP input or webcam dropdown."""
        rtsp_url = self.rtsp_input.text().strip()
        if rtsp_url:
            return rtsp_url
        data = self.device_combo.currentData()
        return data if isinstance(data, int) else -1

    def _start_camera(self) -> None:
        source = self._get_camera_source()
        self._camera = CameraThread(source=source)
        self._camera.raw_frame.connect(self._on_raw_frame)
        self._camera.status_update.connect(
            lambda msg: self.status_label.setText(msg)
        )
        self._camera.start()
        self._camera_running = True

        self.toggle_btn.setText("  STOP CAMERA  ")
        self.toggle_btn.setStyleSheet(self._TOGGLE_BTN_STOP_STYLE)
        self.device_combo.setEnabled(False)
        self.rtsp_input.setEnabled(False)
        self.test_rtsp_btn.setEnabled(False)
        self.launch_btn.setEnabled(True)
        self.status_label.setText("Camera starting...")
        self.status_label.setStyleSheet(
            "color: #8b949e; font-size: 11px; margin-top: 4px;"
        )
        self.camera_started.emit()

    def _stop_camera(self) -> None:
        if self._camera is not None:
            self._camera.raw_frame.disconnect(self._on_raw_frame)
            self._camera.stop()
            self._camera = None
        self._camera_running = False

        self.toggle_btn.setText("  START CAMERA  ")
        self.toggle_btn.setStyleSheet(self._TOGGLE_BTN_STYLE)
        self.device_combo.setEnabled(True)
        self.rtsp_input.setEnabled(True)
        self.test_rtsp_btn.setEnabled(True)
        self.launch_btn.setEnabled(False)
        self.preview_label.clear()
        self.preview_label.setText("Camera preview will appear here")
        self.status_label.setText("Camera stopped")
        self.status_label.setStyleSheet(
            "color: #8b949e; font-size: 11px; margin-top: 4px;"
        )
        self.camera_stopped.emit()

    def _on_raw_frame(self, frame: np.ndarray) -> None:
        """Display the latest raw frame in the preview QLabel."""
        if frame is None or frame.size == 0:
            return
        h, w, ch = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        if pixmap.isNull():
            return
        scaled = pixmap.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    def _on_launch(self) -> None:
        name = self.name_input.text().strip() or "Unknown Candidate"
        code = self.code_input.text().strip() or "UNKNOWN"
        self.launch_session.emit(name, code)

    # ------------------------------------------------------------------
    # Cleanup (called from MainWindow.closeEvent)
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        if self._camera_running:
            self._stop_camera()

    # ------------------------------------------------------------------
    # Stylesheet constants
    # ------------------------------------------------------------------

    _LABEL_STYLE = (
        "color: #8b949e; font-size: 11px; font-weight: bold; "
        "text-transform: uppercase; letter-spacing: 0.5px;"
    )
    _INPUT_STYLE = (
        "QLineEdit, QComboBox {"
        "  background-color: #0d1117; color: #e6edf3; "
        "  border: 1px solid #30363d; border-radius: 4px; "
        "  padding: 8px 10px; font-size: 13px;"
        "}"
        "QLineEdit:focus, QComboBox:focus {"
        "  border-color: #1f6feb;"
        "}"
        "QComboBox::drop-down {"
        "  border: none; padding-right: 8px;"
        "}"
        "QComboBox QAbstractItemView {"
        "  background-color: #0d1117; color: #e6edf3; "
        "  selection-background-color: #1f6feb; border: 1px solid #30363d;"
        "}"
    )
    _TOGGLE_BTN_STYLE = (
        "QPushButton {"
        "  background-color: #238636; color: #ffffff; "
        "  border: none; border-radius: 6px; "
        "  padding: 10px 28px; font-size: 13px; font-weight: bold;"
        "}"
        "QPushButton:hover { background-color: #2ea043; }"
        "QPushButton:pressed { background-color: #1a7f37; }"
    )
    _TOGGLE_BTN_STOP_STYLE = (
        "QPushButton {"
        "  background-color: #da3633; color: #ffffff; "
        "  border: none; border-radius: 6px; "
        "  padding: 10px 28px; font-size: 13px; font-weight: bold;"
        "}"
        "QPushButton:hover { background-color: #f85149; }"
        "QPushButton:pressed { background-color: #b62324; }"
    )
    _SECONDARY_BTN_STYLE = (
        "QPushButton {"
        "  background-color: #21262d; color: #c9d1d9; "
        "  border: 1px solid #30363d; border-radius: 6px; "
        "  padding: 8px 16px; font-size: 12px;"
        "}"
        "QPushButton:hover { background-color: #30363d; }"
    )
    _LAUNCH_BTN_STYLE = (
        "QPushButton {"
        "  background-color: #1f6feb; color: #ffffff; "
        "  border: none; border-radius: 6px; "
        "  padding: 12px 36px; font-size: 14px; font-weight: bold; "
        "  margin-top: 8px;"
        "}"
        "QPushButton:hover { background-color: #388bfd; }"
        "QPushButton:pressed { background-color: #1158c7; }"
        "QPushButton:disabled {"
        "  background-color: #21262d; color: #484f58; "
        "  cursor: not-allowed;"
        "}"
    )

    # ------------------------------------------------------------------
    # Stylesheet constants
    # ------------------------------------------------------------------

    _LABEL_STYLE = (
        "color: #8b949e; font-size: 11px; font-weight: bold; "
        "text-transform: uppercase; letter-spacing: 0.5px;"
    )
    _INPUT_STYLE = (
        "QLineEdit, QComboBox {"
        "  background-color: #0d1117; color: #e6edf3; "
        "  border: 1px solid #30363d; border-radius: 4px; "
        "  padding: 8px 10px; font-size: 13px;"
        "}"
        "QLineEdit:focus, QComboBox:focus {"
        "  border-color: #1f6feb;"
        "}"
        "QComboBox::drop-down {"
        "  border: none; padding-right: 8px;"
        "}"
        "QComboBox QAbstractItemView {"
        "  background-color: #0d1117; color: #e6edf3; "
        "  selection-background-color: #1f6feb; border: 1px solid #30363d;"
        "}"
    )
    _TOGGLE_BTN_STYLE = (
        "QPushButton {"
        "  background-color: #238636; color: #ffffff; "
        "  border: none; border-radius: 6px; "
        "  padding: 10px 28px; font-size: 13px; font-weight: bold;"
        "}"
        "QPushButton:hover { background-color: #2ea043; }"
        "QPushButton:pressed { background-color: #1a7f37; }"
    )
    _TOGGLE_BTN_STOP_STYLE = (
        "QPushButton {"
        "  background-color: #da3633; color: #ffffff; "
        "  border: none; border-radius: 6px; "
        "  padding: 10px 28px; font-size: 13px; font-weight: bold;"
        "}"
        "QPushButton:hover { background-color: #f85149; }"
        "QPushButton:pressed { background-color: #b62324; }"
    )
    _SECONDARY_BTN_STYLE = (
        "QPushButton {"
        "  background-color: #21262d; color: #c9d1d9; "
        "  border: 1px solid #30363d; border-radius: 6px; "
        "  padding: 8px 16px; font-size: 12px;"
        "}"
        "QPushButton:hover { background-color: #30363d; }"
    )
    _LAUNCH_BTN_STYLE = (
        "QPushButton {"
        "  background-color: #1f6feb; color: #ffffff; "
        "  border: none; border-radius: 6px; "
        "  padding: 12px 36px; font-size: 14px; font-weight: bold; "
        "  margin-top: 8px;"
        "}"
        "QPushButton:hover { background-color: #388bfd; }"
        "QPushButton:pressed { background-color: #1158c7; }"
        "QPushButton:disabled {"
        "  background-color: #21262d; color: #484f58; "
        "  cursor: not-allowed;"
        "}"
    )
