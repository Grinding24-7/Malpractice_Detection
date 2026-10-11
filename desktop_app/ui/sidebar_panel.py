"""
sidebar_panel.py — Right-side dashboard panel showing Active Pose,
dual FPS counters (Capture + Inference), 3D head pose angles with
gaze direction, temporal metrics, and a scrolling Anomaly Ticker.
"""

from __future__ import annotations

from datetime import datetime

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class SidebarPanel(QWidget):
    """Right-side panel with live pose info, dual FPS, and anomaly ticker."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(280)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        # ── Header ───────────────────────────────────────────────────────
        header = QLabel("DASHBOARD")
        header.setStyleSheet(
            "color: #868e96; font-size: 12px; font-weight: bold; "
            "text-transform: uppercase; letter-spacing: 1px;"
        )
        layout.addWidget(header)

        # ── Session info ─────────────────────────────────────────────────
        self.session_label = self._value("--", size=11, color="#58a6ff")
        layout.addWidget(self.session_label)

        layout.addSpacing(4)

        # ── Active Pose section ──────────────────────────────────────────
        layout.addWidget(self._section("Active Pose"))
        self.pose_id_label = self._value("--", size=20, bold=True)
        self.pose_name_label = self._value("--", size=13)
        self.confidence_label = self._value("--", size=12, color="#adb5bd")
        layout.addWidget(self.pose_id_label)
        layout.addWidget(self.pose_name_label)
        layout.addWidget(self.confidence_label)

        layout.addSpacing(8)

        # ── 3D Head Pose section ─────────────────────────────────────────
        layout.addWidget(self._section("3D Head Pose"))
        self.yaw_label = self._value("Yaw: 0.0", size=13)
        self.pitch_label = self._value("Pitch: 0.0", size=13)
        self.roll_label = self._value("Roll: 0.0", size=13)
        self.gaze_label = self._value("Gaze: --", size=12, color="#00ffcc")
        layout.addWidget(self.yaw_label)
        layout.addWidget(self.pitch_label)
        layout.addWidget(self.roll_label)
        layout.addWidget(self.gaze_label)

        layout.addSpacing(8)

        # ── Temporal section ─────────────────────────────────────────────
        layout.addWidget(self._section("Temporal Motion"))
        self.yaw_var_label = self._value("Yaw Var: 0.0", size=12, color="#adb5bd")
        self.sustained_label = self._value("Sustained: 0 frames", size=12, color="#adb5bd")
        layout.addWidget(self.yaw_var_label)
        layout.addWidget(self.sustained_label)

        layout.addSpacing(8)

        # ── Performance section (dual FPS) ───────────────────────────────
        layout.addWidget(self._section("Performance"))
        self.capture_fps_label = self._value(
            "Capture FPS: --", size=13, color="#58a6ff"
        )
        self.inference_fps_label = self._value(
            "Inference FPS: --", size=13, color="#ffd43b"
        )
        layout.addWidget(self.capture_fps_label)
        layout.addWidget(self.inference_fps_label)

        layout.addSpacing(8)

        # ── Anomaly Ticker ───────────────────────────────────────────────
        layout.addWidget(self._section("Anomaly Ticker"))
        self.ticker = QTextEdit()
        self.ticker.setReadOnly(True)
        self.ticker.setMaximumHeight(180)
        self.ticker.setStyleSheet(
            "QTextEdit { background-color: #0d1117; color: #ff6b6b; "
            "font-family: Consolas, monospace; font-size: 11px; "
            "border: 1px solid #30363d; padding: 4px; }"
        )
        layout.addWidget(self.ticker)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Public update slots
    # ------------------------------------------------------------------

    def set_session_info(self, candidate_name: str, exam_code: str) -> None:
        self.session_label.setText(f"{candidate_name}  |  {exam_code}")

    def update_pose(self, result: dict) -> None:
        pose_id = result.get("pose_id", "P01")
        # System states (camera fault / startup) and neutral posture states
        # are not anomalies; neutral states get their own green treatment so
        # "writing" never reads as a quiet alert.
        is_system = pose_id in ("NO_SIGNAL", "INITIALIZING")
        neutral_label = result.get("neutral_label") or ""
        is_neutral = bool(neutral_label) and pose_id == "P01"
        is_anomaly = pose_id != "P01" and not is_system
        if is_system:
            color = "#4dabf7"
        elif is_anomaly:
            color = "#ff6b6b"
        else:
            color = "#51cf66"
        self.pose_id_label.setText(f"Pose: {pose_id}")
        self.pose_id_label.setStyleSheet(
            f"color: {color}; font-size: 20px; font-weight: bold; font-family: monospace;"
        )
        if is_neutral:
            # The neutral filter vouched for this frame: name the benign
            # state instead of the generic "Normal Sitting".
            self.pose_name_label.setText(f"{neutral_label} (Neutral)")
        else:
            self.pose_name_label.setText(result.get("pose_name", "--"))
        self.confidence_label.setText(f"Confidence: {result.get('confidence', 0):.0%}")

        # 3D Head Pose
        hp = result.get("head_pose", {})
        yaw = hp.get("yaw", 0.0)
        pitch = hp.get("pitch", 0.0)
        roll = hp.get("roll", 0.0)

        yaw_color = "#ff6b6b" if abs(yaw) > 35 else "#e9ecef"
        pitch_color = "#ff6b6b" if pitch < -25 else "#e9ecef"
        self.yaw_label.setText(f"Yaw: {yaw:.1f}")
        self.yaw_label.setStyleSheet(
            f"color: {yaw_color}; font-size: 13px; font-weight: normal; font-family: Consolas, monospace;"
        )
        self.pitch_label.setText(f"Pitch: {pitch:.1f}")
        self.pitch_label.setStyleSheet(
            f"color: {pitch_color}; font-size: 13px; font-weight: normal; font-family: Consolas, monospace;"
        )
        self.roll_label.setText(f"Roll: {roll:.1f}")

        # Gaze direction
        gaze = result.get("gaze_vector", (0, 0, 1))
        gaze_text = f"Gaze: ({gaze[0]:.2f}, {gaze[1]:.2f}, {gaze[2]:.2f})"
        self.gaze_label.setText(gaze_text)

        # Temporal metrics
        temporal = result.get("temporal_metrics", {})
        yaw_var = temporal.get("yaw_variance", 0.0)
        sustained = int(temporal.get("sustained_anomaly_frames", 0))
        self.yaw_var_label.setText(f"Yaw Var: {yaw_var:.1f}")
        s_color = "#ff6b6b" if sustained > 15 else "#adb5bd"
        self.sustained_label.setText(f"Sustained: {sustained} frames")
        self.sustained_label.setStyleSheet(
            f"color: {s_color}; font-size: 12px; font-weight: normal; font-family: Consolas, monospace;"
        )

    def update_capture_fps(self, fps: float) -> None:
        self.capture_fps_label.setText(f"Capture FPS: {fps:.1f}")

    def update_inference_fps(self, fps: float) -> None:
        self.inference_fps_label.setText(f"Inference FPS: {fps:.1f}")

    def update_anomaly_ticker(self, result: dict) -> None:
        pose_id = result.get("pose_id", "P01")
        # Neutral posture is a non-finding by construction (its verdict is
        # published as P01), and system states are not verdicts at all --
        # neither belongs in a ticker of anomalies.
        if pose_id in ("P01", "NO_SIGNAL", "INITIALIZING"):
            return
        ts = datetime.now().strftime("%H:%M:%S")
        entry = (
            f"[{ts}] {pose_id} - "
            f"{result.get('pose_name', '?')} "
            f"({result.get('confidence', 0):.0%})"
        )
        self.ticker.append(entry)
        sb = self.ticker.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _section(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "color: #868e96; font-size: 10px; font-weight: bold; "
            "text-transform: uppercase; margin-top: 2px;"
        )
        return lbl

    @staticmethod
    def _value(
        text: str,
        size: int = 14,
        bold: bool = False,
        color: str = "#e9ecef",
    ) -> QLabel:
        weight = "bold" if bold else "normal"
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {color}; font-size: {size}px; "
            f"font-weight: {weight}; font-family: Consolas, monospace;"
        )
        return lbl
