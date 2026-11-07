"""
main_window.py — Main PyQt6 application window.

Layout:
    ┌──────────────────────────────────────────────┐
    │  Menu Bar (File | View | Help)                │
    ├──────────────────────────────────────────────┤
    │  QStackedWidget                               │
    │  ┌────────────────────────────────────────┐   │
    │  │ Page 0: HubWidget (Session Setup)      │   │
    │  │ Page 1: Monitoring Dashboard           │   │
    │  │   ┌──────────────┬─────────────────┐   │   │
    │  │   │ VIDEO WIDGET │  SIDEBAR PANEL  │   │   │
    │  │   └──────────────┴─────────────────┘   │   │
    │  └────────────────────────────────────────┘   │
    ├──────────────────────────────────────────────┤
    │  CONSOLE DRAWER (collapsible QDockWidget)     │
    └──────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QMainWindow,
    QMenuBar,
    QStackedWidget,
    QWidget,
)

from desktop_app.core.camera_thread import CameraThread
from desktop_app.core.head_detector import HeadCountWorker, resolve_head_model
from desktop_app.core.inference_worker import InferenceWorker
from desktop_app.ui.hub_widget import HubWidget
from desktop_app.ui.video_widget import VideoWidget
from desktop_app.ui.sidebar_panel import SidebarPanel
from desktop_app.ui.console_widget import ConsoleWidget

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Primary application window with Hub → Proctoring stacked layout."""

    # Page indices for QStackedWidget
    _PAGE_HUB = 0
    _PAGE_MONITORING = 1

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Exam Malpractice Detection — P01-P40 HUD")
        self.setMinimumSize(1100, 700)
        self.resize(1400, 850)

        # Thread references (created on launch)
        self._camera: CameraThread | None = None
        self._worker: InferenceWorker | HeadCountWorker | None = None
        self._threads_running = False
        self._sidebar_timer: QTimer | None = None
        self._last_sidebar_result: dict | None = None

        self._candidate_name = ""
        self._exam_code = ""

        # ── Stacked widget ───────────────────────────────────────────────
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        # Page 0: Hub
        self.hub = HubWidget()
        self.hub.launch_session.connect(self._on_launch_session)
        self._stack.addWidget(self.hub)  # index 0

        # Page 1: Monitoring dashboard
        monitoring_page = QWidget()
        mon_layout = QHBoxLayout(monitoring_page)
        mon_layout.setContentsMargins(4, 4, 4, 4)

        self.video_widget = VideoWidget()
        mon_layout.addWidget(self.video_widget, stretch=3)

        self.sidebar = SidebarPanel()
        mon_layout.addWidget(self.sidebar, stretch=0)

        self._stack.addWidget(monitoring_page)  # index 1

        # ── Console drawer (bottom, collapsible) ─────────────────────────
        self.console = ConsoleWidget()
        self._console_dock = QDockWidget("Console Log")
        self._console_dock.setWidget(self.console)
        self._console_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._console_dock)
        self._console_dock.hide()

        # ── Menu bar ─────────────────────────────────────────────────────
        self._build_menu()

        # ── Apply global dark stylesheet ─────────────────────────────────
        self.setStyleSheet(self._DARK_STYLESHEET)

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        menu_bar: QMenuBar = self.menuBar()

        # File
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction("E&xit", self.close)

        # View
        view_menu = menu_bar.addMenu("&View")
        toggle_console = QAction("Toggle &Console", self)
        toggle_console.setCheckable(True)
        toggle_console.toggled.connect(self._toggle_console)
        view_menu.addAction(toggle_console)

        # Help
        help_menu = menu_bar.addMenu("&Help")
        help_menu.addAction("&About", lambda: None)

    def _toggle_console(self, visible: bool) -> None:
        self._console_dock.setVisible(visible)

    def _on_verdict_changed(self, payload: dict) -> None:
        """Log one console line per published verdict transition.

        The worker emits this only after the stability gate, so this handler
        never sees a per-frame movement/frame-delta update.  Neutral posture
        is named on the event when active; it is displayed on the HUD by the
        result path, not logged continuously.
        """
        pose_id = str(payload.get("pose_id") or "")
        pose_name = str(payload.get("pose_name") or "")
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        seat_id = str(payload.get("seat_id") or "")
        neutral_label = str(payload.get("neutral_label") or "")
        depth_m = payload.get("candidate_depth_m")
        seat_part = f" {seat_id}" if seat_id else ""
        depth_part = (
            f" depth={float(depth_m):.2f}m"
            if isinstance(depth_m, (int, float)) else ""
        )

        if payload.get("source_dead"):
            logger.warning("[NO_SIGNAL] Camera feed lost — monitoring stopped")
        elif payload.get("cleared"):
            if neutral_label:
                logger.info(
                    "[P00] Neutral%s: %s (no alert)%s",
                    seat_part, neutral_label, depth_part,
                )
            else:
                logger.info("[CLEAR]%s returned to P01 Normal%s",
                            seat_part, depth_part)
        else:
            escalation = payload.get("object_escalation") or {}
            label = escalation.get("label") if isinstance(escalation, dict) else ""
            label_part = f" object={label}" if label else ""
            logger.info(
                "[VERDICT]%s %s %s (%.2f)%s%s",
                seat_part, pose_id, pose_name, confidence,
                label_part, depth_part,
            )

    def _on_classroom_initialized(self, info: dict) -> None:
        """Log the SCUT-HEAD seat count and update the window title."""
        count = int(info.get("seated_count", 0) or 0)
        if info.get("locked"):
            logger.info(
                "[INIT] Classroom Seated Count: %d Students Locked", count
            )
            self.setWindowTitle(
                f"Exam Monitoring — {count} Students Locked (SCUT-HEAD)"
            )
        else:
            logger.warning(
                "[INIT] Classroom seats not locked: %s",
                info.get("error") or "no stable heads",
            )

    # ------------------------------------------------------------------
    # Session launch (Hub → Monitoring transition)
    # ------------------------------------------------------------------

    def _on_launch_session(self, candidate_name: str, exam_code: str) -> None:
        """Stop hub camera, create fresh threads, wire signals, switch page."""
        self._candidate_name = candidate_name
        self._exam_code = exam_code

        # Stop hub camera if running
        self.hub.cleanup()

        # Create fresh camera thread.  The hub resolves the source: an image
        # replay list/folder, an RTSP URL, or the selected webcam index.
        source = self.hub._get_camera_source()
        self._camera = CameraThread(source=source)

        # Analysis mode: head occupancy (SCUT-HEAD labels) or pose rules.
        # A head-mode launch without weights falls back to pose rather than
        # showing an empty "0 heads" screen for a model that never loaded.
        mode = self.hub.detection_mode()
        self._worker = None
        if mode == "head":
            head_path = resolve_head_model()
            if head_path is None:
                logger.warning(
                    "Head mode selected but no head weights found; "
                    "falling back to P01-P38 pose rules"
                )
            else:
                logger.info("Head occupancy mode using %s", head_path)
                self._worker = HeadCountWorker(
                    camera=self._camera, model_path=head_path
                )
        if self._worker is None:
            self._worker = InferenceWorker(camera=self._camera)

        # ── Signal wiring ────────────────────────────────────────────────
        # Camera → FPS counter for both modes.  Pose mode also shows raw
        # frames for smoothness; head mode waits for processed HUD frames so
        # dataset replay does not flicker between raw and annotated images.
        # Capture the worker explicitly: shutdown sets self._worker = None
        # before the camera thread has necessarily stopped, and a late
        # raw_frame must not dereference None.
        capture_worker = self._worker
        self._camera.raw_frame.connect(
            lambda _frame, _worker=capture_worker: _worker.tick_capture_fps()
        )
        if not isinstance(self._worker, HeadCountWorker):
            self._camera.raw_frame.connect(self.video_widget.set_frame)
        else:
            self.video_widget.setText("Loading head detector...")
        # Camera status
        self._camera.status_update.connect(
            lambda msg: logger.info("[Camera] %s", msg)
        )

        # Inference → UI.  update_result dispatches head counts vs pose rules.
        # Pose results are polled from the worker's lock-free snapshot slot by
        # a ~60 Hz timer; the Qt signal remains for head mode and tests.  The
        # previous direct signal made the worker allocate/copy a queued event
        # per frame even when the sidebar could not consume it.
        self._worker.frame_processed.connect(self.video_widget.set_frame)
        if isinstance(self._worker, HeadCountWorker):
            self._worker.result_ready.connect(self.sidebar.update_result)
        else:
            self._last_sidebar_result = None
            self._sidebar_timer = QTimer(self)
            self._sidebar_timer.setInterval(16)
            self._sidebar_timer.timeout.connect(self._poll_sidebar_result)
            self._sidebar_timer.start()
        self._worker.fps_updated.connect(self.sidebar.update_inference_fps)
        self._worker.capture_fps_updated.connect(self.sidebar.update_capture_fps)

        # Sidebar session info
        self.sidebar.set_session_info(candidate_name, exam_code)

        # ── Mode-specific wiring ─────────────────────────────────────────
        try:
            self.hub.recording_toggled.disconnect()
        except TypeError:
            pass
        if isinstance(self._worker, HeadCountWorker):
            self._worker.status_update.connect(
                lambda msg: logger.info("[Head] %s", msg)
            )
            self.hub.record_checkbox.setChecked(False)
            self.hub.record_checkbox.setEnabled(False)
            self.sidebar.ticker.clear()
            self.setWindowTitle("Exam Monitoring — Head Occupancy (SCUT-HEAD)")
        else:
            self.hub.recording_toggled.connect(self._worker.set_recording)
            # Event-only console telemetry: the old per-state head/neutral
            # handlers are intentionally not connected (they would duplicate
            # or reintroduce high-frequency lines).
            self._worker.verdict_changed.connect(self._on_verdict_changed)
            self._worker.classroom_initialized.connect(
                self._on_classroom_initialized
            )
            self.hub.record_checkbox.setEnabled(True)
            self.setWindowTitle("Exam Malpractice Detection — P01-P40 HUD")

        # ── Start threads ────────────────────────────────────────────────
        self._camera.start()
        self._worker.start()
        self._threads_running = True

        # ── Switch to monitoring page ────────────────────────────────────
        self._stack.setCurrentIndex(self._PAGE_MONITORING)
        logger.info(
            "Proctoring session launched: candidate=%s, exam=%s",
            candidate_name,
            exam_code,
        )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _poll_sidebar_result(self) -> None:
        """Apply the newest pose result without blocking the inference thread."""
        worker = self._worker
        if worker is None or isinstance(worker, HeadCountWorker):
            return
        latest = worker.latest_result()
        if not latest or latest is self._last_sidebar_result:
            return
        self._last_sidebar_result = latest
        self.sidebar.update_result(latest)

    def _stop_threads(self) -> None:
        """Gracefully stop inference then camera threads."""
        timer = getattr(self, "_sidebar_timer", None)
        if timer is not None:
            timer.stop()
            self._sidebar_timer = None
        if self._worker is not None:
            self._worker.stop()
            self._worker = None
        if self._camera is not None:
            self._camera.stop()
            self._camera = None
        self._threads_running = False

    def closeEvent(self, event) -> None:
        self.hub.cleanup()
        if self._threads_running:
            self._stop_threads()
        self.video_widget.cleanup()
        event.accept()

    # ------------------------------------------------------------------
    # Stylesheet
    # ------------------------------------------------------------------

    _DARK_STYLESHEET = """
    QMainWindow {
        background-color: #161b22;
    }
    QWidget {
        background-color: #161b22;
        color: #e6edf3;
        font-family: Segoe UI, Arial, sans-serif;
    }
    QMenuBar {
        background-color: #0d1117;
        color: #c9d1d9;
        border-bottom: 1px solid #30363d;
    }
    QMenuBar::item:selected {
        background-color: #1f6feb;
    }
    QMenu {
        background-color: #161b22;
        color: #c9d1d9;
        border: 1px solid #30363d;
    }
    QMenu::item:selected {
        background-color: #1f6feb;
    }
    QDockWidget {
        color: #c9d1d9;
        font-size: 12px;
    }
    QDockWidget::title {
        background-color: #0d1117;
        padding: 4px;
        border-bottom: 1px solid #30363d;
    }
    QScrollBar:vertical {
        background-color: #0d1117;
        width: 10px;
    }
    QScrollBar::handle:vertical {
        background-color: #30363d;
        min-height: 20px;
        border-radius: 4px;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    """
