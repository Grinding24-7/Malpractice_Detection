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

from PyQt6.QtCore import Qt
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

    def _on_head_state_changed(
        self, pose_id: str, yaw: float, pitch: float, roll: float
    ) -> None:
        """Log head orientation state changes to the console drawer."""
        logger.info(
            "[HeadPose] State change: %s | Yaw=%.1f Pitch=%.1f Roll=%.1f",
            pose_id, yaw, pitch, roll,
        )

    def _on_neutral_state_changed(self, state: str, label: str) -> None:
        """Log neutral posture transitions to the console drawer.

        An empty state means the student left the neutral posture (or the
        camera faulted); either way it ends the previous line's episode.
        """
        if state:
            logger.info("[P00] Candidate #01: %s (Neutral)", label)
        else:
            logger.info("[P00] Neutral state cleared")

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
        self._camera.raw_frame.connect(lambda _: self._worker.tick_capture_fps())
        if not isinstance(self._worker, HeadCountWorker):
            self._camera.raw_frame.connect(self.video_widget.set_frame)
        else:
            self.video_widget.setText("Loading head detector...")
        # Camera status
        self._camera.status_update.connect(
            lambda msg: logger.info("[Camera] %s", msg)
        )

        # Inference → UI.  update_result dispatches head counts vs pose rules.
        self._worker.frame_processed.connect(self.video_widget.set_frame)
        self._worker.result_ready.connect(self.sidebar.update_result)
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
            self._worker.head_state_changed.connect(self._on_head_state_changed)
            self._worker.neutral_state_changed.connect(self._on_neutral_state_changed)
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

    def _stop_threads(self) -> None:
        """Gracefully stop inference then camera threads."""
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
