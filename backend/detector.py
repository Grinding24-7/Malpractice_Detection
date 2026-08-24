"""
detector.py — Pose estimation + per-candidate heuristics.

Pipeline:
    1. YOLO11n-pose runs at 5 FPS (sub-sampled from 30 FPS stream) in
       Ultralytics ByteTrack tracking mode, so every detected person keeps a
       stable candidate_id across frames.
    2. Keypoint heuristics flag posture anomalies (head-down, body-turn, lean).
    3. @torch.no_grad wrappers + explicit gc prevent memory leaks.
    4. Research hooks for ST-GCN, PnP head pose, and object detection.

Sub-sampling: every 6th frame (30 FPS / 5 FPS = 6).
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants — YOLO11n-pose keypoint indices (COCO 17-keypoint skeleton)
# ---------------------------------------------------------------------------
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


@dataclass
class InferenceResult:
    keypoints: np.ndarray  # shape (N, 17, 3) — x, y, confidence
    boxes: np.ndarray      # shape (N, 4) — xyxy
    confidences: np.ndarray  # shape (N,) — detection confidence per person
    tracker_ids: np.ndarray  # shape (N,) — ByteTrack persistent candidate_ids
    timestamps: float      # time.monotonic() of inference
    anomaly_flags: dict[str, bool] = field(default_factory=dict)


@dataclass
class PoseDetector:
    """
    Wraps YOLO11n-pose with ByteTrack tracking + baseline heuristics.

    Latency target: < 15 ms per inference call on CPU (WSL).
    """

    model: Any = None
    _frame_counter: int = 0
    subsample_rate: int = 6  # process every 6th frame (30 / 5)

    def __post_init__(self) -> None:
        self._load_model()

    def _load_model(self) -> None:
        """Lazy-load YOLO11n-pose.  Import inside method so the module
        can be imported without the dependency installed yet."""
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolo11n-pose.pt")
            # Warm-up: run a dummy forward pass to trigger CUDA/CPU init
            _ = self.model(
                np.zeros((640, 640, 3), dtype=np.uint8), verbose=False
            )
        except ImportError:
            raise RuntimeError(
                "ultralytics not installed.  Run: uv pip install ultralytics"
            )

    @torch.no_grad()
    def _infer(self, frame: np.ndarray) -> InferenceResult:
        """Run YOLO11n-pose in ByteTrack tracking mode.  Decorated with
        @torch.no_grad to prevent autograd graph accumulation (memory leak
        prevention).  `persist=True` keeps candidate_ids stable across frames."""
        results = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml", verbose=False
        )
        r = results[0]

        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            tracker_ids = r.boxes.id.cpu().numpy().astype(np.int64)
        else:
            boxes = np.empty((0, 4))
            tracker_ids = np.empty((0,), dtype=np.int64)
        confidences = (
            r.boxes.conf.cpu().numpy()
            if r.boxes is not None and r.boxes.conf is not None
            else np.empty((0,))
        )
        kpts = (
            r.keypoints.data.cpu().numpy()
            if r.keypoints is not None
            else np.empty((0, 17, 3))
        )

        return InferenceResult(
            keypoints=kpts,
            boxes=boxes,
            confidences=confidences,
            tracker_ids=tracker_ids,
            timestamps=time.monotonic(),
        )

    def should_process(self) -> bool:
        """Sub-sample: return True only every `subsample_rate`-th call."""
        self._frame_counter += 1
        return (self._frame_counter % self.subsample_rate) == 0

    def track(self, frame: np.ndarray) -> InferenceResult:
        """
        Run tracked inference + heuristics on a single frame.

        Unlike `process()`, this always runs (no sub-sampling) so the caller
        (e.g. app.py) controls its own sampling cadence.
        """
        result = self._infer(frame)
        result.anomaly_flags = self._run_heuristics(result)
        return result

    def process(self, frame: np.ndarray) -> Optional[InferenceResult]:
        """
        Run tracked inference + heuristics on a single frame.

        Called by the detector thread (non-blocking w.r.t. the video reader).
        Returns None if the frame is skipped (sub-sampling).
        """
        if not self.should_process():
            return None

        result = self._infer(frame)
        result.anomaly_flags = self._run_heuristics(result)
        return result

    # ------------------------------------------------------------------
    # Baseline heuristics
    # ------------------------------------------------------------------

    def _run_heuristics(self, result: InferenceResult) -> dict[str, bool]:
        """
        Evaluate lightweight posture rules.

        Returns a dict of boolean anomaly flags consumed by main.py.
        """
        flags: dict[str, bool] = {
            "head_down": False,
            "body_turn": False,
            "excessive_lean": False,
            "multi_person": False,
        }

        if result.keypoints.shape[0] == 0:
            return flags

        # --- Head-down detection ---
        # Heuristic: nose y > shoulder midpoint y  (head below shoulders)
        for person in result.keypoints:
            nose_conf = person[NOSE, 2]
            ls_conf = person[LEFT_SHOULDER, 2]
            rs_conf = person[RIGHT_SHOULDER, 2]
            if nose_conf > 0.5 and ls_conf > 0.3 and rs_conf > 0.3:
                nose_y = person[NOSE, 1]
                shoulder_y = (person[LEFT_SHOULDER, 1] + person[RIGHT_SHOULDER, 1]) / 2.0
                if nose_y > shoulder_y + 20.0:  # pixels below shoulders
                    flags["head_down"] = True

        # --- Body-turn detection ---
        # Heuristic: shoulder midpoint x deviates from nose x beyond threshold
        for person in result.keypoints:
            nose_conf = person[NOSE, 2]
            ls_conf = person[LEFT_SHOULDER, 2]
            rs_conf = person[RIGHT_SHOULDER, 2]
            if nose_conf > 0.5 and ls_conf > 0.3 and rs_conf > 0.3:
                nose_x = person[NOSE, 0]
                shoulder_mid_x = (person[LEFT_SHOULDER, 0] + person[RIGHT_SHOULDER, 0]) / 2.0
                if abs(nose_x - shoulder_mid_x) > 40.0:
                    flags["body_turn"] = True

        # --- Excessive lean ---
        # Heuristic: shoulder-to-hip angle far from vertical
        for person in result.keypoints:
            ls_conf = person[LEFT_SHOULDER, 2]
            rs_conf = person[RIGHT_SHOULDER, 2]
            lh_conf = person[LEFT_HIP, 2]
            rh_conf = person[RIGHT_HIP, 2]
            if all(c > 0.3 for c in (ls_conf, rs_conf, lh_conf, rh_conf)):
                shoulder_mid = (
                    (person[LEFT_SHOULDER, 0] + person[RIGHT_SHOULDER, 0]) / 2.0,
                    (person[LEFT_SHOULDER, 1] + person[RIGHT_SHOULDER, 1]) / 2.0,
                )
                hip_mid = (
                    (person[LEFT_HIP, 0] + person[RIGHT_HIP, 0]) / 2.0,
                    (person[LEFT_HIP, 1] + person[RIGHT_HIP, 1]) / 2.0,
                )
                # dx / dy approximates lean angle
                dy = shoulder_mid[1] - hip_mid[1]
                dx = shoulder_mid[0] - hip_mid[0]
                if dy > 0 and abs(dx / dy) > 0.3:
                    flags["excessive_lean"] = True

        # --- Multi-person ---
        if result.keypoints.shape[0] > 1:
            flags["multi_person"] = True

        return flags

    # ------------------------------------------------------------------
    # RESEARCH HOOK A:  Spatio-Temporal Graph Convolutional Network (ST-GCN)
    # ------------------------------------------------------------------
    # Placeholder for future integration:
    #   - ST-GCN takes a skeleton sequence (T x V x C) over a temporal window.
    #   - Captures multi-student interaction (passing papers, looking at
    #     neighbour's screen).
    #   - Replace _run_heuristics() with an ST-GCN forward pass when
    #     labelled interaction data becomes available.
    # ------------------------------------------------------------------
    def _hook_stgcn(self, skeleton_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            skeleton_sequence: (T, V, C) where
                T = temporal window (e.g. 30 frames at 5 FPS = 6 s),
                V = 17 keypoints,
                C = 3 (x, y, confidence) or 2 (x, y) normalised.

        Returns:
            anomaly_logits: (num_classes,) tensor (e.g. [normal, interaction]).
        """
        # TODO: Load pretrained ST-GCN checkpoint → forward pass.
        # TODO: Implement graph adjacency for COCO skeleton topology.
        raise NotImplementedError("ST-GCN hook — future research integration.")

    # ------------------------------------------------------------------
    # RESEARCH HOOK B:  Perspective-n-Point (PnP) 3D Head Pose
    # ------------------------------------------------------------------
    # Placeholder for future integration:
    #   - Facial landmarks (nose, eyes, mouth corners) → solvePnP
    #     with a generic 3D face model.
    #   - Yaw / pitch / roll angles → detect gaze direction toward
    #     another student's screen.
    #   - Significantly more robust than the 2D nose-to-shoulder heuristic.
    # ------------------------------------------------------------------
    def _hook_pnp_head_pose(
        self, image_points: np.ndarray
    ) -> tuple[float, float, float]:
        """
        Args:
            image_points: (N, 2) array of 2D facial landmarks.

        Returns:
            (yaw, pitch, roll) in degrees.
        """
        # TODO:
        #   model_points = np.array([
        #       (0.0, 0.0, 0.0),       # Nose tip
        #       (0.0, -30.0, -10.0),   # Left eye
        #       ...
        #   ])
        #   _, rvec, tvec = cv2.solvePnP(model_points, image_points,
        #                                 camera_matrix, dist_coeffs)
        #   yaw, pitch, roll = cv2.Rodrigues(rvec) ...
        raise NotImplementedError("PnP head pose — future research integration.")

    # ------------------------------------------------------------------
    # RESEARCH HOOK C:  Dynamic Object Detection (phone / cheat sheet)
    # ------------------------------------------------------------------
    # Placeholder for future integration:
    #   - Triggered *only* when posture anomaly is detected (reduce FP).
    #   - Runs a lightweight object detector (e.g. YOLO11n) on a tight
    #     crop around the student's desk region.
    #   - Classes: cell phone, book, cheat sheet, smartwatch.
    # ------------------------------------------------------------------
    def _hook_object_detection(
        self, frame: np.ndarray, person_box: np.ndarray
    ) -> list[dict[str, Any]]:
        """
        Args:
            frame: full-resolution frame (H, W, 3).
            person_box: (4,) xyxy bounding box of the anomalous person.

        Returns:
            List of detections: [{"label": str, "confidence": float, "box": xyxy}].
        """
        # TODO:
        #   crop = frame[person_box[1]:person_box[3], person_box[0]:person_box[2]]
        #   dets = self.obj_model(crop)
        #   filter by class_id in {phone, book, cheat_sheet}
        raise NotImplementedError("Object detection hook — future research integration.")