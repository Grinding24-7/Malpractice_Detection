"""
hud_renderer.py — Draws COCO skeleton, bounding boxes, gaze direction vectors,
spatial bounding-box vectoring, and status overlays directly onto the OpenCV
BGR frame BEFORE QImage conversion.

Uses YOLO11n-pose 17-keypoint COCO skeleton for bone connections.
All drawing happens in-place on a copy; the original frame is never mutated.

Dual FPS overlay shows both Capture FPS (camera) and Inference FPS (model).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import cv2
import numpy as np

# ── COCO 17-keypoint skeleton connections ──────────────────────────────────
COCO_SKELETON: list[tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# ── Colour palette (BGR) ──────────────────────────────────────────────────
_COLOR_SKELETON  = (0, 255, 204)   # cyan
_COLOR_JOINTS    = (0, 204, 255)   # amber
_COLOR_BOX_OK    = (0, 255, 0)     # green
_COLOR_BOX_ALERT = (0, 0, 255)     # red
_COLOR_PANEL_BG  = (20, 20, 20)
_COLOR_PANEL_BD  = (80, 80, 80)
_COLOR_TEXT_W    = (255, 255, 255)
_COLOR_TEXT_G    = (0, 200, 0)
_COLOR_TEXT_R    = (0, 0, 255)
_COLOR_TEXT_CYAN = (255, 200, 0)
_COLOR_TEXT_AMBER = (0, 180, 255)
_COLOR_TEXT_DIM  = (140, 140, 140)
_COLOR_GAZE     = (0, 255, 255)   # yellow for gaze vector
_COLOR_SPATIAL  = (200, 160, 0)   # teal for spatial vectors


def _safe_int(val: float) -> int:
    return max(0, int(val))


def draw_skeleton(
    overlay: np.ndarray,
    keypoints: np.ndarray,
    conf_thr: float = 0.35,
) -> None:
    """Draw COCO skeleton connections and joint dots for a single person."""
    for i, j in COCO_SKELETON:
        if i < len(keypoints) and j < len(keypoints):
            if keypoints[i][2] > conf_thr and keypoints[j][2] > conf_thr:
                pt1 = (_safe_int(keypoints[i][0]), _safe_int(keypoints[i][1]))
                pt2 = (_safe_int(keypoints[j][0]), _safe_int(keypoints[j][1]))
                cv2.line(overlay, pt1, pt2, _COLOR_SKELETON, 2, cv2.LINE_AA)

    for kpt in keypoints:
        if kpt[2] > conf_thr:
            cx, cy = _safe_int(kpt[0]), _safe_int(kpt[1])
            cv2.circle(overlay, (cx, cy), 5, _COLOR_JOINTS, -1, cv2.LINE_AA)
            cv2.circle(overlay, (cx, cy), 5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_gaze_vector(
    overlay: np.ndarray,
    landmarks: list[tuple[float, float, float]],
    gaze_endpoint: tuple[int, int] | None,
) -> None:
    """Draw a 3D nose vector line extending outward indicating gaze direction."""
    if not landmarks or len(landmarks) < 33 or gaze_endpoint is None:
        return

    nose_x = _safe_int(landmarks[0][0])
    nose_y = _safe_int(landmarks[0][1])
    end_x, end_y = gaze_endpoint

    # Draw the gaze ray
    cv2.arrowedLine(
        overlay,
        (nose_x, nose_y),
        (end_x, end_y),
        _COLOR_GAZE,
        2,
        cv2.LINE_AA,
        tipLength=0.15,
    )
    # Draw a small circle at the gaze target
    cv2.circle(overlay, (end_x, end_y), 6, _COLOR_GAZE, 2, cv2.LINE_AA)


def draw_spatial_vectors(
    overlay: np.ndarray,
    boxes: list[list[int]],
    frame_shape: tuple[int, ...],
) -> None:
    """Draw spatial bounding-box vectoring indicators.

    Shows:
    - Frame center crosshair
    - Primary candidate center offset vector
    - Multi-person proximity lines
    """
    fh, fw = frame_shape[:2]
    center_x, center_y = fw // 2, fh // 2

    # Frame center crosshair (subtle)
    cv2.drawMarker(
        overlay,
        (center_x, center_y),
        _COLOR_SPATIAL,
        cv2.MARKER_CROSS,
        15,
        1,
        cv2.LINE_AA,
    )

    if not boxes:
        return

    # Primary candidate (largest bbox)
    primary = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    pcx = (primary[0] + primary[2]) // 2
    pcy = (primary[1] + primary[3]) // 2

    # Offset vector from frame center to candidate center
    cv2.line(
        overlay,
        (center_x, center_y),
        (pcx, pcy),
        _COLOR_SPATIAL,
        1,
        cv2.LINE_AA,
    )
    # Candidate center dot
    cv2.circle(overlay, (pcx, pcy), 4, _COLOR_SPATIAL, -1, cv2.LINE_AA)

    # Multi-person proximity lines
    if len(boxes) > 1:
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                ci = ((boxes[i][0] + boxes[i][2]) // 2, (boxes[i][1] + boxes[i][3]) // 2)
                cj = ((boxes[j][0] + boxes[j][2]) // 2, (boxes[j][1] + boxes[j][3]) // 2)
                cv2.line(overlay, ci, cj, (0, 0, 255), 2, cv2.LINE_AA)
                # Proximity label
                mid = ((ci[0] + cj[0]) // 2, (ci[1] + cj[1]) // 2)
                cv2.putText(
                    overlay, "PROXIMITY", (mid[0] - 40, mid[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA,
                )


def render_hud(
    frame: np.ndarray,
    result: dict[str, Any],
    capture_fps: float,
    inference_fps: float,
) -> np.ndarray:
    """Draw the full HUD overlay onto *frame* and return the annotated copy."""
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    pose_id: str = result.get("pose_id", "P01")
    # NO_SIGNAL and INITIALIZING are system states, not proctoring findings.
    # They must not be styled or reported as alerts, or the operator reads a
    # camera fault as a verdict against a student.
    is_system_state = pose_id in ("NO_SIGNAL", "INITIALIZING")
    is_anomaly = pose_id != "P01" and not is_system_state

    # ── 1. Spatial bounding-box vectoring ─────────────────────────────────
    boxes = result.get("bounding_boxes", [])
    spatial = result.get("spatial_features", {})
    draw_spatial_vectors(overlay, boxes, (h, w))

    # ── 2. Bounding boxes + skeleton + gaze ───────────────────────────────
    all_kpts = result.get("keypoints", [])
    gaze_endpoint = result.get("gaze_endpoint")
    landmarks = result.get("landmarks", [])
    first_landmarks = landmarks[0] if landmarks else []

    for person_idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        box_color = _COLOR_BOX_ALERT if is_anomaly else _COLOR_BOX_OK

        cv2.rectangle(
            overlay,
            (_safe_int(x1), _safe_int(y1)),
            (_safe_int(x2), _safe_int(y2)),
            box_color, 2,
        )

        conf_pct = result.get("confidence", 0) * 100
        label = f"[{pose_id}] Candidate {person_idx + 1} (Conf: {conf_pct:.0f}%)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        lbl_x = _safe_int(x1)
        lbl_y = max(th + 8, _safe_int(y1) - 8)
        cv2.rectangle(
            overlay,
            (lbl_x, lbl_y - th - 6),
            (lbl_x + tw + 8, lbl_y + 4),
            box_color, -1,
        )
        cv2.putText(
            overlay, label, (lbl_x + 4, lbl_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COLOR_TEXT_W, 2, cv2.LINE_AA,
        )

        # Skeleton
        if person_idx < len(all_kpts) and all_kpts[person_idx] is not None:
            kpts = all_kpts[person_idx]
            if isinstance(kpts, np.ndarray) and kpts.ndim == 2 and kpts.shape[0] == 17:
                draw_skeleton(overlay, kpts)

        # Gaze vector (only for primary candidate)
        if person_idx == 0 and first_landmarks:
            draw_gaze_vector(overlay, first_landmarks, gaze_endpoint)

    # ── 3. Status panel (top-left) — dual FPS + head pose ─────────────────
    panel_w, panel_h = 420, 220
    cv2.rectangle(overlay, (5, 5), (5 + panel_w, 5 + panel_h), _COLOR_PANEL_BG, -1)
    cv2.rectangle(overlay, (5, 5), (5 + panel_w, 5 + panel_h), _COLOR_PANEL_BD, 1)

    status_color = _COLOR_TEXT_CYAN if is_system_state else (
        _COLOR_TEXT_R if is_anomaly else _COLOR_TEXT_G
    )
    y0 = 28
    cv2.putText(
        overlay, f"Pose: {pose_id} - {result.get('pose_name', '')}",
        (14, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA,
    )
    # "Confidence" is meaningless for a system state, so it is hidden rather
    # than rendered as 0%, which reads as a failed detection.
    if not is_system_state:
        cv2.putText(
            overlay, f"Confidence: {result.get('confidence', 0):.0%}",
            (14, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COLOR_TEXT_W, 1,
            cv2.LINE_AA,
        )
    cv2.putText(
        overlay, f"Capture FPS:  {capture_fps:.1f}",
        (14, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COLOR_TEXT_CYAN, 1, cv2.LINE_AA,
    )
    cv2.putText(
        overlay, f"Inference FPS: {inference_fps:.1f}",
        (14, y0 + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COLOR_TEXT_AMBER, 1, cv2.LINE_AA,
    )

    hp = result.get("head_pose", {})
    cv2.putText(
        overlay,
        f"Yaw: {hp.get('yaw', 0.0):.1f}  Pitch: {hp.get('pitch', 0.0):.1f}  Roll: {hp.get('roll', 0.0):.1f}",
        (14, y0 + 97), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
    )

    # Temporal / sustained anomaly indicator
    temporal = result.get("temporal_metrics", {})
    sustained = int(temporal.get("sustained_anomaly_frames", 0))
    if sustained > 0:
        cv2.putText(
            overlay, f"Sustained Anomaly: {sustained} frames",
            (14, y0 + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (0, 120, 255) if sustained > 15 else _COLOR_TEXT_DIM, 1, cv2.LINE_AA,
        )

    cv2.putText(
        overlay, f"Persons: {len(boxes)}   Model: YOLO11n-pose",
        (14, y0 + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _COLOR_TEXT_DIM, 1, cv2.LINE_AA,
    )

    # Object detections (phone, book, laptop)
    objects = result.get("objects_detected", [])
    if objects:
        obj_labels = ", ".join(
            f"{o['label']}({o['confidence']:.0%})" for o in objects[:3]
        )
        cv2.putText(
            overlay, f"Objects: {obj_labels}",
            (14, y0 + 165), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (0, 100, 255), 1, cv2.LINE_AA,
        )

    cv2.putText(
        overlay, f"Time: {datetime.now().strftime('%H:%M:%S')}",
        (14, y0 + 185), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _COLOR_TEXT_DIM, 1, cv2.LINE_AA,
    )

    # ── 4. Alert banner ───────────────────────────────────────────────────
    if pose_id == "NO_SIGNAL":
        # A camera fault is not a proctoring alert, so it gets its own banner.
        # The wording matters: it must be impossible to read this as a verdict
        # about a student.
        banner = "!! CAMERA FAULT: NO IMAGE - ROOM NOT MONITORED !!"
        (tw, _), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 3)
        bx = max(10, (w - tw) // 2)
        cv2.rectangle(overlay, (bx - 10, 10), (bx + tw + 10, 55), (60, 60, 60), -1)
        cv2.putText(
            overlay, banner, (bx, 42),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, _COLOR_TEXT_AMBER, 3, cv2.LINE_AA,
        )
    elif pose_id == "INITIALIZING":
        banner = "STARTING UP - ROOM NOT YET ASSESSED"
        (tw, _), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 3)
        bx = max(10, (w - tw) // 2)
        cv2.rectangle(overlay, (bx - 10, 10), (bx + tw + 10, 55), (50, 70, 90), -1)
        cv2.putText(
            overlay, banner, (bx, 42),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, _COLOR_TEXT_CYAN, 3, cv2.LINE_AA,
        )
    elif is_anomaly:
        banner = f"!! ALERT: {pose_id} - {result.get('pose_name', '')} !!"
        (tw, _), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)
        bx = max(10, (w - tw) // 2)
        cv2.rectangle(overlay, (bx - 10, 10), (bx + tw + 10, 55), (0, 0, 180), -1)
        cv2.putText(
            overlay, banner, (bx, 42),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, _COLOR_TEXT_W, 3, cv2.LINE_AA,
        )

    # ── 4b. Object detection boxes ────────────────────────────────────────
    # Colour encodes the spatial relation, because "a phone exists" and "the
    # student is holding a phone" are different claims and the operator needs
    # to see which one is being made.  A remembered (persisted) detection is
    # drawn dashed-dim so a stale box is never read as a live one.
    objects = result.get("objects_detected", [])
    for obj in objects:
        obj_box = obj.get("box", [])
        if len(obj_box) != 4:
            continue
        ox1, oy1, ox2, oy2 = map(_safe_int, obj_box)
        relation = obj.get("relation", "away")
        if relation in ("in_hand", "on_desk", "reachable"):
            color = (0, 60, 255)      # red: within reach, actionable
        elif relation == "lap":
            color = (0, 165, 255)     # orange: near the student
        else:
            color = (140, 140, 140)   # grey: present but not attributable

        persisted = obj.get("age_frames", 0) > 0
        thickness = 1 if persisted else 2
        cv2.rectangle(overlay, (ox1, oy1), (ox2, oy2), color, thickness)
        label = f"{obj['label']} {obj['confidence']:.0%} [{relation}]"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(oy1 - 6, 12)
        cv2.rectangle(
            overlay, (ox1, label_y - 11), (ox1 + tw + 4, label_y + 2),
            (20, 20, 20), -1,
        )
        cv2.putText(
            overlay, label, (ox1 + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )

    # ── 5. Recording indicator ────────────────────────────────────────────
    if result.get("recording"):
        cv2.circle(overlay, (w - 30, 30), 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            overlay, "REC", (w - 65, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA,
        )

    return overlay
