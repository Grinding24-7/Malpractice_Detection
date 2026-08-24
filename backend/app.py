#!/usr/bin/env python3
"""
app.py — Week 3: Multi-Student CCTV tracking with Ultralytics ByteTrack.

Architecture:
    - generate_stream_frames() (thread):
        1. Every raw frame (30 FPS) is appended to per-candidate RAM ring
           buffers (indexed by ByteTrack candidate_id) for currently active
           candidates.
        2. Every 6th frame (5 FPS AI sub-sampling) runs YOLO11-pose in
           ByteTrack tracking mode (persist=True), extracts per-candidate
           normalized pose features, scores heuristics, updates that
           candidate's anomaly counter, and triggers per-candidate evidence
           exports at the alert threshold.
        3. Stale candidates (unseen for > 300 frames / 10 s) are garbage
           collected to bound memory.
    - /api/telemetry reports per-candidate state: [{id, ear_ratio, status}].
"""

import base64
import hashlib
import json
import os
import subprocess
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory, send_file

from dataset_collector import initialize_dataset, log_feature_vector
from detector import PoseDetector
from feature_extractor import extract_normalized_pose_features, extract_pose_features
from retention_policy import start_auto_purge_thread
from temporal_features import (
    HeuristicBaseline,
    PoseWindowManager,
    SequenceDatasetWriter,
    TemporalFeatureExtractor,
    normalize_keypoints,
)

app = Flask(__name__)

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"

VIDEO_SOURCE = os.environ.get("VIDEO_SOURCE", str(BACKEND_DIR / "sample_exam.mp4"))
CAM_INDEX = 0
CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_FPS = 30
SUBSAMPLE = 6  # AI sub-sampling: every 6th frame (30 / 5 FPS)
ANOMALY_THRESHOLD = 10  # sustained anomaly frames before an alert fires
ALERT_COOLDOWN = 10  # seconds between exports for the same candidate
BUFFER_FRAMES = 30 * 30  # per-candidate RAM ring buffer length (30 s @ 30 FPS)
STALE_LOOKBACK_FRAMES = 300  # candidate cleanup after > 10 s unseen

# Week 4: temporal pose-window configuration.
SEQUENCE_LEN = 30  # T pose frames per candidate (~6 s of history at 5 FPS)
TEMPORAL_RECORDING_CHUNK = 32  # labelled sequences buffered before a .pt flush
SEQUENCE_DATASET_PATH = BACKEND_DIR / "sequence_dataset"

# Heuristic thresholds on the scale-normalized features.
HEAD_TURN_EAR_RATIO_MIN = 0.70  # ear_ratio below this => head turned left
HEAD_TURN_EAR_RATIO_MAX = 1.40  # ear_ratio above this => head turned right
PITCH_LEAN_NORM_DROP = 0.90  # norm_vertical_drop above this => leaning/bent

EVIDENCE_DIR = BACKEND_DIR / "evidence_vault"
EVIDENCE_DIR.mkdir(exist_ok=True)

# Week 6 dashboard: classroom registry + dataset media + demo evidence metadata.
CLASSROOMS_PATH = BACKEND_DIR / "classrooms.json"
DATASET_WHITELIST = {"sample_exam.mp4", "multi_classroom_test.mp4", "1_hour_exam_test.mp4"}
CLASSROOM_NAMES = ["Room A — Block 1", "Room B — Block 2", "Room C — Block 3"]

# Week 7 dashboard: offline video upload / analysis jobs.
UPLOAD_DIR = BACKEND_DIR / "upload_jobs"
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_UPLOAD_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_UPLOAD_BYTES = 300 * 1024 * 1024
upload_jobs: dict[str, dict] = {}
upload_jobs_lock = threading.Lock()

POSE_MODEL_PATH = os.environ.get("POSE_MODEL", str(BACKEND_DIR / "yolo11n-pose.pt"))
LABEL_NAMES = {0: "Normal", 1: "Head Turn", 2: "Leaning", 3: "Passing"}

telemetry_state = {
    "active_recording_label": -1,
    "recorded_samples_count": 0,
}
ts_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Per-candidate state (keyed by ByteTrack candidate_id)
# ---------------------------------------------------------------------------
cand_lock = threading.Lock()
candidate_buffers = defaultdict(lambda: deque(maxlen=BUFFER_FRAMES))
candidate_anomaly_counters = defaultdict(int)
candidate_cheating_states = defaultdict(bool)
candidate_last_seen = defaultdict(int)  # last raw-frame index the candidate was active
candidate_last_alert = defaultdict(float)  # time.monotonic() of last export
candidate_features: dict[int, dict] = {}  # id -> latest normalized features
candidate_last_pose: dict[int, tuple] = {}  # id -> (kpts (17,3), box (4,)) for overlay redraw
candidate_body: dict[int, dict] = {}  # id -> {nose:(x,y), shoulder_w, last_v} for velocity_spike
current_active_ids: set[int] = set()  # candidate_ids present in the latest inference
frame_counter = 0  # monotonic raw-frame index driving stale cleanup

# Week 4: per-candidate sliding pose windows + temporal feature extraction.
pose_windows = PoseWindowManager(window_size=SEQUENCE_LEN)
temporal_extractor = TemporalFeatureExtractor(window_size=SEQUENCE_LEN)
temporal_baseline = HeuristicBaseline()
sequence_writer = SequenceDatasetWriter(
    SEQUENCE_DATASET_PATH,
    extractor=temporal_extractor,
    max_pending=TEMPORAL_RECORDING_CHUNK,
)

detector: "PoseDetector | None" = None

# Week 6: a dedicated detector for the webcam `/stream` endpoint so its
# ByteTrack state (stable track ids) is isolated from the CCTV capture thread.
stream_detector: "PoseDetector | None" = None
stream_lock = threading.Lock()


def get_stream_detector() -> "PoseDetector":
    global stream_detector
    if stream_detector is None:
        stream_detector = PoseDetector()
    return stream_detector


def _load_classrooms() -> list:
    if CLASSROOMS_PATH.exists():
        try:
            return json.loads(CLASSROOMS_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save_classrooms(items: list) -> None:
    CLASSROOMS_PATH.write_text(json.dumps(items, indent=2))


# ---------------------------------------------------------------------------
# Offline video upload / analysis pipeline (dashboard Module 1).
# ---------------------------------------------------------------------------

def _job_update(job_id: str, **fields) -> None:
    with upload_jobs_lock:
        job = upload_jobs.get(job_id)
        if job is not None:
            job.update(fields)


def classify_anomaly_type(flags: dict, feats: np.ndarray) -> str | None:
    """Map detector heuristics + pose features to an alert category.

    Order matters: most severe first, so NOTE PASSING > PEEKING > HEAD TURNING.
    Returns None when the frame/person is nominal.
    """
    if not any(flags.values()):
        ear_ratio = float(feats[0])
        norm_vertical_drop = float(feats[1])
        if ear_ratio < HEAD_TURN_EAR_RATIO_MIN or ear_ratio > HEAD_TURN_EAR_RATIO_MAX:
            return "HEAD TURNING"
        if norm_vertical_drop > PITCH_LEAN_NORM_DROP:
            return "PEEKING"
        return None
    if flags.get("multi_person"):
        return "NOTE PASSING"
    if flags.get("head_down") or flags.get("excessive_lean"):
        return "PEEKING"
    if flags.get("body_turn"):
        return "HEAD TURNING"
    return None


TYPE_SEVERITY = {"HEAD TURNING": "medium", "PEEKING": "high", "NOTE PASSING": "high"}


def _build_alert_zones(per_frame: list, fps: float) -> list:
    """Collapse a per-frame alert-type list into merged time zones.

    Zones shorter than ~0.4 s are dropped to suppress single-frame flicker,
    then adjacent zones of the same type are merged.
    """
    zones: list[dict] = []
    current_type: str | None = None
    start_frame = 0
    min_frames = max(3, int(fps * 0.4))

    def flush(end_frame: int) -> None:
        nonlocal current_type, start_frame
        if current_type is not None and (end_frame - start_frame) >= min_frames:
            zones.append(
                {
                    "start": round(start_frame / fps, 2),
                    "end": round(end_frame / fps, 2),
                    "type": current_type,
                    "severity": TYPE_SEVERITY.get(current_type, "medium"),
                }
            )
        current_type = None

    for frame, alert_type in per_frame:
        if alert_type != current_type:
            flush(frame)
            current_type = alert_type
            start_frame = frame
    flush(len(per_frame))

    # Merge adjacent zones of the same type.
    merged: list[dict] = []
    for z in zones:
        if merged and merged[-1]["type"] == z["type"] and z["start"] <= merged[-1]["end"] + 0.05:
            merged[-1]["end"] = z["end"]
        else:
            merged.append(z)
    return merged


def process_upload_job(job_id: str, input_path: Path, output_dir: Path) -> None:
    """Full analysis pipeline for an uploaded test recording.

    Runs the Week 3 tracking (ByteTrack person IDs) + Week 4/5 heuristics per
    frame, writes an H.264-annotated mp4 and a JSON sidecar with the per-frame
    alert log + color-coded timeline zones. Executed on a background thread;
    progress is polled via /api/upload-job/<id>.
    """
    try:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError("could not open uploaded video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _job_update(job_id, status="processing", progress=0, fps=fps, total_frames=total)

        detector = PoseDetector()  # isolated ByteTrack state per job
        per_frame: list = []
        w = h = 0

        # Writer: pipe BGR frames straight into ffmpeg for a browser-safe
        # H.264 mp4 (identical approach to the evidence exporter).
        writer = None

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx == 1:
                h, w = frame.shape[:2]
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "rawvideo",
                    "-pixel_format", "bgr24",
                    "-video_size", f"{w}x{h}",
                    "-framerate", str(round(fps, 3)),
                    "-i", "-",
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    str(output_dir / "annotated.mp4"),
                ]
                writer = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

            result = detector.track(frame)
            alerts = []
            for idx in range(len(result.tracker_ids)):
                cid = int(result.tracker_ids[idx])
                kpts = result.keypoints[idx]
                box = result.boxes[idx]
                conf = (
                    float(result.confidences[idx])
                    if result.confidences.shape[0] > idx
                    else float(kpts[:, 2].max()) if kpts.size else 1.0
                )
                feats = extract_normalized_pose_features(kpts)
                anomaly = evaluate_candidate(feats) or any(result.anomaly_flags.values())
                alert_type = classify_anomaly_type(result.anomaly_flags, feats) if anomaly else None
                if alert_type:
                    alerts.append({"id": cid, "type": alert_type, "conf": round(conf, 3)})
                draw_candidate(frame, kpts, box, cid, anomaly)

            # Frame-level banner when any candidate is anomalous.
            primary = max(alerts, key=lambda a: TYPE_SEVERITY.get(a["type"], "low")) if alerts else None
            if primary:
                color = (0, 0, 255) if primary["type"] != "HEAD TURNING" else (0, 215, 255)
                cv2.rectangle(frame, (0, 0), (w, 46), color, -1)
                cv2.putText(
                    frame, f"ALERT: {primary['type']}",
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2,
                )

            per_frame.append((frame_idx, primary["type"] if primary else None))
            writer.stdin.write(frame.tobytes())

            if frame_idx % 8 == 0:
                _job_update(job_id, progress=round(100 * frame_idx / max(total, 1)))

        writer.stdin.close()
        writer.wait()

        sidecar = {
            "job_id": job_id,
            "status": "done",
            "fps": round(fps, 3),
            "total_frames": frame_idx,
            "duration_s": round(frame_idx / fps, 2),
            "video_url": f"/upload_jobs/{job_id}/annotated.mp4",
            "analysis_url": f"/upload_jobs/{job_id}/analysis.json",
            "zones": _build_alert_zones(per_frame, fps),
            "frames": [
                {"frame": f, "t": round(f / fps, 3), "type": t}
                for f, t in per_frame
            ],
        }
        (output_dir / "analysis.json").write_text(json.dumps(sidecar))
        _job_update(job_id, progress=100, status="done", result=sidecar)
        print(f"[upload] job {job_id} done: {frame_idx} frames, {len(sidecar['zones'])} zone(s)", flush=True)
    except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
        _job_update(job_id, status="error", error=str(exc))
        print(f"[upload] job {job_id} failed: {exc}", flush=True)
    finally:
        input_path.unlink(missing_ok=True)

# ---------------------------------------------------------------------------
# COCO-17 skeleton topology for drawing
# ---------------------------------------------------------------------------
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # face
    (5, 6),  # shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10),  # right arm
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

latest_frame = None
latest_raw_frame = None
frame_lock = threading.Lock()

# Audit: which data source the capture thread resolved to (set by open_capture).
capture_source: dict = {"type": "unknown", "path": None}

telemetry = {"fps": 0.0}
tel_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Feature heuristics
# ---------------------------------------------------------------------------

def evaluate_candidate(features: np.ndarray) -> bool:
    """Return True when one person's normalized pose looks anomalous."""
    ear_ratio = float(features[0])
    norm_vertical_drop = float(features[1])
    nose_conf = float(features[4])
    l_ear_conf = float(features[5])
    r_ear_conf = float(features[6])

    # Skip unreliable keypoints (low-confidence / absent) to avoid false alarms.
    if nose_conf < 0.35 or l_ear_conf < 0.2 or r_ear_conf < 0.2:
        return False

    # Head yaw proxy: nose-to-ear asymmetry deviating from 1.0 => head turn.
    if ear_ratio < HEAD_TURN_EAR_RATIO_MIN or ear_ratio > HEAD_TURN_EAR_RATIO_MAX:
        return True
    # Pitch / lean proxy: eyes dropping toward the shoulder line.
    if norm_vertical_drop > PITCH_LEAN_NORM_DROP:
        return True
    return False


# ---------------------------------------------------------------------------
# Evidence export
# ---------------------------------------------------------------------------

def encode_frame(frame: np.ndarray) -> bytes:
    """JPEG-encode a frame so per-candidate RAM buffers stay memory-bounded."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
    return buf.tobytes() if ok else b""


def export_evidence_job(alert_id: str, candidate_id: int, encoded_frames: list) -> None:
    """Write a candidate's individual RAM buffer to an mp4 clip."""
    frames = [
        cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        for b in encoded_frames
        if b
    ]
    frames = [f for f in frames if f is not None]
    if not frames:
        return
    out_path = EVIDENCE_DIR / f"candidate_{candidate_id}_alert_{alert_id}.mp4"
    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pixel_format", "bgr24",
        "-video_size", f"{w}x{h}",
        "-framerate", "30",
        "-i", "-",
        "-c:v", "libx264",
        "-b:v", "500k",
        "-preset", "fast",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"[export] candidate {candidate_id} -> {out_path.name}", flush=True)


def open_capture():
    global capture_source
    cap = cv2.VideoCapture(CAM_INDEX)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        capture_source = {"type": "webcam", "path": f"device:{CAM_INDEX}"}
        print(f"[capture] webcam device {CAM_INDEX} opened at {CAM_WIDTH}x{CAM_HEIGHT} @ {CAM_FPS}fps", flush=True)
    else:
        print(f"[capture] WARNING: webcam device {CAM_INDEX} unavailable, falling back to {VIDEO_SOURCE}", flush=True)
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        if not cap.isOpened():
            raise RuntimeError("[capture] ERROR: webcam AND video file both failed to open")
        capture_source = {"type": "video_file", "path": VIDEO_SOURCE}
        # Blank-source guard: a black/garbled file yields no detections (and thus
        # no keypoints/overlay). Sample a few frames and warn loudly.
        sampled = [cap.read()[1] for _ in range(8)]
        if any(f is not None for f in sampled) and np.mean([f.mean() for f in sampled if f is not None]) < 5.0:
            print(
                "[capture] WARNING: source frames are effectively BLANK "
                "(mean brightness < 5/255) — ByteTrack will never find a student. "
                f"Set VIDEO_SOURCE to a real exam video (e.g. {BACKEND_DIR / 'sample_exam.mp4'}).",
                flush=True,
            )
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(f"[audit] data source -> {capture_source}", flush=True)
    return cap


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_candidate(frame: np.ndarray, kpts: np.ndarray, box: np.ndarray,
                   candidate_id: int, anomaly: bool) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    color = (0, 0, 255) if anomaly else (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    status = "ANOMALY" if anomaly else "NOMINAL"
    tag = f"ID:{candidate_id} {status}"
    cv2.putText(frame, tag, (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    for (a, b) in SKELETON:
        pa = kpts[a]
        pb = kpts[b]
        if pa[2] > 0.3 and pb[2] > 0.3:
            cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     (255, 0, 255), 2)
    for (x, y, conf) in kpts:
        if conf > 0.3:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)


def redraw_active_overlay(frame: np.ndarray) -> None:
    """Re-render the last known detection overlay on every raw frame.

    Inference only runs on subsampled frames (1 in SUBSAMPLE), so without this
    the box + skeleton would flash on 1 frame out of 6 and effectively not be
    visible to a human viewer. We redraw the most recent pose for each still
    active candidate, dropping poses once they go stale (> 4 s unseen).
    """
    with cand_lock:
        active = list(current_active_ids)
        poses = [
            (cid, kpts, box, candidate_cheating_states.get(cid, False))
            for cid in active
            if cid in candidate_last_pose
            and (frame_counter - candidate_last_seen.get(cid, 0)) <= SUBSAMPLE * 20
            for kpts, box in [candidate_last_pose[cid]]
        ]
    for cid, kpts, box, anomaly in poses:
        draw_candidate(frame, kpts, box, cid, anomaly)


def garbage_collect_stale_candidates() -> None:
    """Drop buffers/counters for candidates unseen for > STALE_LOOKBACK_FRAMES."""
    global frame_counter
    with cand_lock:
        stale = [
            cid for cid, last in candidate_last_seen.items()
            if frame_counter - last > STALE_LOOKBACK_FRAMES
        ]
        for cid in stale:
            candidate_buffers.pop(cid, None)
            candidate_anomaly_counters.pop(cid, None)
            candidate_cheating_states.pop(cid, None)
            candidate_last_seen.pop(cid, None)
            candidate_last_alert.pop(cid, None)
            candidate_features.pop(cid, None)
            candidate_last_pose.pop(cid, None)
            candidate_body.pop(cid, None)
            pose_windows.drop(cid)
        if stale:
            print(f"[gc] removed {len(stale)} stale candidate(s)", flush=True)


def generate_stream_frames():
    """Multi-candidate processing loop (runs in a background thread)."""
    global latest_frame, latest_raw_frame, frame_counter, current_active_ids, detector
    if detector is None:
        detector = PoseDetector()
    cap = open_capture()
    start_time = time.monotonic()
    frames_processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Week 6: keep an un-annotated copy so the dashboard can toggle the
        # detection overlay on/off via /video_feed?overlay=0.
        with frame_lock:
            latest_raw_frame = frame.copy()
        frame_counter += 1
        process_this = (frame_counter % SUBSAMPLE) == 0

        # --- Every raw frame (30 FPS): buffet each currently-active candidate.
        if not process_this:
            encoded = encode_frame(frame)
            with cand_lock:
                active = list(current_active_ids)
            for cid in active:
                with cand_lock:
                    candidate_buffers[cid].append(encoded)
                    candidate_last_seen[cid] = frame_counter
            redraw_active_overlay(frame)
        else:
            # --- AI sub-sampling (5 FPS): tracked inference per candidate.
            result = detector.track(frame)
            boxes = result.boxes
            ids = result.tracker_ids
            kpts = result.keypoints
            with cand_lock:
                current_active_ids = set(int(i) for i in ids.tolist())

            for idx in range(len(ids)):
                cid = int(ids[idx])
                person_kpts = kpts[idx]
                features = extract_normalized_pose_features(person_kpts)
                is_anomaly = evaluate_candidate(features)

                # --- Week 4: temporal pose window + heuristic baseline. ---
                # normalize_keypoints is O(17), pushing to the deque is O(1),
                # and evaluating a (30, 17, 2) window is well under 1 ms, so
                # the 5 FPS inference path is never blocked (reader thread is
                # untouched — it only ever appends raw frames to its buffer).
                xy = normalize_keypoints(person_kpts, frame.shape[1], frame.shape[0])
                if xy is not None:
                    pose_windows.push(cid, xy)
                    if pose_windows.is_ready(cid):
                        seq = pose_windows.window(cid)
                        temporal_flags = temporal_baseline.evaluate(seq)
                        if temporal_flags["anomalous"]:
                            is_anomaly = True
                        with ts_lock:
                            active_label = telemetry_state["active_recording_label"]
                        if active_label in (0, 1, 2, 3):
                            sequence_writer.add(seq, active_label)

                with cand_lock:
                    # This sub-sampled raw frame also enters the candidate's buffer.
                    candidate_buffers[cid].append(encode_frame(frame))
                    candidate_last_seen[cid] = frame_counter
                    candidate_cheating_states[cid] = is_anomaly
                    candidate_anomaly_counters[cid] = (
                        min(candidate_anomaly_counters[cid] + 1, ANOMALY_THRESHOLD * 10)
                        if is_anomaly
                        else 0
                    )
                    candidate_features[cid] = {
                        "ear_ratio": float(features[0]),
                        "norm_vertical_drop": float(features[1]),
                        "box": [round(float(v), 1) for v in boxes[idx]],
                        "confidence": float(result.confidences[idx])
                        if idx < len(result.confidences)
                        else 0.0,
                        "keypoints": [
                            [round(float(x), 1), round(float(y), 1), round(float(c), 3)]
                            for x, y, c in person_kpts
                        ],
                    }
                    candidate_last_pose[cid] = (person_kpts.copy(), boxes[idx].copy())

                    # Head displacement velocity (scale-invariant, for the
                    # monitoring data contract's `velocity_spike`).
                    nose = person_kpts[0]
                    shoulder_w = np.hypot(
                        person_kpts[5][0] - person_kpts[6][0],
                        person_kpts[5][1] - person_kpts[6][1],
                    ) or 40.0
                    prev_body = candidate_body.get(cid)
                    if prev_body is not None:
                        dist = np.hypot(nose[0] - prev_body["nose"][0], nose[1] - prev_body["nose"][1])
                        last_v = min(dist / max(shoulder_w, 1e-3), 2.0)
                    else:
                        last_v = 0.0
                    candidate_body[cid] = {
                        "nose": (float(nose[0]), float(nose[1])),
                        "shoulder_w": float(shoulder_w),
                        "last_v": float(last_v),
                    }

                    count = candidate_anomaly_counters[cid]
                    now = time.monotonic()
                    if (
                        count >= ANOMALY_THRESHOLD
                        and (now - candidate_last_alert[cid]) > ALERT_COOLDOWN
                    ):
                        candidate_last_alert[cid] = now
                        candidate_anomaly_counters[cid] = 0
                        snapshot = list(candidate_buffers[cid])
                        alert_id = str(uuid.uuid4())[:8]
                        threading.Thread(
                            target=export_evidence_job,
                            args=(alert_id, cid, snapshot),
                            daemon=True,
                        ).start()
                        print(f"[alert] candidate {cid} anomaly sustained -> exporting", flush=True)

                draw_candidate(frame, person_kpts, boxes[idx], cid, is_anomaly)

            # Record labelled pose samples for dataset collection, if enabled.
            with ts_lock:
                active_label = telemetry_state["active_recording_label"]
            if active_label in (0, 1, 2, 3):
                for idx in range(len(ids)):
                    legacy = extract_pose_features(kpts[idx])
                    log_feature_vector(legacy, active_label)
                    with ts_lock:
                        telemetry_state["recorded_samples_count"] += 1

            garbage_collect_stale_candidates()

        frames_processed += 1
        elapsed = time.monotonic() - start_time
        with tel_lock:
            telemetry["fps"] = round(
                frames_processed / elapsed if elapsed > 0 else 0, 1
            )

        with frame_lock:
            latest_frame = frame.copy()


def generate_frames(overlay: bool = True):
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.03)
                continue
            source = latest_frame if overlay else latest_raw_frame
            if source is None:
                time.sleep(0.03)
                continue
            ret, jpeg = cv2.imencode(".jpg", source, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                time.sleep(0.03)
                continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        time.sleep(0.03)


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    # Production build (npm run build in frontend/) is preferred when present;
    # otherwise fall back to the legacy single-file dashboard.
    dist_index = FRONTEND_DIR / "dist" / "index.html"
    if dist_index.exists():
        return send_file(dist_index)
    return send_file(FRONTEND_DIR / "index.html")


@app.route("/legacy")
def legacy_dashboard():
    return send_file(FRONTEND_DIR / "legacy_dashboard.html")


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    """Static assets from the Vite production build (frontend/dist/assets)."""
    return send_from_directory(FRONTEND_DIR / "dist" / "assets", filename)


@app.route("/video_feed")
def video_feed():
    overlay = request.args.get("overlay", "1") != "0"
    return Response(
        generate_frames(overlay),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/telemetry")
def get_telemetry():
    with cand_lock:
        candidates = [
            {
                "id": cid,
                "ear_ratio": round(state["ear_ratio"], 4),
                "norm_vertical_drop": round(state["norm_vertical_drop"], 4),
                "anomaly_count": candidate_anomaly_counters.get(cid, 0),
                "status": "ANOMALY"
                if candidate_cheating_states.get(cid, False)
                else "NOMINAL",
                "box": state.get("box", []),
                "confidence": state.get("confidence", 0.0),
                "keypoints": state.get("keypoints", []),
            }
            for cid, state in candidate_features.items()
        ]
        active_candidates = len(candidates)
        anomalous = [c for c in candidates if c["status"] == "ANOMALY"]
        # Aggregate head-pose proxies from the first (primary) candidate so the
        # legacy dashboard fields keep updating.
        if candidates:
            primary = candidates[0]
            tel_tmp_yaw = round(max(-90.0, min(90.0, (1.0 - primary["ear_ratio"]) * 90.0)), 1)
            tel_tmp_pitch = round(
                max(-90.0, min(90.0, (primary["norm_vertical_drop"] - 0.5) * 90.0)), 1
            )
        else:
            tel_tmp_yaw = 0.0
            tel_tmp_pitch = 0.0
    with tel_lock:
        tel = dict(telemetry)
    with ts_lock:
        tel["active_recording_label"] = telemetry_state["active_recording_label"]
        tel["recorded_samples_count"] = telemetry_state["recorded_samples_count"]
    tel["active_candidates"] = active_candidates
    tel["candidates"] = candidates
    tel["yaw"] = tel_tmp_yaw
    tel["pitch"] = tel_tmp_pitch
    tel["face_detected"] = active_candidates > 0
    tel["anomaly_count"] = sum(c["anomaly_count"] for c in candidates)
    tel["anomaly"] = len(anomalous) > 0
    tel["alert"] = len(anomalous) > 0
    tel["temporal_windows_ready"] = pose_windows.ready_count()
    # Audit contract: identify the frame + the data source feeding it.
    with frame_lock:
        tel["frame_id"] = frame_counter
    tel["source_type"] = capture_source.get("type", "unknown")
    tel["source_path"] = capture_source.get("path")
    tel["tracked_students"] = candidates
    return jsonify(tel)


@app.route("/api/monitor/annotations")
def monitor_annotations():
    """Latest model inference in the monitoring data contract:

        {timestamp, frame_id, source_type, source_path, fps,
         students: [{track_id, bbox, status, confidence, keypoints,
                     velocity_spike, ear_ratio, norm_vertical_drop}]}

    `timestamp` is the video-relative time (frame_counter / CAM_FPS) so the
    LiveVisualizer can map each annotation to the frame currently streaming.
    """
    with frame_lock:
        fid = frame_counter
    with cand_lock:
        students = []
        for cid, state in candidate_features.items():
            er = state.get("ear_ratio", 1.0)
            drop = state.get("norm_vertical_drop", 0.0)
            if not candidate_cheating_states.get(cid, False):
                status = "NOMINAL"
            elif candidate_anomaly_counters.get(cid, 0) >= ANOMALY_THRESHOLD:
                status = "PEEKING"
            elif er < HEAD_TURN_EAR_RATIO_MIN or er > HEAD_TURN_EAR_RATIO_MAX:
                status = "HEAD_TURNING"
            elif drop > PITCH_LEAN_NORM_DROP:
                status = "LEANING"
            else:
                status = "HEAD_TURNING"
            students.append(
                {
                    "track_id": cid,
                    "bbox": state.get("box", []),
                    "status": status,
                    "confidence": state.get("confidence", 0.0),
                    "keypoints": state.get("keypoints", []),
                    "velocity_spike": round(candidate_body.get(cid, {}).get("last_v", 0.0), 3),
                    "ear_ratio": round(er, 4),
                    "norm_vertical_drop": round(drop, 4),
                }
            )
        students.sort(key=lambda s: s["track_id"])
    with tel_lock:
        fps = telemetry.get("fps", 0.0)
    return jsonify(
        {
            "timestamp": round(fid / CAM_FPS, 2),
            "frame_id": fid,
            "source_type": capture_source.get("type", "unknown"),
            "source_path": capture_source.get("path"),
            "fps": fps,
            "students": students,
        }
    )


@app.route("/api/record_label", methods=["POST"])
def record_label():
    data = request.get_json(silent=True) or {}
    label = data.get("label", -1)
    try:
        label = int(label)
    except (TypeError, ValueError):
        return jsonify({"error": "label must be an integer"}), 400
    if label not in (-1, 0, 1, 2, 3):
        return jsonify({"error": "label must be in [-1, 0, 1, 2, 3]"}), 400
    with ts_lock:
        telemetry_state["active_recording_label"] = label
        if label == -1:
            telemetry_state["recorded_samples_count"] = 0
    initialize_dataset()
    print(f"[record] active_recording_label -> {label} ({LABEL_NAMES.get(label, 'Off')})", flush=True)
    with ts_lock:
        return jsonify({"ok": True, "active_recording_label": telemetry_state["active_recording_label"],
                        "recorded_samples_count": telemetry_state["recorded_samples_count"]})


@app.route("/vault/<filename>")
def serve_vault(filename):
    return send_from_directory(EVIDENCE_DIR, filename)


@app.route("/api/evidence")
def list_evidence():
    """Evidence vault index.

    NOTE: clip filenames are currently `candidate_<id>_alert_<alert>.mp4`, so
    classroom / type / severity below are deterministic demo values derived
    from the filename + mtime. When the exporter stamps a JSON metadata sidecar
    per clip, replace this derivation with a sidecar read.
    """
    items = []
    for f in sorted(EVIDENCE_DIR.iterdir(), key=os.path.getmtime, reverse=True):
        if not f.is_file():
            continue
        h = hashlib.md5(f.name.encode()).digest()
        mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
        items.append(
            {
                "id": f.name.rsplit(".", 1)[0],
                "name": f.name,
                "url": f"/vault/{f.name}",
                "classroom": CLASSROOM_NAMES[h[0] % len(CLASSROOM_NAMES)],
                "type": ["Head Turn", "Peeking", "Note Passing"][h[1] % 3],
                "severity": ["low", "medium", "high"][h[2] % 3],
                "recorded_at": mtime.isoformat(),
                "duration_s": 30,
            }
        )
    return jsonify(items)


@app.route("/stream", methods=["POST"])
def stream_infer():
    """Webcam inference (dashboard live-testing).

    Accepts a base64 JPEG frame and returns tracked persons (boxes, COCO-17
    keypoints, track ids, confidence, head heuristics) so the browser can draw
    the overlay client-side. Coordinates are returned in the original frame
    pixel space, so the canvas aligns with the webcam snapshot.
    """
    data = request.get_json(silent=True) or {}
    image_b64 = (data.get("image") or "").strip()
    if not image_b64:
        return jsonify({"error": "image (base64 JPEG) required"}), 400
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(image_b64)
    except (ValueError, TypeError):
        return jsonify({"error": "invalid base64 payload"}), 400
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "could not decode frame"}), 400

    detector = get_stream_detector()
    with stream_lock:
        result = detector.track(frame)

    persons = []
    for i, cid in enumerate(result.tracker_ids.tolist()):
        kpts = result.keypoints[i]
        feats = extract_normalized_pose_features(kpts)
        box = result.boxes[i].tolist()
        conf = (
            float(result.confidences[i])
            if result.confidences.shape[0] > i
            else float(kpts[:, 2].max()) if kpts.size else 1.0
        )
        persons.append(
            {
                "id": int(cid),
                "box": [float(v) for v in box],
                "confidence": round(conf, 3),
                "status": "ANOMALY" if evaluate_candidate(feats) else "NOMINAL",
                "keypoints": [[float(x), float(y), float(c)] for (x, y, c) in kpts],
                "head": {
                    "ear_ratio": round(float(feats[0]), 4),
                    "norm_vertical_drop": round(float(feats[1]), 4),
                },
            }
        )
    return jsonify({"persons": persons, "count": len(persons)})


@app.route("/api/classrooms", methods=["GET", "POST"])
def classrooms():
    """Classroom registry (dashboard classroom management)."""
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        if not data.get("name") or not data.get("camera_ip"):
            return jsonify({"error": "name and camera_ip are required"}), 400
        item = {
            "id": f"room-{uuid.uuid4().hex[:8]}",
            "name": data["name"],
            "camera_ip": data["camera_ip"],
            "rtsp_url": data.get("rtsp_url", ""),
            "desk_rows": int(data.get("desk_rows", 6)),
            "desk_cols": int(data.get("desk_cols", 4)),
            "roster": data.get("roster", []),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        items = _load_classrooms()
        items.append(item)
        _save_classrooms(items)
        return jsonify(item), 201
    return jsonify(_load_classrooms())


@app.route("/dataset/<filename>")
def serve_dataset(filename):
    """Classroom dataset videos used by the video-alignment inspector."""
    if filename not in DATASET_WHITELIST:
        return jsonify({"error": "unknown dataset video"}), 404
    return send_from_directory(BACKEND_DIR, filename)


@app.route("/upload-test-video", methods=["POST"])
def upload_test_video():
    """Start an offline analysis job for an uploaded test recording.

    Multipart form field `video` → {job_id}. The pipeline (Week 3 tracking +
    Week 4/5 temporal heuristics) runs on a background thread; poll
    GET /api/upload-job/<id> for progress and the annotated output.
    """
    file = request.files.get("video")
    if file is None or not file.filename:
        return jsonify({"error": "missing video file"}), 400
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTS:
        return jsonify({"error": f"unsupported extension {ext or '(none)'}"}), 400

    job_id = uuid.uuid4().hex[:12]
    output_dir = UPLOAD_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / f"input{ext}"

    with upload_jobs_lock:
        upload_jobs[job_id] = {
            "job_id": job_id,
            "status": "uploading",
            "progress": 0,
            "filename": file.filename,
        }
    file.save(input_path)
    if input_path.stat().st_size > MAX_UPLOAD_BYTES:
        input_path.unlink(missing_ok=True)
        with upload_jobs_lock:
            upload_jobs.pop(job_id, None)
        return jsonify({"error": "file exceeds 300 MB limit"}), 413

    _job_update(job_id, status="queued")
    threading.Thread(
        target=process_upload_job,
        args=(job_id, input_path, output_dir),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id, "status": "queued"}), 202


@app.route("/api/upload-job/<job_id>")
def upload_job_status(job_id):
    with upload_jobs_lock:
        job = upload_jobs.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job"}), 404
    return jsonify(job)


@app.route("/upload_jobs/<job_id>/<filename>")
def serve_upload_job_file(job_id, filename):
    """Serve a job's annotated video / analysis sidecar to the dashboard."""
    job_dir = UPLOAD_DIR / job_id
    if not job_dir.is_dir():
        return jsonify({"error": "unknown job"}), 404
    return send_from_directory(job_dir, filename)


if __name__ == "__main__":
    def audit_data_sources() -> None:
        """Print exactly what the runtime reads, so the data path is verifiable.

        The live pipeline is real-video only: cv2.VideoCapture on the webcam or
        VIDEO_SOURCE (no synthetic generator). The collected feature datasets
        (pose_dataset*.csv, sequence_dataset/) are training-side artefacts and
        are NOT consumed at runtime — documented here for the audit.
        """
        print("\n[audit] ===== backend data-path audit =====", flush=True)
        print(f"[audit] detector model : {POSE_MODEL_PATH} (exists={os.path.exists(POSE_MODEL_PATH)})", flush=True)
        print(f"[audit] capture source : VIDEO_SOURCE={VIDEO_SOURCE} (exists={os.path.exists(VIDEO_SOURCE)})", flush=True)
        for name in ("pose_dataset.csv", "pose_dataset_cctv.csv"):
            p = BACKEND_DIR / name
            print(f"[audit] training csv   : {p} (exists={p.exists()}, bytes={p.stat().st_size if p.exists() else 0})", flush=True)
        print(f"[audit] sequence ds dir: {SEQUENCE_DATASET_PATH} (exists={SEQUENCE_DATASET_PATH.is_dir()})", flush=True)
        print(f"[audit] evidence vault : {EVIDENCE_DIR}", flush=True)
        print(f"[audit] upload jobs    : {UPLOAD_DIR}", flush=True)
        print("[audit] runtime dataset: REAL video via cv2.VideoCapture (no synthetic generator active)", flush=True)
        print("[audit] ===== end audit =====\n", flush=True)

    audit_data_sources()
    initialize_dataset()
    start_auto_purge_thread(EVIDENCE_DIR, retention_seconds=86400, check_interval=3600)
    threading.Thread(target=generate_stream_frames, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)