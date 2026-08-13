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

import os
import subprocess
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory, send_file

from dataset_collector import initialize_dataset, log_feature_vector
from detector import PoseDetector
from feature_extractor import extract_normalized_pose_features, extract_pose_features
from retention_policy import start_auto_purge_thread

app = Flask(__name__)

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"

VIDEO_SOURCE = os.environ.get("VIDEO_SOURCE", str(BACKEND_DIR / "1_hour_exam_test.mp4"))
CAM_INDEX = 0
CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_FPS = 30
SUBSAMPLE = 6  # AI sub-sampling: every 6th frame (30 / 5 FPS)
ANOMALY_THRESHOLD = 10  # sustained anomaly frames before an alert fires
ALERT_COOLDOWN = 10  # seconds between exports for the same candidate
BUFFER_FRAMES = 30 * 30  # per-candidate RAM ring buffer length (30 s @ 30 FPS)
STALE_LOOKBACK_FRAMES = 300  # candidate cleanup after > 10 s unseen

# Heuristic thresholds on the scale-normalized features.
HEAD_TURN_EAR_RATIO_MIN = 0.70  # ear_ratio below this => head turned left
HEAD_TURN_EAR_RATIO_MAX = 1.40  # ear_ratio above this => head turned right
PITCH_LEAN_NORM_DROP = 0.90  # norm_vertical_drop above this => leaning/bent

EVIDENCE_DIR = BACKEND_DIR / "evidence_vault"
EVIDENCE_DIR.mkdir(exist_ok=True)

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
current_active_ids: set[int] = set()  # candidate_ids present in the latest inference
frame_counter = 0  # monotonic raw-frame index driving stale cleanup

detector: "PoseDetector | None" = None

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
frame_lock = threading.Lock()

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
    cap = cv2.VideoCapture(CAM_INDEX)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        print(f"[capture] webcam device {CAM_INDEX} opened at {CAM_WIDTH}x{CAM_HEIGHT} @ {CAM_FPS}fps", flush=True)
        return cap
    print(f"[capture] WARNING: webcam device {CAM_INDEX} unavailable, falling back to {VIDEO_SOURCE}", flush=True)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if cap.isOpened():
        return cap
    raise RuntimeError("[capture] ERROR: webcam AND video file both failed to open")


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
        if stale:
            print(f"[gc] removed {len(stale)} stale candidate(s)", flush=True)


def generate_stream_frames():
    """Multi-candidate processing loop (runs in a background thread)."""
    global latest_frame, frame_counter, current_active_ids, detector
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


def generate_frames():
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.03)
                continue
            ret, jpeg = cv2.imencode(".jpg", latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
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
    return send_file(FRONTEND_DIR / "index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


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
    return jsonify(tel)


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
    files = sorted(EVIDENCE_DIR.iterdir(), key=os.path.getmtime, reverse=True)
    return jsonify([f.name for f in files if f.is_file()])


if __name__ == "__main__":
    initialize_dataset()
    start_auto_purge_thread(EVIDENCE_DIR, retention_seconds=86400, check_interval=3600)
    threading.Thread(target=generate_stream_frames, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)