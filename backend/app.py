#!/usr/bin/env python3
import os, subprocess, threading, time, uuid
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory, send_file

from dataset_collector import initialize_dataset, log_feature_vector
from feature_extractor import extract_pose_features
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
SUBSAMPLE = 6
ANOMALY_THRESHOLD = 10
ALERT_COOLDOWN = 10
YAW_LIMIT = 30
PITCH_LIMIT = 30
EVIDENCE_DIR = BACKEND_DIR / "evidence_vault"
EVIDENCE_DIR.mkdir(exist_ok=True)

POSE_MODEL_PATH = os.environ.get("POSE_MODEL", str(BACKEND_DIR / "yolo11n-pose.pt"))
LABEL_NAMES = {0: "Normal", 1: "Head Turn", 2: "Leaning", 3: "Passing"}

telemetry_state = {
    "active_recording_label": -1,
    "recorded_samples_count": 0,
}
ts_lock = threading.Lock()

from ultralytics import YOLO

pose_model = YOLO(POSE_MODEL_PATH)

import urllib.request

HAAR_PATH = Path("/tmp/haarcascade_frontalface_default.xml")
if not HAAR_PATH.exists():
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, str(HAAR_PATH))
face_cascade = cv2.CascadeClassifier(str(HAAR_PATH))

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -15.0, -5.0),
    (-30.0, -30.0, -10.0),
    (30.0, -30.0, -10.0),
    (-25.0, -50.0, -5.0),
    (25.0, -50.0, -5.0),
], dtype=np.float64)

FOCAL = 700
CAM_MAT = np.array([[FOCAL, 0, 640], [0, FOCAL, 360], [0, 0, 1]], dtype=np.float64)
DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)

ring_buffer = deque(maxlen=900)
buf_lock = threading.Lock()
latest_frame = None
frame_lock = threading.Lock()

telemetry = {
    "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
    "face_detected": False, "anomaly": False,
    "anomaly_count": 0, "alert": False, "fps": 0.0,
}
tel_lock = threading.Lock()

anomaly_counter = 0
last_alert_time = 0


def estimate_landmarks(x, y, w, h):
    return np.array([
        (x + w // 2, int(y + h * 0.62)),
        (x + w // 2, int(y + h * 0.45)),
        (x + int(w * 0.3), int(y + h * 0.33)),
        (x + int(w * 0.7), int(y + h * 0.33)),
        (x + int(w * 0.25), int(y + h * 0.72)),
        (x + int(w * 0.75), int(y + h * 0.72)),
    ], dtype=np.float64)


def compute_head_pose(landmarks):
    _, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS, landmarks, CAM_MAT, DIST_COEFFS,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return float(np.degrees(x)), float(np.degrees(y)), float(np.degrees(z))


def export_evidence(frames):
    alert_id = str(uuid.uuid4())[:8]
    out_path = EVIDENCE_DIR / f"alert_{alert_id}.mp4"
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


def process_frames():
    global latest_frame, anomaly_counter, last_alert_time
    cap = open_capture()
    frame_idx = 0
    start_time = time.time()
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        with buf_lock:
            ring_buffer.append(frame.copy())
        process_this = (frame_idx % SUBSAMPLE) == 0
        frame_idx += 1
        yaw = pitch = roll = 0.0
        face_detected = False
        is_anomaly = False
        features = None
        if process_this:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
            )
            if len(faces) > 0:
                face_detected = True
                fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
                landmarks = estimate_landmarks(fx, fy, fw, fh)
                yaw, pitch, roll = compute_head_pose(landmarks)
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
                for lx, ly in landmarks.astype(int):
                    cv2.circle(frame, (lx, ly), 3, (0, 0, 255), -1)
                is_anomaly = abs(yaw) > YAW_LIMIT or abs(pitch) > PITCH_LIMIT
                anomaly_counter = anomaly_counter + 1 if is_anomaly else 0
                t_now = time.time()
                if anomaly_counter >= ANOMALY_THRESHOLD and (t_now - last_alert_time) > ALERT_COOLDOWN:
                    last_alert_time = t_now
                    anomaly_counter = 0
                    with buf_lock:
                        export_frames = list(ring_buffer)
                    threading.Thread(target=export_evidence, args=(export_frames,), daemon=True).start()
            else:
                anomaly_counter = 0

            pose_results = pose_model(frame, verbose=False)
            if pose_results and pose_results[0].keypoints is not None:
                kpts = pose_results[0].keypoints.data.cpu().numpy()
                if kpts.shape[0] > 0:
                    primary = kpts[0]
                    features = extract_pose_features(primary)
                    for lx, ly in primary[:, :2].astype(int):
                        cv2.circle(frame, (int(lx), int(ly)), 3, (255, 0, 255), -1)

            if features is not None:
                with ts_lock:
                    active = telemetry_state["active_recording_label"]
                if active in (0, 1, 2, 3):
                    log_feature_vector(features, active)
                    with ts_lock:
                        telemetry_state["recorded_samples_count"] += 1

        frame_count += 1
        elapsed = time.time() - start_time
        with tel_lock:
            telemetry["yaw"] = round(yaw, 1)
            telemetry["pitch"] = round(pitch, 1)
            telemetry["roll"] = round(roll, 1)
            telemetry["face_detected"] = face_detected
            telemetry["anomaly"] = is_anomaly
            telemetry["anomaly_count"] = anomaly_counter
            telemetry["alert"] = anomaly_counter >= ANOMALY_THRESHOLD
            telemetry["fps"] = round(frame_count / elapsed if elapsed > 0 else 0, 1)
        cv2.putText(frame, f"Yaw:{yaw:.1f} Pitch:{pitch:.1f} Roll:{roll:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Anomaly:{anomaly_counter}/{ANOMALY_THRESHOLD}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if is_anomaly else (0, 255, 0), 2)
        with ts_lock:
            rec_label = telemetry_state["active_recording_label"]
            rec_count = telemetry_state["recorded_samples_count"]
        rec_status = "OFF" if rec_label not in (0, 1, 2, 3) else LABEL_NAMES[rec_label]
        rec_color = (0, 0, 255) if rec_label in (0, 1, 2, 3) else (128, 128, 128)
        cv2.putText(frame, f"REC[{rec_status}] samples:{rec_count}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, rec_color, 2)
        if anomaly_counter >= ANOMALY_THRESHOLD:
            cv2.putText(frame, "*** CHEATING ALERT ***", (frame.shape[1] // 2 - 200, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
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


@app.route("/")
def index():
    return send_file(FRONTEND_DIR / "index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/telemetry")
def get_telemetry():
    with tel_lock:
        tel = dict(telemetry)
    with ts_lock:
        tel["active_recording_label"] = telemetry_state["active_recording_label"]
        tel["recorded_samples_count"] = telemetry_state["recorded_samples_count"]
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
    threading.Thread(target=process_frames, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
