# Malpractice Detection

Intelligent **offline exam malpractice detection system** — analyzes a proctored exam video feed (file, webcam, or RTRT stream) in real time and flags suspicious behavior: head turns, body rotation, leaning, absent seating, unauthorized objects, and more.

Everything runs locally. No cloud, no uploads.

## Architecture

Three surfaces share the same detection logic:

| Component | Path | Description |
|---|---|---|
| **Flask backend + dashboard** | `backend/app.py` | Flask API (`/video_feed`, `/api/telemetry`, `/api/evidence`), serves the web dashboard, evidence vault, RTSP reconnect |
| **Headless pipeline** | `backend/main.py` | Threaded pipeline: `VideoStreamReader` → bounded queue → `DetectorRunner` → `ExporterThread` |
| **Desktop app (PyQt6)** | `desktop_app/` | Full GUI with P01–P40 rule engine, HUD overlay, dataset logger, console |
| **Web dashboard** | `frontend/` | Static page — Tailwind (CDN) + Chart.js, polls telemetry/evidence endpoints |

Supporting pieces:

- `backend/detector.py` — YOLO11n-pose at 5 FPS + heuristic rules
- `backend/stgcn.py` — ST-GCN skeleton action recognition (30-frame window, 17 joints)
- `backend/exporter.py` — async evidence clips (15 s pre/post-event buffer)
- `models/desk/` — custom YOLO training for desk objects (paper, pen, notebook — COCO lacks them)
- `train_classifier.py` — trains RandomForest/XGBoost pose classifiers from logged features
- `tools/` — evaluation and test scripts (`test_e2e_pipeline.py`, `eval_objects.py`, `calibrate_body_turn.py`, …)

### Detection rules

**Backend (Flask / pipeline):**

| Rule | Trigger |
|---|---|
| `head_down` | pitch < -25° |
| `body_turn` | yaw > 35° |
| `excessive_lean` | sustained torso lean |
| `multi_person` | more than one person in frame |
| `object_detected` | unauthorized object in desk region |

Each alert carries a severity: `low` / `medium` / `high`.

**Desktop P01–P40:** head turn (L/R, 30°/60°), head down/up, lean, body turn, looking away, note peeking, talking, standing, absent seat, head on desk, hand to mouth, writing under desk, second-screen gaze, hidden face, earbuds, object transfer, frequent glancing, crossed arms, multiple persons, unusual object, collaboration, and more.

Object rules combine desk-region detection with spatial relations (`in_hand`, `reachable`, `lap`, `away`), nearest-person attribution, persistence tracks, and confidence/stability gates.

## Setup

Requires **Python ≥ 3.11** and `ffmpeg` on PATH.

```bash
# with uv (uses uv.lock)
uv sync

# or with pip
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

> **Model weights are not committed.** YOLO weights (`yolo11n*.pt`) download automatically from the ultralytics registry on first run. MediaPipe `.task` models go in `assets/` — see the [MediaPipe Pose Landmarker docs](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) for downloads.

## Running

```bash
# Flask backend + web dashboard  →  http://localhost:5000
python backend/app.py

# Headless threaded pipeline (Ctrl+C to stop)
python backend/main.py

# PyQt6 desktop app
python -m desktop_app.main

# Train pose classifier (P01–P40) from logged features
python train_classifier.py --csv data/proctoring_features.csv --model models/pose_rf_classifier.pkl

# Tests / evaluation scripts
python tools/test_e2e_pipeline.py
```

Set the video source via environment variable (defaults to a sample exam video; `0` = webcam):

```bash
set VIDEO_SOURCE=0                # webcam
set VIDEO_SOURCE=C:\path\to\exam.mp4
python backend/app.py
```

## Project layout

```
├── backend/           Flask app, headless pipeline, YOLO/ST-GCN detectors, exporter
├── frontend/          Proctoring dashboard (static HTML + Tailwind + Chart.js)
├── desktop_app/       PyQt6 GUI (core rule engine, HUD, camera/inference threads)
├── models/            Custom desk-object YOLO training
├── tools/             Evaluation, calibration, and test scripts
├── data/              Feature CSVs and dataset configs
├── assets/            MediaPipe pose landmarker models (not committed)
├── train_classifier.py
└── requirements.txt / pyproject.toml / uv.lock
```

## Data handling

All processing is local. Evidence clips are written to `backend/evidence_vault/` and event exports to `backend/exports/` — both are git-ignored.
