# Malpractice Detection

Intelligent **offline exam malpractice detection system** — analyzes a proctored exam video feed (file, webcam, or RTSP stream) in real time and flags suspicious behavior: head turns, body rotation, leaning, absent seating, unauthorized objects, and more.

Everything runs locally. No cloud, no uploads.

## Architecture

Three surfaces share the same detection logic:

| Component | Path | Description |
|---|---|---|
| **Flask backend + dashboard** | `backend/app.py` | Flask API (`/video_feed`, `/api/telemetry`, `/api/evidence`), serves the web dashboard, evidence vault, RTSP reconnect |
| **Headless pipeline** | `backend/main.py` | Threaded pipeline: `VideoStreamReader` → bounded queue → `DetectorRunner` → `ExporterThread` |
| **Desktop app (PyQt6)** | `desktop_app/` | Full GUI with P01–P40 rule engine **and a separate SCUT-HEAD head-occupancy mode**, HUD overlay, dataset logger, console |
| **Web dashboard** | `frontend/` | Static page — Tailwind (CDN) + Chart.js, polls telemetry/evidence endpoints |

Supporting pieces:

- `backend/detector.py` — YOLO11n-pose at 5 FPS + heuristic rules
- `backend/stgcn.py` — ST-GCN skeleton action recognition (30-frame window, 17 joints)
- `backend/exporter.py` — async evidence clips (15 s pre/post-event buffer)
- `models/desk/` — custom YOLO training for desk objects (paper, pen, notebook — COCO lacks them)
- `models/head/` — one-class YOLO11n head detector trained on SCUT-HEAD
- `desktop_app/core/head_detector.py` — head-occupancy worker and HUD (separate from pose rules)
- `train_classifier.py` — trains RandomForest/XGBoost pose classifiers from logged features
- `tools/` — evaluation and test scripts (`test_e2e_pipeline.py`, `eval_objects.py`, `calibrate_body_turn.py`, …), plus the SCUT-HEAD ingest/export/train/benchmark pipeline

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

**Desktop P01–P38 (reachable set):** head turn (L/R, 30°/60°), head down/up, lean, body turn, looking away, note peeking, looking down, standing, absent seat, head on desk, hand to mouth, writing under desk, frequent glancing, chin rest, crossed arms, multiple persons, unusual object, phone usability. IDs P17/P24/P27/P34/P37/P39/P40 were removed — no sensor in this system can measure them.

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

### CUDA setup (recommended for training)

The default `requirements.txt` install can resolve a CPU-only PyTorch wheel. With `uv`, `uv sync` already pins the CUDA 13 wheel index and installs `torch 2.14.1+cu130` / `torchvision 0.29.1+cu130`. For pip, install the CUDA build explicitly so YOLO training, the sequence trainer, and inference use the GPU:

```bash
pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.14.0+cu130 torchvision==0.29.0+cu130
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If a `+cpu` wheel is already installed, add `--force-reinstall --no-deps`: a requirement such as `torch==2.14.0` also matches `2.14.0+cpu`, because pip ignores the local version label when matching.

## Head occupancy with SCUT-HEAD

[SCUT-HEAD](https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release) is a real classroom dataset containing **head bounding boxes only** — no pose keypoints and no behaviour labels. The desktop app therefore uses it for head detection and occupancy counting, in a mode that is deliberately separate from the P01–P38 pose rules. It is not used to train the behaviour classifier or to populate `data/proctoring_features.csv`.

> **License / use:** SCUT-HEAD is released for **academic research only**. `datasets/` and `*.pt` are git-ignored, so neither the dataset nor trained weights are committed. Review the dataset repository's terms before any non-research use.

### Dataset and baseline pipeline

```bash
# 1. Download both parts, extract, validate, and build manifest.csv
python tools/scut_head_ingest.py

# 2. Export a one-class YOLO dataset, data.yaml, and desktop replay lists
python tools/scut_head_export.py

# 3. Train the head detector on the GPU (auto -> 6gb preset on CUDA)
python tools/train_head_detector.py --preset auto --epochs 50

# 4. Benchmark boxes/counts; CUDA + FP16 is selected automatically
python tools/benchmark_head_detector.py --device auto
```

Training presets:

| Preset | Resolution | Batch | Workers | Cache | AMP | Use |
|---|---:|---:|---:|---|---|---|
| `6gb` (auto on CUDA) | 640 | 16 | 6 | RAM | on | RTX 3050 6 GB and similar |
| `4gb` | 512 | 8 | 4 | RAM | on | Heavier desktop/UI use alongside training |
| `cpu` | 640 | 16 | 0 | disk | off | Reproduction of the CPU baseline |

The `6gb`/`4gb` presets pass `nbs=64`, so Ultralytics accumulates micro-batches up to a nominal batch of 64 without the VRAM cost. If CUDA still runs out of memory, the script halves the physical batch and retries (up to three attempts) instead of failing. `--cache ram` avoids re-decoding JPEGs every epoch; remove `--fraction` and `--freeze` for a full-quality run. Explicit `--imgsz`, `--batch`, `--workers`, `--cache`, or `--amp/--no-amp` override the preset.

The export writes `datasets/SCUT_HEAD/yolo/` with `images/{train,val,test}`, `labels/{train,val,test}`, `data.yaml`, and replay lists such as `replay/A_test.txt`. Images are hard-linked when possible, so the export does not duplicate the dataset. The trained app weights are copied to `models/head/scut_head_yolo11n.pt`, and the run summary is written next to them as `train.summary.json`.

**CPU note:** the `cpu` preset remains available for reproducibility. At `imgsz=640`, training costs roughly 0.5 s/image on this machine; one full epoch over 2,543 images is about 20 minutes (the recorded 8-epoch subset baseline took 1,452 s). Use a GPU and remove `--fraction`/`--freeze` for a stronger model.

### Desktop modes and sources

Launch the app with `python -m desktop_app.main`. On the Hub:

1. Choose **Detection Mode**:
   - **Head count (SCUT-HEAD)** — runs the head detector, draws head boxes, and reports occupancy. This is the dataset-backed mode.
   - **P01–P38 pose rules** — the original pose/behaviour pipeline.
2. Choose a source: webcam, RTSP URL, or an **Image Folder / Replay List**. A replay list can be `.txt` (one image path per line) or `.csv` with an `image` column. **Use SCUT-HEAD Test** fills in `datasets/SCUT_HEAD/yolo/replay/A_test.txt` when the export exists.
3. Click **START CAMERA**, then **Launch Proctoring Session**.

Head mode loops the replay images at `IP_REPLAY_FPS` (default 2.0) and shows `Heads: N` plus the image position in the sidebar. For SCUT-HEAD replay lists it also draws the dataset's annotated head boxes in **magenta** and shows the ground-truth count beside the detector output; live sources have no GT. Pose mode never reports head counts, and head mode never claims a P01–P38 behaviour. If head weights are missing, the app logs a warning and falls back to pose mode.

Relevant environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `IP_TORCH_DEVICE` | `auto` | `auto`, `cuda`, or `cpu` for all PyTorch inference |
| `IP_INFERENCE_HALF` | `1` | FP16 inference on CUDA (`0` forces FP32) |
| `IP_HEAD_DEVICE` | follows `IP_TORCH_DEVICE` | Device override for head mode |
| `IP_HEAD_HALF` | follows `IP_INFERENCE_HALF` | FP16 override for head mode |
| `IP_NUM_WORKERS` | host-derived (max 6 on Windows) | DataLoader workers for custom training loops |
| `IP_AMP` | `1` | Enable AMP in the sequence trainer on CUDA |
| `IP_HEAD_MODEL` | `scut_head_yolo11n.pt` | Head weights filename |
| `IP_HEAD_MODEL_DIR` | `models/head` | First directory to search |
| `IP_HEAD_CONF` | `0.25` | Detection confidence threshold |
| `IP_HEAD_IMGSZ` | `640` | Detector input size |
| `IP_HEAD_FPS` | `5.0` | Head inference rate cap |
| `IP_REPLAY_FPS` | `2.0` | Image-replay frame rate |

`tools/benchmark_head_detector.py` accepts the same device policy explicitly: `--device auto|cpu|0` and `--half/--no-half`. On CPU, FP16 is always disabled. For a CPU-only benchmark, run `python tools/benchmark_head_detector.py --device cpu --no-half`.

### Baseline results (honest caveats)

The included model was trained for 8 epochs on 20% of the training split (509 images), with the first 10 layers frozen. On the full 1,000-image test split it scores:

| Metric | Value |
|---|---:|
| Box precision | 0.800 |
| Box recall | 0.663 |
| Box F1 | 0.725 |
| Head-count MAE | 5.11 heads/image |
| Head-count bias | −4.59 heads/image |
| Exact-count frames | 17.1% |

The model undercounts crowded scenes (for example 59 predicted vs 83 annotated heads on one frame). These numbers measure head detection/counting only; SCUT-HEAD cannot measure cheating behaviour. The pose-model proxy baseline covers just 23.3% of annotated head centres and has a count MAE of 21.5, which is why a dedicated head detector is used instead. Full report: `datasets/SCUT_HEAD/benchmark/head_benchmark.md` (git-ignored). Train on the full split with a GPU before treating any count as operational.

## Running

```bash
# Flask backend + web dashboard  →  http://localhost:5000
python backend/app.py

# Headless threaded pipeline (Ctrl+C to stop)
python backend/main.py

# PyQt6 desktop app
python -m desktop_app.main

# Train pose classifier (P01–P40) from logged features.
# The parsed feature matrix is cached as <csv>.feature_cache.npz; rerun with
# --rebuild-cache after editing the CSV, or --no-cache to bypass it entirely.
python train_classifier.py --csv data/proctoring_features.csv --model models/pose_rf_classifier.pkl

# Tests / evaluation scripts
python tools/test_e2e_pipeline.py
python tools/test_head_mode.py   # skips if local SCUT-HEAD assets are absent
python tools/test_dead_source.py
```

Custom PyTorch training (`backend/temporal_training.py`, `backend/evaluate.py`, `backend/ablation_study.py`) resolves `torch.device("cuda" if torch.cuda.is_available() else "cpu")` automatically, enables FP16 autocast and a GradScaler on CUDA, and uses pinned memory with persistent workers when a GPU is present. Set `IP_TORCH_DEVICE=cpu` to force the CPU path, or `IP_AMP=0` to train in FP32.

Set the video source via environment variable (defaults to a sample exam video; `0` = webcam):

```bash
set VIDEO_SOURCE=0                # webcam
set VIDEO_SOURCE=C:\path\to\exam.mp4
python backend/app.py
```

## Packaging the desktop app

Install the runtime and build dependencies, then run the release script:

```bash
python -m pip install -r requirements.txt -r requirements-build.txt
python tools/build_release.py --clean
```

This produces `dist/IntelliProctor/IntelliProctor.exe` and an
`IntelliProctor.lnk` shortcut inside the deployment bundle. To also place a
shortcut on the current user's Desktop:

```bash
python tools/build_release.py --clean --desktop
```

Desktop creation is opt-in; a normal build never modifies the user's Desktop.
Use `--force` to replace an existing Desktop shortcut. The shortcut points at
the packaged `.exe`, starts it in the bundle directory, and uses the
application icon embedded in the executable, so it remains valid if the
deployment folder is moved. Pass `--icon path/to/icon.ico` only when a
separate shortcut icon is required.

The build needs the locally trained (git-ignored) artefacts:

- `assets/pose_landmarker_*.task` (at least the configured MediaPipe variant)
- `models/pose_rf_classifier.pkl` and its `.metrics.json` sidecar
- `models/head/scut_head_yolo11n.pt`
- `backend/yolo11n-pose.pt` and the COCO fallback `backend/yolo11n.pt`

The shortcut helper can also be used on its own:

```bash
python tools/create_shortcut.py                      # bundle shortcut
python tools/create_shortcut.py --dest desktop       # current Desktop
python tools/create_shortcut.py --dest path --dest-dir release/
```

On Windows the helper writes a real `.lnk` through `WScript.Shell` using only
the Python standard library and PowerShell. Linux and macOS fall back to a
`.desktop` file and a `.command` launcher.

## Project layout

```
├── backend/           Flask app, headless pipeline, YOLO/ST-GCN detectors, exporter
├── frontend/          Proctoring dashboard (static HTML + Tailwind + Chart.js)
├── desktop_app/       PyQt6 GUI (core rule engine, HUD, camera/inference threads)
├── models/            Custom desk-object and SCUT-HEAD head-detector training
├── tools/             Evaluation, calibration, SCUT-HEAD pipeline, and test scripts
├── data/              Feature CSVs and dataset configs
├── datasets/          Local datasets (SCUT-HEAD; git-ignored, research-only)
├── assets/            MediaPipe pose landmarker models (not committed)
├── train_classifier.py
└── requirements.txt / pyproject.toml / uv.lock
```

## Data handling

All processing is local. Evidence clips are written to `backend/evidence_vault/` and event exports to `backend/exports/` — both are git-ignored.

`datasets/` is git-ignored as well. SCUT-HEAD is academic-research-only; its images, manifests, exported YOLO labels, and replay lists must not be committed or redistributed with this project. Trained weights (`*.pt`) are also git-ignored.
