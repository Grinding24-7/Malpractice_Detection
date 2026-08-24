# Backend Data-Flow & Rendering Audit

Date: 2026-08-15
Scope: Week 3 CCTV ByteTrack data path, keypoint rendering on the dashboard, and
the "synthetic vs real video" question.

## Findings

### 1. Runtime is real video — there is no synthetic generator in the live path
- `SyntheticPoseGenerator` / `MockKeypointDataset` do not exist anywhere in the
  repo. `grep` finds synthetic code only in:
  - `temporal_features.py` (`make_synthetic_sequence`, smoke tests)
  - `temporal_training.py` (`_make_peeking_sequence`, `__main__` demo)
  - `frontend/src/data/mockData.js` (offline UI fallback only)
- The live capture loop reads real video: webcam device 0, else
  `VIDEO_SOURCE` (env, default now `backend/sample_exam.mp4`) via
  `cv2.VideoCapture`. `main.py` (Week 3 CLI) is separate and defaulted to
  `sample_exam.mp4`.
- The training-side artefacts (`pose_dataset.csv`,
  `pose_dataset_cctv.csv` 574 KB, `sequence_dataset/`) are real but are
  **training-only** — they are not consumed by the runtime pipeline. Runtime
  inference is live ByteTrack on the stream. This is by design; documented at
  startup (see `[audit]` log).

### 2. ROOT CAUSE of "keypoints not rendering": the fallback video was blank
- `backend/1_hour_exam_test.mp4` (108,000 frames) is effectively **black**
  (mean brightness ~1.7/255 across every sampled frame). ByteTrack never found
  a person, so nothing was ever drawn.
- `backend/sample_exam.mp4` is real content (mean ~165–173).
- Default `VIDEO_SOURCE` was `1_hour_exam_test.mp4`; it is now `sample_exam.mp4`.

### 3. Secondary rendering bug: annotations only on 1-in-6 frames (flicker)
- Inference is sub-sampled to 5 FPS (`SUBSAMPLE = 6`), and
  `draw_candidate()` only ran inside the subsampled branch, so the box +
  skeleton flashed on 1 of every 6 frames (invisible in practice).
- Fixed: `redraw_active_overlay()` re-renders each active candidate's last
  known pose on every 30 FPS frame (stale after ~4 s unseen).

### 4. ByteTrack is genuinely executing
- `detector.py:_infer` runs `model.track(frame, persist=True,
  tracker="bytetrack.yaml")`; confirmed on the live stream: 1 person tracked,
  id 1, conf 0.939, 17 keypoints, ear_ratio heuristics scoring ANOMALY.

## Changes made (all in `backend/app.py`)
1. Default `VIDEO_SOURCE` → `sample_exam.mp4` (was the blank
   `1_hour_exam_test.mp4`; still overridable via env).
2. `open_capture()` now records the resolved source in `capture_source` and
   warns loudly if the chosen source is blank (mean brightness < 5/255).
3. Persistent overlay: `candidate_last_pose` dict + `redraw_active_overlay()`
   draw skeletons on every frame; poses GC'd with stale candidates.
4. Startup audit log (`[audit]` block) prints detector model path, capture
   source, training CSV sizes, and the real-video runtime dataset note.
5. `/api/telemetry` now carries the audit contract keys:
   - `frame_id` (monotonic raw frame counter)
   - `source_type` (`webcam` | `video_file`), `source_path`
   - `tracked_students` = full candidate list incl. `box`, `confidence`, and
     JSON-encodable `keypoints` (17×3) — alongside the legacy `candidates`.
   - `fps` was already present.

## Verification (live server)
- `[audit] data source -> {'type': 'video_file', 'path': '.../sample_exam.mp4'}`
- `/api/telemetry`: `frame_id` incrementing, `tracked_students` = 1 student
  (id 1, conf 0.939, 17 kpts, box [206,64,528,428]).
- `/video_feed?overlay=1`: 10/10 consecutive frames contain magenta skeleton +
  yellow keypoints (was ~1/6). `overlay=0` returns raw frames with no overlay.
- `app.py` compiles clean.

## Remaining notes
- `main.py` (Week 3 headless orchestrator) logs alerts only; it never writes
  annotated video output — annotation rendering lives in `app.py`/dashboard.
- `/stream` (webcam POST) already returned full keypoints JSON; it uses an
  isolated detector and is unchanged.
