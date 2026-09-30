"""
endpoints.py — FastAPI router for upload processing, job status, and webcam toggle.

Endpoints:
    POST /api/v1/upload-test-video  — Multipart file upload → background inference job
    GET  /api/v1/job-status/{id}    — Pollable progress + detected anomaly timestamps
    POST /api/v1/webcam/toggle      — Controls local webcam ingestion (ON/OFF)
"""

from __future__ import annotations

import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector import PoseDetector
from feature_extractor import extract_normalized_pose_features

router = APIRouter(prefix="/api/v1", tags=["endpoints"])

BACKEND_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BACKEND_DIR / "upload_jobs"
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_UPLOAD_BYTES = 300 * 1024 * 1024

HEAD_TURN_EAR_RATIO_MIN = 0.70
HEAD_TURN_EAR_RATIO_MAX = 1.40
PITCH_LEAN_NORM_DROP = 0.90

# In-memory job registry
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

# Webcam state
_webcam_state = {"active": False, "source_index": 0}


def _job_update(job_id: str, **fields) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job.update(fields)


def _evaluate_candidate(features: np.ndarray) -> bool:
    ear_ratio = float(features[0])
    norm_vertical_drop = float(features[1])
    nose_conf = float(features[4])
    l_ear_conf = float(features[5])
    r_ear_conf = float(features[6])
    if nose_conf < 0.35 or l_ear_conf < 0.2 or r_ear_conf < 0.2:
        return False
    if ear_ratio < HEAD_TURN_EAR_RATIO_MIN or ear_ratio > HEAD_TURN_EAR_RATIO_MAX:
        return True
    if norm_vertical_drop > PITCH_LEAN_NORM_DROP:
        return True
    return False


def _classify_anomaly(features: np.ndarray, flags: dict) -> Optional[str]:
    if not any(flags.values()):
        ear_ratio = float(features[0])
        norm_vertical_drop = float(features[1])
        if ear_ratio < HEAD_TURN_EAR_RATIO_MIN or ear_ratio > HEAD_TURN_EAR_RATIO_MAX:
            return "HEAD_TURNING"
        if norm_vertical_drop > PITCH_LEAN_NORM_DROP:
            return "PEEKING"
        return None
    if flags.get("multi_person"):
        return "NOTE_PASSING"
    if flags.get("head_down") or flags.get("excessive_lean"):
        return "PEEKING"
    if flags.get("body_turn"):
        return "HEAD_TURNING"
    return None


def _build_alert_zones(per_frame: list, fps: float) -> list:
    zones = []
    current_type = None
    start_frame = 0
    min_frames = max(3, int(fps * 0.4))
    severity = {"HEAD_TURNING": "medium", "PEEKING": "high", "NOTE_PASSING": "high"}

    def flush(end_frame: int):
        nonlocal current_type, start_frame
        if current_type is not None and (end_frame - start_frame) >= min_frames:
            zones.append({
                "start": round(start_frame / fps, 2),
                "end": round(end_frame / fps, 2),
                "type": current_type,
                "severity": severity.get(current_type, "medium"),
            })
        current_type = None

    for frame_idx, alert_type in per_frame:
        if alert_type != current_type:
            flush(frame_idx)
            current_type = alert_type
            start_frame = frame_idx
    flush(len(per_frame))

    merged = []
    for z in zones:
        if merged and merged[-1]["type"] == z["type"] and z["start"] <= merged[-1]["end"] + 0.05:
            merged[-1]["end"] = z["end"]
        else:
            merged.append(z)
    return merged


def process_upload_job(job_id: str, input_path: Path, output_dir: Path) -> None:
    """Background inference pipeline for uploaded test recordings."""
    try:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError("could not open uploaded video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _job_update(job_id, status="processing", progress=0, fps=fps, total_frames=total)

        detector = PoseDetector()
        per_frame = []
        w = h = 0
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
                    "ffmpeg", "-y", "-f", "rawvideo",
                    "-pixel_format", "bgr24",
                    "-video_size", f"{w}x{h}",
                    "-framerate", str(round(fps, 3)),
                    "-i", "-",
                    "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "23", "-pix_fmt", "yuv420p",
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
                    if idx < len(result.confidences)
                    else float(kpts[:, 2].max()) if kpts.size else 1.0
                )
                feats = extract_normalized_pose_features(kpts)
                is_anomaly = _evaluate_candidate(feats)
                alert_type = _classify_anomaly(result.anomaly_flags, feats) if is_anomaly else None
                if alert_type:
                    alerts.append({"id": cid, "type": alert_type, "conf": round(conf, 3)})

                x1, y1, x2, y2 = (int(v) for v in box)
                color = (0, 0, 255) if is_anomaly else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{cid}", (x1, max(y1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            primary = max(alerts, key=lambda a: {"NOTE_PASSING": 3, "PEEKING": 2, "HEAD_TURNING": 1}.get(a["type"], 0)) if alerts else None
            if primary:
                bcolor = (0, 0, 255) if primary["type"] != "HEAD_TURNING" else (0, 215, 255)
                cv2.rectangle(frame, (0, 0), (w, 46), bcolor, -1)
                cv2.putText(frame, f"ALERT: {primary['type']}", (12, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

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
        (output_dir / "analysis.json").write_text(
            __import__("json").dumps(sidecar)
        )
        _job_update(job_id, progress=100, status="done", result=sidecar)
        print(f"[upload] job {job_id} done: {frame_idx} frames", flush=True)
    except Exception as exc:
        _job_update(job_id, status="error", error=str(exc))
        print(f"[upload] job {job_id} failed: {exc}", flush=True)
    finally:
        input_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Upload endpoint
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    job_id: str
    status: str


@router.post("/upload-test-video", response_model=UploadResponse)
async def upload_test_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
):
    """
    Multipart file upload accepting MP4/AVI clips.
    Saves footage, kicks off background inference, returns a job UUID.
    """
    if not video.filename:
        raise HTTPException(status_code=400, detail="missing video filename")

    ext = Path(video.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"unsupported extension {ext}")

    job_id = uuid.uuid4().hex[:12]
    output_dir = UPLOAD_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / f"input{ext}"

    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "uploading",
            "progress": 0,
            "filename": video.filename,
        }

    content = await video.read()
    if len(content) > MAX_UPLOAD_BYTES:
        with _jobs_lock:
            _jobs.pop(job_id, None)
        raise HTTPException(status_code=413, detail="file exceeds 300 MB limit")

    input_path.write_bytes(content)

    _job_update(job_id, status="queued")
    background_tasks.add_task(process_upload_job, job_id, input_path, output_dir)

    return UploadResponse(job_id=job_id, status="queued")


# ---------------------------------------------------------------------------
# Job status endpoint
# ---------------------------------------------------------------------------

@router.get("/job-status/{job_id}")
async def job_status(job_id: str):
    """Pollable progress endpoint returning processing percentage and detected anomaly timestamps."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return job


# ---------------------------------------------------------------------------
# Webcam toggle endpoint
# ---------------------------------------------------------------------------

class WebcamToggleRequest(BaseModel):
    active: bool
    source_index: int = 0


class WebcamToggleResponse(BaseModel):
    active: bool
    source_index: int


@router.post("/webcam/toggle", response_model=WebcamToggleResponse)
async def webcam_toggle(req: WebcamToggleRequest):
    """
    Controls local webcam ingestion (ON/OFF) and state synchronization.
    Returns the updated webcam state.
    """
    _webcam_state["active"] = req.active
    _webcam_state["source_index"] = req.source_index
    print(f"[webcam] toggle -> active={req.active}, source={req.source_index}", flush=True)
    return WebcamToggleResponse(**_webcam_state)
