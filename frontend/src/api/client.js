// ---------------------------------------------------------------------------
// API client — single source of truth for backend connectivity.
//
// Week 6 FastAPI endpoints (new):
//   WS   /api/v1/stream/ws       — WebSocket telemetry + compressed frames
//   GET  /api/v1/stream/mjpeg    — MJPEG stream with ByteTrack overlays
//   POST /api/v1/upload-test-video — Multipart upload → background job
//   GET  /api/v1/job-status/<id>  — Pollable progress endpoint
//   POST /api/v1/webcam/toggle    — Webcam ingestion ON/OFF
//
// Legacy Flask endpoints (still available):
//   GET  /video_feed              — MJPEG CCTV stream
//   POST /stream                  — Webcam inference (per-frame)
//   GET  /api/telemetry           — Live candidate state
//   GET  /api/monitor/annotations — Monitoring data contract
//   GET  /api/evidence            — Evidence vault metadata
//   GET  /api/classrooms          — Classroom registry
//
// Every call has a graceful offline fallback so the UI remains usable (with a
// "DEMO" tag) when the backend is not running.
// ---------------------------------------------------------------------------

export const API_BASE = '' // same-origin in prod; Vite proxy in dev

// Week 6: WebSocket base URL (FastAPI on port 8000 in dev, same-origin in prod)
export const WS_BASE = (() => {
  if (typeof window === 'undefined') return ''
  const host = window.location.hostname
  // In dev mode (Vite on 5173), connect directly to FastAPI on 8000
  if (window.location.port === '5173') return `ws://${host}:8000`
  // In production, same-origin WebSocket (wss:// if HTTPS)
  return window.location.protocol === 'https:' ? `wss://${host}` : `ws://${host}`
})()

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  })
  if (!res.ok) {
    throw new Error(`API ${path} failed: ${res.status} ${res.statusText}`)
  }
  const text = await res.text()
  return text ? JSON.parse(text) : null
}

/** MJPEG classroom CCTV stream URL (served by /video_feed). */
export const cctvStreamUrl = (overlay = true) =>
  `${API_BASE}/video_feed${overlay ? '' : '?overlay=0'}`

/** Absolute URL for an evidence clip in the vault. */
export const clipUrl = (filename) => `${API_BASE}/vault/${encodeURIComponent(filename)}`

/** Absolute URL for a classroom dataset video used by the alignment view. */
export const datasetUrl = (filename) => `${API_BASE}/dataset/${encodeURIComponent(filename)}`

/**
 * Live telemetry — {fps, active_candidates, anomaly_count, candidates[],
 * yaw, pitch, alert, ...}. Polled by the live view.
 */
export const fetchTelemetry = () => request('/api/telemetry')

/**
 * Latest model inference frame (monitoring data contract):
 * {timestamp, frame_id, source_type, source_path, fps, students: [{track_id,
 * bbox, status, confidence, keypoints, velocity_spike, ...}]}. Polled by the
 * Model Inspector's LiveVisualizer; falls back to demo data offline.
 */
export const fetchMonitorAnnotations = () => request('/api/monitor/annotations')

/**
 * POST a webcam snapshot to the backend and get back tracked persons
 * (boxes, COCO-17 keypoints, track ids, confidence) + the aggregate alert.
 * @param {string} imageDataUrl  base64 JPEG from react-webcam
 */
export const postStreamFrame = (imageDataUrl) =>
  request('/stream', {
    method: 'POST',
    body: JSON.stringify({ image: imageDataUrl, quality: 72 }),
  })

/**
 * Evidence vault list. Backend returns [{id, name, url, classroom, type,
 * severity, recorded_at, duration_s}] — see /api/evidence in app.py.
 */
export const fetchEvidence = () => request('/api/evidence')

/** Registered classrooms: [{id, name, camera_ip, rtsp_url, desk_rows, ...}]. */
export const fetchClassrooms = () => request('/api/classrooms')

/** Register a new classroom CCTV configuration. */
export const createClassroom = (payload) =>
  request('/api/classrooms', { method: 'POST', body: JSON.stringify(payload) })

/** Absolute URL for a completed analysis job's artifacts. */
export const uploadJobUrl = (path) => `${API_BASE}${path}`

/**
 * Upload a test recording and start the offline analysis pipeline
 * (POST /upload-test-video). Returns {job_id}.
 */
export const uploadTestVideo = async (file) => {
  const form = new FormData()
  form.append('video', file)
  const res = await fetch(`${API_BASE}/upload-test-video`, { method: 'POST', body: form })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.error || `upload failed: ${res.status}`)
  }
  return res.json()
}

/** Poll the status of an analysis job — {status, progress, result, error}. */
export const getUploadJob = (jobId) => request(`/api/upload-job/${jobId}`)

// ---------------------------------------------------------------------------
// Week 6: FastAPI streaming endpoints
// ---------------------------------------------------------------------------

/** WebSocket URL for the real-time streaming endpoint. */
export const streamWsUrl = () => `${WS_BASE}/api/v1/stream/ws`

/** MJPEG stream URL (FastAPI backend, processed frames). */
export const streamMjpegUrl = () => `${API_BASE}/api/v1/stream/mjpeg`

/**
 * Upload a test recording via the FastAPI backend (POST /api/v1/upload-test-video).
 * Returns {job_id, status}.
 */
export const uploadTestVideoV1 = async (file) => {
  const form = new FormData()
  form.append('video', file)
  const res = await fetch(`${API_BASE}/api/v1/upload-test-video`, { method: 'POST', body: form })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || body.error || `upload failed: ${res.status}`)
  }
  return res.json()
}

/** Poll the status of a FastAPI analysis job — {status, progress, result, error}. */
export const getJobStatusV1 = (jobId) => request(`/api/v1/job-status/${jobId}`)

/** Toggle webcam ingestion on/off (POST /api/v1/webcam/toggle). */
export const toggleWebcam = (active, sourceIndex = 0) =>
  request('/api/v1/webcam/toggle', {
    method: 'POST',
    body: JSON.stringify({ active, source_index: sourceIndex }),
  })

/** Health check for the FastAPI streaming backend. */
export const healthCheck = () => request('/api/v1/health')
