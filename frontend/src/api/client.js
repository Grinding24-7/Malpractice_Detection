// ---------------------------------------------------------------------------
// API client — single source of truth for backend connectivity.
//
// Endpoint contract (spec)   -> Flask backend route
// --------------------------   ------------------------------------------------
//   GET  /stream (MJPEG)     -> /video_feed            (backend-annotated CCTV)
//   POST /stream (frame)     -> /stream                (webcam inference, JSON)
//   GET  /evidence           -> /api/evidence          (evidence clip metadata)
//   GET  /vault/<clip>       -> /vault/<filename>      (mp4 clip download)
//   GET  /classrooms         -> /api/classrooms        (registered classrooms)
//   POST /classrooms         -> /api/classrooms        (register a classroom)
//   GET  /telemetry          -> /api/telemetry         (live candidate state)
//   GET  /dataset/<file>     -> /dataset/<file>        (training videos)
//   POST /upload-test-video -> /upload-test-video      (offline analysis job)
//   GET  /upload-job/<id>   -> /api/upload-job/<id>    (job progress/result)
//
// Every call has a graceful offline fallback so the UI remains usable (with a
// "DEMO" tag) when the Flask backend is not running.
// ---------------------------------------------------------------------------

export const API_BASE = '' // same-origin in prod; Vite proxy in dev

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
