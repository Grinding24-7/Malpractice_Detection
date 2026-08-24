// ---------------------------------------------------------------------------
// Mock / fallback data.
//
// These structures mirror what the backend API returns so the dashboard stays
// fully interactive even when the Flask service is offline. Records loaded from
// the real API always take precedence (the views tag fallback data with
// `source: 'demo'` so it is never mistaken for live evidence).
// ---------------------------------------------------------------------------

export const DATASET_VIDEOS = [
  { id: 'sample_exam.mp4', name: 'Sample Exam' },
  { id: 'multi_classroom_test.mp4', name: 'Multi-Classroom Test' },
  { id: '1_hour_exam_test.mp4', name: '1-Hour Exam Test' },
]

export const MOCK_CLASSROOMS = [
  {
    id: 'room-a',
    name: 'Room A — Block 1',
    camera_ip: '192.168.10.21',
    rtsp_url: 'rtsp://192.168.10.21:554/live',
    desk_rows: 6,
    desk_cols: 4,
    roster: Array.from({ length: 24 }, (_, i) => `Student ${i + 1}`),
    created_at: '2026-08-01T09:00:00Z',
    source: 'demo',
  },
  {
    id: 'room-b',
    name: 'Room B — Block 2',
    camera_ip: '192.168.10.22',
    rtsp_url: 'rtsp://192.168.10.22:554/live',
    desk_rows: 5,
    desk_cols: 5,
    roster: Array.from({ length: 25 }, (_, i) => `Student ${i + 1}`),
    created_at: '2026-08-01T09:10:00Z',
    source: 'demo',
  },
]

const SEVERITIES = ['low', 'medium', 'high']
const TYPES = ['Head Turn', 'Peeking', 'Note Passing']

export const MOCK_EVIDENCE = Array.from({ length: 12 }, (_, i) => {
  const m = new Date(Date.UTC(2026, 7, 12 - (i % 6), 8 + (i % 9), (i * 7) % 60))
  return {
    id: `candidate_${(i % 4) + 1}_alert_demo${i}`,
    name: `candidate_${(i % 4) + 1}_alert_demo${i}.mp4`,
    classroom: i % 2 ? 'Room B — Block 2' : 'Room A — Block 1',
    type: TYPES[i % 3],
    severity: SEVERITIES[i % 3],
    recorded_at: m.toISOString(),
    duration_s: 30,
    source: 'demo',
  }
})

const MOCK_PERSONS = [
  { id: 1, box: [60, 90, 210, 430], confidence: 0.93, status: 'ANOMALY' },
  { id: 2, box: [430, 120, 580, 450], confidence: 0.88, status: 'NOMINAL' },
]

// COCO-17 skeleton topology used by the overlay canvas.
export const SKELETON = [
  [0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9],
  [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15],
  [12, 14], [14, 16],
]

// Visual contract for the inspector: status -> stroke color + human label.
// GREEN = normal, AMBER = suspicion threshold breached, RED = malpractice.
export const STATUS_META = {
  NOMINAL: { color: '#34d399', label: 'Normal Writing' },
  HEAD_TURNING: { color: '#fbbf24', label: 'Head Turning' },
  LEANING: { color: '#fbbf24', label: 'Leaning' },
  PEEKING: { color: '#f87171', label: 'Peeking / Alert' },
  ANOMALY: { color: '#f87171', label: 'Anomaly' },
}

/** Dataset videos play back at the capture FPS (matches CAM_FPS in app.py). */
export const DEMO_FPS = 30

// Deterministic pseudo-random for stable demo keypoints.
function mulberry32(seed) {
  return function () {
    seed |= 0
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// 17 keypoints roughly matching a seated person (x, y, confidence).
const BASE_SKELETON = [
  [0.50, 0.18], [0.46, 0.20], [0.54, 0.20], [0.42, 0.30], [0.58, 0.30],
  [0.32, 0.42], [0.68, 0.42], [0.26, 0.55], [0.74, 0.55], [0.24, 0.70],
  [0.76, 0.70], [0.34, 0.46], [0.66, 0.46], [0.36, 0.62], [0.64, 0.62],
  [0.38, 0.78], [0.62, 0.78],
]

/** Demo keypoints for one person, jittered per call for a "live" feel. */
export function demoKeypoints(frame = 0) {
  const rng = mulberry32(frame * 7919 + 13)
  const turn = Math.sin(frame * 0.21) * 0.06
  return BASE_SKELETON.map(([x, y], i) => {
    const jx = (rng() - 0.5) * 0.02
    const jy = (rng() - 0.5) * 0.02
    const shifted = i === 0 || i === 1 || i === 2 ? x + turn : x
    return [shifted + jx, y + jy, 0.82 + rng() * 0.15]
  })
}

/** Demo persons with boxes derived from the skeleton bounds. */
export function demoPersons(frame = 0) {
  return MOCK_PERSONS.map((p) => {
    const kpts = demoKeypoints(frame + p.id * 137)
    const xs = kpts.map((k) => k[0])
    const ys = kpts.map((k) => k[1])
    const [x1, y1, x2, y2] = p.box
    const w = x2 - x1
    const h = y2 - y1
    return {
      ...p,
      box: [
        Math.round(x1 + Math.min(...xs) * w * 0.9),
        Math.round(y1 + Math.min(...ys) * h * 0.9),
        Math.round(x1 + Math.max(...xs) * w * 1.1),
        Math.round(y1 + Math.max(...ys) * h * 1.1),
      ],
      keypoints: kpts,
      head: { ear_ratio: 0.6 + 0.3 * Math.abs(Math.sin(frame * 0.13)), norm_vertical_drop: 0.4 + 0.4 * Math.abs(Math.cos(frame * 0.11)) },
    }
  })
}

export function demoAlert(frame) {
  const mod = Math.floor(frame / 40) % 3
  const types = ['NORMAL', 'HEAD TURNING', 'PEEKING']
  const active = mod !== 0
  return { active, type: types[mod], demo: true }
}

/**
 * Demo inference frame in the monitoring data contract:
 *   {timestamp, frame_id, source_type, students: [{track_id, bbox, status,
 *    confidence, keypoints, velocity_spike, ear_ratio, norm_vertical_drop}]}
 * Deterministic in `frame` so scrubbing the dataset video shows stable poses.
 */
export function demoAnnotations(frame = 0) {
  const persons = demoPersons(frame)
  const phase = Math.floor(frame / 40) % 3
  const statuses = ['NOMINAL', 'HEAD_TURNING', 'PEEKING']
  const students = persons.map((p) => {
    const s = p.id === 1 && phase !== 0 ? statuses[phase] : 'NOMINAL'
    return {
      track_id: p.id,
      bbox: p.box,
      status: s,
      confidence: p.confidence,
      keypoints: p.keypoints,
      velocity_spike: Number((0.12 + 0.45 * Math.abs(Math.sin(frame * 0.21 + p.id * 2.7))).toFixed(3)),
      ear_ratio: p.head.ear_ratio,
      norm_vertical_drop: p.head.norm_vertical_drop,
    }
  })
  return {
    timestamp: Number((frame / DEMO_FPS).toFixed(2)),
    frame_id: frame,
    source_type: 'demo',
    source_path: null,
    fps: 30,
    students,
  }
}

/** COCO wrist indices used by the hand-motion metric. */
const WRISTS = [9, 10]

/**
 * Hand spatial velocity between two keypoint frames, normalized by the max
 * shoulder width in the scene (scale-invariant, mirrors velocity_spike).
 * Returns 0 when either frame has no usable wrist keypoints.
 */
export function handVelocity(prevStudents, nextStudents) {
  const width = Math.max(
    ...nextStudents.map((s) => {
      const k = s.keypoints || []
      if (k.length < 7) return 40
      return Math.hypot(k[5][0] - k[6][0], k[5][1] - k[6][1])
    }),
    40,
  )
  let best = 0
  for (const n of nextStudents) {
    const p = prevStudents.find((s) => s.track_id === n.track_id)
    if (!p) continue
    const nk = n.keypoints || []
    const pk = p.keypoints || []
    if (nk.length < 11 || pk.length < 11) continue
    for (const wi of WRISTS) {
      const d = Math.hypot(nk[wi][0] - pk[wi][0], nk[wi][1] - pk[wi][1])
      best = Math.max(best, d / width)
    }
  }
  return Number(Math.min(best, 2).toFixed(3))
}

/**
 * Deterministic (B, T, F) pseudo feature tensor for the alignment inspector.
 * B = samples, T = frames @ 5 fps, F = channels. Mirrors the Week 4/5 tensor
 * shape used by `TemporalFeatureExtractor` (F = 117 in production).
 */
export function makeFeatureTensor(b = 4, t = 150, f = 117) {
  const rng = mulberry32(20260815)
  const tensor = Array.from({ length: b }, () =>
    Array.from({ length: t }, () =>
      Array.from({ length: f }, () => rng() * 0.15),
    ),
  )
  // Inject deterministic "velocity spikes" (head-motion bursts) across a few
  // frames so the aligned chart has visible peaks.
  for (let bi = 0; bi < b; bi++) {
    for (let s = 0; s < 5; s++) {
      const start = 12 + s * 28
      for (let dt = 0; dt < 5; dt++) {
        const tIdx = start + dt
        if (tIdx < t) {
          for (let fi = 0; fi < f; fi++) {
            tensor[bi][tIdx][fi] += (0.6 - dt * 0.1) * (0.5 + rng() * 0.5)
          }
        }
      }
    }
  }
  return tensor
}

/** Downstream view: per-frame aggregate velocity (norm of channel deltas). */
export function velocitySeries(tensor) {
  const b = tensor.length
  const t = tensor[0].length
  const f = tensor[0][0].length
  return Array.from({ length: t }, (_, ti) => {
    let total = 0
    for (let bi = 0; bi < b; bi++) {
      for (let fi = 0; fi < f; fi++) {
        if (ti > 0) {
          const d = tensor[bi][ti][fi] - tensor[bi][ti - 1][fi]
          total += d * d
        }
      }
    }
    return Number(Math.sqrt(total / Math.max(b * f, 1)).toFixed(4))
  })
}
