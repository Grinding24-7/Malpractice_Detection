import { useEffect, useRef } from 'react'
import { SKELETON, STATUS_META } from '../data/mockData'

/**
 * Draw engine for the Model Inspector (data contract students[]).
 *
 * Rendered client-side so box color, skeleton and labels can react to the
 * live model output without waiting for a backend-reencoded frame:
 *   - GREEN box  = normal writing behaviour
 *   - AMBER box  = suspicion threshold breached (head turning / leaning)
 *   - RED box    = malpractice alert (peeking / sustained anomaly)
 *   - magenta skeleton + yellow joints over the COCO-17 keypoints
 *   - a horizontal velocity gauge under each box (velocity_spike)
 *   - `ID #xx | 94%` label with persistent ByteTrack track_id
 *
 * Coordinates are already in video pixel space (1280x720), matching the
 * object-contain video/canvas layout used across the dashboard.
 */

export const drawStudents = (ctx, students, W, H, opts = {}) => {
  const { showSkeleton = true, showTrail = true, selected = null } = opts
  ctx.clearRect(0, 0, W, H)
  if (!Array.isArray(students)) return

  for (const s of students) {
    const [x1, y1, x2, y2] = s.bbox
    const meta = STATUS_META[s.status] || STATUS_META.NOMINAL
    const color = meta.color
    const isSel = selected === s.track_id

    // Bounding box — thicker + white corner ticks when selected.
    ctx.lineWidth = isSel ? 4 : 2.5
    ctx.strokeStyle = color
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
    if (isSel) {
      ctx.strokeStyle = '#ffffff'
      ctx.lineWidth = 1.5
      const tick = 12
      for (const [cx, cy, dx, dy] of [
        [x1, y1, 1, 1], [x2, y1, -1, 1], [x1, y2, 1, -1], [x2, y2, -1, -1],
      ]) {
        ctx.beginPath()
        ctx.moveTo(cx + dx * tick, cy)
        ctx.lineTo(cx, cy)
        ctx.lineTo(cx, cy + dy * tick)
        ctx.stroke()
      }
    }

    // Velocity spike gauge beneath the box.
    if (showTrail && typeof s.velocity_spike === 'number') {
      const gw = x2 - x1
      const gy = y2 + 6
      ctx.fillStyle = 'rgba(15, 23, 42, 0.45)'
      ctx.fillRect(x1, gy, gw, 4)
      ctx.fillStyle = color
      ctx.fillRect(x1, gy, gw * Math.min(1, s.velocity_spike), 4)
    }

    // Track ID + confidence label.
    const tag = `ID #${String(s.track_id).padStart(2, '0')} | ${Math.round((s.confidence || 0) * 100)}%`
    ctx.font = 'bold 12px ui-monospace, monospace'
    const tw = ctx.measureText(tag).width
    const ly = Math.max(y1 - 22, 0)
    ctx.fillStyle = color
    ctx.fillRect(x1, ly, tw + 10, 18)
    ctx.fillStyle = '#0f172a'
    ctx.fillText(tag, x1 + 5, ly + 13)

    // 17-keypoint skeleton.
    if (showSkeleton && Array.isArray(s.keypoints) && s.keypoints.length >= 17) {
      const kp = s.keypoints
      ctx.strokeStyle = '#e879f9'
      ctx.lineWidth = 2
      for (const [a, b] of SKELETON) {
        const pa = kp[a]
        const pb = kp[b]
        if (pa && pb && (pa[2] ?? 1) > 0.3 && (pb[2] ?? 1) > 0.3) {
          ctx.beginPath()
          ctx.moveTo(pa[0], pa[1])
          ctx.lineTo(pb[0], pb[1])
          ctx.stroke()
        }
      }
      ctx.fillStyle = '#fde047'
      for (const k of kp) {
        if ((k[2] ?? 1) > 0.3) {
          ctx.beginPath()
          ctx.arc(k[0], k[1], 3, 0, Math.PI * 2)
          ctx.fill()
        }
      }
    }
  }
}

/**
 * InferenceOverlay — absolute-positioned canvas that renders `students` on top
 * of a video/img element that shares the same aspect-ratio box.
 */
export default function InferenceOverlay({ students = [], selected = null, enabled = true }) {
  const ref = useRef(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const W = canvas.clientWidth || 640
    const H = canvas.clientHeight || 360
    const dpr = window.devicePixelRatio || 1
    canvas.width = W * dpr
    canvas.height = H * dpr
    const ctx = canvas.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    if (enabled) drawStudents(ctx, students, W, H, { selected })
    else ctx.clearRect(0, 0, W, H)
  }, [students, selected, enabled])

  return (
    <canvas
      ref={ref}
      className="pointer-events-none absolute inset-0 h-full w-full object-contain"
    />
  )
}
