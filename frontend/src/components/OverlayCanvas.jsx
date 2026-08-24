import { useEffect, useRef } from 'react'
import { SKELETON } from '../data/mockData'

/**
 * OverlayCanvas — draws the real-time detection overlay on top of the live
 * webcam snapshot:
 *   - bounding boxes + track IDs + confidence score per person
 *   - COCO-17 keypoint skeleton + joints
 *
 * Coordinates come from POST /stream responses (already in webcam pixel
 * space). When `persons` is empty the canvas is cleared.
 */
export default function OverlayCanvas({ persons = [], width = 1280, height = 720, enabled = true }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const dpr = window.devicePixelRatio || 1
    canvas.width = width
    canvas.height = height
    canvas.style.width = `${width}px`
    canvas.style.height = `${height}px`
    canvas.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0)

    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, width, height)
    if (!enabled) return

    for (const p of persons) {
      const [x1, y1, x2, y2] = p.box
      const anomaly = p.status === 'ANOMALY'
      // Bounding box + track id + confidence.
      ctx.strokeStyle = anomaly ? '#ef4444' : '#34d399'
      ctx.lineWidth = 3
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
      const tag = `ID:${p.id} ${(p.confidence * 100).toFixed(0)}%`
      ctx.font = 'bold 14px ui-monospace, monospace'
      const tw = ctx.measureText(tag).width
      ctx.fillStyle = anomaly ? '#ef4444' : '#10b981'
      ctx.fillRect(x1, Math.max(y1 - 22, 0), tw + 10, 20)
      ctx.fillStyle = '#ffffff'
      ctx.fillText(tag, x1 + 5, Math.max(y1 - 7, 14))

      // Skeleton.
      if (p.keypoints) {
        ctx.strokeStyle = '#a78bfa'
        ctx.lineWidth = 2
        for (const [a, b] of SKELETON) {
          const pa = p.keypoints[a]
          const pb = p.keypoints[b]
          if (pa && pb && pa[2] > 0.3 && pb[2] > 0.3) {
            ctx.beginPath()
            ctx.moveTo(pa[0], pa[1])
            ctx.lineTo(pb[0], pb[1])
            ctx.stroke()
          }
        }
        for (const [x, y, c] of p.keypoints) {
          if (c > 0.3) {
            ctx.fillStyle = '#fbbf24'
            ctx.beginPath()
            ctx.arc(x, y, 3.5, 0, Math.PI * 2)
            ctx.fill()
          }
        }
      }
    }
  }, [persons, width, height, enabled])

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none absolute inset-0 h-full w-full object-contain"
    />
  )
}
