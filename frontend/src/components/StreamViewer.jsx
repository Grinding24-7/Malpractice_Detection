/**
 * StreamViewer.jsx — HTML5 Canvas consumer for real-time WebSocket video stream.
 *
 * Renders:
 *   - Incoming base64 JPEG frames on a <canvas> element
 *   - ByteTrack bounding boxes (color-coded by prediction status)
 *   - 17-point COCO pose skeletons
 *   - Status tags (NORMAL / HEAD_TURNING / PEEKING / SUSPICIOUS)
 *   - Velocity spike gauges per student
 *
 * Theme integration:
 *   - Reads dark/light state from ThemeProvider context
 *   - Maps status → canvas stroke colors:
 *       NORMAL     → #22c55e (green)
 *       HEAD_TURNING → #f59e0b (amber)
 *       PEEKING    → #ef4444 (red)
 *       SUSPICIOUS → #ef4444 (red)
 */

import { useEffect, useRef, useMemo } from 'react'
import { useVideoStream } from '../hooks/useVideoStream'
import { WS_BASE } from '../api/client'

// COCO-17 skeleton topology
const SKELETON = [
  [0, 1], [0, 2], [1, 3], [2, 4],        // face
  [5, 6],                                   // shoulders
  [5, 7], [7, 9],                           // left arm
  [6, 8], [8, 10],                          // right arm
  [5, 11], [6, 12], [11, 12],              // torso
  [11, 13], [13, 15],                       // left leg
  [12, 14], [14, 16],                       // right leg
]

// Status → color mapping (matches CSS theme variables)
const STATUS_COLORS = {
  NORMAL:       '#22c55e',
  HEAD_TURNING: '#f59e0b',
  PEEKING:      '#ef4444',
  SUSPICIOUS:   '#ef4444',
  NOMINAL:      '#22c55e',
  ANOMALY:      '#ef4444',
}

const STATUS_LABELS = {
  NORMAL:       'Normal Writing',
  HEAD_TURNING: 'Head Turning',
  PEEKING:      'Peeking',
  SUSPICIOUS:   'Suspicious',
  NOMINAL:      'Normal',
  ANOMALY:      'Anomaly',
}

/**
 * Draw students' bounding boxes, skeletons, and status tags on a canvas context.
 */
function drawOverlay(ctx, students, W, H, dark) {
  ctx.clearRect(0, 0, W, H)
  if (!Array.isArray(students) || students.length === 0) return

  for (const s of students) {
    const bbox = s.bbox
    if (!bbox || bbox.length < 4) continue
    const [x1, y1, x2, y2] = bbox
    const pred = s.prediction || 'NORMAL'
    const color = STATUS_COLORS[pred] || STATUS_COLORS.NORMAL

    // Bounding box
    ctx.lineWidth = 2.5
    ctx.strokeStyle = color
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

    // Status label tag
    const label = STATUS_LABELS[pred] || pred
    const tag = `ID #${String(s.track_id).padStart(2, '0')} | ${label}`
    ctx.font = 'bold 11px ui-monospace, monospace'
    const tw = ctx.measureText(tag).width
    const ly = Math.max(y1 - 22, 0)
    ctx.fillStyle = color
    ctx.fillRect(x1, ly, tw + 10, 18)
    ctx.fillStyle = dark ? '#0f172a' : '#ffffff'
    ctx.fillText(tag, x1 + 5, ly + 13)

    // Confidence badge
    const confText = `${Math.round((s.confidence || 0) * 100)}%`
    ctx.font = '10px ui-monospace, monospace'
    const ctw = ctx.measureText(confText).width
    ctx.fillStyle = 'rgba(0,0,0,0.6)'
    ctx.fillRect(x1 + tw + 14, ly, ctw + 8, 18)
    ctx.fillStyle = '#ffffff'
    ctx.fillText(confText, x1 + tw + 18, ly + 13)

    // Velocity spike gauge beneath box
    if (typeof s.velocity_spike === 'number') {
      const gw = x2 - x1
      const gy = y2 + 6
      ctx.fillStyle = 'rgba(15, 23, 42, 0.45)'
      ctx.fillRect(x1, gy, gw, 4)
      ctx.fillStyle = color
      ctx.fillRect(x1, gy, gw * Math.min(1, s.velocity_spike), 4)
    }

    // 17-keypoint skeleton
    const kps = s.keypoints
    if (Array.isArray(kps) && kps.length >= 17) {
      ctx.strokeStyle = '#e879f9'
      ctx.lineWidth = 2
      for (const [a, b] of SKELETON) {
        const pa = kps[a]
        const pb = kps[b]
        if (pa && pb && (pa[2] ?? 1) > 0.3 && (pb[2] ?? 1) > 0.3) {
          ctx.beginPath()
          ctx.moveTo(pa[0], pa[1])
          ctx.lineTo(pb[0], pb[1])
          ctx.stroke()
        }
      }
      // Keypoint dots
      ctx.fillStyle = '#fde047'
      for (const k of kps) {
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
 * StreamViewer — self-contained component that connects to the WebSocket
 * stream and renders video frames + overlay on an HTML5 canvas.
 *
 * Props:
 *   wsUrl     — WebSocket endpoint (default: ws://localhost:8000/api/v1/stream/ws)
 *   dark      — theme mode (default: true)
 *   className — optional CSS class for the container
 *   showOverlay — render detection overlay (default: true)
 */
export default function StreamViewer({
  wsUrl = `${WS_BASE}/api/v1/stream/ws`,
  dark = true,
  className = '',
  showOverlay = true,
}) {
  const canvasRef = useRef(null)
  const imgRef = useRef(null)
  const animFrameRef = useRef(null)
  const { connected, frame, students, frameId, error } = useVideoStream(wsUrl)

  // Draw frame + overlay whenever new data arrives
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !frame) return

    const ctx = canvas.getContext('2d')
    const img = imgRef.current || new Image()
    imgRef.current = img

    img.onload = () => {
      // Size canvas to match image
      if (canvas.width !== img.naturalWidth || canvas.height !== img.naturalHeight) {
        canvas.width = img.naturalWidth
        canvas.height = img.naturalHeight
      }
      ctx.drawImage(img, 0, 0)
      if (showOverlay) {
        drawOverlay(ctx, students, canvas.width, canvas.height, dark)
      }
    }

    img.src = `data:image/jpeg;base64,${frame}`
  }, [frame, students, dark, showOverlay])

  return (
    <div className={`relative overflow-hidden ${className}`}>
      {/* Connection status badge */}
      <div className="absolute top-2 right-2 z-10 flex items-center gap-2">
        <span
          className={`inline-block h-2 w-2 rounded-full ${
            connected ? 'bg-green-400 animate-pulse' : 'bg-red-400'
          }`}
        />
        <span className="text-xs font-mono opacity-70">
          {connected ? `LIVE — F#${frameId}` : 'RECONNECTING...'}
        </span>
      </div>

      {/* Error overlay */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/60 z-20">
          <div className="text-red-400 text-sm font-mono">{error}</div>
        </div>
      )}

      {/* Canvas stream */}
      <canvas
        ref={canvasRef}
        className="w-full h-full object-contain bg-black"
        style={{ imageRendering: 'auto' }}
      />

      {/* Placeholder when no frame yet */}
      {!frame && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-gray-400 text-sm font-mono animate-pulse">
            {connected ? 'Waiting for first frame...' : 'Connecting to stream...'}
          </div>
        </div>
      )}
    </div>
  )
}
