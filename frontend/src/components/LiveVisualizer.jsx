import { useEffect, useRef, useState } from 'react'
import { Activity, Layers, MousePointer2 } from 'lucide-react'
import InferenceOverlay from './InferenceOverlay'
import { cctvStreamUrl, fetchMonitorAnnotations } from '../api/client'
import { demoAnnotations } from '../data/mockData'

const POLL_MS = 250

/**
 * LiveVisualizer — Module: Real-time Draw Engine.
 *
 * Streams the raw classroom CCTV footage (overlay=0) and superimposes the model
 * inference client-side, reading the /api/monitor/annotations data contract:
 * ByteTrack boxes colored by state, 17-keypoint skeletons, velocity gauges.
 * Polls the backend ~4 Hz; falls back to deterministic demo data when offline.
 */
export default function LiveVisualizer({ selected, onSelect, onAnnotations }) {
  const [overlay, setOverlay] = useState(true)
  const [imgKey, setImgKey] = useState(0)
  const [online, setOnline] = useState(true)
  const [annotations, setAnnotations] = useState(() => demoAnnotations(0))
  const boxRef = useRef(null)
  const demoRef = useRef(0)

  useEffect(() => {
    let stop = false
    const poll = async () => {
      try {
        const a = await fetchMonitorAnnotations()
        if (stop) return
        setAnnotations(a)
        setOnline(true)
      } catch {
        if (stop) return
        demoRef.current += 3
        setAnnotations(demoAnnotations(demoRef.current))
        setOnline(false)
      }
    }
    poll()
    const id = setInterval(poll, POLL_MS)
    return () => {
      stop = true
      clearInterval(id)
    }
  }, [])

  // Stream the live annotation frame up to the TelemetryPanel.
  useEffect(() => {
    if (onAnnotations) onAnnotations(annotations)
  }, [annotations, onAnnotations])

  const students = annotations.students || []
  const badge = online ? 'LIVE' : 'DEMO'

  // Hit test: map a click on the video box to a student (video is 1280x720).
  const handleClick = (e) => {
    if (!boxRef.current) return
    const rect = boxRef.current.getBoundingClientRect()
    const x = ((e.clientX - rect.left) / rect.width) * 1280
    const y = ((e.clientY - rect.top) / rect.height) * 720
    let best = null
    let bestDist = Infinity
    for (const s of students) {
      const [x1, y1, x2, y2] = s.bbox
      if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
        onSelect(s.track_id)
        return
      }
      const cx = (x1 + x2) / 2
      const cy = (y1 + y2) / 2
      const d = Math.hypot(x - cx, y - cy)
      if (d < bestDist && d < 160) {
        bestDist = d
        best = s.track_id
      }
    }
    if (best != null) onSelect(best)
  }

  return (
    <div className="glass flex flex-col overflow-hidden rounded-2xl">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-4 py-3">
        <div className="flex items-center gap-2">
          <Activity size={17} className="text-accent" />
          <span className="text-sm font-semibold">Live Visualizer</span>
          <span
            className={`rounded-full px-2 py-0.5 text-[10px] font-bold tracking-wider ${
              online ? 'bg-emerald-500/15 text-emerald-500' : 'bg-warning/15 text-warning'
            }`}
          >
            {badge}
          </span>
        </div>
        <div className="flex items-center gap-3">
          <label className="flex cursor-pointer items-center gap-2 text-xs font-medium text-muted">
            <input
              type="checkbox"
              checked={overlay}
              onChange={(e) => {
                setOverlay(e.target.checked)
                setImgKey((k) => k + 1)
              }}
              className="accent-indigo-500"
            />
            <Layers size={13} />
            Inference overlay
          </label>
          <span className="hidden font-mono text-xs text-muted sm:block">
            frame {annotations.frame_id} · t={annotations.timestamp}s · {annotations.fps || 0} fps
          </span>
        </div>
      </div>

      <div
        ref={boxRef}
        onClick={handleClick}
        className="relative aspect-video w-full cursor-crosshair bg-black"
      >
        <img
          key={imgKey}
          src={cctvStreamUrl(false)}
          className="h-full w-full object-contain"
          alt="Raw classroom CCTV feed"
        />
        <InferenceOverlay students={overlay ? students : []} selected={selected} />
        <div className="pointer-events-none absolute left-2 top-2 rounded-md bg-black/60 px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-white/80">
          Raw source
        </div>
        <div className="pointer-events-none absolute bottom-2 right-2 flex items-center gap-1 rounded-md bg-black/60 px-2 py-1 text-[10px] text-white/70">
          <MousePointer2 size={11} />
          click a student to inspect
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-line px-4 py-3">
        <div className="flex flex-wrap items-center gap-3 text-[11px] text-muted">
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-emerald-400" /> Normal
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-amber-400" /> Suspicion
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-red-400" /> Alert
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-fuchsia-400" /> Skeleton
          </span>
        </div>
        <span className="text-xs text-muted">
          {students.length} tracked {students.length === 1 ? 'student' : 'students'} ·{' '}
          {annotations.source_type}
        </span>
      </div>
    </div>
  )
}
