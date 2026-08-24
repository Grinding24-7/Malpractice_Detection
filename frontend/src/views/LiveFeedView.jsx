import { useEffect, useMemo, useState } from 'react'
import { MonitorPlay, Layers } from 'lucide-react'
import StatCard from '../components/StatCard'
import AlertBadge from '../components/AlertBadge'
import WebcamPanel, { deriveAlert } from '../components/WebcamPanel'
import { useAsync } from '../hooks/useAsync'
import { cctvStreamUrl, fetchTelemetry } from '../api/client'

/** Derive the aggregate alert from /api/telemetry candidate states. */
function telemetryAlert(tel) {
  if (!tel || !Array.isArray(tel.candidates) || tel.candidates.length === 0) {
    return { active: false, type: 'NORMAL' }
  }
  const any = tel.candidates.some((c) => c.status === 'ANOMALY')
  if (!any) return { active: false, type: 'NORMAL' }
  const turning = tel.candidates.some(
    (c) => c.ear_ratio < 0.7 || c.ear_ratio > 1.4,
  )
  const peeking = tel.candidates.some((c) => c.norm_vertical_drop > 0.9)
  if (peeking) return { active: true, type: 'PEEKING' }
  if (turning) return { active: true, type: 'HEAD TURNING' }
  return { active: true, type: 'NOTE PASSING' }
}

/**
 * LiveFeedView — Module 1: Real-Time Testing & Model Inference.
 * Left: classroom CCTV MJPEG stream with a toggleable backend overlay.
 * Right: interactive webcam inference panel.
 */
export default function LiveFeedView({ classroom }) {
  const [cctvOverlay, setCctvOverlay] = useState(true)
  const [backendUp, setBackendUp] = useState(true)
  const [imgKey, setImgKey] = useState(0)

  const { data: tel } = useAsync(
    () => fetchTelemetry().catch(() => null),
    [],
    600,
  )

  useEffect(() => {
    if (tel) setBackendUp(true)
    else setBackendUp(false)
  }, [tel])

  const alert = useMemo(() => telemetryAlert(tel), [tel])
  const candidates = (tel && tel.candidates) || []

  return (
    <div className="fade-in space-y-5">
      {/* KPI strip */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard label="FPS" value={tel ? tel.fps : '—'} tone="accent" />
        <StatCard label="Active Candidates" value={candidates.length} tone="accent2" />
        <StatCard label="Anomaly Count" value={tel ? tel.anomaly_count : '—'} tone="danger" />
        <StatCard label="Temporal Windows" value={tel ? tel.temporal_windows_ready : '—'} tone="muted" />
      </div>

      {/* Live panels */}
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
        {/* Classroom CCTV */}
        <div className="glass flex flex-col overflow-hidden rounded-2xl">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-4 py-3">
            <div className="flex items-center gap-2">
              <MonitorPlay size={17} className="text-accent-2" />
              <span className="text-sm font-semibold">
                Classroom CCTV — {classroom || 'All'}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <label className="flex cursor-pointer items-center gap-2 text-xs font-medium text-muted">
                <input
                  type="checkbox"
                  checked={cctvOverlay}
                  onChange={(e) => {
                    setCctvOverlay(e.target.checked)
                    setImgKey((k) => k + 1) // re-request with overlay param
                  }}
                  className="accent-indigo-500"
                />
                <Layers size={13} />
                Overlay
              </label>
              <AlertBadge type={alert.type} demo={!backendUp} />
            </div>
          </div>

          <div className="relative aspect-video w-full bg-black">
            {backendUp ? (
              <img
                key={imgKey}
                src={cctvStreamUrl(cctvOverlay)}
                className="h-full w-full object-contain"
                alt="Classroom CCTV feed"
              />
            ) : (
              <div className="flex h-full w-full flex-col items-center justify-center gap-3 p-6 text-center text-muted">
                <MonitorPlay size={40} />
                <p className="text-sm font-medium">Backend is offline</p>
                <p className="max-w-sm text-xs">
                  Start the Flask server (port 5000) to view the live CCTV
                  stream and detection overlay.
                </p>
              </div>
            )}
          </div>

          <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-3 text-xs text-muted">
            <span>Track IDs, keypoints & confidence drawn by the backend</span>
            <span>
              {tel ? `${tel.active_candidates} active · ${tel.fps || 0} fps` : 'no telemetry'}
            </span>
          </div>
        </div>

        {/* Webcam inference */}
        <WebcamPanel />
      </div>
    </div>
  )
}
