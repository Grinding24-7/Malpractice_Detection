import { useMemo, useState } from 'react'
import { Activity, Frame, Users, AlertTriangle } from 'lucide-react'
import StatCard from '../components/StatCard'
import LiveVisualizer from '../components/LiveVisualizer'
import DatasetPlaybackSync from '../components/DatasetPlaybackSync'
import TelemetryPanel from '../components/TelemetryPanel'
import { useAsync } from '../hooks/useAsync'
import { fetchTelemetry } from '../api/client'

/**
 * InspectorView — Dataset & Model Monitoring Dashboard.
 *
 * Combines the three monitoring modules:
 *   1. LiveVisualizer       — real-time draw engine over the CCTV stream
 *   2. TelemetryPanel       — velocity sparklines + per-student breakdown
 *   3. DatasetPlaybackSync  — frame-locked raw vs model-monitoring playback
 * Selection of a student track is shared between the visualizer and telemetry.
 */
export default function InspectorView() {
  const [selected, setSelected] = useState(null)
  const [liveAnnotations, setLiveAnnotations] = useState(null)

  const { data: tel } = useAsync(
    () => fetchTelemetry().catch(() => null),
    [],
    600,
  )

  const activeCount = (liveAnnotations && liveAnnotations.students) || []
  const alertActive = activeCount.some(
    (s) => s.status !== 'NOMINAL',
  )

  const stats = useMemo(
    () => ({
      fps: tel ? tel.fps : '—',
      frame: liveAnnotations ? liveAnnotations.frame_id : '—',
      students: activeCount.length,
      status: alertActive ? 'ALERT' : 'NOMINAL',
    }),
    [tel, liveAnnotations, alertActive],
  )

  return (
    <div className="fade-in space-y-5">
      {/* KPI strip */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard
          label="Model FPS"
          value={stats.fps}
          icon={Activity}
          tone="accent"
        />
        <StatCard
          label="Frame ID"
          value={stats.frame}
          icon={Frame}
          tone="accent2"
        />
        <StatCard
          label="Tracked Students"
          value={stats.students}
          icon={Users}
          tone="muted"
        />
        <StatCard
          label="Model Status"
          value={stats.status}
          icon={AlertTriangle}
          tone={alertActive ? 'danger' : 'accent'}
        />
      </div>

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-3">
        <div className="xl:col-span-2">
          <LiveVisualizer
            selected={selected}
            onSelect={setSelected}
            onAnnotations={setLiveAnnotations}
          />
        </div>
        <TelemetryPanel
          annotations={liveAnnotations}
          telemetry={tel}
          selected={selected}
          onSelect={setSelected}
        />
      </div>

      <DatasetPlaybackSync />
    </div>
  )
}
