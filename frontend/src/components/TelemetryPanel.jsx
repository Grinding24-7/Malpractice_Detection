import { useEffect, useMemo, useRef } from 'react'
import { Line } from 'react-chartjs-2'
import { Activity, Hand, ScanEye, AlertTriangle } from 'lucide-react'
import { chartPalette } from '../theme'
import { handVelocity, STATUS_META } from '../data/mockData'

const HISTORY = 120

/**
 * TelemetryPanel — Module: Real-Time Feature Telemetry & Velocity Charts.
 *
 * Left: live sparklines of head displacement velocity (Δp) and hand spatial
 * movement against frame time. Right: per-student breakdown for the selected
 * track — head angle deviation vs threshold, temporal window buffer, and the
 * model classification output with confidence.
 */
export default function TelemetryPanel({ annotations, telemetry, selected, onSelect }) {
  const palette = chartPalette()
  const histRef = useRef([])
  const prevRef = useRef([])

  // Append one sample per new annotation frame (head = max velocity_spike,
  // hand = wrist displacement between consecutive frames).
  const hist = useMemo(() => {
    const students = (annotations && annotations.students) || []
    const frameId = annotations && annotations.frame_id
    const h = histRef.current
    const last = h.length ? h[h.length - 1] : null
    if (last && last.frame === frameId) return h
    const head = students.length
      ? Math.max(...students.map((s) => s.velocity_spike || 0))
      : 0
    const hand = prevRef.current.length ? handVelocity(prevRef.current, students) : 0
    h.push({
      frame: frameId,
      t: (annotations && annotations.timestamp) || 0,
      head,
      hand,
    })
    if (h.length > HISTORY) h.shift()
    if (students.length) prevRef.current = students
    return h
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [annotations])

  // Keep the chart in sync after a manual cap shift.
  useEffect(() => {
    // no-op — `hist` memo above owns the buffer.
  }, [hist])

  const selectedStudent = useMemo(() => {
    const students = (annotations && annotations.students) || []
    if (!students.length) return null
    return (
      students.find((s) => s.track_id === selected) ||
      students[0]
    )
  }, [annotations, selected])

  const headAngle = selectedStudent
    ? Math.min(90, Math.abs((selectedStudent.ear_ratio ?? 1) - 1) * 90)
    : 0
  const angleHot = headAngle > 30
  const windows = telemetry ? telemetry.temporal_windows_ready : null

  const spark = (data, color, label) => ({
    labels: hist.map((s) => String((s.t || 0).toFixed(1))),
    datasets: [
      {
        label,
        data,
        borderColor: color,
        backgroundColor: color + '22',
        fill: true,
        tension: 0.35,
        pointRadius: 0,
        borderWidth: 2,
      },
    ],
  })

  const sparkOptions = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      plugins: { legend: { display: false }, tooltip: { enabled: true } },
      scales: {
        x: { display: false },
        y: { display: false },
      },
    }),
    [],
  )

  return (
    <div className="glass flex flex-col overflow-hidden rounded-2xl">
      <div className="flex items-center justify-between border-b border-line px-4 py-3">
        <span className="flex items-center gap-2 text-sm font-semibold">
          <ScanEye size={17} className="text-accent-2" />
          Telemetry & Velocity
        </span>
        <span className="font-mono text-xs text-muted">last {hist.length} frames</span>
      </div>

      <div className="grid grid-cols-1 gap-3 p-4">
        {/* Sparklines */}
        <div className="rounded-xl border border-line p-3">
          <div className="mb-1 flex items-center justify-between text-xs">
            <span className="flex items-center gap-1.5 font-medium text-muted">
              <Activity size={13} className="text-accent" />
              Head displacement velocity (Δp)
            </span>
            <span className="font-mono text-accent">
              {hist.length ? hist[hist.length - 1].head.toFixed(2) : '—'}
            </span>
          </div>
          <div className="h-14">
            <Line data={spark(hist.map((s) => s.head), palette.accent, 'Head Δp')} options={sparkOptions} />
          </div>
        </div>

        <div className="rounded-xl border border-line p-3">
          <div className="mb-1 flex items-center justify-between text-xs">
            <span className="flex items-center gap-1.5 font-medium text-muted">
              <Hand size={13} className="text-accent-2" />
              Hand spatial movement
            </span>
            <span className="font-mono text-accent-2">
              {hist.length ? hist[hist.length - 1].hand.toFixed(2) : '—'}
            </span>
          </div>
          <div className="h-14">
            <Line data={spark(hist.map((s) => s.hand), palette.accent2, 'Hand motion')} options={sparkOptions} />
          </div>
        </div>

        {/* Per-student breakdown */}
        <div className="rounded-xl border border-line p-3">
          <div className="mb-2 flex items-center justify-between text-xs font-medium text-muted">
            <span>Per-Student Breakdown</span>
            <span>click a row to inspect</span>
          </div>
          {!selectedStudent ? (
            <p className="py-6 text-center text-xs text-muted">No tracked students yet.</p>
          ) : (
            <>
              <ul className="mb-3 space-y-1.5">
                {((annotations && annotations.students) || []).map((s) => {
                  const meta = STATUS_META[s.status] || STATUS_META.NOMINAL
                  const active = s.track_id === selectedStudent.track_id
                  return (
                    <li key={s.track_id}>
                      <button
                        onClick={() => onSelect(s.track_id)}
                        className={`flex w-full items-center justify-between rounded-lg px-2 py-1.5 text-xs transition ${
                          active ? 'bg-accent/10 text-ink' : 'hover:bg-surface-2 text-muted'
                        }`}
                      >
                        <span className="flex items-center gap-2">
                          <span className="h-2 w-2 rounded-full" style={{ background: meta.color }} />
                          <span className="font-mono font-semibold">Student #{String(s.track_id).padStart(2, '0')}</span>
                          <span className="text-[10px]">{meta.label}</span>
                        </span>
                        <span className="font-mono">{Math.round((s.confidence || 0) * 100)}%</span>
                      </button>
                    </li>
                  )
                })}
              </ul>

              <div className="space-y-2 border-t border-line pt-3 text-xs">
                <div className="flex items-center justify-between">
                  <span className="text-muted">Active Track ID</span>
                  <span className="font-mono font-semibold text-ink">
                    Student #{String(selectedStudent.track_id).padStart(2, '0')}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted">Head Angle Deviation</span>
                  <span className={`font-mono font-semibold ${angleHot ? 'text-danger' : 'text-accent'}`}>
                    {headAngle.toFixed(0)}° <span className="font-normal text-muted">(Thr: 30°)</span>
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted">Temporal Window Buffer</span>
                  <span className="font-mono font-semibold text-ink">
                    {windows ?? '—'} / 30 frames
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted">Model Classification</span>
                  <span
                    className={`font-mono font-semibold ${
                      selectedStudent.status === 'NOMINAL' ? 'text-accent' : 'text-danger'
                    }`}
                  >
                    {selectedStudent.status}{' '}
                    <span className="font-normal text-muted">
                      (Confidence: {(selectedStudent.confidence || 0).toFixed(2)})
                    </span>
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted">Velocity Spike</span>
                  <span className="font-mono font-semibold text-ink">
                    {(selectedStudent.velocity_spike || 0).toFixed(2)}
                  </span>
                </div>
              </div>

              {angleHot && (
                <div className="mt-3 flex items-center gap-2 rounded-lg bg-danger/10 px-3 py-2 text-[11px] font-medium text-danger">
                  <AlertTriangle size={13} />
                  Head angle exceeds the 30° suspicion threshold.
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
