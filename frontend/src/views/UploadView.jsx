import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { UploadCloud, ScanSearch, AlertTriangle, Loader2, FileVideo } from 'lucide-react'
import DropZone from '../components/DropZone'
import SideBySidePlayer from '../components/SideBySidePlayer'
import StatCard from '../components/StatCard'
import LoadingSkeleton from '../components/LoadingSkeleton'
import { getUploadJob, uploadJobUrl, uploadTestVideo } from '../api/client'

const TYPE_STYLE = {
  'HEAD TURNING': 'border-amber-500/60 bg-amber-500/10 text-amber-600 dark:text-amber-400',
  'PEEKING': 'border-red-500/60 bg-red-500/10 text-red-600 dark:text-red-400',
  'NOTE PASSING': 'border-red-500/60 bg-red-500/10 text-red-600 dark:text-red-400',
}

/**
 * UploadView — Module 1: File Upload & Demo Video Showcase.
 * Drag-drop a recording, hit "Analyze Recording" (POST /upload-test-video),
 * poll the background job, then compare the original with the annotated output
 * using a color-coded alert timeline for quick skipping to cheating moments.
 */
export default function UploadView() {
  const [file, setFile] = useState(null)
  const [jobId, setJobId] = useState(null)
  const [job, setJob] = useState(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [error, setError] = useState(null)
  const rawUrlRef = useRef(null)

  // Local object URL for the raw recording preview (revoked when replaced).
  useEffect(() => {
    if (rawUrlRef.current) URL.revokeObjectURL(rawUrlRef.current)
    rawUrlRef.current = file ? URL.createObjectURL(file) : null
    return () => {
      if (rawUrlRef.current) URL.revokeObjectURL(rawUrlRef.current)
      rawUrlRef.current = null
    }
  }, [file])

  const analyze = async () => {
    if (!file || analyzing) return
    setAnalyzing(true)
    setError(null)
    setJob(null)
    setJobId(null)
    try {
      const res = await uploadTestVideo(file)
      setJobId(res.job_id)
    } catch (err) {
      setError(err.message || String(err))
      setAnalyzing(false)
    }
  }

  // Poll job progress.
  useEffect(() => {
    if (!jobId) return undefined
    const poll = async () => {
      try {
        const j = await getUploadJob(jobId)
        setJob(j)
        if (j.status === 'done' || j.status === 'error') {
          setAnalyzing(false)
        }
      } catch {
        /* transient — keep polling */
      }
    }
    poll()
    const id = setInterval(poll, 1500)
    return () => clearInterval(id)
  }, [jobId])

  const result = job && job.result ? job.result : null
  const progress = job ? job.progress : 0
  const zones = (result && result.zones) || []

  const typeCounts = useMemo(() => {
    const counts = { 'HEAD TURNING': 0, 'PEEKING': 0, 'NOTE PASSING': 0 }
    for (const z of zones) counts[z.type] = (counts[z.type] || 0) + 1
    return counts
  }, [zones])

  const frameAlerts = useMemo(
    () => ((result && result.frames) || []).filter((f) => f.type),
    [result],
  )

  const reset = () => {
    setJobId(null)
    setJob(null)
    setFile(null)
    setError(null)
  }

  return (
    <div className="fade-in space-y-5">
      {/* Uploader + analyze trigger */}
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-[1fr_auto]">
        <DropZone file={file} onFile={setFile} />
        <div className="flex flex-col justify-between gap-3">
          <div className="glass flex-1 rounded-2xl p-4 text-xs text-muted">
            <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold text-ink">
              <ScanSearch size={15} className="text-accent" /> Inference pipeline
            </h3>
            <ul className="list-inside list-disc space-y-1">
              <li>Week 3 — ByteTrack person tracking</li>
              <li>Week 4 — temporal pose windows & heuristics</li>
              <li>Week 5 — spatial-temporal anomaly classification</li>
              <li>Annotated H.264 output + alert timeline</li>
            </ul>
          </div>
          <button
            onClick={analyze}
            disabled={!file || analyzing}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-accent to-accent-2 px-5 py-3 text-sm font-semibold text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {analyzing ? (
              <>
                <Loader2 size={16} className="animate-spin" />
                Analyzing… {job ? `${progress}%` : ''}
              </>
            ) : (
              <>
                <UploadCloud size={16} />
                Analyze Recording
              </>
            )}
          </button>
        </div>
      </div>

      {/* Progress bar while the backend job runs */}
      {analyzing && job && (
        <div className="glass rounded-2xl p-4">
          <div className="mb-2 flex items-center justify-between text-xs">
            <span className="font-medium text-ink">
              {job.status === 'queued' ? 'Queued…' : 'Running inference on recording'}
            </span>
            <span className="font-mono text-muted">{progress}%</span>
          </div>
          <div className="h-2.5 w-full overflow-hidden rounded-full bg-surface-2">
            <div
              className="h-full rounded-full bg-gradient-to-r from-accent to-accent-2 transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {error && (
        <div className="flex items-center gap-2 rounded-2xl border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-danger">
          <AlertTriangle size={16} /> {error} — is the backend running on port 5000?
        </div>
      )}

      {/* Result */}
      {analyzing && !job && !error && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <LoadingSkeleton className="aspect-video" />
          <LoadingSkeleton className="aspect-video" />
        </div>
      )}

      {result && (
        <>
          {/* KPI strip */}
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <StatCard label="Duration" value={`${result.duration_s}s`} tone="accent" />
            <StatCard label="Frames" value={result.total_frames} tone="muted" />
            <StatCard label="Head Turning" value={typeCounts['HEAD TURNING']} tone="warning" />
            <StatCard label="Peeking / Passing" value={typeCounts['PEEKING'] + typeCounts['NOTE PASSING']} tone="danger" />
          </div>

          {/* Side-by-side playback */}
          <SideBySidePlayer
            rawUrl={rawUrlRef.current}
            processedUrl={uploadJobUrl(result.video_url)}
            zones={zones}
            duration={result.duration_s}
          />

          <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
            {/* Detected zones */}
            <div className="glass rounded-2xl p-4">
              <div className="mb-3 flex items-center justify-between">
                <h3 className="text-sm font-semibold">Detected Zones</h3>
                <button
                  onClick={reset}
                  className="text-xs font-medium text-accent hover:underline"
                >
                  New recording
                </button>
              </div>
              {zones.length === 0 ? (
                <p className="text-sm text-muted">No anomalies detected in this recording.</p>
              ) : (
                <ul className="space-y-2">
                  {zones.map((z, i) => (
                    <li
                      key={i}
                      className={`flex items-center gap-3 rounded-xl border px-3 py-2 text-xs ${TYPE_STYLE[z.type] || ''}`}
                    >
                      <AlertTriangle size={14} />
                      <span className="font-semibold">{z.type}</span>
                      <span className="ml-auto font-mono text-muted">
                        {z.start}s – {z.end}s
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            {/* Frame-by-frame anomaly flags */}
            <div className="glass max-h-80 overflow-y-auto rounded-2xl p-4">
              <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold">
                <FileVideo size={15} className="text-accent" />
                Frame-by-Frame Anomaly Flags
                <span className="ml-auto rounded-full bg-surface-2 px-2 py-0.5 font-mono text-[10px] text-muted">
                  {frameAlerts.length} flagged
                </span>
              </h3>
              {frameAlerts.length === 0 ? (
                <p className="text-sm text-muted">No flagged frames.</p>
              ) : (
                <table className="w-full text-left text-xs">
                  <thead className="text-muted">
                    <tr className="border-b border-line">
                      <th className="py-1.5 pr-3 font-medium">Frame</th>
                      <th className="py-1.5 pr-3 font-medium">Time</th>
                      <th className="py-1.5 font-medium">Alert</th>
                    </tr>
                  </thead>
                  <tbody>
                    {frameAlerts.slice(0, 200).map((f) => (
                      <tr key={f.frame} className="border-b border-line/40">
                        <td className="py-1.5 pr-3 font-mono text-ink">{f.frame}</td>
                        <td className="py-1.5 pr-3 font-mono text-muted">{f.t}s</td>
                        <td className={`py-1.5 font-semibold ${f.type === 'HEAD TURNING' ? 'text-amber-600 dark:text-amber-400' : 'text-red-600 dark:text-red-400'}`}>
                          {f.type}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
