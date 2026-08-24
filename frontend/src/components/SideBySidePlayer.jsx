import { useCallback, useEffect, useRef, useState } from 'react'
import { Play, Pause, RotateCcw, MonitorPlay, ScanLine } from 'lucide-react'
import AlertTimeline from './AlertTimeline'

/**
 * SideBySidePlayer — original raw recording (left) next to the processed,
 * annotated output (right). The two videos are kept in sync: play/pause/seek
 * on either drives the other, and the timeline's color-coded zones jump both
 * to the start of a detected cheating moment.
 */
export default function SideBySidePlayer({ rawUrl, processedUrl, zones = [], duration = 0 }) {
  const rawRef = useRef(null)
  const procRef = useRef(null)
  const [playing, setPlaying] = useState(false)
  const [time, setTime] = useState(0)
  const [procDuration, setProcDuration] = useState(0)
  const syncingRef = useRef(false)

  const seekBoth = useCallback((t) => {
    syncingRef.current = true
    for (const v of [rawRef.current, procRef.current]) {
      if (v) {
        try {
          v.currentTime = t
        } catch {
          /* video may not be seekable yet */
        }
      }
    }
    syncingRef.current = false
    setTime(t)
  }, [])

  const togglePlay = () => {
    const master = procRef.current || rawRef.current
    if (!master) return
    if (master.paused) {
      for (const v of [rawRef.current, procRef.current]) if (v) v.play()
      setPlaying(true)
    } else {
      for (const v of [rawRef.current, procRef.current]) if (v) v.pause()
      setPlaying(false)
    }
  }

  // Keep the two videos time-synced during playback.
  const onTimeUpdate = (source) => (e) => {
    if (syncingRef.current) return
    const v = e.currentTarget
    if (source === 'proc') {
      setTime(v.currentTime)
      const other = rawRef.current
      if (other && Math.abs(other.currentTime - v.currentTime) > 0.25) {
        syncingRef.current = true
        other.currentTime = v.currentTime
        syncingRef.current = false
      }
    } else {
      setTime(v.currentTime)
    }
  }

  useEffect(() => {
    if (procRef.current && procDuration === 0) {
      const onMeta = () => setProcDuration(procRef.current.duration || 0)
      procRef.current.addEventListener('loadedmetadata', onMeta)
      return () => procRef.current && procRef.current.removeEventListener('loadedmetadata', onMeta)
    }
  }, [procDuration])

  const endDuration = duration || procDuration

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {/* Raw original */}
        <div className="overflow-hidden rounded-2xl border border-line bg-black">
          <div className="flex items-center gap-2 border-b border-white/10 bg-surface px-3 py-2">
            <MonitorPlay size={14} className="text-muted" />
            <span className="text-xs font-semibold">Original Recording</span>
          </div>
          <video
            ref={rawRef}
            src={rawUrl}
            controls={false}
            onTimeUpdate={onTimeUpdate('raw')}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            className="aspect-video w-full"
          />
        </div>
        {/* Processed annotated output */}
        <div className="relative overflow-hidden rounded-2xl border border-line bg-black">
          <div className="flex items-center gap-2 border-b border-white/10 bg-surface px-3 py-2">
            <ScanLine size={14} className="text-accent" />
            <span className="text-xs font-semibold">Processed Output</span>
            <span className="ml-auto rounded-full bg-accent/15 px-2 py-0.5 text-[10px] font-bold text-accent">
              AI OVERLAY
            </span>
          </div>
          <video
            ref={procRef}
            src={processedUrl}
            controls={false}
            onTimeUpdate={onTimeUpdate('proc')}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            className="aspect-video w-full"
          />
        </div>
      </div>

      {/* Transport + timeline */}
      <div className="glass rounded-2xl p-4">
        <div className="mb-3 flex items-center gap-2">
          <button
            onClick={togglePlay}
            className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-accent to-accent-2 px-4 py-2 text-sm font-semibold text-white transition hover:opacity-90"
          >
            {playing ? <Pause size={15} /> : <Play size={15} />}
            {playing ? 'Pause' : 'Play'}
          </button>
          <button
            onClick={() => seekBoth(0)}
            className="rounded-xl border border-line p-2 text-muted transition hover:text-ink"
            aria-label="Restart"
          >
            <RotateCcw size={15} />
          </button>
          <span className="ml-auto text-xs text-muted">
            {zones.length} detected {zones.length === 1 ? 'zone' : 'zones'}
          </span>
        </div>
        <AlertTimeline
          duration={endDuration}
          zones={zones}
          currentTime={time}
          onSeek={seekBoth}
        />
      </div>
    </div>
  )
}
