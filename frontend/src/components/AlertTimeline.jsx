import { useCallback, useRef } from 'react'

const ZONE_COLORS = {
  'HEAD TURNING': '#f59e0b', // amber/yellow — moderate
  'PEEKING': '#ef4444',      // red — high
  'NOTE PASSING': '#ef4444', // red — high
}

/**
 * AlertTimeline — color-coded scrubber for the annotated playback.
 * Red zones = peeking / note-passing, yellow zones = head-turning. Clicking a
 * zone (or anywhere on the bar) seeks both videos. The playhead tracks the
 * processed video's currentTime.
 */
export default function AlertTimeline({ duration, zones = [], currentTime, onSeek }) {
  const barRef = useRef(null)

  const seekFromEvent = useCallback(
    (e) => {
      if (!barRef.current) return
      const rect = barRef.current.getBoundingClientRect()
      const ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width))
      onSeek(ratio * (duration || 0))
    },
    [duration, onSeek],
  )

  const pct = duration > 0 ? Math.min(100, (currentTime / duration) * 100) : 0
  const activeZone = zones.find((z) => currentTime >= z.start && currentTime <= z.end)

  return (
    <div>
      {/* Zone legend */}
      <div className="mb-1.5 flex flex-wrap items-center gap-3 text-[11px] text-muted">
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm bg-red-500" /> Peeking / Note passing
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm bg-amber-500" /> Head turning
        </span>
        <span className="ml-auto font-mono">
          {currentTime.toFixed(1)}s / {duration.toFixed(1)}s
        </span>
      </div>

      {/* Scrubber */}
      <div
        ref={barRef}
        onClick={seekFromEvent}
        className="relative h-8 w-full cursor-pointer overflow-hidden rounded-xl border border-line bg-surface-2"
        role="slider"
        aria-label="Timeline scrubber"
        aria-valuemin={0}
        aria-valuemax={Math.round(duration || 0)}
        aria-valuenow={Math.round(currentTime)}
        tabIndex={0}
        onKeyDown={(e) => {
          const step = 2
          if (e.key === 'ArrowRight') onSeek(Math.min(duration, currentTime + step))
          if (e.key === 'ArrowLeft') onSeek(Math.max(0, currentTime - step))
        }}
      >
        {zones.map((z, i) => {
          const left = (z.start / duration) * 100
          const width = ((z.end - z.start) / duration) * 100
          return (
            <div
              key={`${z.start}-${i}`}
              title={`${z.type}: ${z.start}s – ${z.end}s (${z.severity})`}
              onClick={(e) => {
                e.stopPropagation()
                onSeek(z.start)
              }}
              className="absolute top-0 h-full opacity-70 transition hover:opacity-100"
              style={{
                left: `${left}%`,
                width: `${width}%`,
                background: ZONE_COLORS[z.type] || '#ef4444',
              }}
            />
          )
        })}
        {/* Playhead */}
        <div
          className="pointer-events-none absolute top-0 h-full w-0.5 bg-ink"
          style={{ left: `${pct}%` }}
        />
      </div>

      {activeZone && (
        <div className="mt-1.5 inline-flex items-center gap-2 rounded-full bg-surface-2 px-2.5 py-1 text-[11px] font-medium text-ink">
          <span
            className="h-2 w-2 rounded-full"
            style={{ background: ZONE_COLORS[activeZone.type] }}
          />
          Inside {activeZone.type} zone
          <span className="font-mono text-muted">
            {activeZone.start}s – {activeZone.end}s
          </span>
        </div>
      )}
    </div>
  )
}
