import { useMemo, useState } from 'react'
import { Vault, Play, CalendarDays, Clock } from 'lucide-react'
import Modal from '../components/Modal'
import LoadingSkeleton from '../components/LoadingSkeleton'
import { useAsync } from '../hooks/useAsync'
import { clipUrl, fetchEvidence } from '../api/client'
import { MOCK_EVIDENCE } from '../data/mockData'

const SEVERITY_STYLE = {
  low: 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400',
  medium: 'bg-amber-500/15 text-amber-600 dark:text-amber-400',
  high: 'bg-red-500/15 text-red-600 dark:text-red-400',
}

/** Mock bbox/keypoint log rendered in the quick-view modal. */
function KeypointLog({ id }) {
  const rows = useMemo(() => {
    return Array.from({ length: 12 }, (_, i) => {
      const base = id.charCodeAt(id.length - 1) * 7 + i * 13
      const x = 40 + ((base * 37) % 200)
      const y = 60 + ((base * 53) % 220)
      return {
        frame: i * 5,
        id: (base % 4) + 1,
        box: `[${x}, ${y}, ${x + 180}, ${y + 320}]`,
        conf: (0.62 + ((base % 30) / 100)).toFixed(2),
        keypoints: 17,
      }
    })
  }, [id])
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-xs">
        <thead>
          <tr className="border-b border-line text-muted">
            <th className="py-1.5 pr-3 font-medium">Frame</th>
            <th className="py-1.5 pr-3 font-medium">Track ID</th>
            <th className="py-1.5 pr-3 font-medium">BBox</th>
            <th className="py-1.5 pr-3 font-medium">Conf</th>
            <th className="py-1.5 font-medium">KPts</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.frame} className="border-b border-line/50 font-mono">
              <td className="py-1.5 pr-3 text-ink">{r.frame}</td>
              <td className="py-1.5 pr-3 text-ink">#{r.id}</td>
              <td className="py-1.5 pr-3 text-muted">{r.box}</td>
              <td className="py-1.5 pr-3 text-accent">{r.conf}</td>
              <td className="py-1.5 text-muted">{r.keypoints}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-[11px] text-muted">
        Demo log — the live exporter stamps a JSON sidecar per clip (see{' '}
        <code className="rounded bg-surface-2 px-1">/api/evidence</code>).
      </p>
    </div>
  )
}

/**
 * EvidenceVaultView — Module 3: Evidence Vault.
 * Grid of auto-flagged clips with filter bar (date / classroom / severity /
 * malpractice type) and a quick-view modal with replay + bbox keypoint logs.
 */
export default function EvidenceVaultView({ classroom }) {
  const { data, loading } = useAsync(
    () => fetchEvidence().catch(() => MOCK_EVIDENCE),
    [],
  )

  const evidence = data && data.length ? data : MOCK_EVIDENCE
  const [filters, setFilters] = useState({ date: '', classroom: '', severity: '', type: '' })
  const [selected, setSelected] = useState(null)

  const classrooms = useMemo(
    () => [...new Set(evidence.map((e) => e.classroom).filter(Boolean))],
    [evidence],
  )
  const types = useMemo(
    () => [...new Set(evidence.map((e) => e.type).filter(Boolean))],
    [evidence],
  )

  const filtered = useMemo(() => {
    return evidence.filter((e) => {
      if (filters.classroom && e.classroom !== filters.classroom) return false
      if (filters.severity && e.severity !== filters.severity) return false
      if (filters.type && e.type !== filters.type) return false
      if (filters.date) {
        const d = (e.recorded_at || '').slice(0, 10)
        if (d !== filters.date) return false
      }
      if (classroom && e.classroom !== classroom) return false
      return true
    })
  }, [evidence, filters, classroom])

  const setFilter = (key, value) => setFilters((f) => ({ ...f, [key]: value }))

  const filterClass = 'rounded-xl border border-line bg-surface px-3 py-2 text-sm outline-none transition focus:border-accent'

  return (
    <div className="fade-in space-y-5">
      <div className="glass flex flex-col gap-3 rounded-2xl px-4 py-3 lg:flex-row lg:items-center">
        <div className="flex items-center gap-2">
          <Vault size={17} className="text-accent" />
          <span className="text-sm font-semibold">Evidence Vault</span>
          <span className="rounded-full bg-surface-2 px-2 py-0.5 text-xs text-muted">
            {filtered.length} clip{filtered.length === 1 ? '' : 's'}
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <input
            type="date"
            value={filters.date}
            onChange={(e) => setFilter('date', e.target.value)}
            className={filterClass}
            aria-label="Filter by date"
          />
          <select
            value={filters.classroom}
            onChange={(e) => setFilter('classroom', e.target.value)}
            className={filterClass}
            aria-label="Filter by classroom"
          >
            <option value="">All classrooms</option>
            {classrooms.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
          <select
            value={filters.severity}
            onChange={(e) => setFilter('severity', e.target.value)}
            className={filterClass}
            aria-label="Filter by severity"
          >
            <option value="">All severities</option>
            {['low', 'medium', 'high'].map((s) => (
              <option key={s} value={s}>
                {s[0].toUpperCase() + s.slice(1)}
              </option>
            ))}
          </select>
          <select
            value={filters.type}
            onChange={(e) => setFilter('type', e.target.value)}
            className={filterClass}
            aria-label="Filter by malpractice type"
          >
            <option value="">All types</option>
            {types.map((t) => (
              <option key={t}>{t}</option>
            ))}
          </select>
          {(filters.date || filters.classroom || filters.severity || filters.type) && (
            <button
              onClick={() => setFilters({ date: '', classroom: '', severity: '', type: '' })}
              className="text-xs font-medium text-accent hover:underline"
            >
              Clear filters
            </button>
          )}
        </div>
      </div>

      {loading ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <LoadingSkeleton key={i} className="h-52" />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {filtered.map((e) => (
            <button
              key={e.id}
              onClick={() => setSelected(e)}
              className="glass group overflow-hidden rounded-2xl text-left transition-transform duration-200 hover:-translate-y-0.5"
            >
              <div className="relative aspect-video bg-black">
                <video
                  src={clipUrl(e.name)}
                  muted
                  preload="metadata"
                  className="h-full w-full object-cover opacity-80 transition group-hover:opacity-100"
                />
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="rounded-full bg-black/60 p-3 text-white opacity-0 transition group-hover:opacity-100">
                    <Play size={20} className="ml-0.5" />
                  </span>
                </div>
                <span className={`absolute right-2 top-2 rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide ${SEVERITY_STYLE[e.severity] || SEVERITY_STYLE.low}`}>
                  {e.severity}
                </span>
              </div>
              <div className="space-y-1.5 p-3">
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate font-mono text-xs text-ink">{e.name}</span>
                </div>
                <div className="flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-muted">
                  <span>{e.classroom}</span>
                  <span className="rounded-full bg-accent-2/15 px-2 py-0.5 font-medium text-accent-2">
                    {e.type}
                  </span>
                </div>
                <div className="flex items-center gap-3 text-[11px] text-muted">
                  <span className="flex items-center gap-1">
                    <CalendarDays size={11} />
                    {(e.recorded_at || '').replace('T', ' ').slice(0, 16)}
                  </span>
                  <span className="flex items-center gap-1">
                    <Clock size={11} /> {e.duration_s}s
                  </span>
                </div>
              </div>
            </button>
          ))}
          {filtered.length === 0 && (
            <div className="col-span-full py-16 text-center text-muted">
              <Vault size={36} className="mx-auto mb-3 opacity-40" />
              No evidence clips match the current filters.
            </div>
          )}
        </div>
      )}

      {/* Quick-view modal */}
      <Modal
        open={!!selected}
        onClose={() => setSelected(null)}
        title={selected ? `Evidence — ${selected.name}` : ''}
        wide
      >
        {selected && (
          <div className="space-y-4">
            <video src={clipUrl(selected.name)} controls autoPlay className="aspect-video w-full rounded-xl bg-black" />
            <div className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
              <div className="rounded-xl bg-surface-2 p-3">
                <div className="text-[10px] uppercase tracking-wider text-muted">Classroom</div>
                <div className="mt-1 font-medium">{selected.classroom || '—'}</div>
              </div>
              <div className="rounded-xl bg-surface-2 p-3">
                <div className="text-[10px] uppercase tracking-wider text-muted">Type</div>
                <div className="mt-1 font-medium text-accent-2">{selected.type || '—'}</div>
              </div>
              <div className="rounded-xl bg-surface-2 p-3">
                <div className="text-[10px] uppercase tracking-wider text-muted">Severity</div>
                <div className="mt-1 font-medium capitalize">{selected.severity || '—'}</div>
              </div>
              <div className="rounded-xl bg-surface-2 p-3">
                <div className="text-[10px] uppercase tracking-wider text-muted">Recorded</div>
                <div className="mt-1 font-medium">{(selected.recorded_at || '').replace('T', ' ').slice(0, 16)}</div>
              </div>
            </div>
            <div>
              <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted">
                Saved bounding-box & keypoint log
              </div>
              <KeypointLog id={selected.id} />
            </div>
          </div>
        )}
      </Modal>
    </div>
  )
}
