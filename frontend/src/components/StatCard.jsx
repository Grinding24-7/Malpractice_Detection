import { Activity, Users, AlertTriangle, Video } from 'lucide-react'

/** StatCard — small glass KPI used across views. */
export default function StatCard({ label, value, icon, tone = 'accent' }) {
  const toneMap = {
    accent: 'text-accent',
    accent2: 'text-accent-2',
    danger: 'text-danger',
    warning: 'text-warning',
    muted: 'text-muted',
  }
  const Icon = icon || Activity
  return (
    <div className="glass rounded-2xl p-4">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium uppercase tracking-wide text-muted">
          {label}
        </span>
        <Icon size={18} className={toneMap[tone]} />
      </div>
      <div className="mt-2 font-mono text-2xl font-semibold text-ink">{value}</div>
    </div>
  )
}

export { Activity, Users, AlertTriangle, Video }
