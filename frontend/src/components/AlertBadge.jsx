import { AlertTriangle, ShieldCheck } from 'lucide-react'

const TYPE_STYLES = {
  'NORMAL': { color: 'bg-emerald-500', text: 'text-emerald-600 dark:text-emerald-400', label: 'NORMAL' },
  'HEAD TURNING': { color: 'bg-amber-500', text: 'text-amber-600 dark:text-amber-400', label: 'HEAD TURNING' },
  'PEEKING': { color: 'bg-orange-500', text: 'text-orange-600 dark:text-orange-400', label: 'PEEKING' },
  'NOTE PASSING': { color: 'bg-red-500', text: 'text-red-600 dark:text-red-400', label: 'NOTE PASSING' },
}

/**
 * AlertBadge — flashing live state indicator.
 * Shows NORMAL / HEAD TURNING / PEEKING / NOTE PASSING. An optional `demo`
 * flag disables the flash animation so demo state is never mistaken for a
 * real classroom alert.
 */
export default function AlertBadge({ type = 'NORMAL', demo = false, className = '' }) {
  const style = TYPE_STYLES[type] || TYPE_STYLES.NORMAL
  const active = type !== 'NORMAL'
  return (
    <div
      className={`glass flex items-center gap-2 rounded-full px-4 py-2 ${active && !demo ? 'alert-pulse border-red-500/60' : ''} ${className}`}
    >
      {active ? (
        <AlertTriangle size={16} className="text-red-500" />
      ) : (
        <ShieldCheck size={16} className="text-emerald-500" />
      )}
      <span className={`text-sm font-bold tracking-wider ${style.text}`}>
        {style.label}
      </span>
      {demo && (
        <span className="rounded-full bg-surface-3 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-muted">
          demo
        </span>
      )}
    </div>
  )
}
