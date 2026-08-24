/**
 * LoadingSkeleton — shimmer placeholders used while API data is in flight.
 * `variant`: 'card' | 'video' | 'line'.
 */
export default function LoadingSkeleton({ variant = 'card', className = '' }) {
  if (variant === 'line') {
    return <div className={`skeleton h-4 w-full rounded-full ${className}`} />
  }
  if (variant === 'video') {
    return (
      <div
        className={`skeleton flex aspect-video w-full items-center justify-center rounded-2xl ${className}`}
      >
        <span className="text-sm text-muted">Loading stream…</span>
      </div>
    )
  }
  return (
    <div className={`skeleton h-40 rounded-2xl ${className}`} />
  )
}
