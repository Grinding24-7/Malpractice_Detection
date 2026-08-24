import { useEffect } from 'react'
import { X } from 'lucide-react'

/** Modal — accessible dialog with backdrop blur, ESC close and scroll lock. */
export default function Modal({ open, onClose, title, children, wide = false }) {
  useEffect(() => {
    if (!open) return
    const onKey = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    document.body.style.overflow = 'hidden'
    return () => {
      window.removeEventListener('keydown', onKey)
      document.body.style.overflow = ''
    }
  }, [open, onClose])

  if (!open) return null
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 fade-in"
      role="dialog"
      aria-modal="true"
    >
      <div
        className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm"
        onClick={onClose}
      />
      <div
        className={`glass relative flex max-h-[90vh] w-full flex-col overflow-hidden rounded-2xl ${
          wide ? 'max-w-4xl' : 'max-w-lg'
        }`}
      >
        <div className="flex items-center justify-between border-b border-line px-5 py-4">
          <h3 className="text-base font-semibold text-ink">{title}</h3>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-muted transition hover:bg-surface-2 hover:text-ink"
            aria-label="Close"
          >
            <X size={18} />
          </button>
        </div>
        <div className="overflow-y-auto p-5">{children}</div>
      </div>
    </div>
  )
}
