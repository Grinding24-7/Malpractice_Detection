import { createContext, useContext, useState } from 'react'
import { AlertTriangle, CheckCircle2 } from 'lucide-react'

/**
 * ToastProvider — minimal toast notifications for form feedback
 * (e.g. "Classroom registered") with auto-dismiss + slide-in animation.
 */
const ToastContext = createContext(null)

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([])

  const push = (message, kind = 'success') => {
    const id = Date.now() + Math.random()
    setToasts((t) => [...t, { id, message, kind }])
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 3500)
  }

  return (
    <ToastContext.Provider value={{ push }}>
      {children}
      <div className="fixed bottom-5 right-5 z-[60] flex flex-col gap-2">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={`glass fade-in flex items-center gap-3 rounded-xl px-4 py-3 text-sm shadow-lg ${
              t.kind === 'error'
                ? 'border-l-4 !border-l-red-500'
                : 'border-l-4 !border-l-emerald-500'
            }`}
          >
            {t.kind === 'error' ? (
              <AlertTriangle size={18} className="text-red-500" />
            ) : (
              <CheckCircle2 size={18} className="text-emerald-500" />
            )}
            <span className="text-ink">{t.message}</span>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}

export function useToast() {
  const ctx = useContext(ToastContext)
  if (!ctx) throw new Error('useToast must be used within <ToastProvider>')
  return ctx
}
