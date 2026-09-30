import { useState } from 'react'
import { ShieldCheck, LayoutDashboard, ChevronLeft, ChevronRight } from 'lucide-react'

/**
 * Sidebar — responsive glass navigation. Collapses to icon-only on small
 * screens (hamburger opens a slide-over) and to compact rail on hover.
 */
export default function Sidebar({ nav, active, onSelect }) {
  const [mobileOpen, setMobileOpen] = useState(false)
  const [collapsed, setCollapsed] = useState(false)

  const items = nav.map((n) => {
    const Icon = n.icon
    const isActive = n.id === active
    return (
      <button
        key={n.id}
        onClick={() => {
          onSelect(n.id)
          setMobileOpen(false)
        }}
        className={`group relative flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-all duration-200 ${
          isActive
            ? 'bg-accent/15 text-accent'
            : 'text-muted hover:bg-surface-2 hover:text-ink'
        }`}
        title={n.label}
      >
        <Icon size={18} className="shrink-0" />
        {!collapsed && <span className="truncate">{n.label}</span>}
        {isActive && (
          <span className="absolute right-2 h-5 w-1 rounded-full bg-accent" />
        )}
      </button>
    )
  })

  return (
    <>
      {/* Mobile slide-over */}
      <aside
        className={`fixed inset-y-0 left-0 z-50 w-64 transform transition-transform duration-300 lg:hidden ${
          mobileOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="glass flex h-full flex-col p-4">{renderBrand()}{items}</div>
        <button
          className="absolute -right-10 top-4 rounded-full bg-surface p-2 text-muted"
          onClick={() => setMobileOpen(false)}
          aria-label="Close navigation"
        >
          <ChevronLeft size={16} />
        </button>
      </aside>
      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-slate-900/50 lg:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Desktop rail */}
      <aside
        className={`hidden shrink-0 flex-col border-r border-line transition-all duration-300 lg:flex ${
          collapsed ? 'w-16' : 'w-60'
        }`}
      >
        <div className="flex h-full flex-col p-4">
          {renderBrand()}
          <nav className="flex flex-1 flex-col gap-1">{items}</nav>
          <button
            onClick={() => setCollapsed((c) => !c)}
            className="mt-2 flex items-center justify-center gap-2 rounded-xl px-3 py-2 text-muted transition hover:bg-surface-2 hover:text-ink"
            title={collapsed ? 'Expand' : 'Collapse'}
          >
            {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
            {!collapsed && <span className="text-xs">Collapse</span>}
          </button>
        </div>
      </aside>

      {/* Mobile topbar trigger */}
      <button
        onClick={() => setMobileOpen(true)}
        className="fixed left-4 top-4 z-30 rounded-xl bg-surface p-2 shadow-lg lg:hidden"
        aria-label="Open navigation"
      >
        <LayoutDashboard size={18} />
      </button>
    </>
  )

  function renderBrand() {
    return (
      <div className="mb-6 flex items-center gap-2.5 px-1">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-accent to-accent-2 text-white shadow-lg">
          <ShieldCheck size={18} />
        </div>
        <div className={collapsed ? 'hidden' : ''}>
          <div className="text-sm font-bold leading-tight">Proctoring AI</div>
          <div className="text-[10px] uppercase tracking-widest text-muted">
            Malpractice Detection
          </div>
        </div>
      </div>
    )
  }
}
