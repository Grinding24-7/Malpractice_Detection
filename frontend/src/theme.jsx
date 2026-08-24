import React, { createContext, useContext, useEffect, useMemo, useState } from 'react'

const STORAGE_KEY = 'md-theme'
const ThemeContext = createContext(null)

/**
 * ThemeProvider — owns the Dark/Light state.
 *
 * - Defaults to the OS `prefers-color-scheme` when nothing is stored.
 * - Persists the user's choice in `localStorage['md-theme']`.
 * - Toggles the `.dark` class on <html>; every color in the app is driven by
 *   the CSS custom properties declared in src/index.css, so toggling the class
 *   re-themes Tailwind utilities AND Chart.js charts in one shot.
 */
export function ThemeProvider({ children }) {
  const [dark, setDark] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY)
      if (saved === 'dark') return true
      if (saved === 'light') return false
      return window.matchMedia('(prefers-color-scheme: dark)').matches
    } catch {
      return true
    }
  })

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    try {
      localStorage.setItem(STORAGE_KEY, dark ? 'dark' : 'light')
    } catch {
      /* storage unavailable (private mode) — theme still applies for this session */
    }
  }, [dark])

  const value = useMemo(
    () => ({
      dark,
      toggle: () => setDark((d) => !d),
    }),
    [dark],
  )

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export function useTheme() {
  const ctx = useContext(ThemeContext)
  if (!ctx) throw new Error('useTheme must be used within <ThemeProvider>')
  return ctx
}

/**
 * Chart palettes that follow the current theme. The values mirror the CSS
 * variables in index.css so charts stay consistent with the rest of the UI.
 */
export function chartPalette() {
  const css = (name, fallback) => {
    if (typeof document === 'undefined') return fallback
    return getComputedStyle(document.documentElement)
      .getPropertyValue(name)
      .trim() || fallback
  }
  return {
    text: css('--md-text', '#0f172a'),
    muted: css('--md-muted', '#64748b'),
    grid: css('--md-border', '#e2e8f0'),
    accent: css('--md-accent', '#10b981'),
    accent2: css('--md-accent-2', '#6366f1'),
    danger: css('--md-danger', '#ef4444'),
    warning: css('--md-warning', '#f59e0b'),
    surface: css('--md-surface', '#ffffff'),
  }
}
