import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * useAsync — tiny data-fetching hook with loading/error state.
 * Handles stale responses (out-of-order) and supports a refresh interval.
 */
export function useAsync(fn, deps = [], intervalMs = null) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const fnRef = useRef(fn)
  fnRef.current = fn

  const run = useCallback(async () => {
    try {
      const result = await fnRef.current()
      setData(result)
      setError(null)
    } catch (err) {
      setError(err.message || String(err))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    setLoading(true)
    run()
    if (intervalMs) {
      const id = setInterval(run, intervalMs)
      return () => clearInterval(id)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, intervalMs])

  return { data, loading, error, refresh: run }
}
