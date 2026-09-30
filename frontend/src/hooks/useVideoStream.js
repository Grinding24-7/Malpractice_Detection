/**
 * useVideoStream.js — Custom React hook for WebSocket video stream consumption.
 *
 * Features:
 *   - Auto-reconnect with exponential backoff on disconnect
 *   - Parses incoming JSON telemetry + base64 JPEG frames
 *   - Exposes frame data, connection state, and error status
 *   - Cleanup on unmount (closes WebSocket gracefully)
 *
 * Usage:
 *   const { connected, frame, students, frameId, error } = useVideoStream(wsUrl)
 */

import { useCallback, useEffect, useRef, useState } from 'react'

const RECONNECT_BASE_MS = 1000
const RECONNECT_MAX_MS = 15000
const MAX_FRAME_QUEUE = 3  // drop old frames to prevent stutter

export function useVideoStream(wsUrl) {
  const [connected, setConnected] = useState(false)
  const [frame, setFrame] = useState(null)       // base64 JPEG string
  const [students, setStudents] = useState([])    // telemetry array
  const [frameId, setFrameId] = useState(0)
  const [error, setError] = useState(null)

  const wsRef = useRef(null)
  const reconnectMs = useRef(RECONNECT_BASE_MS)
  const reconnectTimer = useRef(null)
  const unmounted = useRef(false)
  const frameQueue = useRef([])

  const cleanup = useCallback(() => {
    if (reconnectTimer.current) {
      clearTimeout(reconnectTimer.current)
      reconnectTimer.current = null
    }
    if (wsRef.current) {
      try { wsRef.current.close() } catch { /* ignore */ }
      wsRef.current = null
    }
  }, [])

  const connect = useCallback(() => {
    if (unmounted.current || !wsUrl) return
    cleanup()

    try {
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onopen = () => {
        setConnected(true)
        setError(null)
        reconnectMs.current = RECONNECT_BASE_MS
        console.log('[useVideoStream] connected')
      }

      ws.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data)

          // Backpressure: skip frame if queue is backing up
          if (frameQueue.current.length >= MAX_FRAME_QUEUE) {
            frameQueue.current.shift()
          }
          frameQueue.current.push(data)

          // Only render the latest frame
          const latest = frameQueue.current[frameQueue.current.length - 1]
          if (latest.frame_jpeg) {
            setFrame(latest.frame_jpeg)
          }
          if (latest.students) {
            setStudents(latest.students)
          }
          if (latest.frame_id) {
            setFrameId(latest.frame_id)
          }
        } catch (err) {
          console.warn('[useVideoStream] parse error:', err)
        }
      }

      ws.onerror = (evt) => {
        console.error('[useVideoStream] error:', evt)
        setError('WebSocket error')
      }

      ws.onclose = () => {
        setConnected(false)
        wsRef.current = null
        if (!unmounted.current) {
          const delay = reconnectMs.current
          reconnectMs.current = Math.min(reconnectMs.current * 1.5, RECONNECT_MAX_MS)
          console.log(`[useVideoStream] reconnecting in ${delay}ms...`)
          reconnectTimer.current = setTimeout(connect, delay)
        }
      }
    } catch (err) {
      setError(err.message)
      // Retry
      if (!unmounted.current) {
        reconnectTimer.current = setTimeout(connect, reconnectMs.current)
        reconnectMs.current = Math.min(reconnectMs.current * 1.5, RECONNECT_MAX_MS)
      }
    }
  }, [wsUrl, cleanup])

  useEffect(() => {
    unmounted.current = false
    connect()
    return () => {
      unmounted.current = true
      cleanup()
    }
  }, [connect, cleanup])

  return { connected, frame, students, frameId, error }
}
