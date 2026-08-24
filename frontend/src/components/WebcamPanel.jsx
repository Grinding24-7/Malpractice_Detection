import { useCallback, useEffect, useRef, useState } from 'react'
import Webcam from 'react-webcam'
import { Camera, CameraOff } from 'lucide-react'
import OverlayCanvas from './OverlayCanvas'
import AlertBadge from './AlertBadge'
import { postStreamFrame } from '../api/client'
import { demoPersons, demoAlert } from '../data/mockData'

const CAPTURE_MS = 700

/** Derive the dominant alert type from per-person features. */
export function deriveAlert(persons) {
  if (!persons || persons.length === 0) return { active: false, type: 'NORMAL' }
  const any = persons.some((p) => p.status === 'ANOMALY')
  if (!any) return { active: false, type: 'NORMAL' }
  const turning = persons.some((p) => {
    const r = p.head ? p.head.ear_ratio : 1
    return r < 0.7 || r > 1.4
  })
  const peeking = persons.some((p) => (p.head ? p.head.norm_vertical_drop : 0) > 0.9)
  if (peeking) return { active: true, type: 'PEEKING' }
  if (turning) return { active: true, type: 'HEAD TURNING' }
  return { active: true, type: 'NOTE PASSING' }
}

/**
 * WebcamPanel — interactive webcam toggle for testing real-time predictions.
 *
 * When switched ON the browser webcam is opened and a low-rate snapshot is
 * POSTed to the backend `/stream` endpoint. The response (tracked boxes +
 * COCO-17 keypoints + track ids + confidence) is drawn by OverlayCanvas and
 * drives the AlertBadge. If the backend is unreachable the panel falls back
 * to a clearly-labelled DEMO overlay so the UI stays testable.
 */
export default function WebcamPanel({ className = '' }) {
  const webcamRef = useRef(null)
  const [enabled, setEnabled] = useState(false)
  const [overlay, setOverlay] = useState(true)
  const [persons, setPersons] = useState([])
  const [frame, setFrame] = useState(0)
  const [backendUp, setBackendUp] = useState(true)
  const busyRef = useRef(false)
  const frameRef = useRef(0)

  const snapshot = useCallback(async () => {
    const img = webcamRef.current && webcamRef.current.getScreenshot()
    if (!img || busyRef.current) return
    busyRef.current = true
    frameRef.current += 1
    const f = frameRef.current
    setFrame(f)
    try {
      const res = await postStreamFrame(img)
      setPersons(res.persons || [])
      setBackendUp(true)
    } catch {
      // Backend offline — fall back to the demo overlay (tagged "demo").
      setBackendUp(false)
      setPersons(demoPersons(f))
    } finally {
      busyRef.current = false
    }
  }, [])

  useEffect(() => {
    if (!enabled) {
      setPersons([])
      return undefined
    }
    const id = setInterval(snapshot, CAPTURE_MS)
    return () => clearInterval(id)
  }, [enabled, snapshot])

  const alert = deriveAlert(persons)
  const streamActive = backendUp && enabled

  return (
    <div className={`glass flex flex-col overflow-hidden rounded-2xl ${className}`}>
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-4 py-3">
        <div className="flex items-center gap-2">
          <Camera size={17} className="text-accent" />
          <span className="text-sm font-semibold">Webcam Inference</span>
        </div>
        <div className="flex items-center gap-2">
          <label className="flex cursor-pointer items-center gap-2 text-xs font-medium text-muted">
            <input
              type="checkbox"
              checked={overlay}
              onChange={(e) => setOverlay(e.target.checked)}
              className="accent-emerald-500"
            />
            Keypoint overlay
          </label>
          {/* Toggle ON/OFF switch */}
          <button
            onClick={() => setEnabled((e) => !e)}
            className={`relative inline-flex h-7 w-12 items-center rounded-full transition-colors ${
              enabled ? 'bg-accent' : 'bg-surface-3'
            }`}
            aria-pressed={enabled}
            role="switch"
            aria-label="Toggle webcam"
          >
            <span
              className={`inline-block h-5 w-5 transform rounded-full bg-white shadow transition-transform ${
                enabled ? 'translate-x-6' : 'translate-x-1'
              }`}
            />
          </button>
        </div>
      </div>

      <div className="relative aspect-video w-full bg-black">
        {enabled ? (
          <>
            <Webcam
              ref={webcamRef}
              audio={false}
              screenshotFormat="image/jpeg"
              screenshotQuality={0.72}
              videoConstraints={{ width: 1280, height: 720, facingMode: 'user' }}
              className="h-full w-full object-contain"
              mirrored
            />
            <OverlayCanvas
              persons={persons}
              width={1280}
              height={720}
              enabled={overlay}
            />
            {!backendUp && (
              <div className="absolute left-3 top-3 rounded-full bg-black/70 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider text-amber-300">
                Backend offline — demo overlay
              </div>
            )}
          </>
        ) : (
          <div className="flex h-full w-full flex-col items-center justify-center gap-2 text-muted">
            <CameraOff size={36} />
            <p className="text-sm">Webcam is off</p>
            <p className="max-w-xs text-center text-xs opacity-80">
              Toggle the switch above to start streaming frames to{' '}
              <code className="rounded bg-surface-2 px-1">POST /stream</code>{' '}
              for live model predictions.
            </p>
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3">
        <AlertBadge type={alert.type} demo={!streamActive} />
        {enabled && (
          <div className="flex items-center gap-3 text-xs text-muted">
            <span className="flex items-center gap-1">
              <span
                className={`h-2 w-2 rounded-full ${streamActive ? 'bg-emerald-500' : 'bg-amber-400'}`}
              />
              {streamActive ? 'Live' : 'Demo'} · {persons.length} tracked
            </span>
            <span className="hidden sm:inline">frame #{frame}</span>
          </div>
        )}
      </div>
    </div>
  )
}
