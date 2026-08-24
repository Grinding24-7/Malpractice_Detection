import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Clapperboard,
  Play,
  Pause,
  RotateCcw,
  SkipBack,
  SkipForward,
  ScanLine,
} from 'lucide-react'
import { datasetUrl } from '../api/client'
import { DATASET_VIDEOS, DEMO_FPS, demoAnnotations } from '../data/mockData'
import { drawStudents } from './InferenceOverlay'

/**
 * DatasetPlaybackSync — Module: Time-Synchronized Dataset Inspector.
 *
 * Left: raw source CCTV dataset video. Right: the model monitoring view —
 * bounding boxes, ByteTrack tracking vectors and normalized keypoints drawn
 * client-side over the SAME video element, so both windows are frame-locked by
 * construction. Controls (play / pause / scrub / step ±1 frame) drive one
 * playhead. Demo keypoints are deterministic in the playhead frame so
 * scrubbing shows stable, reproducible poses.
 */
export default function DatasetPlaybackSync() {
  const [videoName, setVideoName] = useState(DATASET_VIDEOS[0].id)
  const [playing, setPlaying] = useState(false)
  const [time, setTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [demo, setDemo] = useState(false)
  const videoRef = useRef(null)
  const rightRef = useRef(null)
  const rafRef = useRef(null)

  const draw = useCallback(() => {
    const v = videoRef.current
    const canvas = rightRef.current
    if (v && canvas) {
      const W = canvas.clientWidth || 640
      const H = canvas.clientHeight || 360
      const dpr = window.devicePixelRatio || 1
      if (canvas.width !== W * dpr) {
        canvas.width = W * dpr
        canvas.height = H * dpr
      }
      const ctx = canvas.getContext('2d')
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.drawImage(v, 0, 0, W, H)
      const frame = Math.floor(v.currentTime * DEMO_FPS)
      const ann = demoAnnotations(frame)
      drawStudents(ctx, ann.students, W, H, {})
    }
    rafRef.current = requestAnimationFrame(draw)
  }, [])

  useEffect(() => {
    rafRef.current = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(rafRef.current)
  }, [draw])

  const onTime = () => {
    const v = videoRef.current
    setTime(v ? v.currentTime : 0)
    setDemo(v ? Math.floor(v.currentTime * DEMO_FPS) % 120 >= 60 : false)
  }

  const toggle = () => {
    const v = videoRef.current
    if (!v) return
    if (v.paused) {
      v.play()
      setPlaying(true)
    } else {
      v.pause()
      setPlaying(false)
    }
  }

  const restart = () => {
    const v = videoRef.current
    if (v) v.currentTime = 0
    setPlaying(false)
  }

  const step = (n) => {
    const v = videoRef.current
    if (!v) return
    v.pause()
    setPlaying(false)
    v.currentTime = Math.min(Math.max(v.currentTime + n / DEMO_FPS, 0), v.duration || 0)
  }

  const scrub = (e) => {
    const v = videoRef.current
    if (!v) return
    v.currentTime = Number(e.target.value)
    setTime(v.currentTime)
  }

  const fmt = (s) => {
    const m = Math.floor(s / 60)
    return `${m}:${String(Math.floor(s % 60)).padStart(2, '0')}`
  }

  const load = (name) => {
    setVideoName(name)
    setTime(0)
    setDuration(0)
    setPlaying(false)
  }

  return (
    <div className="glass flex flex-col overflow-hidden rounded-2xl">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-4 py-3">
        <div className="flex items-center gap-2">
          <ScanLine size={17} className="text-accent-2" />
          <span className="text-sm font-semibold">Dataset Playback Sync</span>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <select
            value={videoName}
            onChange={(e) => load(e.target.value)}
            className="rounded-xl border border-line bg-surface px-3 py-1.5 text-sm outline-none"
            aria-label="Dataset video"
          >
            {DATASET_VIDEOS.map((v) => (
              <option key={v.id} value={v.id}>
                {v.name}
              </option>
            ))}
          </select>
          {/* Playback transport */}
          <div className="flex items-center gap-1 rounded-xl border border-line p-1">
            <button
              onClick={toggle}
              className="rounded-lg bg-accent p-1.5 text-white transition hover:opacity-90"
              aria-label={playing ? 'Pause' : 'Play'}
            >
              {playing ? <Pause size={14} /> : <Play size={14} />}
            </button>
            <button
              onClick={() => step(-1)}
              className="rounded-lg p-1.5 text-muted transition hover:bg-surface-2 hover:text-ink"
              aria-label="Step back one frame"
            >
              <SkipBack size={14} />
            </button>
            <button
              onClick={() => step(1)}
              className="rounded-lg p-1.5 text-muted transition hover:bg-surface-2 hover:text-ink"
              aria-label="Step forward one frame"
            >
              <SkipForward size={14} />
            </button>
            <button
              onClick={restart}
              className="rounded-lg p-1.5 text-muted transition hover:bg-surface-2 hover:text-ink"
              aria-label="Restart"
            >
              <RotateCcw size={14} />
            </button>
          </div>
          <span className="font-mono text-xs text-muted">
            {fmt(time)} / {fmt(duration || 0)}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-0 md:grid-cols-2">
        {/* Left: raw source */}
        <div className="border-b border-line md:border-b-0 md:border-r">
          <div className="flex items-center justify-between px-4 py-2">
            <span className="text-[11px] font-bold uppercase tracking-wider text-muted">
              Raw source
            </span>
            <span className="font-mono text-[10px] text-muted">{videoName}</span>
          </div>
          <video
            ref={videoRef}
            src={datasetUrl(videoName)}
            preload="metadata"
            className="aspect-video w-full bg-black"
            onTimeUpdate={onTime}
            onLoadedMetadata={() => setDuration(videoRef.current?.duration || 0)}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
          />
        </div>

        {/* Right: model monitoring view (same video, client overlay) */}
        <div>
          <div className="flex items-center justify-between px-4 py-2">
            <span className="text-[11px] font-bold uppercase tracking-wider text-accent">
              Model monitoring
            </span>
            <span
              className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${
                demo ? 'bg-warning/15 text-warning' : 'bg-accent/15 text-accent'
              }`}
            >
              {demo ? 'ALERT PHASE' : 'NOMINAL PHASE'}
            </span>
          </div>
          <canvas ref={rightRef} className="aspect-video w-full bg-black" />
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-line px-4 py-3">
        <span className="text-xs text-muted">
          Frame-locked playback · demo pose overlay (deterministic in frame) · ByteTrack
          tracking vectors + normalized keypoints active
        </span>
        <span className="font-mono text-xs text-muted">
          frame {Math.floor(time * DEMO_FPS)} @ {DEMO_FPS} fps
        </span>
      </div>
    </div>
  )
}
