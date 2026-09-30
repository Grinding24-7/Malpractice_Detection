import { useEffect, useMemo, useRef, useState } from 'react'
import { Line } from 'react-chartjs-2'
import { Clapperboard, Play, Pause, RotateCcw, AlignVerticalSpaceAround } from 'lucide-react'
import { chartPalette, useTheme } from '../theme'
import { datasetUrl } from '../api/client'
import {
  DATASET_VIDEOS,
  makeFeatureTensor,
  velocitySeries,
} from '../data/mockData'

const FPS = 5 // feature frames per second (matches the 5 FPS AI sub-sampling)

/**
 * FeatureHeatmap — compact (channels x time) visualization of the (B, T, F)
 * tensor. The active column tracks the video playhead so you can see which
 * feature frame corresponds to the frame currently on screen.
 */
function FeatureHeatmap({ tensor, sample, currentFrame }) {
  const canvasRef = useRef(null)
  const rows = 24 // downsampled channel rows

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const t = tensor[sample]
    const T = t.length
    const F = t[0].length
    const W = canvas.clientWidth || 600
    const H = 90
    canvas.width = W
    canvas.height = H
    const ctx = canvas.getContext('2d')
    const cellW = W / T
    const cellH = H / rows

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < T; c++) {
        // Average a slice of feature channels for the row.
        let v = 0
        const step = Math.max(1, Math.floor(F / rows))
        for (let fi = r * step; fi < Math.min(F, (r + 1) * step); fi++) {
          v += t[c][fi]
        }
        v = v / Math.max(step, 1)
        const norm = Math.min(1, v / 0.9)
        ctx.fillStyle = `rgba(16, 185, 129, ${norm})`
        ctx.fillRect(c * cellW, r * cellH, Math.ceil(cellW) + 0.5, Math.ceil(cellH) + 0.5)
      }
    }
    // Playhead column.
    if (currentFrame >= 0 && currentFrame < T) {
      ctx.fillStyle = 'rgba(99, 102, 241, 0.85)'
      ctx.fillRect(Math.floor(currentFrame * cellW), 0, 3, H)
    }
  }, [tensor, sample, currentFrame])

  return <canvas ref={canvasRef} className="w-full" style={{ height: 90 }} />
}

/**
 * AlignmentView — Module 2: Dataset & Video Alignment Inspector.
 * Split-screen view mapping the (B, T, F) feature tensor directly against the
 * playing classroom dataset video, with a time-synchronized keypoint velocity
 * chart and a frame-aligned heatmap.
 */
export default function AlignmentView() {
  const { dark } = useTheme()
  const [videoName, setVideoName] = useState(DATASET_VIDEOS[0].id)
  const [sample, setSample] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [currentFrame, setCurrentFrame] = useState(0)
  const videoRef = useRef(null)

  const tensor = useMemo(() => makeFeatureTensor(4, 150, 117), [])
  const velocity = useMemo(() => velocitySeries(tensor), [tensor])
  const B = tensor.length
  const T = tensor[0].length
  const F = tensor[0][0].length

  const onTimeUpdate = () => {
    const v = videoRef.current
    if (v) setCurrentFrame(Math.floor(v.currentTime * FPS))
  }

  const palette = chartPalette()
  const chartData = {
    labels: Array.from({ length: T }, (_, i) => i),
    datasets: [
      {
        label: 'Keypoint velocity',
        data: velocity,
        borderColor: palette.accent,
        backgroundColor: palette.accent + '22',
        fill: true,
        tension: 0.3,
        pointRadius: 0,
      },
      // Spike markers — frames where velocity crosses the threshold.
      {
        label: 'Velocity spike',
        data: velocity.map((v, i) =>
          i > 0 && v > 0.5 && velocity[i - 1] <= 0.5 ? v : null,
        ),
        borderColor: palette.danger,
        backgroundColor: palette.danger,
        pointRadius: 4,
        pointStyle: 'triangle',
      },
    ],
  }

  const chartOptions = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      interaction: { mode: 'nearest', intersect: false },
      plugins: {
        legend: { labels: { color: palette.muted, usePointStyle: true } },
        // Playhead: vertical line at the frame currently playing in the video.
        annotation: {
          annotations: {
            playhead: {
              type: 'line',
              xMin: currentFrame,
              xMax: currentFrame,
              borderColor: palette.accent2,
              borderWidth: 2,
              label: {
                display: true,
                content: `t=${currentFrame}`,
                position: 'start',
                color: palette.surface,
                backgroundColor: palette.accent2,
                font: { size: 9 },
              },
            },
          },
        },
      },
      scales: {
        x: { title: { display: true, text: 'Frame index (5 fps)', color: palette.muted }, ticks: { color: palette.muted }, grid: { color: palette.grid } },
        y: { title: { display: true, text: 'Velocity (Δ-features)', color: palette.muted }, ticks: { color: palette.muted }, grid: { color: palette.grid } },
      },
    }),
    [palette, currentFrame],
  )

  const togglePlay = () => {
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

  return (
    <div className="fade-in space-y-5">
      <div className="glass flex flex-wrap items-center justify-between gap-3 rounded-2xl px-4 py-3">
        <div className="flex items-center gap-2">
          <AlignVerticalSpaceAround size={17} className="text-accent-2" />
          <span className="text-sm font-semibold">
            Dataset ↔ Feature Alignment
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <select
            value={videoName}
            onChange={(e) => setVideoName(e.target.value)}
            className="rounded-xl border border-line bg-surface px-3 py-1.5 text-sm outline-none"
            aria-label="Dataset video"
          >
            {DATASET_VIDEOS.map((v) => (
              <option key={v.id} value={v.id}>
                {v.name}
              </option>
            ))}
          </select>
          <select
            value={sample}
            onChange={(e) => setSample(Number(e.target.value))}
            className="rounded-xl border border-line bg-surface px-3 py-1.5 text-sm outline-none"
            aria-label="Feature sample"
          >
            {Array.from({ length: B }, (_, i) => (
              <option key={i} value={i}>
                Sample {i + 1}
              </option>
            ))}
          </select>
          <span className="rounded-full bg-surface-2 px-2.5 py-1 font-mono text-xs text-muted">
            B={B} · T={T} · F={F}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
        {/* Left: playing classroom video */}
        <div className="glass overflow-hidden rounded-2xl">
          <div className="flex items-center justify-between border-b border-line px-4 py-3">
            <span className="flex items-center gap-2 text-sm font-semibold">
              <Clapperboard size={16} className="text-accent" />
              {videoName}
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={togglePlay}
                className="rounded-xl bg-accent p-2 text-white transition hover:opacity-90"
                aria-label={playing ? 'Pause' : 'Play'}
              >
                {playing ? <Pause size={15} /> : <Play size={15} />}
              </button>
              <button
                onClick={() => {
                  const v = videoRef.current
                  if (v) {
                    v.currentTime = 0
                    setCurrentFrame(0)
                  }
                }}
                className="rounded-xl border border-line p-2 text-muted transition hover:text-ink"
                aria-label="Restart"
              >
                <RotateCcw size={15} />
              </button>
            </div>
          </div>
          <div className="bg-black">
            <video
              ref={videoRef}
              src={datasetUrl(videoName)}
              controls={false}
              className="aspect-video w-full"
              onTimeUpdate={onTimeUpdate}
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
            />
          </div>
        </div>

        {/* Right: synchronized feature analysis */}
        <div className="glass flex flex-col overflow-hidden rounded-2xl">
          <div className="border-b border-line px-4 py-3 text-sm font-semibold">
            Time-Synchronized Feature Chart
          </div>
          <div className="relative h-64 p-3">
            <Line data={chartData} options={chartOptions} />
          </div>
          <div className="border-t border-line px-4 py-3">
            <div className="mb-2 flex items-center justify-between text-xs text-muted">
              <span className="flex items-center gap-2">
                <span className="inline-block h-2.5 w-2.5 rounded-sm bg-emerald-500" />
                Keypoint velocity per frame
              </span>
              <span>
                frame <span className="font-mono text-ink">{currentFrame}</span> / {T}
              </span>
            </div>
            <FeatureHeatmap
              tensor={tensor}
              sample={sample}
              currentFrame={currentFrame}
            />
            <p className="mt-2 text-[11px] text-muted">
              (B, T, F) heatmap — column colors show the active feature window
              aligned to the video playhead.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
