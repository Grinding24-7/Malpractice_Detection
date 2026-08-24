import { useCallback, useRef, useState } from 'react'
import { UploadCloud, FileVideo, X } from 'lucide-react'

const ACCEPTED = ['video/mp4', 'video/avi', 'video/x-msvideo', 'video/quicktime', 'video/x-matroska']
const ACCEPTED_EXT = ['.mp4', '.avi', '.mov', '.mkv']

export function isValidVideo(file) {
  if (ACCEPTED.includes(file.type)) return true
  const name = file.name.toLowerCase()
  return ACCEPTED_EXT.some((ext) => name.endsWith(ext))
}

/**
 * DropZone — drag-and-drop video uploader for the demo showcase.
 * Accepts .mp4 / .avi / .mov (and .mkv), validates, and lifts the file +
 * a local object URL up to the parent via `onFile`.
 */
export default function DropZone({ file, onFile }) {
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

  const accept = useCallback((candidate) => {
    if (!candidate) return
    if (!isValidVideo(candidate)) {
      setError(`Unsupported file "${candidate.name}". Use .mp4, .avi or .mov.`)
      return
    }
    setError(null)
    onFile(candidate)
  }, [onFile])

  const onDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    accept(e.dataTransfer.files && e.dataTransfer.files[0])
  }

  const clear = (e) => {
    e.stopPropagation()
    onFile(null)
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current && inputRef.current.click()}
      className={`group relative flex min-h-[180px] cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed p-6 text-center transition-all ${
        dragging
          ? 'border-accent bg-accent/10 scale-[1.01]'
          : 'border-line bg-surface hover:border-accent/60 hover:bg-surface-2/50'
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".mp4,.avi,.mov,.mkv,video/*"
        className="hidden"
        onChange={(e) => accept(e.target.files && e.target.files[0])}
      />

      {file ? (
        <div className="flex w-full flex-col items-center gap-3">
          <video
            src={URL.createObjectURL(file)}
            muted
            preload="metadata"
            className="max-h-44 w-full max-w-md rounded-xl bg-black"
            onClick={(e) => e.stopPropagation()}
          />
          <div className="flex items-center gap-2 text-sm">
            <FileVideo size={15} className="text-accent" />
            <span className="font-medium text-ink">{file.name}</span>
            <span className="rounded-full bg-surface-2 px-2 py-0.5 font-mono text-[11px] text-muted">
              {(file.size / 1048576).toFixed(1)} MB
            </span>
            <button
              onClick={clear}
              className="rounded-lg p-1 text-muted transition hover:bg-surface-2 hover:text-danger"
              aria-label="Remove file"
            >
              <X size={15} />
            </button>
          </div>
        </div>
      ) : (
        <>
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-accent/15 text-accent">
            <UploadCloud size={24} />
          </div>
          <div>
            <p className="text-sm font-semibold text-ink">
              Drag & drop a recording here
            </p>
            <p className="mt-1 text-xs text-muted">
              or click to browse — .mp4, .avi, .mov
            </p>
          </div>
        </>
      )}

      {error && (
        <div className="absolute bottom-3 left-1/2 w-max max-w-full -translate-x-1/2 rounded-full bg-danger/15 px-3 py-1 text-xs font-medium text-danger">
          {error}
        </div>
      )}
    </div>
  )
}
