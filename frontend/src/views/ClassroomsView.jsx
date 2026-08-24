import { useState } from 'react'
import { School, Camera, Grid3X3, Users, Plus, Trash2 } from 'lucide-react'
import LoadingSkeleton from '../components/LoadingSkeleton'
import { useAsync } from '../hooks/useAsync'
import { useToast } from '../components/Toast'
import { createClassroom, fetchClassrooms } from '../api/client'
import { MOCK_CLASSROOMS } from '../data/mockData'

const EMPTY = {
  name: '',
  camera_ip: '',
  rtsp_url: '',
  desk_rows: '6',
  desk_cols: '4',
  roster: '',
}

/**
 * ClassroomsView — Module 4: Classroom Management & Registration.
 * Form to register new classroom CCTV configurations (camera / RTSP, desk grid
 * layout, student roster) persisted via POST /api/classrooms, plus a selector
 * of all registered rooms.
 */
export default function ClassroomsView() {
  const { data, loading, refresh } = useAsync(
    () => fetchClassrooms().catch(() => MOCK_CLASSROOMS),
    [],
  )
  const { push } = useToast()
  const classrooms = data && data.length ? data : MOCK_CLASSROOMS
  const [form, setForm] = useState(EMPTY)
  const [submitting, setSubmitting] = useState(false)

  const set = (key) => (e) => setForm((f) => ({ ...f, [key]: e.target.value }))

  const submit = async (e) => {
    e.preventDefault()
    if (!form.name.trim() || !form.camera_ip.trim()) {
      push('Name and camera IP are required.', 'error')
      return
    }
    setSubmitting(true)
    const payload = {
      name: form.name.trim(),
      camera_ip: form.camera_ip.trim(),
      rtsp_url: form.rtsp_url.trim(),
      desk_rows: Number(form.desk_rows) || 1,
      desk_cols: Number(form.desk_cols) || 1,
      roster: form.roster
        .split('\n')
        .map((s) => s.trim())
        .filter(Boolean),
    }
    try {
      await createClassroom(payload)
      push(`Classroom "${payload.name}" registered.`)
      setForm(EMPTY)
      refresh()
    } catch {
      // Backend offline — keep the demo classroom list but notify clearly.
      push('Backend offline — classroom not persisted (demo mode).', 'error')
    } finally {
      setSubmitting(false)
    }
  }

  const inputClass =
    'w-full rounded-xl border border-line bg-surface px-3 py-2 text-sm outline-none transition focus:border-accent'

  return (
    <div className="fade-in space-y-5">
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
        {/* Registration form */}
        <form
          onSubmit={submit}
          className="glass rounded-2xl p-5"
        >
          <div className="mb-4 flex items-center gap-2">
            <Plus size={17} className="text-accent" />
            <h2 className="text-sm font-semibold">Register Classroom</h2>
          </div>

          <div className="space-y-3">
            <div>
              <label className="mb-1 block text-xs font-medium text-muted">Room name</label>
              <input
                value={form.name}
                onChange={set('name')}
                placeholder="Room C — Block 3"
                className={inputClass}
              />
            </div>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <div>
                <label className="mb-1 block text-xs font-medium text-muted">Camera IP</label>
                <input
                  value={form.camera_ip}
                  onChange={set('camera_ip')}
                  placeholder="192.168.10.23"
                  className={inputClass}
                />
              </div>
              <div>
                <label className="mb-1 block text-xs font-medium text-muted">RTSP stream URL</label>
                <input
                  value={form.rtsp_url}
                  onChange={set('rtsp_url')}
                  placeholder="rtsp://192.168.10.23:554/live"
                  className={inputClass}
                />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="mb-1 block text-xs font-medium text-muted">Desk rows</label>
                <input
                  type="number"
                  min="1"
                  value={form.desk_rows}
                  onChange={set('desk_rows')}
                  className={inputClass}
                />
              </div>
              <div>
                <label className="mb-1 block text-xs font-medium text-muted">Desk columns</label>
                <input
                  type="number"
                  min="1"
                  value={form.desk_cols}
                  onChange={set('desk_cols')}
                  className={inputClass}
                />
              </div>
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-muted">
                Student roster <span className="normal-case text-muted">(one per line)</span>
              </label>
              <textarea
                value={form.roster}
                onChange={set('roster')}
                rows="4"
                placeholder={'Alice\nBob\nCarol'}
                className={`${inputClass} resize-none font-mono`}
              />
            </div>
            <button
              type="submit"
              disabled={submitting}
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-accent to-accent-2 px-4 py-2.5 text-sm font-semibold text-white transition hover:opacity-90 disabled:opacity-50"
            >
              <Plus size={16} />
              {submitting ? 'Registering…' : 'Register classroom'}
            </button>
          </div>
        </form>

        {/* Registered classrooms */}
        <div>
          <div className="mb-3 flex items-center gap-2">
            <School size={17} className="text-accent-2" />
            <h2 className="text-sm font-semibold">Registered Classrooms</h2>
          </div>

          {loading ? (
            <div className="space-y-3">
              <LoadingSkeleton className="h-32" />
              <LoadingSkeleton className="h-32" />
            </div>
          ) : (
            <div className="space-y-3">
              {classrooms.map((c) => (
                <div key={c.id || c.name} className="glass rounded-2xl p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="flex items-center gap-2">
                        <School size={15} className="text-accent-2" />
                        <span className="text-sm font-semibold">{c.name}</span>
                        {c.source === 'demo' && (
                          <span className="rounded-full bg-surface-3 px-2 py-0.5 text-[10px] uppercase text-muted">
                            demo
                          </span>
                        )}
                      </div>
                      <div className="mt-1 flex items-center gap-2 text-xs text-muted">
                        <Camera size={12} />
                        <span className="font-mono">{c.camera_ip}</span>
                        <span className="hidden md:inline">· {c.rtsp_url}</span>
                      </div>
                    </div>
                    <button
                      onClick={() => push('Demo mode — delete disabled.', 'error')}
                      className="rounded-lg p-1.5 text-muted transition hover:bg-surface-2 hover:text-danger"
                      aria-label="Delete classroom"
                    >
                      <Trash2 size={15} />
                    </button>
                  </div>
                  <div className="mt-3 flex flex-wrap gap-2 text-[11px] text-muted">
                    <span className="flex items-center gap-1 rounded-full bg-surface-2 px-2.5 py-1">
                      <Grid3X3 size={11} /> {c.desk_rows}×{c.desk_cols} desk grid
                    </span>
                    <span className="flex items-center gap-1 rounded-full bg-surface-2 px-2.5 py-1">
                      <Users size={11} /> {c.roster.length} students
                    </span>
                    <span className="flex items-center gap-1 rounded-full bg-surface-2 px-2.5 py-1">
                      <Camera size={11} /> CCTV
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
