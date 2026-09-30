import { useEffect, useMemo } from 'react'
import { Moon, Sun, School } from 'lucide-react'
import { useTheme } from '../theme'
import { useAsync } from '../hooks/useAsync'
import { fetchClassrooms } from '../api/client'
import { MOCK_CLASSROOMS } from '../data/mockData'

/**
 * Topbar — page title, global classroom selector (GET /classrooms) and the
 * Dark/Light theme toggle. The classroom selection is lifted to App so every
 * view can filter on it.
 */
export default function Topbar({ title, onClassroomChange, activeClassroom }) {
  const { dark, toggle } = useTheme()

  const { data } = useAsync(
    () => fetchClassrooms().catch(() => MOCK_CLASSROOMS),
    [],
  )

  const classrooms = useMemo(() => (data && data.length ? data : MOCK_CLASSROOMS), [data])
  const active = activeClassroom || (classrooms[0] ? classrooms[0].name : 'No classroom')

  useEffect(() => {
    if (!activeClassroom && classrooms[0]) onClassroomChange(classrooms[0].name)
  }, [classrooms, activeClassroom, onClassroomChange])

  return (
    <header className="glass sticky top-0 z-20 flex items-center justify-between gap-3 border-b border-line px-5 py-3 lg:px-8">
      <h1 className="text-lg font-semibold tracking-tight">{title}</h1>

      <div className="flex items-center gap-2">
        <label className="flex items-center gap-2 rounded-xl border border-line bg-surface px-3 py-2">
          <School size={15} className="text-muted" />
          <select
            value={active}
            onChange={(e) => onClassroomChange(e.target.value)}
            className="bg-transparent text-sm font-medium text-ink outline-none"
            aria-label="Active classroom"
          >
            {classrooms.map((c) => (
              <option key={c.id || c.name} value={c.name}>
                {c.name}
              </option>
            ))}
          </select>
        </label>

        <button
          onClick={toggle}
          className="rounded-xl border border-line bg-surface p-2 text-muted transition hover:text-ink"
          aria-label="Toggle dark / light theme"
          title="Toggle theme"
        >
          {dark ? <Sun size={17} /> : <Moon size={17} />}
        </button>
      </div>
    </header>
  )
}
