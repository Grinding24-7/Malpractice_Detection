import { useState } from 'react'
import { UploadCloud, Radio, AlignVerticalSpaceAround, Vault, School, ScanEye } from 'lucide-react'
import UploadView from './views/UploadView'
import LiveFeedView from './views/LiveFeedView'
import AlignmentView from './views/AlignmentView'
import EvidenceVaultView from './views/EvidenceVaultView'
import ClassroomsView from './views/ClassroomsView'
import InspectorView from './views/InspectorView'
import Sidebar from './components/Sidebar'
import Topbar from './components/Topbar'

const NAV = [
  { id: 'upload', label: 'Upload & Demo', icon: UploadCloud },
  { id: 'live', label: 'Live Monitoring', icon: Radio },
  { id: 'inspector', label: 'Model Inspector', icon: ScanEye },
  { id: 'alignment', label: 'Video Alignment', icon: AlignVerticalSpaceAround },
  { id: 'evidence', label: 'Evidence Vault', icon: Vault },
  { id: 'classrooms', label: 'Classrooms', icon: School },
]

/**
 * App — root layout. View switching is state-driven (no router dependency so
 * deep links never 404 behind the Flask static file server). The active
 * classroom selected in the Topbar is shared with every view via prop.
 */
export default function App() {
  const [view, setView] = useState('upload')
  const [activeClassroom, setActiveClassroom] = useState(null)

  const nav = NAV.find((n) => n.id === view)
  const renderView = () => {
    switch (view) {
      case 'live':
        return <LiveFeedView classroom={activeClassroom} />
      case 'inspector':
        return <InspectorView classroom={activeClassroom} />
      case 'alignment':
        return <AlignmentView classroom={activeClassroom} />
      case 'evidence':
        return <EvidenceVaultView classroom={activeClassroom} />
      case 'classrooms':
        return <ClassroomsView />
      default:
        return <UploadView />
    }
  }

  return (
    <div className="flex h-full min-h-screen bg-bg text-ink">
      <Sidebar nav={NAV} active={view} onSelect={setView} />
      <div className="flex min-w-0 flex-1 flex-col">
        <Topbar
          title={nav.label}
          onClassroomChange={setActiveClassroom}
          activeClassroom={activeClassroom}
        />
        <main className="flex-1 overflow-y-auto px-5 py-5 lg:px-8">
          {renderView()}
        </main>
      </div>
    </div>
  )
}
