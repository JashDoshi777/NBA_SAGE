import { useState, useEffect } from 'react'
import './index.css'

// Icons
import {
  IconLive, IconTarget, IconChart, IconTrophy, IconCrown,
  IconVs, IconRefresh, IconBarChart
} from './icons'

// Pages
import LiveGames from './pages/LiveGames'
import Predictions from './pages/Predictions'
import Accuracy from './pages/Accuracy'
import MvpRace from './pages/MvpRace'
import Championship from './pages/Championship'
import HeadToHead from './pages/HeadToHead'
import Analytics from './pages/Analytics'

// Chevron icon for collapse toggle
function IconChevron({ className = '', direction = 'left' }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      {direction === 'left' ? (
        <polyline points="15 18 9 12 15 6"></polyline>
      ) : (
        <polyline points="9 18 15 12 9 6"></polyline>
      )}
    </svg>
  )
}

function App() {
  const [activePage, setActivePage] = useState('live')
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [sidebarHovered, setSidebarHovered] = useState(false)

  const handleRefresh = () => {
    setIsRefreshing(true)
    window.location.reload()
  }

  const navSections = [
    {
      title: 'Games',
      items: [
        { id: 'live', name: 'Live Games', icon: IconLive },
        { id: 'predictions', name: 'Predictions', icon: IconTarget },
      ]
    },
    {
      title: 'Analysis',
      items: [
        { id: 'h2h', name: 'Head to Head', icon: IconVs },
        { id: 'analytics', name: 'Analytics', icon: IconBarChart },
        { id: 'accuracy', name: 'Model Accuracy', icon: IconChart },
      ]
    },
    {
      title: 'Rankings',
      items: [
        { id: 'mvp', name: 'MVP Race', icon: IconTrophy },
        { id: 'championship', name: 'Championship', icon: IconCrown },
      ]
    }
  ]

  const renderPage = () => {
    switch (activePage) {
      case 'live': return <LiveGames />
      case 'predictions': return <Predictions />
      case 'accuracy': return <Accuracy />
      case 'analytics': return <Analytics />
      case 'mvp': return <MvpRace />
      case 'championship': return <Championship />
      case 'h2h': return <HeadToHead />
      default: return <LiveGames />
    }
  }

  // Determine if sidebar should be expanded (not collapsed OR being hovered)
  const isExpanded = !sidebarCollapsed || sidebarHovered

  return (
    <div className={`app-layout ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
      {/* Sidebar */}
      <aside
        className={`sidebar ${sidebarCollapsed ? 'collapsed' : ''} ${sidebarHovered ? 'hovered' : ''}`}
        onMouseEnter={() => sidebarCollapsed && setSidebarHovered(true)}
        onMouseLeave={() => setSidebarHovered(false)}
      >
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <div className="sidebar-logo-icon">
              <img
                src="https://cdn.nba.com/logos/leagues/logo-nba.svg"
                alt="NBA"
                style={{ width: '32px', height: '32px' }}
              />
            </div>
            <span className="sidebar-logo-text">NBA sage</span>
          </div>
          <button
            className="sidebar-toggle"
            onClick={() => {
              setSidebarCollapsed(!sidebarCollapsed)
              setSidebarHovered(false)
            }}
            title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            <IconChevron direction={sidebarCollapsed ? 'right' : 'left'} />
          </button>
        </div>

        <nav className="sidebar-nav">
          {navSections.map((section) => (
            <div key={section.title} className="nav-section">
              <div className="nav-section-title">{section.title}</div>
              {section.items.map((item) => (
                <div
                  key={item.id}
                  className={`nav-item ${activePage === item.id ? 'active' : ''}`}
                  onClick={() => setActivePage(item.id)}
                  title={sidebarCollapsed && !sidebarHovered ? item.name : ''}
                >
                  <item.icon className="nav-icon" />
                  <span className="nav-text">{item.name}</span>
                </div>
              ))}
            </div>
          ))}
        </nav>

        <div className="sidebar-footer">
          <button
            className="btn btn-ghost btn-block"
            onClick={handleRefresh}
            disabled={isRefreshing}
            title={sidebarCollapsed && !sidebarHovered ? 'Refresh Data' : ''}
          >
            <IconRefresh className={`nav-icon ${isRefreshing ? 'spinning' : ''}`} />
            <span className="nav-text">Refresh Data</span>
          </button>
          <div className="sidebar-version">
            NBA sage
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        {renderPage()}
      </main>
    </div>
  )
}

export default App

