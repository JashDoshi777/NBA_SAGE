import { useState, useEffect } from 'react'
import { TeamLogo, getTeamName } from '../teamLogos'

// API call for standings
async function getStandings() {
    const response = await fetch('http://localhost:8000/api/standings')
    return response.json()
}

function Standings() {
    const [standings, setStandings] = useState({ east: [], west: [] })
    const [loading, setLoading] = useState(true)
    const [activeConference, setActiveConference] = useState('east')

    useEffect(() => {
        getStandings()
            .then(data => {
                setStandings(data)
                setLoading(false)
            })
            .catch(err => {
                console.error('Failed to load standings:', err)
                setLoading(false)
            })
    }, [])

    if (loading) {
        return (
            <div className="loading">
                <div className="spinner"></div>
                <p className="loading-text">Loading standings...</p>
            </div>
        )
    }

    const currentStandings = activeConference === 'east' ? standings.east : standings.west

    return (
        <div className="animate-fadeIn">
            <div className="page-header">
                <h1 className="page-title">Season Standings</h1>
                <p className="page-description">2025-26 NBA Conference standings</p>
            </div>

            {/* Conference Tabs */}
            <div style={{ display: 'flex', gap: 'var(--space-2)', marginBottom: 'var(--space-6)' }}>
                <button
                    className={`btn ${activeConference === 'east' ? 'btn-primary' : 'btn-secondary'}`}
                    onClick={() => setActiveConference('east')}
                >
                    Eastern Conference
                </button>
                <button
                    className={`btn ${activeConference === 'west' ? 'btn-primary' : 'btn-secondary'}`}
                    onClick={() => setActiveConference('west')}
                >
                    Western Conference
                </button>
            </div>

            {/* Standings Table */}
            <div className="table-container">
                <table className="data-table">
                    <thead>
                        <tr>
                            <th style={{ width: '50px' }}>Rank</th>
                            <th>Team</th>
                            <th style={{ textAlign: 'center' }}>W</th>
                            <th style={{ textAlign: 'center' }}>L</th>
                            <th style={{ textAlign: 'center' }}>PCT</th>
                            <th style={{ textAlign: 'center' }}>GB</th>
                            <th style={{ textAlign: 'center' }}>Streak</th>
                        </tr>
                    </thead>
                    <tbody>
                        {currentStandings?.length > 0 ? currentStandings.map((team, idx) => (
                            <tr key={team.team || idx}>
                                <td style={{ fontWeight: '600', color: idx < 6 ? 'var(--accent-success)' : idx < 10 ? 'var(--text-secondary)' : 'var(--text-muted)' }}>
                                    {idx + 1}
                                </td>
                                <td>
                                    <div className="table-team">
                                        <TeamLogo abbrev={team.team_abbrev || team.team} size="sm" />
                                        <span style={{ fontWeight: '500' }}>{team.team_name || getTeamName(team.team_abbrev || team.team)}</span>
                                    </div>
                                </td>
                                <td style={{ textAlign: 'center', fontFamily: 'var(--font-mono)' }}>{team.wins || 0}</td>
                                <td style={{ textAlign: 'center', fontFamily: 'var(--font-mono)' }}>{team.losses || 0}</td>
                                <td style={{ textAlign: 'center', fontFamily: 'var(--font-mono)', color: 'var(--accent-primary)' }}>
                                    {((team.win_pct || 0) * 100).toFixed(1)}%
                                </td>
                                <td style={{ textAlign: 'center', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                                    {team.gb || '-'}
                                </td>
                                <td style={{ textAlign: 'center' }}>
                                    <span className={`badge ${team.streak?.includes('W') ? 'badge-success' : 'badge-danger'}`}>
                                        {team.streak || '-'}
                                    </span>
                                </td>
                            </tr>
                        )) : (
                            <tr>
                                <td colSpan="7" style={{ textAlign: 'center', padding: 'var(--space-8)', color: 'var(--text-muted)' }}>
                                    No standings data available. API endpoint may need to be added.
                                </td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    )
}

export default Standings
