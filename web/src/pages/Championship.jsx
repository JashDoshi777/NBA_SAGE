import { useState, useEffect } from 'react'
import { getChampionshipOdds } from '../api'
import { TeamLogo } from '../teamLogos'

function Championship() {
    const [teams, setTeams] = useState([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        getChampionshipOdds()
            .then(data => {
                setTeams(data.teams || [])
                setLoading(false)
            })
            .catch(err => {
                console.error('Failed to load championship odds:', err)
                setLoading(false)
            })
    }, [])

    if (loading) {
        return (
            <div className="loading">
                <div className="spinner"></div>
                <p className="loading-text">Loading championship odds...</p>
            </div>
        )
    }

    const top4 = teams.slice(0, 4)
    const rest = teams.slice(4)

    return (
        <div className="animate-fadeIn">
            <div className="page-header">
                <h1 className="page-title">Championship Odds</h1>
                <p className="page-description">2025-26 NBA Championship probability rankings</p>
            </div>

            {/* Top 4 Contenders */}
            {top4.length > 0 && (
                <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)', marginBottom: 'var(--space-8)' }}>
                    {top4.map((team, idx) => (
                        <div
                            key={team.team}
                            className="stat-card animate-slideUp"
                            style={{
                                textAlign: 'center',
                                animationDelay: `${idx * 0.1}s`,
                                border: idx === 0 ? '1px solid var(--accent-primary)' : 'var(--border-subtle)'
                            }}
                        >
                            <div style={{
                                fontSize: '0.6875rem',
                                fontWeight: '600',
                                color: 'var(--text-muted)',
                                marginBottom: 'var(--space-3)'
                            }}>
                                #{idx + 1} CONTENDER
                            </div>
                            <TeamLogo abbrev={team.team} size="lg" style={{ margin: '0 auto var(--space-3)' }} />
                            <div style={{ fontSize: '1.125rem', fontWeight: '600', marginBottom: 'var(--space-2)' }}>
                                {team.team}
                            </div>
                            <div className="stat-value accent" style={{ fontSize: '2.5rem' }}>
                                {team.odds}%
                            </div>
                            <div className="stat-label">Championship Odds</div>
                            <div style={{
                                marginTop: 'var(--space-3)',
                                fontSize: '0.75rem',
                                color: 'var(--text-muted)',
                                fontFamily: 'var(--font-mono)'
                            }}>
                                Win Rate: {team.win_pct}%
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {/* Full Rankings */}
            {rest.length > 0 && (
                <div className="table-container">
                    <table className="data-table">
                        <thead>
                            <tr>
                                <th style={{ width: '60px' }}>Rank</th>
                                <th>Team</th>
                                <th style={{ textAlign: 'right' }}>Championship Odds</th>
                                <th style={{ textAlign: 'right' }}>Win Rate</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rest.map((team) => (
                                <tr key={team.team}>
                                    <td style={{ fontWeight: '600', color: 'var(--text-muted)' }}>#{team.rank}</td>
                                    <td>
                                        <div className="table-team">
                                            <TeamLogo abbrev={team.team} size="sm" />
                                            <span style={{ fontWeight: '500' }}>{team.team}</span>
                                        </div>
                                    </td>
                                    <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--accent-primary)', fontWeight: '600' }}>
                                        {team.odds}%
                                    </td>
                                    <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                                        {team.win_pct}%
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    )
}

export default Championship
