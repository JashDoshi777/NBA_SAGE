import { useState, useEffect } from 'react'
import { getMvpRace } from '../api'
import { TeamLogo } from '../teamLogos'

function MvpRace() {
    const [candidates, setCandidates] = useState([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        getMvpRace()
            .then(data => {
                setCandidates(data.candidates || [])
                setLoading(false)
            })
            .catch(err => {
                console.error('Failed to load MVP race:', err)
                setLoading(false)
            })
    }, [])

    if (loading) {
        return (
            <div className="loading">
                <div className="spinner"></div>
                <p className="loading-text">Loading MVP race...</p>
            </div>
        )
    }

    const top3 = candidates.slice(0, 3)
    const rest = candidates.slice(3)

    return (
        <div className="animate-fadeIn">
            <div className="page-header">
                <h1 className="page-title">MVP Race</h1>
                <p className="page-description">2025-26 MVP candidates ranked by performance metrics</p>
            </div>

            {/* Top 3 Podium */}
            {top3.length > 0 && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 'var(--space-4)', marginBottom: 'var(--space-8)' }}>
                    {top3.map((player, idx) => {
                        const rankStyles = [
                            { border: '1px solid #FFD700', boxShadow: '0 0 20px rgba(255, 215, 0, 0.1)' },
                            { border: '1px solid #C0C0C0' },
                            { border: '1px solid #CD7F32' },
                        ]
                        const rankLabels = ['1ST', '2ND', '3RD']

                        return (
                            <div
                                key={player.name}
                                className="card animate-slideUp"
                                style={{
                                    ...rankStyles[idx],
                                    textAlign: 'center',
                                    animationDelay: `${idx * 0.1}s`
                                }}
                            >
                                <div style={{
                                    fontSize: '0.6875rem',
                                    fontWeight: '700',
                                    letterSpacing: '0.1em',
                                    color: idx === 0 ? '#FFD700' : idx === 1 ? '#C0C0C0' : '#CD7F32',
                                    marginBottom: 'var(--space-4)'
                                }}>
                                    {rankLabels[idx]}
                                </div>

                                <div style={{
                                    width: '64px',
                                    height: '64px',
                                    borderRadius: '50%',
                                    background: 'var(--bg-elevated)',
                                    margin: '0 auto var(--space-3)',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    fontSize: '1.5rem',
                                    fontWeight: '700',
                                    color: 'var(--accent-primary)'
                                }}>
                                    {player.name.split(' ').map(n => n[0]).join('')}
                                </div>

                                <h3 style={{ marginBottom: 'var(--space-2)' }}>{player.name}</h3>
                                <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginBottom: 'var(--space-4)' }}>
                                    {player.ppg} PPG / {player.rpg} RPG / {player.apg} APG
                                </p>

                                <div style={{
                                    fontSize: '2rem',
                                    fontWeight: '700',
                                    color: 'var(--accent-primary)',
                                    marginBottom: 'var(--space-1)'
                                }}>
                                    {player.mvp_score}
                                </div>
                                <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                                    MVP Score
                                </div>
                            </div>
                        )
                    })}
                </div>
            )}

            {/* Full Rankings */}
            {rest.length > 0 && (
                <div className="table-container">
                    <table className="data-table">
                        <thead>
                            <tr>
                                <th style={{ width: '60px' }}>Rank</th>
                                <th>Player</th>
                                <th style={{ textAlign: 'right' }}>PPG</th>
                                <th style={{ textAlign: 'right' }}>RPG</th>
                                <th style={{ textAlign: 'right' }}>APG</th>
                                <th style={{ textAlign: 'right' }}>MVP Score</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rest.map((player) => (
                                <tr key={player.name}>
                                    <td style={{ fontWeight: '600', color: 'var(--text-muted)' }}>#{player.rank}</td>
                                    <td style={{ fontWeight: '500' }}>{player.name}</td>
                                    <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>{player.ppg}</td>
                                    <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>{player.rpg}</td>
                                    <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>{player.apg}</td>
                                    <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--accent-primary)', fontWeight: '600' }}>
                                        {player.mvp_score}
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

export default MvpRace
