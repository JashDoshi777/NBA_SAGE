import { useState, useEffect } from 'react'
import { getAccuracy } from '../api'
import { IconRefresh } from '../icons'

function Accuracy() {
    const [data, setData] = useState({ stats: {}, recent_predictions: [] })
    const [loading, setLoading] = useState(true)

    const fetchData = async () => {
        setLoading(true)
        try {
            const result = await getAccuracy()
            setData(result)
        } catch (err) {
            console.error('Failed to load accuracy:', err)
        } finally {
            setLoading(false)
        }
    }

    useEffect(() => {
        fetchData()
        // Auto-refresh every 60 seconds
        const interval = setInterval(fetchData, 60000)
        return () => clearInterval(interval)
    }, [])

    if (loading) {
        return (
            <div className="loading">
                <div className="spinner"></div>
                <p className="loading-text">Loading accuracy stats...</p>
            </div>
        )
    }

    const { stats, recent_predictions } = data
    const overallAccuracy = stats.overall_accuracy || 0
    const byConfidence = stats.by_confidence || {}

    return (
        <div className="animate-fadeIn">
            <div className="page-header">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="page-title">Model Accuracy</h1>
                        <p className="page-description">Track prediction performance and model reliability</p>
                    </div>
                    <button className="btn btn-secondary" onClick={fetchData}>
                        <IconRefresh className="nav-icon" />
                        Refresh
                    </button>
                </div>
            </div>

            {/* Primary Stats Grid */}
            <div className="stats-grid" style={{ marginBottom: 'var(--space-6)' }}>
                <div className="stat-card">
                    <div className="stat-label">Total Predictions</div>
                    <div className="stat-value">{stats.total_predictions || 0}</div>
                </div>
                <div className="stat-card">
                    <div className="stat-label">Completed Games</div>
                    <div className="stat-value">{stats.completed_games || 0}</div>
                </div>
                <div className="stat-card">
                    <div className="stat-label">Correct Predictions</div>
                    <div className="stat-value accent">{stats.correct_predictions || 0}</div>
                </div>
                <div className="stat-card">
                    <div className="stat-label">Overall Accuracy</div>
                    <div className="stat-value accent" style={{ fontSize: '2.5rem' }}>
                        {(overallAccuracy * 100).toFixed(1)}%
                    </div>
                </div>
            </div>

            {/* Detailed Metrics */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-6)', marginBottom: 'var(--space-8)' }}>
                {/* Performance Metrics */}
                <div className="card">
                    <h3 style={{ marginBottom: 'var(--space-4)' }}>Performance Metrics</h3>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-4)' }}>
                        <div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Current Streak</div>
                            <div style={{ fontSize: '1.5rem', fontWeight: '700', color: stats.streak_type === 'W' ? 'var(--accent-success)' : 'var(--accent-danger)' }}>
                                {stats.current_streak || 0}{stats.streak_type || ''}
                            </div>
                        </div>
                        <div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Last 10 Games</div>
                            <div style={{ fontSize: '1.5rem', fontWeight: '700' }}>
                                {stats.last_10_record || '0-0'}
                                <span style={{ fontSize: '0.875rem', color: 'var(--text-muted)', marginLeft: 'var(--space-2)' }}>
                                    ({((stats.last_10_accuracy || 0) * 100).toFixed(0)}%)
                                </span>
                            </div>
                        </div>
                        <div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Pending Predictions</div>
                            <div style={{ fontSize: '1.5rem', fontWeight: '700', color: 'var(--accent-warning)' }}>
                                {stats.pending_predictions || 0}
                            </div>
                        </div>
                        <div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Avg Probability</div>
                            <div style={{ fontSize: '1.5rem', fontWeight: '700' }}>
                                {((stats.avg_probability_correct || 0) * 100).toFixed(1)}%
                            </div>
                        </div>
                    </div>
                </div>

                {/* Home vs Away */}
                <div className="card">
                    <h3 style={{ marginBottom: 'var(--space-4)' }}>Home vs Away Picks</h3>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-4)' }}>
                        <div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Home Team Picks</div>
                            <div style={{ fontSize: '1.5rem', fontWeight: '700', color: 'var(--accent-primary)' }}>
                                {((stats.home_pick_accuracy || 0) * 100).toFixed(1)}%
                            </div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                {stats.home_picks_total || 0} picks
                            </div>
                        </div>
                        <div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Away Team Picks</div>
                            <div style={{ fontSize: '1.5rem', fontWeight: '700', color: 'var(--accent-secondary)' }}>
                                {((stats.away_pick_accuracy || 0) * 100).toFixed(1)}%
                            </div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                {stats.away_picks_total || 0} picks
                            </div>
                        </div>
                    </div>

                    {/* Visual bar */}
                    <div style={{ marginTop: 'var(--space-4)' }}>
                        <div style={{ display: 'flex', height: '8px', borderRadius: '4px', overflow: 'hidden' }}>
                            <div style={{
                                flex: stats.home_picks_total || 1,
                                background: 'var(--accent-primary)',
                                opacity: stats.home_pick_accuracy > stats.away_pick_accuracy ? 1 : 0.5
                            }}></div>
                            <div style={{
                                flex: stats.away_picks_total || 1,
                                background: 'var(--accent-secondary)',
                                opacity: stats.away_pick_accuracy > stats.home_pick_accuracy ? 1 : 0.5
                            }}></div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Accuracy by Confidence */}
            {Object.keys(byConfidence).length > 0 && (
                <div style={{ marginBottom: 'var(--space-8)' }}>
                    <h3 style={{ marginBottom: 'var(--space-4)' }}>Accuracy by Confidence Level</h3>
                    <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
                        {['high', 'medium', 'low'].map((conf) => {
                            const confData = byConfidence[conf] || { accuracy: 0, correct: 0, total: 0 }
                            const accuracyPercent = (confData.accuracy * 100).toFixed(1)
                            return (
                                <div key={conf} className="stat-card" style={{ textAlign: 'center' }}>
                                    <span className={`badge confidence-${conf}`} style={{ marginBottom: 'var(--space-3)' }}>
                                        {conf.toUpperCase()}
                                    </span>
                                    <div className="stat-value" style={{ fontSize: '2rem' }}>{accuracyPercent}%</div>
                                    <div className="stat-label" style={{ marginTop: 'var(--space-2)' }}>
                                        {confData.correct}/{confData.total} correct
                                    </div>
                                    {/* Progress ring visual */}
                                    <div style={{
                                        marginTop: 'var(--space-3)',
                                        height: '4px',
                                        background: 'var(--bg-tertiary)',
                                        borderRadius: '2px',
                                        overflow: 'hidden'
                                    }}>
                                        <div style={{
                                            height: '100%',
                                            width: `${accuracyPercent}%`,
                                            background: conf === 'high' ? 'var(--accent-success)' : conf === 'medium' ? 'var(--accent-warning)' : 'var(--accent-danger)',
                                            transition: 'width 0.3s ease'
                                        }}></div>
                                    </div>
                                </div>
                            )
                        })}
                    </div>
                </div>
            )}

            {/* Accuracy by Team */}
            {stats.by_team && Object.keys(stats.by_team).length > 0 && (
                <div style={{ marginBottom: 'var(--space-8)' }}>
                    <h3 style={{ marginBottom: 'var(--space-4)' }}>Accuracy by Team Predicted</h3>
                    <div className="table-container">
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Team</th>
                                    <th style={{ textAlign: 'center' }}>Correct</th>
                                    <th style={{ textAlign: 'center' }}>Total</th>
                                    <th style={{ textAlign: 'right' }}>Accuracy</th>
                                </tr>
                            </thead>
                            <tbody>
                                {Object.entries(stats.by_team)
                                    .sort((a, b) => b[1].accuracy - a[1].accuracy)
                                    .map(([team, data]) => (
                                        <tr key={team}>
                                            <td style={{ fontWeight: '500' }}>{team}</td>
                                            <td style={{ textAlign: 'center' }}>{data.correct}</td>
                                            <td style={{ textAlign: 'center', color: 'var(--text-muted)' }}>{data.total}</td>
                                            <td style={{ textAlign: 'right', color: 'var(--accent-primary)', fontWeight: '600', fontFamily: 'var(--font-mono)' }}>
                                                {(data.accuracy * 100).toFixed(1)}%
                                            </td>
                                        </tr>
                                    ))
                                }
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {/* Recent Predictions */}
            <div className="table-container">
                <div style={{ padding: 'var(--space-4) var(--space-4) 0', borderBottom: 'var(--border-subtle)' }}>
                    <h3 style={{ marginBottom: 'var(--space-4)' }}>Recent Predictions</h3>
                </div>

                {recent_predictions.length === 0 ? (
                    <div className="empty-state">
                        <p className="empty-state-title">No Predictions Yet</p>
                        <p className="empty-state-text">Visit the Live Games page to start tracking predictions.</p>
                    </div>
                ) : (
                    <table className="data-table">
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Matchup</th>
                                <th>Prediction</th>
                                <th style={{ textAlign: 'center' }}>Confidence</th>
                                <th style={{ textAlign: 'center' }}>Result</th>
                            </tr>
                        </thead>
                        <tbody>
                            {recent_predictions.map((pred, idx) => {
                                const isPending = pred.is_correct === -1
                                const isCorrect = pred.is_correct === 1

                                return (
                                    <tr key={idx}>
                                        <td style={{ color: 'var(--text-muted)', fontSize: '0.8125rem' }}>
                                            {pred.game_date || 'N/A'}
                                        </td>
                                        <td>
                                            <span style={{ fontWeight: '500' }}>{pred.away_team || 'N/A'}</span>
                                            <span style={{ color: 'var(--text-muted)' }}> @ </span>
                                            <span style={{ fontWeight: '500' }}>{pred.home_team || 'N/A'}</span>
                                        </td>
                                        <td style={{ color: 'var(--accent-primary)', fontWeight: '500' }}>
                                            {pred.predicted_winner || 'N/A'}
                                            <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem', marginLeft: 'var(--space-2)' }}>
                                                ({((pred.home_win_prob > 0.5 ? pred.home_win_prob : pred.away_win_prob) * 100 || 50).toFixed(0)}%)
                                            </span>
                                        </td>
                                        <td style={{ textAlign: 'center' }}>
                                            <span className={`badge confidence-${pred.confidence || 'medium'}`}>
                                                {(pred.confidence || 'medium').toUpperCase()}
                                            </span>
                                        </td>
                                        <td style={{ textAlign: 'center' }}>
                                            {isPending ? (
                                                <span className="badge badge-neutral">PENDING</span>
                                            ) : isCorrect ? (
                                                <span className="badge badge-success">CORRECT</span>
                                            ) : (
                                                <span className="badge badge-danger">WRONG</span>
                                            )}
                                        </td>
                                    </tr>
                                )
                            })}
                        </tbody>
                    </table>
                )}
            </div>
        </div>
    )
}

export default Accuracy
