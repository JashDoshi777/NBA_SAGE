import { useState, useEffect } from 'react'
import { predictGame, getTeams } from '../api'
import { TeamLogo, getTeamName } from '../teamLogos'

function Predictions() {
    const [teams, setTeams] = useState([])
    const [homeTeam, setHomeTeam] = useState('LAL')
    const [awayTeam, setAwayTeam] = useState('BOS')
    const [prediction, setPrediction] = useState(null)
    const [loading, setLoading] = useState(false)

    useEffect(() => {
        getTeams().then(data => {
            setTeams(data.teams || [])
        }).catch(console.error)
    }, [])

    const handlePredict = async () => {
        if (!homeTeam || !awayTeam || homeTeam === awayTeam) return

        setLoading(true)
        try {
            const result = await predictGame(homeTeam, awayTeam)
            setPrediction(result)
        } catch (err) {
            console.error('Prediction failed:', err)
        } finally {
            setLoading(false)
        }
    }

    const homeProb = prediction ? (prediction.home_win_probability * 100) : 50
    const awayProb = prediction ? (prediction.away_win_probability * 100) : 50

    return (
        <div className="animate-fadeIn">
            <div className="page-header">
                <h1 className="page-title">Game Predictions</h1>
                <p className="page-description">Select teams to get AI-powered win probabilities</p>
            </div>

            {/* Team Selector */}
            <div className="card" style={{ marginBottom: 'var(--space-6)' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: 'var(--space-6)', alignItems: 'end' }}>
                    {/* Away Team */}
                    <div className="form-group" style={{ marginBottom: 0 }}>
                        <label className="form-label">Away Team</label>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
                            <TeamLogo abbrev={awayTeam} size="md" />
                            <select
                                className="form-select"
                                value={awayTeam}
                                onChange={(e) => setAwayTeam(e.target.value)}
                            >
                                {teams.map(team => (
                                    <option key={team.id} value={team.abbrev}>{team.abbrev} - {getTeamName(team.abbrev)}</option>
                                ))}
                            </select>
                        </div>
                    </div>

                    <div style={{ color: 'var(--text-dim)', fontWeight: '600', paddingBottom: 'var(--space-3)' }}>@</div>

                    {/* Home Team */}
                    <div className="form-group" style={{ marginBottom: 0 }}>
                        <label className="form-label">Home Team</label>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
                            <TeamLogo abbrev={homeTeam} size="md" />
                            <select
                                className="form-select"
                                value={homeTeam}
                                onChange={(e) => setHomeTeam(e.target.value)}
                            >
                                {teams.map(team => (
                                    <option key={team.id} value={team.abbrev}>{team.abbrev} - {getTeamName(team.abbrev)}</option>
                                ))}
                            </select>
                        </div>
                    </div>
                </div>

                <button
                    className="btn btn-primary btn-lg btn-block"
                    onClick={handlePredict}
                    disabled={loading || homeTeam === awayTeam}
                    style={{ marginTop: 'var(--space-6)' }}
                >
                    {loading ? 'Analyzing...' : 'Generate Prediction'}
                </button>
            </div>

            {/* Prediction Result */}
            {prediction && (
                <div className="card animate-slideUp">
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: 'var(--space-8)', alignItems: 'center' }}>
                        {/* Away Team */}
                        <div style={{ textAlign: 'center' }}>
                            <TeamLogo abbrev={prediction.away_team} size="xl" />
                            <div style={{ marginTop: 'var(--space-3)' }}>
                                <div style={{ fontSize: '1.25rem', fontWeight: '600' }}>{prediction.away_team}</div>
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                    ELO: {prediction.away_elo?.toFixed(0) || 'N/A'}
                                </div>
                            </div>
                            <div style={{ fontSize: '2.5rem', fontWeight: '700', color: 'var(--accent-secondary)', marginTop: 'var(--space-4)' }}>
                                {awayProb.toFixed(1)}%
                            </div>
                        </div>

                        {/* Prediction Center */}
                        <div style={{ textAlign: 'center' }}>
                            <div style={{ fontSize: '0.6875rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', marginBottom: 'var(--space-2)' }}>
                                Predicted Winner
                            </div>
                            <div style={{ fontSize: '1.75rem', fontWeight: '700', color: 'var(--accent-primary)' }}>
                                {prediction.predicted_winner}
                            </div>
                            <span className={`badge confidence-${prediction.confidence}`} style={{ marginTop: 'var(--space-3)' }}>
                                {prediction.confidence?.toUpperCase()} CONFIDENCE
                            </span>
                            <div style={{ marginTop: 'var(--space-4)', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                ELO Difference: {prediction.elo_diff > 0 ? '+' : ''}{prediction.elo_diff?.toFixed(0)}
                            </div>
                        </div>

                        {/* Home Team */}
                        <div style={{ textAlign: 'center' }}>
                            <TeamLogo abbrev={prediction.home_team} size="xl" />
                            <div style={{ marginTop: 'var(--space-3)' }}>
                                <div style={{ fontSize: '1.25rem', fontWeight: '600' }}>{prediction.home_team}</div>
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                    ELO: {prediction.home_elo?.toFixed(0) || 'N/A'}
                                </div>
                            </div>
                            <div style={{ fontSize: '2.5rem', fontWeight: '700', color: 'var(--accent-primary)', marginTop: 'var(--space-4)' }}>
                                {homeProb.toFixed(1)}%
                            </div>
                        </div>
                    </div>

                    {/* Probability Bar */}
                    <div className="probability-bar-container" style={{ marginTop: 'var(--space-8)' }}>
                        <div className="probability-bar" style={{ height: '8px' }}>
                            <div className="probability-fill-away" style={{ width: `${awayProb}%` }}></div>
                            <div className="probability-fill-home" style={{ width: `${homeProb}%` }}></div>
                        </div>
                        <div className="probability-labels" style={{ marginTop: 'var(--space-3)' }}>
                            <span>{prediction.away_team}: {awayProb.toFixed(1)}%</span>
                            <span>{prediction.home_team}: {homeProb.toFixed(1)}%</span>
                        </div>
                    </div>

                    {/* Factors */}
                    {prediction.factors && prediction.factors.length > 0 && (
                        <div style={{ marginTop: 'var(--space-6)', paddingTop: 'var(--space-6)', borderTop: 'var(--border-subtle)' }}>
                            <h4 style={{ marginBottom: 'var(--space-4)', color: 'var(--text-secondary)' }}>Key Factors</h4>
                            <ul style={{ listStyle: 'none' }}>
                                {prediction.factors.map((factor, idx) => (
                                    <li key={idx} style={{
                                        padding: 'var(--space-2) 0',
                                        color: 'var(--text-muted)',
                                        fontSize: '0.875rem'
                                    }}>
                                        • {factor}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}

export default Predictions
