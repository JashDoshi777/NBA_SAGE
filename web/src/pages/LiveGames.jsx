import { useState, useEffect } from 'react'
import { getLiveGames } from '../api'
import { TeamLogo } from '../teamLogos'
import { IconRefresh } from '../icons'

// Confidence Meter Component - Visual gauge showing MODEL CONFIDENCE
// This shows how confident the model is in its prediction, not the win probability
// Confidence is calculated as: how far the win probability is from 50%
// 50% win prob = 0% confidence (could go either way)
// 75% win prob = 50% confidence (25% away from 50%, on 0-50 scale = 50%)
// 100% win prob = 100% confidence (50% away from 50%)
function ConfidenceMeter({ winProbability }) {
    // winProbability is expected to be between 50 and 100 (the higher prob of the two teams)
    // Calculate confidence as distance from 50%, normalized to 0-100 scale
    // The max distance is 50 (going from 50% to 100%), so we multiply by 2 to get 0-100
    const confidence = Math.round((Math.abs(winProbability - 50) / 50) * 100);

    // Normalize for the arc display (0 to 1)
    const normalizedValue = confidence / 100;

    // Color based on confidence level
    const getColor = () => {
        if (confidence >= 60) return '#22c55e'; // Green - High confidence
        if (confidence >= 35) return '#eab308'; // Yellow - Medium confidence
        return '#ef4444'; // Red - Low confidence
    };

    const getLabel = () => {
        if (confidence >= 60) return 'HIGH';
        if (confidence >= 35) return 'MEDIUM';
        return 'LOW';
    };

    // SVG arc calculation
    const radius = 40;
    const strokeWidth = 8;
    const circumference = Math.PI * radius; // Half circle
    const fillLength = normalizedValue * circumference;

    return (
        <div className="confidence-meter">
            <svg width="100" height="60" viewBox="0 0 100 60">
                {/* Background arc */}
                <path
                    d="M 10 50 A 40 40 0 0 1 90 50"
                    fill="none"
                    stroke="rgba(255,255,255,0.1)"
                    strokeWidth={strokeWidth}
                    strokeLinecap="round"
                />
                {/* Filled arc */}
                <path
                    d="M 10 50 A 40 40 0 0 1 90 50"
                    fill="none"
                    stroke={getColor()}
                    strokeWidth={strokeWidth}
                    strokeLinecap="round"
                    strokeDasharray={`${fillLength} ${circumference}`}
                    style={{
                        filter: `drop-shadow(0 0 6px ${getColor()})`,
                        transition: 'stroke-dasharray 0.5s ease-out'
                    }}
                />
            </svg>
            <div className="confidence-value" style={{ color: getColor() }}>
                {confidence}%
            </div>
            <div className="confidence-label" style={{ color: getColor() }}>
                {getLabel()}
            </div>
        </div>
    );
}


// Fetch roster for a team - use relative URL in production
const API_BASE = import.meta.env.DEV ? 'http://localhost:8000' : '';
async function getTeamRoster(teamAbbrev) {
    const response = await fetch(`${API_BASE}/api/roster/${teamAbbrev}`)
    return response.json()
}

function LiveGames() {
    const [data, setData] = useState({ live: [], final: [], upcoming: [] })
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [lastRefresh, setLastRefresh] = useState(new Date())

    const fetchGames = async () => {
        try {
            const result = await getLiveGames()
            setData(result)
            setLastRefresh(new Date())
            setError(null)
        } catch (err) {
            setError('Failed to connect to API. Make sure the server is running.')
        } finally {
            setLoading(false)
        }
    }

    useEffect(() => {
        fetchGames()
        const interval = setInterval(fetchGames, 15000)
        return () => clearInterval(interval)
    }, [])

    if (loading) {
        return (
            <div className="loading">
                <div className="spinner"></div>
                <p className="loading-text">Loading games...</p>
            </div>
        )
    }

    if (error) {
        return (
            <div className="empty-state">
                <p className="empty-state-title">Connection Error</p>
                <p className="empty-state-text">{error}</p>
                <button className="btn btn-primary" onClick={fetchGames} style={{ marginTop: 'var(--space-4)' }}>
                    Retry
                </button>
            </div>
        )
    }

    const hasGames = data.live?.length > 0 || data.final?.length > 0 || data.upcoming?.length > 0

    return (
        <div className="animate-fadeIn">
            <div className="page-header">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="page-title">Live Games</h1>
                        <p className="page-description">
                            Last updated: {lastRefresh.toLocaleTimeString()} • Auto-refreshes every 15s
                        </p>
                    </div>
                    <button className="btn btn-secondary" onClick={fetchGames}>
                        <IconRefresh className="nav-icon" />
                        Refresh
                    </button>
                </div>
            </div>

            {!hasGames ? (
                <div className="empty-state">
                    <p className="empty-state-title">No Games Today</p>
                    <p className="empty-state-text">Check back later for scheduled NBA games.</p>
                </div>
            ) : (
                <>
                    {/* Live Games */}
                    {data.live?.length > 0 && (
                        <section className="section">
                            <div className="section-header">
                                <span className="section-title flex items-center gap-2">
                                    <span className="live-dot"></span>
                                    In Progress
                                </span>
                            </div>
                            <div className="stagger">
                                {data.live.map((game) => (
                                    <GameCard key={game.game_id} game={game} isLive={true} />
                                ))}
                            </div>
                        </section>
                    )}

                    {/* Final Games */}
                    {data.final?.length > 0 && (
                        <section className="section">
                            <div className="section-header">
                                <span className="section-title">Completed</span>
                            </div>
                            <div className="stagger">
                                {data.final.map((game) => (
                                    <GameCard key={game.game_id} game={game} isFinal={true} />
                                ))}
                            </div>
                        </section>
                    )}

                    {/* Upcoming Games */}
                    {data.upcoming?.length > 0 && (
                        <section className="section">
                            <div className="section-header">
                                <span className="section-title">Upcoming Today</span>
                            </div>
                            <div className="stagger">
                                {data.upcoming.map((game) => (
                                    <GameCard key={game.game_id} game={game} showLineups={true} />
                                ))}
                            </div>
                        </section>
                    )}
                </>
            )}
        </div>
    )
}

function GameCard({ game, isLive, isFinal, showLineups }) {
    const prediction = game.prediction || {}
    const homeProb = (prediction.home_win_probability || 0.5) * 100
    const awayProb = (prediction.away_win_probability || 0.5) * 100

    const [showRosters, setShowRosters] = useState(false)
    const [homeRoster, setHomeRoster] = useState([])
    const [awayRoster, setAwayRoster] = useState([])
    const [loadingRosters, setLoadingRosters] = useState(false)

    const fetchRosters = async () => {
        if (homeRoster.length > 0 && awayRoster.length > 0) {
            setShowRosters(!showRosters)
            return
        }

        setLoadingRosters(true)
        try {
            const [homeData, awayData] = await Promise.all([
                getTeamRoster(game.home_team),
                getTeamRoster(game.away_team)
            ])
            setHomeRoster(homeData.starters || [])
            setAwayRoster(awayData.starters || [])
            setShowRosters(true)
        } catch (err) {
            console.error('Failed to fetch rosters:', err)
        } finally {
            setLoadingRosters(false)
        }
    }

    return (
        <div className={`game-card animate-slideUp ${isLive ? 'live' : ''}`}>
            {/* Header */}
            <div className="game-header">
                <div className="game-status" style={{ color: isLive ? 'var(--accent-danger)' : isFinal ? 'var(--text-muted)' : 'var(--text-secondary)' }}>
                    {isLive && (
                        <span className="flex items-center gap-2">
                            <span className="live-dot"></span>
                            {game.period && `Q${game.period}`} {game.clock || 'LIVE'}
                        </span>
                    )}
                    {isFinal && 'FINAL'}
                    {!isLive && !isFinal && (game.status_text || 'Scheduled')}
                </div>
                <span className="game-time">{game.game_date}</span>
            </div>

            {/* Matchup */}
            <div className="game-matchup">
                {/* Away Team */}
                <div className="team-block away">
                    <div className="team-info">
                        <TeamLogo abbrev={game.away_team} size="lg" />
                        <div className="team-details">
                            <span className="team-name">{game.away_team}</span>
                            <span className="team-record">{game.away_record || '0-0'}</span>
                        </div>
                    </div>
                    {(isLive || isFinal) ? (
                        <div className="team-score">{game.away_score}</div>
                    ) : (
                        <div className="team-probability">{awayProb.toFixed(0)}%</div>
                    )}
                </div>

                {/* Center - Prediction with Confidence Meter */}
                <div className="game-center">
                    {!isFinal && prediction.predicted_winner && (
                        <div className="prediction-indicator">
                            <div className="prediction-label">Prediction</div>
                            <div className="prediction-pick">{prediction.predicted_winner}</div>
                            <span className={`badge confidence-${prediction.confidence || 'medium'}`} style={{ marginTop: 'var(--space-2)' }}>
                                {(prediction.confidence || 'medium').toUpperCase()}
                            </span>

                            {/* Confidence Meter */}
                            <div className="confidence-meter-container">
                                <ConfidenceMeter winProbability={Math.max(homeProb, awayProb)} />
                            </div>
                        </div>
                    )}
                    {isFinal && (
                        <div className="prediction-indicator">
                            <div className="prediction-label">Predicted</div>
                            <div className="prediction-pick">{prediction.predicted_winner || 'N/A'}</div>
                            <span className={`badge ${game.prediction_correct ? 'badge-success' : 'badge-danger'}`} style={{ marginTop: 'var(--space-2)' }}>
                                {game.prediction_correct ? 'CORRECT' : 'WRONG'}
                            </span>
                        </div>
                    )}
                </div>

                {/* Home Team */}
                <div className="team-block home">
                    <div className="team-info">
                        <TeamLogo abbrev={game.home_team} size="lg" />
                        <div className="team-details">
                            <span className="team-name">{game.home_team}</span>
                            <span className="team-record">{game.home_record || '0-0'}</span>
                        </div>
                    </div>
                    {(isLive || isFinal) ? (
                        <div className="team-score">{game.home_score}</div>
                    ) : (
                        <div className="team-probability">{homeProb.toFixed(0)}%</div>
                    )}
                </div>
            </div>

            {/* Probability Bar */}
            {!isFinal && (
                <div className="probability-bar-container">
                    <div className="probability-bar">
                        <div className="probability-fill-away" style={{ width: `${awayProb}%` }}></div>
                        <div className="probability-fill-home" style={{ width: `${homeProb}%` }}></div>
                    </div>
                    <div className="probability-labels">
                        <span>{game.away_team}: {awayProb.toFixed(1)}%</span>
                        <span>{game.home_team}: {homeProb.toFixed(1)}%</span>
                    </div>
                </div>
            )}

            {/* Starting Lineups Toggle */}
            {!isFinal && (
                <div style={{ marginTop: 'var(--space-4)', borderTop: 'var(--border-subtle)', paddingTop: 'var(--space-4)' }}>
                    <button
                        className="btn btn-ghost btn-block"
                        onClick={fetchRosters}
                        disabled={loadingRosters}
                    >
                        {loadingRosters ? 'Loading...' : showRosters ? 'Hide Starting Lineups' : 'Show Projected Starting 5'}
                    </button>

                    {/* Rosters Display */}
                    {showRosters && (
                        <div style={{ marginTop: 'var(--space-4)', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-6)' }}>
                            {/* Away Team Roster */}
                            <div>
                                <div style={{ fontSize: '0.75rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', marginBottom: 'var(--space-3)' }}>
                                    {game.away_team} Starters
                                </div>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
                                    {awayRoster.length > 0 ? awayRoster.map((player, idx) => (
                                        <div key={idx} style={{
                                            display: 'flex',
                                            justifyContent: 'space-between',
                                            padding: 'var(--space-2) var(--space-3)',
                                            background: 'var(--bg-elevated)',
                                            borderRadius: 'var(--radius-sm)',
                                            fontSize: '0.8125rem'
                                        }}>
                                            <span style={{ fontWeight: '500' }}>{idx + 1}. {player.name}</span>
                                            <span style={{ color: 'var(--accent-primary)', fontFamily: 'var(--font-mono)' }}>{player.pts || 0} PPG</span>
                                        </div>
                                    )) : <span style={{ color: 'var(--text-muted)', fontSize: '0.8125rem' }}>Lineup unavailable</span>}
                                </div>
                            </div>

                            {/* Home Team Roster */}
                            <div>
                                <div style={{ fontSize: '0.75rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', marginBottom: 'var(--space-3)' }}>
                                    {game.home_team} Starters
                                </div>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
                                    {homeRoster.length > 0 ? homeRoster.map((player, idx) => (
                                        <div key={idx} style={{
                                            display: 'flex',
                                            justifyContent: 'space-between',
                                            padding: 'var(--space-2) var(--space-3)',
                                            background: 'var(--bg-elevated)',
                                            borderRadius: 'var(--radius-sm)',
                                            fontSize: '0.8125rem'
                                        }}>
                                            <span style={{ fontWeight: '500' }}>{idx + 1}. {player.name}</span>
                                            <span style={{ color: 'var(--accent-primary)', fontFamily: 'var(--font-mono)' }}>{player.pts || 0} PPG</span>
                                        </div>
                                    )) : <span style={{ color: 'var(--text-muted)', fontSize: '0.8125rem' }}>Lineup unavailable</span>}
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}

export default LiveGames
