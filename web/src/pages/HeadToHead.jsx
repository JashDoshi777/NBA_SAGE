import { useState, useEffect } from 'react'
import { getTeams, predictGame, getTeamStats } from '../api'
import { TeamLogo, getTeamName } from '../teamLogos'

// Stats Comparison Bar Component
function StatBar({ label, value1, value2, team1, team2, higherIsBetter = true }) {
    const max = Math.max(value1, value2);
    const width1 = max > 0 ? (value1 / max) * 100 : 50;
    const width2 = max > 0 ? (value2 / max) * 100 : 50;

    const team1Better = higherIsBetter ? value1 >= value2 : value1 <= value2;

    return (
        <div className="stat-comparison-row">
            <div className="stat-bar-container">
                <div className="stat-bar left" style={{ width: `${width1}%`, background: team1Better ? 'var(--accent-success)' : 'var(--bg-tertiary)' }} />
                <span className="stat-value-label left" style={{ color: team1Better ? 'var(--accent-success)' : 'var(--text-secondary)' }}>{value1}</span>
            </div>
            <div className="stat-label-center">{label}</div>
            <div className="stat-bar-container right">
                <span className="stat-value-label right" style={{ color: !team1Better ? 'var(--accent-success)' : 'var(--text-secondary)' }}>{value2}</span>
                <div className="stat-bar right" style={{ width: `${width2}%`, background: !team1Better ? 'var(--accent-success)' : 'var(--bg-tertiary)' }} />
            </div>
        </div>
    );
}

// Recent Form Display
function RecentForm({ form }) {
    if (!form || form.length === 0) return null;

    return (
        <div className="recent-form">
            <div className="form-badges">
                {form.map((game, i) => (
                    <span
                        key={i}
                        className={`form-badge ${game.result === 'W' ? 'win' : 'loss'}`}
                        title={`${game.result} vs ${game.opponent} (${game.score})`}
                    >
                        {game.result}
                    </span>
                ))}
            </div>
        </div>
    );
}

function HeadToHead() {
    const [teams, setTeams] = useState([])
    const [team1, setTeam1] = useState('LAL')
    const [team2, setTeam2] = useState('BOS')
    const [comparison, setComparison] = useState(null)
    const [teamStats, setTeamStats] = useState({ team1: null, team2: null })
    const [loading, setLoading] = useState(false)
    const [statsLoading, setStatsLoading] = useState(false)

    useEffect(() => {
        getTeams().then(data => {
            setTeams(data.teams || [])
        }).catch(console.error)
    }, [])

    const handleCompare = async () => {
        if (!team1 || !team2 || team1 === team2) return

        setLoading(true)
        setStatsLoading(true)
        try {
            // Get predictions for both scenarios
            const [homeGame, awayGame] = await Promise.all([
                predictGame(team1, team2), // Team1 at home
                predictGame(team2, team1), // Team2 at home
            ])

            setComparison({
                team1,
                team2,
                homeGame,  // Team1 hosting Team2
                awayGame,  // Team2 hosting Team1
            })

            // Fetch detailed team stats in parallel
            const [stats1, stats2] = await Promise.all([
                getTeamStats(team1),
                getTeamStats(team2)
            ])

            setTeamStats({ team1: stats1, team2: stats2 })
        } catch (err) {
            console.error('Comparison failed:', err)
        } finally {
            setLoading(false)
            setStatsLoading(false)
        }
    }

    const stats1 = teamStats.team1;
    const stats2 = teamStats.team2;

    return (
        <div className="animate-fadeIn">
            <div className="page-header">
                <h1 className="page-title">Head to Head</h1>
                <p className="page-description">Compare two teams across different scenarios</p>
            </div>

            {/* Team Selector */}
            <div className="card" style={{ marginBottom: 'var(--space-6)' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: 'var(--space-6)', alignItems: 'end' }}>
                    {/* Team 1 */}
                    <div className="form-group" style={{ marginBottom: 0 }}>
                        <label className="form-label">Team 1</label>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
                            <TeamLogo abbrev={team1} size="md" />
                            <select
                                className="form-select"
                                value={team1}
                                onChange={(e) => setTeam1(e.target.value)}
                            >
                                {teams.map(team => (
                                    <option key={team.id} value={team.abbrev}>{team.abbrev} - {getTeamName(team.abbrev)}</option>
                                ))}
                            </select>
                        </div>
                    </div>

                    <div style={{ color: 'var(--text-dim)', fontWeight: '600', paddingBottom: 'var(--space-3)' }}>VS</div>

                    {/* Team 2 */}
                    <div className="form-group" style={{ marginBottom: 0 }}>
                        <label className="form-label">Team 2</label>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
                            <TeamLogo abbrev={team2} size="md" />
                            <select
                                className="form-select"
                                value={team2}
                                onChange={(e) => setTeam2(e.target.value)}
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
                    onClick={handleCompare}
                    disabled={loading || team1 === team2}
                    style={{ marginTop: 'var(--space-6)' }}
                >
                    {loading ? 'Analyzing...' : 'Compare Teams'}
                </button>
            </div>

            {/* Comparison Results */}
            {comparison && (
                <div className="animate-slideUp">
                    {/* Team Records Header */}
                    <div className="stats-grid" style={{ marginBottom: 'var(--space-6)' }}>
                        <div className="stat-card" style={{ textAlign: 'center' }}>
                            <TeamLogo abbrev={comparison.team1} size="lg" style={{ margin: '0 auto var(--space-3)' }} />
                            <div style={{ fontSize: '1.125rem', fontWeight: '600', marginBottom: 'var(--space-2)' }}>
                                {getTeamName(comparison.team1)}
                            </div>
                            {stats1?.record && (
                                <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: 'var(--space-2)' }}>
                                    {stats1.record.wins}-{stats1.record.losses}
                                </div>
                            )}
                            <div className="stat-value accent">
                                {comparison.homeGame?.home_elo?.toFixed(0) || 'N/A'}
                            </div>
                            <div className="stat-label">ELO Rating</div>
                            {stats1?.recent_form && (
                                <div style={{ marginTop: 'var(--space-3)' }}>
                                    <div style={{ fontSize: '0.625rem', color: 'var(--text-muted)', marginBottom: 'var(--space-1)' }}>LAST 5</div>
                                    <RecentForm form={stats1.recent_form} />
                                </div>
                            )}
                        </div>

                        <div className="stat-card" style={{ textAlign: 'center', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 'var(--space-2)' }}>ELO DIFFERENCE</div>
                            <div style={{ fontSize: '2rem', fontWeight: '700', color: comparison.homeGame?.elo_diff > 0 ? 'var(--accent-success)' : 'var(--accent-danger)' }}>
                                {comparison.homeGame?.elo_diff > 0 ? '+' : ''}{comparison.homeGame?.elo_diff?.toFixed(0) || 0}
                            </div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 'var(--space-2)' }}>
                                {comparison.homeGame?.elo_diff > 0 ? `${comparison.team1} favored` : `${comparison.team2} favored`}
                            </div>
                        </div>

                        <div className="stat-card" style={{ textAlign: 'center' }}>
                            <TeamLogo abbrev={comparison.team2} size="lg" style={{ margin: '0 auto var(--space-3)' }} />
                            <div style={{ fontSize: '1.125rem', fontWeight: '600', marginBottom: 'var(--space-2)' }}>
                                {getTeamName(comparison.team2)}
                            </div>
                            {stats2?.record && (
                                <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: 'var(--space-2)' }}>
                                    {stats2.record.wins}-{stats2.record.losses}
                                </div>
                            )}
                            <div className="stat-value accent">
                                {comparison.homeGame?.away_elo?.toFixed(0) || 'N/A'}
                            </div>
                            <div className="stat-label">ELO Rating</div>
                            {stats2?.recent_form && (
                                <div style={{ marginTop: 'var(--space-3)' }}>
                                    <div style={{ fontSize: '0.625rem', color: 'var(--text-muted)', marginBottom: 'var(--space-1)' }}>LAST 5</div>
                                    <RecentForm form={stats2.recent_form} />
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Stats Comparison */}
                    {stats1?.stats && stats2?.stats && (
                        <div className="card" style={{ marginBottom: 'var(--space-6)' }}>
                            <h3 style={{ marginBottom: 'var(--space-4)', textAlign: 'center' }}>Season Statistics</h3>
                            <div className="stats-comparison">
                                <StatBar label="PPG" value1={stats1.stats.ppg} value2={stats2.stats.ppg} team1={team1} team2={team2} />
                                <StatBar label="Opp PPG" value1={stats1.stats.opp_ppg} value2={stats2.stats.opp_ppg} team1={team1} team2={team2} higherIsBetter={false} />
                                <StatBar label="RPG" value1={stats1.stats.rpg} value2={stats2.stats.rpg} team1={team1} team2={team2} />
                                <StatBar label="APG" value1={stats1.stats.apg} value2={stats2.stats.apg} team1={team1} team2={team2} />
                                <StatBar label="FG%" value1={stats1.stats.fg_pct} value2={stats2.stats.fg_pct} team1={team1} team2={team2} />
                                <StatBar label="3P%" value1={stats1.stats.fg3_pct} value2={stats2.stats.fg3_pct} team1={team1} team2={team2} />
                            </div>
                        </div>
                    )}

                    {/* Key Players */}
                    {(stats1?.key_players?.length > 0 || stats2?.key_players?.length > 0) && (
                        <div className="card" style={{ marginBottom: 'var(--space-6)' }}>
                            <h3 style={{ marginBottom: 'var(--space-4)', textAlign: 'center' }}>Key Players</h3>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-4)' }}>
                                <div>
                                    <div style={{ textAlign: 'center', marginBottom: 'var(--space-2)' }}>
                                        <TeamLogo abbrev={team1} size="sm" />
                                        <span style={{ marginLeft: 'var(--space-2)', fontWeight: '600' }}>{team1}</span>
                                    </div>
                                    {stats1?.key_players?.map((player, i) => (
                                        <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-2) 0', borderBottom: 'var(--border-subtle)' }}>
                                            <span>{player.name}</span>
                                            <span style={{ color: 'var(--text-muted)' }}>{player.position}</span>
                                        </div>
                                    ))}
                                </div>
                                <div>
                                    <div style={{ textAlign: 'center', marginBottom: 'var(--space-2)' }}>
                                        <TeamLogo abbrev={team2} size="sm" />
                                        <span style={{ marginLeft: 'var(--space-2)', fontWeight: '600' }}>{team2}</span>
                                    </div>
                                    {stats2?.key_players?.map((player, i) => (
                                        <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: 'var(--space-2) 0', borderBottom: 'var(--border-subtle)' }}>
                                            <span>{player.name}</span>
                                            <span style={{ color: 'var(--text-muted)' }}>{player.position}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Scenario Cards */}
                    <h3 style={{ marginBottom: 'var(--space-4)' }}>Win Probabilities by Venue</h3>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-4)' }}>
                        {/* Scenario 1: Team1 at Home */}
                        <div className="card">
                            <div className="card-header">
                                <span className="card-title">{comparison.team1} Home Game</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-4)' }}>
                                <div style={{ textAlign: 'center' }}>
                                    <TeamLogo abbrev={comparison.team1} size="md" />
                                    <div style={{ fontSize: '1.5rem', fontWeight: '700', color: 'var(--accent-primary)', marginTop: 'var(--space-2)' }}>
                                        {(comparison.homeGame?.home_win_probability * 100).toFixed(1)}%
                                    </div>
                                </div>
                                <div style={{ color: 'var(--text-dim)' }}>vs</div>
                                <div style={{ textAlign: 'center' }}>
                                    <TeamLogo abbrev={comparison.team2} size="md" />
                                    <div style={{ fontSize: '1.5rem', fontWeight: '700', color: 'var(--accent-secondary)', marginTop: 'var(--space-2)' }}>
                                        {(comparison.homeGame?.away_win_probability * 100).toFixed(1)}%
                                    </div>
                                </div>
                            </div>
                            <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)', textAlign: 'center' }}>
                                Prediction: <span style={{ color: 'var(--accent-primary)', fontWeight: '600' }}>{comparison.homeGame?.predicted_winner}</span>
                            </div>
                        </div>

                        {/* Scenario 2: Team2 at Home */}
                        <div className="card">
                            <div className="card-header">
                                <span className="card-title">{comparison.team2} Home Game</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-4)' }}>
                                <div style={{ textAlign: 'center' }}>
                                    <TeamLogo abbrev={comparison.team2} size="md" />
                                    <div style={{ fontSize: '1.5rem', fontWeight: '700', color: 'var(--accent-primary)', marginTop: 'var(--space-2)' }}>
                                        {(comparison.awayGame?.home_win_probability * 100).toFixed(1)}%
                                    </div>
                                </div>
                                <div style={{ color: 'var(--text-dim)' }}>vs</div>
                                <div style={{ textAlign: 'center' }}>
                                    <TeamLogo abbrev={comparison.team1} size="md" />
                                    <div style={{ fontSize: '1.5rem', fontWeight: '700', color: 'var(--accent-secondary)', marginTop: 'var(--space-2)' }}>
                                        {(comparison.awayGame?.away_win_probability * 100).toFixed(1)}%
                                    </div>
                                </div>
                            </div>
                            <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)', textAlign: 'center' }}>
                                Prediction: <span style={{ color: 'var(--accent-primary)', fontWeight: '600' }}>{comparison.awayGame?.predicted_winner}</span>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}

export default HeadToHead
