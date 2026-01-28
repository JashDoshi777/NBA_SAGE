import { useState, useEffect } from 'react'
import {
    LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
    XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
    AreaChart, Area, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
    ScatterChart, Scatter, ZAxis
} from 'recharts'

// API Base URL
const API_BASE = import.meta.env.DEV ? 'http://localhost:8000' : '';

function Analytics() {
    const [data, setData] = useState(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        fetchAnalytics()
    }, [])

    const fetchAnalytics = async () => {
        try {
            const response = await fetch(`${API_BASE}/api/analytics`)
            const result = await response.json()
            setData(result)
        } catch (err) {
            console.error('Failed to fetch analytics:', err)
        } finally {
            setLoading(false)
        }
    }

    if (loading) {
        return (
            <div className="loading">
                <div className="spinner"></div>
                <p className="loading-text">Loading analytics...</p>
            </div>
        )
    }

    // Chart colors
    const COLORS = ['#4ade80', '#facc15', '#f87171', '#60a5fa', '#a78bfa']
    const GRADIENT_COLORS = {
        primary: '#4ade80',
        secondary: '#60a5fa',
        accent: '#a78bfa'
    }

    // Data with fallbacks
    const accuracyTrend = data?.accuracy_trend || []
    const teamAccuracy = data?.team_accuracy || []
    const confidenceDistribution = data?.confidence_distribution || []
    const calibrationData = data?.calibration || []
    const overallStats = data?.overall || { total_predictions: 0, correct: 0, accuracy: 0, avg_confidence: 0 }

    // New advanced data
    const eloScatter = data?.elo_scatter || []
    const radarData = data?.radar_data || []
    const homeAwaySplit = data?.home_away_split || { home: { accuracy: 0 }, away: { accuracy: 0 } }
    const streakData = data?.streak_data || []
    const currentStreak = data?.current_streak || { count: 0, type: 'W' }
    const topMatchups = data?.top_matchups || []

    // Custom tooltip styles
    const tooltipStyle = {
        background: 'rgba(0,0,0,0.9)',
        border: '1px solid rgba(255,255,255,0.2)',
        borderRadius: '8px',
        padding: '8px 12px'
    }

    return (
        <div className="animate-fadeIn">
            <div className="page-header">
                <h1 className="page-title">Analytics Dashboard</h1>
                <p className="page-description">
                    Advanced model performance and NBA statistics
                </p>
            </div>

            {/* Stats Overview */}
            <div className="stats-grid" style={{ marginBottom: 'var(--space-6)' }}>
                <div className="stat-card">
                    <div className="stat-value accent">{overallStats.total_predictions}</div>
                    <div className="stat-label">Total Predictions</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value" style={{ color: 'var(--accent-success)' }}>
                        {overallStats.correct}
                    </div>
                    <div className="stat-label">Correct Predictions</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value accent">{overallStats.accuracy}%</div>
                    <div className="stat-label">Overall Accuracy</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value" style={{
                        color: currentStreak.type === 'W' ? '#4ade80' : '#f87171'
                    }}>
                        {currentStreak.count}{currentStreak.type}
                    </div>
                    <div className="stat-label">Current Streak</div>
                </div>
            </div>

            {/* Prediction Streak Timeline */}
            <div className="card" style={{ marginBottom: 'var(--space-6)' }}>
                <div className="card-header">
                    <span className="card-title">Prediction Streak Timeline</span>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        Last {streakData.length} predictions
                    </span>
                </div>
                <div style={{
                    display: 'flex',
                    gap: '4px',
                    padding: 'var(--space-4)',
                    flexWrap: 'wrap',
                    justifyContent: 'center'
                }}>
                    {streakData.map((item, idx) => (
                        <div
                            key={idx}
                            title={`${item.matchup} - ${item.date}`}
                            style={{
                                width: '32px',
                                height: '32px',
                                borderRadius: '6px',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                fontWeight: '700',
                                fontSize: '0.75rem',
                                background: item.result === 'W'
                                    ? 'linear-gradient(135deg, #22c55e, #16a34a)'
                                    : 'linear-gradient(135deg, #ef4444, #dc2626)',
                                color: 'white',
                                cursor: 'pointer',
                                transition: 'transform 0.2s',
                            }}
                            onMouseEnter={(e) => e.target.style.transform = 'scale(1.2)'}
                            onMouseLeave={(e) => e.target.style.transform = 'scale(1)'}
                        >
                            {item.result}
                        </div>
                    ))}
                </div>
            </div>

            {/* Charts Grid - Row 1 */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-6)', marginBottom: 'var(--space-6)' }}>

                {/* Radar Chart - Model Dimensions */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Model Performance Radar</span>
                    </div>
                    <div style={{ height: '300px' }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <RadarChart cx="50%" cy="50%" outerRadius="70%" data={radarData}>
                                <PolarGrid stroke="rgba(255,255,255,0.1)" />
                                <PolarAngleAxis
                                    dataKey="dimension"
                                    tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 11 }}
                                />
                                <PolarRadiusAxis
                                    angle={90}
                                    domain={[0, 100]}
                                    tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 10 }}
                                />
                                <Radar
                                    name="Accuracy %"
                                    dataKey="value"
                                    stroke="#4ade80"
                                    fill="#4ade80"
                                    fillOpacity={0.3}
                                    strokeWidth={2}
                                />
                                <Tooltip contentStyle={tooltipStyle} />
                            </RadarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* ELO vs Win% Scatter */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">ELO Rating vs Win %</span>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                            Bubble size = Games played
                        </span>
                    </div>
                    <div style={{ height: '300px' }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                <XAxis
                                    type="number"
                                    dataKey="elo"
                                    name="ELO"
                                    domain={['dataMin - 50', 'dataMax + 50']}
                                    stroke="rgba(255,255,255,0.5)"
                                    fontSize={11}
                                    label={{ value: 'ELO Rating', position: 'bottom', fill: 'rgba(255,255,255,0.5)', fontSize: 11 }}
                                />
                                <YAxis
                                    type="number"
                                    dataKey="winPct"
                                    name="Win %"
                                    domain={[20, 80]}
                                    stroke="rgba(255,255,255,0.5)"
                                    fontSize={11}
                                />
                                <ZAxis
                                    type="number"
                                    dataKey="gamesPlayed"
                                    range={[50, 400]}
                                    name="Games"
                                />
                                <Tooltip
                                    cursor={{ strokeDasharray: '3 3' }}
                                    contentStyle={tooltipStyle}
                                    formatter={(value, name) => [value, name]}
                                    labelFormatter={(label) => `Team: ${eloScatter.find(t => t.elo === label)?.team || ''}`}
                                />
                                <Scatter
                                    name="Teams"
                                    data={eloScatter}
                                    fill="#60a5fa"
                                >
                                    {eloScatter.map((entry, index) => (
                                        <Cell
                                            key={`cell-${index}`}
                                            fill={entry.conference === 'East' ? '#60a5fa' : '#f97316'}
                                        />
                                    ))}
                                </Scatter>
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                    <div style={{
                        display: 'flex',
                        justifyContent: 'center',
                        gap: 'var(--space-4)',
                        padding: 'var(--space-2)',
                        fontSize: '0.75rem'
                    }}>
                        <span><span style={{ color: '#60a5fa' }}>●</span> East</span>
                        <span><span style={{ color: '#f97316' }}>●</span> West</span>
                    </div>
                </div>
            </div>

            {/* Charts Grid - Row 2 */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-6)', marginBottom: 'var(--space-6)' }}>

                {/* Accuracy Trend */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Accuracy Trend (7 Days)</span>
                    </div>
                    <div style={{ height: '280px' }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={accuracyTrend}>
                                <defs>
                                    <linearGradient id="colorAccuracy" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor={GRADIENT_COLORS.primary} stopOpacity={0.3} />
                                        <stop offset="95%" stopColor={GRADIENT_COLORS.primary} stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                <XAxis dataKey="date" stroke="rgba(255,255,255,0.5)" fontSize={11} />
                                <YAxis domain={[0, 100]} stroke="rgba(255,255,255,0.5)" fontSize={11} />
                                <Tooltip contentStyle={tooltipStyle} />
                                <Area
                                    type="monotone"
                                    dataKey="accuracy"
                                    stroke={GRADIENT_COLORS.primary}
                                    strokeWidth={2}
                                    fill="url(#colorAccuracy)"
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Home vs Away Performance */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Home vs Away Accuracy</span>
                    </div>
                    <div style={{ height: '280px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <div style={{ display: 'flex', gap: 'var(--space-8)', alignItems: 'flex-end' }}>
                            {/* Home Bar */}
                            <div style={{ textAlign: 'center' }}>
                                <div style={{
                                    width: '80px',
                                    height: `${homeAwaySplit.home?.accuracy * 2.5}px`,
                                    background: 'linear-gradient(180deg, #4ade80, #22c55e)',
                                    borderRadius: '8px 8px 0 0',
                                    display: 'flex',
                                    alignItems: 'flex-start',
                                    justifyContent: 'center',
                                    paddingTop: 'var(--space-2)',
                                    minHeight: '40px'
                                }}>
                                    <span style={{ fontWeight: '700', fontSize: '1.25rem' }}>
                                        {homeAwaySplit.home?.accuracy}%
                                    </span>
                                </div>
                                <div style={{
                                    marginTop: 'var(--space-2)',
                                    fontWeight: '600',
                                    color: 'var(--text-secondary)'
                                }}>
                                    🏠 Home
                                </div>
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                    {homeAwaySplit.home?.correct}/{homeAwaySplit.home?.total}
                                </div>
                            </div>

                            {/* Away Bar */}
                            <div style={{ textAlign: 'center' }}>
                                <div style={{
                                    width: '80px',
                                    height: `${homeAwaySplit.away?.accuracy * 2.5}px`,
                                    background: 'linear-gradient(180deg, #60a5fa, #3b82f6)',
                                    borderRadius: '8px 8px 0 0',
                                    display: 'flex',
                                    alignItems: 'flex-start',
                                    justifyContent: 'center',
                                    paddingTop: 'var(--space-2)',
                                    minHeight: '40px'
                                }}>
                                    <span style={{ fontWeight: '700', fontSize: '1.25rem' }}>
                                        {homeAwaySplit.away?.accuracy}%
                                    </span>
                                </div>
                                <div style={{
                                    marginTop: 'var(--space-2)',
                                    fontWeight: '600',
                                    color: 'var(--text-secondary)'
                                }}>
                                    ✈️ Away
                                </div>
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                    {homeAwaySplit.away?.correct}/{homeAwaySplit.away?.total}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Mini Heatmap - Top Matchups */}
            {topMatchups.length > 0 && (
                <div className="card" style={{ marginBottom: 'var(--space-6)' }}>
                    <div className="card-header">
                        <span className="card-title">Top Team Matchup Predictions</span>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                            Home team win probability
                        </span>
                    </div>
                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(4, 1fr)',
                        gap: '8px',
                        padding: 'var(--space-4)'
                    }}>
                        {topMatchups.map((matchup, idx) => {
                            const prob = matchup.homeWinProb
                            const hue = prob > 50 ? 120 : 0  // Green for >50%, Red for <50%
                            const saturation = Math.abs(prob - 50) * 2  // More intense as further from 50
                            const lightness = 35
                            return (
                                <div
                                    key={idx}
                                    style={{
                                        background: `hsl(${hue}, ${saturation}%, ${lightness}%)`,
                                        borderRadius: '8px',
                                        padding: 'var(--space-3)',
                                        textAlign: 'center',
                                        transition: 'transform 0.2s',
                                        cursor: 'pointer'
                                    }}
                                    onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.05)'}
                                    onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
                                >
                                    <div style={{ fontSize: '0.75rem', marginBottom: '4px', opacity: 0.8 }}>
                                        {matchup.away} @ {matchup.home}
                                    </div>
                                    <div style={{ fontSize: '1.25rem', fontWeight: '700' }}>
                                        {prob}%
                                    </div>
                                </div>
                            )
                        })}
                    </div>
                </div>
            )}

            {/* Charts Grid - Row 3 */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-6)', marginBottom: 'var(--space-6)' }}>

                {/* Confidence Distribution */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Confidence Distribution</span>
                    </div>
                    <div style={{ height: '280px' }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={confidenceDistribution}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={50}
                                    outerRadius={90}
                                    paddingAngle={5}
                                    dataKey="value"
                                    label={({ name, percent }) => `${(percent * 100).toFixed(0)}%`}
                                    labelLine={false}
                                >
                                    {confidenceDistribution.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.color} />
                                    ))}
                                </Pie>
                                <Tooltip contentStyle={tooltipStyle} />
                                <Legend
                                    verticalAlign="bottom"
                                    height={36}
                                    formatter={(value) => <span style={{ color: 'rgba(255,255,255,0.8)', fontSize: '0.75rem' }}>{value}</span>}
                                />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Calibration Chart */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Prediction Calibration</span>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                            Predicted vs Actual Win %
                        </span>
                    </div>
                    <div style={{ height: '280px' }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={calibrationData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                <XAxis
                                    dataKey="predicted"
                                    stroke="rgba(255,255,255,0.5)"
                                    fontSize={11}
                                />
                                <YAxis
                                    stroke="rgba(255,255,255,0.5)"
                                    fontSize={11}
                                    domain={[50, 90]}
                                />
                                <Tooltip contentStyle={tooltipStyle} />
                                <Legend />
                                <Line
                                    type="monotone"
                                    dataKey="predicted"
                                    stroke="rgba(255,255,255,0.3)"
                                    strokeDasharray="5 5"
                                    name="Perfect"
                                    dot={false}
                                />
                                <Line
                                    type="monotone"
                                    dataKey="actual"
                                    stroke={GRADIENT_COLORS.accent}
                                    strokeWidth={2}
                                    name="Actual"
                                    dot={{ fill: GRADIENT_COLORS.accent, r: 4 }}
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            {/* Team Accuracy Bar */}
            <div className="card" style={{ marginBottom: 'var(--space-6)' }}>
                <div className="card-header">
                    <span className="card-title">Accuracy by Team</span>
                </div>
                <div style={{ height: '300px' }}>
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={teamAccuracy} layout="vertical">
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                            <XAxis type="number" domain={[0, 100]} stroke="rgba(255,255,255,0.5)" fontSize={11} />
                            <YAxis dataKey="team" type="category" stroke="rgba(255,255,255,0.5)" fontSize={11} width={40} />
                            <Tooltip
                                contentStyle={tooltipStyle}
                                formatter={(value) => [`${value}%`, 'Accuracy']}
                            />
                            <Bar dataKey="accuracy" fill={GRADIENT_COLORS.secondary} radius={[0, 4, 4, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Recent Predictions Table */}
            <div className="card">
                <div className="card-header">
                    <span className="card-title">Recent Predictions</span>
                </div>
                <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                                <th style={tableHeaderStyle}>Date</th>
                                <th style={tableHeaderStyle}>Matchup</th>
                                <th style={tableHeaderStyle}>Prediction</th>
                                <th style={tableHeaderStyle}>Confidence</th>
                                <th style={tableHeaderStyle}>Result</th>
                            </tr>
                        </thead>
                        <tbody>
                            {(data?.recent_predictions || []).map((pred, idx) => (
                                <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                                    <td style={tableCellStyle}>{pred.date}</td>
                                    <td style={tableCellStyle}>{pred.matchup}</td>
                                    <td style={tableCellStyle}><strong>{pred.prediction}</strong></td>
                                    <td style={tableCellStyle}>
                                        <span style={{
                                            color: pred.confidence > 70 ? '#4ade80' : pred.confidence > 60 ? '#facc15' : '#f87171'
                                        }}>
                                            {pred.confidence}%
                                        </span>
                                    </td>
                                    <td style={tableCellStyle}>
                                        <span className={`badge ${pred.correct ? 'badge-success' : 'badge-danger'}`}>
                                            {pred.correct ? '✓' : '✗'}
                                        </span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    )
}

const tableHeaderStyle = {
    textAlign: 'left',
    padding: 'var(--space-3) var(--space-4)',
    fontSize: '0.75rem',
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    color: 'var(--text-muted)'
}

const tableCellStyle = {
    padding: 'var(--space-3) var(--space-4)',
    fontSize: '0.875rem'
}

export default Analytics
