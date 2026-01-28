import { useState } from 'react'
import { IconSearch } from '../icons'

// Simulated player data - in production this would come from API
const MOCK_PLAYERS = [
    { id: 1, name: 'LeBron James', team: 'LAL', ppg: 25.4, rpg: 7.2, apg: 8.1, fg_pct: 54.2, position: 'SF' },
    { id: 2, name: 'Stephen Curry', team: 'GSW', ppg: 26.8, rpg: 4.5, apg: 5.2, fg_pct: 45.1, position: 'PG' },
    { id: 3, name: 'Giannis Antetokounmpo', team: 'MIL', ppg: 31.2, rpg: 11.5, apg: 5.8, fg_pct: 61.2, position: 'PF' },
    { id: 4, name: 'Nikola Jokic', team: 'DEN', ppg: 26.5, rpg: 12.2, apg: 9.1, fg_pct: 58.3, position: 'C' },
    { id: 5, name: 'Jayson Tatum', team: 'BOS', ppg: 27.0, rpg: 8.1, apg: 4.6, fg_pct: 47.1, position: 'SF' },
    { id: 6, name: 'Luka Doncic', team: 'DAL', ppg: 33.2, rpg: 9.1, apg: 9.5, fg_pct: 48.7, position: 'PG' },
    { id: 7, name: 'Kevin Durant', team: 'PHX', ppg: 27.5, rpg: 6.5, apg: 5.2, fg_pct: 52.4, position: 'SF' },
    { id: 8, name: 'Joel Embiid', team: 'PHI', ppg: 34.1, rpg: 11.0, apg: 5.7, fg_pct: 54.8, position: 'C' },
    { id: 9, name: 'Shai Gilgeous-Alexander', team: 'OKC', ppg: 31.5, rpg: 5.5, apg: 6.2, fg_pct: 53.5, position: 'SG' },
    { id: 10, name: 'Anthony Edwards', team: 'MIN', ppg: 26.0, rpg: 5.8, apg: 5.0, fg_pct: 46.2, position: 'SG' },
];

function PlayerStats() {
    const [searchQuery, setSearchQuery] = useState('')
    const [selectedPlayer, setSelectedPlayer] = useState(null)

    const filteredPlayers = MOCK_PLAYERS.filter(player =>
        player.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        player.team.toLowerCase().includes(searchQuery.toLowerCase())
    )

    return (
        <div className="animate-fadeIn">
            <div className="page-header">
                <h1 className="page-title">Player Stats</h1>
                <p className="page-description">Search and compare NBA player statistics</p>
            </div>

            {/* Search */}
            <div className="card" style={{ marginBottom: 'var(--space-6)' }}>
                <div style={{ position: 'relative' }}>
                    <IconSearch className="nav-icon" style={{
                        position: 'absolute',
                        left: 'var(--space-4)',
                        top: '50%',
                        transform: 'translateY(-50%)',
                        color: 'var(--text-muted)'
                    }} />
                    <input
                        type="text"
                        className="form-input"
                        placeholder="Search players or teams..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        style={{ paddingLeft: 'var(--space-10)' }}
                    />
                </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: selectedPlayer ? '1fr 1fr' : '1fr', gap: 'var(--space-6)' }}>
                {/* Player List */}
                <div className="table-container">
                    <table className="data-table">
                        <thead>
                            <tr>
                                <th>Player</th>
                                <th>Team</th>
                                <th>Pos</th>
                                <th style={{ textAlign: 'right' }}>PPG</th>
                                <th style={{ textAlign: 'right' }}>RPG</th>
                                <th style={{ textAlign: 'right' }}>APG</th>
                            </tr>
                        </thead>
                        <tbody>
                            {filteredPlayers.map((player) => (
                                <tr
                                    key={player.id}
                                    onClick={() => setSelectedPlayer(player)}
                                    style={{ cursor: 'pointer', background: selectedPlayer?.id === player.id ? 'var(--bg-elevated)' : undefined }}
                                >
                                    <td style={{ fontWeight: '500' }}>{player.name}</td>
                                    <td style={{ color: 'var(--text-muted)' }}>{player.team}</td>
                                    <td style={{ color: 'var(--text-muted)' }}>{player.position}</td>
                                    <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--accent-primary)' }}>{player.ppg}</td>
                                    <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>{player.rpg}</td>
                                    <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>{player.apg}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>

                {/* Player Detail Card */}
                {selectedPlayer && (
                    <div className="card animate-slideUp">
                        <div style={{ textAlign: 'center', marginBottom: 'var(--space-6)' }}>
                            <div style={{
                                width: '80px',
                                height: '80px',
                                borderRadius: '50%',
                                background: 'var(--bg-elevated)',
                                margin: '0 auto var(--space-4)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                fontSize: '2rem',
                                fontWeight: '700',
                                color: 'var(--accent-primary)'
                            }}>
                                {selectedPlayer.name.split(' ').map(n => n[0]).join('')}
                            </div>
                            <h2 style={{ marginBottom: 'var(--space-2)' }}>{selectedPlayer.name}</h2>
                            <p style={{ color: 'var(--text-muted)' }}>
                                {selectedPlayer.team} • {selectedPlayer.position}
                            </p>
                        </div>

                        <div className="stats-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
                            <div className="stat-card">
                                <div className="stat-value accent">{selectedPlayer.ppg}</div>
                                <div className="stat-label">Points Per Game</div>
                            </div>
                            <div className="stat-card">
                                <div className="stat-value">{selectedPlayer.rpg}</div>
                                <div className="stat-label">Rebounds Per Game</div>
                            </div>
                            <div className="stat-card">
                                <div className="stat-value">{selectedPlayer.apg}</div>
                                <div className="stat-label">Assists Per Game</div>
                            </div>
                            <div className="stat-card">
                                <div className="stat-value">{selectedPlayer.fg_pct}%</div>
                                <div className="stat-label">Field Goal %</div>
                            </div>
                        </div>

                        <div style={{ marginTop: 'var(--space-4)', fontSize: '0.75rem', color: 'var(--text-muted)', textAlign: 'center' }}>
                            Note: Player stats are currently mock data. Connect to NBA API for live stats.
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}

export default PlayerStats
