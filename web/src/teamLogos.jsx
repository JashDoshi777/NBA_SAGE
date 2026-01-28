/**
 * NBA team logo URLs from official NBA CDN
 */

// Team ID to abbreviation mapping
export const TEAM_IDS = {
    1610612737: 'ATL', 1610612738: 'BOS', 1610612751: 'BKN', 1610612766: 'CHA',
    1610612741: 'CHI', 1610612739: 'CLE', 1610612742: 'DAL', 1610612743: 'DEN',
    1610612765: 'DET', 1610612744: 'GSW', 1610612745: 'HOU', 1610612754: 'IND',
    1610612746: 'LAC', 1610612747: 'LAL', 1610612763: 'MEM', 1610612748: 'MIA',
    1610612749: 'MIL', 1610612750: 'MIN', 1610612740: 'NOP', 1610612752: 'NYK',
    1610612760: 'OKC', 1610612753: 'ORL', 1610612755: 'PHI', 1610612756: 'PHX',
    1610612757: 'POR', 1610612758: 'SAC', 1610612759: 'SAS', 1610612761: 'TOR',
    1610612762: 'UTA', 1610612764: 'WAS'
};

// Abbreviation to Team ID mapping
export const ABBREV_TO_ID = Object.fromEntries(
    Object.entries(TEAM_IDS).map(([id, abbrev]) => [abbrev, id])
);

// Full team names
export const TEAM_NAMES = {
    ATL: 'Atlanta Hawks', BOS: 'Boston Celtics', BKN: 'Brooklyn Nets',
    CHA: 'Charlotte Hornets', CHI: 'Chicago Bulls', CLE: 'Cleveland Cavaliers',
    DAL: 'Dallas Mavericks', DEN: 'Denver Nuggets', DET: 'Detroit Pistons',
    GSW: 'Golden State Warriors', HOU: 'Houston Rockets', IND: 'Indiana Pacers',
    LAC: 'LA Clippers', LAL: 'Los Angeles Lakers', MEM: 'Memphis Grizzlies',
    MIA: 'Miami Heat', MIL: 'Milwaukee Bucks', MIN: 'Minnesota Timberwolves',
    NOP: 'New Orleans Pelicans', NYK: 'New York Knicks', OKC: 'Oklahoma City Thunder',
    ORL: 'Orlando Magic', PHI: 'Philadelphia 76ers', PHX: 'Phoenix Suns',
    POR: 'Portland Trail Blazers', SAC: 'Sacramento Kings', SAS: 'San Antonio Spurs',
    TOR: 'Toronto Raptors', UTA: 'Utah Jazz', WAS: 'Washington Wizards'
};

/**
 * Get team logo URL - using NBA CDN with team ID
 * @param {string} abbrev - Team abbreviation (e.g., 'LAL')
 * @returns {string} Logo URL
 */
export function getTeamLogo(abbrev) {
    const teamId = ABBREV_TO_ID[(abbrev || '').toUpperCase()];
    if (!teamId) {
        // Return NBA logo as fallback
        return 'https://cdn.nba.com/logos/leagues/logo-nba.svg';
    }
    // NBA CDN format - using primary logo path
    return `https://cdn.nba.com/logos/nba/${teamId}/primary/L/logo.svg`;
}

/**
 * Get team full name
 * @param {string} abbrev - Team abbreviation
 * @returns {string} Full team name
 */
export function getTeamName(abbrev) {
    return TEAM_NAMES[abbrev] || abbrev;
}

/**
 * Team logo component
 */
export function TeamLogo({ abbrev, size = 'md', className = '' }) {
    const sizes = {
        xs: 'team-logo team-logo-xs',
        sm: 'team-logo team-logo-sm',
        md: 'team-logo',
        lg: 'team-logo team-logo-lg',
        xl: 'team-logo team-logo-xl',
    };

    return (
        <img
            src={getTeamLogo(abbrev)}
            alt={`${abbrev} logo`}
            className={`${sizes[size] || sizes.md} ${className}`}
            onError={(e) => {
                e.target.style.opacity = '0.3';
            }}
        />
    );
}

export default TeamLogo;
