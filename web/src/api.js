/**
 * API utilities for NBA ML Predictor
 */

// Use relative URL in production, localhost in development
const API_BASE = import.meta.env.DEV ? 'http://localhost:8000/api' : '/api';


/**
 * Generic fetch wrapper with error handling
 */
async function fetchAPI(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

/**
 * Get live games with predictions
 */
export async function getLiveGames() {
    return fetchAPI('/games/live');
}

/**
 * Get upcoming games
 */
export async function getUpcomingGames(days = 7) {
    return fetchAPI(`/games/upcoming?days=${days}`);
}

/**
 * Get prediction for a single game
 */
export async function predictGame(home, away) {
    return fetchAPI(`/predict?home=${home}&away=${away}`);
}

/**
 * Get model accuracy stats
 */
export async function getAccuracy() {
    return fetchAPI('/accuracy');
}

/**
 * Get MVP race standings
 */
export async function getMvpRace() {
    return fetchAPI('/mvp');
}

/**
 * Get championship odds
 */
export async function getChampionshipOdds() {
    return fetchAPI('/championship');
}

/**
 * Get list of all teams
 */
export async function getTeams() {
    return fetchAPI('/teams');
}

/**
 * Health check
 */
export async function healthCheck() {
    return fetchAPI('/health');
}

/**
 * Get detailed team statistics for head-to-head comparison
 */
export async function getTeamStats(team) {
    return fetchAPI(`/team-stats?team=${team}`);
}
