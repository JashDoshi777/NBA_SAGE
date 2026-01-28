"""
NBA ML Prediction System - Live Data Collector
===============================================
Real-time data collection from NBA Live API endpoints.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
import time

from nba_api.live.nba.endpoints import scoreboard, boxscore

from src.config import API_CONFIG, NBA_TEAMS

logger = logging.getLogger(__name__)


class LiveDataCollector:
    """
    Collects live game data from NBA API.
    
    Uses nba_api.live endpoints:
    - scoreboard.ScoreBoard() for today's games with live scores
    - boxscore.BoxScore(game_id) for detailed game box scores
    """
    
    GAME_STATUS_MAP = {
        1: "NOT_STARTED",
        2: "IN_PROGRESS", 
        3: "FINAL"
    }
    
    def __init__(self):
        self._last_scoreboard_fetch = None
        self._cached_scoreboard = None
        self._cache_ttl = 10  # Seconds to cache scoreboard
    
    def get_live_scoreboard(self, force_refresh: bool = False) -> List[Dict]:
        """
        Get today's games with live scores.
        
        Returns list of games with:
        - game_id, game_code
        - home_team, away_team (tricodes)
        - home_score, away_score
        - status (NOT_STARTED, IN_PROGRESS, FINAL)
        - period, clock
        - home_record, away_record
        """
        # Use cache if available and fresh
        if not force_refresh and self._cached_scoreboard:
            if self._last_scoreboard_fetch:
                elapsed = (datetime.now() - self._last_scoreboard_fetch).total_seconds()
                if elapsed < self._cache_ttl:
                    return self._cached_scoreboard
        
        try:
            sb = scoreboard.ScoreBoard()
            games_data = sb.games.get_dict()
            
            games_list = []
            for game in games_data:
                game_status = game.get("gameStatus", 1)
                home_team = game.get("homeTeam", {})
                away_team = game.get("awayTeam", {})
                
                # Parse periods for quarter scores
                home_periods = [p.get("score", 0) for p in home_team.get("periods", [])]
                away_periods = [p.get("score", 0) for p in away_team.get("periods", [])]
                
                games_list.append({
                    "game_id": game.get("gameId", ""),
                    "game_code": game.get("gameCode", ""),
                    "game_date": game.get("gameEt", "")[:10] if game.get("gameEt") else "",
                    "game_time_utc": game.get("gameTimeUTC", ""),
                    "game_time_et": game.get("gameEt", ""),
                    
                    # Teams
                    "home_team": home_team.get("teamTricode", ""),
                    "away_team": away_team.get("teamTricode", ""),
                    "home_team_id": home_team.get("teamId", 0),
                    "away_team_id": away_team.get("teamId", 0),
                    "home_team_name": home_team.get("teamName", ""),
                    "away_team_name": away_team.get("teamName", ""),
                    
                    # Scores
                    "home_score": home_team.get("score", 0),
                    "away_score": away_team.get("score", 0),
                    "home_periods": home_periods,
                    "away_periods": away_periods,
                    
                    # Status
                    "status": self.GAME_STATUS_MAP.get(game_status, "UNKNOWN"),
                    "status_text": game.get("gameStatusText", ""),
                    "period": game.get("period", 0),
                    "clock": game.get("gameClock", ""),
                    
                    # Records
                    "home_wins": home_team.get("wins", 0),
                    "home_losses": home_team.get("losses", 0),
                    "away_wins": away_team.get("wins", 0),
                    "away_losses": away_team.get("losses", 0),
                    "home_record": f"{home_team.get('wins', 0)}-{home_team.get('losses', 0)}",
                    "away_record": f"{away_team.get('wins', 0)}-{away_team.get('losses', 0)}",
                    
                    # Leaders (for in-progress/final games)
                    "home_leader": game.get("gameLeaders", {}).get("homeLeaders", {}),
                    "away_leader": game.get("gameLeaders", {}).get("awayLeaders", {}),
                })
            
            # Update cache
            self._cached_scoreboard = games_list
            self._last_scoreboard_fetch = datetime.now()
            
            logger.info(f"Fetched {len(games_list)} games from Live Scoreboard")
            return games_list
            
        except Exception as e:
            logger.error(f"Failed to fetch live scoreboard: {e}")
            return self._cached_scoreboard or []
    
    def get_game_boxscore(self, game_id: str) -> Optional[Dict]:
        """
        Get detailed box score for a specific game.
        
        Returns:
            Dict with game details, team stats, player stats
        """
        try:
            box = boxscore.BoxScore(game_id)
            game_data = box.game.get_dict()
            
            return {
                "game_id": game_data.get("gameId", game_id),
                "game_status": game_data.get("gameStatus", 1),
                "game_status_text": game_data.get("gameStatusText", ""),
                "period": game_data.get("period", 0),
                "clock": game_data.get("gameClock", ""),
                
                "home_team": game_data.get("homeTeam", {}),
                "away_team": game_data.get("awayTeam", {}),
                
                "arena": game_data.get("arena", {}),
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch boxscore for {game_id}: {e}")
            return None
    
    def get_game_status(self, game_id: str) -> str:
        """
        Get current status for a specific game.
        
        Returns:
            'NOT_STARTED', 'IN_PROGRESS', or 'FINAL'
        """
        games = self.get_live_scoreboard()
        for game in games:
            if game["game_id"] == game_id:
                return game["status"]
        return "UNKNOWN"
    
    def get_winner(self, game_id: str) -> Optional[str]:
        """
        Get the winner of a completed game.
        
        Returns:
            Team tricode of winner, or None if game not finished
        """
        games = self.get_live_scoreboard()
        for game in games:
            if game["game_id"] == game_id:
                if game["status"] == "FINAL":
                    if game["home_score"] > game["away_score"]:
                        return game["home_team"]
                    else:
                        return game["away_team"]
        return None
    
    def get_games_by_status(self, status: str) -> List[Dict]:
        """
        Filter games by status.
        
        Args:
            status: 'NOT_STARTED', 'IN_PROGRESS', or 'FINAL'
        """
        games = self.get_live_scoreboard()
        return [g for g in games if g["status"] == status]
    
    def get_live_games(self) -> List[Dict]:
        """Get all currently in-progress games."""
        return self.get_games_by_status("IN_PROGRESS")
    
    def get_final_games(self) -> List[Dict]:
        """Get all completed games from today."""
        return self.get_games_by_status("FINAL")
    
    def get_upcoming_games(self) -> List[Dict]:
        """Get all not-yet-started games from today."""
        return self.get_games_by_status("NOT_STARTED")
    
    def format_game_summary(self, game: Dict) -> str:
        """Format a game into a readable summary string."""
        status = game["status"]
        away = game["away_team"]
        home = game["home_team"]
        
        if status == "NOT_STARTED":
            return f"{away} @ {home} - {game['status_text']}"
        elif status == "IN_PROGRESS":
            return f"{away} {game['away_score']} @ {home} {game['home_score']} - {game['status_text']}"
        else:  # FINAL
            return f"{away} {game['away_score']} @ {home} {game['home_score']} - FINAL"


# =============================================================================
# CLI INTERFACE
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    collector = LiveDataCollector()
    
    print("\n=== Today's NBA Games ===\n")
    games = collector.get_live_scoreboard()
    
    if not games:
        print("No games scheduled for today")
    else:
        for game in games:
            print(collector.format_game_summary(game))
            if game["status"] == "IN_PROGRESS":
                print(f"  Q{game['period']} {game['clock']}")
            print()
