"""
NBA ML Prediction System - Comprehensive Data Collector
========================================================
Full data collection from NBA API with all available endpoints:
- Games, Team Stats, Player Stats (basic)
- Advanced Metrics, Clutch Stats, Hustle Stats
- Box Scores, Standings, Play Types
"""

import time
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from tqdm import tqdm
import logging

from nba_api.stats.endpoints import (
    # Basic endpoints
    leaguegamefinder,
    leaguestandings,
    leaguedashteamstats,
    leaguedashplayerstats,
    playergamelog,
    boxscoretraditionalv2,
    boxscoreadvancedv2,
    leagueleaders,
    # Advanced endpoints
    teamestimatedmetrics,
    playerestimatedmetrics,
    leaguedashteamclutch,
    leaguedashplayerclutch,
    leaguehustlestatsteam,
    leaguehustlestatsplayer,
    leaguedashptteamdefend,
    leaguedashptstats,
    leaguestandingsv3,
    teamyearbyyearstats,
    # Box score variants
    boxscoremiscv2,
    boxscorescoringv2,
    boxscoreusagev2,
    # Shooting
    leaguedashteamptshot,
    leaguedashplayerptshot,
)
from nba_api.stats.static import teams, players

from src.config import (
    API_CONFIG, 
    SEASON_STRINGS, 
    API_CACHE_DIR, 
    RAW_DATA_DIR,
    NBA_TEAMS
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# RETRY DECORATOR WITH EXPONENTIAL BACKOFF
# =============================================================================
def retry_with_backoff(func):
    """Decorator to retry API calls with exponential backoff."""
    def wrapper(*args, **kwargs):
        backoff = API_CONFIG.initial_backoff
        last_exception = None
        
        for attempt in range(API_CONFIG.max_retries + 1):
            try:
                time.sleep(API_CONFIG.base_delay)
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < API_CONFIG.max_retries:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff = min(backoff * API_CONFIG.backoff_multiplier, API_CONFIG.max_backoff)
                else:
                    logger.error(f"All {API_CONFIG.max_retries + 1} attempts failed for {func.__name__}")
        
        raise last_exception
    return wrapper


# =============================================================================
# CACHE MANAGER
# =============================================================================
class CacheManager:
    """Manages caching of API responses with per-endpoint, per-season storage."""
    
    def __init__(self, cache_dir: Path = API_CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = cache_dir / "checkpoint.json"
    
    def get_cache_path(self, endpoint: str, season: str, entity_id: Optional[str] = None) -> Path:
        if entity_id:
            return self.cache_dir / f"{endpoint}_{season}_{entity_id}.parquet"
        return self.cache_dir / f"{endpoint}_{season}.parquet"
    
    def is_cached(self, endpoint: str, season: str, entity_id: Optional[str] = None) -> bool:
        return self.get_cache_path(endpoint, season, entity_id).exists()
    
    def load_cached(self, endpoint: str, season: str, entity_id: Optional[str] = None) -> Optional[pd.DataFrame]:
        cache_path = self.get_cache_path(endpoint, season, entity_id)
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        return None
    
    def save_to_cache(self, df: pd.DataFrame, endpoint: str, season: str, entity_id: Optional[str] = None):
        cache_path = self.get_cache_path(endpoint, season, entity_id)
        df.to_parquet(cache_path, index=False)
        logger.info(f"Cached {len(df)} rows to {cache_path.name}")
    
    def load_checkpoint(self) -> Dict[str, Any]:
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {"completed_seasons": [], "last_endpoint": None, "last_season": None}
    
    def save_checkpoint(self, checkpoint: Dict[str, Any]):
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def mark_season_complete(self, endpoint: str, season: str):
        checkpoint = self.load_checkpoint()
        key = f"{endpoint}_{season}"
        if key not in checkpoint["completed_seasons"]:
            checkpoint["completed_seasons"].append(key)
        checkpoint["last_endpoint"] = endpoint
        checkpoint["last_season"] = season
        self.save_checkpoint(checkpoint)
    
    def is_season_complete(self, endpoint: str, season: str) -> bool:
        checkpoint = self.load_checkpoint()
        return f"{endpoint}_{season}" in checkpoint["completed_seasons"]


# =============================================================================
# GAME DATA COLLECTOR
# =============================================================================
class GameDataCollector:
    """Collects game-level data."""
    
    def __init__(self):
        self.cache = CacheManager()
    
    @retry_with_backoff
    def _fetch_season_games(self, season: str) -> pd.DataFrame:
        games = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            league_id_nullable="00"
        )
        return games.get_data_frames()[0]
    
    def get_season_games(self, season: str, force_refresh: bool = False) -> pd.DataFrame:
        if not force_refresh and self.cache.is_cached("games", season):
            logger.info(f"Loading cached games for {season}")
            return self.cache.load_cached("games", season)
        
        logger.info(f"Fetching games for {season} from API...")
        df = self._fetch_season_games(season)
        self.cache.save_to_cache(df, "games", season)
        self.cache.mark_season_complete("games", season)
        return df
    
    def collect_all_seasons(self, seasons: List[str] = None, force_refresh: bool = False) -> pd.DataFrame:
        if seasons is None:
            seasons = SEASON_STRINGS
        
        all_games = []
        for season in tqdm(seasons, desc="Collecting games"):
            if not force_refresh and self.cache.is_season_complete("games", season):
                df = self.cache.load_cached("games", season)
            else:
                df = self.get_season_games(season, force_refresh)
            all_games.append(df)
        
        combined = pd.concat(all_games, ignore_index=True)
        combined.to_parquet(RAW_DATA_DIR / "all_games.parquet", index=False)
        logger.info(f"Saved {len(combined)} total games to all_games.parquet")
        return combined


# =============================================================================
# TEAM DATA COLLECTOR (ENHANCED)
# =============================================================================
class TeamDataCollector:
    """Collects comprehensive team statistics."""
    
    def __init__(self):
        self.cache = CacheManager()
    
    @retry_with_backoff
    def _fetch_team_stats(self, season: str) -> pd.DataFrame:
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            per_mode_detailed="PerGame"
        )
        return stats.get_data_frames()[0]
    
    @retry_with_backoff
    def _fetch_team_advanced(self, season: str) -> pd.DataFrame:
        """Fetch advanced team metrics: NET_RTG, PACE, PIE, TS%, eFG%"""
        try:
            stats = teamestimatedmetrics.TeamEstimatedMetrics(season=season)
            return stats.get_data_frames()[0]
        except Exception as e:
            logger.warning(f"TeamEstimatedMetrics failed for {season}: {e}")
            return pd.DataFrame()
    
    @retry_with_backoff
    def _fetch_team_clutch(self, season: str) -> pd.DataFrame:
        """Fetch clutch stats: performance in close games"""
        try:
            stats = leaguedashteamclutch.LeagueDashTeamClutch(
                season=season,
                clutch_time="Last 5 Minutes",
                point_diff=5
            )
            return stats.get_data_frames()[0]
        except Exception as e:
            logger.warning(f"Team clutch stats failed for {season}: {e}")
            return pd.DataFrame()
    
    @retry_with_backoff
    def _fetch_team_hustle(self, season: str) -> pd.DataFrame:
        """Fetch hustle stats: deflections, loose balls, charges"""
        try:
            stats = leaguehustlestatsteam.LeagueHustleStatsTeam(season=season)
            return stats.get_data_frames()[0]
        except Exception as e:
            logger.warning(f"Team hustle stats failed for {season}: {e}")
            return pd.DataFrame()
    
    @retry_with_backoff
    def _fetch_team_defense(self, season: str) -> pd.DataFrame:
        """Fetch defensive stats"""
        try:
            stats = leaguedashptteamdefend.LeagueDashPtTeamDefend(
                season=season,
                defense_category="Overall"
            )
            return stats.get_data_frames()[0]
        except Exception as e:
            logger.warning(f"Team defense stats failed for {season}: {e}")
            return pd.DataFrame()
    
    @retry_with_backoff
    def _fetch_team_shooting(self, season: str) -> pd.DataFrame:
        """Fetch team shooting stats"""
        try:
            stats = leaguedashteamptshot.LeagueDashTeamPtShot(season=season)
            return stats.get_data_frames()[0]
        except Exception as e:
            logger.warning(f"Team shooting stats failed for {season}: {e}")
            return pd.DataFrame()
    
    @retry_with_backoff
    def _fetch_standings(self, season: str) -> pd.DataFrame:
        """Fetch standings with win streaks"""
        try:
            standings = leaguestandingsv3.LeagueStandingsV3(
                season=season,
                league_id="00"
            )
            return standings.get_data_frames()[0]
        except Exception as e:
            logger.warning(f"Standings failed for {season}: {e}")
            return pd.DataFrame()
    
    def get_team_stats(self, season: str, force_refresh: bool = False) -> pd.DataFrame:
        if not force_refresh and self.cache.is_cached("team_stats", season):
            logger.info(f"Loading cached team stats for {season}")
            return self.cache.load_cached("team_stats", season)
        
        logger.info(f"Fetching team stats for {season}...")
        df = self._fetch_team_stats(season)
        self.cache.save_to_cache(df, "team_stats", season)
        return df
    
    def get_team_advanced(self, season: str, force_refresh: bool = False) -> pd.DataFrame:
        if not force_refresh and self.cache.is_cached("team_advanced", season):
            return self.cache.load_cached("team_advanced", season)
        
        logger.info(f"Fetching team advanced metrics for {season}...")
        df = self._fetch_team_advanced(season)
        if not df.empty:
            self.cache.save_to_cache(df, "team_advanced", season)
        return df
    
    def get_team_clutch(self, season: str, force_refresh: bool = False) -> pd.DataFrame:
        if not force_refresh and self.cache.is_cached("team_clutch", season):
            return self.cache.load_cached("team_clutch", season)
        
        logger.info(f"Fetching team clutch stats for {season}...")
        df = self._fetch_team_clutch(season)
        if not df.empty:
            self.cache.save_to_cache(df, "team_clutch", season)
        return df
    
    def get_team_hustle(self, season: str, force_refresh: bool = False) -> pd.DataFrame:
        if not force_refresh and self.cache.is_cached("team_hustle", season):
            return self.cache.load_cached("team_hustle", season)
        
        logger.info(f"Fetching team hustle stats for {season}...")
        df = self._fetch_team_hustle(season)
        if not df.empty:
            self.cache.save_to_cache(df, "team_hustle", season)
        return df
    
    def get_team_defense(self, season: str, force_refresh: bool = False) -> pd.DataFrame:
        if not force_refresh and self.cache.is_cached("team_defense", season):
            return self.cache.load_cached("team_defense", season)
        
        logger.info(f"Fetching team defense stats for {season}...")
        df = self._fetch_team_defense(season)
        if not df.empty:
            self.cache.save_to_cache(df, "team_defense", season)
        return df
    
    def get_standings(self, season: str, force_refresh: bool = False) -> pd.DataFrame:
        if not force_refresh and self.cache.is_cached("standings", season):
            return self.cache.load_cached("standings", season)
        
        logger.info(f"Fetching standings for {season}...")
        df = self._fetch_standings(season)
        if not df.empty:
            self.cache.save_to_cache(df, "standings", season)
        return df
    
    def collect_all_seasons(self, seasons: List[str] = None, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        if seasons is None:
            seasons = SEASON_STRINGS
        
        results = {
            "team_stats": [],
            "team_advanced": [],
            "team_clutch": [],
            "team_hustle": [],
            "team_defense": [],
            "standings": []
        }
        
        for season in tqdm(seasons, desc="Collecting team data"):
            # Basic stats
            df = self.get_team_stats(season, force_refresh)
            df["SEASON"] = season
            results["team_stats"].append(df)
            
            # Advanced metrics
            df = self.get_team_advanced(season, force_refresh)
            if not df.empty:
                df["SEASON"] = season
                results["team_advanced"].append(df)
            
            # Clutch stats
            df = self.get_team_clutch(season, force_refresh)
            if not df.empty:
                df["SEASON"] = season
                results["team_clutch"].append(df)
            
            # Hustle stats
            df = self.get_team_hustle(season, force_refresh)
            if not df.empty:
                df["SEASON"] = season
                results["team_hustle"].append(df)
            
            # Defense stats
            df = self.get_team_defense(season, force_refresh)
            if not df.empty:
                df["SEASON"] = season
                results["team_defense"].append(df)
            
            # Standings
            df = self.get_standings(season, force_refresh)
            if not df.empty:
                df["SEASON"] = season
                results["standings"].append(df)
        
        # Save all combined data
        for key, data in results.items():
            if data:
                combined = pd.concat(data, ignore_index=True)
                combined.to_parquet(RAW_DATA_DIR / f"all_{key}.parquet", index=False)
                logger.info(f"Saved {len(combined)} rows to all_{key}.parquet")
        
        return results


# =============================================================================
# PLAYER DATA COLLECTOR (ENHANCED)
# =============================================================================
class PlayerDataCollector:
    """Collects comprehensive player statistics."""
    
    def __init__(self):
        self.cache = CacheManager()
    
    @retry_with_backoff
    def _fetch_player_stats(self, season: str) -> pd.DataFrame:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed="PerGame"
        )
        return stats.get_data_frames()[0]
    
    @retry_with_backoff
    def _fetch_player_advanced(self, season: str) -> pd.DataFrame:
        """Fetch advanced player metrics: PER, USG%, TS%, eFG%"""
        try:
            stats = playerestimatedmetrics.PlayerEstimatedMetrics(season=season)
            return stats.get_data_frames()[0]
        except Exception as e:
            logger.warning(f"PlayerEstimatedMetrics failed for {season}: {e}")
            return pd.DataFrame()
    
    @retry_with_backoff
    def _fetch_player_clutch(self, season: str) -> pd.DataFrame:
        """Fetch player clutch stats"""
        try:
            stats = leaguedashplayerclutch.LeagueDashPlayerClutch(
                season=season,
                clutch_time="Last 5 Minutes",
                point_diff=5
            )
            return stats.get_data_frames()[0]
        except Exception as e:
            logger.warning(f"Player clutch stats failed for {season}: {e}")
            return pd.DataFrame()
    
    @retry_with_backoff
    def _fetch_player_hustle(self, season: str) -> pd.DataFrame:
        """Fetch player hustle stats"""
        try:
            stats = leaguehustlestatsplayer.LeagueHustleStatsPlayer(season=season)
            return stats.get_data_frames()[0]
        except Exception as e:
            logger.warning(f"Player hustle stats failed for {season}: {e}")
            return pd.DataFrame()
    
    @retry_with_backoff
    def _fetch_player_shooting(self, season: str) -> pd.DataFrame:
        """Fetch player shooting stats"""
        try:
            stats = leaguedashplayerptshot.LeagueDashPlayerPtShot(season=season)
            return stats.get_data_frames()[0]
        except Exception as e:
            logger.warning(f"Player shooting stats failed for {season}: {e}")
            return pd.DataFrame()
    
    def get_player_stats(self, season: str, force_refresh: bool = False) -> pd.DataFrame:
        if not force_refresh and self.cache.is_cached("player_stats", season):
            logger.info(f"Loading cached player stats for {season}")
            return self.cache.load_cached("player_stats", season)
        
        logger.info(f"Fetching player stats for {season}...")
        df = self._fetch_player_stats(season)
        self.cache.save_to_cache(df, "player_stats", season)
        return df
    
    def get_player_advanced(self, season: str, force_refresh: bool = False) -> pd.DataFrame:
        if not force_refresh and self.cache.is_cached("player_advanced", season):
            return self.cache.load_cached("player_advanced", season)
        
        logger.info(f"Fetching player advanced metrics for {season}...")
        df = self._fetch_player_advanced(season)
        if not df.empty:
            self.cache.save_to_cache(df, "player_advanced", season)
        return df
    
    def get_player_clutch(self, season: str, force_refresh: bool = False) -> pd.DataFrame:
        if not force_refresh and self.cache.is_cached("player_clutch", season):
            return self.cache.load_cached("player_clutch", season)
        
        logger.info(f"Fetching player clutch stats for {season}...")
        df = self._fetch_player_clutch(season)
        if not df.empty:
            self.cache.save_to_cache(df, "player_clutch", season)
        return df
    
    def get_player_hustle(self, season: str, force_refresh: bool = False) -> pd.DataFrame:
        if not force_refresh and self.cache.is_cached("player_hustle", season):
            return self.cache.load_cached("player_hustle", season)
        
        logger.info(f"Fetching player hustle stats for {season}...")
        df = self._fetch_player_hustle(season)
        if not df.empty:
            self.cache.save_to_cache(df, "player_hustle", season)
        return df
    
    def collect_all_seasons(self, seasons: List[str] = None, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        if seasons is None:
            seasons = SEASON_STRINGS
        
        results = {
            "player_stats": [],
            "player_advanced": [],
            "player_clutch": [],
            "player_hustle": []
        }
        
        for season in tqdm(seasons, desc="Collecting player data"):
            # Basic stats
            df = self.get_player_stats(season, force_refresh)
            df["SEASON"] = season
            results["player_stats"].append(df)
            
            # Advanced metrics
            df = self.get_player_advanced(season, force_refresh)
            if not df.empty:
                df["SEASON"] = season
                results["player_advanced"].append(df)
            
            # Clutch stats
            df = self.get_player_clutch(season, force_refresh)
            if not df.empty:
                df["SEASON"] = season
                results["player_clutch"].append(df)
            
            # Hustle stats
            df = self.get_player_hustle(season, force_refresh)
            if not df.empty:
                df["SEASON"] = season
                results["player_hustle"].append(df)
        
        # Save all combined data
        for key, data in results.items():
            if data:
                combined = pd.concat(data, ignore_index=True)
                combined.to_parquet(RAW_DATA_DIR / f"all_{key}.parquet", index=False)
                logger.info(f"Saved {len(combined)} rows to all_{key}.parquet")
        
        return results


# =============================================================================
# LEAGUE LEADERS COLLECTOR
# =============================================================================
class LeagueLeadersCollector:
    """Collects league leaders data."""
    
    def __init__(self):
        self.cache = CacheManager()
    
    @retry_with_backoff
    def _fetch_leaders(self, season: str, stat_category: str = "PTS") -> pd.DataFrame:
        leaders = leagueleaders.LeagueLeaders(
            season=season,
            stat_category_abbreviation=stat_category
        )
        return leaders.get_data_frames()[0]
    
    def get_leaders(self, season: str, force_refresh: bool = False) -> pd.DataFrame:
        if not force_refresh and self.cache.is_cached("leaders", season):
            return self.cache.load_cached("leaders", season)
        
        logger.info(f"Fetching league leaders for {season}...")
        df = self._fetch_leaders(season)
        self.cache.save_to_cache(df, "leaders", season)
        return df


# =============================================================================
# MASTER COLLECTOR
# =============================================================================
class NBADataCollector:
    """Master collector that orchestrates ALL data collection."""
    
    def __init__(self):
        self.games = GameDataCollector()
        self.teams = TeamDataCollector()
        self.players = PlayerDataCollector()
        self.leaders = LeagueLeadersCollector()
        self.cache = CacheManager()
    
    def collect_all(self, seasons: List[str] = None, force_refresh: bool = False):
        """Collect ALL data for specified seasons."""
        if seasons is None:
            seasons = SEASON_STRINGS
        
        logger.info(f"Starting COMPREHENSIVE data collection for {len(seasons)} seasons...")
        logger.info(f"Seasons: {seasons[0]} to {seasons[-1]}")
        logger.info("This will take several hours. Data is cached, so it can resume if interrupted.\n")
        
        # Collect games
        logger.info("=" * 50)
        logger.info("=== PHASE 1: Collecting Games ===")
        logger.info("=" * 50)
        self.games.collect_all_seasons(seasons, force_refresh)
        
        # Collect team stats (all types)
        logger.info("\n" + "=" * 50)
        logger.info("=== PHASE 2: Collecting Team Stats (6 data types) ===")
        logger.info("=" * 50)
        self.teams.collect_all_seasons(seasons, force_refresh)
        
        # Collect player stats (all types)
        logger.info("\n" + "=" * 50)
        logger.info("=== PHASE 3: Collecting Player Stats (4 data types) ===")
        logger.info("=" * 50)
        self.players.collect_all_seasons(seasons, force_refresh)
        
        logger.info("\n" + "=" * 50)
        logger.info("=== DATA COLLECTION COMPLETE ===")
        logger.info("=" * 50)
        logger.info(f"Data saved to: {RAW_DATA_DIR}")
        
        # List all generated files
        parquet_files = list(RAW_DATA_DIR.glob("*.parquet"))
        logger.info(f"\nGenerated {len(parquet_files)} data files:")
        for f in parquet_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            logger.info(f"  - {f.name} ({size_mb:.2f} MB)")


# =============================================================================
# CLI INTERFACE
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NBA Comprehensive Data Collector")
    parser.add_argument("--seasons", nargs="+", help="Specific seasons to collect (e.g., 2023-24)")
    parser.add_argument("--force", action="store_true", help="Force refresh, ignore cache")
    parser.add_argument("--games-only", action="store_true", help="Only collect games")
    parser.add_argument("--teams-only", action="store_true", help="Only collect team stats")
    parser.add_argument("--players-only", action="store_true", help="Only collect player stats")
    parser.add_argument("--test", action="store_true", help="Test with single season")
    
    args = parser.parse_args()
    
    collector = NBADataCollector()
    
    if args.test:
        print("Running in test mode (single season 2024-25)...")
        collector.collect_all(["2024-25"], args.force)
        print("Test complete!")
    elif args.games_only:
        collector.games.collect_all_seasons(args.seasons, args.force)
    elif args.teams_only:
        collector.teams.collect_all_seasons(args.seasons, args.force)
    elif args.players_only:
        collector.players.collect_all_seasons(args.seasons, args.force)
    else:
        collector.collect_all(args.seasons, args.force)
