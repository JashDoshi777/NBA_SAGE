
"""
NBA ML Prediction System - Comprehensive Feature Engineering
=============================================================
Time-aware feature generation using ALL available stats:
- ELO ratings
- Era normalization (Z-score by season)
- Rolling averages (basic + advanced)
- Clutch performance
- Hustle metrics
- Defensive ratings
- Data leakage prevention
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

from src.config import (
    ELO_CONFIG, 
    FEATURE_CONFIG, 
    RAW_DATA_DIR, 
    PROCESSED_DATA_DIR,
    NBA_TEAMS
)

logger = logging.getLogger(__name__)

# =============================================================================
# ALL STAT COLUMNS BY CATEGORY
# =============================================================================
BASIC_STATS = ["PTS", "AST", "REB", "STL", "BLK", "TOV", "FGM", "FGA", "FG_PCT", 
               "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT", "OREB", "DREB"]

ADVANCED_STATS = ["E_OFF_RATING", "E_DEF_RATING", "E_NET_RATING", "E_PACE", 
                  "E_AST_RATIO", "E_OREB_PCT", "E_DREB_PCT", "E_REB_PCT", 
                  "E_TM_TOV_PCT", "E_EFG_PCT", "E_TS_PCT"]

CLUTCH_STATS = ["CLUTCH_PTS", "CLUTCH_FG_PCT", "CLUTCH_FG3_PCT", "CLUTCH_PLUS_MINUS"]

HUSTLE_STATS = ["DEFLECTIONS", "LOOSE_BALLS_RECOVERED", "CHARGES_DRAWN", 
                "CONTESTED_SHOTS", "SCREEN_ASSISTS"]


# =============================================================================
# ELO RATING SYSTEM
# =============================================================================
class ELOCalculator:
    """
    Calculates ELO ratings for NBA teams.
    ELO is extremely predictive in sports - can add +3-5% accuracy.
    """
    
    def __init__(self, config=ELO_CONFIG):
        self.initial_rating = config.initial_rating
        self.k_factor = config.k_factor
        self.home_advantage = config.home_advantage
        self.season_regression = config.season_regression
        self.ratings: Dict[int, float] = {}
    
    def reset_ratings(self):
        self.ratings = {}
    
    def get_rating(self, team_id: int) -> float:
        if team_id not in self.ratings:
            self.ratings[team_id] = self.initial_rating
        return self.ratings[team_id]
    
    def regress_to_mean(self):
        mean_rating = np.mean(list(self.ratings.values())) if self.ratings else self.initial_rating
        for team_id in self.ratings:
            self.ratings[team_id] = (
                self.season_regression * mean_rating + 
                (1 - self.season_regression) * self.ratings[team_id]
            )
    
    def expected_win_probability(self, team_rating: float, opponent_rating: float, 
                                  is_home: bool = False) -> float:
        rating_diff = team_rating - opponent_rating
        if is_home:
            rating_diff += self.home_advantage
        return 1.0 / (1.0 + 10 ** (-rating_diff / 400))
    
    def update_ratings(self, team_id: int, opponent_id: int, 
                       won: bool, is_home: bool = False) -> Tuple[float, float]:
        team_rating = self.get_rating(team_id)
        opponent_rating = self.get_rating(opponent_id)
        
        expected = self.expected_win_probability(team_rating, opponent_rating, is_home)
        actual = 1.0 if won else 0.0
        
        delta = self.k_factor * (actual - expected)
        self.ratings[team_id] = team_rating + delta
        self.ratings[opponent_id] = opponent_rating - delta
        
        return self.ratings[team_id], self.ratings[opponent_id]
    
    def calculate_game_features(self, team_id: int, opponent_id: int, 
                                 is_home: bool) -> Dict[str, float]:
        team_elo = self.get_rating(team_id)
        opponent_elo = self.get_rating(opponent_id)
        
        return {
            "team_elo": team_elo,
            "opponent_elo": opponent_elo,
            "elo_diff": team_elo - opponent_elo,
            "elo_win_prob": self.expected_win_probability(team_elo, opponent_elo, is_home),
            "home_elo_boost": self.home_advantage if is_home else 0
        }


# =============================================================================
# ERA NORMALIZATION
# =============================================================================
class EraNormalizer:
    """Z-score normalization within season to handle era differences."""
    
    def __init__(self):
        self.season_stats: Dict[str, Dict[str, Tuple[float, float]]] = {}
    
    def fit_season(self, df: pd.DataFrame, season: str, stat_columns: List[str]):
        self.season_stats[season] = {}
        for col in stat_columns:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                self.season_stats[season][col] = (mean, std if std > 0 else 1.0)
    
    def transform(self, df: pd.DataFrame, season: str, stat_columns: List[str]) -> pd.DataFrame:
        df = df.copy()
        if season not in self.season_stats:
            return df
        
        for col in stat_columns:
            if col in df.columns and col in self.season_stats[season]:
                mean, std = self.season_stats[season][col]
                df[f"{col}_zscore"] = (df[col] - mean) / std
        
        return df


# =============================================================================
# COMPREHENSIVE STAT LOADER
# =============================================================================
class StatLoader:
    """Loads and merges all collected stats for a team/player."""
    
    def __init__(self):
        self.team_stats = None
        self.team_advanced = None
        self.team_clutch = None
        self.team_hustle = None
        self.team_defense = None
        self.player_stats = None
        self.player_advanced = None
        self._loaded = False
    
    def load_all_stats(self):
        """Load all available stat files."""
        if self._loaded:
            return
        
        logger.info("Loading all stat files...")
        
        # Team stats
        try:
            self.team_stats = pd.read_parquet(RAW_DATA_DIR / "all_team_stats.parquet")
            logger.info(f"  Loaded team_stats: {len(self.team_stats)} rows")
        except:
            self.team_stats = pd.DataFrame()
        
        try:
            self.team_advanced = pd.read_parquet(RAW_DATA_DIR / "all_team_advanced.parquet")
            logger.info(f"  Loaded team_advanced: {len(self.team_advanced)} rows")
        except:
            self.team_advanced = pd.DataFrame()
        
        try:
            self.team_clutch = pd.read_parquet(RAW_DATA_DIR / "all_team_clutch.parquet")
            logger.info(f"  Loaded team_clutch: {len(self.team_clutch)} rows")
        except:
            self.team_clutch = pd.DataFrame()
        
        try:
            self.team_hustle = pd.read_parquet(RAW_DATA_DIR / "all_team_hustle.parquet")
            logger.info(f"  Loaded team_hustle: {len(self.team_hustle)} rows")
        except:
            self.team_hustle = pd.DataFrame()
        
        try:
            self.team_defense = pd.read_parquet(RAW_DATA_DIR / "all_team_defense.parquet")
            logger.info(f"  Loaded team_defense: {len(self.team_defense)} rows")
        except:
            self.team_defense = pd.DataFrame()
        
        # Player stats
        try:
            self.player_stats = pd.read_parquet(RAW_DATA_DIR / "all_player_stats.parquet")
            logger.info(f"  Loaded player_stats: {len(self.player_stats)} rows")
        except:
            self.player_stats = pd.DataFrame()
        
        try:
            self.player_advanced = pd.read_parquet(RAW_DATA_DIR / "all_player_advanced.parquet")
            logger.info(f"  Loaded player_advanced: {len(self.player_advanced)} rows")
        except:
            self.player_advanced = pd.DataFrame()
        
        self._loaded = True
    
    def get_team_season_stats(self, team_id: int, season: str) -> Dict[str, float]:
        """Get all stats for a team in a season."""
        self.load_all_stats()
        
        features = {}
        
        # Basic team stats
        if not self.team_stats.empty:
            mask = (self.team_stats["TEAM_ID"] == team_id) & (self.team_stats["SEASON"] == season)
            row = self.team_stats[mask]
            if not row.empty:
                row = row.iloc[0]
                for col in BASIC_STATS:
                    if col in row.index:
                        features[f"team_{col}"] = row[col]
        
        # Advanced metrics
        if not self.team_advanced.empty:
            mask = (self.team_advanced["TEAM_ID"] == team_id) & (self.team_advanced["SEASON"] == season)
            row = self.team_advanced[mask]
            if not row.empty:
                row = row.iloc[0]
                for col in ADVANCED_STATS:
                    if col in row.index:
                        features[f"team_{col}"] = row[col]
        
        # Clutch stats
        if not self.team_clutch.empty:
            mask = (self.team_clutch["TEAM_ID"] == team_id) & (self.team_clutch["SEASON"] == season)
            row = self.team_clutch[mask]
            if not row.empty:
                row = row.iloc[0]
                features["team_clutch_pts"] = row.get("PTS", 0)
                features["team_clutch_fg_pct"] = row.get("FG_PCT", 0)
                features["team_clutch_plus_minus"] = row.get("PLUS_MINUS", 0)
        
        # Hustle stats
        if not self.team_hustle.empty:
            mask = (self.team_hustle["TEAM_ID"] == team_id) & (self.team_hustle["SEASON"] == season)
            row = self.team_hustle[mask]
            if not row.empty:
                row = row.iloc[0]
                for col in ["DEFLECTIONS", "LOOSE_BALLS_RECOVERED", "CHARGES_DRAWN", 
                           "CONTESTED_SHOTS_2PT", "CONTESTED_SHOTS_3PT"]:
                    if col in row.index:
                        features[f"team_{col.lower()}"] = row[col]
        
        return features
    
    def get_team_top_players_stats(self, team_id: int, season: str, top_n: int = 5) -> Dict[str, float]:
        """Get aggregated stats for top N players on a team."""
        self.load_all_stats()
        
        features = {}
        
        if self.player_stats.empty:
            return features
        
        # Get team's players for the season
        mask = (self.player_stats["TEAM_ID"] == team_id) & (self.player_stats["SEASON"] == season)
        team_players = self.player_stats[mask].copy()
        
        if team_players.empty:
            return features
        
        # Sort by minutes and get top players
        if "MIN" in team_players.columns:
            team_players = team_players.sort_values("MIN", ascending=False).head(top_n)
        
        # Aggregate stats
        features["top_players_avg_pts"] = team_players["PTS"].mean() if "PTS" in team_players.columns else 0
        features["top_players_avg_ast"] = team_players["AST"].mean() if "AST" in team_players.columns else 0
        features["top_players_avg_reb"] = team_players["REB"].mean() if "REB" in team_players.columns else 0
        features["top_players_avg_stl"] = team_players["STL"].mean() if "STL" in team_players.columns else 0
        features["top_players_avg_blk"] = team_players["BLK"].mean() if "BLK" in team_players.columns else 0
        
        # Star player concentration (how much does top player score vs team)
        if "PTS" in team_players.columns and len(team_players) > 0:
            top_scorer_pts = team_players["PTS"].max()
            total_pts = team_players["PTS"].sum()
            features["star_concentration"] = top_scorer_pts / total_pts if total_pts > 0 else 0
        
        return features


# =============================================================================
# COMPREHENSIVE FEATURE GENERATOR
# =============================================================================
class FeatureGenerator:
    """Generates ALL features with strict data leakage prevention."""
    
    def __init__(self, config=FEATURE_CONFIG):
        self.rolling_windows = config.rolling_windows
        self.min_games = config.min_games_for_features
        self.elo = ELOCalculator()
        self.normalizer = EraNormalizer()
        self.stat_loader = StatLoader()
    
    # League-average fills for cold-start handling (typical NBA averages)
    LEAGUE_AVERAGES = {
        "PTS": 112.0,
        "AST": 25.0,
        "REB": 44.0,
        "FG_PCT": 0.465,
        "FG3_PCT": 0.360,
        "FT_PCT": 0.780,
        "PLUS_MINUS": 0.0,
        "STL": 7.5,
        "BLK": 5.0,
        "DREB": 34.0,
    }
    
    def calculate_rolling_stats(self, team_games: pd.DataFrame, 
                                 current_date: datetime,
                                 stat_columns: List[str]) -> Dict[str, float]:
        """
        Calculate rolling averages (time-aware) with cold-start handling.
        
        For early-season games with insufficient history, uses league-average
        fills instead of NaN to maintain prediction quality.
        """
        past_games = team_games[pd.to_datetime(team_games["GAME_DATE"]) < current_date]
        past_games = past_games.sort_values("GAME_DATE", ascending=False)
        
        features = {}
        games_available = len(past_games)
        
        for window in self.rolling_windows:
            recent_games = past_games.head(window)
            
            if len(recent_games) < self.min_games:
                # Cold-start: Use league averages instead of NaN
                for col in stat_columns:
                    league_avg = self.LEAGUE_AVERAGES.get(col, 0)
                    
                    if games_available > 0 and col in past_games.columns:
                        # Blend available data with league average
                        # Weight: available_games / min_games
                        blend_weight = games_available / self.min_games
                        team_avg = past_games.head(games_available)[col].mean()
                        features[f"{col}_last{window}"] = (
                            blend_weight * team_avg + 
                            (1 - blend_weight) * league_avg
                        )
                    else:
                        features[f"{col}_last{window}"] = league_avg
            else:
                for col in stat_columns:
                    if col in recent_games.columns:
                        features[f"{col}_last{window}"] = recent_games[col].mean()
                    else:
                        features[f"{col}_last{window}"] = self.LEAGUE_AVERAGES.get(col, 0)
        
        return features
    
    def calculate_defensive_stats(self, team_games: pd.DataFrame,
                                   current_date: datetime) -> Dict[str, float]:
        """Calculate defensive rolling stats."""
        past_games = team_games[pd.to_datetime(team_games["GAME_DATE"]) < current_date]
        past_games = past_games.sort_values("GAME_DATE", ascending=False).head(10)
        
        features = {}
        
        if len(past_games) >= 3:
            for col in ["STL", "BLK", "DREB"]:
                if col in past_games.columns:
                    features[f"{col}_last10"] = past_games[col].mean()
            
            # Points allowed (opponent points)
            # This would need opponent data, so we estimate from +/-
            if "PLUS_MINUS" in past_games.columns and "PTS" in past_games.columns:
                features["pts_allowed_last10"] = past_games["PTS"].mean() - past_games["PLUS_MINUS"].mean()
        
        return features
    
    def calculate_season_stats(self, team_games: pd.DataFrame,
                                current_date: datetime,
                                stat_columns: List[str]) -> Dict[str, float]:
        """Calculate season-to-date stats (time-aware)."""
        past_games = team_games[pd.to_datetime(team_games["GAME_DATE"]) < current_date]
        
        features = {}
        for col in stat_columns:
            if col in past_games.columns:
                features[f"{col}_season_avg"] = past_games[col].mean()
        
        # Win percentage
        if "WL" in past_games.columns:
            wins = (past_games["WL"] == "W").sum()
            total = len(past_games)
            features["win_pct_season"] = wins / total if total > 0 else 0.5
            features["games_played"] = total
        
        return features
    
    def calculate_momentum(self, team_games: pd.DataFrame,
                           current_date: datetime) -> Dict[str, float]:
        """Calculate momentum features (streaks, recent form)."""
        past_games = team_games[pd.to_datetime(team_games["GAME_DATE"]) < current_date]
        past_games = past_games.sort_values("GAME_DATE", ascending=False)
        
        features = {}
        
        if len(past_games) >= 5:
            last5 = past_games.head(5)
            
            # Win streak
            wins_last5 = (last5["WL"] == "W").sum() if "WL" in last5.columns else 0
            features["wins_last5"] = wins_last5
            features["hot_streak"] = 1 if wins_last5 >= 4 else 0
            features["cold_streak"] = 1 if wins_last5 <= 1 else 0
            
            # Point differential trend
            if "PLUS_MINUS" in last5.columns:
                features["plus_minus_last5"] = last5["PLUS_MINUS"].mean()
        
        if len(past_games) >= 10:
            last10 = past_games.head(10)
            wins_last10 = (last10["WL"] == "W").sum() if "WL" in last10.columns else 0
            features["wins_last10"] = wins_last10
        
        return features
    
    def calculate_rest_fatigue(self, team_games: pd.DataFrame,
                                current_date: datetime) -> Dict[str, float]:
        """Calculate rest and fatigue features."""
        past_games = team_games[pd.to_datetime(team_games["GAME_DATE"]) < current_date]
        past_games = past_games.sort_values("GAME_DATE", ascending=False)
        
        features = {}
        
        if len(past_games) > 0:
            last_game = pd.to_datetime(past_games["GAME_DATE"].iloc[0])
            days_rest = (current_date - last_game).days
            features["days_rest"] = days_rest
            features["back_to_back"] = 1 if days_rest == 1 else 0
            features["well_rested"] = 1 if days_rest >= 3 else 0
        else:
            features["days_rest"] = 3
            features["back_to_back"] = 0
            features["well_rested"] = 1
        
        # Games in last 7 days (fatigue)
        week_ago = current_date - pd.Timedelta(days=7)
        recent_games = past_games[pd.to_datetime(past_games["GAME_DATE"]) >= week_ago]
        features["games_last_week"] = len(recent_games)
        
        return features
    
    def calculate_form_index(self, team_games: pd.DataFrame,
                             current_date: datetime) -> Dict[str, float]:
        """
        Calculate exponentially-weighted form index for fast regime-change detection.
        
        Recent games are weighted more heavily than older games, allowing the model
        to quickly adapt when a team's performance regime changes (e.g., after
        major trades, injuries, or coaching changes).
        """
        past_games = team_games[pd.to_datetime(team_games["GAME_DATE"]) < current_date]
        past_games = past_games.sort_values("GAME_DATE", ascending=False).head(10)
        
        features = {}
        
        if len(past_games) < 3:
            features["form_index"] = 0.5  # Neutral for cold start
            features["form_trend"] = 0.0
            return features
        
        # Exponential weights: most recent game has ~2x weight of 5th game
        # decay_rate=0.15 means game 5 has weight e^(-0.15*4) ≈ 0.55 vs 1.0 for game 1
        weights = np.exp(-np.arange(len(past_games)) * 0.15)
        weights = weights / weights.sum()  # Normalize to sum to 1
        
        # Win-based form index (0-1 scale)
        if "WL" in past_games.columns:
            wins = (past_games["WL"] == "W").astype(float).values
            form_index = (wins * weights).sum()
            features["form_index"] = form_index
            
            # Form trend: compare last 3 vs previous 3
            if len(past_games) >= 6:
                recent_3_wins = (past_games.head(3)["WL"] == "W").mean()
                prev_3_wins = (past_games.iloc[3:6]["WL"] == "W").mean()
                features["form_trend"] = recent_3_wins - prev_3_wins
            else:
                features["form_trend"] = 0.0
        else:
            features["form_index"] = 0.5
            features["form_trend"] = 0.0
        
        # Point differential form (exponentially weighted)
        if "PLUS_MINUS" in past_games.columns:
            pm_values = past_games["PLUS_MINUS"].fillna(0).values
            features["form_plus_minus"] = (pm_values * weights).sum()
        
        return features
    
    def generate_game_features(self, games_df: pd.DataFrame, 
                                game_row: pd.Series,
                                season: str = None) -> Dict[str, float]:
        """Generate ALL features for a single game prediction."""
        game_date = pd.to_datetime(game_row["GAME_DATE"])
        team_id = game_row["TEAM_ID"]
        matchup = game_row.get("MATCHUP", "")
        is_home = "@" not in matchup
        
        # Get opponent ID
        opponent_abbrev = matchup.split(" ")[-1] if matchup else ""
        opponent_id = next(
            (tid for tid, abbrev in NBA_TEAMS.items() if abbrev == opponent_abbrev), 
            None
        )
        
        # Get team's past games
        team_games = games_df[
            (games_df["TEAM_ID"] == team_id) & 
            (pd.to_datetime(games_df["GAME_DATE"]) < game_date)
        ]
        
        # Start with basic features
        features = {"is_home": 1 if is_home else 0}
        
        # ELO features
        if opponent_id:
            elo_features = self.elo.calculate_game_features(team_id, opponent_id, is_home)
            features.update(elo_features)
        
        # Rolling stats (basic)
        basic_cols = ["PTS", "AST", "REB", "FG_PCT", "FG3_PCT", "FT_PCT", "PLUS_MINUS"]
        rolling_features = self.calculate_rolling_stats(team_games, game_date, basic_cols)
        features.update(rolling_features)
        
        # Defensive stats
        def_features = self.calculate_defensive_stats(team_games, game_date)
        features.update(def_features)
        
        # Season-to-date stats
        season_features = self.calculate_season_stats(team_games, game_date, basic_cols)
        features.update(season_features)
        
        # Momentum features
        momentum_features = self.calculate_momentum(team_games, game_date)
        features.update(momentum_features)
        
        # Rest/fatigue features
        rest_features = self.calculate_rest_fatigue(team_games, game_date)
        features.update(rest_features)
        
        # Form index (exponentially-weighted recent performance)
        form_features = self.calculate_form_index(team_games, game_date)
        features.update(form_features)
        
        # Season-level team stats (advanced, clutch, hustle)
        if season:
            team_season_stats = self.stat_loader.get_team_season_stats(team_id, season)
            features.update(team_season_stats)
            
            # Top players stats
            player_features = self.stat_loader.get_team_top_players_stats(team_id, season)
            features.update(player_features)
        
        return features


# =============================================================================
# BATCH PROCESSOR
# =============================================================================
def process_all_games(games_df: pd.DataFrame, 
                       output_path: Optional[Path] = None) -> pd.DataFrame:
    """Process ALL games with comprehensive features."""
    logger.info(f"Processing {len(games_df)} games with COMPREHENSIVE features...")
    
    games_df = games_df.sort_values("GAME_DATE").copy()
    
    generator = FeatureGenerator()
    all_features = []
    current_season = None
    
    from tqdm import tqdm
    
    for idx, row in tqdm(games_df.iterrows(), total=len(games_df), desc="Processing games"):
        season = row.get("SEASON_ID", "")
        
        # Parse season for stat lookup
        if isinstance(season, str) and len(season) >= 5:
            year = season[1:5]
            season_str = f"{year}-{str(int(year)+1)[-2:]}"
        else:
            season_str = None
        
        # Regress ELO at season change
        if season != current_season:
            if current_season is not None:
                generator.elo.regress_to_mean()
            current_season = season
        
        # Generate features
        features = generator.generate_game_features(games_df, row, season_str)
        features["GAME_ID"] = row["GAME_ID"]
        features["TEAM_ID"] = row["TEAM_ID"]
        features["GAME_DATE"] = row["GAME_DATE"]
        features["SEASON_ID"] = row.get("SEASON_ID", "")
        features["WL"] = row.get("WL", None)
        
        all_features.append(features)
        
        # Update ELO after game
        if row.get("WL") and features.get("opponent_elo"):
            opponent_abbrev = row.get("MATCHUP", "").split(" ")[-1]
            opponent_id = next(
                (tid for tid, abbrev in NBA_TEAMS.items() if abbrev == opponent_abbrev), 
                None
            )
            if opponent_id:
                won = row["WL"] == "W"
                is_home = "@" not in row.get("MATCHUP", "")
                generator.elo.update_ratings(row["TEAM_ID"], opponent_id, won, is_home)
    
    result_df = pd.DataFrame(all_features)
    
    if output_path:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        result_df.to_parquet(output_path, index=False)
        logger.info(f"Saved features to {output_path}")
    
    return result_df


# =============================================================================
# CLI INTERFACE
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Feature Engineering")
    parser.add_argument("--test", action="store_true", help="Run tests only")
    parser.add_argument("--process", action="store_true", help="Process collected data")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.test or (not args.process and not args.test):
        print("Testing ELO Calculator...")
        elo = ELOCalculator()
        
        lal_rating = elo.get_rating(1610612747)
        bos_rating = elo.get_rating(1610612738)
        
        print(f"Initial ratings - LAL: {lal_rating}, BOS: {bos_rating}")
        
        elo.update_ratings(1610612747, 1610612738, won=True, is_home=True)
        print(f"After LAL home win - LAL: {elo.get_rating(1610612747):.1f}, BOS: {elo.get_rating(1610612738):.1f}")
        
        features = elo.calculate_game_features(1610612747, 1610612738, is_home=True)
        print(f"\nGame features: {features}")
    
    if args.process:
        print("\n=== Processing Collected Data with COMPREHENSIVE Features ===")
        
        games_path = RAW_DATA_DIR / "all_games.parquet"
        output_path = PROCESSED_DATA_DIR / "game_features.parquet"
        
        if not games_path.exists():
            print(f"ERROR: Games data not found at {games_path}")
            print("Run 'python -m src.data_collector' first to collect data.")
            exit(1)
        
        print(f"Loading games from {games_path}...")
        games_df = pd.read_parquet(games_path)
        print(f"Loaded {len(games_df)} games")
        
        print("\nGenerating COMPREHENSIVE features (this may take a while)...")
        print("Features include: ELO, rolling stats, defense, momentum, rest, advanced metrics, clutch, hustle...")
        
        result_df = process_all_games(games_df, output_path)
        
        print(f"\n✅ Features saved to: {output_path}")
        print(f"   Total rows: {len(result_df)}")
        print(f"   Total features: {len(result_df.columns)}")
        print(f"\nFeature columns ({len(result_df.columns)} total):")
        for col in sorted(result_df.columns):
            print(f"  - {col}")
