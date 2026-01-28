"""
NBA ML Prediction System - MVP Predictor
=========================================
Model to predict MVP based on player performance, team success, 
and historical MVP similarity.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import logging

from src.config import MODELS_DIR, RAW_DATA_DIR

logger = logging.getLogger(__name__)

# =============================================================================
# HISTORICAL MVP PROFILES
# =============================================================================
# Historical MVP seasons (approximate stats for similarity comparison)
HISTORICAL_MVP_PROFILES = {
    "2023-24": {"player": "Nikola Jokic", "ppg": 26.4, "rpg": 12.4, "apg": 9.0, "ws": 17.8, "team_wins": 57},
    "2022-23": {"player": "Joel Embiid", "ppg": 33.1, "rpg": 10.2, "apg": 4.2, "ws": 14.3, "team_wins": 54},
    "2021-22": {"player": "Nikola Jokic", "ppg": 27.1, "rpg": 13.8, "apg": 7.9, "ws": 15.2, "team_wins": 48},
    "2020-21": {"player": "Nikola Jokic", "ppg": 26.4, "rpg": 10.8, "apg": 8.3, "ws": 15.6, "team_wins": 47},
    "2019-20": {"player": "Giannis Antetokounmpo", "ppg": 29.5, "rpg": 13.6, "apg": 5.6, "ws": 14.4, "team_wins": 56},
    "2018-19": {"player": "Giannis Antetokounmpo", "ppg": 27.7, "rpg": 12.5, "apg": 5.9, "ws": 14.4, "team_wins": 60},
    "2017-18": {"player": "James Harden", "ppg": 30.4, "rpg": 5.4, "apg": 8.8, "ws": 15.4, "team_wins": 65},
    "2016-17": {"player": "Russell Westbrook", "ppg": 31.6, "rpg": 10.7, "apg": 10.4, "ws": 13.1, "team_wins": 47},
    "2015-16": {"player": "Stephen Curry", "ppg": 30.1, "rpg": 5.4, "apg": 6.7, "ws": 17.9, "team_wins": 73},
}


# =============================================================================
# MVP PREDICTOR
# =============================================================================
class MVPPredictor:
    """
    Predicts MVP vote share using gradient boosting with narrative features.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.trained = False
    
    def calculate_mvp_similarity(self, player_stats: Dict) -> float:
        """
        Calculate cosine similarity to historical MVP profiles.
        Captures voter psychology by finding players who "look like" past MVPs.
        """
        # Create feature vector for player
        player_vector = np.array([
            player_stats.get("ppg", 0),
            player_stats.get("rpg", 0),
            player_stats.get("apg", 0),
            player_stats.get("ws", 0),
            player_stats.get("team_wins", 0) / 82  # Normalize to 0-1
        ]).reshape(1, -1)
        
        # Create matrix of historical MVP profiles
        mvp_vectors = []
        for season, profile in HISTORICAL_MVP_PROFILES.items():
            mvp_vectors.append([
                profile["ppg"],
                profile["rpg"],
                profile["apg"],
                profile["ws"],
                profile["team_wins"] / 82
            ])
        
        mvp_matrix = np.array(mvp_vectors)
        
        # Normalize
        if len(mvp_matrix) > 0:
            mvp_matrix_normalized = self.scaler.fit_transform(mvp_matrix)
            player_normalized = self.scaler.transform(player_vector)
            
            # Calculate similarity to each MVP season
            similarities = cosine_similarity(player_normalized, mvp_matrix_normalized)[0]
            
            # Return max similarity (closest to any MVP)
            return float(np.max(similarities))
        
        return 0.0
    
    def calculate_narrative_features(self, player_stats: Dict, 
                                      prev_season_stats: Optional[Dict] = None) -> Dict:
        """
        Calculate narrative momentum features that voters care about.
        """
        features = {}
        
        # Stat improvement year-over-year
        if prev_season_stats:
            features["ppg_improvement"] = player_stats.get("ppg", 0) - prev_season_stats.get("ppg", 0)
            features["rpg_improvement"] = player_stats.get("rpg", 0) - prev_season_stats.get("rpg", 0)
            features["apg_improvement"] = player_stats.get("apg", 0) - prev_season_stats.get("apg", 0)
        else:
            features["ppg_improvement"] = 0
            features["rpg_improvement"] = 0
            features["apg_improvement"] = 0
        
        # Team success
        features["team_wins"] = player_stats.get("team_wins", 0)
        features["team_win_pct"] = player_stats.get("team_wins", 41) / 82
        
        # Games played (durability matters)
        features["games_played"] = player_stats.get("gp", 0)
        features["games_played_pct"] = player_stats.get("gp", 0) / 82
        
        return features
    
    def prepare_features(self, player_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare all features for MVP prediction."""
        features = player_df.copy()
        
        # Calculate MVP similarity for each player
        features["mvp_similarity"] = features.apply(
            lambda row: self.calculate_mvp_similarity({
                "ppg": row.get("PTS", 0),
                "rpg": row.get("REB", 0),
                "apg": row.get("AST", 0),
                "ws": row.get("WS", 10),  # Default if not available
                "team_wins": row.get("TEAM_WINS", 41)
            }), axis=1
        )
        
        return features
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_columns: List[str]):
        """Train the MVP prediction model."""
        self.feature_columns = feature_columns
        
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        self.model.fit(X, y)
        self.trained = True
        logger.info("MVP model trained")
    
    def predict_vote_share(self, X: np.ndarray) -> np.ndarray:
        """Predict MVP vote share (0-1 scale)."""
        if not self.trained:
            raise ValueError("Model not trained")
        return self.model.predict(X)
    
    def rank_candidates(self, player_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Rank MVP candidates and return top N.
        Uses real stats-based scoring formula.
        """
        df = player_df.copy()
        
        # MVP score based on stats available from NBA API
        # Weighted formula considering:
        # - Scoring (30%): Points per game
        # - Playmaking (20%): Assists per game
        # - Rebounding (15%): Rebounds per game
        # - Defense (10%): Steals + Blocks
        # - Efficiency (10%): Plus/Minus and FG%
        # - Team Success (15%): Team win percentage
        
        pts = df.get("PTS", pd.Series([0]*len(df))).fillna(0)
        ast = df.get("AST", pd.Series([0]*len(df))).fillna(0)
        reb = df.get("REB", pd.Series([0]*len(df))).fillna(0)
        stl = df.get("STL", pd.Series([0]*len(df))).fillna(0)
        blk = df.get("BLK", pd.Series([0]*len(df))).fillna(0)
        plus_minus = df.get("PLUS_MINUS", pd.Series([0]*len(df))).fillna(0)
        fg_pct = df.get("FG_PCT", pd.Series([0.45]*len(df))).fillna(0.45)
        team_win_pct = df.get("TEAM_WIN_PCT", pd.Series([0.5]*len(df))).fillna(0.5)
        
        df["mvp_score"] = (
            pts * 1.0 +                    # Points (raw weight)
            ast * 2.0 +                    # Assists (weighted more for playmaking)
            reb * 1.0 +                    # Rebounds
            (stl + blk) * 1.5 +            # Defense
            plus_minus * 0.3 +             # Impact metric
            fg_pct * 20 +                  # Efficiency bonus
            team_win_pct * 30              # Team success (big factor for MVP)
        )
        
        # Add MVP similarity if we can calculate it
        if "mvp_similarity" not in df.columns:
            df = self.prepare_features(df)
        
        if "mvp_similarity" in df.columns:
            df["mvp_score"] = df["mvp_score"] + df["mvp_similarity"].fillna(0) * 10
        
        # Sort and return top candidates
        df = df.sort_values("mvp_score", ascending=False)
        
        # Ensure columns exist for return
        if "mvp_similarity" not in df.columns:
            df["mvp_similarity"] = 0.0
        
        return df.head(top_n)[["PLAYER_NAME", "PTS", "REB", "AST", "mvp_score", "mvp_similarity"]]
    
    def save(self, path: Path = None):
        """Save model to disk."""
        if path is None:
            path = MODELS_DIR / "mvp_predictor.joblib"
        
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "trained": self.trained
        }, path)
        logger.info(f"Saved MVP model to {path}")
    
    def load(self, path: Path = None):
        """Load model from disk."""
        if path is None:
            path = MODELS_DIR / "mvp_predictor.joblib"
        
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_columns = data["feature_columns"]
        self.trained = data["trained"]


# =============================================================================
# CLI INTERFACE
# =============================================================================
if __name__ == "__main__":
    print("Testing MVP Similarity Calculator...")
    
    predictor = MVPPredictor()
    
    # Test with a hypothetical MVP-caliber season
    test_stats = {
        "ppg": 28.5,
        "rpg": 12.0,
        "apg": 8.5,
        "ws": 15.0,
        "team_wins": 55
    }
    
    similarity = predictor.calculate_mvp_similarity(test_stats)
    print(f"MVP Similarity Score: {similarity:.3f}")
    
    # Test narrative features
    prev_stats = {"ppg": 25.0, "rpg": 10.0, "apg": 7.0}
    narrative = predictor.calculate_narrative_features(test_stats, prev_stats)
    print(f"Narrative Features: {narrative}")
