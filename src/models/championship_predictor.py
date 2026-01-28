"""
NBA ML Prediction System - Championship Predictor
==================================================
Model to predict NBA Finals winner with playoff experience features.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import xgboost as xgb
import joblib
import logging

from src.config import MODELS_DIR, NBA_TEAMS

logger = logging.getLogger(__name__)

# =============================================================================
# PLAYOFF EXPERIENCE INDEX
# =============================================================================
class PlayoffExperienceCalculator:
    """
    Calculates playoff experience index for teams.
    Teams with playoff experience perform better in postseason.
    """
    
    # Historical playoff appearances (last 5 years weight)
    PLAYOFF_HISTORY = {
        "BOS": {"appearances": 5, "finals": 2, "championships": 1},
        "MIA": {"appearances": 4, "finals": 2, "championships": 0},
        "DEN": {"appearances": 4, "finals": 1, "championships": 1},
        "GSW": {"appearances": 5, "finals": 3, "championships": 2},
        "PHX": {"appearances": 3, "finals": 1, "championships": 0},
        "MIL": {"appearances": 5, "finals": 1, "championships": 1},
        "LAL": {"appearances": 4, "finals": 1, "championships": 1},
        "DAL": {"appearances": 3, "finals": 1, "championships": 0},
        "CLE": {"appearances": 3, "finals": 0, "championships": 0},
        "OKC": {"appearances": 2, "finals": 0, "championships": 0},
    }
    
    def calculate_experience_index(self, team_abbrev: str, 
                                    core_continuity: float = 0.8) -> float:
        """
        Calculate playoff experience index.
        
        Args:
            team_abbrev: Team abbreviation
            core_continuity: How much of the playoff core remains (0-1)
        
        Returns:
            Experience index (0-1 scale)
        """
        history = self.PLAYOFF_HISTORY.get(team_abbrev, {
            "appearances": 0, "finals": 0, "championships": 0
        })
        
        # Weight factors
        appearance_weight = 0.3
        finals_weight = 0.4
        championship_weight = 0.3
        
        # Calculate raw score
        raw_score = (
            history["appearances"] / 5 * appearance_weight +
            history["finals"] / 3 * finals_weight +
            history["championships"] / 2 * championship_weight
        )
        
        # Apply core continuity discount
        adjusted_score = raw_score * core_continuity
        
        return min(adjusted_score, 1.0)


# =============================================================================
# CHAMPIONSHIP PREDICTOR
# =============================================================================
class ChampionshipPredictor:
    """
    Predicts championship probability for each team.
    """
    
    def __init__(self):
        self.model = None
        self.experience_calc = PlayoffExperienceCalculator()
        self.feature_columns = None
        self.trained = False
    
    def calculate_team_strength(self, team_stats: Dict) -> float:
        """Calculate overall team strength rating."""
        # Weighted combination of team metrics
        strength = (
            team_stats.get("win_pct", 0.5) * 30 +
            team_stats.get("net_rating", 0) * 0.5 +
            team_stats.get("elo", 1500) / 100 +
            team_stats.get("playoff_experience", 0) * 10
        )
        return strength
    
    def calculate_injury_sensitivity(self, team_stats: Dict) -> float:
        """
        Calculate how dependent team is on star players.
        High sensitivity = risky championship bet.
        """
        # Simplified: use points concentration
        top_scorer_pts = team_stats.get("top_scorer_ppg", 25)
        team_ppg = team_stats.get("team_ppg", 110)
        
        # High concentration = high sensitivity
        concentration = top_scorer_pts / team_ppg if team_ppg > 0 else 0.3
        
        return concentration
    
    def prepare_features(self, team_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for championship prediction."""
        df = team_df.copy()
        
        # Add playoff experience
        df["playoff_experience"] = df["TEAM_ABBREVIATION"].apply(
            lambda x: self.experience_calc.calculate_experience_index(x)
        )
        
        # Calculate strength rating
        df["strength_rating"] = df.apply(lambda row: self.calculate_team_strength({
            "win_pct": row.get("W_PCT", 0.5),
            "net_rating": row.get("NET_RATING", 0),
            "elo": row.get("ELO", 1500),
            "playoff_experience": row.get("playoff_experience", 0)
        }), axis=1)
        
        return df
    
    def predict_probabilities(self, team_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict championship probability for each team.
        Uses a formula-based approach if model not trained.
        """
        df = self.prepare_features(team_df)
        
        # Calculate raw championship scores
        df["champ_score"] = (
            df.get("W_PCT", 0.5) * 40 +
            df.get("NET_RATING", 0) * 2 +
            df["playoff_experience"] * 20 +
            df["strength_rating"] * 0.5
        )
        
        # Convert to probabilities (softmax-like normalization)
        total = df["champ_score"].sum()
        df["champ_probability"] = df["champ_score"] / total if total > 0 else 1/len(df)
        
        # Sort by probability
        df = df.sort_values("champ_probability", ascending=False)
        
        return df[["TEAM_ABBREVIATION", "W_PCT", "playoff_experience", 
                   "strength_rating", "champ_probability"]]
    
    def get_top_contenders(self, team_df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
        """Get top championship contenders."""
        probs = self.predict_probabilities(team_df)
        return probs.head(top_n)
    
    def simulate_playoff_bracket(self, team_df: pd.DataFrame) -> Dict:
        """
        Simple playoff bracket simulation.
        Returns predicted conference champions and finals winner.
        """
        probs = self.predict_probabilities(team_df)
        
        # Split by conference (simplified - top 8 each)
        # In reality, would use actual standings
        top_teams = probs.head(16)
        
        east_teams = top_teams.head(8)  # Simplified
        west_teams = top_teams.tail(8)
        
        # Pick conference champions (highest probability each)
        east_champ = east_teams.iloc[0]["TEAM_ABBREVIATION"]
        west_champ = west_teams.iloc[0]["TEAM_ABBREVIATION"]
        
        # Finals winner
        finals_winner = probs.iloc[0]["TEAM_ABBREVIATION"]
        
        return {
            "east_champion": east_champ,
            "west_champion": west_champ,
            "finals_winner": finals_winner,
            "champion_probability": probs.iloc[0]["champ_probability"]
        }
    
    def save(self, path: Path = None):
        """Save model to disk."""
        if path is None:
            path = MODELS_DIR / "championship_predictor.joblib"
        
        joblib.dump({
            "model": self.model,
            "feature_columns": self.feature_columns,
            "trained": self.trained
        }, path)
        logger.info(f"Saved championship model to {path}")
    
    def load(self, path: Path = None):
        """Load model from disk."""
        if path is None:
            path = MODELS_DIR / "championship_predictor.joblib"
        
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
        self.trained = data["trained"]


# =============================================================================
# CLI INTERFACE
# =============================================================================
if __name__ == "__main__":
    print("Testing Championship Predictor...")
    
    # Create sample team data
    sample_teams = pd.DataFrame({
        "TEAM_ABBREVIATION": ["BOS", "DEN", "MIL", "PHX", "GSW", "MIA", "LAL", "DAL"],
        "W_PCT": [0.68, 0.65, 0.63, 0.60, 0.58, 0.55, 0.52, 0.50],
        "NET_RATING": [8.5, 6.2, 5.1, 4.0, 3.5, 2.0, 1.5, 0.5]
    })
    
    predictor = ChampionshipPredictor()
    
    print("\nChampionship Probabilities:")
    probs = predictor.get_top_contenders(sample_teams)
    print(probs.to_string(index=False))
    
    print("\nPlayoff Bracket Simulation:")
    bracket = predictor.simulate_playoff_bracket(sample_teams)
    for k, v in bracket.items():
        print(f"  {k}: {v}")
