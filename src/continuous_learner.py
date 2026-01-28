"""
NBA ML Prediction System - Continuous Learner
==============================================
Handles incremental model updates with new game data.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import time

import pandas as pd
import numpy as np

from src.config import (
    RAW_DATA_DIR, 
    PROCESSED_DATA_DIR, 
    MODELS_DIR,
    SEASON_STRINGS,
    API_CONFIG
)
from src.data_collector import GameDataCollector, TeamDataCollector, PlayerDataCollector, CacheManager
from src.feature_engineering import FeatureGenerator
from src.preprocessing import DataPreprocessor
from src.models.game_predictor import GamePredictor, train_game_predictor
from src.live_data_collector import LiveDataCollector
from src.prediction_tracker import PredictionTracker

logger = logging.getLogger(__name__)


class ContinuousLearner:
    """
    Handles model updates with new game data.
    
    Workflow:
    1. Ingest completed games from live API
    2. Update raw data files
    3. Re-run feature engineering
    4. Retrain model (incremental or full)
    """
    
    def __init__(self):
        self.game_collector = GameDataCollector()
        self.live_collector = LiveDataCollector()
        self.prediction_tracker = PredictionTracker()
        self.feature_gen = FeatureGenerator()
        self.cache = CacheManager()
        
        # Checkpoint file for tracking last ingested game
        self.checkpoint_file = PROCESSED_DATA_DIR / "continuous_learning_checkpoint.json"
    
    def _load_checkpoint(self) -> Dict:
        """Load checkpoint of last processed game."""
        import json
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                # Ensure both fields exist for backward compatibility
                if "trained_game_ids" not in checkpoint:
                    checkpoint["trained_game_ids"] = []
                if "prediction_game_ids" not in checkpoint:
                    checkpoint["prediction_game_ids"] = checkpoint.get("last_game_ids", [])
                return checkpoint
        return {"last_game_date": None, "last_game_ids": [], "trained_game_ids": [], "prediction_game_ids": []}
    
    def _save_checkpoint(self, checkpoint: Dict):
        """Save checkpoint after processing."""
        import json
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def ingest_completed_games(self) -> int:
        """
        Fetch completed games from live API and add to training data.
        
        Returns:
            Number of new games ingested
        """
        logger.info("Checking for completed games to ingest...")
        
        # Get completed games from today
        final_games = self.live_collector.get_final_games()
        
        if not final_games:
            logger.info("No completed games to ingest")
            return 0
        
        # Load checkpoint
        checkpoint = self._load_checkpoint()
        prediction_ids = set(checkpoint.get("prediction_game_ids", checkpoint.get("last_game_ids", [])))
        
        # Filter to games not yet tracked for predictions
        new_for_predictions = [g for g in final_games if g["game_id"] not in prediction_ids]
        
        # Update prediction tracker with results (for ALL new games)
        for game in new_for_predictions:
            winner = game["home_team"] if game["home_score"] > game["away_score"] else game["away_team"]
            self.prediction_tracker.update_result(
                game_id=game["game_id"],
                actual_winner=winner,
                home_score=game["home_score"],
                away_score=game["away_score"]
            )
            prediction_ids.add(game["game_id"])
        
        # Update prediction checkpoint immediately (tracks which games have been processed for accuracy)
        checkpoint["prediction_game_ids"] = list(prediction_ids)[-100:]
        checkpoint["last_game_ids"] = checkpoint["prediction_game_ids"]  # Backward compat
        checkpoint["last_game_date"] = datetime.now().isoformat()
        self._save_checkpoint(checkpoint)
        
        if new_for_predictions:
            logger.info(f"Updated predictions for {len(new_for_predictions)} games")
        
        # For TRAINING, check against trained_game_ids (different from prediction tracking)
        trained_ids = set(checkpoint.get("trained_game_ids", []))
        new_for_training = [g for g in final_games if g["game_id"] not in trained_ids]
        
        if not new_for_training:
            logger.info("All completed games already trained on")
            return 0
        
        logger.info(f"Found {len(new_for_training)} new games for training")
        
        # Append new games to raw data for training
        self._append_games_to_raw_data(new_for_training)
        
        # Store the game IDs to be marked as trained AFTER successful training
        # This is stored temporarily; mark_games_as_trained() will persist them
        self._pending_training_games = [g["game_id"] for g in new_for_training]
        
        logger.info(f"Ingested {len(new_for_training)} new games for training")
        return len(new_for_training)
    
    def mark_games_as_trained(self, game_ids: List[str] = None):
        """
        Mark games as successfully trained on. Only call after training succeeds.
        
        Args:
            game_ids: List of game IDs to mark as trained. If None, uses pending games.
        """
        if game_ids is None:
            game_ids = getattr(self, '_pending_training_games', [])
        
        if not game_ids:
            return
        
        checkpoint = self._load_checkpoint()
        trained_ids = set(checkpoint.get("trained_game_ids", []))
        trained_ids.update(game_ids)
        
        checkpoint["trained_game_ids"] = list(trained_ids)[-100:]  # Keep last 100
        self._save_checkpoint(checkpoint)
        
        logger.info(f"Marked {len(game_ids)} games as trained")
        self._pending_training_games = []
    
    def _append_games_to_raw_data(self, games: List[Dict]):
        """Append new game data to raw parquet files."""
        try:
            # Load existing games
            games_file = RAW_DATA_DIR / "all_games.parquet"
            if games_file.exists():
                existing_df = pd.read_parquet(games_file)
            else:
                existing_df = pd.DataFrame()
            
            # Convert new games to DataFrame
            new_rows = []
            for game in games:
                new_rows.append({
                    "GAME_ID": game["game_id"],
                    "GAME_DATE": game.get("game_date", ""),
                    "HOME_TEAM_ID": game["home_team_id"],
                    "VISITOR_TEAM_ID": game["away_team_id"],
                    "HOME_TEAM_ABBREVIATION": game["home_team"],
                    "VISITOR_TEAM_ABBREVIATION": game["away_team"],
                    "PTS_home": game["home_score"],
                    "PTS_away": game["away_score"],
                    "HOME_TEAM_WINS": 1 if game["home_score"] > game["away_score"] else 0,
                    "SEASON_ID": self._get_current_season_id(),
                })
            
            new_df = pd.DataFrame(new_rows)
            
            # Append and deduplicate
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["GAME_ID"], keep="last")
            
            # Save
            combined_df.to_parquet(games_file, index=False)
            logger.info(f"Updated raw games data: {len(combined_df)} total games")
            
        except Exception as e:
            logger.error(f"Failed to append games to raw data: {e}")
    
    def _get_current_season_id(self) -> str:
        """Get current NBA season ID."""
        now = datetime.now()
        year = now.year if now.month >= 10 else now.year - 1
        return f"2{year}"  # e.g., "22024" for 2024-25 season
    
    def update_features(self) -> bool:
        """
        Re-run feature engineering with updated data.
        
        Returns:
            True if successful
        """
        logger.info("Updating features with new data...")
        
        try:
            # Re-run feature generation
            self.feature_gen.process_all_data(force_regenerate=True)
            logger.info("Feature update complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update features: {e}")
            return False
    
    def retrain_model(self, incremental: bool = True) -> Dict:
        """
        Retrain the game prediction model.
        
        Args:
            incremental: If True, use warm start from existing model.
                        If False, train from scratch.
        
        Returns:
            Dict with training metrics
        """
        logger.info(f"Retraining model (incremental={incremental})...")
        
        try:
            import joblib
            
            # Load dataset
            dataset_path = PROCESSED_DATA_DIR / "game_dataset.joblib"
            if not dataset_path.exists():
                logger.error("Dataset not found. Run preprocessing first.")
                return {"error": "Dataset not found"}
            
            dataset = joblib.load(dataset_path)
            
            # Train model
            metrics = train_game_predictor(dataset)
            
            logger.info(f"Model retrained. Accuracy: {metrics.get('test_accuracy', 0):.2%}")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to retrain model: {e}")
            return {"error": str(e)}
    
    def run_update_cycle(self, retrain: bool = True) -> Dict:
        """
        Full update cycle: ingest -> features -> retrain.
        
        Args:
            retrain: Whether to retrain model after updating data
        
        Returns:
            Dict with cycle results
        """
        logger.info("Starting continuous learning update cycle...")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "games_ingested": 0,
            "features_updated": False,
            "model_retrained": False,
            "metrics": {}
        }
        
        # Step 1: Ingest completed games
        games_ingested = self.ingest_completed_games()
        results["games_ingested"] = games_ingested
        
        if games_ingested == 0:
            logger.info("No new games to process, skipping update")
            return results
        
        # Step 2: Update features
        features_updated = self.update_features()
        results["features_updated"] = features_updated
        
        if not features_updated:
            logger.warning("Feature update failed, skipping retrain")
            return results
        
        # Step 3: Retrain model (if requested)
        if retrain:
            # Rebuild dataset
            preprocessor = DataPreprocessor()
            preprocessor.build_dataset()
            
            # Retrain
            metrics = self.retrain_model(incremental=True)
            results["model_retrained"] = "error" not in metrics
            results["metrics"] = metrics
            
            # Only mark games as trained AFTER successful training
            if results["model_retrained"]:
                self.mark_games_as_trained()
                logger.info("Training successful - games marked as trained")
            else:
                logger.warning("Training failed - games will be retried next cycle")
        
        logger.info("Update cycle complete")
        return results


# =============================================================================
# CLI INTERFACE
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="NBA Continuous Learning System")
    parser.add_argument("--ingest", action="store_true", help="Ingest completed games only")
    parser.add_argument("--update", action="store_true", help="Full update cycle")
    parser.add_argument("--no-retrain", action="store_true", help="Skip model retraining")
    
    args = parser.parse_args()
    
    learner = ContinuousLearner()
    
    if args.ingest:
        print("\n=== Ingesting Completed Games ===\n")
        count = learner.ingest_completed_games()
        print(f"Ingested {count} new games")
    
    elif args.update:
        print("\n=== Running Update Cycle ===\n")
        results = learner.run_update_cycle(retrain=not args.no_retrain)
        print(f"Games ingested: {results['games_ingested']}")
        print(f"Features updated: {results['features_updated']}")
        print(f"Model retrained: {results['model_retrained']}")
        if results['metrics']:
            print(f"Test accuracy: {results['metrics'].get('test_accuracy', 'N/A')}")
    
    else:
        print("Use --ingest or --update")
        print("\nUsage:")
        print("  python -m src.continuous_learner --ingest")
        print("  python -m src.continuous_learner --update")
        print("  python -m src.continuous_learner --update --no-retrain")
