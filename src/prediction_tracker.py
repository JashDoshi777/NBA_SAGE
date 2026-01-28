"""
NBA ML Prediction System - Prediction Tracker
==============================================
Tracks predictions and measures accuracy using ChromaDB Cloud with local fallback.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
import json
import hashlib
from pathlib import Path

from src.config import CHROMADB_CONFIG, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


class PredictionTracker:
    """
    Tracks predictions and measures accuracy over time.
    
    Uses ChromaDB Cloud if available, otherwise falls back to local JSON storage.
    Stores predictions before games and updates with results after completion.
    Provides accuracy statistics by team, confidence level, and over time.
    """
    
    def __init__(self):
        """Initialize storage - try ChromaDB Cloud, fallback to local JSON."""
        self.collection = None
        self.client = None
        self._use_local = False
        self._local_file = PROCESSED_DATA_DIR / "predictions_local.json"
        self._local_data: List[Dict] = []
        
        # Try ChromaDB Cloud first
        try:
            import chromadb
            
            # Try CloudClient (the official way for Chroma Cloud)
            self.client = chromadb.CloudClient(
                tenant=CHROMADB_CONFIG.tenant,
                database=CHROMADB_CONFIG.database,
                api_key=CHROMADB_CONFIG.api_key,
            )
            
            # Get or create collection for predictions
            self.collection = self.client.get_or_create_collection(
                name=CHROMADB_CONFIG.collection_name,
                metadata={"description": "NBA game predictions with results"}
            )
            
            logger.info("Connected to ChromaDB Cloud successfully")
            
        except Exception as e:
            logger.warning(f"ChromaDB Cloud unavailable ({e}), using local JSON storage")
            self._use_local = True
            self._load_local_data()
    
    def _generate_id(self, game_id: str, prediction_date: str) -> str:
        """Generate unique ID for a prediction."""
        return hashlib.md5(f"{game_id}_{prediction_date}".encode()).hexdigest()
    
    def _load_local_data(self):
        """Load predictions from local JSON file."""
        if self._local_file.exists():
            try:
                with open(self._local_file, 'r') as f:
                    self._local_data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load local data: {e}")
                self._local_data = []
        else:
            self._local_data = []
    
    def _save_local_data(self):
        """Save predictions to local JSON file."""
        try:
            with open(self._local_file, 'w') as f:
                json.dump(self._local_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save local data: {e}")
    
    def _find_local_prediction(self, game_id: str) -> Optional[int]:
        """Find index of prediction by game_id in local data."""
        for i, pred in enumerate(self._local_data):
            if pred.get("game_id") == game_id:
                return i
        return None
    
    def save_prediction(self, game_id: str, prediction: Dict) -> bool:
        """
        Store a prediction before game starts.
        
        Args:
            game_id: NBA game ID
            prediction: Dict with home_team, away_team, predicted_winner,
                       home_win_prob, confidence, etc.
        
        Returns:
            True if saved successfully
        """
        now = datetime.now().isoformat()
        doc_id = self._generate_id(game_id, now[:10])
        
        # Prepare metadata
        metadata = {
            "id": doc_id,
            "game_id": game_id,
            "game_date": prediction.get("game_date", now[:10]),
            "home_team": prediction.get("home_team", ""),
            "away_team": prediction.get("away_team", ""),
            "predicted_winner": prediction.get("predicted_winner", ""),
            "home_win_prob": float(prediction.get("home_win_probability", 0.5)),
            "away_win_prob": float(prediction.get("away_win_probability", 0.5)),
            "confidence": prediction.get("confidence", "medium"),
            "home_elo": float(prediction.get("home_elo", 1500)),
            "away_elo": float(prediction.get("away_elo", 1500)),
            "actual_winner": "",  # Empty until game completes
            "is_correct": -1,  # -1 = pending, 0 = wrong, 1 = correct
            "created_at": now,
            "updated_at": now,
        }
        
        # Use local storage if ChromaDB not available
        if self._use_local:
            try:
                # Check if exists and update, otherwise append
                idx = self._find_local_prediction(game_id)
                if idx is not None:
                    self._local_data[idx] = metadata
                else:
                    self._local_data.append(metadata)
                self._save_local_data()
                logger.info(f"Saved prediction for game {game_id} (local)")
                return True
            except Exception as e:
                logger.error(f"Failed to save prediction locally: {e}")
                return False
        
        # Use ChromaDB Cloud
        if not self.collection:
            logger.warning("ChromaDB not available, prediction not saved")
            return False
        
        try:
            # Document text for semantic search
            doc_text = (
                f"NBA Game: {prediction.get('away_team')} @ {prediction.get('home_team')} "
                f"on {metadata['game_date']}. "
                f"Predicted winner: {metadata['predicted_winner']} "
                f"with {metadata['confidence']} confidence "
                f"({metadata['home_win_prob']:.1%} home win probability)"
            )
            
            # Upsert (update if exists, insert if not)
            self.collection.upsert(
                ids=[doc_id],
                documents=[doc_text],
                metadatas=[metadata]
            )
            
            logger.info(f"Saved prediction for game {game_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save prediction: {e}")
            return False
    
    def update_result(self, game_id: str, actual_winner: str, 
                     home_score: int = 0, away_score: int = 0) -> bool:
        """
        Update prediction with actual game result.
        
        Args:
            game_id: NBA game ID
            actual_winner: Team tricode of actual winner
            home_score: Final home score
            away_score: Final away score
        
        Returns:
            True if updated successfully
        """
        # Handle local storage
        if self._use_local:
            try:
                idx = self._find_local_prediction(game_id)
                if idx is None:
                    logger.warning(f"No prediction found for game {game_id}")
                    return False
                
                pred = self._local_data[idx]
                predicted_winner = pred.get("predicted_winner", "")
                is_correct = 1 if predicted_winner == actual_winner else 0
                
                pred["actual_winner"] = actual_winner
                pred["is_correct"] = is_correct
                pred["home_score"] = home_score
                pred["away_score"] = away_score
                pred["updated_at"] = datetime.now().isoformat()
                
                self._local_data[idx] = pred
                self._save_local_data()
                
                result_text = "CORRECT ✓" if is_correct else "WRONG ✗"
                logger.info(f"Updated result for game {game_id}: {result_text} (local)")
                return True
                
            except Exception as e:
                logger.error(f"Failed to update result locally: {e}")
                return False
        
        # Handle ChromaDB Cloud
        if not self.collection:
            return False
        
        try:
            # Find the prediction for this game
            results = self.collection.get(
                where={"game_id": game_id},
                include=["metadatas", "documents"]
            )
            
            if not results["ids"]:
                logger.warning(f"No prediction found for game {game_id}")
                return False
            
            doc_id = results["ids"][0]
            metadata = results["metadatas"][0]
            
            # Check if prediction was correct
            predicted_winner = metadata.get("predicted_winner", "")
            is_correct = 1 if predicted_winner == actual_winner else 0
            
            # Update metadata
            metadata["actual_winner"] = actual_winner
            metadata["is_correct"] = is_correct
            metadata["home_score"] = home_score
            metadata["away_score"] = away_score
            metadata["updated_at"] = datetime.now().isoformat()
            
            # Update document text
            result_text = "CORRECT ✓" if is_correct else "WRONG ✗"
            doc_text = (
                f"NBA Game: {metadata['away_team']} @ {metadata['home_team']}. "
                f"Predicted: {predicted_winner}, Actual: {actual_winner}. "
                f"Result: {result_text}"
            )
            
            self.collection.update(
                ids=[doc_id],
                documents=[doc_text],
                metadatas=[metadata]
            )
            
            logger.info(f"Updated result for game {game_id}: {result_text}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update result: {e}")
            return False
    
    def get_prediction(self, game_id: str) -> Optional[Dict]:
        """Get prediction for a specific game."""
        # Handle local storage
        if self._use_local:
            idx = self._find_local_prediction(game_id)
            if idx is not None:
                return self._local_data[idx]
            return None
        
        # Handle ChromaDB
        if not self.collection:
            return None
        
        try:
            results = self.collection.get(
                where={"game_id": game_id},
                include=["metadatas"]
            )
            
            if results["ids"]:
                return results["metadatas"][0]
            return None
            
        except Exception as e:
            logger.error(f"Failed to get prediction: {e}")
            return None
    
    def get_recent_predictions(self, n: int = 20) -> List[Dict]:
        """Get N most recent predictions with results."""
        # Handle local storage
        if self._use_local:
            predictions = sorted(
                self._local_data,
                key=lambda x: x.get("created_at", ""),
                reverse=True
            )
            return predictions[:n]
        
        # Handle ChromaDB
        if not self.collection:
            return []
        
        try:
            # Get all predictions and sort by date
            results = self.collection.get(
                include=["metadatas"]
            )
            
            if not results["ids"]:
                return []
            
            predictions = results["metadatas"]
            
            # Sort by created_at descending
            predictions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
            return predictions[:n]
            
        except Exception as e:
            logger.error(f"Failed to get recent predictions: {e}")
            return []
    
    def _calculate_accuracy_from_predictions(self, predictions: List[Dict]) -> Dict:
        """Calculate accuracy stats from a list of predictions."""
        if not predictions:
            return {
                "total_predictions": 0,
                "completed_games": 0,
                "correct_predictions": 0,
                "overall_accuracy": 0.0,
                "by_confidence": {},
                "by_team": {},
            }
        
        # Filter to completed games only
        completed = [p for p in predictions if p.get("is_correct", -1) >= 0]
        correct = [p for p in completed if p.get("is_correct") == 1]
        
        # By confidence level
        confidence_stats = {}
        for conf in ["high", "medium", "low"]:
            conf_preds = [p for p in completed if p.get("confidence") == conf]
            conf_correct = [p for p in conf_preds if p.get("is_correct") == 1]
            if conf_preds:
                confidence_stats[conf] = {
                    "total": len(conf_preds),
                    "correct": len(conf_correct),
                    "accuracy": len(conf_correct) / len(conf_preds)
                }
        
        # By team predicted
        team_stats = {}
        for pred in completed:
            team = pred.get("predicted_winner", "")
            if team not in team_stats:
                team_stats[team] = {"total": 0, "correct": 0}
            team_stats[team]["total"] += 1
            if pred.get("is_correct") == 1:
                team_stats[team]["correct"] += 1
        
        for team in team_stats:
            total = team_stats[team]["total"]
            team_stats[team]["accuracy"] = team_stats[team]["correct"] / total if total > 0 else 0
        
        return {
            "total_predictions": len(predictions),
            "completed_games": len(completed),
            "correct_predictions": len(correct),
            "overall_accuracy": len(correct) / len(completed) if completed else 0.0,
            "by_confidence": confidence_stats,
            "by_team": team_stats,
        }
    
    def get_accuracy_stats(self) -> Dict:
        """
        Calculate comprehensive accuracy statistics.
        
        Returns:
            Dict with overall accuracy, by confidence, by team
        """
        # Handle local storage
        if self._use_local:
            return self._calculate_accuracy_from_predictions(self._local_data)
        
        # Handle ChromaDB
        if not self.collection:
            return {
                "total_predictions": 0,
                "completed_games": 0,
                "correct_predictions": 0,
                "overall_accuracy": 0.0,
                "by_confidence": {},
                "by_team": {},
            }
        
        try:
            results = self.collection.get(include=["metadatas"])
            
            if not results["ids"]:
                return {
                    "total_predictions": 0,
                    "completed_games": 0,
                    "correct_predictions": 0,
                    "overall_accuracy": 0.0,
                    "by_confidence": {},
                    "by_team": {},
                }
            
            predictions = results["metadatas"]
            
            # Filter to completed games only
            completed = [p for p in predictions if p.get("is_correct", -1) >= 0]
            correct = [p for p in completed if p.get("is_correct") == 1]
            
            # By confidence level
            confidence_stats = {}
            for conf in ["high", "medium", "low"]:
                conf_preds = [p for p in completed if p.get("confidence") == conf]
                conf_correct = [p for p in conf_preds if p.get("is_correct") == 1]
                if conf_preds:
                    confidence_stats[conf] = {
                        "total": len(conf_preds),
                        "correct": len(conf_correct),
                        "accuracy": len(conf_correct) / len(conf_preds)
                    }
            
            # By team predicted
            team_stats = {}
            for pred in completed:
                team = pred.get("predicted_winner", "")
                if team not in team_stats:
                    team_stats[team] = {"total": 0, "correct": 0}
                team_stats[team]["total"] += 1
                if pred.get("is_correct") == 1:
                    team_stats[team]["correct"] += 1
            
            for team in team_stats:
                total = team_stats[team]["total"]
                team_stats[team]["accuracy"] = team_stats[team]["correct"] / total if total > 0 else 0
            
            return {
                "total_predictions": len(predictions),
                "completed_games": len(completed),
                "correct_predictions": len(correct),
                "overall_accuracy": len(correct) / len(completed) if completed else 0.0,
                "by_confidence": confidence_stats,
                "by_team": team_stats,
            }
            
        except Exception as e:
            logger.error(f"Failed to get accuracy stats: {e}")
            return {
                "total_predictions": 0,
                "completed_games": 0,
                "correct_predictions": 0,
                "overall_accuracy": 0.0,
                "by_confidence": {},
                "by_team": {},
                "error": str(e)
            }
    
    def get_pending_predictions(self) -> List[Dict]:
        """Get predictions for games not yet completed."""
        if not self.collection:
            return []
        
        try:
            results = self.collection.get(
                where={"is_correct": -1},
                include=["metadatas"]
            )
            
            return results.get("metadatas", [])
            
        except Exception as e:
            logger.error(f"Failed to get pending predictions: {e}")
            return []


# =============================================================================
# CLI INTERFACE
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    tracker = PredictionTracker()
    
    print("\n=== Prediction Tracker Stats ===\n")
    
    stats = tracker.get_accuracy_stats()
    print(f"Total Predictions: {stats['total_predictions']}")
    print(f"Completed Games: {stats['completed_games']}")
    print(f"Correct Predictions: {stats['correct_predictions']}")
    print(f"Overall Accuracy: {stats['overall_accuracy']:.1%}")
    
    if stats['by_confidence']:
        print("\nBy Confidence Level:")
        for conf, data in stats['by_confidence'].items():
            print(f"  {conf.upper()}: {data['correct']}/{data['total']} ({data['accuracy']:.1%})")
    
    print("\n=== Recent Predictions ===\n")
    recent = tracker.get_recent_predictions(5)
    for pred in recent:
        status = "✓" if pred.get("is_correct") == 1 else "✗" if pred.get("is_correct") == 0 else "⏳"
        print(f"{status} {pred.get('away_team')} @ {pred.get('home_team')} - Predicted: {pred.get('predicted_winner')}")
