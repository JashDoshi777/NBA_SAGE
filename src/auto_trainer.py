"""
NBA ML Prediction System - Auto Training Scheduler
===================================================
Background scheduler that automatically trains the model on new game data.
Runs within the Streamlit app or as a standalone service.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Optional
import atexit

logger = logging.getLogger(__name__)


class AutoTrainer:
    """
    Automatic model training scheduler.
    
    Runs background tasks to:
    1. Ingest completed games every hour
    2. Retrain the model daily
    3. Update prediction results after games
    """
    
    _instance: Optional['AutoTrainer'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern - only one auto trainer instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Track last run times
        self._last_ingest = None
        self._last_retrain = None
        self._last_results_check = None
        
        # Intervals (in seconds)
        self.INGEST_INTERVAL = 3600  # 1 hour
        self.RETRAIN_INTERVAL = 86400  # 24 hours (daily)
        self.RESULTS_CHECK_INTERVAL = 1800  # 30 minutes
        
        logger.info("AutoTrainer initialized")
    
    def start(self):
        """Start the background training scheduler."""
        if self._running:
            logger.info("AutoTrainer already running")
            return
        
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        # Register cleanup on exit
        atexit.register(self.stop)
        
        logger.info("AutoTrainer started - background training enabled")
    
    def stop(self):
        """Stop the background scheduler."""
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        
        logger.info("AutoTrainer stopped")
    
    def _run_loop(self):
        """Main background loop - checks for tasks to run."""
        logger.info("AutoTrainer loop started")
        
        while not self._stop_event.is_set():
            try:
                now = datetime.now()
                
                # Check and update prediction results (every 30 min)
                if self._should_run(self._last_results_check, self.RESULTS_CHECK_INTERVAL):
                    self._check_results()
                    self._last_results_check = now
                
                # Ingest completed games (every hour)
                if self._should_run(self._last_ingest, self.INGEST_INTERVAL):
                    self._ingest_games()
                    self._last_ingest = now
                
                # Retrain model only after all daily games are complete
                # NBA games typically end by 1 AM ET, so we retrain at 4 AM ET (safe window)
                # 4 AM ET = 1:30 PM IST
                if self._should_run(self._last_retrain, self.RETRAIN_INTERVAL):
                    if self._all_daily_games_complete():
                        self._retrain_model()
                        self._last_retrain = now
                    else:
                        logger.info("AutoTrainer: Waiting for all games to complete before retrain")
                
            except Exception as e:
                logger.error(f"AutoTrainer error: {e}")
            
            # Sleep for 5 minutes between checks
            self._stop_event.wait(300)
    
    def _all_daily_games_complete(self) -> bool:
        """Check if all of today's games have completed."""
        try:
            from src.live_data_collector import LiveDataCollector
            collector = LiveDataCollector()
            
            # Get live games - if any are still in progress, don't retrain
            live_games = collector.get_live_games()
            if live_games:
                logger.info(f"AutoTrainer: {len(live_games)} games still in progress")
                return False
            
            # Get upcoming games - if any haven't started, don't retrain yet
            upcoming = collector.get_upcoming_games()
            if upcoming:
                logger.info(f"AutoTrainer: {len(upcoming)} games haven't started yet")
                return False
            
            # All games completed (or no games today)
            return True
            
        except Exception as e:
            logger.warning(f"Could not check game status: {e}")
            # Default to checking time - after 4 AM ET (safe window)
            hour = datetime.now().hour
            # 4 AM ET ≈ 1:30 PM IST, 9 AM UTC
            return hour >= 4 or hour < 12  # Between 4 AM and noon
    
    def _should_run(self, last_run: Optional[datetime], interval: int) -> bool:
        """Check if enough time has passed since last run."""
        if last_run is None:
            return True
        return (datetime.now() - last_run).total_seconds() >= interval
    
    def _check_results(self):
        """Check completed games and update prediction results."""
        logger.info("AutoTrainer: Checking prediction results...")
        try:
            from src.prediction_pipeline import PredictionPipeline
            pipeline = PredictionPipeline()
            updated = pipeline.check_prediction_results()
            logger.info(f"AutoTrainer: Updated {len(updated)} prediction results")
        except Exception as e:
            logger.error(f"AutoTrainer: Failed to check results: {e}")
    
    def _ingest_games(self):
        """Ingest completed games into training data."""
        logger.info("AutoTrainer: Ingesting completed games...")
        try:
            from src.continuous_learner import ContinuousLearner
            learner = ContinuousLearner()
            count = learner.ingest_completed_games()
            logger.info(f"AutoTrainer: Ingested {count} new games")
        except Exception as e:
            logger.error(f"AutoTrainer: Failed to ingest games: {e}")
    
    def _retrain_model(self):
        """Full model retrain cycle."""
        logger.info("AutoTrainer: Starting daily model retrain...")
        try:
            from src.continuous_learner import ContinuousLearner
            learner = ContinuousLearner()
            results = learner.run_update_cycle(retrain=True)
            
            if results.get("model_retrained"):
                accuracy = results.get("metrics", {}).get("test_accuracy", 0)
                logger.info(f"AutoTrainer: Model retrained! Accuracy: {accuracy:.2%}")
            else:
                logger.info("AutoTrainer: No new data to retrain on")
                
        except Exception as e:
            logger.error(f"AutoTrainer: Failed to retrain model: {e}")
    
    def get_status(self) -> dict:
        """Get current auto-trainer status."""
        return {
            "running": self._running,
            "last_ingest": self._last_ingest.isoformat() if self._last_ingest else None,
            "last_retrain": self._last_retrain.isoformat() if self._last_retrain else None,
            "last_results_check": self._last_results_check.isoformat() if self._last_results_check else None,
            "next_ingest_in": self._time_until_next(self._last_ingest, self.INGEST_INTERVAL),
            "next_retrain_in": self._time_until_next(self._last_retrain, self.RETRAIN_INTERVAL),
        }
    
    def _time_until_next(self, last_run: Optional[datetime], interval: int) -> str:
        """Human-readable time until next run."""
        if last_run is None:
            return "Soon"
        
        elapsed = (datetime.now() - last_run).total_seconds()
        remaining = max(0, interval - elapsed)
        
        if remaining < 60:
            return f"{int(remaining)}s"
        elif remaining < 3600:
            return f"{int(remaining / 60)}m"
        else:
            return f"{int(remaining / 3600)}h {int((remaining % 3600) / 60)}m"
    
    def force_ingest(self):
        """Force an immediate game ingestion."""
        threading.Thread(target=self._ingest_games, daemon=True).start()
    
    def run_training_cycle(self) -> dict:
        """
        Public method for external training requests (e.g., from server.py).
        Uses ContinuousLearner for the actual training.
        
        Returns:
            Dict with 'success', 'accuracy', 'skipped', and optionally 'error' keys
        """
        try:
            from src.continuous_learner import ContinuousLearner
            
            logger.info("AutoTrainer: Running training cycle via ContinuousLearner...")
            learner = ContinuousLearner()
            result = learner.run_update_cycle(retrain=True)
            
            accuracy = result.get("metrics", {}).get("test_accuracy", 0)
            success = result.get("model_retrained", False)
            games_ingested = result.get("games_ingested", 0)
            
            if success:
                logger.info(f"AutoTrainer: Training cycle complete. Accuracy: {accuracy:.2%}")
                return {
                    "success": True,
                    "skipped": False,
                    "accuracy": accuracy,
                    "games_ingested": games_ingested
                }
            elif games_ingested == 0:
                # No new data - this is NOT an error, just nothing to train on
                logger.info(f"AutoTrainer: No new games available for training")
                return {
                    "success": True,  # Not a failure
                    "skipped": True,  # Just skipped
                    "reason": "No new games to train on",
                    "accuracy": accuracy,
                    "games_ingested": 0
                }
            else:
                # Games ingested but training failed
                logger.warning(f"AutoTrainer: Training failed after ingesting {games_ingested} games")
                return {
                    "success": False,
                    "skipped": False,
                    "error": "Training failed - will retry next cycle",
                    "accuracy": accuracy,
                    "games_ingested": games_ingested
                }
        except Exception as e:
            logger.error(f"AutoTrainer: Training cycle failed: {e}")
            return {"success": False, "skipped": False, "error": str(e), "accuracy": 0}
    
    def force_retrain(self):
        """Force an immediate model retrain."""
        threading.Thread(target=self._retrain_model, daemon=True).start()


# Global instance
auto_trainer = AutoTrainer()


# =============================================================================
# CLI INTERFACE
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="NBA Auto Training Scheduler")
    parser.add_argument("--start", action="store_true", help="Start the scheduler")
    parser.add_argument("--status", action="store_true", help="Show scheduler status")
    
    args = parser.parse_args()
    
    if args.status:
        status = auto_trainer.get_status()
        print("\n=== Auto Trainer Status ===\n")
        print(f"Running: {'Yes ✓' if status['running'] else 'No'}")
        print(f"Last Ingest: {status['last_ingest'] or 'Never'}")
        print(f"Last Retrain: {status['last_retrain'] or 'Never'}")
        print(f"Next Ingest In: {status['next_ingest_in']}")
        print(f"Next Retrain In: {status['next_retrain_in']}")
    
    elif args.start:
        print("\n=== Starting Auto Trainer ===\n")
        print("Background training enabled!")
        print("- Checks prediction results every 30 minutes")
        print("- Ingests completed games every 1 hour")
        print("- Retrains model every 24 hours")
        print("\nPress Ctrl+C to stop...\n")
        
        auto_trainer.start()
        
        try:
            while True:
                time.sleep(60)
                status = auto_trainer.get_status()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Running... Next ingest: {status['next_ingest_in']}, Next retrain: {status['next_retrain_in']}")
        except KeyboardInterrupt:
            print("\nStopping...")
            auto_trainer.stop()
    
    else:
        print("Use --start to begin auto training or --status to check status")
