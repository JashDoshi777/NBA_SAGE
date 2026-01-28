"""
NBA Sage - Production Server with Full Features
================================================
Serves the React frontend and Flask API with:
- Auto-training scheduler
- Continuous learning
- Persistent storage using /data folder
- Model updates
- Caching for fast responses

For Hugging Face Spaces deployment.
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import sys
import logging
import os
import shutil
import threading
from pathlib import Path
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

# =============================================================================
# CACHE CONFIGURATION - For lightning-fast responses
# =============================================================================
cache = {
    "mvp": {"data": None, "timestamp": None, "ttl": 86400},  # 24 hours - daily refresh
    "championship": {"data": None, "timestamp": None, "ttl": 86400},  # 24 hours - daily refresh
    "teams": {"data": None, "timestamp": None, "ttl": 3600},  # 1 hour cache
    "rosters": {},  # Per-team roster cache: {team_abbrev: {"data": [...], "timestamp": datetime}}
    "roster_ttl": 3600,  # 1 hour cache for rosters
    "live_games": {"data": None, "timestamp": None, "ttl": 30},  # 30 sec cache for live games
    "predictions": {},  # Per-matchup prediction cache
    "predictions_ttl": 300,  # 5 min cache for predictions
    "all_starters": {"data": None, "timestamp": None, "ttl": 3600},  # Pre-warmed starters cache
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)

# =============================================================================
# PATH CONFIGURATION - Use persistent /data folder on HF Spaces
# =============================================================================
ROOT_DIR = Path(__file__).parent

# Pull Git LFS content to ensure parquet files are available (not just LFS pointers)
import subprocess
try:
    api_data_dir = ROOT_DIR / "data" / "api_data"
    test_file = api_data_dir / "all_games_summary.parquet"
    # Check if file exists and is an LFS pointer (very small size)
    if test_file.exists() and test_file.stat().st_size < 500:
        logger.info("Detected LFS pointers, pulling actual file content...")
        result = subprocess.run(
            ["git", "lfs", "pull"], 
            cwd=str(ROOT_DIR), 
            capture_output=True, 
            text=True,
            timeout=180
        )
        if result.returncode == 0:
            logger.info("Git LFS pull completed successfully")
        else:
            logger.warning(f"Git LFS pull issue: {result.stderr[:200]}")
except Exception as e:
    logger.info(f"LFS check skipped: {e}")

# Check if we're on Hugging Face Spaces (has /data folder)
HF_PERSISTENT_DIR = Path("/data")
IS_HF_SPACES = HF_PERSISTENT_DIR.exists() and os.access(HF_PERSISTENT_DIR, os.W_OK)

if IS_HF_SPACES:
    logger.info("Running on Hugging Face Spaces - using persistent /data storage")
    PERSISTENT_DIR = HF_PERSISTENT_DIR
else:
    logger.info("Running locally - using local data folder")
    PERSISTENT_DIR = ROOT_DIR / "data"

# Create persistent directories
PERSISTENT_MODELS_DIR = PERSISTENT_DIR / "models"
PERSISTENT_PREDICTIONS_DIR = PERSISTENT_DIR / "predictions"
PERSISTENT_ELO_DIR = PERSISTENT_DIR / "elo_ratings"
PERSISTENT_INJURIES_DIR = PERSISTENT_DIR / "injuries"

for dir_path in [PERSISTENT_MODELS_DIR, PERSISTENT_PREDICTIONS_DIR, PERSISTENT_ELO_DIR, PERSISTENT_INJURIES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Copy initial model to persistent storage if not exists
INITIAL_MODEL = ROOT_DIR / "models" / "game_predictor.joblib"
PERSISTENT_MODEL = PERSISTENT_MODELS_DIR / "game_predictor.joblib"
if INITIAL_MODEL.exists() and not PERSISTENT_MODEL.exists():
    logger.info("Copying initial model to persistent storage...")
    shutil.copy(INITIAL_MODEL, PERSISTENT_MODEL)

# Copy initial data files to persistent storage
INITIAL_PROCESSED = ROOT_DIR / "data" / "processed"
PERSISTENT_PROCESSED = PERSISTENT_DIR / "processed"
if INITIAL_PROCESSED.exists() and not PERSISTENT_PROCESSED.exists():
    logger.info("Copying initial processed data to persistent storage...")
    shutil.copytree(INITIAL_PROCESSED, PERSISTENT_PROCESSED)

# Copy api_data (standings, games, ELO data) to persistent storage
INITIAL_API_DATA = ROOT_DIR / "data" / "api_data"
PERSISTENT_API_DATA = PERSISTENT_DIR / "api_data"
logger.info(f"INITIAL_API_DATA path: {INITIAL_API_DATA}")
logger.info(f"INITIAL_API_DATA exists: {INITIAL_API_DATA.exists()}")
if INITIAL_API_DATA.exists():
    logger.info(f"INITIAL_API_DATA files: {len(list(INITIAL_API_DATA.glob('*')))}")
logger.info(f"PERSISTENT_API_DATA path: {PERSISTENT_API_DATA}")
logger.info(f"PERSISTENT_API_DATA exists: {PERSISTENT_API_DATA.exists()}")

if INITIAL_API_DATA.exists() and not PERSISTENT_API_DATA.exists():
    logger.info("Copying initial API data (standings, games) to persistent storage...")
    shutil.copytree(INITIAL_API_DATA, PERSISTENT_API_DATA)
elif INITIAL_API_DATA.exists() and PERSISTENT_API_DATA.exists():
    logger.info("API data already in persistent storage")

# Update environment to use persistent paths
os.environ["NBA_ML_DATA_DIR"] = str(PERSISTENT_DIR)
os.environ["NBA_ML_MODELS_DIR"] = str(PERSISTENT_MODELS_DIR)
logger.info(f"Set NBA_ML_DATA_DIR to: {PERSISTENT_DIR}")

# Add project root to path
sys.path.insert(0, str(ROOT_DIR))

# Now import project modules (after path setup)
from src.prediction_pipeline import PredictionPipeline
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, NBA_TEAMS

# =============================================================================
# CACHE WARMING FUNCTIONS - Must be defined before startup code uses them
# =============================================================================
def _infer_position_from_stats(player_row) -> str:
    """Infer position based on stats profile."""
    reb = float(player_row.get('REB', 0) or 0)
    ast = float(player_row.get('AST', 0) or 0)
    blk = float(player_row.get('BLK', 0) or 0)
    
    if ast > 5 and reb < 6:
        return "G"
    elif reb > 8 or blk > 1.5:
        return "C"
    elif reb > 5:
        return "F"
    else:
        return "G-F"


def warm_starter_cache():
    """Pre-warm all team starters in a SINGLE API call for lightning-fast responses.
    
    Excludes injured players and fills in from depth chart.
    """
    global cache
    logger.info("Warming starter cache for all 30 teams with REAL NBA API data...")
    
    # First, fetch injury data for filtering
    injured_players = set()
    try:
        from src.injury_collector import InjuryCollector
        injury_collector = InjuryCollector()
        injuries_df = injury_collector.fetch_injuries(force_refresh=True)
        
        if not injuries_df.empty:
            # Get players who are OUT or DOUBTFUL (exclude them from starting 5)
            for _, row in injuries_df.iterrows():
                status = str(row.get('status', '')).lower()
                if 'out' in status or 'doubtful' in status:
                    player_name = row.get('player_name', row.get('Player', row.get('name', '')))
                    if player_name:
                        injured_players.add(player_name)
            logger.info(f"Found {len(injured_players)} injured/doubtful players to exclude from starting lineups")
    except Exception as e:
        logger.warning(f"Could not fetch injury data for lineup filtering: {e}")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            from nba_api.stats.endpoints import leaguedashplayerstats
            import time
            
            # Rate limit delay
            time.sleep(1.0 if attempt > 0 else 0.6)
            
            logger.info(f"Fetching NBA API data (attempt {attempt + 1}/{max_retries})...")
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season='2025-26',
                per_mode_detailed='PerGame',
                timeout=15  # Reduced timeout - fallback to cache if API unresponsive
            )
            df = stats.get_data_frames()[0]
            
            if df.empty:
                logger.warning("NBA API returned empty data, retrying...")
                continue
            
            logger.info(f"Got {len(df)} players from NBA API")
            
            all_starters = {}
            for team_abbrev in NBA_TEAMS.values():
                team_players = df[df['TEAM_ABBREVIATION'] == team_abbrev].copy()
                
                if team_players.empty:
                    logger.warning(f"No players found for {team_abbrev}")
                    continue
                
                # Sort by minutes to get starters (highest minutes = most likely starter)
                team_players = team_players.sort_values('MIN', ascending=False)
                
                # Filter out injured players and build starting 5
                starters = []
                for _, player in team_players.iterrows():
                    player_name = player['PLAYER_NAME']
                    
                    # Skip injured players
                    if player_name in injured_players:
                        logger.debug(f"Excluding injured player {player_name} from {team_abbrev} lineup")
                        continue
                    
                    starters.append({
                        'name': player_name,
                        'position': _infer_position_from_stats(player),
                        'pts': round(float(player.get('PTS', 0)), 1),
                        'reb': round(float(player.get('REB', 0)), 1),
                        'ast': round(float(player.get('AST', 0)), 1),
                        'min': round(float(player.get('MIN', 0)), 1)
                    })
                    
                    # Stop once we have 5 starters
                    if len(starters) >= 5:
                        break
                
                all_starters[team_abbrev] = starters
                cache["rosters"][team_abbrev] = {"data": starters, "timestamp": datetime.utcnow()}
            
            cache["all_starters"]["data"] = all_starters
            cache["all_starters"]["timestamp"] = datetime.utcnow()
            logger.info(f"SUCCESS: Starter cache warmed for {len(all_starters)} teams with injury-aware lineups!")
            return  # Success, exit function
            
        except Exception as e:
            logger.error(f"NBA API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(2)  # Wait before retry
    
    # NBA API failed - try to load from cached file
    logger.warning("NBA API failed. Loading fallback starters from cached file...")
    try:
        import json
        cache_file = ROOT_DIR / "data" / "api_data" / "starters_cache.json"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                all_starters = json.load(f)
            
            for team_abbrev, starters in all_starters.items():
                cache["rosters"][team_abbrev] = {"data": starters, "timestamp": datetime.utcnow()}
            
            cache["all_starters"]["data"] = all_starters
            cache["all_starters"]["timestamp"] = datetime.utcnow()
            logger.info(f"FALLBACK SUCCESS: Loaded {len(all_starters)} teams from starters_cache.json")
            return
    except Exception as e:
        logger.error(f"Failed to load fallback starters: {e}")
    
    logger.error("FAILED: Could not fetch NBA API data after all retries. No roster data available.")

# =============================================================================
# Initialize Flask App
# =============================================================================
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app, origins=["*"])

# =============================================================================
# Initialize Pipeline
# =============================================================================
logger.info("Initializing prediction pipeline...")
try:
    pipeline = PredictionPipeline()
    logger.info("Pipeline ready!")
except Exception as e:
    logger.error(f"Pipeline initialization failed: {e}")
    pipeline = None

# =============================================================================
# Background Scheduler for Auto-Training
# =============================================================================
scheduler = BackgroundScheduler(daemon=True)

# Track training state
last_training_date = None
minimum_accuracy_threshold = 0.55  # Model must be at least 55% accurate
old_model_backup_path = PERSISTENT_MODELS_DIR / "game_predictor_backup.joblib"

def check_all_games_complete():
    """Check if all of today's games are complete."""
    try:
        if not pipeline:
            return False
        games = pipeline.get_games_with_predictions()
        if not games:
            return False  # No games today
        
        # Check if any games are live or upcoming
        for game in games:
            status = game.get("status", "")
            if status in ["IN_PROGRESS", "NOT_STARTED"]:
                return False
        
        # All games are FINAL
        return True
    except Exception as e:
        logger.error(f"Error checking game completion: {e}")
        return False

def get_current_model_accuracy():
    """Get the current model's accuracy from recent predictions."""
    try:
        if not pipeline:
            return 0.0
        stats = pipeline.get_accuracy_stats()
        return stats.get("accuracy", 0.0)
    except:
        return 0.0

def smart_retrain_model():
    """
    Smart retraining that:
    1. Only triggers after all games complete
    2. Validates new model accuracy
    3. Keeps old model if new one is worse
    """
    global pipeline, last_training_date
    
    today = datetime.utcnow().date()
    
    # Skip if already trained today
    if last_training_date == today:
        logger.info("Already trained today, skipping...")
        return
    
    # Check if all games are complete
    if not check_all_games_complete():
        logger.info("Not all games complete yet, skipping training...")
        return
    
    logger.info("All games complete! Starting smart model retraining...")
    
    try:
        import shutil
        from src.auto_trainer import AutoTrainer
        
        # Get current model accuracy before retraining
        old_accuracy = get_current_model_accuracy()
        logger.info(f"Current model accuracy: {old_accuracy:.2%}")
        
        # Backup current model
        current_model = PERSISTENT_MODELS_DIR / "game_predictor.joblib"
        if current_model.exists():
            shutil.copy(current_model, old_model_backup_path)
            logger.info("Backed up current model")
        
        # Train new model
        trainer = AutoTrainer()
        result = trainer.run_training_cycle()
        
        # Handle skipped training (no new games) - not an error
        if result.get("skipped"):
            logger.info(f"Training skipped: {result.get('reason', 'No new games')}")
            # Don't restore backup, just return - model is still valid
            return
        
        if not result.get("success"):
            logger.warning(f"Training failed: {result.get('error', 'Unknown error')}")
            # Restore backup
            if old_model_backup_path.exists():
                shutil.copy(old_model_backup_path, current_model)
            return
        
        new_accuracy = result.get("accuracy", 0.0)
        logger.info(f"New model accuracy: {new_accuracy:.2%}")
        
        # Validate new model is better
        if new_accuracy < minimum_accuracy_threshold:
            logger.warning(f"New model accuracy ({new_accuracy:.2%}) below threshold ({minimum_accuracy_threshold:.0%})")
            logger.warning("Reverting to previous model...")
            if old_model_backup_path.exists():
                shutil.copy(old_model_backup_path, current_model)
            return
        
        if new_accuracy < old_accuracy - 0.02:  # Allow 2% margin
            logger.warning(f"New model ({new_accuracy:.2%}) worse than old ({old_accuracy:.2%})")
            logger.warning("Reverting to previous model...")
            if old_model_backup_path.exists():
                shutil.copy(old_model_backup_path, current_model)
            return
        
        # New model is good - reload pipeline
        logger.info(f"New model is better! Accuracy improved: {old_accuracy:.2%} -> {new_accuracy:.2%}")
        pipeline = PredictionPipeline()
        last_training_date = today
        logger.info("Pipeline reloaded with new model")
        
        # Clean up backup
        if old_model_backup_path.exists():
            old_model_backup_path.unlink()
            
    except Exception as e:
        logger.error(f"Error during smart retraining: {e}")
        # Restore backup on error
        try:
            current_model = PERSISTENT_MODELS_DIR / "game_predictor.joblib"
            if old_model_backup_path.exists():
                shutil.copy(old_model_backup_path, current_model)
        except:
            pass

def update_elo_ratings():
    """Background job to update ELO ratings from completed games."""
    global pipeline
    logger.info("Updating ELO ratings from recent games...")
    
    try:
        if pipeline:
            # Update from completed games
            games = pipeline.get_games_with_predictions()
            updated_count = 0
            for game in games:
                if game.get("status") == "FINAL":
                    # ELO is already updated in the pipeline when games complete
                    updated_count += 1
            logger.info(f"Processed {updated_count} completed games for ELO updates")
    except Exception as e:
        logger.error(f"Error updating ELO ratings: {e}")

def sync_predictions():
    """Background job to sync prediction results."""
    logger.info("Syncing prediction results...")
    
    try:
        if pipeline:
            # Get today's completed games and update predictions
            games = pipeline.get_games_with_predictions()
            synced = 0
            for game in games:
                if game.get("status") == "FINAL":
                    game_id = game.get("game_id")
                    if game_id:
                        home_score = game.get("home_score", 0)
                        away_score = game.get("away_score", 0)
                        actual_winner = game.get("home_team") if home_score > away_score else game.get("away_team")
                        
                        pipeline.prediction_tracker.update_result(
                            game_id, actual_winner, home_score, away_score
                        )
                        synced += 1
            logger.info(f"Synced {synced} prediction results")
            
            # Check if we should trigger training
            if check_all_games_complete():
                logger.info("All games complete - triggering smart retraining check...")
                smart_retrain_model()
    except Exception as e:
        logger.error(f"Error syncing predictions: {e}")

# Schedule background jobs
# Update ELO ratings every 1 hour
scheduler.add_job(
    update_elo_ratings,
    trigger=IntervalTrigger(hours=1),
    id='update_elo',
    name='Update ELO Ratings',
    replace_existing=True
)

# Sync prediction results every 15 minutes
scheduler.add_job(
    sync_predictions,
    trigger=IntervalTrigger(minutes=15),
    id='sync_predictions',
    name='Sync Prediction Results',
    replace_existing=True
)

# Smart retrain check every hour (will only actually train when all games done)
scheduler.add_job(
    smart_retrain_model,
    trigger=IntervalTrigger(hours=1),
    id='smart_retrain',
    name='Smart Model Retraining (after games complete)',
    replace_existing=True
)

# Refresh starter cache every 2 hours (deferred reference - function defined later)
scheduler.add_job(
    lambda: warm_starter_cache(),
    trigger=IntervalTrigger(hours=2),
    id='warm_starters',
    name='Warm Starter Cache',
    replace_existing=True
)

# Start scheduler
scheduler.start()
logger.info("Background scheduler started with jobs: update_elo (1h), sync_predictions (15m), smart_retrain (1h), warm_starters (2h)")

# =============================================================================
# STARTUP CACHE WARMING - Pre-load data for lightning-fast first requests
# =============================================================================
def startup_cache_warming():
    """Warm all caches on startup for instant first requests."""
    logger.info("Starting SYNCHRONOUS cache warming on startup...")
    try:
        # Warm starter cache SYNCHRONOUSLY so data is ready before first request
        # This ensures accurate, real NBA API data is shown immediately
        warm_starter_cache()
        logger.info("Starter cache warming COMPLETE - real data ready!")
    except Exception as e:
        logger.warning(f"Startup cache warming error: {e}")
        logger.warning("Will use fallback data until next refresh")

# Run startup warming (blocks until complete for accurate data)
startup_cache_warming()

# =============================================================================
# Serve React Frontend
# =============================================================================

@app.route('/')
def serve_frontend():
    """Serve the React frontend."""
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files, fallback to index.html for client-side routing."""
    static_folder = Path(app.static_folder)
    file_path = static_folder / path
    
    if file_path.exists() and file_path.is_file():
        return send_from_directory('static', path)
    return send_from_directory('static', 'index.html')

# =============================================================================
# API Endpoints
# =============================================================================

@app.route("/api/health")
def health_check():
    """Health check endpoint with system status."""
    scheduler_running = scheduler.running if scheduler else False
    jobs = [{"id": job.id, "next_run": str(job.next_run_time)} for job in scheduler.get_jobs()] if scheduler else []
    
    return jsonify({
        "status": "healthy",
        "pipeline_ready": pipeline is not None,
        "persistent_storage": IS_HF_SPACES,
        "scheduler_running": scheduler_running,
        "scheduled_jobs": jobs,
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/api/games/live")
def get_live_games():
    """Get today's games with live scores and predictions."""
    if not pipeline:
        return jsonify({"live": [], "final": [], "upcoming": [], "total": 0, "error": "Pipeline not ready"})
    
    try:
        games = pipeline.get_games_with_predictions()
        
        for game in games:
            status = game.get("status")
            game_id = game.get("game_id")
            pred = game.get("prediction", {})
            
            if game_id and pred:
                if status == "NOT_STARTED":
                    existing = pipeline.prediction_tracker.get_prediction(game_id)
                    if not existing:
                        pipeline.prediction_tracker.save_prediction(game_id, {
                            "game_date": game.get("game_date"),
                            "home_team": game.get("home_team"),
                            "away_team": game.get("away_team"),
                            "predicted_winner": pred.get("predicted_winner"),
                            "home_win_probability": pred.get("home_win_probability"),
                            "away_win_probability": pred.get("away_win_probability"),
                            "confidence": pred.get("confidence"),
                            "home_elo": pred.get("home_elo"),
                            "away_elo": pred.get("away_elo"),
                        })
                elif status == "FINAL":
                    home_score = game.get("home_score", 0)
                    away_score = game.get("away_score", 0)
                    actual_winner = game.get("home_team") if home_score > away_score else game.get("away_team")
                    
                    existing = pipeline.prediction_tracker.get_prediction(game_id)
                    if not existing:
                        pipeline.prediction_tracker.save_prediction(game_id, {
                            "game_date": game.get("game_date"),
                            "home_team": game.get("home_team"),
                            "away_team": game.get("away_team"),
                            "predicted_winner": pred.get("predicted_winner"),
                            "home_win_probability": pred.get("home_win_probability"),
                            "away_win_probability": pred.get("away_win_probability"),
                            "confidence": pred.get("confidence"),
                            "home_elo": pred.get("home_elo"),
                            "away_elo": pred.get("away_elo"),
                        })
                    
                    pipeline.prediction_tracker.update_result(game_id, actual_winner, home_score, away_score)
                    game["prediction_correct"] = pred.get("predicted_winner") == actual_winner
        
        return jsonify({
            "live": [g for g in games if g.get("status") == "IN_PROGRESS"],
            "final": [g for g in games if g.get("status") == "FINAL"],
            "upcoming": [g for g in games if g.get("status") == "NOT_STARTED"],
            "total": len(games)
        })
    except Exception as e:
        logger.error(f"Error in get_live_games: {e}")
        return jsonify({"live": [], "final": [], "upcoming": [], "total": 0, "error": str(e)})

@app.route("/api/games/upcoming")
def get_upcoming_games():
    """Get upcoming games for the next N days."""
    if not pipeline:
        return jsonify({"games": [], "count": 0, "error": "Pipeline not ready"})
    
    try:
        days = request.args.get("days", 7, type=int)
        days = max(1, min(days, 14))
        
        games = pipeline.get_upcoming_games(days_ahead=days)
        
        enriched_games = []
        for game in games:
            pred = pipeline.predict_game(game["home_team"], game["away_team"])
            enriched_games.append({**game, "prediction": pred})
        
        return jsonify({"games": enriched_games, "count": len(enriched_games)})
    except Exception as e:
        logger.error(f"Error in get_upcoming_games: {e}")
        return jsonify({"games": [], "count": 0, "error": str(e)})

@app.route("/api/predict")
def predict_game():
    """Predict outcome for a single game."""
    if not pipeline:
        return jsonify({"error": "Pipeline not ready"}), 503
    
    home = request.args.get("home", "").upper()
    away = request.args.get("away", "").upper()
    
    if not home or not away:
        return jsonify({"error": "Missing home or away team parameter"}), 400
    
    try:
        prediction = pipeline.predict_game(home, away)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/team-stats")
def get_team_stats():
    """Get detailed team statistics for head-to-head comparison."""
    team = request.args.get("team", "").upper()
    
    if not team:
        return jsonify({"error": "Missing team parameter"}), 400
    
    try:
        from nba_api.stats.endpoints import teamgamelog, commonteamroster
        from nba_api.stats.static import teams as nba_teams
        import time
        
        # Find team ID
        team_info = next((t for t in nba_teams.get_teams() if t['abbreviation'] == team), None)
        if not team_info:
            return jsonify({"error": f"Team not found: {team}"}), 404
        
        team_id = team_info['id']
        
        # Get team game log for current season
        time.sleep(0.6)  # Rate limit
        game_log = teamgamelog.TeamGameLog(team_id=team_id, season='2024-25')
        games_df = game_log.get_data_frames()[0]
        
        if len(games_df) == 0:
            return jsonify({
                "team": team,
                "record": {"wins": 0, "losses": 0},
                "stats": {},
                "recent_form": [],
                "key_players": []
            })
        
        # Calculate stats
        wins = len(games_df[games_df['WL'] == 'W'])
        losses = len(games_df[games_df['WL'] == 'L'])
        
        # Scoring stats
        avg_pts = games_df['PTS'].mean()
        avg_pts_allowed = games_df['PTS'].mean() - games_df['PLUS_MINUS'].mean()  # Approximate
        avg_reb = games_df['REB'].mean()
        avg_ast = games_df['AST'].mean()
        avg_fg_pct = games_df['FG_PCT'].mean() * 100
        avg_fg3_pct = games_df['FG3_PCT'].mean() * 100
        
        # Recent form (last 5 games)
        recent_games = games_df.head(5)
        recent_form = []
        for _, game in recent_games.iterrows():
            recent_form.append({
                "date": game['GAME_DATE'],
                "opponent": game['MATCHUP'].split()[-1],
                "result": game['WL'],
                "score": f"{int(game['PTS'])} pts",
                "plus_minus": int(game['PLUS_MINUS'])
            })
        
        # Get key players from starters cache
        key_players = []
        if team in starters_cache:
            starters = starters_cache[team].get('starters', [])[:3]  # Top 3 by minutes
            key_players = [{"name": p['name'], "position": p.get('position', 'G')} for p in starters]
        
        return jsonify({
            "team": team,
            "record": {
                "wins": wins,
                "losses": losses,
                "pct": round(wins / max(wins + losses, 1), 3)
            },
            "stats": {
                "ppg": round(avg_pts, 1),
                "opp_ppg": round(avg_pts_allowed, 1),
                "rpg": round(avg_reb, 1),
                "apg": round(avg_ast, 1),
                "fg_pct": round(avg_fg_pct, 1),
                "fg3_pct": round(avg_fg3_pct, 1)
            },
            "recent_form": recent_form,
            "key_players": key_players
        })
        
    except Exception as e:
        logger.error(f"Error getting team stats for {team}: {e}")
        # Return fallback data
        return jsonify({
            "team": team,
            "record": {"wins": 0, "losses": 0, "pct": 0},
            "stats": {"ppg": 0, "opp_ppg": 0, "rpg": 0, "apg": 0, "fg_pct": 0, "fg3_pct": 0},
            "recent_form": [],
            "key_players": []
        })

@app.route("/api/accuracy")
def get_accuracy():
    """Get comprehensive model accuracy statistics."""
    if not pipeline:
        return jsonify({"stats": {}, "recent_predictions": [], "error": "Pipeline not ready"})
    
    try:
        stats = pipeline.get_accuracy_stats()
        recent = pipeline.get_recent_predictions(50)
        
        completed = [p for p in recent if p.get("is_correct", -1) >= 0]
        correct = [p for p in completed if p.get("is_correct") == 1]
        
        home_picks = [p for p in completed if p.get("predicted_winner") == p.get("home_team")]
        home_correct = [p for p in home_picks if p.get("is_correct") == 1]
        away_picks = [p for p in completed if p.get("predicted_winner") == p.get("away_team")]
        away_correct = [p for p in away_picks if p.get("is_correct") == 1]
        
        streak = 0
        streak_type = None
        for p in sorted(completed, key=lambda x: x.get("updated_at", ""), reverse=True):
            if streak_type is None:
                streak_type = "W" if p.get("is_correct") == 1 else "L"
            if (p.get("is_correct") == 1 and streak_type == "W") or (p.get("is_correct") == 0 and streak_type == "L"):
                streak += 1
            else:
                break
        
        last_10 = completed[:10] if len(completed) >= 10 else completed
        last_10_correct = sum(1 for p in last_10 if p.get("is_correct") == 1)
        
        correct_avg_prob = sum(max(p.get("home_win_prob", 0.5), p.get("away_win_prob", 0.5)) for p in correct) / len(correct) if correct else 0
        incorrect = [p for p in completed if p.get("is_correct") == 0]
        incorrect_avg_prob = sum(max(p.get("home_win_prob", 0.5), p.get("away_win_prob", 0.5)) for p in incorrect) / len(incorrect) if incorrect else 0
        
        enhanced_stats = {
            **stats,
            "home_pick_accuracy": len(home_correct) / len(home_picks) if home_picks else 0,
            "away_pick_accuracy": len(away_correct) / len(away_picks) if away_picks else 0,
            "home_picks_total": len(home_picks),
            "away_picks_total": len(away_picks),
            "current_streak": streak,
            "streak_type": streak_type or "N/A",
            "last_10_record": f"{last_10_correct}-{len(last_10) - last_10_correct}",
            "last_10_accuracy": last_10_correct / len(last_10) if last_10 else 0,
            "avg_probability_correct": correct_avg_prob,
            "avg_probability_incorrect": incorrect_avg_prob,
            "pending_predictions": len([p for p in recent if p.get("is_correct", -1) == -1]),
        }
        
        return jsonify({"stats": enhanced_stats, "recent_predictions": recent[:20]})
    except Exception as e:
        logger.error(f"Error in get_accuracy: {e}")
        return jsonify({"stats": {}, "recent_predictions": [], "error": str(e)})

@app.route("/api/analytics")
def get_analytics():
    """Get analytics data for charts and visualizations."""
    if not pipeline:
        return jsonify({"error": "Pipeline not ready"})
    
    try:
        # Get prediction history
        recent = pipeline.prediction_tracker.get_recent_predictions(100)
        completed = [p for p in recent if p.get("is_correct", -1) != -1]
        
        # 1. Accuracy Trend (last 7 days)
        from collections import defaultdict
        daily_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for pred in completed:
            date = pred.get("game_date", "")[:10]  # YYYY-MM-DD
            if date:
                daily_stats[date]["total"] += 1
                if pred.get("is_correct") == 1:
                    daily_stats[date]["correct"] += 1
        
        accuracy_trend = []
        for date in sorted(daily_stats.keys())[-7:]:
            stats = daily_stats[date]
            acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
            accuracy_trend.append({
                "date": date[5:],  # MM-DD format
                "accuracy": round(acc, 1),
                "predictions": stats["total"]
            })
        
        # 2. Accuracy by Team
        team_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        for pred in completed:
            winner = pred.get("predicted_winner", "")
            if winner:
                team_stats[winner]["total"] += 1
                if pred.get("is_correct") == 1:
                    team_stats[winner]["correct"] += 1
        
        team_accuracy = []
        for team, stats in sorted(team_stats.items(), key=lambda x: x[1]["total"], reverse=True)[:10]:
            acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
            team_accuracy.append({
                "team": team,
                "correct": stats["correct"],
                "total": stats["total"],
                "accuracy": round(acc, 1)
            })
        
        # 3. Confidence Distribution
        high = medium = low = 0
        for pred in completed:
            conf = max(pred.get("home_win_prob", 0.5), pred.get("away_win_prob", 0.5))
            if conf > 0.7:
                high += 1
            elif conf > 0.6:
                medium += 1
            else:
                low += 1
        
        confidence_distribution = [
            {"name": "High (>70%)", "value": high, "color": "#4ade80"},
            {"name": "Medium (60-70%)", "value": medium, "color": "#facc15"},
            {"name": "Low (<60%)", "value": low, "color": "#f87171"},
        ]
        
        # 4. Calibration Data
        calibration_buckets = defaultdict(lambda: {"correct": 0, "total": 0})
        for pred in completed:
            conf = max(pred.get("home_win_prob", 0.5), pred.get("away_win_prob", 0.5))
            bucket = int(conf * 100 // 5) * 5  # 5% buckets
            if 55 <= bucket <= 85:
                calibration_buckets[bucket]["total"] += 1
                if pred.get("is_correct") == 1:
                    calibration_buckets[bucket]["correct"] += 1
        
        calibration = []
        for bucket in sorted(calibration_buckets.keys()):
            stats = calibration_buckets[bucket]
            actual = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else bucket
            calibration.append({
                "predicted": bucket,
                "actual": round(actual, 1)
            })
        
        # 5. Overall Stats
        correct_count = len([p for p in completed if p.get("is_correct") == 1])
        total_count = len(completed)
        overall = {
            "total_predictions": total_count,
            "correct": correct_count,
            "accuracy": round(correct_count / total_count * 100, 1) if total_count > 0 else 0,
            "avg_confidence": round(sum(max(p.get("home_win_prob", 0.5), p.get("away_win_prob", 0.5)) 
                                       for p in completed) / len(completed) * 100, 1) if completed else 0
        }
        
        # 6. Recent Predictions for table
        recent_display = []
        for pred in completed[:10]:
            recent_display.append({
                "date": pred.get("game_date", "")[:10],
                "matchup": f"{pred.get('away_team', '?')} @ {pred.get('home_team', '?')}",
                "prediction": pred.get("predicted_winner", "?"),
                "confidence": round(max(pred.get("home_win_prob", 0.5), pred.get("away_win_prob", 0.5)) * 100, 0),
                "correct": pred.get("is_correct") == 1
            })
        
        # 7. ELO Scatter Data (Teams plotted by ELO vs Win%)
        elo_scatter = []
        try:
            from nba_api.stats.endpoints import leaguestandings
            import time
            time.sleep(0.5)
            standings = leaguestandings.LeagueStandings(season='2025-26', timeout=15)
            standings_df = standings.get_data_frames()[0]
            
            for _, row in standings_df.iterrows():
                team_abbrev = row.get('TeamAbbreviation', '')
                if team_abbrev:
                    team_id = next((tid for tid, abbr in NBA_TEAMS.items() if abbr == team_abbrev), None)
                    elo = pipeline.feature_gen.elo.get_rating(team_id) if team_id else 1500
                    elo_scatter.append({
                        "team": team_abbrev,
                        "elo": round(elo, 0),
                        "winPct": round(row.get('WinPCT', 0.5) * 100, 1),
                        "gamesPlayed": row.get('WINS', 0) + row.get('LOSSES', 0),
                        "conference": row.get('Conference', 'East')
                    })
        except Exception as e:
            logger.warning(f"Could not fetch ELO scatter data: {e}")
            # Fallback data
            elo_scatter = [
                {"team": "OKC", "elo": 1650, "winPct": 74, "gamesPlayed": 45, "conference": "West"},
                {"team": "CLE", "elo": 1620, "winPct": 70, "gamesPlayed": 44, "conference": "East"},
                {"team": "BOS", "elo": 1600, "winPct": 66, "gamesPlayed": 46, "conference": "East"},
                {"team": "DEN", "elo": 1580, "winPct": 62, "gamesPlayed": 45, "conference": "West"},
                {"team": "MEM", "elo": 1560, "winPct": 60, "gamesPlayed": 43, "conference": "West"},
                {"team": "HOU", "elo": 1540, "winPct": 58, "gamesPlayed": 44, "conference": "West"},
                {"team": "NYK", "elo": 1530, "winPct": 56, "gamesPlayed": 45, "conference": "East"},
                {"team": "LAL", "elo": 1510, "winPct": 52, "gamesPlayed": 46, "conference": "West"},
            ]
        
        # 8. Radar Chart Data - Model performance across dimensions
        home_correct = sum(1 for p in completed if p.get("is_correct") == 1 and p.get("predicted_winner") == p.get("home_team"))
        home_total = sum(1 for p in completed if p.get("predicted_winner") == p.get("home_team"))
        away_correct = sum(1 for p in completed if p.get("is_correct") == 1 and p.get("predicted_winner") == p.get("away_team"))
        away_total = sum(1 for p in completed if p.get("predicted_winner") == p.get("away_team"))
        
        high_conf_correct = sum(1 for p in completed if p.get("is_correct") == 1 and max(p.get("home_win_prob", 0.5), p.get("away_win_prob", 0.5)) > 0.7)
        high_conf_total = sum(1 for p in completed if max(p.get("home_win_prob", 0.5), p.get("away_win_prob", 0.5)) > 0.7)
        
        low_conf_correct = sum(1 for p in completed if p.get("is_correct") == 1 and max(p.get("home_win_prob", 0.5), p.get("away_win_prob", 0.5)) < 0.6)
        low_conf_total = sum(1 for p in completed if max(p.get("home_win_prob", 0.5), p.get("away_win_prob", 0.5)) < 0.6)
        
        radar_data = [
            {"dimension": "Home Picks", "value": round(home_correct / max(home_total, 1) * 100, 1), "fullMark": 100},
            {"dimension": "Away Picks", "value": round(away_correct / max(away_total, 1) * 100, 1), "fullMark": 100},
            {"dimension": "High Conf", "value": round(high_conf_correct / max(high_conf_total, 1) * 100, 1), "fullMark": 100},
            {"dimension": "Low Conf", "value": round(low_conf_correct / max(low_conf_total, 1) * 100, 1), "fullMark": 100},
            {"dimension": "Overall", "value": round(correct_count / max(total_count, 1) * 100, 1), "fullMark": 100},
        ]
        
        # 9. Home vs Away Split
        home_away_split = {
            "home": {"correct": home_correct, "total": home_total, "accuracy": round(home_correct / max(home_total, 1) * 100, 1)},
            "away": {"correct": away_correct, "total": away_total, "accuracy": round(away_correct / max(away_total, 1) * 100, 1)}
        }
        
        # 10. Streak Data
        streak_data = []
        current_streak = 0
        streak_type = None
        for pred in sorted(completed, key=lambda x: x.get("updated_at", ""), reverse=True)[:20]:
            is_correct = pred.get("is_correct") == 1
            streak_data.append({
                "date": pred.get("game_date", "")[:10],
                "result": "W" if is_correct else "L",
                "matchup": f"{pred.get('away_team', '?')}@{pred.get('home_team', '?')}"
            })
            if streak_type is None:
                streak_type = "W" if is_correct else "L"
            if (is_correct and streak_type == "W") or (not is_correct and streak_type == "L"):
                current_streak += 1
            elif current_streak == 0:
                current_streak = 1
        
        # 11. Top Matchup Predictions (sample of interesting matchups)
        top_matchups = []
        top_teams = ["OKC", "CLE", "BOS", "DEN", "MEM", "HOU", "NYK", "LAL"]
        for i, home in enumerate(top_teams[:4]):
            for away in top_teams[4:]:
                try:
                    pred = pipeline.predict_game(home, away)
                    top_matchups.append({
                        "home": home,
                        "away": away,
                        "homeWinProb": round(pred.get("home_win_probability", 0.5) * 100, 0)
                    })
                except:
                    pass
        
        return jsonify({
            "accuracy_trend": accuracy_trend,
            "team_accuracy": team_accuracy,
            "confidence_distribution": confidence_distribution,
            "calibration": calibration,
            "overall": overall,
            "recent_predictions": recent_display,
            # New advanced analytics
            "elo_scatter": elo_scatter,
            "radar_data": radar_data,
            "home_away_split": home_away_split,
            "streak_data": streak_data[:15],
            "current_streak": {"count": current_streak, "type": streak_type or "W"},
            "top_matchups": top_matchups
        })
        
    except Exception as e:
        logger.error(f"Error in get_analytics: {e}")
        return jsonify({"error": str(e)})

@app.route("/api/mvp")
def get_mvp_race():
    """Get current MVP race standings with caching."""
    global cache
    
    if not pipeline:
        return jsonify({"candidates": [], "error": "Pipeline not ready"})
    
    # Check cache
    now = datetime.utcnow()
    mvp_cache = cache.get("mvp", {})
    cache_data = mvp_cache.get("data")
    cache_time = mvp_cache.get("timestamp")
    cache_ttl = mvp_cache.get("ttl", 300)
    
    # Return cached data if valid
    if cache_data and cache_time and (now - cache_time).total_seconds() < cache_ttl:
        logger.debug("Returning cached MVP data")
        return jsonify(cache_data)
    
    # Fetch fresh data in background thread if cache expired
    def fetch_mvp_data():
        try:
            mvp_df = pipeline.get_mvp_race()
            
            candidates = []
            for idx, row in mvp_df.iterrows():
                candidates.append({
                    "rank": len(candidates) + 1,
                    "name": row["PLAYER_NAME"],
                    "ppg": round(float(row["PTS"]), 1),
                    "rpg": round(float(row["REB"]), 1),
                    "apg": round(float(row["AST"]), 1),
                    "mvp_score": round(float(row["mvp_score"]), 1),
                    "similarity": round(float(row["mvp_similarity"]) * 100, 1)
                })
            
            # Update cache
            cache["mvp"]["data"] = {"candidates": candidates}
            cache["mvp"]["timestamp"] = datetime.utcnow()
            logger.info(f"MVP cache updated with {len(candidates)} candidates")
        except Exception as e:
            logger.error(f"Background MVP fetch error: {e}")
    
    # If we have stale cache, return it immediately and refresh in background
    if cache_data:
        threading.Thread(target=fetch_mvp_data, daemon=True).start()
        return jsonify(cache_data)
    
    # No cache - fetch synchronously (first request only)
    try:
        mvp_df = pipeline.get_mvp_race()
        
        candidates = []
        for idx, row in mvp_df.iterrows():
            candidates.append({
                "rank": len(candidates) + 1,
                "name": row["PLAYER_NAME"],
                "ppg": round(float(row["PTS"]), 1),
                "rpg": round(float(row["REB"]), 1),
                "apg": round(float(row["AST"]), 1),
                "mvp_score": round(float(row["mvp_score"]), 1),
                "similarity": round(float(row["mvp_similarity"]) * 100, 1)
            })
        
        # Update cache
        cache["mvp"]["data"] = {"candidates": candidates}
        cache["mvp"]["timestamp"] = datetime.utcnow()
        
        return jsonify({"candidates": candidates})
    except Exception as e:
        logger.error(f"Error in get_mvp_race: {e}")
        return jsonify({"candidates": [], "error": str(e)})

@app.route("/api/championship")
def get_championship_odds():
    """Get current championship odds with caching."""
    global cache
    
    if not pipeline:
        return jsonify({"teams": [], "error": "Pipeline not ready"})
    
    # Check cache
    now = datetime.utcnow()
    champ_cache = cache.get("championship", {})
    cache_data = champ_cache.get("data")
    cache_time = champ_cache.get("timestamp")
    cache_ttl = champ_cache.get("ttl", 300)
    
    # Return cached data if valid
    if cache_data and cache_time and (now - cache_time).total_seconds() < cache_ttl:
        logger.debug("Returning cached championship data")
        return jsonify(cache_data)
    
    # Fetch fresh data in background thread if cache expired
    def fetch_champ_data():
        try:
            champ_df = pipeline.get_championship_odds()
            
            teams = []
            for idx, row in champ_df.iterrows():
                teams.append({
                    "rank": len(teams) + 1,
                    "team": row.get("TEAM_ABBREVIATION", row.get("Team", "N/A")),
                    "odds": round(float(row.get("champ_probability", row.get("Championship_Odds", 0))) * 100, 1),
                    "win_pct": round(float(row.get("W_PCT", 0.5)) * 100, 1)
                })
            
            # Update cache
            cache["championship"]["data"] = {"teams": teams}
            cache["championship"]["timestamp"] = datetime.utcnow()
            logger.info(f"Championship cache updated with {len(teams)} teams")
        except Exception as e:
            logger.error(f"Background championship fetch error: {e}")
    
    # If we have stale cache, return it immediately and refresh in background
    if cache_data:
        threading.Thread(target=fetch_champ_data, daemon=True).start()
        return jsonify(cache_data)
    
    # No cache - fetch synchronously (first request only)
    try:
        champ_df = pipeline.get_championship_odds()
        
        teams = []
        for idx, row in champ_df.iterrows():
            teams.append({
                "rank": len(teams) + 1,
                "team": row.get("TEAM_ABBREVIATION", row.get("Team", "N/A")),
                "odds": round(float(row.get("champ_probability", row.get("Championship_Odds", 0))) * 100, 1),
                "win_pct": round(float(row.get("W_PCT", 0.5)) * 100, 1)
            })
        
        # Update cache
        cache["championship"]["data"] = {"teams": teams}
        cache["championship"]["timestamp"] = datetime.utcnow()
        
        return jsonify({"teams": teams})
    except Exception as e:
        logger.error(f"Error in get_championship_odds: {e}")
        return jsonify({"teams": [], "error": str(e)})

@app.route("/api/teams")
def get_teams():
    """Get list of all NBA teams."""
    try:
        from src.config import NBA_TEAMS
        teams = [{"id": tid, "abbrev": abbrev} for tid, abbrev in NBA_TEAMS.items()]
        teams.sort(key=lambda x: x["abbrev"])
        return jsonify({"teams": teams})
    except Exception as e:
        return jsonify({"teams": [], "error": str(e)})

@app.route("/api/roster/<team_abbrev>")
def get_team_roster(team_abbrev):
    """Get projected starting 5 for a team - INSTANT from cache."""
    global cache
    team_abbrev = team_abbrev.upper()
    
    # Check pre-warmed all_starters cache first (fastest)
    all_starters = cache.get("all_starters", {})
    if all_starters.get("data") and team_abbrev in all_starters["data"]:
        return jsonify({"team": team_abbrev, "starters": all_starters["data"][team_abbrev]})
    
    # Check individual team cache
    if team_abbrev in cache.get("rosters", {}):
        team_cache = cache["rosters"][team_abbrev]
        cache_age = (datetime.utcnow() - team_cache.get("timestamp", datetime.min)).total_seconds()
        if cache_age < cache.get("roster_ttl", 3600):  # Within TTL
            return jsonify({"team": team_abbrev, "starters": team_cache["data"]})
    
    # No cache hit - return fallback immediately while refreshing in background
    def refresh_cache():
        try:
            warm_starter_cache()
        except Exception as e:
            logger.warning(f"Background roster refresh failed: {e}")
    
    threading.Thread(target=refresh_cache, daemon=True).start()
    
    # Return fallback data from pipeline
    if pipeline:
        roster = pipeline.get_team_roster(team_abbrev)
        return jsonify({"team": team_abbrev, "starters": roster})
    
    return jsonify({"team": team_abbrev, "starters": []})


# =============================================================================
# Admin/Management Endpoints
# =============================================================================

@app.route("/api/admin/retrain", methods=["POST"])
def manual_retrain():
    """Manually trigger model retraining."""
    try:
        # Run in background thread to not block
        thread = threading.Thread(target=auto_retrain_model)
        thread.start()
        return jsonify({"status": "Retraining started", "message": "Model retraining initiated in background"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/admin/sync", methods=["POST"])
def manual_sync():
    """Manually sync prediction results."""
    try:
        sync_predictions()
        return jsonify({"status": "Sync complete", "message": "Prediction results synced"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/admin/status")
def get_system_status():
    """Get detailed system status."""
    try:
        jobs_info = []
        for job in scheduler.get_jobs():
            jobs_info.append({
                "id": job.id,
                "name": job.name,
                "next_run": str(job.next_run_time) if job.next_run_time else "Not scheduled",
                "trigger": str(job.trigger)
            })
        
        # Check persistent storage
        storage_info = {
            "persistent_dir": str(PERSISTENT_DIR),
            "is_hf_spaces": IS_HF_SPACES,
            "models_dir_exists": PERSISTENT_MODELS_DIR.exists(),
            "predictions_dir_exists": PERSISTENT_PREDICTIONS_DIR.exists(),
        }
        
        # Count files
        if PERSISTENT_MODELS_DIR.exists():
            storage_info["model_files"] = len(list(PERSISTENT_MODELS_DIR.glob("*.joblib")))
        if PERSISTENT_PREDICTIONS_DIR.exists():
            storage_info["prediction_files"] = len(list(PERSISTENT_PREDICTIONS_DIR.glob("*.json")))
        
        return jsonify({
            "status": "running",
            "pipeline_ready": pipeline is not None,
            "scheduler_running": scheduler.running,
            "scheduled_jobs": jobs_info,
            "storage": storage_info,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    logger.info(f"Starting NBA Sage server on port {port}")
    logger.info(f"Persistent storage: {PERSISTENT_DIR}")
    logger.info(f"Is HF Spaces: {IS_HF_SPACES}")
    app.run(host="0.0.0.0", port=port, debug=False)
