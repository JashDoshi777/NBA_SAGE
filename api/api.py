"""
NBA ML Prediction System - Flask Backend
=========================================
REST API for the React frontend.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
import logging
from pathlib import Path

# Configure logging to reduce verbosity
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("src.injury_collector").setLevel(logging.WARNING)
logging.getLogger("src.prediction_tracker").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prediction_pipeline import PredictionPipeline

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"])

# Initialize prediction pipeline (ELO ratings loaded on startup)
print("Initializing prediction pipeline...")
pipeline = PredictionPipeline()
print("Pipeline ready!")


@app.route("/api/health")
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "pipeline_ready": pipeline is not None})


@app.route("/api/games/live")
def get_live_games():
    """Get today's games with live scores and predictions."""
    games = pipeline.get_games_with_predictions()
    
    # Process each game - save predictions for upcoming, update results for completed
    for game in games:
        status = game.get("status")
        game_id = game.get("game_id")
        pred = game.get("prediction", {})
        
        if game_id and pred:
            if status == "NOT_STARTED":
                # Only save if prediction doesn't already exist
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
                # Update result for completed game
                home_score = game.get("home_score", 0)
                away_score = game.get("away_score", 0)
                actual_winner = game.get("home_team") if home_score > away_score else game.get("away_team")
                
                # Check if prediction exists, if not save it first
                existing = pipeline.prediction_tracker.get_prediction(game_id)
                if not existing:
                    # Save prediction first (for games completed before tracking started)
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
                
                # Now update with result
                pipeline.prediction_tracker.update_result(
                    game_id,
                    actual_winner,
                    home_score,
                    away_score
                )
                
                # Mark whether our prediction was correct
                game["prediction_correct"] = pred.get("predicted_winner") == actual_winner
    
    # Separate by status
    return jsonify({
        "live": [g for g in games if g.get("status") == "IN_PROGRESS"],
        "final": [g for g in games if g.get("status") == "FINAL"],
        "upcoming": [g for g in games if g.get("status") == "NOT_STARTED"],
        "total": len(games)
    })


@app.route("/api/games/upcoming")
def get_upcoming_games():
    """Get upcoming games for the next N days."""
    days = request.args.get("days", 7, type=int)
    days = max(1, min(days, 14))  # Clamp between 1-14
    
    games = pipeline.get_upcoming_games(days_ahead=days)
    
    # Add predictions to each game
    enriched_games = []
    for game in games:
        pred = pipeline.predict_game(game["home_team"], game["away_team"])
        enriched_games.append({
            **game,
            "prediction": pred
        })
    
    return jsonify({"games": enriched_games, "count": len(enriched_games)})


@app.route("/api/predict")
def predict_game():
    """Predict outcome for a single game."""
    home = request.args.get("home", "").upper()
    away = request.args.get("away", "").upper()
    
    if not home or not away:
        return jsonify({"error": "Missing home or away team parameter"}), 400
    
    prediction = pipeline.predict_game(home, away)
    return jsonify(prediction)


@app.route("/api/accuracy")
def get_accuracy():
    """Get comprehensive model accuracy statistics."""
    stats = pipeline.get_accuracy_stats()
    recent = pipeline.get_recent_predictions(50)  # Get more for analysis
    
    # Calculate additional metrics
    completed = [p for p in recent if p.get("is_correct", -1) >= 0]
    correct = [p for p in completed if p.get("is_correct") == 1]
    
    # Home vs Away accuracy
    home_picks = [p for p in completed if p.get("predicted_winner") == p.get("home_team")]
    home_correct = [p for p in home_picks if p.get("is_correct") == 1]
    away_picks = [p for p in completed if p.get("predicted_winner") == p.get("away_team")]
    away_correct = [p for p in away_picks if p.get("is_correct") == 1]
    
    # Current streak
    streak = 0
    streak_type = None
    for p in sorted(completed, key=lambda x: x.get("updated_at", ""), reverse=True):
        if streak_type is None:
            streak_type = "W" if p.get("is_correct") == 1 else "L"
        if (p.get("is_correct") == 1 and streak_type == "W") or (p.get("is_correct") == 0 and streak_type == "L"):
            streak += 1
        else:
            break
    
    # Last 10 games
    last_10 = completed[:10] if len(completed) >= 10 else completed
    last_10_correct = sum(1 for p in last_10 if p.get("is_correct") == 1)
    
    # Average win probability for correct vs incorrect predictions
    correct_avg_prob = sum(max(p.get("home_win_prob", 0.5), p.get("away_win_prob", 0.5)) for p in correct) / len(correct) if correct else 0
    incorrect = [p for p in completed if p.get("is_correct") == 0]
    incorrect_avg_prob = sum(max(p.get("home_win_prob", 0.5), p.get("away_win_prob", 0.5)) for p in incorrect) / len(incorrect) if incorrect else 0
    
    # Build enhanced response
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
    
    return jsonify({
        "stats": enhanced_stats,
        "recent_predictions": recent[:20]  # Return 20 most recent for display
    })


@app.route("/api/mvp")
def get_mvp_race():
    """Get current MVP race standings."""
    mvp_df = pipeline.get_mvp_race()
    
    # Convert DataFrame to list of dicts
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
    
    return jsonify({"candidates": candidates})


@app.route("/api/championship")
def get_championship_odds():
    """Get current championship odds."""
    champ_df = pipeline.get_championship_odds()
    
    # Convert DataFrame to list of dicts
    teams = []
    for idx, row in champ_df.iterrows():
        # ChampionshipPredictor returns: TEAM_ABBREVIATION, W_PCT, playoff_experience, strength_rating, champ_probability
        teams.append({
            "rank": len(teams) + 1,
            "team": row.get("TEAM_ABBREVIATION", row.get("Team", "N/A")),
            "odds": round(float(row.get("champ_probability", row.get("Championship_Odds", 0))) * 100, 1),
            "win_pct": round(float(row.get("W_PCT", 0.5)) * 100, 1)
        })
    
    return jsonify({"teams": teams})


@app.route("/api/teams")
def get_teams():
    """Get list of all NBA teams."""
    from src.config import NBA_TEAMS
    
    teams = [{"id": tid, "abbrev": abbrev} for tid, abbrev in NBA_TEAMS.items()]
    teams.sort(key=lambda x: x["abbrev"])
    
    return jsonify({"teams": teams})


@app.route("/api/roster/<team_abbrev>")
def get_team_roster(team_abbrev):
    """Get projected starting 5 for a team using LIVE 2025-26 season stats."""
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats
        import time
        
        # Fetch current season player stats from NBA API
        time.sleep(0.5)  # Rate limiting
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season='2025-26',
            per_mode_detailed='PerGame'
        )
        df = stats.get_data_frames()[0]
        
        # Filter by team
        team_abbrev = team_abbrev.upper()
        team_players = df[df['TEAM_ABBREVIATION'] == team_abbrev].copy()
        
        if team_players.empty:
            return jsonify({"team": team_abbrev, "starters": []})
        
        # Sort by minutes played (starters play the most minutes)
        team_players = team_players.sort_values('MIN', ascending=False)
        
        # Get top 5 players (projected starters)
        starters = []
        for _, player in team_players.head(5).iterrows():
            starters.append({
                'name': player['PLAYER_NAME'],
                'position': player.get('POSITION', ''),
                'pts': round(float(player['PTS']), 1),
                'reb': round(float(player.get('REB', 0)), 1),
                'ast': round(float(player.get('AST', 0)), 1),
                'min': round(float(player.get('MIN', 0)), 1)
            })
        
        return jsonify({"team": team_abbrev, "starters": starters})
        
    except Exception as e:
        print(f"Error fetching roster for {team_abbrev}: {e}")
        # Fallback to pipeline method
        roster = pipeline.get_team_roster(team_abbrev.upper())
        return jsonify({"team": team_abbrev.upper(), "starters": roster})


@app.route("/api/standings")
def get_standings():
    """Get current NBA standings by conference."""
    # Team name to abbreviation mapping for fallback - includes team names, cities, and variants
    TEAM_NAME_TO_ABBREV = {
        # Team nicknames
        "hawks": "ATL", "celtics": "BOS", "nets": "BKN", "hornets": "CHA",
        "bulls": "CHI", "cavaliers": "CLE", "cavs": "CLE", "mavericks": "DAL", "mavs": "DAL",
        "nuggets": "DEN", "pistons": "DET", "warriors": "GSW", "dubs": "GSW",
        "rockets": "HOU", "pacers": "IND", "clippers": "LAC", "lakers": "LAL",
        "grizzlies": "MEM", "heat": "MIA", "bucks": "MIL", "timberwolves": "MIN", "wolves": "MIN",
        "pelicans": "NOP", "pels": "NOP", "knicks": "NYK", "thunder": "OKC",
        "magic": "ORL", "76ers": "PHI", "sixers": "PHI", "suns": "PHX",
        "trail blazers": "POR", "blazers": "POR", "trailblazers": "POR",
        "kings": "SAC", "spurs": "SAS", "raptors": "TOR", "jazz": "UTA", "wizards": "WAS",
        # City names
        "atlanta": "ATL", "boston": "BOS", "brooklyn": "BKN", "charlotte": "CHA",
        "chicago": "CHI", "cleveland": "CLE", "dallas": "DAL", "denver": "DEN",
        "detroit": "DET", "golden state": "GSW", "houston": "HOU", "indiana": "IND",
        "los angeles c": "LAC", "la c": "LAC", "los angeles l": "LAL", "la l": "LAL",
        "memphis": "MEM", "miami": "MIA", "milwaukee": "MIL", "minnesota": "MIN",
        "new orleans": "NOP", "new york": "NYK", "oklahoma city": "OKC", "oklahoma": "OKC",
        "orlando": "ORL", "philadelphia": "PHI", "phoenix": "PHX", "portland": "POR",
        "sacramento": "SAC", "san antonio": "SAS", "toronto": "TOR", "utah": "UTA", "washington": "WAS"
    }
    
    def get_abbrev_from_name(team_name):
        """Extract abbreviation from team name."""
        team_name_lower = team_name.lower().strip()
        for name_part, abbrev in TEAM_NAME_TO_ABBREV.items():
            if name_part in team_name_lower:
                return abbrev
        return ""
    
    try:
        from nba_api.stats.endpoints import leaguestandings
        import time
        
        time.sleep(0.5)
        standings = leaguestandings.LeagueStandings(season='2025-26')
        df = standings.get_data_frames()[0]
        
        # Debug: print column names on first run
        print(f"Standings columns: {list(df.columns)}")
        
        east = []
        west = []
        
        for _, row in df.iterrows():
            # Try multiple possible column names for team abbreviation
            abbrev = ""
            for col_name in ["TeamSlug", "TeamAbbreviation", "TEAM_ABBREVIATION", "team_abbreviation"]:
                if col_name in row.index and row.get(col_name):
                    abbrev = str(row.get(col_name)).upper().strip()
                    break
            
            # Build team name from city + name
            team_city = str(row.get("TeamCity", ""))
            team_name = str(row.get("TeamName", ""))
            full_team_name = f"{team_city} {team_name}".strip()
            
            # If abbreviation still not found, extract from team name
            if not abbrev:
                abbrev = get_abbrev_from_name(full_team_name)
            
            team_data = {
                "team": abbrev,  # Use abbreviation for frontend TeamLogo component
                "team_abbrev": abbrev,  # Duplicate for clarity
                "team_name": full_team_name,
                "wins": int(row.get("WINS", 0)),
                "losses": int(row.get("LOSSES", 0)),
                "win_pct": float(row.get("WinPCT", 0)),
                "gb": str(row.get("ConferenceGamesBack", "-")),
                "streak": str(row.get("strCurrentStreak", "-")),
                "conference": row.get("Conference", ""),
            }
            
            if row.get("Conference") == "East":
                east.append(team_data)
            else:
                west.append(team_data)
        
        # Sort by wins descending
        east.sort(key=lambda x: (-x["wins"], x["losses"]))
        west.sort(key=lambda x: (-x["wins"], x["losses"]))
        
        return jsonify({"east": east, "west": west})
        
    except Exception as e:
        print(f"Error fetching standings: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"east": [], "west": [], "error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
