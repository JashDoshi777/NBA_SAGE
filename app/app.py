"""
NBA ML Prediction System - Streamlit Dashboard
===============================================
Comprehensive web interface with live scores and prediction tracking.
Auto-training runs in background while the dashboard is active.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prediction_pipeline import PredictionPipeline
from src.config import NBA_TEAMS, LIVE_REFRESH_INTERVAL
from src.auto_trainer import auto_trainer

# Start auto-trainer when app loads (runs in background thread)
if not auto_trainer._running:
    auto_trainer.start()

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="NBA ML Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh import (used on Live Games page)
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# Custom CSS for premium look
st.markdown("""
<style>
    /* Main theme */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Cards */
    .prediction-card {
        background: linear-gradient(145deg, #1f2937 0%, #111827 100%);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    /* Live game card */
    .live-game-card {
        background: linear-gradient(145deg, #1f2937 0%, #111827 100%);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        border: 2px solid #10b981;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.2);
    }
    
    /* Live indicator */
    .live-indicator {
        background: #ef4444;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Team names */
    .team-name {
        font-size: 28px;
        font-weight: 700;
        color: #fff;
    }
    
    /* Score display */
    .score-display {
        font-size: 48px;
        font-weight: 800;
        color: #00d4ff;
    }
    
    /* Win probability */
    .win-prob {
        font-size: 48px;
        font-weight: 800;
        background: linear-gradient(90deg, #00d4ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Confidence badges */
    .confidence-high {
        background: linear-gradient(90deg, #10b981, #059669);
        padding: 4px 12px;
        border-radius: 20px;
        color: white;
        font-weight: 600;
    }
    
    .confidence-medium {
        background: linear-gradient(90deg, #f59e0b, #d97706);
        padding: 4px 12px;
        border-radius: 20px;
        color: white;
        font-weight: 600;
    }
    
    .confidence-low {
        background: linear-gradient(90deg, #ef4444, #dc2626);
        padding: 4px 12px;
        border-radius: 20px;
        color: white;
        font-weight: 600;
    }
    
    /* Result badges */
    .result-correct {
        background: linear-gradient(90deg, #10b981, #059669);
        padding: 4px 12px;
        border-radius: 20px;
        color: white;
        font-weight: 600;
    }
    
    .result-wrong {
        background: linear-gradient(90deg, #ef4444, #dc2626);
        padding: 4px 12px;
        border-radius: 20px;
        color: white;
        font-weight: 600;
    }
    
    /* Stats */
    .stat-value {
        font-size: 32px;
        font-weight: 700;
        color: #00d4ff;
    }
    
    .stat-label {
        font-size: 14px;
        color: #9ca3af;
        text-transform: uppercase;
    }
    
    /* Accuracy meter */
    .accuracy-meter {
        font-size: 64px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #10b981, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Header */
    .main-header {
        font-size: 48px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #00d4ff, #7c3aed, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 32px;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #111827;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# INITIALIZE PIPELINE
# =============================================================================
@st.cache_resource
def get_pipeline():
    return PredictionPipeline()

pipeline = get_pipeline()

# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
st.sidebar.markdown("# 🏀 NBA ML Predictor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🔴 Live Games", "🎮 Game Predictions", "📈 Model Accuracy", "🏆 MVP Race", "👑 Championship Odds", "📊 Team Explorer"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

# Refresh button for live data
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Auto-trainer status
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Auto Training")
trainer_status = auto_trainer.get_status()

if trainer_status["running"]:
    st.sidebar.success("✓ Active")
    st.sidebar.caption(f"Next game ingest: {trainer_status['next_ingest_in']}")
    st.sidebar.caption(f"Next model retrain: {trainer_status['next_retrain_in']}")
else:
    st.sidebar.warning("⚠ Not running")
    if st.sidebar.button("▶ Start Auto-Training"):
        auto_trainer.start()
        st.rerun()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def render_confidence_badge(confidence: str) -> str:
    """Render a confidence badge HTML."""
    return f'<span class="confidence-{confidence}">{confidence.upper()}</span>'

def render_result_badge(is_correct: bool) -> str:
    """Render a result badge HTML."""
    if is_correct:
        return '<span class="result-correct">✓ CORRECT</span>'
    else:
        return '<span class="result-wrong">✗ WRONG</span>'

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_projected_starters(team_abbr: str) -> list:
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
        team_players = df[df['TEAM_ABBREVIATION'] == team_abbr].copy()
        
        if team_players.empty:
            return []
        
        # Sort by minutes played (starters play the most minutes)
        team_players = team_players.sort_values('MIN', ascending=False)
        
        # Get top 5 players (projected starters)
        starters = []
        for _, player in team_players.head(5).iterrows():
            starters.append({
                'name': player['PLAYER_NAME'],
                'pts': player['PTS'],
                'reb': player.get('REB', 0),
                'ast': player.get('AST', 0)
            })
        
        return starters
        
    except Exception as e:
        # Fallback to local data if API fails
        try:
            import pandas as pd
            from pathlib import Path
            
            player_stats_path = Path("data/cache/player_stats_2025-26.parquet")
            if not player_stats_path.exists():
                return []
            
            df = pd.read_parquet(player_stats_path)
            team_players = df[df['TEAM_ABBREVIATION'] == team_abbr].copy()
            
            if team_players.empty:
                return []
            
            team_players = team_players.sort_values('MIN', ascending=False)
            
            starters = []
            for _, player in team_players.head(5).iterrows():
                starters.append({
                    'name': player.get('PLAYER_NAME', 'Unknown'),
                    'pts': player.get('PTS', 0),
                    'reb': player.get('REB', 0),
                    'ast': player.get('AST', 0)
                })
            
            return starters
        except:
            return []


def render_probability_bar(prob: float, label: str = "") -> None:
    """Render a probability bar with Plotly."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': label, 'font': {'size': 16, 'color': 'white'}},
        number={'suffix': '%', 'font': {'size': 40, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'white'},
            'bar': {'color': '#7c3aed'},
            'bgcolor': '#1f2937',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': '#374151'},
                {'range': [50, 100], 'color': '#4c1d95'}
            ],
        }
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE: LIVE GAMES
# =============================================================================
if page == "🔴 Live Games":
    # Auto-refresh every 30 seconds for live scores
    if HAS_AUTOREFRESH:
        st_autorefresh(interval=LIVE_REFRESH_INTERVAL * 1000, limit=None, key="live_refresh")
    
    st.markdown('<h1 class="main-header">🔴 Live NBA Games</h1>', unsafe_allow_html=True)
    
    # Get all today's games with live data
    games = pipeline.get_games_with_predictions()
    
    if not games:
        st.info("📅 No NBA games scheduled for today. Check back later!")
    else:
        # Separate by status
        live_games = [g for g in games if g.get("status") == "IN_PROGRESS"]
        final_games = [g for g in games if g.get("status") == "FINAL"]
        upcoming_games = [g for g in games if g.get("status") == "NOT_STARTED"]
        
        # =================================================================
        # LIVE GAMES
        # =================================================================
        if live_games:
            st.markdown("### 🔴 In Progress")
            for game in live_games:
                pred = game["prediction"]
                with st.container():
                    st.markdown(f"""
                        <div class="live-game-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span class="live-indicator">● LIVE</span>
                                <span style="color: #9ca3af;">Q{game.get('period', '?')} {game.get('clock', '')}</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns([2, 1, 2])
                    
                    with col1:
                        st.markdown(f"### {game['away_team']}")
                        st.markdown(f"<div class='score-display'>{game['away_score']}</div>", unsafe_allow_html=True)
                        st.caption(f"Record: {game.get('away_record', 'N/A')}")
                    
                    with col2:
                        st.markdown("<div style='text-align: center; padding-top: 20px;'>", unsafe_allow_html=True)
                        st.markdown(f"""
                            <div style='font-size: 14px; color: #9ca3af;'>PREDICTED</div>
                            <div style='font-size: 24px; font-weight: 700; color: #7c3aed;'>{pred['predicted_winner']}</div>
                            {render_confidence_badge(pred['confidence'])}
                        """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"### {game['home_team']} 🏠")
                        st.markdown(f"<div class='score-display'>{game['home_score']}</div>", unsafe_allow_html=True)
                        st.caption(f"Record: {game.get('home_record', 'N/A')}")
                    
                    st.markdown("---")
        
        # =================================================================
        # FINAL GAMES
        # =================================================================
        if final_games:
            st.markdown("### ✅ Completed Games")
            for game in final_games:
                pred = game["prediction"]
                is_correct = game.get("prediction_correct", False)
                
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 1, 2, 1])
                    
                    with col1:
                        st.markdown(f"### {game['away_team']}")
                        st.markdown(f"<div class='score-display'>{game['away_score']}</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("<div style='text-align: center; padding-top: 20px;'>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size: 24px; color: #9ca3af;'>FINAL</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"### {game['home_team']} 🏠")
                        st.markdown(f"<div class='score-display'>{game['home_score']}</div>", unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown("<div style='text-align: center; padding-top: 20px;'>", unsafe_allow_html=True)
                        st.markdown(f"""
                            <div style='font-size: 12px; color: #9ca3af;'>PREDICTED: {pred['predicted_winner']}</div>
                            <div style='margin-top: 8px;'>{render_result_badge(is_correct)}</div>
                        """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("---")
        
        # =================================================================
        # UPCOMING GAMES
        # =================================================================
        if upcoming_games:
            st.markdown("### ⏳ Upcoming Today")
            for game in upcoming_games:
                pred = game["prediction"]
                
                with st.expander(f"🏀 {game['away_team']} @ {game['home_team']} - {game.get('status_text', 'TBD')}", expanded=False):
                    col1, col2, col3 = st.columns([2, 1, 2])
                    
                    with col1:
                        st.markdown(f"### {game['away_team']}")
                        st.metric("Win Probability", f"{pred['away_win_probability']:.0%}")
                        st.caption(f"ELO: {pred['away_elo']:.0f}")
                    
                    with col2:
                        st.markdown("<div style='text-align: center; padding-top: 20px;'>", unsafe_allow_html=True)
                        st.markdown(f"""
                            <div style='font-size: 14px; color: #9ca3af;'>PICK</div>
                            <div style='font-size: 28px; font-weight: 700; color: #10b981;'>{pred['predicted_winner']}</div>
                            {render_confidence_badge(pred['confidence'])}
                        """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"### {game['home_team']} 🏠")
                        st.metric("Win Probability", f"{pred['home_win_probability']:.0%}")
                        st.caption(f"ELO: {pred['home_elo']:.0f}")
                    
                    # Projected Starting Lineups
                    st.markdown("---")
                    st.markdown("#### 👥 Projected Starting 5")
                    
                    lineup_col1, lineup_col2 = st.columns(2)
                    
                    with lineup_col1:
                        st.markdown(f"**{game['away_team']}**")
                        away_starters = get_projected_starters(game['away_team'])
                        if away_starters:
                            for i, player in enumerate(away_starters, 1):
                                st.markdown(f"{i}. {player['name']} ({player['pts']:.1f} PPG)")
                        else:
                            st.caption("Lineup data unavailable")
                    
                    with lineup_col2:
                        st.markdown(f"**{game['home_team']}**")
                        home_starters = get_projected_starters(game['home_team'])
                        if home_starters:
                            for i, player in enumerate(home_starters, 1):
                                st.markdown(f"{i}. {player['name']} ({player['pts']:.1f} PPG)")
                        else:
                            st.caption("Lineup data unavailable")

# =============================================================================
# PAGE: GAME PREDICTIONS
# =============================================================================
elif page == "🎮 Game Predictions":
    st.markdown('<h1 class="main-header">Game Predictions</h1>', unsafe_allow_html=True)
    
    # Quick game predictor
    st.markdown("### 🎯 Quick Prediction")
    col1, col2, col3 = st.columns([2, 1, 2])
    
    team_list = sorted(NBA_TEAMS.values())
    
    with col1:
        away_team = st.selectbox("Away Team", team_list, index=team_list.index("BOS") if "BOS" in team_list else 0)
    
    with col2:
        st.markdown("<h2 style='text-align: center; padding-top: 25px;'>@</h2>", unsafe_allow_html=True)
    
    with col3:
        home_team = st.selectbox("Home Team", team_list, index=team_list.index("LAL") if "LAL" in team_list else 1)
    
    if st.button("🔮 Predict Game", type="primary", use_container_width=True):
        with st.spinner("Analyzing matchup..."):
            prediction = pipeline.predict_game(home_team, away_team)
        
        st.markdown("---")
        
        # Display prediction
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.markdown(f"### {away_team}")
            st.metric("Win Probability", f"{prediction['away_win_probability']:.1%}")
            st.metric("ELO Rating", f"{prediction['away_elo']:.0f}")
        
        with col2:
            winner = prediction['predicted_winner']
            confidence = prediction['confidence']
            st.markdown(f"""
                <div style='text-align: center; padding: 20px;'>
                    <div style='font-size: 18px; color: #9ca3af;'>Predicted Winner</div>
                    <div class='team-name' style='font-size: 48px;'>{winner}</div>
                    <div style='margin-top: 10px;'>
                        {render_confidence_badge(confidence)}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            render_probability_bar(prediction['home_win_probability'], f"{home_team} Win Probability")
        
        with col3:
            st.markdown(f"### {home_team}")
            st.metric("Win Probability", f"{prediction['home_win_probability']:.1%}")
            st.metric("ELO Rating", f"{prediction['home_elo']:.0f}")
            st.caption("🏠 Home Court Advantage")
        
        # Key factors
        st.markdown("---")
        st.markdown("### 📊 Key Factors")
        for factor in prediction['factors']:
            st.markdown(f"• {factor}")
    
    # ==========================================================================
    # Upcoming Games with Predictions
    # ==========================================================================
    st.markdown("---")
    st.markdown("### 📅 Upcoming NBA Games")
    
    # Get upcoming games
    upcoming_games = pipeline.get_upcoming_games(days_ahead=3)
    
    # Group by date
    games_by_date = {}
    for game in upcoming_games:
        date = game['date']
        if date not in games_by_date:
            games_by_date[date] = []
        games_by_date[date].append(game)
    
    # Display games by date
    for date, games in games_by_date.items():
        day_name = games[0]['day_name']
        st.markdown(f"#### 📆 {day_name} - {date}")
        
        for game in games:
            home = game['home_team']
            away = game['away_team']
            time = game['time']
            
            # Get prediction
            pred = pipeline.predict_game(home, away)
            
            with st.expander(f"🏀 {away} @ {home} - {time}", expanded=False):
                col1, col2, col3 = st.columns([2, 1, 2])
                
                with col1:
                    st.markdown(f"### {away}")
                    st.metric("Win Probability", f"{pred['away_win_probability']:.0%}")
                
                with col2:
                    winner = pred['predicted_winner']
                    confidence = pred['confidence']
                    st.markdown(f"""
                        <div style='text-align: center; padding-top: 20px;'>
                            <div style='font-size: 14px; color: #9ca3af;'>PICK</div>
                            <div style='font-size: 36px; font-weight: 800; color: #10b981;'>{winner}</div>
                            <div class='confidence-{confidence}'>{confidence.upper()}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"### {home} 🏠")
                    st.metric("Win Probability", f"{pred['home_win_probability']:.0%}")
                
                # Projected Starting Lineups
                st.markdown("---")
                st.markdown("#### 👥 Projected Starting 5")
                
                lineup_col1, lineup_col2 = st.columns(2)
                
                with lineup_col1:
                    st.markdown(f"**{away}**")
                    away_starters = get_projected_starters(away)
                    if away_starters:
                        for i, player in enumerate(away_starters, 1):
                            st.markdown(f"{i}. {player['name']} ({player['pts']:.1f} PPG)")
                    else:
                        st.caption("Lineup data unavailable")
                
                with lineup_col2:
                    st.markdown(f"**{home}**")
                    home_starters = get_projected_starters(home)
                    if home_starters:
                        for i, player in enumerate(home_starters, 1):
                            st.markdown(f"{i}. {player['name']} ({player['pts']:.1f} PPG)")
                    else:
                        st.caption("Lineup data unavailable")

# =============================================================================
# PAGE: MODEL ACCURACY
# =============================================================================
elif page == "📈 Model Accuracy":
    st.markdown('<h1 class="main-header">Model Accuracy Tracker</h1>', unsafe_allow_html=True)
    
    # Get accuracy stats
    stats = pipeline.get_accuracy_stats()
    
    # Main accuracy display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class='prediction-card' style='text-align: center;'>
                <div class='stat-label'>Total Predictions</div>
                <div class='stat-value'>{}</div>
            </div>
        """.format(stats.get('total_predictions', 0)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='prediction-card' style='text-align: center;'>
                <div class='stat-label'>Completed Games</div>
                <div class='stat-value'>{}</div>
            </div>
        """.format(stats.get('completed_games', 0)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class='prediction-card' style='text-align: center;'>
                <div class='stat-label'>Correct Predictions</div>
                <div class='stat-value'>{}</div>
            </div>
        """.format(stats.get('correct_predictions', 0)), unsafe_allow_html=True)
    
    with col4:
        accuracy = stats.get('overall_accuracy', 0)
        st.markdown(f"""
            <div class='prediction-card' style='text-align: center;'>
                <div class='stat-label'>Overall Accuracy</div>
                <div class='accuracy-meter'>{accuracy:.1%}</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Accuracy by confidence level
    st.markdown("### 📊 Accuracy by Confidence Level")
    
    by_confidence = stats.get('by_confidence', {})
    if by_confidence:
        conf_cols = st.columns(3)
        for i, conf in enumerate(['high', 'medium', 'low']):
            if conf in by_confidence:
                data = by_confidence[conf]
                with conf_cols[i]:
                    color = "#10b981" if conf == "high" else "#f59e0b" if conf == "medium" else "#ef4444"
                    st.markdown(f"""
                        <div class='prediction-card' style='text-align: center;'>
                            <div style='font-size: 18px; color: {color}; font-weight: 600;'>{conf.upper()}</div>
                            <div class='stat-value'>{data['accuracy']:.1%}</div>
                            <div style='color: #9ca3af;'>{data['correct']}/{data['total']} correct</div>
                        </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("No predictions tracked yet. Predictions will be tracked automatically as games are played.")
    
    st.markdown("---")
    
    # Recent predictions
    st.markdown("### 📋 Recent Predictions")
    
    recent = pipeline.get_recent_predictions(20)
    if recent:
        for pred in recent:
            is_pending = pred.get('is_correct', -1) == -1
            is_correct = pred.get('is_correct') == 1
            
            status_icon = "⏳" if is_pending else "✅" if is_correct else "❌"
            result_text = "Pending" if is_pending else ("Correct" if is_correct else "Wrong")
            
            st.markdown(f"""
                {status_icon} **{pred.get('away_team', 'N/A')}** @ **{pred.get('home_team', 'N/A')}** 
                - Predicted: {pred.get('predicted_winner', 'N/A')} ({pred.get('confidence', 'N/A')}) 
                - {result_text}
            """)
    else:
        st.info("No predictions recorded yet. Make predictions on the Live Games page!")
    
    # Check and update results button
    if st.button("🔄 Check Game Results", type="primary"):
        with st.spinner("Checking completed games..."):
            updated = pipeline.check_prediction_results()
        if updated:
            st.success(f"Updated {len(updated)} prediction results!")
            st.rerun()
        else:
            st.info("No new results to update.")

# =============================================================================
# PAGE: MVP RACE
# =============================================================================
elif page == "🏆 MVP Race":
    st.markdown('<h1 class="main-header">MVP Race 2025-26</h1>', unsafe_allow_html=True)
    
    # Get MVP rankings with caching
    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def get_cached_mvp_race():
        return pipeline.get_mvp_race()
    
    mvp_df = get_cached_mvp_race()
    
    # Top 3 podium
    st.markdown("### 🥇 Top MVP Candidates")
    
    cols = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]
    
    for i, (idx, row) in enumerate(mvp_df.head(3).iterrows()):
        with cols[i]:
            st.markdown(f"""
                <div class='prediction-card' style='text-align: center;'>
                    <div style='font-size: 48px;'>{medals[i]}</div>
                    <div style='font-size: 24px; font-weight: 700; color: white;'>{row['PLAYER_NAME']}</div>
                    <div style='font-size: 14px; color: #9ca3af; margin-top: 8px;'>
                        {row['PTS']:.1f} PPG | {row['REB']:.1f} RPG | {row['AST']:.1f} APG
                    </div>
                    <div style='margin-top: 12px;'>
                        <span class='stat-value'>{row['mvp_score']:.1f}</span>
                        <span class='stat-label'> MVP Score</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Full rankings table
    st.markdown("### 📋 Full Rankings")
    
    display_df = mvp_df.copy()
    display_df.index = range(1, len(display_df) + 1)
    display_df.columns = ["Player", "PPG", "RPG", "APG", "MVP Score", "Historical Similarity"]
    
    st.dataframe(
        display_df.style.format({
            "PPG": "{:.1f}",
            "RPG": "{:.1f}",
            "APG": "{:.1f}",
            "MVP Score": "{:.2f}",
            "Historical Similarity": "{:.1%}"
        }).background_gradient(subset=["MVP Score"], cmap="Purples"),
        use_container_width=True
    )

# =============================================================================
# PAGE: CHAMPIONSHIP ODDS
# =============================================================================
elif page == "👑 Championship Odds":
    st.markdown('<h1 class="main-header">Championship Odds 2025-26</h1>', unsafe_allow_html=True)
    
    # Get championship odds
    champ_df = pipeline.get_championship_odds()
    
    # Top contenders
    st.markdown("### 🏆 Top Championship Contenders")
    
    cols = st.columns(4)
    for i, (idx, row) in enumerate(champ_df.head(4).iterrows()):
        with cols[i]:
            st.markdown(f"""
                <div class='prediction-card' style='text-align: center;'>
                    <div style='font-size: 36px; font-weight: 700; color: white;'>{row['TEAM_ABBREVIATION']}</div>
                    <div class='win-prob'>{row['champ_probability']:.1%}</div>
                    <div style='font-size: 12px; color: #9ca3af;'>Championship Probability</div>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Championship probability visualization
    st.markdown("### 📊 Championship Probability Distribution")
    
    fig = px.pie(
        champ_df,
        values="champ_probability",
        names="TEAM_ABBREVIATION",
        color_discrete_sequence=px.colors.sequential.Purples_r,
        hole=0.4
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE: TEAM EXPLORER
# =============================================================================
elif page == "📊 Team Explorer":
    st.markdown('<h1 class="main-header">Team Explorer</h1>', unsafe_allow_html=True)
    
    # Team selection
    selected_team = st.selectbox("Select Team", sorted(NBA_TEAMS.values()))
    
    # Team info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
            <div class='prediction-card'>
                <div style='font-size: 48px; font-weight: 700; color: white; text-align: center;'>{selected_team}</div>
                <div style='text-align: center; color: #9ca3af;'>2025-26 Season</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🏥 Injury Report")
        injuries = pipeline.injury_collector.get_injury_summary(selected_team)
        
        if injuries["total_injuries"] > 0:
            for player in injuries["players"][:5]:
                status_color = "🔴" if "Out" in player["status"] else "🟡" if "Questionable" in player["status"] else "🟢"
                st.markdown(f"{status_color} **{player['name']}** - {player['status']}")
        else:
            st.success("✅ No injured players!")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #6b7280; padding: 20px;'>
        <p>🏀 NBA ML Predictor | Built with XGBoost + LightGBM Ensemble</p>
        <p style='font-size: 12px;'>Live scores via NBA API | Predictions tracked in ChromaDB</p>
    </div>
""", unsafe_allow_html=True)
