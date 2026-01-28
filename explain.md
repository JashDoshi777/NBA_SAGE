# NBA Sage - Technical Explanation

> **An AI-powered NBA game prediction system with real-time data, machine learning, and a modern web interface.**

---

## 🎯 What Does This Project Do?

NBA Sage is a full-stack application that:

1. **Predicts NBA game outcomes** before they happen
2. **Shows live scores** with real-time updates
3. **Tracks prediction accuracy** over time
4. **Calculates MVP race standings** based on current stats
5. **Estimates championship odds** for all 30 teams

---

## 🏆 Key Features

| Feature | Description |
|---------|-------------|
| **Live Game Dashboard** | Real-time scores, game status, win probabilities |
| **Win Predictions** | Probability % for each team to win |
| **Starting 5 Lineups** | Projected starters with PPG stats from NBA API |
| **MVP Race** | Top 10 MVP candidates with scores |
| **Championship Odds** | All 30 teams ranked by title probability |
| **Model Accuracy** | Track how well predictions perform over time |

---

## 🛠️ Technology Stack

### Backend (Python)
| Technology | Purpose |
|------------|---------|
| **Flask** | REST API framework |
| **nba_api** | Official NBA data (stats.nba.com) |
| **XGBoost + LightGBM** | Machine learning ensemble model |
| **APScheduler** | Background job scheduling |
| **ChromaDB Cloud** | Persistent prediction storage |
| **Pandas/NumPy** | Data processing |

### Frontend (React)
| Technology | Purpose |
|------------|---------|
| **React 18** | UI framework |
| **Vite** | Build tool & dev server |
| **Custom CSS** | Modern design system |

### Infrastructure
| Technology | Purpose |
|------------|---------|
| **Docker** | Container deployment |
| **Hugging Face Spaces** | Cloud hosting |
| **Git LFS** | Large file versioning |

---

## 🔬 How Predictions Work

### The Prediction Algorithm

Predictions are made using a **multi-factor formula**:

```
Win Probability = Log5 Formula of:
├── 40% - Current Season Record (Win %)
├── 30% - Recent Form (Last 10 games performance)
├── 20% - ELO Rating (Historical team strength)
└── 10% - Baseline

Adjustments Applied:
├── +3.5% for Home Court Advantage
└── -2% per Injury Impact Point
```

### ELO Rating System

ELO is a chess-inspired rating system adapted for NBA:

- **Starting rating**: 1500 (average team)
- **K-factor**: 20 (how much ratings change per game)
- **Home advantage**: +100 ELO points equivalent
- **Season regression**: Ratings regress 25% to mean each season

**How it works:**
- Win against better team → Big ELO gain
- Win against weaker team → Small ELO gain
- Lose against better team → Small ELO loss
- Lose against weaker team → Big ELO loss

---

## 📊 Data Sources

### Real-Time Data
- **NBA Live API** (`nba_api.live`)
  - Live scores updated every 30 seconds
  - Game status (scheduled, in progress, final)
  - Box scores and player stats

### Historical Data
- **NBA Stats API** (`nba_api.stats`)
  - 23 years of game data (2003-2026)
  - Team statistics (basic, advanced, clutch, hustle)
  - Player statistics
  - Current season stats for predictions

### Data Storage
- **Parquet files**: Cached API responses (~140 files)
- **ChromaDB Cloud**: Prediction history and accuracy tracking
- **Joblib files**: Trained ML model and processed datasets

---

## 🧠 Machine Learning Components

### Trained Model: XGBoost + LightGBM Ensemble

Two gradient boosting models trained on 41,000+ historical games:

```
Game Features ──┬──► XGBoost (50%) ──┐
                │                    │──► Ensemble Prediction
                └──► LightGBM (50%) ─┘
```

**Features Used:**
- ELO ratings and differentials
- Rolling averages (5, 10, 20 game windows)
- Rest days and back-to-back games
- Home/away status
- Season record statistics

### Training Pipeline

```
Data Collection ──► Feature Engineering ──► Model Training ──► Evaluation
         │                  │                    │
         ▼                  ▼                    ▼
    NBA API Data      ELO Calculation      XGBoost+LightGBM
                      Era Normalization
                      Rolling Windows
```

### Auto-Training System

The system automatically retrains itself:

1. **Ingests completed games** every hour
2. **Waits for all daily games** to complete
3. **Compares new model accuracy** to existing
4. **Only updates if improved** (prevents regression)

---

## 🌐 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        React Frontend                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │LiveGames │ │Predictions│ │MVP Race  │ │ Accuracy │           │
│  └────▲─────┘ └────▲─────┘ └────▲─────┘ └────▲─────┘           │
└───────┼────────────┼────────────┼────────────┼──────────────────┘
        │            │            │            │
        └────────────┴─────┬──────┴────────────┘
                           │ REST API
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Flask Server                                │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐    │
│  │   Endpoints    │  │    Caching     │  │  Scheduler     │    │
│  │  /api/live     │  │  In-Memory     │  │  APScheduler   │    │
│  │  /api/roster   │  │  1-hour rosters│  │  Auto-retrain  │    │
│  └────────┬───────┘  └────────────────┘  └────────────────┘    │
│           │                                                      │
│           ▼                                                      │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              Prediction Pipeline                        │    │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │    │
│  │  │Live Collector│ │Feature Gen  │ │ ELO System  │       │    │
│  │  └─────────────┘ └─────────────┘ └─────────────┘       │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      External Services                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  NBA API    │  │ ChromaDB    │  │ Hugging Face│             │
│  │ stats.nba   │  │   Cloud     │  │   Spaces    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
NBA ML/
├── server.py              # Production server (Hugging Face)
├── api/api.py             # Development server
│
├── src/                   # Core logic
│   ├── prediction_pipeline.py   # Main orchestrator
│   ├── feature_engineering.py   # ELO + features
│   ├── data_collector.py        # Historical data
│   ├── live_data_collector.py   # Real-time data
│   ├── prediction_tracker.py    # Accuracy tracking
│   └── models/
│       └── game_predictor.py    # ML model
│
├── web/                   # React frontend
│   └── src/
│       ├── App.jsx
│       ├── pages/         # UI pages
│       └── index.css      # Design system
│
├── data/
│   └── api_data/          # 140+ parquet files
│
└── models/
    └── game_predictor.joblib  # Trained model (9.6KB)
```

---

## 🚀 Deployment

### Local Development
```bash
# Backend
python api/api.py  # Runs on localhost:8000

# Frontend
cd web && npm run dev  # Runs on localhost:5173
```

### Production (Hugging Face Spaces)
```bash
# Docker container
python server.py  # Serves both API + React on port 7860
```

---

## 📈 Performance & Accuracy

### Prediction Accuracy
- **Overall**: Tracked via ChromaDB Cloud
- **By Confidence**: High/Medium/Low confidence splits
- **By Team**: Per-team prediction accuracy

### Speed Optimizations
- **In-memory caching**: Roster data cached for 1 hour
- **Startup warming**: All 30 teams pre-loaded on server start
- **Background refresh**: Cache updated every 2 hours

---

## 🔮 Future Improvements

1. **Integrate ML model** into live predictions (currently formula-based)
2. **Add player-level features** (injuries, rest days per player)
3. **Implement spread predictions** (margin of victory)
4. **Add playoff predictions** with series outcomes

---

## 📊 Stats at a Glance

| Metric | Value |
|--------|-------|
| Historical games | 41,000+ |
| Seasons covered | 23 (2003-2026) |
| Teams tracked | 30 |
| ML model type | XGBoost + LightGBM |
| API endpoints | 10+ |
| Frontend pages | 6 |
---

## 📋 Complete ML Feature List (90+ Features)

The model uses approximately **90 features** organized into these categories:

### 1️⃣ ELO Rating Features (5 features)
| Feature | Description |
|---------|-------------|
| `team_elo` | Team's current ELO rating |
| `opponent_elo` | Opponent's current ELO rating |
| `elo_diff` | Difference between team and opponent ELO |
| `elo_win_prob` | Expected win probability from ELO |
| `home_elo_boost` | ELO boost for home court (100 points) |

### 2️⃣ Basic Stats - Rolling Averages (21 features)
For each of 7 stats × 3 windows (5, 10, 20 games):

| Base Stat | Windows |
|-----------|---------|
| `PTS` (Points) | `PTS_last5`, `PTS_last10`, `PTS_last20` |
| `AST` (Assists) | `AST_last5`, `AST_last10`, `AST_last20` |
| `REB` (Rebounds) | `REB_last5`, `REB_last10`, `REB_last20` |
| `FG_PCT` (Field Goal %) | `FG_PCT_last5`, `FG_PCT_last10`, `FG_PCT_last20` |
| `FG3_PCT` (3-Point %) | `FG3_PCT_last5`, `FG3_PCT_last10`, `FG3_PCT_last20` |
| `FT_PCT` (Free Throw %) | `FT_PCT_last5`, `FT_PCT_last10`, `FT_PCT_last20` |
| `PLUS_MINUS` (Point Diff) | `PLUS_MINUS_last5`, `PLUS_MINUS_last10`, `PLUS_MINUS_last20` |

### 3️⃣ Season Statistics (9 features)
| Feature | Description |
|---------|-------------|
| `PTS_season_avg` | Season average points |
| `AST_season_avg` | Season average assists |
| `REB_season_avg` | Season average rebounds |
| `FG_PCT_season_avg` | Season field goal % |
| `FG3_PCT_season_avg` | Season 3-point % |
| `FT_PCT_season_avg` | Season free throw % |
| `PLUS_MINUS_season_avg` | Season point differential |
| `win_pct_season` | Season win percentage |
| `games_played` | Games played in season |

### 4️⃣ Defensive Features (4 features)
| Feature | Description |
|---------|-------------|
| `STL_last10` | Steals per game (last 10) |
| `BLK_last10` | Blocks per game (last 10) |
| `DREB_last10` | Defensive rebounds (last 10) |
| `pts_allowed_last10` | Points allowed (last 10) |

### 5️⃣ Momentum Features (6 features)
| Feature | Description |
|---------|-------------|
| `wins_last5` | Wins in last 5 games (0-5) |
| `wins_last10` | Wins in last 10 games (0-10) |
| `hot_streak` | 1 if 4+ wins in last 5 |
| `cold_streak` | 1 if 1 or fewer wins in last 5 |
| `plus_minus_last5` | Point differential trend |
| `form_trend` | Comparison of last 3 vs previous 3 |

### 6️⃣ Rest & Fatigue Features (4 features)
| Feature | Description |
|---------|-------------|
| `days_rest` | Days since last game |
| `back_to_back` | 1 if playing consecutive days |
| `well_rested` | 1 if 3+ days rest |
| `games_last_week` | Games played in last 7 days |

### 7️⃣ Form Index Features (3 features)
| Feature | Description |
|---------|-------------|
| `form_index` | Exponentially-weighted recent performance (0-1) |
| `form_trend` | Trend direction (improving/declining) |
| `form_plus_minus` | Weighted point differential |

### 8️⃣ Basic Stat Columns (17 raw features)
```python
BASIC_STATS = [
    "PTS", "AST", "REB", "STL", "BLK", "TOV",
    "FGM", "FGA", "FG_PCT",
    "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT",
    "OREB", "DREB"
]
```

### 9️⃣ Advanced Team Stats (11 features)
```python
ADVANCED_STATS = [
    "E_OFF_RATING",    # Offensive Rating
    "E_DEF_RATING",    # Defensive Rating
    "E_NET_RATING",    # Net Rating
    "E_PACE",          # Pace (possessions per game)
    "E_AST_RATIO",     # Assist Ratio
    "E_OREB_PCT",      # Offensive Rebound %
    "E_DREB_PCT",      # Defensive Rebound %
    "E_REB_PCT",       # Total Rebound %
    "E_TM_TOV_PCT",    # Team Turnover %
    "E_EFG_PCT",       # Effective FG%
    "E_TS_PCT"         # True Shooting %
]
```

### 🔟 Clutch Stats (4 features)
```python
CLUTCH_STATS = [
    "CLUTCH_PTS",          # Points in clutch time
    "CLUTCH_FG_PCT",       # FG% in clutch
    "CLUTCH_FG3_PCT",      # 3PT% in clutch
    "CLUTCH_PLUS_MINUS"    # +/- in clutch
]
```

### 1️⃣1️⃣ Hustle Stats (5 features)
```python
HUSTLE_STATS = [
    "DEFLECTIONS",             # Passes deflected
    "LOOSE_BALLS_RECOVERED",   # Loose balls recovered
    "CHARGES_DRAWN",           # Offensive fouls drawn
    "CONTESTED_SHOTS",         # Shots contested
    "SCREEN_ASSISTS"           # Screen assists
]
```

### 1️⃣2️⃣ Top Player Stats (6 features)
| Feature | Description |
|---------|-------------|
| `top_players_avg_pts` | Avg points of top 5 players |
| `top_players_avg_ast` | Avg assists of top 5 players |
| `top_players_avg_reb` | Avg rebounds of top 5 players |
| `top_players_avg_stl` | Avg steals of top 5 players |
| `top_players_avg_blk` | Avg blocks of top 5 players |
| `star_concentration` | % of scoring from top player |

### 1️⃣3️⃣ Game Context (1 feature)
| Feature | Description |
|---------|-------------|
| `is_home` | 1 if home team, 0 if away |

---

## 📊 Feature Summary

| Category | Feature Count |
|----------|---------------|
| ELO Ratings | 5 |
| Rolling Averages (5/10/20) | 21 |
| Season Statistics | 9 |
| Defensive Stats | 4 |
| Momentum Features | 6 |
| Rest/Fatigue | 4 |
| Form Index | 3 |
| Advanced Team Stats | 11 |
| Clutch Stats | 4 |
| Hustle Stats | 5 |
| Top Player Stats | 6 |
| Game Context | 1 |
| **TOTAL** | **~79 core features** |

*Plus Z-score normalized versions of stats for era adjustment = **90+ total features***

---

*Built with Python, React, and a passion for basketball analytics* 🏀
