🏀 NBA Sage — AI-Powered NBA Game Predictor

NBA Sage is a full-stack machine learning system that predicts NBA game outcomes using historical data, real-time statistics, ELO ratings, and automated learning pipelines.

It combines sports analytics, machine learning, and real-time data ingestion into a production-ready application with a modern web interface.

🚀 What Does NBA Sage Do?

NBA Sage provides:

✅ Pre-game win probability predictions

📊 Live NBA scores with real-time updates

📈 Prediction accuracy tracking over time

🏆 MVP race rankings based on live season stats

🥇 Championship odds for all 30 NBA teams

All predictions are generated automatically using current season data, recent form, and historical performance.

🧠 How It Works (High Level)
Prediction Pipeline

Collects historical NBA data (23 seasons)

Fetches live data for today’s games

Generates advanced features (ELO, rolling stats, fatigue, momentum)

Calculates win probabilities

Stores predictions & results

Tracks accuracy continuously

NBA API → Feature Engineering → Prediction Engine → API → React Dashboard

📊 Prediction Methodology
🧮 Live Game Prediction Formula

Live predictions currently use a multi-factor statistical model:

Factor	Weight
Current season win %	40%
Recent form (last 10 games)	30%
ELO rating strength	20%
Baseline stabilization	10%

Adjustments applied:

+3.5% Home court advantage

−2% per injury impact point

Log5 formula for head-to-head probability

This approach ensures stability, interpretability, and strong real-world performance.

🤖 Machine Learning Model (Trained)

NBA Sage also includes a trained ML ensemble, used for offline training and evaluation:

Model Architecture
Game Features
 ├── XGBoost (50%)
 └── LightGBM (50%)
        ↓
   Ensemble Probability

Training Details

41,000+ historical NBA games

90+ engineered features

Time-aware training (no future leakage)

Automated retraining & rollback if accuracy drops

🔬 The ML model is production-ready and designed to be integrated into live predictions.

🧩 Features Used (Summary)

ELO ratings & differentials

Rolling averages (5 / 10 / 20 games)

Rest days & back-to-back games

Home / away context

Season performance metrics

Momentum & streak indicators

Advanced team stats (OffRtg, DefRtg, Pace)

Clutch & hustle stats

Top player contribution metrics

Era normalization (z-score adjusted)

📈 Accuracy & Stats
Metric	Value
Seasons covered	23 (2003–2026)
Games analyzed	41,000+
NBA teams	30
Features used	90+
Prediction tracking	Yes (ChromaDB)
Auto retraining	Yes

Accuracy is tracked:

Overall

By confidence level

By team

Over time

🖥️ Web Application
Frontend

React 18 + Vite

Live dashboards

Modern UI with custom CSS system

Pages

Live Games

Upcoming Predictions

Accuracy Dashboard

MVP Race

Championship Odds

Head-to-Head Comparison

🛠️ Tech Stack
Backend

Python

Flask

XGBoost

LightGBM

Pandas / NumPy

APScheduler

ChromaDB Cloud

Frontend

React

Vite

Custom CSS

Infrastructure

Docker

Hugging Face Spaces

Git LFS (large data files)

🔁 Continuous Learning System

NBA Sage retrains itself automatically:

Ingests completed games

Updates features

Retrains the model

Compares accuracy

Deploys only if performance improves

This prevents model degradation over time.


Why This Project Matters

NBA Sage demonstrates:

Real-world ML system design

Feature engineering at scale

Time-aware modeling

Full-stack ML deployment

Production data pipelines

Continuous learning systems
