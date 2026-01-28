# 🏀 NBA Sage – AI-Powered NBA Game Predictor

NBA Sage is a full-stack machine learning system that predicts NBA game outcomes using historical data, real-time statistics, ELO ratings, and automated learning pipelines.

It combines sports analytics, machine learning, and real-time data ingestion into a production-ready web application.

---

## 🚀 What Does NBA Sage Do?

NBA Sage provides:

- Pre-game win probability predictions  
- Live NBA scores with real-time updates  
- Prediction accuracy tracking over time  
- MVP race rankings based on live season stats  
- Championship odds for all 30 NBA teams  

---

## 🧠 How It Works (High Level)

1. Collects historical NBA data across multiple seasons  
2. Fetches live game and team data  
3. Generates advanced features (ELO, rolling stats, rest, momentum)  
4. Calculates win probabilities  
5. Stores predictions and tracks results  

**Data Flow**

NBA API → Feature Engineering → Prediction Engine → REST API → React Dashboard

---

## 📊 Prediction Methodology

### Live Prediction Formula

Live predictions are generated using a multi-factor statistical model:

- 40% Current season win percentage  
- 30% Recent form (last 10 games)  
- 20% ELO rating strength  
- 10% Baseline stabilization  

**Adjustments**
- +3.5% home court advantage  
- −2% per injury impact point  
- Log5 formula for head-to-head probability  

This approach prioritizes stability, interpretability, and real-world reliability.

---

## 🤖 Machine Learning Model

NBA Sage includes a trained machine learning ensemble used for offline training and evaluation.

**Model Type**
- XGBoost classifier
- LightGBM classifier
- 50/50 weighted ensemble

**Training**
- Trained on 41,000+ historical NBA games  
- Time-aware training (no future data leakage)  
- Automated retraining with accuracy-based rollback  

---

## 🧩 Feature Engineering (Summary)

- ELO ratings and differentials  
- Rolling averages (5, 10, 20 games)  
- Rest days and back-to-back games  
- Home / away context  
- Season performance metrics  
- Momentum and streak indicators  
- Advanced team stats (OffRtg, DefRtg, Pace)  
- Clutch and hustle stats  
- Top player contribution metrics  
- Era-normalized features (z-score)  

Total features used: **90+**

---

## 📈 Stats at a Glance

| Metric | Value |
|------|------|
| Seasons covered | 23 (2003–2026) |
| Games analyzed | 41,000+ |
| NBA teams | 30 |
| Features used | 90+ |
| Prediction tracking | Yes |
| Auto retraining | Yes |

---

## 🖥️ Web Application

**Frontend**
- React 18
- Vite
- Custom CSS design system

**Pages**
- Live Games  
- Upcoming Predictions  
- Accuracy Dashboard  
- MVP Race  
- Championship Odds  
- Head-to-Head Comparison  

---

## 🛠️ Tech Stack

**Backend**
- Python
- Flask
- XGBoost
- LightGBM
- Pandas, NumPy
- APScheduler
- ChromaDB Cloud

**Frontend**
- React
- Vite

**Infrastructure**
- Docker
- Hugging Face Spaces
- Git LFS

---

## 🔁 Continuous Learning

The system automatically:

1. Ingests completed games  
2. Updates features  
3. Retrains the model  
4. Compares new accuracy with previous versions  
5. Deploys only if performance improves  



