"""
NBA ML Prediction System - Configuration
=========================================
Central configuration for data collection, model training, and predictions.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# =============================================================================
# PATHS
# =============================================================================
import os
PROJECT_ROOT = Path(__file__).parent.parent
# Use environment variable for data dir if set (for HF Spaces persistent storage)
DATA_DIR = Path(os.environ.get("NBA_ML_DATA_DIR", str(PROJECT_ROOT / "data")))
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
API_CACHE_DIR = DATA_DIR / "api_data"
MODELS_DIR = Path(os.environ.get("NBA_ML_MODELS_DIR", str(PROJECT_ROOT / "models")))

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, API_CACHE_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SEASONS
# =============================================================================
# Extended dataset: 23 years (2003-2026) for comprehensive training
INITIAL_SEASON_START = 2003
INITIAL_SEASON_END = 2026  # Current 2025-26 season
FULL_SEASON_START = 2003  # Full dataset starts from 2003

def get_season_strings(start_year: int = INITIAL_SEASON_START, 
                       end_year: int = INITIAL_SEASON_END) -> List[str]:
    """Generate season strings like '2003-04', '2004-05', etc."""
    return [f"{year}-{str(year+1)[-2:]}" for year in range(start_year, end_year)]

SEASON_STRINGS = get_season_strings()

# =============================================================================
# CHROMADB CONFIGURATION
# =============================================================================
import os

@dataclass
class ChromaDBConfig:
    """Configuration for ChromaDB prediction tracking.
    Reads from environment variables for security, with fallback defaults.
    """
    tenant: str = os.environ.get("CHROMADB_TENANT", "70e82e68-9fa7-4224-9975-d49d355f6328")
    database: str = os.environ.get("CHROMADB_DATABASE", "NBA_ML")
    api_key: str = os.environ.get("CHROMADB_API_KEY", "ck-2bXunZK4X3BFSPHtwLG2Ki9xr5r6ZPxzADESDperHweT")
    collection_name: str = "predictions"
    
CHROMADB_CONFIG = ChromaDBConfig()

# =============================================================================
# LIVE DATA CONFIGURATION
# =============================================================================
LIVE_REFRESH_INTERVAL = 15  # Seconds between live score refreshes

# =============================================================================
# API CONFIGURATION
# =============================================================================
@dataclass
class APIConfig:
    """Configuration for NBA API requests with robustness features."""
    base_delay: float = 0.6  # Base delay between requests (seconds)
    max_retries: int = 3  # Maximum retry attempts
    initial_backoff: float = 2.0  # Initial backoff in seconds
    max_backoff: float = 60.0  # Maximum backoff in seconds
    backoff_multiplier: float = 2.0  # Exponential backoff multiplier
    timeout: int = 30  # Request timeout in seconds
    
API_CONFIG = APIConfig()

# =============================================================================
# ELO CONFIGURATION
# =============================================================================
@dataclass
class ELOConfig:
    """Configuration for ELO rating calculations."""
    initial_rating: float = 1500.0
    k_factor: float = 20.0  # How much ratings change per game
    home_advantage: float = 100.0  # ELO points added for home team
    season_regression: float = 0.85  # Regress to mean at season start (85% = only 15% carryover)
    
ELO_CONFIG = ELOConfig()

# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================
@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    min_games_for_features: int = 5  # Minimum games before generating features
    
FEATURE_CONFIG = FeatureConfig()

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
@dataclass
class ModelConfig:
    """Configuration for model training."""
    test_seasons: List[str] = field(default_factory=lambda: ["2024-25"])
    val_seasons: List[str] = field(default_factory=lambda: ["2023-24"])
    random_state: int = 42
    
    # XGBoost defaults
    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    })
    
    # LightGBM defaults
    lgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1
    })
    
MODEL_CONFIG = ModelConfig()

# =============================================================================
# TEAM MAPPINGS
# =============================================================================
# NBA Team IDs (for reference)
NBA_TEAMS = {
    1610612737: "ATL", 1610612738: "BOS", 1610612739: "CLE", 1610612740: "NOP",
    1610612741: "CHI", 1610612742: "DAL", 1610612743: "DEN", 1610612744: "GSW",
    1610612745: "HOU", 1610612746: "LAC", 1610612747: "LAL", 1610612748: "MIA",
    1610612749: "MIL", 1610612750: "MIN", 1610612751: "BKN", 1610612752: "NYK",
    1610612753: "ORL", 1610612754: "IND", 1610612755: "PHI", 1610612756: "PHX",
    1610612757: "POR", 1610612758: "SAC", 1610612759: "SAS", 1610612760: "OKC",
    1610612761: "TOR", 1610612762: "UTA", 1610612763: "MEM", 1610612764: "WAS",
    1610612765: "DET", 1610612766: "CHA"
}

# =============================================================================
# INJURY STATUS WEIGHTS
# =============================================================================
INJURY_IMPACT = {
    "Out": 1.0,  # Full impact - player not available
    "Doubtful": 0.8,
    "Questionable": 0.5,
    "Probable": 0.2,
    "Available": 0.0
}
