"""
NBA ML Prediction System - Preprocessing
=========================================
Data cleaning and transformation with:
- Time-aware train/val/test splits
- Dynamic feature detection (uses ALL available features)
- Missing value handling
- Feature scaling
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import logging

from src.config import MODEL_CONFIG, PROCESSED_DATA_DIR, MODELS_DIR

logger = logging.getLogger(__name__)

# =============================================================================
# COLUMNS TO EXCLUDE FROM FEATURES
# =============================================================================
EXCLUDE_COLUMNS = [
    "GAME_ID", "TEAM_ID", "GAME_DATE", "SEASON_ID", "SEASON", 
    "WL", "target", "MATCHUP", "TEAM_NAME", "TEAM_ABBREVIATION",
    "PLAYER_ID", "PLAYER_NAME"
]


# =============================================================================
# SEASON-BASED SPLITTER (NO DATA LEAKAGE)
# =============================================================================
class SeasonBasedSplitter:
    """Splits data by season to prevent data leakage."""
    
    def __init__(self, 
                 test_seasons: List[str] = None,
                 val_seasons: List[str] = None):
        self.test_seasons = test_seasons or MODEL_CONFIG.test_seasons
        self.val_seasons = val_seasons or MODEL_CONFIG.val_seasons
    
    def split(self, df: pd.DataFrame, 
              season_column: str = "SEASON") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Extract season from SEASON_ID if needed
        if season_column not in df.columns and "SEASON_ID" in df.columns:
            df = df.copy()
            df[season_column] = df["SEASON_ID"].apply(self._parse_season_id)
        
        test_mask = df[season_column].isin(self.test_seasons)
        val_mask = df[season_column].isin(self.val_seasons)
        train_mask = ~(test_mask | val_mask)
        
        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        test_df = df[test_mask].copy()
        
        logger.info(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def _parse_season_id(self, season_id: str) -> str:
        if isinstance(season_id, str) and len(season_id) == 5:
            year = int(season_id[1:])
            return f"{year}-{str(year+1)[-2:]}"
        return str(season_id)


# =============================================================================
# DATA PREPROCESSOR
# =============================================================================
class DataPreprocessor:
    """Handles missing values, scaling, and data preparation."""
    
    def __init__(self, feature_columns: List[str] = None):
        self.feature_columns = feature_columns
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.fitted = False
    
    def fit(self, df: pd.DataFrame, feature_columns: List[str] = None):
        if feature_columns:
            self.feature_columns = feature_columns
        
        X = df[self.feature_columns].values
        X_imputed = self.imputer.fit_transform(X)
        self.scaler.fit(X_imputed)
        
        self.fitted = True
        logger.info(f"Preprocessor fitted on {len(self.feature_columns)} features")
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        X = df[self.feature_columns].values
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        
        return X_scaled
    
    def fit_transform(self, df: pd.DataFrame, feature_columns: List[str] = None) -> np.ndarray:
        self.fit(df, feature_columns)
        return self.transform(df)
    
    def save(self, path: Path = None):
        if path is None:
            path = MODELS_DIR / "preprocessor.joblib"
        
        joblib.dump({
            "feature_columns": self.feature_columns,
            "scaler": self.scaler,
            "imputer": self.imputer
        }, path)
        logger.info(f"Saved preprocessor to {path}")
    
    def load(self, path: Path = None):
        if path is None:
            path = MODELS_DIR / "preprocessor.joblib"
        
        data = joblib.load(path)
        self.feature_columns = data["feature_columns"]
        self.scaler = data["scaler"]
        self.imputer = data["imputer"]
        self.fitted = True
        logger.info(f"Loaded preprocessor from {path}")


# =============================================================================
# DATASET BUILDER - USES ALL AVAILABLE FEATURES
# =============================================================================
class GameDatasetBuilder:
    """Builds train/val/test datasets using ALL available features."""
    
    def __init__(self):
        self.splitter = SeasonBasedSplitter()
        self.preprocessor = DataPreprocessor()
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Dynamically detect ALL numeric feature columns.
        Excludes ID columns, target, and non-numeric columns.
        """
        feature_columns = []
        
        for col in df.columns:
            # Skip excluded columns
            if col in EXCLUDE_COLUMNS:
                continue
            
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            # Skip columns with all NaN
            if df[col].isna().all():
                continue
            
            feature_columns.append(col)
        
        return sorted(feature_columns)
    
    def build_dataset(self, features_df: pd.DataFrame, 
                       target_column: str = "WL",
                       use_all_features: bool = True) -> Dict:
        """
        Build complete dataset for training.
        
        Args:
            features_df: DataFrame with features
            target_column: Column to predict
            use_all_features: If True, uses ALL available numeric features
        """
        # Remove rows without target
        df = features_df.dropna(subset=[target_column]).copy()
        
        # Convert WL to binary
        df["target"] = (df[target_column] == "W").astype(int)
        
        # Split by season
        train_df, val_df, test_df = self.splitter.split(df)
        
        # Get feature columns - USE ALL AVAILABLE
        if use_all_features:
            feature_columns = self._get_feature_columns(df)
            logger.info(f"Detected {len(feature_columns)} numeric feature columns")
        else:
            # Fallback to basic features
            feature_columns = [
                "team_elo", "opponent_elo", "elo_diff", "elo_win_prob", "is_home",
                "PTS_last5", "PTS_last10", "AST_last5", "REB_last5",
                "win_pct_season", "days_rest", "back_to_back"
            ]
            feature_columns = [c for c in feature_columns if c in df.columns]
        
        logger.info(f"\n=== FEATURES USED FOR TRAINING ({len(feature_columns)} total) ===")
        for i, col in enumerate(feature_columns):
            logger.info(f"  {i+1:3}. {col}")
        
        # Fit preprocessor on training data
        self.preprocessor.fit(train_df, feature_columns)
        
        # Transform all splits
        X_train = self.preprocessor.transform(train_df)
        X_val = self.preprocessor.transform(val_df)
        X_test = self.preprocessor.transform(test_df)
        
        y_train = train_df["target"].values
        y_val = val_df["target"].values
        y_test = test_df["target"].values
        
        logger.info(f"\n=== DATASET SUMMARY ===")
        logger.info(f"  Training samples: {len(y_train)}")
        logger.info(f"  Validation samples: {len(y_val)}")
        logger.info(f"  Test samples: {len(y_test)}")
        logger.info(f"  Features: {len(feature_columns)}")
        
        return {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "feature_columns": feature_columns,
            "preprocessor": self.preprocessor,
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df
        }
    
    def save_dataset(self, dataset: Dict, name: str = "game_dataset"):
        path = PROCESSED_DATA_DIR / f"{name}.joblib"
        joblib.dump(dataset, path)
        logger.info(f"Saved dataset to {path}")
    
    def load_dataset(self, name: str = "game_dataset") -> Dict:
        path = PROCESSED_DATA_DIR / f"{name}.joblib"
        return joblib.load(path)


# =============================================================================
# CLI INTERFACE
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocessing")
    parser.add_argument("--build", action="store_true", help="Build dataset from features")
    parser.add_argument("--test", action="store_true", help="Run tests")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.build:
        print("=== Building Dataset from Features ===")
        
        features_path = PROCESSED_DATA_DIR / "game_features.parquet"
        
        if not features_path.exists():
            print(f"ERROR: Features not found at {features_path}")
            print("Run 'python -m src.feature_engineering --process' first.")
            exit(1)
        
        print(f"Loading features from {features_path}...")
        features_df = pd.read_parquet(features_path)
        print(f"Loaded {len(features_df)} rows")
        
        builder = GameDatasetBuilder()
        dataset = builder.build_dataset(features_df, use_all_features=True)
        builder.save_dataset(dataset)
        
        print(f"\n✅ Dataset saved!")
        print(f"   Training samples: {len(dataset['y_train'])}")
        print(f"   Features used: {len(dataset['feature_columns'])}")
    
    elif args.test:
        print("Testing Season-Based Splitter...")
        sample_data = pd.DataFrame({
            "SEASON": ["2022-23"] * 100 + ["2023-24"] * 50 + ["2024-25"] * 25,
            "feature1": np.random.randn(175),
            "WL": np.random.choice(["W", "L"], 175)
        })
        
        splitter = SeasonBasedSplitter()
        train, val, test = splitter.split(sample_data)
        
        print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    else:
        print("Use --build to build dataset or --test to run tests")
