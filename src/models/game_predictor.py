"""
NBA ML Prediction System - Game Predictor
==========================================
XGBoost + LightGBM ensemble for game win prediction.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging

from src.config import MODEL_CONFIG, MODELS_DIR
# Import preprocessing so pickle can find DataPreprocessor class
from src.preprocessing import DataPreprocessor, GameDatasetBuilder

logger = logging.getLogger(__name__)

# =============================================================================
# GAME PREDICTOR MODEL
# =============================================================================
class GamePredictor:
    """
    Ensemble model for predicting game outcomes.
    Uses XGBoost + LightGBM with weighted averaging.
    """
    
    def __init__(self, 
                 xgb_weight: float = 0.5,
                 lgb_weight: float = 0.5):
        self.xgb_weight = xgb_weight
        self.lgb_weight = lgb_weight
        
        self.xgb_model = None
        self.lgb_model = None
        self.feature_columns = None
        self.trained = False
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              feature_columns: List[str] = None):
        """
        Train both XGBoost and LightGBM models.
        """
        self.feature_columns = feature_columns
        
        logger.info("Training XGBoost model...")
        self.xgb_model = xgb.XGBClassifier(**MODEL_CONFIG.xgb_params)
        
        if X_val is not None:
            self.xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.xgb_model.fit(X_train, y_train)
        
        logger.info("Training LightGBM model...")
        self.lgb_model = lgb.LGBMClassifier(**MODEL_CONFIG.lgb_params)
        
        if X_val is not None:
            self.lgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)]
            )
        else:
            self.lgb_model.fit(X_train, y_train)
        
        self.trained = True
        logger.info("Training complete!")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict win probabilities using ensemble.
        
        Returns:
            Array of shape (n_samples, 2) with [loss_prob, win_prob]
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        xgb_proba = self.xgb_model.predict_proba(X)
        lgb_proba = self.lgb_model.predict_proba(X)
        
        # Weighted average
        ensemble_proba = (
            self.xgb_weight * xgb_proba + 
            self.lgb_weight * lgb_proba
        )
        
        return ensemble_proba
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict win/loss (1/0)."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
    
    def predict_with_confidence(self, X: np.ndarray) -> List[Dict]:
        """
        Predict with detailed confidence information.
        Shows individual model predictions and disagreement.
        """
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        lgb_proba = self.lgb_model.predict_proba(X)[:, 1]
        ensemble_proba = self.predict_proba(X)[:, 1]
        
        results = []
        for i in range(len(X)):
            # Check model disagreement
            disagreement = abs(xgb_proba[i] - lgb_proba[i])
            
            results.append({
                "win_probability": ensemble_proba[i],
                "xgb_probability": xgb_proba[i],
                "lgb_probability": lgb_proba[i],
                "model_disagreement": disagreement,
                "confidence": "high" if disagreement < 0.1 else ("medium" if disagreement < 0.2 else "low"),
                "prediction": "WIN" if ensemble_proba[i] >= 0.5 else "LOSS"
            })
        
        return results
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive model evaluation.
        
        Returns:
            Dict with accuracy, brier score, and other metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "brier_score": brier_score_loss(y, y_proba),
            "log_loss": log_loss(y, y_proba)
        }
        
        # Individual model metrics
        xgb_pred = self.xgb_model.predict(X)
        lgb_pred = self.lgb_model.predict(X)
        
        metrics["xgb_accuracy"] = accuracy_score(y, xgb_pred)
        metrics["lgb_accuracy"] = accuracy_score(y, lgb_pred)
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from both models."""
        if not self.trained or self.feature_columns is None:
            return pd.DataFrame()
        
        xgb_importance = self.xgb_model.feature_importances_
        lgb_importance = self.lgb_model.feature_importances_
        
        df = pd.DataFrame({
            "feature": self.feature_columns,
            "xgb_importance": xgb_importance,
            "lgb_importance": lgb_importance,
            "avg_importance": (xgb_importance + lgb_importance) / 2
        })
        
        return df.sort_values("avg_importance", ascending=False)
    
    def explain_prediction(self, X: np.ndarray, top_n: int = 5) -> List[Dict]:
        """
        Explain predictions using feature importance.
        Returns top N contributing features for each prediction.
        """
        if not self.trained or self.feature_columns is None:
            return []
        
        importance = self.get_feature_importance()
        top_features = importance.head(top_n)["feature"].tolist()
        
        explanations = []
        for i in range(len(X)):
            feature_contributions = []
            for j, feat in enumerate(self.feature_columns):
                if feat in top_features:
                    feature_contributions.append({
                        "feature": feat,
                        "value": X[i, j],
                        "importance": importance[importance["feature"] == feat]["avg_importance"].values[0]
                    })
            
            # Sort by importance
            feature_contributions.sort(key=lambda x: x["importance"], reverse=True)
            
            explanations.append({
                "top_features": feature_contributions[:top_n],
                "prediction": self.predict(X[i:i+1])[0]
            })
        
        return explanations
    
    def save(self, path: Path = None):
        """Save model to disk."""
        if path is None:
            path = MODELS_DIR / "game_predictor.joblib"
        
        joblib.dump({
            "xgb_model": self.xgb_model,
            "lgb_model": self.lgb_model,
            "xgb_weight": self.xgb_weight,
            "lgb_weight": self.lgb_weight,
            "feature_columns": self.feature_columns,
            "trained": self.trained
        }, path)
        logger.info(f"Saved model to {path}")
    
    def load(self, path: Path = None):
        """Load model from disk."""
        if path is None:
            path = MODELS_DIR / "game_predictor.joblib"
        
        data = joblib.load(path)
        self.xgb_model = data["xgb_model"]
        self.lgb_model = data["lgb_model"]
        self.xgb_weight = data["xgb_weight"]
        self.lgb_weight = data["lgb_weight"]
        self.feature_columns = data["feature_columns"]
        self.trained = data["trained"]
        logger.info(f"Loaded model from {path}")


# =============================================================================
# TRAINING PIPELINE
# =============================================================================
def train_game_predictor(dataset: Dict) -> GamePredictor:
    """
    Full training pipeline for game predictor.
    """
    logger.info("Starting game predictor training...")
    
    model = GamePredictor()
    model.train(
        X_train=dataset["X_train"],
        y_train=dataset["y_train"],
        X_val=dataset["X_val"],
        y_val=dataset["y_val"],
        feature_columns=dataset["feature_columns"]
    )
    
    # Evaluate on all splits
    logger.info("\n=== Training Metrics ===")
    train_metrics = model.evaluate(dataset["X_train"], dataset["y_train"])
    logger.info(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
    
    logger.info("\n=== Validation Metrics ===")
    val_metrics = model.evaluate(dataset["X_val"], dataset["y_val"])
    logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
    logger.info(f"Val Brier Score: {val_metrics['brier_score']:.4f}")
    
    logger.info("\n=== Test Metrics ===")
    test_metrics = model.evaluate(dataset["X_test"], dataset["y_test"])
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Brier Score: {test_metrics['brier_score']:.4f}")
    
    # Check if we meet target
    if test_metrics["accuracy"] >= 0.65:
        logger.info("✓ Target accuracy (>65%) achieved!")
    else:
        logger.warning(f"✗ Below target accuracy. Got {test_metrics['accuracy']:.2%}")
    
    # Feature importance
    logger.info("\n=== Top Features ===")
    importance = model.get_feature_importance()
    print(importance.head(10))
    
    # Save model
    model.save()
    
    return model


# =============================================================================
# CLI INTERFACE
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Game Predictor Training")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate existing model")
    
    args = parser.parse_args()
    
    if args.train:
        from src.preprocessing import GameDatasetBuilder
        
        logging.basicConfig(level=logging.INFO)
        
        print("Loading dataset...")
        builder = GameDatasetBuilder()
        
        try:
            dataset = builder.load_dataset()
            print(f"Loaded dataset with {len(dataset['feature_columns'])} features")
        except FileNotFoundError:
            print("No dataset found. Please run 'python -m src.preprocessing --build' first.")
            exit(1)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
        
        model = train_game_predictor(dataset)
        print("\nTraining complete!")
    
    elif args.evaluate:
        model = GamePredictor()
        model.load()
        
        from src.preprocessing import GameDatasetBuilder
        builder = GameDatasetBuilder()
        dataset = builder.load_dataset()
        
        metrics = model.evaluate(dataset["X_test"], dataset["y_test"])
        print("\n=== Test Metrics ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
    else:
        print("Use --train to train or --evaluate to evaluate")
