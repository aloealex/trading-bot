# models/xgboost_model.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class XGBoostTradingModel:
    """
    Modello XGBoost per la predizione del gradimento.
    """
    
    def __init__(self,
                 n_estimators: int = 1000,
                 max_depth: int = 6,
                 learning_rate: float = 0.01,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 random_state: int = 42):
        
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'random_state': random_state,
            'verbosity': 0
        }
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Addestra il modello XGBoost."""
        X_scaled = self.scaler.fit_transform(X)
        
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
        else:
            X_train = X_scaled
            X_val = self.scaler.transform(X_val)
            y_train = y
            y_val = y_val
        
        self.model = xgb.XGBRegressor(**self.params)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        self.is_trained = True
        
        return {
            'train_rmse': np.sqrt(np.mean((y_train - self.model.predict(X_train)) ** 2)),
            'val_rmse': np.sqrt(np.mean((y_val - self.model.predict(X_val)) ** 2)),
            'best_iteration': self.model.best_iteration if hasattr(self.model, 'best_iteration') else len(self.model.estimators_)
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice il gradimento."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return np.clip(predictions, -1, 1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predice le probabilità."""
        predictions = self.predict(X)
        prob = 1 / (1 + np.exp(-predictions * 3))
        return np.vstack([1 - prob, prob]).T
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Restituisce l'importanza delle feature."""
        if not self.is_trained or self.model is None:
            return {}
        
        importance = self.model.feature_importances_
        return {f'feature_{i}': imp for i, imp in enumerate(importance)}
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predice con confidenza."""
        predictions = self.predict(X)
        
        if hasattr(self.model, 'get_booster'):
            booster = self.model.get_booster()
            # Simula la confidenza
            confidence = 1 - np.abs(predictions) * 0.5
        else:
            confidence = 1 - np.abs(predictions)
        
        return predictions, np.clip(confidence, 0, 1)