# models/xgboost_model.py
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

import xgboost as xgb

logger = logging.getLogger(__name__)


def chronological_split(X: np.ndarray, y: np.ndarray, val_fraction: float = 0.2):
    """Split temporale: le ultime osservazioni vanno in validazione (niente shuffle sulle serie storiche)."""
    split = int(len(X) * (1 - val_fraction))
    return X[:split], X[split:], y[:split], y[split:]


class XGBoostTradingModel:
    """
    Modello XGBoost che stima il rendimento futuro (target del dataset).
    La conversione in "gradimento" (-1..+1) avviene in models.predict.
    """

    name = 'xgboost'

    def __init__(self,
                 n_estimators: int = 1000,
                 max_depth: int = 4,
                 learning_rate: float = 0.01,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 min_child_weight: int = 20,
                 early_stopping_rounds: int = 50,
                 random_state: int = 42,
                 feature_names: Optional[List[str]] = None):
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'min_child_weight': min_child_weight,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            # in xgboost >= 2 early stopping si passa al costruttore, non a fit()
            'early_stopping_rounds': early_stopping_rounds,
            'random_state': random_state,
            'verbosity': 0,
        }
        self.feature_names = feature_names
        self.model = None
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Addestra il modello. Se non viene passata una validazione usa l'ultimo 20% dei dati."""
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = chronological_split(X, y)
        else:
            X_train, y_train = X, y

        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        self.is_trained = True

        pred_val = self.model.predict(X_val)
        return {
            'train_rmse': float(np.sqrt(np.mean((y_train - self.model.predict(X_train)) ** 2))),
            'val_rmse': float(np.sqrt(np.mean((y_val - pred_val) ** 2))),
            'val_corr': float(np.corrcoef(pred_val, y_val)[0, 1]) if np.std(pred_val) > 0 else 0.0,
            'best_iteration': int(getattr(self.model, 'best_iteration', self.params['n_estimators'])),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Modello non addestrato")
        return self.model.predict(np.asarray(X, dtype=float))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilità (indicativa) di rendimento positivo."""
        pred = self.predict(X)
        scale = np.std(pred) or 1.0
        prob = 1 / (1 + np.exp(-pred / scale))
        return np.vstack([1 - prob, prob]).T

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_trained or self.model is None:
            return {}
        importance = self.model.feature_importances_
        names = self.feature_names or [f'feature_{i}' for i in range(len(importance))]
        return {n: float(v) for n, v in zip(names, importance)}

    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Confidenza come accordo tra gli alberi: si confronta la predizione con quella
        ottenuta usando solo la prima metà degli alberi.
        """
        pred = self.predict(X)
        best = getattr(self.model, 'best_iteration', None) or self.params['n_estimators']
        half = self.model.predict(np.asarray(X, dtype=float), iteration_range=(0, max(1, best // 2)))
        agree = np.sign(pred) == np.sign(half)
        confidence = np.where(agree, 0.75, 0.4)
        return pred, confidence
