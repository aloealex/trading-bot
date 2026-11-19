# models/ensemble_model.py
"""
Ensemble di modelli: XGBoost + rete neurale (se PyTorch è installato).
I pesi sono inversamente proporzionali all'errore sul set di VALIDAZIONE.
"""
import logging
from typing import Dict, List, Optional

import numpy as np

from .xgboost_model import XGBoostTradingModel, chronological_split
from .neural_network import NeuralTradingModel, TORCH_AVAILABLE

logger = logging.getLogger(__name__)


class EnsembleModel:
    name = 'ensemble'

    def __init__(self, models: Optional[List] = None, feature_names: Optional[List[str]] = None):
        if models is None:
            models = [XGBoostTradingModel(feature_names=feature_names)]
            if TORCH_AVAILABLE:
                models.append(NeuralTradingModel(feature_names=feature_names))
            else:
                logger.warning("PyTorch non installato: l'ensemble userà solo XGBoost")
        self.models = models
        self.feature_names = feature_names
        self.weights: Dict[str, float] = {}
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        if X_val is None or y_val is None:
            X, X_val, y, y_val = chronological_split(np.asarray(X), np.asarray(y))

        metrics = {}
        for model in self.models:
            for k, v in model.train(X, y, X_val, y_val).items():
                metrics[f'{model.name}_{k}'] = v
        self.is_trained = True

        # pesi dall'errore di validazione (prima si usava X_val=None e il codice andava in errore)
        preds = self._get_base_predictions(X_val)
        errors = [np.mean((preds[:, i] - y_val) ** 2) for i in range(len(self.models))]
        inv = [1 / (e + 1e-12) for e in errors]
        total = sum(inv)
        self.weights = {m.name: w / total for m, w in zip(self.models, inv)}
        ens = self.predict(X_val)
        metrics['val_corr'] = float(np.corrcoef(ens, y_val)[0, 1]) if np.std(ens) > 0 else 0.0
        metrics['weights'] = dict(self.weights)
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Modello non addestrato")
        preds = self._get_base_predictions(X)
        w = np.array([self.weights.get(m.name, 1 / len(self.models)) for m in self.models])
        return preds @ (w / w.sum())

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pred = self.predict(X)
        scale = np.std(pred) or 1.0
        prob = 1 / (1 + np.exp(-pred / scale))
        return np.vstack([1 - prob, prob]).T

    def predict_with_confidence(self, X: np.ndarray):
        """Confidenza alta quando i modelli concordano sul segno della previsione."""
        preds = self._get_base_predictions(X)
        signs = np.sign(preds)
        agreement = np.abs(signs.mean(axis=1))  # 1 = tutti d'accordo
        confidence = 0.4 + 0.5 * agreement
        return self.predict(X), confidence

    def get_feature_importance(self) -> Dict[str, float]:
        importance: Dict[str, float] = {}
        for model in self.models:
            imp = model.get_feature_importance()
            total = sum(imp.values()) or 1.0
            weight = self.weights.get(model.name, 1 / len(self.models))
            for k, v in imp.items():
                importance[k] = importance.get(k, 0.0) + weight * v / total
        return dict(sorted(importance.items(), key=lambda kv: -kv[1]))

    def _get_base_predictions(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([m.predict(X) for m in self.models])
