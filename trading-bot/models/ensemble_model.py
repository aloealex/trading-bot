# models/ensemble_model.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression

from .base import BaseTradingModel
from .xgboost_model import XGBoostTradingModel
from .neural_network import NeuralTradingModel

class EnsembleModel(BaseTradingModel):
    """
    Ensemble di modelli per la predizione del gradimento.
    
    Combina:
    - XGBoost (robusto, feature importance)
    - Rete Neurale (non lineare, pattern complessi)
    - Media ponderata basata sulle performance recenti
    """
    
    def __init__(self,
                 models: Optional[List[BaseTradingModel]] = None,
                 weights: Optional[Dict[str, float]] = None,
                 meta_model: Optional[BaseTradingModel] = None):
        
        super().__init__("Ensemble")
        
        self.models = models or [
            XGBoostTradingModel(n_estimators=500, max_depth=4),
            NeuralTradingModel(hidden_dims=[256, 128, 64], epochs=50)
        ]
        
        self.weights = weights or {}
        self.meta_model = meta_model
        self.is_trained = False
        self.performance_history = []
    
    def train(self, X: np.ndarray, y: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Addestra tutti i modelli dell'ensemble.
        """
        # Addestra ogni modello
        for model in self.models:
            model.train(X, y, X_val, y_val)
            self.is_trained = True
        
        # Addestra il meta-modello (stacking)
        if self.meta_model is not None:
            # Genera le predizioni dei modelli base
            base_predictions = self._get_base_predictions(X)
            
            # Addestra il meta-modello
            self.meta_model.train(base_predictions, y, X_val, y_val)
        
        # Calcola i pesi basati sulle performance
        self._update_weights(X_val, y_val if y_val is not None else y)
        
        return {'n_models': len(self.models)}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice il gradimento usando l'ensemble.
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Predizioni di tutti i modelli
        predictions = self._get_base_predictions(X)
        
        # Se abbiamo un meta-modello, usalo
        if self.meta_model is not None:
            return self.meta_model.predict(predictions)
        
        # Altrimenti, media ponderata
        if self.weights:
            weighted_sum = 0
            total_weight = 0
            for i, model in enumerate(self.models):
                weight = self.weights.get(model.name, 1.0 / len(self.models))
                weighted_sum += weight * predictions[:, i]
                total_weight += weight
            return weighted_sum / total_weight
        
        # Media semplice
        return np.mean(predictions, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predice le probabilità.
        """
        predictions = self.predict(X)
        
        # Converti in probabilità
        prob_buy = (predictions + 1) / 2
        prob_sell = 1 - prob_buy
        
        return np.vstack([prob_sell, prob_buy]).T
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Aggrega le importanze delle feature.
        """
        importance = {}
        
        for model in self.models:
            model_importance = model.get_feature_importance()
            for feature, imp in model_importance.items():
                if feature not in importance:
                    importance[feature] = 0
                importance[feature] += imp
        
        # Normalizza
        total = sum(importance.values())
        if total > 0:
            for feature in importance:
                importance[feature] /= total
        
        return importance
    
    def _get_base_predictions(self, X: np.ndarray) -> np.ndarray:
        """Ottiene le predizioni di tutti i modelli base."""
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))
        return np.column_stack(predictions)
    
    def _update_weights(self, X: np.ndarray, y: np.ndarray):
        """
        Aggiorna i pesi basati sulle performance recenti.
        """
        predictions = self._get_base_predictions(X)
        
        # Calcola l'errore per ogni modello
        errors = []
        for i in range(len(self.models)):
            mse = np.mean((predictions[:, i] - y) ** 2)
            errors.append(mse)
        
        # Pesi inversamente proporzionali all'errore
        inv_errors = [1 / (e + 1e-8) for e in errors]
        total = sum(inv_errors)
        
        for i, model in enumerate(self.models):
            self.weights[model.name] = inv_errors[i] / total
    
    def get_ensemble_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Calcola le metriche dell'ensemble.
        """
        predictions = self.predict(X)
        
        return {
            'mse': np.mean((predictions - y) ** 2),
            'mae': np.mean(np.abs(predictions - y)),
            'correlation': np.corrcoef(predictions, y)[0, 1],
            'model_weights': self.weights
        }

class AdaptiveWeightedEnsemble(EnsembleModel):
    """
    Ensemble con pesi che si adattano alle condizioni di mercato.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.market_regime_weights = {}
    
    def predict(self, X: np.ndarray, market_regime: str = 'neutral') -> np.ndarray:
        """
        Predice usando pesi specifici per regime di mercato.
        """
        if market_regime in self.market_regime_weights:
            weights = self.market_regime_weights[market_regime]
        else:
            weights = self.weights
        
        predictions = self._get_base_predictions(X)
        
        weighted_sum = 0
        total_weight = 0
        for i, model in enumerate(self.models):
            weight = weights.get(model.name, 1.0 / len(self.models))
            weighted_sum += weight * predictions[:, i]
            total_weight += weight
        
        return weighted_sum / total_weight
    
    def update_regime_weights(self, X: np.ndarray, y: np.ndarray, regimes: List[str]):
        """
        Aggiorna i pesi per ogni regime di mercato.
        """
        unique_regimes = set(regimes)
        
        for regime in unique_regimes:
            idx = [i for i, r in enumerate(regimes) if r == regime]
            if idx:
                X_regime = X[idx]
                y_regime = y[idx]
                
                # Calcola i pesi per questo regime
                predictions = self._get_base_predictions(X_regime)
                
                errors = []
                for i in range(len(self.models)):
                    mse = np.mean((predictions[:, i] - y_regime) ** 2)
                    errors.append(mse)
                
                inv_errors = [1 / (e + 1e-8) for e in errors]
                total = sum(inv_errors)
                
                weights = {}
                for i, model in enumerate(self.models):
                    weights[model.name] = inv_errors[i] / total
                
                self.market_regime_weights[regime] = weights