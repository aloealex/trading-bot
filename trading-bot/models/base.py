# models/base.py
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pickle
import json
import os

@dataclass
class ModelPrediction:
    """Predizione del modello"""
    symbol: str
    timestamp: datetime
    gradimento: float          # -1 a +1
    confidence: float          # 0 a 1
    probabilities: Dict[str, float]  # Probabilità per classe
    features: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelMetrics:
    """Metriche di performance del modello"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_return: float
    std_return: float
    confidence_interval: Tuple[float, float]

class BaseTradingModel(ABC):
    """Classe base per tutti i modelli di trading"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.is_trained = False
        self.feature_importance = None
        self.metrics = None
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, 
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Addestra il modello"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice il gradimento per nuovi dati"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predice le probabilità delle classi"""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Restituisce l'importanza delle feature"""
        pass
    
    def save(self, path: str):
        """Salva il modello su disco"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str):
        """Carica il modello da disco"""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Valuta le performance del modello"""
        predictions = self.predict(X)
        
        # Metriche di classificazione
        from sklearn.metrics import (accuracy_score, precision_score, 
                                    recall_score, f1_score, roc_auc_score)
        
        # Converti i target in classi (se necessario)
        y_class = np.where(y > 0, 1, 0)
        pred_class = np.where(predictions > 0, 1, 0)
        
        accuracy = accuracy_score(y_class, pred_class)
        precision = precision_score(y_class, pred_class, zero_division=0)
        recall = recall_score(y_class, pred_class, zero_division=0)
        f1 = f1_score(y_class, pred_class, zero_division=0)
        
        try:
            auc = roc_auc_score(y_class, predictions)
        except:
            auc = 0.5
        
        # Metriche di trading
        returns = predictions * 0.01  # Assumiamo 1% di rendimento per predizione
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        cumulative = (1 + returns).cumprod()
        max_drawdown = (cumulative / cumulative.expanding().max() - 1).min()
        win_rate = (returns > 0).mean()
        avg_return = returns.mean()
        std_return = returns.std()
        
        # Intervallo di confidenza
        ci_lower = np.percentile(returns, 2.5)
        ci_upper = np.percentile(returns, 97.5)
        
        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            avg_return=avg_return,
            std_return=std_return,
            confidence_interval=(ci_lower, ci_upper)
        )