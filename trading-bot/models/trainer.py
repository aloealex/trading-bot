# models/trainer.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .data_pipeline import DataPipeline
from .xgboost_model import XGBoostTradingModel
from .neural_network import NeuralTradingModel
from .ensemble_model import EnsembleModel
from .hyperopt import HyperparameterOptimizer

class ModelTrainer:
    """
    Addestramento e gestione dei modelli.
    """
    
    def __init__(self,
                 data: pd.DataFrame,
                 model_type: str = 'ensemble',
                 test_size: float = 0.2,
                 validation_size: float = 0.2,
                 random_state: int = 42):
        
        self.data = data
        self.model_type = model_type
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        
        self.pipeline = DataPipeline()
        self.model = None
        self.metrics = None
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepara i dati per l'addestramento.
        """
        # Crea feature
        df = self.pipeline.create_features(self.data)
        
        # Prepara i dati
        X, y = self.pipeline.prepare_data(df)
        
        # Split temporale (no leakage)
        split_idx = int(len(X) * (1 - self.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Validation split
        val_idx = int(len(X_train) * (1 - self.validation_size))
        X_val, X_train_final = X_train[val_idx:], X_train[:val_idx]
        y_val, y_train_final = y_train[val_idx:], y_train[:val_idx]
        
        return X_train_final, X_val, X_test, y_train_final, y_val, y_test
    
    def train(self) -> Dict[str, float]:
        """
        Addestra il modello.
        """
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data()
        
        # Crea il modello
        if self.model_type == 'xgboost':
            self.model = XGBoostTradingModel()
        elif self.model_type == 'neural':
            self.model = NeuralTradingModel()
        elif self.model_type == 'ensemble':
            self.model = EnsembleModel()
        else:
            raise ValueError(f"Model type {self.model_type} non supportato")
        
        # Addestra
        train_metrics = self.model.train(X_train, y_train, X_val, y_val)
        
        # Valuta
        self.metrics = self.model.evaluate(X_test, y_test)
        
        return train_metrics
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Ottimizza gli iperparametri.
        """
        optimizer = HyperparameterOptimizer()
        
        if self.model_type == 'xgboost':
            return optimizer.optimize_xgboost(X, y)
        elif self.model_type == 'neural':
            return optimizer.optimize_neural_network(X, y)
        elif self.model_type == 'ensemble':
            return optimizer.optimize_ensemble(X, y)
        else:
            raise ValueError(f"Model type {self.model_type} non supportato")
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Valuta il modello su nuovi dati.
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        return self.model.evaluate(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice il gradimento.
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        return self.model.predict(X)
    
    def save(self, path: str):
        """
        Salva il modello e la pipeline.
        """
        import pickle
        import os
        
        os.makedirs(path, exist_ok=True)
        
        # Salva il modello
        self.model.save(os.path.join(path, 'model.pkl'))
        
        # Salva la pipeline
        with open(os.path.join(path, 'pipeline.pkl'), 'wb') as f:
            pickle.dump(self.pipeline, f)
        
        # Salva le metriche
        if self.metrics:
            pd.Series(self.metrics.__dict__).to_csv(os.path.join(path, 'metrics.csv'))
    
    @classmethod
    def load(cls, path: str):
        """
        Carica un modello salvato.
        """
        import pickle
        
        # Carica il modello
        from .base import BaseTradingModel
        model = BaseTradingModel.load(os.path.join(path, 'model.pkl'))
        
        # Carica la pipeline
        with open(os.path.join(path, 'pipeline.pkl'), 'rb') as f:
            pipeline = pickle.load(f)
        
        # Crea il trainer
        trainer = cls(pd.DataFrame())
        trainer.model = model
        trainer.pipeline = pipeline
        
        return trainer
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Restituisce un riassunto del modello.
        """
        summary = {
            'model_type': self.model_type,
            'is_trained': self.model.is_trained if self.model else False,
            'feature_count': len(self.pipeline.feature_names) if self.pipeline else 0,
            'metrics': self.metrics.__dict__ if self.metrics else None,
            'feature_importance': self.model.get_feature_importance() if self.model else None
        }
        
        return summary