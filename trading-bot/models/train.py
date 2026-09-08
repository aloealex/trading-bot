# models/train.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pickle
import os
import logging

from .xgboost_model import XGBoostTradingModel
from .neural_network import NeuralTradingModel
from .ensemble_model import EnsembleModel
from ..signals.ensemble import SignalEnsemble

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Addestramento e gestione dei modelli.
    """
    
    def __init__(self,
                 model_type: str = 'ensemble',
                 features: Optional[List[str]] = None,
                 model_params: Optional[Dict] = None):
        self.model_type = model_type
        self.features = features or []
        self.model_params = model_params or {}
        self.model = None
        self.signal_engine = SignalEnsemble()
        self.is_trained = False
    
    def prepare_data(self, data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara i dati per l'addestramento."""
        X_list = []
        y_list = []
        
        for symbol, df in data.items():
            if df.empty:
                continue
            
            # Estrai le feature
            features = self._extract_features(df)
            X_list.append(features)
            
            # Target: rendimento futuro a 5 giorni
            y = df['close'].shift(-5) / df['close'] - 1
            y_list.append(y)
        
        X = np.vstack([x for x in X_list if len(x) > 0])
        y = np.concatenate([y for y in y_list if len(y) > 0])
        
        return X, y
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Estrae le feature da un DataFrame."""
        # Calcola le feature tecniche
        close = df['close']
        returns = df['returns']
        
        features = {}
        
        # Momentum
        for period in [5, 10, 20, 50, 125]:
            features[f'momentum_{period}'] = (close / close.shift(period) - 1)
        
        # SMA
        for period in [20, 50, 200]:
            sma = close.rolling(period).mean()
            features[f'price_vs_sma_{period}'] = close / sma - 1
        
        # Volatility
        for period in [20, 50]:
            features[f'volatility_{period}'] = returns.rolling(period).std() * np.sqrt(252)
        
        # RSI
        features['rsi_14'] = self._calculate_rsi(close)
        
        # Efficiency Ratio
        features['efficiency_ratio'] = self._calculate_efficiency_ratio(close)
        
        # Clenow score
        features['clenow_score'] = self.signal_engine.clenow.calculate(close) or 0
        
        # Raschke signal
        features['raschke_signal'] = self.signal_engine.raschke.calculate(close) or 0
        
        # Risk metrics
        risk_metrics = self.signal_engine.risk.calculate_metrics(close)
        if risk_metrics:
            features['risk_volatility'] = risk_metrics.volatility
            features['risk_var_95'] = risk_metrics.var_95
            features['risk_sharpe'] = risk_metrics.sharpe_ratio
            features['risk_drawdown'] = risk_metrics.max_drawdown
        
        # Crea il DataFrame delle feature
        df_features = pd.DataFrame(features)
        
        # Rimuovi NaN
        df_features = df_features.dropna()
        
        return df_features.values
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcola l'RSI."""
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_efficiency_ratio(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calcola l'Efficiency Ratio."""
        returns = prices.diff()
        net_move = abs(prices - prices.shift(window))
        sum_move = returns.rolling(window).apply(lambda x: abs(x).sum())
        
        return net_move / sum_move
    
    def train(self, data: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, float]:
        """Addestra il modello."""
        X, y = self.prepare_data(data)
        
        if len(X) == 0:
            logger.error("Nessun dato disponibile per l'addestramento")
            return {}
        
        # Crea il modello
        if self.model_type == 'xgboost':
            self.model = XGBoostTradingModel(**self.model_params)
        elif self.model_type == 'neural':
            self.model = NeuralTradingModel(**self.model_params)
        elif self.model_type == 'ensemble':
            self.model = EnsembleModel(**self.model_params)
        else:
            raise ValueError(f"Model type {self.model_type} non supportato")
        
        # Addestra
        metrics = self.model.train(X, y, **kwargs)
        self.is_trained = True
        
        logger.info(f"Modello {self.model_type} addestrato con successo")
        
        return metrics
    
    def save(self, path: str) -> None:
        """Salva il modello su disco."""
        os.makedirs(path, exist_ok=True)
        
        with open(os.path.join(path, 'model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(os.path.join(path, 'features.pkl'), 'wb') as f:
            pickle.dump(self.features, f)
        
        logger.info(f"Modello salvato in {path}")
    
    def load(self, path: str) -> None:
        """Carica il modello da disco."""
        with open(os.path.join(path, 'model.pkl'), 'rb') as f:
            self.model = pickle.load(f)
        
        with open(os.path.join(path, 'features.pkl'), 'rb') as f:
            self.features = pickle.load(f)
        
        self.is_trained = True
        logger.info(f"Modello caricato da {path}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Restituisce un riassunto del modello."""
        return {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'feature_count': len(self.features),
            'model_params': self.model_params,
            'feature_importance': self.model.get_feature_importance() if self.model else {}
        }