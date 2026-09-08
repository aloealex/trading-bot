# models/predict.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from ..signals.ensemble import SignalEnsemble

logger = logging.getLogger(__name__)

class ModelPredictor:
    """
    Predictore per il modello di trading.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.features = []
        self.signal_engine = SignalEnsemble()
        
        if model_path:
            self.load(model_path)
    
    def load(self, path: str) -> None:
        """Carica il modello da disco."""
        import pickle
        import os
        
        with open(os.path.join(path, 'model.pkl'), 'rb') as f:
            self.model = pickle.load(f)
        
        with open(os.path.join(path, 'features.pkl'), 'rb') as f:
            self.features = pickle.load(f)
        
        logger.info(f"Modello caricato da {path}")
    
    def predict_symbol(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predice il gradimento per un singolo simbolo.
        """
        if self.model is None:
            return {'gradimento': 0.0, 'confidence': 0.0, 'error': 'Modello non caricato'}
        
        try:
            # Estrai le feature
            features = self._extract_features(df)
            
            if features is None:
                return {'gradimento': 0.0, 'confidence': 0.0, 'error': 'Feature non disponibili'}
            
            # Predici
            gradimento = self.model.predict(features.reshape(1, -1))[0]
            
            # Confidenza
            if hasattr(self.model, 'predict_with_confidence'):
                gradimento, confidence = self.model.predict_with_confidence(features.reshape(1, -1))
                confidence = confidence[0]
            else:
                confidence = 1 - abs(gradimento)
            
            return {
                'symbol': df.get('symbol', [''])[0] if 'symbol' in df.columns else '',
                'gradimento': float(gradimento),
                'confidence': float(confidence),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Errore nella predizione: {e}")
            return {'gradimento': 0.0, 'confidence': 0.0, 'error': str(e)}
    
    def predict_batch(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Predice per un batch di simboli.
        """
        results = {}
        
        for symbol, df in data.items():
            if not df.empty:
                result = self.predict_symbol(df)
                results[symbol] = result
        
        return results
    
    def _extract_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Estrae le feature da un DataFrame.
        """
        close = df['close']
        returns = df['returns'] if 'returns' in df.columns else close.pct_change()
        
        features = {}
        
        # Momentum
        for period in [5, 10, 20, 50, 125]:
            if period in self.features or not self.features:
                features[f'momentum_{period}'] = (close / close.shift(period) - 1).iloc[-1]
        
        # SMA
        for period in [20, 50, 200]:
            if f'price_vs_sma_{period}' in self.features or not self.features:
                sma = close.rolling(period).mean()
                features[f'price_vs_sma_{period}'] = (close.iloc[-1] / sma.iloc[-1] - 1)
        
        # Volatility
        for period in [20, 50]:
            if f'volatility_{period}' in self.features or not self.features:
                features[f'volatility_{period}'] = returns.iloc[-period:].std() * np.sqrt(252)
        
        # RSI
        if 'rsi_14' in self.features or not self.features:
            features['rsi_14'] = self._calculate_rsi(close).iloc[-1]
        
        # Efficiency Ratio
        if 'efficiency_ratio' in self.features or not self.features:
            features['efficiency_ratio'] = self._calculate_efficiency_ratio(close).iloc[-1]
        
        # Clenow score
        if 'clenow_score' in self.features or not self.features:
            features['clenow_score'] = self.signal_engine.clenow.calculate(close) or 0
        
        # Raschke signal
        if 'raschke_signal' in self.features or not self.features:
            features['raschke_signal'] = self.signal_engine.raschke.calculate(close) or 0
        
        # Risk metrics
        if any(f in self.features for f in ['risk_volatility', 'risk_var_95', 'risk_sharpe', 'risk_drawdown']) or not self.features:
            risk_metrics = self.signal_engine.risk.calculate_metrics(close)
            if risk_metrics:
                features['risk_volatility'] = risk_metrics.volatility
                features['risk_var_95'] = risk_metrics.var_95
                features['risk_sharpe'] = risk_metrics.sharpe_ratio
                features['risk_drawdown'] = risk_metrics.max_drawdown
        
        # Verifica che tutte le feature siano presenti
        if self.features:
            missing = [f for f in self.features if f not in features]
            if missing:
                logger.warning(f"Feature mancanti: {missing}")
                return None
        
        # Crea il vettore delle feature
        feature_names = self.features or list(features.keys())
        feature_values = [features.get(f, 0) for f in feature_names]
        
        return np.array(feature_values)
    
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