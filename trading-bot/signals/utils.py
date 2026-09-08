# signals/utils.py
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from datetime import datetime, timedelta

class SignalUtils:
    """Utility varie per i segnali."""
    
    @staticmethod
    def smooth_signal(signal: np.ndarray, window: int = 5) -> np.ndarray:
        """Smussa un segnale con media mobile."""
        return pd.Series(signal).rolling(window).mean().values
    
    @staticmethod
    def detect_crossovers(series1: np.ndarray, series2: np.ndarray) -> List[int]:
        """
        Rileva i crossover tra due serie.
        Returns: lista di indici dove avviene il crossover.
        """
        crossovers = []
        for i in range(1, len(series1)):
            if (series1[i-1] < series2[i-1] and series1[i] > series2[i]) or \
               (series1[i-1] > series2[i-1] and series1[i] < series2[i]):
                crossovers.append(i)
        return crossovers
    
    @staticmethod
    def normalize_signal(signal: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """
        Normalizza un segnale.
        
        Methods:
            - zscore: (x - mean) / std
            - minmax: (x - min) / (max - min)
            - tanh: tanh(x / std)
        """
        if method == 'zscore':
            return (signal - np.mean(signal)) / np.std(signal)
        elif method == 'minmax':
            return (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-8)
        elif method == 'tanh':
            return np.tanh(signal / np.std(signal))
        return signal
    
    @staticmethod
    def calculate_rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
        """Calcola lo Sharpe Ratio rolling."""
        rolling_mean = returns.rolling(window).mean() * 252
        rolling_std = returns.rolling(window).std() * np.sqrt(252)
        return rolling_mean / rolling_std
    
    @staticmethod
    def calculate_beta(prices1: pd.Series, prices2: pd.Series, window: int = 252) -> float:
        """Calcola il beta di un titolo rispetto a un benchmark."""
        if len(prices1) < window or len(prices2) < window:
            return 0.0
        
        returns1 = prices1.pct_change().dropna()
        returns2 = prices2.pct_change().dropna()
        
        # Allinea le serie
        min_len = min(len(returns1), len(returns2), window)
        returns1 = returns1.iloc[-min_len:]
        returns2 = returns2.iloc[-min_len:]
        
        covariance = np.cov(returns1, returns2)[0, 1]
        variance = np.var(returns2)
        
        return covariance / variance if variance > 0 else 0.0
    
    @staticmethod
    def calculate_correlation(prices1: pd.Series, prices2: pd.Series, window: int = 60) -> float:
        """Calcola la correlazione rolling tra due titoli."""
        if len(prices1) < window or len(prices2) < window:
            return 0.0
        
        returns1 = prices1.pct_change().dropna()
        returns2 = prices2.pct_change().dropna()
        
        return returns1.iloc[-window:].corr(returns2.iloc[-window:])
    
    @staticmethod
    def calculate_risk_adjusted_rank(scores: pd.Series, risk_metrics: pd.DataFrame) -> pd.Series:
        """
        Calcola un ranking corretto per il rischio.
        """
        if scores.empty or risk_metrics.empty:
            return pd.Series()
        
        # Sharpe-like score: rendimento / rischio
        risk_adjusted = scores / (risk_metrics['volatility'] + 0.01)
        
        return risk_adjusted.sort_values(ascending=False)