# signals/base.py
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SignalResult:
    """Risultato standardizzato di un segnale"""
    symbol: str
    timestamp: datetime
    value: float
    confidence: float = 0.5
    components: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseSignal(ABC):
    """Classe base per tutti i generatori di segnale"""
    
    def __init__(self, name: str):
        self.name = name
        self._cache = {}
    
    @abstractmethod
    def calculate(self, prices: pd.Series, **kwargs) -> Optional[float]:
        """Calcola il segnale da una serie di prezzi"""
        pass
    
    @abstractmethod
    def explain(self, prices: pd.Series) -> Dict[str, Any]:
        """Spiega il ragionamento dietro il segnale"""
        pass
    
    def validate_input(self, prices: pd.Series, min_length: int = 50) -> bool:
        """Valida i dati di input"""
        if prices is None or len(prices) < min_length:
            return False
        if prices.isnull().any():
            return False
        if (prices <= 0).any():
            return False
        return True
    
    def normalize(self, value: float, min_val: float = -1, max_val: float = 1) -> float:
        """Normalizza un valore in un range specificato"""
        return np.clip(value, min_val, max_val)

class SignalUtils:
    """Utility per i segnali"""
    
    @staticmethod
    def calculate_returns(prices: pd.Series, period: int = 1) -> pd.Series:
        """Calcola i rendimenti"""
        return prices.pct_change(period)
    
    @staticmethod
    def calculate_log_returns(prices: pd.Series) -> pd.Series:
        """Calcola i rendimenti logaritmici"""
        return np.log(prices / prices.shift(1))
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
        """Calcola la volatilità rolling"""
        return returns.rolling(window).std() * np.sqrt(252)
    
    @staticmethod
    def calculate_sharpe(returns: pd.Series, window: int = 252) -> float:
        """Calcola lo Sharpe Ratio"""
        if len(returns) < window:
            return 0.0
        rolling_returns = returns.iloc[-window:]
        if rolling_returns.std() == 0:
            return 0.0
        return (rolling_returns.mean() / rolling_returns.std()) * np.sqrt(252)
    
    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> float:
        """Calcola il drawdown massimo"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative / running_max - 1)
        return drawdown.min()
    
    @staticmethod
    def calculate_efficiency_ratio(prices: pd.Series, window: int = 20) -> float:
        """
        Calcola l'Efficiency Ratio di Kaufman.
        ER = |Prezzo_Oggi - Prezzo_N_giorni_fa| / Somma_movimenti_giornalieri
        """
        if len(prices) < window:
            return 0.0
        
        returns = prices.diff()
        net_move = abs(prices.iloc[-1] - prices.iloc[-window])
        sum_move = returns.iloc[-window:].abs().sum()
        
        if sum_move == 0:
            return 0.0
        
        return net_move / sum_move