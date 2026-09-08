# signals/clenow.py
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class ClenowMomentum:
    """
    Momentum Score di Andreas Clenow.
    
    Formula: Momentum = Annualized_Slope * R²
    Dove:
        - Annualized_Slope = (exp(slope) ^ 252 - 1) * 100
        - R² = coefficiente di determinazione della regressione
    """
    
    def __init__(self, window: int = 125, min_observations: int = 30):
        self.window = window
        self.min_observations = min_observations
    
    def calculate(self, prices: pd.Series) -> Optional[float]:
        """
        Calcola il momentum score.
        """
        if len(prices) < self.min_observations:
            return None
        
        window = min(self.window, len(prices))
        ts = prices.iloc[-window:]
        
        x = np.arange(len(ts))
        log_ts = np.log(ts)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_ts)
        
        annualized_slope = (np.power(np.exp(slope), 252) - 1) * 100
        r_squared = r_value ** 2
        score = annualized_slope * r_squared
        
        return np.clip(score, -100, 100)
    
    def calculate_ranked(self, prices_dict: Dict[str, pd.Series]) -> pd.Series:
        """
        Calcola i momentum score per un dizionario di simboli.
        """
        scores = {}
        for symbol, prices in prices_dict.items():
            score = self.calculate(prices)
            if score is not None:
                scores[symbol] = score
        
        return pd.Series(scores).sort_values(ascending=False)
    
    def explain(self, prices: pd.Series) -> Dict[str, Any]:
        """
        Spiega il calcolo del momentum score.
        """
        if len(prices) < self.min_observations:
            return {'error': 'Dati insufficienti'}
        
        window = min(self.window, len(prices))
        ts = prices.iloc[-window:]
        x = np.arange(len(ts))
        log_ts = np.log(ts)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_ts)
        
        annualized_slope = (np.power(np.exp(slope), 252) - 1) * 100
        r_squared = r_value ** 2
        score = annualized_slope * r_squared
        
        return {
            'score': np.clip(score, -100, 100),
            'window': window,
            'slope': slope,
            'slope_annualized': annualized_slope,
            'r_squared': r_squared,
            'intercept': intercept,
            'std_error': std_err,
            'p_value': p_value,
            'start_price': ts.iloc[0],
            'end_price': ts.iloc[-1],
            'total_return': (ts.iloc[-1] / ts.iloc[0] - 1) * 100
        }

class ClenowRanking:
    """
    Sistema di ranking basato sul momentum score di Clenow.
    """
    
    def __init__(self, window: int = 125, top_n: int = 10):
        self.momentum = ClenowMomentum(window)
        self.top_n = top_n
    
    def rank_symbols(self, prices_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Classifica i simboli per momentum score.
        """
        scores = self.momentum.calculate_ranked(prices_dict)
        
        df = pd.DataFrame({
            'symbol': scores.index,
            'momentum_score': scores.values,
            'rank': range(1, len(scores) + 1)
        })
        
        return df
    
    def get_top_symbols(self, prices_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        """Restituisce i top N simboli per momentum."""
        df = self.rank_symbols(prices_dict)
        return df.head(self.top_n)
    
    def get_weighted_signals(self, prices_dict: Dict[str, pd.Series]) -> Dict[str, float]:
        """Restituisce segnali pesati per tutti i simboli."""
        scores = self.momentum.calculate_ranked(prices_dict)
        
        if scores.empty:
            return {}
        
        max_score = max(abs(scores.max()), abs(scores.min()))
        if max_score > 0:
            normalized = scores / max_score
        else:
            normalized = pd.Series(0, index=scores.index)
        
        return normalized.to_dict()