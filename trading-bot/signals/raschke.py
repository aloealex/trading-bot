# signals/raschke.py
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class RaschkePatterns:
    """
    Pattern di Linda Bradford Raschke basati sull'oscillatore 3/10.
    """
    
    def __init__(self, 
                 fast: int = 3, 
                 slow: int = 10,
                 divergence_window: int = 10,
                 smoothing: int = 16):
        self.fast = fast
        self.slow = slow
        self.divergence_window = divergence_window
        self.smoothing = smoothing
    
    def _calculate_oscillator(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calcola l'oscillatore 3/10 e la slow line."""
        sma_fast = prices.rolling(self.fast).mean()
        sma_slow = prices.rolling(self.slow).mean()
        oscillator = sma_fast - sma_slow
        slow_line = oscillator.rolling(self.smoothing).mean()
        return oscillator, slow_line
    
    def first_cross(self, prices: pd.Series, direction: str = 'buy') -> int:
        """
        First Cross Buy/Sell.
        """
        oscillator, slow_line = self._calculate_oscillator(prices)
        
        if len(slow_line) < 5:
            return 0
        
        if direction == 'buy':
            cross_up = (slow_line.iloc[-3] < 0) and (slow_line.iloc[-2] > 0)
            pullback = oscillator.iloc[-1] < 0
            return 1 if cross_up and pullback else 0
        
        elif direction == 'sell':
            cross_down = (slow_line.iloc[-3] > 0) and (slow_line.iloc[-2] < 0)
            pullback = oscillator.iloc[-1] > 0
            return 1 if cross_down and pullback else 0
        
        return 0
    
    def divergence(self, prices: pd.Series) -> Tuple[int, float]:
        """
        Rileva divergenze tra prezzo e oscillatore.
        
        Returns:
            Tuple: (tipo, forza)
            tipo: 1 = rialzista, -1 = ribassista, 0 = nessuna
        """
        oscillator, slow_line = self._calculate_oscillator(prices)
        
        if len(prices) < self.divergence_window + 10:
            return 0, 0.0
        
        recent_prices = prices.iloc[-self.divergence_window:]
        recent_osc = oscillator.iloc[-self.divergence_window:]
        
        current_price = recent_prices.iloc[-1]
        current_osc = recent_osc.iloc[-1]
        
        # Divergenza ribassista
        if current_price > recent_prices.iloc[0] and current_osc < recent_osc.iloc[0]:
            if current_price > recent_prices.max():
                price_diff = (current_price - recent_prices.max()) / recent_prices.max()
                osc_diff = (recent_osc.max() - current_osc) / abs(recent_osc.max())
                strength = min(1.0, (price_diff + osc_diff) / 2)
                return -1, strength
        
        # Divergenza rialzista
        if current_price < recent_prices.iloc[0] and current_osc > recent_osc.iloc[0]:
            if current_price < recent_prices.min():
                price_diff = (recent_prices.min() - current_price) / recent_prices.min()
                osc_diff = (current_osc - recent_osc.min()) / abs(recent_osc.min())
                strength = min(1.0, (price_diff + osc_diff) / 2)
                return 1, strength
        
        return 0, 0.0
    
    def anti_pattern(self, prices: pd.Series) -> int:
        """
        Anti pattern: bull/bear flag in un trading range.
        """
        oscillator, slow_line = self._calculate_oscillator(prices)
        
        if len(slow_line) < 10:
            return 0
        
        slow_turning_up = (slow_line.iloc[-3] < slow_line.iloc[-2] < slow_line.iloc[-1])
        osc_approach = abs(oscillator.iloc[-1] - slow_line.iloc[-1]) < 0.1
        
        price_range = (prices.iloc[-10:].max() - prices.iloc[-10:].min()) / prices.iloc[-10:].mean()
        is_consolidating = price_range < 0.03
        
        if slow_turning_up and osc_approach and is_consolidating:
            return 1
        
        return 0
    
    def calculate(self, prices: pd.Series) -> Optional[float]:
        """
        Calcola il segnale combinato dei pattern Raschke.
        """
        if len(prices) < 50:
            return None
        
        signal = 0.0
        total_weight = 0.0
        
        if self.first_cross(prices, 'buy'):
            signal += 0.5
            total_weight += 1.0
        elif self.first_cross(prices, 'sell'):
            signal -= 0.5
            total_weight += 1.0
        
        div_type, div_strength = self.divergence(prices)
        if div_type == 1:
            signal += 0.6 * div_strength
            total_weight += div_strength
        elif div_type == -1:
            signal -= 0.6 * div_strength
            total_weight += div_strength
        
        if self.anti_pattern(prices):
            signal += 0.3
            total_weight += 1.0
        
        if total_weight > 0:
            return np.clip(signal / total_weight, -1, 1)
        
        return 0.0
    
    def calculate_components(self, prices: pd.Series) -> Dict[str, Any]:
        """Calcola tutte le componenti del segnale Raschke."""
        if len(prices) < 50:
            return {'error': 'Dati insufficienti'}
        
        oscillator, slow_line = self._calculate_oscillator(prices)
        
        return {
            'oscillator': oscillator.iloc[-1] if len(oscillator) > 0 else 0,
            'slow_line': slow_line.iloc[-1] if len(slow_line) > 0 else 0,
            'first_cross_buy': self.first_cross(prices, 'buy'),
            'first_cross_sell': self.first_cross(prices, 'sell'),
            'divergence_type': self.divergence(prices)[0],
            'divergence_strength': self.divergence(prices)[1],
            'anti_pattern': self.anti_pattern(prices),
            'combined_signal': self.calculate(prices)
        }
    
    def explain(self, prices: pd.Series) -> Dict[str, Any]:
        """Spiega il ragionamento dietro il segnale."""
        components = self.calculate_components(prices)
        
        explanation = {
            'signal': components.get('combined_signal', 0),
            'components': components,
            'interpretation': []
        }
        
        if components.get('first_cross_buy'):
            explanation['interpretation'].append('First Cross Buy: trend reversal up')
        if components.get('first_cross_sell'):
            explanation['interpretation'].append('First Cross Sell: trend reversal down')
        if components.get('divergence_type') == 1:
            explanation['interpretation'].append(f'Bullish divergence (strength: {components["divergence_strength"]:.2f})')
        if components.get('divergence_type') == -1:
            explanation['interpretation'].append(f'Bearish divergence (strength: {components["divergence_strength"]:.2f})')
        if components.get('anti_pattern'):
            explanation['interpretation'].append('Anti pattern: consolidation/flag')
        
        if not explanation['interpretation']:
            explanation['interpretation'].append('No clear pattern detected')
        
        return explanation