# risk/position_sizing.py
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class PositionSizer:
    """
    Gestione della dimensione della posizione.
    """
    
    def __init__(self,
                 capital: float,
                 max_positions: int = 10,
                 kelly_fraction: float = 0.25,
                 volatility_target: float = 0.15,
                 max_position_size: float = 0.20):
        
        self.capital = capital
        self.max_positions = max_positions
        self.kelly_fraction = kelly_fraction
        self.volatility_target = volatility_target
        self.max_position_size = max_position_size
        self.trade_history = []
    
    def kelly_sizing(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calcola la frazione di Kelly."""
        if avg_loss == 0:
            return 0.0
        
        b = avg_win / avg_loss
        q = 1 - win_rate
        f = (win_rate * b - q) / b
        
        f = f * self.kelly_fraction
        return max(0, min(f, 0.20))
    
    def volatility_sizing(self, volatility: float, capital: float, price: float) -> int:
        """Dimensiona la posizione in base alla volatilità."""
        if volatility == 0 or price == 0:
            return 0
        
        daily_vol = volatility / np.sqrt(252)
        target_risk = self.volatility_target * capital / np.sqrt(252)
        
        position_value = target_risk / daily_vol
        shares = int(position_value / price)
        
        return max(0, shares)
    
    def calculate_position_size(self,
                               signal: float,
                               price: float,
                               volatility: float,
                               capital: float) -> Dict:
        """Calcola la dimensione della posizione."""
        max_position_value = capital * self.max_position_size
        
        # Kelly sizing
        win_rate = self._calculate_win_rate()
        avg_win = self._calculate_avg_win()
        avg_loss = self._calculate_avg_loss()
        kelly_fraction = self.kelly_sizing(win_rate, avg_win, avg_loss)
        
        # Volatility sizing
        volatility_shares = self.volatility_sizing(volatility, capital, price)
        volatility_value = volatility_shares * price
        
        # Sizing basato sul segnale
        signal_factor = abs(signal)
        base_position_value = min(max_position_value * signal_factor, capital * 0.10)
        
        # Combina i metodi
        if kelly_fraction > 0:
            kelly_value = capital * kelly_fraction * signal_factor
        else:
            kelly_value = base_position_value
        
        final_value = min(kelly_value, volatility_value * 1.5) if volatility_value > 0 else kelly_value
        final_value = min(final_value, max_position_value)
        
        if final_value < capital * 0.01 and signal > 0.3:
            final_value = capital * 0.01
        
        shares = int(final_value / price) if price > 0 else 0
        
        return {
            'shares': shares,
            'value': shares * price,
            'capital_percent': (shares * price) / capital * 100 if capital > 0 else 0,
            'kelly_fraction': kelly_fraction,
            'volatility_shares': volatility_shares,
            'method': 'kelly_volatility'
        }
    
    def _calculate_win_rate(self) -> float:
        """Calcola il win rate storico."""
        if not self.trade_history:
            return 0.55
        wins = sum(1 for t in self.trade_history if t.get('pnl', 0) > 0)
        return wins / len(self.trade_history) if self.trade_history else 0.5
    
    def _calculate_avg_win(self) -> float:
        """Calcola il profitto medio."""
        wins = [t.get('pnl', 0) for t in self.trade_history if t.get('pnl', 0) > 0]
        return np.mean(wins) if wins else 0.01
    
    def _calculate_avg_loss(self) -> float:
        """Calcola la perdita media."""
        losses = [t.get('pnl', 0) for t in self.trade_history if t.get('pnl', 0) < 0]
        return abs(np.mean(losses)) if losses else 0.01
    
    def record_trade(self, trade: Dict):
        """Registra un trade per l'aggiornamento dei parametri."""
        self.trade_history.append(trade)