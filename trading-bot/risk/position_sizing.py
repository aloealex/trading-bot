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
                 max_position_size: float = 0.20,
                 base_fraction: float = 0.10,
                 min_trades_for_kelly: int = 20):
        
        self.capital = capital
        self.max_positions = max_positions
        self.kelly_fraction = kelly_fraction
        self.volatility_target = volatility_target
        self.max_position_size = max_position_size
        self.base_fraction = base_fraction
        self.min_trades_for_kelly = min_trades_for_kelly
        self.trade_history = []
    
    def kelly_sizing(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calcola la frazione di Kelly."""
        if avg_loss == 0 or avg_win == 0:
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
        
        signal_factor = min(1.0, abs(signal))

        # Dimensione base: quota fissa dell'equity, scalata per l'intensità del segnale
        # (stessa regola del backtest, così i risultati sono confrontabili).
        base_value = capital * self.base_fraction * signal_factor

        # Kelly frazionato solo con uno storico sufficiente: con pochi trade la stima
        # è puro rumore (prima, con lo storico vuoto, si investiva ~1% per titolo).
        kelly_fraction = 0.0
        if len(self.trade_history) >= self.min_trades_for_kelly:
            kelly_fraction = self.kelly_sizing(self._calculate_win_rate(),
                                               self._calculate_avg_win(),
                                               self._calculate_avg_loss())
            base_value = capital * max(kelly_fraction, self.base_fraction * 0.25) * signal_factor

        # Tetto di volatilità: titoli molto volatili ricevono meno capitale
        volatility_shares = self.volatility_sizing(volatility, capital, price)
        volatility_value = volatility_shares * price
        final_value = min(base_value, volatility_value) if volatility_value > 0 else base_value
        final_value = min(final_value, max_position_value)
        
        shares = int(final_value / price) if price > 0 else 0
        
        return {
            'shares': shares,
            'value': shares * price,
            'capital_percent': (shares * price) / capital * 100 if capital > 0 else 0,
            'kelly_fraction': kelly_fraction,
            'volatility_shares': volatility_shares,
            'method': 'kelly' if kelly_fraction > 0 else 'base'
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