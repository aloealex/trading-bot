# risk/stop_loss.py
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class StopLossManager:
    """
    Gestione degli stop loss dinamici.
    """
    
    def __init__(self,
                 fixed_stop: float = 0.05,
                 trailing_stop: float = 0.03,
                 atr_multiplier: float = 2.0,
                 max_holding_days: int = 30,
                 take_profit: Optional[float] = None):
        
        self.fixed_stop = fixed_stop
        self.trailing_stop = trailing_stop
        self.take_profit = take_profit
        self.atr_multiplier = atr_multiplier
        self.max_holding_days = max_holding_days
        
        self.positions = {}  # symbol -> position_data
    
    def add_position(self,
                     symbol: str,
                     entry_price: float,
                     entry_date: datetime,
                     side: str = 'buy'):
        """Aggiunge una posizione da monitorare."""
        self.positions[symbol] = {
            'entry_price': entry_price,
            'entry_date': entry_date,
            'side': side,
            'highest_price': entry_price,
            'lowest_price': entry_price,
            'current_stop': None,
            'stop_type': None
        }
        # Inizializza subito lo stop: prima restava None e check_stop andava in errore
        # (confronto float <= None) se veniva chiamato prima di update_prices.
        self._update_stop(symbol, entry_price)
    
    def remove_position(self, symbol: str):
        """Rimuove una posizione dal monitoraggio."""
        if symbol in self.positions:
            del self.positions[symbol]
    
    def update_prices(self, symbol: str, current_price: float, atr: Optional[float] = None):
        """Aggiorna i prezzi per una posizione."""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        
        if pos['side'] == 'buy':
            if current_price > pos['highest_price']:
                pos['highest_price'] = current_price
            if current_price < pos['lowest_price']:
                pos['lowest_price'] = current_price
        else:  # sell
            if current_price < pos['lowest_price']:
                pos['lowest_price'] = current_price
            if current_price > pos['highest_price']:
                pos['highest_price'] = current_price
        
        # Aggiorna lo stop
        self._update_stop(symbol, current_price, atr)
    
    def _update_stop(self, symbol: str, current_price: float, atr: Optional[float] = None):
        """Aggiorna lo stop loss per una posizione."""
        pos = self.positions[symbol]
        entry = pos['entry_price']
        
        stop_price = None
        stop_type = None
        
        if pos['side'] == 'buy':
            # Fixed stop
            fixed_stop_price = entry * (1 - self.fixed_stop)
            
            # Trailing stop
            trailing_stop_price = pos['highest_price'] * (1 - self.trailing_stop)
            
            # ATR stop (se disponibile)
            atr_stop_price = None
            if atr:
                atr_stop_price = current_price - atr * self.atr_multiplier
            
            # Prendi il più alto tra i vari stop (più vicino al prezzo)
            candidates = {'stop_loss': fixed_stop_price, 'trailing_stop': trailing_stop_price}
            if atr_stop_price:
                candidates['atr_stop'] = atr_stop_price
            stop_type, stop_price = max(candidates.items(), key=lambda kv: kv[1])
        
        else:  # sell
            fixed_stop_price = entry * (1 + self.fixed_stop)
            trailing_stop_price = pos['lowest_price'] * (1 + self.trailing_stop)
            
            atr_stop_price = None
            if atr:
                atr_stop_price = current_price + atr * self.atr_multiplier
            
            candidates = {'stop_loss': fixed_stop_price, 'trailing_stop': trailing_stop_price}
            if atr_stop_price:
                candidates['atr_stop'] = atr_stop_price
            stop_type, stop_price = min(candidates.items(), key=lambda kv: kv[1])
        
        pos['current_stop'] = stop_price
        pos['stop_type'] = stop_type
    
    def check_stop(self, symbol: str, current_price: float, current_date: datetime) -> Tuple[bool, str]:
        """
        Verifica se lo stop è stato attivato.
        
        Returns:
            Tuple: (stop_hit, reason)
        """
        if symbol not in self.positions:
            return False, ''
        
        pos = self.positions[symbol]
        if pos['current_stop'] is None:
            self._update_stop(symbol, current_price)
        
        # Stop (fisso o trailing)
        if pos['side'] == 'buy':
            if current_price <= pos['current_stop']:
                return True, pos.get('stop_type') or 'stop_loss'
        else:  # sell
            if current_price >= pos['current_stop']:
                return True, pos.get('stop_type') or 'stop_loss'
        
        # Take profit
        if self.take_profit:
            entry = pos['entry_price']
            if pos['side'] == 'buy' and current_price >= entry * (1 + self.take_profit):
                return True, 'take_profit'
            if pos['side'] == 'sell' and current_price <= entry * (1 - self.take_profit):
                return True, 'take_profit'
        
        # Check max holding days
        holding_days = (current_date - pos['entry_date']).days
        if holding_days >= self.max_holding_days:
            return True, 'max_holding'
        
        return False, ''
    
    def get_stop_price(self, symbol: str) -> Optional[float]:
        """Restituisce il prezzo di stop corrente."""
        if symbol in self.positions:
            return self.positions[symbol].get('current_stop')
        return None
    
    def get_position_info(self, symbol: str) -> Optional[Dict]:
        """Restituisce le informazioni di una posizione."""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict:
        """Restituisce tutte le posizioni monitorate."""
        return self.positions.copy()
    
    def calculate_atr_stop(self, entry_price: float, atr: float, side: str = 'buy') -> float:
        """Calcola lo stop basato su ATR."""
        if side == 'buy':
            return entry_price - atr * self.atr_multiplier
        else:
            return entry_price + atr * self.atr_multiplier


class DynamicStopLoss(StopLossManager):
    """
    Stop loss dinamico con adattamento alla volatilità.
    """
    
    def __init__(self,
                 base_stop: float = 0.05,
                 volatility_multiplier: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.base_stop = base_stop
        self.volatility_multiplier = volatility_multiplier
    
    def _update_stop(self, symbol: str, current_price: float, atr: Optional[float] = None):
        """Aggiorna lo stop con adattamento alla volatilità."""
        pos = self.positions[symbol]
        entry = pos['entry_price']
        
        # Calcola la volatilità relativa
        if atr and entry > 0:
            volatility_ratio = atr / entry
            # Adatta lo stop alla volatilità
            adjusted_stop = self.base_stop * (1 + volatility_ratio * self.volatility_multiplier)
            adjusted_stop = min(adjusted_stop, 0.15)  # Max 15%
        else:
            adjusted_stop = self.base_stop
        
        if pos['side'] == 'buy':
            stop_price = entry * (1 - adjusted_stop)
            # Trailing stop
            trailing_price = pos['highest_price'] * (1 - self.trailing_stop)
            stop_price = max(stop_price, trailing_price)
        else:
            stop_price = entry * (1 + adjusted_stop)
            trailing_price = pos['lowest_price'] * (1 + self.trailing_stop)
            stop_price = min(stop_price, trailing_price)
        
        pos['current_stop'] = stop_price
        pos['stop_type'] = 'dynamic'