# execution/portfolio.py
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PortfolioManager:
    """
    Gestione del portafoglio.
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Dict] = {}
        self.transactions: List[Dict] = []
        self.value_history: List[Dict] = []
        self._last_update = datetime.now()
    
    def update_position(self, symbol: str, quantity: int, price: float):
        """Aggiorna una posizione."""
        if quantity == 0:
            if symbol in self.positions:
                del self.positions[symbol]
            return
        
        if symbol in self.positions:
            # Aggiorna posizione esistente
            pos = self.positions[symbol]
            old_quantity = pos['quantity']
            
            # Calcola il costo medio
            total_cost = pos['avg_cost'] * old_quantity + price * quantity
            new_quantity = old_quantity + quantity
            
            pos['quantity'] = new_quantity
            pos['avg_cost'] = total_cost / new_quantity if new_quantity > 0 else 0
            pos['current_price'] = price
            pos['market_value'] = new_quantity * price
            pos['unrealized_pnl'] = pos['market_value'] - new_quantity * pos['avg_cost']
            pos['unrealized_pnl_percent'] = (pos['unrealized_pnl'] / (new_quantity * pos['avg_cost'])) * 100 if new_quantity * pos['avg_cost'] > 0 else 0
        
        else:
            # Nuova posizione
            self.positions[symbol] = {
                'symbol': symbol,
                'quantity': quantity,
                'avg_cost': price,
                'current_price': price,
                'market_value': quantity * price,
                'unrealized_pnl': 0,
                'unrealized_pnl_percent': 0,
                'entry_date': datetime.now()
            }
        
        self._last_update = datetime.now()
    
    def record_transaction(self,
                          symbol: str,
                          side: str,
                          quantity: int,
                          price: float,
                          order_id: Optional[str] = None):
        """Registra una transazione."""
        transaction = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'value': quantity * price,
            'timestamp': datetime.now(),
            'order_id': order_id
        }
        
        self.transactions.append(transaction)
        
        # Aggiorna il cash
        if side == 'buy':
            self.cash -= quantity * price
        else:
            self.cash += quantity * price
        
        # Aggiorna la posizione
        if side == 'buy':
            self.update_position(symbol, quantity, price)
        else:
            self.update_position(symbol, -quantity, price)
        
        # Registra lo storico del valore
        self._record_value()
    
    def _record_value(self):
        """Registra il valore corrente del portafoglio."""
        total_value = self.cash + self.get_positions_value()
        self.value_history.append({
            'timestamp': datetime.now(),
            'total_value': total_value,
            'cash': self.cash,
            'positions_value': self.get_positions_value(),
            'positions_count': len(self.positions)
        })
    
    def get_positions_value(self) -> float:
        """Calcola il valore totale delle posizioni."""
        return sum(pos['market_value'] for pos in self.positions.values())
    
    def get_total_value(self) -> float:
        """Calcola il valore totale del portafoglio."""
        return self.cash + self.get_positions_value()
    
    def get_total_return(self) -> float:
        """Calcola il rendimento totale."""
        return (self.get_total_value() - self.initial_capital) / self.initial_capital
    
    def get_position_pnl(self, symbol: str) -> Dict:
        """Calcola il PnL di una posizione."""
        if symbol not in self.positions:
            return {}
        
        pos = self.positions[symbol]
        realized_pnl = 0
        unrealized_pnl = pos.get('unrealized_pnl', 0)
        
        # Calcola il realized PnL dalle transazioni
        for tx in self.transactions:
            if tx['symbol'] == symbol and tx['side'] == 'sell':
                # Calcola il costo medio al momento della vendita
                # Implementazione semplificata
                pass
        
        return {
            'symbol': symbol,
            'quantity': pos['quantity'],
            'avg_cost': pos['avg_cost'],
            'current_price': pos['current_price'],
            'market_value': pos['market_value'],
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_percent': pos.get('unrealized_pnl_percent', 0),
            'realized_pnl': realized_pnl,
            'total_pnl': realized_pnl + unrealized_pnl
        }
    
    def get_positions_pnl(self) -> pd.DataFrame:
        """Calcola il PnL di tutte le posizioni."""
        data = []
        for symbol in self.positions:
            pnl = self.get_position_pnl(symbol)
            if pnl:
                data.append(pnl)
        
        return pd.DataFrame(data)
    
    def get_portfolio_metrics(self) -> Dict:
        """Calcola le metriche del portafoglio."""
        total_value = self.get_total_value()
        positions_value = self.get_positions_value()
        
        # Calcola la concentrazione
        concentration = {}
        for symbol, pos in self.positions.items():
            if total_value > 0:
                concentration[symbol] = pos['market_value'] / total_value
        
        # Calcola il drawdown
        if self.value_history:
            values = [v['total_value'] for v in self.value_history]
            max_value = max(values)
            current_value = values[-1] if values else total_value
            drawdown = (current_value - max_value) / max_value if max_value > 0 else 0
        else:
            drawdown = 0
        
        return {
            'total_value': total_value,
            'cash': self.cash,
            'positions_value': positions_value,
            'positions_count': len(self.positions),
            'total_return': self.get_total_return(),
            'drawdown': drawdown,
            'concentration': concentration,
            'largest_position': max(concentration.values()) if concentration else 0,
            'transactions_count': len(self.transactions),
            'last_update': self._last_update
        }
    
    def get_position_weights(self) -> Dict[str, float]:
        """Calcola i pesi delle posizioni."""
        total = self.get_positions_value()
        if total == 0:
            return {}
        return {symbol: pos['market_value'] / total for symbol, pos in self.positions.items()}
    
    def get_cash_allocations(self) -> Dict:
        """Calcola l'allocazione del cash."""
        return {
            'cash_percent': (self.cash / self.get_total_value()) * 100 if self.get_total_value() > 0 else 0,
            'invested_percent': (self.get_positions_value() / self.get_total_value()) * 100 if self.get_total_value() > 0 else 0
        }