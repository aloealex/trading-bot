# monitoring/logger.py
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
import os

class TradingLogger:
    """
    Sistema di logging per il trading.
    """
    
    def __init__(self,
                 log_file: str = 'trading.log',
                 json_log: bool = True,
                 level: str = 'INFO'):
        
        self.log_file = log_file
        self.json_log = json_log
        self.level = getattr(logging, level.upper(), logging.INFO)
        
        self._setup_logger()
        
        self.trades = []
        self.signals = []
        self.errors = []
    
    def _setup_logger(self):
        """Configura il logger."""
        self.logger = logging.getLogger('trading_bot')
        self.logger.setLevel(self.level)
        
        # Handler per file
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(self.level)
        
        # Formatter
        if self.json_log:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        else:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Handler per console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def log_trade(self,
                  symbol: str,
                  side: str,
                  quantity: int,
                  price: float,
                  order_id: Optional[str] = None,
                  **kwargs):
        """Registra un trade."""
        entry = {
            'type': 'trade',
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'value': quantity * price,
            'order_id': order_id,
            **kwargs
        }
        
        self.trades.append(entry)
        
        if self.json_log:
            self.logger.info(json.dumps(entry))
        else:
            self.logger.info(f"TRADE: {symbol} {side} {quantity} @ {price:.2f}")
    
    def log_signal(self,
                   symbol: str,
                   signal: float,
                   confidence: float,
                   components: Dict[str, float],
                   **kwargs):
        """Registra un segnale."""
        entry = {
            'type': 'signal',
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'components': components,
            **kwargs
        }
        
        self.signals.append(entry)
        
        if self.json_log:
            self.logger.info(json.dumps(entry))
        else:
            self.logger.info(f"SIGNAL: {symbol} {signal:.3f} ({confidence:.2%})")
    
    def log_error(self,
                  error: str,
                  context: Optional[Dict] = None,
                  **kwargs):
        """Registra un errore."""
        entry = {
            'type': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': error,
            'context': context or {},
            **kwargs
        }
        
        self.errors.append(entry)
        
        if self.json_log:
            self.logger.error(json.dumps(entry))
        else:
            self.logger.error(f"ERROR: {error}")
    
    def log_info(self, message: str, **kwargs):
        """Registra un messaggio informativo."""
        entry = {
            'type': 'info',
            'timestamp': datetime.now().isoformat(),
            'message': message,
            **kwargs
        }
        
        if self.json_log:
            self.logger.info(json.dumps(entry))
        else:
            self.logger.info(message)
    
    def log_performance(self, metrics: Dict[str, Any]):
        """Registra le metriche di performance."""
        entry = {
            'type': 'performance',
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        if self.json_log:
            self.logger.info(json.dumps(entry))
        else:
            self.logger.info(f"PERFORMANCE: {json.dumps(metrics, default=str)}")
    
    def get_trades(self, symbol: Optional[str] = None) -> list:
        """Restituisce i trades registrati."""
        if symbol:
            return [t for t in self.trades if t.get('symbol') == symbol]
        return self.trades.copy()
    
    def get_signals(self, symbol: Optional[str] = None) -> list:
        """Restituisce i segnali registrati."""
        if symbol:
            return [s for s in self.signals if s.get('symbol') == symbol]
        return self.signals.copy()
    
    def get_errors(self) -> list:
        """Restituisce gli errori registrati."""
        return self.errors.copy()
    
    def clear_logs(self):
        """Pulisce i log."""
        self.trades = []
        self.signals = []
        self.errors = []
    
    def export_logs(self, filepath: str):
        """Esporta i log in formato JSON."""
        logs = {
            'trades': self.trades,
            'signals': self.signals,
            'errors': self.errors,
            'exported_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(logs, f, indent=2, default=str)