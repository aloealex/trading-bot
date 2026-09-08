# deployment/deploy.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import json
import os
import signal
import sys
import time

from ..signals.ensemble import SignalEnsemble
from ..models.trainer import ModelTrainer
from ..execution.broker import get_broker
from ..execution.orders import OrderManager, OrderSide, OrderType
from ..risk.position_sizing import PositionSizer

class TradingBotDeployment:
    """
    Sistema di deployment per il bot di trading in produzione.
    """
    
    def __init__(self,
                 config_path: str = 'config/production.json',
                 log_level: str = 'INFO'):
        
        # Carica la configurazione
        self.config = self._load_config(config_path)
        
        # Configura il logging
        self._setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
        
        # Componenti
        self.broker = None
        self.signal_engine = SignalEnsemble()
        self.order_manager = OrderManager()
        self.position_sizer = PositionSizer(
            capital=self.config.get('initial_capital', 100000),
            max_positions=self.config.get('max_positions', 10)
        )
        
        # Stato
        self.is_running = False
        self.symbols = self.config.get('symbols', [])
        self.positions = {}
        self.capital = self.config.get('initial_capital', 100000)
        self.last_update = None
        self.daily_pnl = 0
        
        # Carica il modello ML se configurato
        self.model = None
        if self.config.get('use_ml', False):
            model_path = self.config.get('model_path', './saved_models/ensemble_model')
            self.model = ModelTrainer.load(model_path)
            self.logger.info(f"Modello ML caricato da {model_path}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Carica la configurazione da file JSON"""
        default_config = {
            'broker': 'alpaca',
            'broker_config': {
                'paper': True
            },
            'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
            'initial_capital': 100000,
            'max_positions': 10,
            'max_position_size': 0.20,
            'min_position_size': 0.01,
            'buy_threshold': 0.3,
            'sell_threshold': -0.3,
            'update_interval': 300,  # secondi
            'use_ml': False,
            'stop_loss': 0.05,
            'take_profit': 0.15,
            'trailing_stop': 0.03,
            'max_holding_days': 30,
            'risk_per_trade': 0.02,
            'position_sizing': 'kelly'
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                default_config.update(config)
        
        return default_config
    
    def _setup_logging(self, log_level: str):
        """Configura il sistema di logging"""
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Handler per console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        # Handler per file
        file_handler = logging.FileHandler('trading_bot.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        # Configura il root logger
        logging.basicConfig(
            level=numeric_level,
            handlers=[console_handler, file_handler]
        )
    
    def initialize(self) -> bool:
        """
        Inizializza il bot e connette al broker.
        """
        self.logger.info("Inizializzazione del bot...")
        
        # Connetti al broker
        broker_type = self.config.get('broker', 'alpaca')
        broker_config = self.config.get('broker_config', {})
        
        self.broker = get_broker(broker_type, **broker_config)
        if not self.broker.connect():
            self.logger.error("Impossibile connettersi al broker")
            return False
        
        # Recupera il capitale
        self.capital = self.broker.get_account_balance()
        if self.capital == 0:
            self.capital = self.config.get('initial_capital', 100000)
            self.logger.warning(f"Saldo non disponibile, uso capitale iniziale: {self.capital}")
        else:
            self.logger.info(f"Capitale: ${self.capital:.2f}")
        
        # Recupera le posizioni esistenti
        self.positions = self.broker.get_positions()
        self.logger.info(f"Posizioni esistenti: {len(self.positions)}")
        
        # Setup segnali di shutdown
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)
        
        self.is_running = True
        self.last_update = datetime.now()
        
        self.logger.info("Bot inizializzato con successo")
        return True
    
    def _shutdown_handler(self, signum, frame):
        """Gestisce lo shutdown del bot"""
        self.logger.info(f"Ricevuto segnale {signum}, arresto...")
        self.shutdown()
        sys.exit(0)
    
    def get_market_data(self, symbol: str) -> pd.DataFrame:
        """
        Recupera i dati di mercato.
        """
        try:
            # Dati storici
            historical = self.broker.get_historical_data(
                symbol, 
                self.config.get('lookback_days', 252),
                '1d'
            )
            
            # Prezzo corrente
            current_price = self.broker.get_current_price(symbol)
            if current_price > 0:
                # Aggiorna l'ultimo prezzo
                if not historical.empty:
                    historical.iloc[-1, historical.columns.get_loc('close')] = current_price
            
            return historical
            
        except Exception as e:
            self.logger.error(f"Errore nel recupero dati per {symbol}: {e}")
            return pd.DataFrame()
    
    def evaluate_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Valuta un singolo simbolo.
        """
        try:
            data = self.get_market_data(symbol)
            if data.empty:
                return {'symbol': symbol, 'gradimento': 0.0}
            
            prices = data['close']
            current_price = prices.iloc[-1]
            
            # Genera il segnale
            if self.model is not None:
                # Usa il modello ML
                features = self.signal_engine.extract_features(prices)
                gradimento = self.model.predict(features)
            else:
                # Usa il motore di segnali diretto
                signal_result = self.signal_engine.calculate(prices)
                gradimento = signal_result.gradimento
            
            return {
                'symbol': symbol,
                'gradimento': gradimento,
                'price': current_price,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Errore nella valutazione di {symbol}: {e}")
            return {'symbol': symbol, 'gradimento': 0.0}
    
    def evaluate_all_symbols(self) -> List[Dict[str, Any]]:
        """
        Valuta tutti i simboli del watchlist.
        """
        results = []
        for symbol in self.symbols:
            result = self.evaluate_symbol(symbol)
            results.append(result)
        
        # Ordina per gradimento decrescente
        results.sort(key=lambda x: x.get('gradimento', 0), reverse=True)
        
        return results
    
    def execute_trades(self, evaluations: List[Dict[str, Any]]):
        """
        Esegue gli ordini in base alle valutazioni.
        """
        # Aggiorna il capitale
        self.capital = self.broker.get_account_balance()
        
        # Gestione ordini di chiusura
        for symbol in list(self.positions.keys()):
            # Trova la valutazione del simbolo
            eval_data = next((e for e in evaluations if e.get('symbol') == symbol), None)
            if eval_data is None:
                continue
            
            gradimento = eval_data.get('gradimento', 0)
            
            # Verifica se chiudere la posizione
            if gradimento < self.config.get('sell_threshold', -0.3):
                self._close_position(symbol, eval_data.get('price', 0))
        
        # Gestione ordini di apertura
        available_slots = self.config.get('max_positions', 10) - len(self.positions)
        
        for eval_data in evaluations:
            if available_slots <= 0:
                break
            
            symbol = eval_data.get('symbol')
            gradimento = eval_data.get('gradimento', 0)
            price = eval_data.get('price', 0)
            
            # Verifica se aprire una nuova posizione
            if (gradimento > self.config.get('buy_threshold', 0.3) and
                symbol not in self.positions):
                
                # Calcola la dimensione della posizione
                position = self.position_sizer.calculate_position_size(
                    signal=gradimento,
                    price=price,
                    volatility=0.2,  # Da calcolare
                    capital=self.capital
                )
                
                if position['shares'] > 0:
                    # Invia l'ordine
                    order = self.order_manager.create_order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=position['shares'],
                        order_type=OrderType.MARKET
                    )
                    
                    executed_order = self.broker.place_order(order)
                    
                    if executed_order.status == OrderStatus.FILLED:
                        self.positions[symbol] = {
                            'quantity': position['shares'],
                            'entry_price': executed_order.filled_price or price,
                            'entry_date': datetime.now(),
                            'gradimento': gradimento
                        }
                        
                        self.logger.info(
                            f"ACQUISTATO {symbol}: {position['shares']} shares "
                            f"@ ${executed_order.filled_price:.2f} "
                            f"(gradimento: {gradimento:.3f})"
                        )
                        
                        available_slots -= 1
    
    def _close_position(self, symbol: str, current_price: float):
        """
        Chiude una posizione esistente.
        """
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        quantity = position['quantity']
        
        order = self.order_manager.create_order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=OrderType.MARKET
        )
        
        executed_order = self.broker.place_order(order)
        
        if executed_order.status == OrderStatus.FILLED:
            pnl = (current_price - position['entry_price']) * quantity
            pnl_percent = (current_price / position['entry_price'] - 1) * 100
            
            self.logger.info(
                f"VENDUTO {symbol}: {quantity} shares "
                f"@ ${executed_order.filled_price:.2f} "
                f"(PnL: ${pnl:.2f}, {pnl_percent:.2f}%)"
            )
            
            self.daily_pnl += pnl
            del self.positions[symbol]
    
    def run_cycle(self):
        """
        Esegue un singolo ciclo di trading.
        """
        self.logger.info("--- Inizio ciclo di trading ---")
        
        try:
            # 1. Valuta tutti i simboli
            evaluations = self.evaluate_all_symbols()
            
            # 2. Log dei migliori segnali
            top_signals = evaluations[:5]
            for eval_data in top_signals:
                self.logger.info(
                    f"  {eval_data['symbol']}: "
                    f"Gradimento = {eval_data['gradimento']:.3f}"
                )
            
            # 3. Esegui i trades
            self.execute_trades(evaluations)
            
            # 4. Aggiorna lo stato
            self.last_update = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Errore nel ciclo di trading: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        self.logger.info("--- Fine ciclo di trading ---")
    
    def run(self):
        """
        Avvia il loop principale del bot.
        """
        if not self.initialize():
            self.logger.error("Inizializzazione fallita")
            return
        
        self.logger.info("Bot avviato in produzione")
        self.logger.info(f"Watchlist: {self.symbols}")
        self.logger.info(f"Intervallo: {self.config.get('update_interval', 300)}s")
        
        interval = self.config.get('update_interval', 300)
        
        while self.is_running:
            try:
                # Esegui un ciclo
                self.run_cycle()
                
                # Aspetta fino al prossimo ciclo
                for _ in range(interval // 5):
                    if not self.is_running:
                        break
                    time.sleep(5)
                    
            except KeyboardInterrupt:
                self.logger.info("Interruzione ricevuta...")
                break
            except Exception as e:
                self.logger.error(f"Errore nel loop principale: {e}")
                time.sleep(60)  # Aspetta 1 minuto prima di riprovare
        
        self.shutdown()
    
    def shutdown(self):
        """Spegne il bot in modo ordinato."""
        self.logger.info("Spegnimento del bot...")
        self.is_running = False
        
        # Chiudi le posizioni se configurato
        if self.config.get('close_on_shutdown', False):
            self.logger.info("Chiusura posizioni aperte...")
            for symbol in list(self.positions.keys()):
                current_price = self.broker.get_current_price(symbol)
                self._close_position(symbol, current_price)
        
        self.broker.disconnect()
        self.logger.info("Bot spento.")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Restituisce lo stato attuale del bot.
        """
        return {
            'is_running': self.is_running,
            'capital': self.capital,
            'positions': self.positions,
            'daily_pnl': self.daily_pnl,
            'last_update': self.last_update,
            'symbols': self.symbols,
            'total_trades': len(self.order_manager.closed_orders)
        }