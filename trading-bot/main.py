# main.py
#!/usr/bin/env python
"""
Punto di ingresso principale per il bot di trading.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from dotenv import load_dotenv

# Carica le variabili d'ambiente
load_dotenv()

from config.settings import Settings, BrokerSettings, TradingSettings, SignalSettings, ModelSettings
from config.symbols import Watchlist
from signals.ensemble import SignalEnsemble
from models.train import ModelTrainer
from models.predict import ModelPredictor
from risk.position_sizing import PositionSizer
from risk.stop_loss import StopLossManager
from risk.portfolio_risk import PortfolioRiskManager
from execution.broker import get_broker
from execution.orders import OrderManager
from execution.portfolio import PortfolioManager
from monitoring.logger import TradingLogger
from monitoring.alerts import AlertSystem
from monitoring.dashboard import TradingDashboard
from backtest.engine import BacktestEngine

def setup_logging(level: str = 'INFO'):
    """Configura il logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trading_bot.log'),
            logging.StreamHandler()
        ]
    )

class TradingBot:
    """
    Bot di trading automatico completo.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Componenti
        self.broker = None
        self.signal_engine = SignalEnsemble(
            weights=settings.signals.ensemble_weights
        )
        self.order_manager = OrderManager()
        self.portfolio_manager = PortfolioManager(
            initial_capital=settings.trading.initial_capital
        )
        self.position_sizer = PositionSizer(
            capital=settings.trading.initial_capital,
            max_positions=settings.trading.max_positions,
            kelly_fraction=settings.trading.kelly_fraction,
            volatility_target=settings.trading.volatility_target,
            max_position_size=settings.trading.max_position_size
        )
        self.stop_loss_manager = StopLossManager(
            fixed_stop=settings.trading.stop_loss,
            trailing_stop=settings.trading.trailing_stop,
            max_holding_days=settings.trading.max_holding_days
        )
        self.risk_manager = PortfolioRiskManager(
            initial_capital=settings.trading.initial_capital,
            max_portfolio_risk=settings.trading.max_drawdown,
            max_concentration=settings.trading.max_position_size
        )
        
        self.logger = TradingLogger()
        self.alerts = AlertSystem()
        
        self.model = None
        self.predictor = None
        self.is_running = False
        self.symbols = settings.symbols
        self.last_update = None
    
    def initialize(self) -> bool:
        """Inizializza il bot."""
        self.logger.log_info("Inizializzazione del bot...")
        
        # Connetti al broker
        broker_config = self.settings.broker
        self.broker = get_broker(
            broker_config.broker_type,
            api_key=broker_config.alpaca_api_key,
            secret_key=broker_config.alpaca_secret_key,
            paper=True
        )
        
        if not self.broker.connect():
            self.logger.log_error("Impossibile connettersi al broker")
            return False
        
        # Carica il modello ML se configurato
        if self.settings.model.model_type != 'direct':
            model_path = self.settings.model.model_path
            if os.path.exists(model_path):
                self.predictor = ModelPredictor(model_path)
                self.logger.log_info(f"Modello caricato da {model_path}")
            else:
                self.logger.log_info("Nessun modello trovato, uso segnali diretti")
        
        # Aggiorna il capitale
        capital = self.broker.get_account_balance()
        if capital > 0:
            self.portfolio_manager.cash = capital
            self.portfolio_manager.initial_capital = capital
        
        self.is_running = True
        self.last_update = datetime.now()
        
        self.logger.log_info("Bot inizializzato con successo")
        return True
    
    def get_market_data(self, symbol: str) -> pd.DataFrame:
        """Recupera i dati di mercato."""
        try:
            df = self.broker.get_historical_data(
                symbol,
                self.settings.model.lookback_days,
                '1d'
            )
            
            # Aggiungi il prezzo corrente
            current_price = self.broker.get_current_price(symbol)
            if current_price > 0 and not df.empty:
                df.loc[df.index[-1], 'close'] = current_price
            
            return df
            
        except Exception as e:
            self.logger.log_error(f"Errore recupero dati per {symbol}", {'error': str(e)})
            return pd.DataFrame()
    
    def evaluate_symbol(self, symbol: str) -> Dict:
        """Valuta un singolo simbolo."""
        try:
            data = self.get_market_data(symbol)
            if data.empty:
                return {'symbol': symbol, 'gradimento': 0.0, 'error': 'No data'}
            
            prices = data['close']
            current_price = prices.iloc[-1]
            
            # Usa il modello ML o i segnali diretti
            if self.predictor:
                result = self.predictor.predict_symbol(data)
                gradimento = result.get('gradimento', 0)
                confidence = result.get('confidence', 0.5)
            else:
                signal_result = self.signal_engine.calculate(prices, symbol)
                gradimento = signal_result.gradimento
                confidence = signal_result.confidence
            
            # Calcola il rischio
            risk_metrics = self.signal_engine.risk.calculate_metrics(prices)
            volatility = risk_metrics.volatility if risk_metrics else 0.2
            
            return {
                'symbol': symbol,
                'gradimento': gradimento,
                'confidence': confidence,
                'price': current_price,
                'volatility': volatility,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.log_error(f"Errore valutazione {symbol}", {'error': str(e)})
            return {'symbol': symbol, 'gradimento': 0.0, 'error': str(e)}
    
    def evaluate_all_symbols(self) -> List[Dict]:
        """Valuta tutti i simboli."""
        results = []
        for symbol in self.symbols:
            result = self.evaluate_symbol(symbol)
            results.append(result)
        
        results.sort(key=lambda x: x.get('gradimento', 0), reverse=True)
        return results
    
    def execute_trades(self, evaluations: List[Dict]):
        """Esegue gli ordini."""
        # Aggiorna il capitale
        capital = self.broker.get_account_balance()
        if capital > 0:
            self.portfolio_manager.cash = capital
        
        # Posizioni correnti
        positions = self.broker.get_positions()
        
        # Chiudi posizioni con segnale negativo
        for symbol in list(positions.keys()):
            eval_data = next((e for e in evaluations if e.get('symbol') == symbol), None)
            if eval_data is None:
                continue
            
            gradimento = eval_data.get('gradimento', 0)
            price = eval_data.get('price', 0)
            
            if gradimento < self.settings.trading.sell_threshold:
                self._close_position(symbol, price, 'signal')
        
        # Apri nuove posizioni
        available_slots = self.settings.trading.max_positions - len(positions)
        
        for eval_data in evaluations:
            if available_slots <= 0:
                break
            
            symbol = eval_data.get('symbol')
            gradimento = eval_data.get('gradimento', 0)
            price = eval_data.get('price', 0)
            volatility = eval_data.get('volatility', 0.2)
            
            if (gradimento > self.settings.trading.buy_threshold and
                symbol not in positions):
                
                # Calcola la dimensione
                position = self.position_sizer.calculate_position_size(
                    signal=gradimento,
                    price=price,
                    volatility=volatility,
                    capital=self.portfolio_manager.cash
                )
                
                if position['shares'] > 0:
                    self._open_position(symbol, position['shares'], price, gradimento)
                    available_slots -= 1
    
    def _open_position(self, symbol: str, quantity: int, price: float, signal: float):
        """Apre una nuova posizione."""
        order = self.order_manager.create_order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET
        )
        
        executed = self.broker.place_order(order)
        
        if executed.status == OrderStatus.FILLED:
            self.portfolio_manager.record_transaction(
                symbol=symbol,
                side='buy',
                quantity=quantity,
                price=executed.filled_price or price
            )
            
            self.stop_loss_manager.add_position(
                symbol=symbol,
                entry_price=executed.filled_price or price,
                entry_date=datetime.now(),
                side='buy'
            )
            
            self.logger.log_trade(
                symbol=symbol,
                side='BUY',
                quantity=quantity,
                price=executed.filled_price or price,
                order_id=executed.order_id
            )
            
            self.alerts.send_trade_alert(
                symbol=symbol,
                side='BUY',
                quantity=quantity,
                price=executed.filled_price or price,
                signal=signal
            )
    
    def _close_position(self, symbol: str, price: float, reason: str):
        """Chiude una posizione esistente."""
        position = self.broker.get_positions().get(symbol, 0)
        if position <= 0:
            return
        
        order = self.order_manager.create_order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=position,
            order_type=OrderType.MARKET
        )
        
        executed = self.broker.place_order(order)
        
        if executed.status == OrderStatus.FILLED:
            self.portfolio_manager.record_transaction(
                symbol=symbol,
                side='sell',
                quantity=position,
                price=executed.filled_price or price
            )
            
            self.stop_loss_manager.remove_position(symbol)
            
            self.logger.log_trade(
                symbol=symbol,
                side='SELL',
                quantity=position,
                price=executed.filled_price or price,
                order_id=executed.order_id
            )
            
            self.alerts.send_trade_alert(
                symbol=symbol,
                side='SELL',
                quantity=position,
                price=executed.filled_price or price,
                reason=reason
            )
    
    def run_cycle(self):
        """Esegue un ciclo di trading."""
        self.logger.log_info("--- Inizio ciclo di trading ---")
        
        try:
            evaluations = self.evaluate_all_symbols()
            
            # Log dei migliori segnali
            for eval_data in evaluations[:5]:
                self.logger.log_info(
                    f"{eval_data['symbol']}: "
                    f"Gradimento = {eval_data['gradimento']:.3f}"
                )
            
            self.execute_trades(evaluations)
            
            # Aggiorna il rischio di portafoglio
            for symbol, pos in self.broker.get_positions().items():
                price = self.broker.get_current_price(symbol)
                if price > 0:
                    self.risk_manager.update_position(symbol, pos, price)
            
            risk_metrics = self.risk_manager.calculate_risk_metrics()
            limits_ok, violations = self.risk_manager.check_portfolio_limits()
            
            if not limits_ok:
                self.logger.log_warning("Violazione limiti di rischio", {'violations': violations})
                for violation in violations:
                    self.alerts.send_alert('warning', f"Limite rischio: {violation}", 'warning')
            
            self.last_update = datetime.now()
            
        except Exception as e:
            self.logger.log_error(f"Errore nel ciclo: {e}")
            self.alerts.send_error_alert(str(e))
        
        self.logger.log_info("--- Fine ciclo di trading ---")
    
    def run(self):
        """Avvia il loop principale."""
        if not self.initialize():
            return
        
        self.logger.log_info("Bot avviato")
        self.logger.log_info(f"Watchlist: {self.symbols}")
        self.logger.log_info(f"Intervallo: {self.settings.update_interval}s")
        
        import time
        while self.is_running:
            try:
                self.run_cycle()
                
                for _ in range(self.settings.update_interval // 5):
                    if not self.is_running:
                        break
                    time.sleep(5)
                    
            except KeyboardInterrupt:
                self.logger.log_info("Interruzione ricevuta")
                break
            except Exception as e:
                self.logger.log_error(f"Errore nel loop: {e}")
                time.sleep(60)
        
        self.shutdown()
    
    def shutdown(self):
        """Spegne il bot."""
        self.logger.log_info("Spegnimento del bot...")
        self.is_running = False
        
        if self.settings.close_on_shutdown:
            self.logger.log_info("Chiusura posizioni aperte...")
            for symbol in list(self.broker.get_positions().keys()):
                price = self.broker.get_current_price(symbol)
                self._close_position(symbol, price, 'shutdown')
        
        if self.broker:
            self.broker.disconnect()
        
        self.logger.log_info("Bot spento")

def main():
    """Punto di ingresso principale."""
    parser = argparse.ArgumentParser(description='Trading Bot')
    parser.add_argument('--config', '-c', default='config/production.json',
                       help='Percorso del file di configurazione')
    parser.add_argument('--mode', '-m', choices=['trade', 'dashboard', 'backtest'],
                       default='trade', help='Modalità di esecuzione')
    parser.add_argument('--log-level', '-l', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Livello di logging')
    parser.add_argument('--symbols', '-s', nargs='+', help='Lista di simboli')
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Carica le impostazioni
    settings = Settings()
    if args.symbols:
        settings.symbols = args.symbols
    
    if args.mode == 'trade':
        bot = TradingBot(settings)
        bot.run()
    
    elif args.mode == 'dashboard':
        bot = TradingBot(settings)
        if bot.initialize():
            dashboard = TradingDashboard(bot)
            dashboard.run()
    
    elif args.mode == 'backtest':
        logger.info("Modalità backtest")
        # Implementazione backtest
        from backtest.engine import BacktestEngine
        # ... 

if __name__ == '__main__':
    main()