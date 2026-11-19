#!/usr/bin/env python
# main.py
"""
Punto di ingresso del bot di trading.

Modalità:
    python main.py --mode signals                 # classifica dei segnali di oggi (nessun ordine)
    python main.py --mode trade                   # bot live (paper di default) con il broker in config
    python main.py --mode trade --broker sim      # bot con broker SIMULATO (nessun ordine reale)
    python main.py --mode backtest --years 5      # backtest dei segnali sullo storico
    python main.py --mode train --years 8         # addestra il modello ML e lo salva
    python main.py --mode walkforward --ml        # validazione walk-forward (segnali o ML)
    python main.py --mode dashboard               # dashboard Streamlit (legge lo stato del bot)

Opzioni dati: --csv-dir CARTELLA (un CSV per simbolo) oppure yfinance (default).
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# il bot si avvia dalla cartella del progetto: rendi importabili i package locali
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from config.settings import Settings
from signals.ensemble import SignalEnsemble
from models.predict import ModelPredictor
from risk.position_sizing import PositionSizer
from risk.stop_loss import StopLossManager
from risk.portfolio_risk import PortfolioRiskManager
from execution.broker import BrokerInterface, get_broker
from execution.orders import OrderManager, OrderSide, OrderType, OrderStatus
from execution.portfolio import PortfolioManager
from monitoring.logger import TradingLogger
from monitoring.alerts import AlertSystem

logger = logging.getLogger(__name__)


def setup_logging(level: str = 'INFO'):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler('trading_bot.log', encoding='utf-8'), logging.StreamHandler()],
    )


def load_settings(path: Optional[str]) -> Settings:
    if path and os.path.exists(path):
        return Settings.from_json(path)
    if path:
        logger.warning(f"File di configurazione {path} non trovato: uso i valori predefiniti")
    return Settings()


def make_broker(settings: Settings, **overrides) -> BrokerInterface:
    b = settings.broker
    kwargs = dict(
        api_key=b.alpaca_api_key,
        secret_key=b.alpaca_secret_key,
        paper=b.paper,
        data_feed=b.data_feed,
        host=b.ib_host,
        port=b.ib_port,
        client_id=b.ib_client_id,
        initial_cash=settings.trading.initial_capital,
        commission=settings.trading.commission,
        slippage=settings.trading.slippage,
    )
    kwargs.update(overrides)
    return get_broker(overrides.pop('broker_type', None) or b.broker_type, **kwargs)


class TradingBot:
    """Bot di trading automatico (solo posizioni long)."""

    def __init__(self, settings: Settings, broker: Optional[BrokerInterface] = None):
        self.settings = settings
        t = settings.trading

        self.broker = broker
        self.signal_engine = SignalEnsemble.from_settings(settings.signals)
        self.order_manager = OrderManager()
        self.portfolio_manager = PortfolioManager(initial_capital=t.initial_capital)
        self.position_sizer = PositionSizer(
            capital=t.initial_capital,
            max_positions=t.max_positions,
            kelly_fraction=t.kelly_fraction,
            volatility_target=t.volatility_target,
            max_position_size=t.max_position_size,
            base_fraction=t.position_size,
        )
        self.stop_loss_manager = StopLossManager(
            fixed_stop=t.stop_loss,
            trailing_stop=t.trailing_stop,
            max_holding_days=t.max_holding_days,
            take_profit=t.take_profit,
        )
        self.risk_manager = PortfolioRiskManager(
            initial_capital=t.initial_capital,
            max_portfolio_risk=t.max_drawdown,
            max_concentration=t.max_position_size,
        )
        self.trade_log = TradingLogger(level=settings.log_level)
        self.alerts = AlertSystem.from_settings(settings.notifications)

        self.predictor: Optional[ModelPredictor] = None
        self.is_running = False
        self.symbols = list(settings.symbols)
        self.last_update = None
        self.last_evaluations: List[Dict] = []
        self._price_cache: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ #
    def initialize(self) -> bool:
        self.trade_log.log_info("Inizializzazione del bot...")
        if self.broker is None:
            self.broker = make_broker(self.settings)
        if not self.broker.connect():
            self.trade_log.log_error("Impossibile connettersi al broker")
            return False

        m = self.settings.model
        if m.use_ml:
            if ModelPredictor.exists(m.model_path):
                self.predictor = ModelPredictor(m.model_path)
                self.trade_log.log_info(f"Modello ML caricato da {m.model_path}")
            else:
                self.trade_log.log_warning(
                    f"use_ml attivo ma nessun modello in {m.model_path}: uso i segnali diretti. "
                    f"Addestralo con: python main.py --mode train")

        equity = self.broker.get_account_balance()
        if equity > 0:
            self.portfolio_manager.initial_capital = equity
        self.portfolio_manager.cash = self.broker.get_cash() or equity

        # posizioni già aperte sul conto: mettile sotto stop loss (prezzo d'ingresso ignoto -> prezzo attuale)
        for symbol, qty in self.broker.get_positions().items():
            price = self.broker.get_current_price(symbol)
            if qty > 0 and price > 0:
                self.portfolio_manager.update_position(symbol, qty, price)
                self.stop_loss_manager.add_position(symbol, price, datetime.now(), 'buy')
                self.trade_log.log_info(f"Posizione esistente: {symbol} x{qty} (stop calcolato da {price:.2f})")

        self.is_running = True
        self.last_update = datetime.now()
        self.trade_log.log_info(f"Bot inizializzato. Equity: {equity:,.2f}")
        return True

    # ------------------------------------------------------------------ #
    def get_market_data(self, symbol: str) -> pd.DataFrame:
        try:
            df = self.broker.get_historical_data(symbol, self.settings.model.lookback_days, '1d')
            if df.empty:
                return df
            current = self.broker.get_current_price(symbol)
            if current > 0:
                df = df.copy()
                df.iloc[-1, df.columns.get_loc('close')] = current
            self._price_cache[symbol] = df
            return df
        except Exception as e:
            self.trade_log.log_error(f"Errore recupero dati per {symbol}", {'error': str(e)})
            return pd.DataFrame()

    def evaluate_symbol(self, symbol: str) -> Dict:
        try:
            data = self.get_market_data(symbol)
            if data.empty:
                return {'symbol': symbol, 'gradimento': 0.0, 'error': 'Nessun dato'}
            prices = data['close']
            price = float(prices.iloc[-1])

            result = self.signal_engine.calculate(prices, symbol)
            volatility = result.risk_metrics.volatility if result.risk_metrics else \
                float(prices.pct_change().iloc[-60:].std() * np.sqrt(252))

            if self.predictor is not None:
                ml = self.predictor.predict_symbol(data)
                gradimento, confidence = ml.get('gradimento', 0.0), ml.get('confidence', 0.5)
                source = 'ml'
            else:
                gradimento, confidence = result.gradimento, result.confidence
                source = 'segnali'

            return {
                'symbol': symbol,
                'gradimento': float(gradimento),
                'confidence': float(confidence),
                'price': price,
                'volatility': float(volatility),
                'source': source,
                'components': {k: float(v) for k, v in result.components.items()},
                'interpretation': result.interpretation,
                'timestamp': datetime.now(),
            }
        except Exception as e:
            self.trade_log.log_error(f"Errore valutazione {symbol}", {'error': str(e)})
            return {'symbol': symbol, 'gradimento': 0.0, 'error': str(e)}

    def evaluate_all_symbols(self) -> List[Dict]:
        results = [self.evaluate_symbol(s) for s in self.symbols]
        results.sort(key=lambda x: x.get('gradimento', 0), reverse=True)
        self.last_evaluations = results
        return results

    # ------------------------------------------------------------------ #
    def _submit(self, symbol: str, side: OrderSide, quantity: int):
        order = self.order_manager.create_order(symbol=symbol, side=side, quantity=quantity,
                                                order_type=OrderType.MARKET)
        executed = self.broker.place_order(order)
        self.order_manager.record_result(executed)
        return executed

    def check_exits(self, evaluations: List[Dict]) -> None:
        """Stop loss / trailing / take profit / max holding / segnale negativo."""
        positions = self.broker.get_positions()
        by_symbol = {e['symbol']: e for e in evaluations}
        now = datetime.now()
        for symbol, qty in positions.items():
            if qty <= 0:
                continue
            price = by_symbol.get(symbol, {}).get('price') or self.broker.get_current_price(symbol)
            if not price:
                continue
            if symbol not in self.stop_loss_manager.positions:
                self.stop_loss_manager.add_position(symbol, price, now, 'buy')
            self.stop_loss_manager.update_prices(symbol, price)
            hit, reason = self.stop_loss_manager.check_stop(symbol, price, now)
            if not hit:
                g = by_symbol.get(symbol, {}).get('gradimento')
                if g is not None and g < self.settings.trading.sell_threshold:
                    hit, reason = True, 'signal'
            if hit:
                self._close_position(symbol, price, reason)

    def execute_entries(self, evaluations: List[Dict]) -> None:
        t = self.settings.trading
        positions = self.broker.get_positions()
        pending = {o.symbol for o in self.broker.get_open_orders()}
        slots = t.max_positions - len(positions) - len(pending)
        equity = self.broker.get_account_balance()
        cash = self.broker.get_cash()

        for ev in evaluations:
            if slots <= 0:
                break
            symbol, g = ev.get('symbol'), ev.get('gradimento', 0.0)
            if ev.get('error') or g <= t.buy_threshold or symbol in positions or symbol in pending:
                continue
            price = ev.get('price', 0.0)
            sizing = self.position_sizer.calculate_position_size(
                signal=g, price=price, volatility=ev.get('volatility', 0.2), capital=equity)
            shares = min(sizing['shares'], int(cash * 0.99 / price) if price > 0 else 0)
            if shares * price < equity * t.min_position_size:
                continue
            if shares > 0:
                if self._open_position(symbol, shares, price, g):
                    cash -= shares * price
                    slots -= 1

    def _open_position(self, symbol: str, quantity: int, price: float, signal: float) -> bool:
        executed = self._submit(symbol, OrderSide.BUY, quantity)
        if executed.status != OrderStatus.FILLED:
            if executed.status == OrderStatus.PENDING:
                self.trade_log.log_info(f"Ordine {symbol} in attesa di esecuzione")
            else:
                self.trade_log.log_warning(f"Ordine di acquisto {symbol} non eseguito: {executed.status.value}"
                                           + (f" ({executed.notes})" if executed.notes else ''))
            return False
        fill = executed.filled_price or price
        qty = executed.filled_quantity or quantity
        self.portfolio_manager.record_transaction(symbol, 'buy', qty, fill, executed.order_id)
        self.stop_loss_manager.add_position(symbol, fill, datetime.now(), 'buy')
        self.trade_log.log_trade(symbol=symbol, side='BUY', quantity=qty, price=fill,
                                 order_id=executed.order_id, gradimento=round(signal, 3))
        self.alerts.send_trade_alert(symbol=symbol, side='BUY', quantity=qty, price=fill, signal=signal)
        return True

    def _close_position(self, symbol: str, price: float, reason: str) -> bool:
        quantity = self.broker.get_positions().get(symbol, 0)
        if quantity <= 0:
            return False
        executed = self._submit(symbol, OrderSide.SELL, quantity)
        if executed.status != OrderStatus.FILLED:
            self.trade_log.log_warning(f"Ordine di vendita {symbol} non eseguito: {executed.status.value}")
            return False
        fill = executed.filled_price or price
        info = self.stop_loss_manager.get_position_info(symbol) or {}
        entry = info.get('entry_price')
        pnl_pct = (fill / entry - 1) if entry else 0.0
        self.position_sizer.record_trade({'symbol': symbol, 'pnl': pnl_pct})
        self.portfolio_manager.record_transaction(symbol, 'sell', quantity, fill, executed.order_id)
        self.stop_loss_manager.remove_position(symbol)
        self.risk_manager.remove_position(symbol)
        self.trade_log.log_trade(symbol=symbol, side='SELL', quantity=quantity, price=fill,
                                 order_id=executed.order_id, reason=reason, pnl_pct=round(pnl_pct * 100, 2))
        self.alerts.send_trade_alert(symbol=symbol, side='SELL', quantity=quantity, price=fill, reason=reason)
        return True

    # ------------------------------------------------------------------ #
    def reconcile(self) -> None:
        """
        Allinea lo stato interno al conto del broker (la fonte di verità):
        ordini rimasti 'in attesa' che il broker non ha più aperti, posizioni eseguite
        in ritardo o chiuse a mano, liquidità effettiva (commissioni comprese).
        """
        broker_open = {o.broker_order_id for o in self.broker.get_open_orders()}
        for oid, order in list(self.order_manager.open_orders.items()):
            if order.broker_order_id not in broker_open:
                self.order_manager.update_order(oid, OrderStatus.CANCELLED)

        positions = self.broker.get_positions()
        for symbol in list(self.portfolio_manager.positions):
            if positions.get(symbol, 0) <= 0:
                self.portfolio_manager.positions.pop(symbol, None)
                self.stop_loss_manager.remove_position(symbol)
                self.risk_manager.remove_position(symbol)
        for symbol, qty in positions.items():
            known = self.portfolio_manager.positions.get(symbol, {}).get('quantity', 0)
            if qty != known:
                price = self.broker.get_current_price(symbol)
                if price > 0:
                    self.portfolio_manager.update_position(symbol, qty - known, price)
                    if symbol not in self.stop_loss_manager.positions:
                        self.stop_loss_manager.add_position(symbol, price, datetime.now(), 'buy')
                    self.trade_log.log_info(f"Riconciliazione: {symbol} ora {qty} azioni sul conto")
        cash = self.broker.get_cash()
        if cash:
            self.portfolio_manager.cash = cash

    def run_cycle(self) -> None:
        self.trade_log.log_info("--- Inizio ciclo di trading ---")
        try:
            self.reconcile()
            evaluations = self.evaluate_all_symbols()
            for ev in evaluations[:5]:
                if 'error' in ev:
                    self.trade_log.log_info(f"{ev['symbol']}: {ev['error']}")
                else:
                    self.trade_log.log_info(f"{ev['symbol']}: gradimento {ev['gradimento']:+.3f} "
                                            f"(conf. {ev['confidence']:.0%}, {ev['source']})")

            if self.broker.is_market_open():
                self.check_exits(evaluations)
                self.execute_entries(evaluations)
            else:
                self.trade_log.log_info("Mercato chiuso: nessun ordine in questo ciclo")

            # rischio di portafoglio
            prices = {}
            for symbol, qty in self.broker.get_positions().items():
                df = self._price_cache.get(symbol)
                if df is None or df.empty:
                    continue
                prices[symbol] = float(df['close'].iloc[-1])
                self.risk_manager.update_position(symbol, qty, prices[symbol])
                self.risk_manager.update_price_history(symbol, df['close'])
            self.portfolio_manager.mark_to_market(prices)
            self.risk_manager.set_cash(self.broker.get_cash())
            ok, violations = self.risk_manager.check_portfolio_limits()
            for v in violations:
                self.trade_log.log_warning(f"Limite di rischio: {v}")
                self.alerts.send_alert('risk', f"Limite rischio: {v}", 'warning')

            self.last_update = datetime.now()
            self.save_state()
        except Exception as e:
            logger.exception("Errore nel ciclo")
            self.trade_log.log_error(f"Errore nel ciclo: {e}")
            self.alerts.send_error_alert(str(e))
        self.trade_log.log_info("--- Fine ciclo di trading ---")

    def run(self, max_cycles: Optional[int] = None) -> None:
        if not self.initialize():
            return
        self.trade_log.log_info(f"Bot avviato. Watchlist: {self.symbols}")
        self.trade_log.log_info(f"Intervallo: {self.settings.update_interval}s")
        cycles = 0
        try:
            while self.is_running:
                self.run_cycle()
                cycles += 1
                if max_cycles and cycles >= max_cycles:
                    break
                for _ in range(max(1, self.settings.update_interval // 5)):
                    if not self.is_running:
                        break
                    time.sleep(5)
        except KeyboardInterrupt:
            self.trade_log.log_info("Interruzione ricevuta (Ctrl+C)")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self.trade_log.log_info("Spegnimento del bot...")
        self.is_running = False
        if self.broker is None:
            return
        if self.settings.close_on_shutdown:
            self.trade_log.log_info("Chiusura posizioni aperte...")
            for symbol in list(self.broker.get_positions()):
                self._close_position(symbol, self.broker.get_current_price(symbol), 'shutdown')
        self.save_state()
        self.broker.disconnect()
        self.trade_log.log_info("Bot spento")

    # ------------------------------------------------------------------ #
    def get_status(self) -> Dict:
        positions = []
        for symbol, pos in self.portfolio_manager.positions.items():
            positions.append({
                'symbol': symbol,
                'quantity': pos['quantity'],
                'avg_cost': pos['avg_cost'],
                'current_price': pos['current_price'],
                'unrealized_pnl': pos['unrealized_pnl'],
                'unrealized_pnl_percent': pos['unrealized_pnl_percent'],
                'stop': self.stop_loss_manager.get_stop_price(symbol),
            })
        return {
            'updated_at': datetime.now().isoformat(timespec='seconds'),
            'is_running': self.is_running,
            'broker': self.settings.broker.broker_type,
            'paper': self.settings.broker.paper,
            'equity': self.broker.get_account_balance() if self.broker else 0.0,
            'cash': self.broker.get_cash() if self.broker else 0.0,
            'positions': positions,
            'evaluations': [{k: v for k, v in e.items() if k != 'timestamp'} for e in self.last_evaluations],
            'equity_history': [{'timestamp': v['timestamp'].isoformat(timespec='seconds'),
                                'total_value': v['total_value']}
                               for v in self.portfolio_manager.value_history[-2000:]],
            'trades': self.trade_log.get_trades()[-200:],
            'order_summary': self.order_manager.get_summary(),
        }

    def save_state(self) -> None:
        path = self.settings.state_file
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            tmp = path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(self.get_status(), f, indent=2, default=str, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception as e:
            logger.warning(f"Impossibile salvare lo stato: {e}")


# ---------------------------------------------------------------------- #
# Modalità offline: segnali, backtest, train, walk-forward
# ---------------------------------------------------------------------- #
def _load_history(args, settings: Settings) -> Dict[str, pd.DataFrame]:
    from data.loaders import load_prices
    data = load_prices(settings.symbols, csv_dir=args.csv_dir, years=args.years)
    if not data:
        raise SystemExit("Nessun dato storico caricato (controlla la connessione o --csv-dir)")
    print(f"Dati caricati: {len(data)} simboli, "
          f"{min(df.index[0] for df in data.values()).date()} -> {max(df.index[-1] for df in data.values()).date()}")
    return data


def run_signals(args, settings: Settings) -> None:
    from data.loaders import load_prices
    data = load_prices(settings.symbols, csv_dir=args.csv_dir, years=2)
    engine = SignalEnsemble.from_settings(settings.signals)
    rows = []
    for symbol, df in data.items():
        r = engine.calculate(df['close'], symbol)
        rows.append({'simbolo': symbol, 'gradimento': r.gradimento, 'confidenza': r.confidence,
                     **{k: round(float(v), 3) for k, v in r.components.items()},
                     'azione': engine.get_trading_signal(r)['action']})
    out = pd.DataFrame(rows).sort_values('gradimento', ascending=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))


def run_backtest(args, settings: Settings) -> None:
    from backtest.engine import BacktestEngine
    t = settings.trading
    data = _load_history(args, settings)

    if args.ml:
        path = settings.model.model_path
        if not ModelPredictor.exists(path):
            raise SystemExit(f"Nessun modello in {path}. Addestralo prima con --mode train")
        predictor = ModelPredictor(path)
        print(f"Segnali dal modello ML ({predictor.meta['model_type']}). ATTENZIONE: valuta solo il periodo "
              f"di test {predictor.meta['test_period']}; sul periodo di training il risultato è falsato.")
        signals = {s: predictor.predict_series(df) for s, df in data.items()}
    else:
        # Monte Carlo più leggero per velocità: le metriche usate dal gradimento cambiano pochissimo
        engine = SignalEnsemble.from_settings(settings.signals, mc_simulations=args.mc_sims)
        signals = {}
        for symbol, df in data.items():
            print(f"Calcolo segnali {symbol}...", flush=True)
            signals[symbol] = engine.signal_series(df['close'])

    bt = BacktestEngine(
        initial_capital=t.initial_capital,
        commission=t.commission,
        slippage=t.slippage,
        max_positions=t.max_positions,
        risk_per_trade=args.position_size or t.position_size,
        max_position_size=t.max_position_size,
        entry_threshold=t.buy_threshold,
        exit_threshold=t.sell_threshold,
    )
    result = bt.run(data, signals, stop_loss=t.stop_loss, take_profit=t.take_profit,
                    trailing_stop=t.trailing_stop, max_holding_days=t.max_holding_days)
    print_backtest(result)

    # confronto con "compra e tieni" equipesato sugli stessi titoli e periodo
    start = result.equity_curve.index[0]
    closes = pd.DataFrame({s: df['close'] for s, df in data.items()}).loc[start:].ffill().bfill()
    bh = (closes / closes.iloc[0]).mean(axis=1)
    print(f"\nRiferimento compra-e-tieni (equipesato): {bh.iloc[-1] - 1:+.1%}")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        result.equity_curve.to_csv(os.path.join(args.output, 'equity.csv'), header=['equity'])
        bt.trades_dataframe().to_csv(os.path.join(args.output, 'trades.csv'), index=False)
        print(f"Risultati salvati in {args.output}/")


def print_backtest(result) -> None:
    print("\n=== Risultati backtest ===")
    for k, v in result.summary().items():
        if isinstance(v, float) and k in ('Rendimento totale', 'Rendimento annuo', 'Volatilità annua',
                                          'Max drawdown', 'Win rate', 'Esposizione media'):
            print(f"{k:28s} {v:+.2%}")
        elif isinstance(v, float):
            print(f"{k:28s} {v:,.2f}")
        else:
            print(f"{k:28s} {v}")
    if result.trades:
        reasons = pd.Series([t.exit_reason for t in result.trades]).value_counts()
        print("Uscite per motivo:", ', '.join(f"{k}={v}" for k, v in reasons.items()))


def run_train(args, settings: Settings) -> None:
    from models.train import ModelTrainer
    data = _load_history(args, settings)
    model_type = args.model_type or settings.model.model_type
    trainer = ModelTrainer(model_type=model_type)
    meta = trainer.train(data)
    path = args.model_path or settings.model.model_path
    trainer.save(path)
    print(f"\nModello '{model_type}' salvato in {path}")
    print(f"Training: {meta['train_period']}  |  Test (mai visto): {meta['test_period']}")
    print("Metriche sul periodo di test:")
    for k, v in meta['test_metrics'].items():
        print(f"  {k:24s} {v:+.4f}")
    top = sorted(trainer.model.get_feature_importance().items(), key=lambda kv: -kv[1])[:8]
    if top:
        print("Feature più importanti:", ', '.join(f"{k} ({v:.2f})" for k, v in top))


def run_walkforward(args, settings: Settings) -> None:
    from backtest.walk_forward import WalkForwardAnalyzer
    from models.train import ModelTrainer
    t = settings.trading
    data = _load_history(args, settings)
    wf = WalkForwardAnalyzer(train_window=args.train_window, test_window=args.test_window,
                             step_size=args.test_window)
    trainer = ModelTrainer(model_type=args.model_type or settings.model.model_type) if args.ml else None
    engine = SignalEnsemble.from_settings(settings.signals, mc_simulations=args.mc_sims)
    res = wf.run(data, signal_generator=engine, model_trainer=trainer,
                 initial_capital=t.initial_capital, commission=t.commission, slippage=t.slippage,
                 max_positions=t.max_positions, risk_per_trade=args.position_size or t.position_size,
                 entry_threshold=t.buy_threshold, exit_threshold=t.sell_threshold)
    print(res.to_string(index=False))
    print("\nSintesi:")
    for k, v in wf.analyze_results().items():
        print(f"  {k:22s} {v:.4f}" if isinstance(v, float) else f"  {k:22s} {v}")


def run_dashboard(args, settings: Settings) -> None:
    app = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'monitoring', 'dashboard.py')
    env = dict(os.environ, BOT_STATE_FILE=os.path.abspath(settings.state_file))
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', app], env=env, check=False)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description='Trading Bot')
    parser.add_argument('--config', '-c', default='config/production.json', help='File di configurazione JSON')
    parser.add_argument('--mode', '-m', default='trade',
                        choices=['trade', 'signals', 'backtest', 'train', 'walkforward', 'dashboard'])
    parser.add_argument('--log-level', '-l', default=None, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--symbols', '-s', nargs='+', help='Lista di simboli (sovrascrive la config)')
    parser.add_argument('--broker', choices=['alpaca', 'ib', 'sim'], help='Sovrascrive il broker della config')
    parser.add_argument('--cycles', type=int, default=None, help='trade: numero di cicli e poi stop')
    parser.add_argument('--csv-dir', default=None, help='Cartella di CSV (uno per simbolo) invece di yfinance')
    parser.add_argument('--years', type=float, default=5, help='Anni di storico da scaricare')
    parser.add_argument('--ml', action='store_true', help='backtest/walkforward: usa il modello ML')
    parser.add_argument('--model-type', choices=['xgboost', 'neural', 'ensemble'], default=None)
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--position-size', type=float, default=None,
                        help='backtest: quota di equity per posizione (x |gradimento|); default dalla config')
    parser.add_argument('--mc-sims', type=int, default=1000, help='backtest: simulazioni Monte Carlo')
    parser.add_argument('--train-window', type=int, default=504)
    parser.add_argument('--test-window', type=int, default=126)
    parser.add_argument('--output', '-o', default=None, help='backtest: cartella dove salvare equity e trade')
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    setup_logging(args.log_level or settings.log_level)
    if args.symbols:
        settings.symbols = [s.upper() for s in args.symbols]
    if args.broker:
        settings.broker.broker_type = args.broker
    if args.model_path:
        settings.model.model_path = args.model_path

    if args.mode == 'trade':
        b = settings.broker
        if b.broker_type == 'alpaca' and not b.paper:
            logger.warning("ATTENZIONE: conto Alpaca REALE (paper=false). Gli ordini useranno denaro vero.")
        broker = None
        if settings.broker.broker_type == 'sim' and args.csv_dir:
            from data.loaders import load_csv_dir
            broker = make_broker(settings, data=load_csv_dir(args.csv_dir, settings.symbols))
        TradingBot(settings, broker=broker).run(max_cycles=args.cycles)
    elif args.mode == 'signals':
        run_signals(args, settings)
    elif args.mode == 'backtest':
        run_backtest(args, settings)
    elif args.mode == 'train':
        run_train(args, settings)
    elif args.mode == 'walkforward':
        run_walkforward(args, settings)
    elif args.mode == 'dashboard':
        run_dashboard(args, settings)


if __name__ == '__main__':
    main()
