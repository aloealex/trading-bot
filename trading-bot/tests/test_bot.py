# tests/test_bot.py
"""Test del ciclo completo del bot con il broker simulato (nessuna rete, nessun ordine reale)."""
import json
import os

import pytest

from config.settings import Settings
from execution.broker import SimulatedBroker
from main import TradingBot, load_settings


def _settings(tmp_path, symbols):
    s = Settings()
    s.symbols = list(symbols)
    s.broker.broker_type = 'sim'
    s.signals.mc_simulations = 300
    s.state_file = str(tmp_path / 'state.json')
    s.trading.max_positions = 3
    s.trading.buy_threshold = 0.2
    return s


def test_bot_molti_cicli(universe, tmp_path):
    s = _settings(tmp_path, universe)
    broker = SimulatedBroker(initial_cash=100_000, data=universe, start_index=320)
    bot = TradingBot(s, broker=broker)
    assert bot.initialize()

    max_seen = 0
    for _ in range(150):
        bot.run_cycle()
        max_seen = max(max_seen, len(broker.get_positions()))
        broker.advance()

    filled = [o for o in broker.filled_orders]
    sells = [o for o in filled if o.side.value == 'sell']
    assert len(filled) > 10 and len(sells) > 3          # compra E vende (stop/segnali funzionano)
    assert max_seen <= s.trading.max_positions
    assert len(bot.order_manager.open_orders) == 0       # nessun ordine "appeso"
    # il portafoglio interno coincide con quello del broker
    assert {k: v['quantity'] for k, v in bot.portfolio_manager.positions.items()} == broker.get_positions()

    state = json.load(open(s.state_file))
    assert state['broker'] == 'sim' and len(state['evaluations']) == len(universe)


def test_bot_errore_dati_non_blocca(universe, tmp_path):
    s = _settings(tmp_path, list(universe) + ['NONESISTE'])
    broker = SimulatedBroker(data=universe, start_index=320)
    bot = TradingBot(s, broker=broker)
    bot.initialize()
    bot.run_cycle()
    errs = [e for e in bot.last_evaluations if 'error' in e]
    assert [e['symbol'] for e in errs] == ['NONESISTE']


def test_config_production_json(monkeypatch):
    monkeypatch.setenv('ALPACA_API_KEY', 'chiave')
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'production.json')
    s = load_settings(path)
    assert s.broker.alpaca_api_key == 'chiave'
    assert s.broker.paper is True
    assert s.trading.max_positions == 8
    assert s.signals.ensemble_weights['risk'] > 0


def test_riconciliazione_con_il_conto(universe, tmp_path):
    """Posizione aperta 'a mano' sul conto e ordine rimasto appeso: il bot si riallinea."""
    from execution.orders import Order, OrderSide
    s = _settings(tmp_path, universe)
    broker = SimulatedBroker(initial_cash=100_000, data=universe, start_index=320)
    bot = TradingBot(s, broker=broker)
    bot.initialize()
    broker.place_order(Order('CCC', OrderSide.BUY, 5))           # eseguito fuori dal bot
    stale = bot.order_manager.create_order('AAA', OrderSide.BUY, 1)  # ordine mai concluso
    bot.reconcile()
    assert bot.portfolio_manager.positions['CCC']['quantity'] == 5
    assert 'CCC' in bot.stop_loss_manager.positions
    assert stale.order_id not in bot.order_manager.open_orders
    assert bot.portfolio_manager.cash == pytest.approx(broker.get_cash())
