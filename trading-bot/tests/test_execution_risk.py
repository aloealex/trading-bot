# tests/test_execution_risk.py
from datetime import datetime

import pytest

from execution.orders import OrderManager, OrderSide, OrderStatus
from execution.portfolio import PortfolioManager
from execution.broker import SimulatedBroker, get_broker, AlpacaBroker
from risk.stop_loss import StopLossManager
from risk.position_sizing import PositionSizer
from risk.portfolio_risk import PortfolioRiskManager


def test_ordini_non_bloccano_dopo_20():
    """Prima gli ordini restavano 'aperti' per sempre e al 21° il bot si fermava."""
    om = OrderManager(max_open_orders=20)
    for _ in range(50):
        o = om.create_order('AAA', OrderSide.BUY, 1)
        o.status, o.filled_price = OrderStatus.FILLED, 10.0
        om.record_result(o)
    assert len(om.open_orders) == 0
    assert om.get_positions() == {'AAA': 50}


def test_costo_medio_vendita_parziale():
    pm = PortfolioManager(10_000)
    pm.record_transaction('AAA', 'buy', 10, 100)
    pm.record_transaction('AAA', 'buy', 10, 110)
    pm.record_transaction('AAA', 'sell', 5, 120)
    pos = pm.positions['AAA']
    assert pos['quantity'] == 15
    assert pos['avg_cost'] == pytest.approx(105)
    assert pm.transactions[-1]['realized_pnl'] == pytest.approx(75)
    pm.record_transaction('AAA', 'sell', 15, 100)
    assert 'AAA' not in pm.positions
    assert pm.cash == pytest.approx(10_000 - 1000 - 1100 + 600 + 1500)


def test_stop_attivo_subito_e_take_profit():
    m = StopLossManager(fixed_stop=0.05, trailing_stop=0.10, take_profit=0.15)
    m.add_position('AAA', 100.0, datetime(2024, 1, 1))
    assert m.check_stop('AAA', 94.0, datetime(2024, 1, 2)) == (True, 'stop_loss')
    assert m.check_stop('AAA', 116.0, datetime(2024, 1, 2)) == (True, 'take_profit')
    assert m.check_stop('AAA', 101.0, datetime(2024, 3, 1))[1] == 'max_holding'


def test_sizing_proporzionale_al_segnale():
    ps = PositionSizer(capital=100_000, base_fraction=0.10, max_position_size=0.20)
    small = ps.calculate_position_size(0.3, 100, 0.25, 100_000)['value']
    big = ps.calculate_position_size(0.9, 100, 0.25, 100_000)['value']
    assert small == pytest.approx(3000, rel=0.02) and big == pytest.approx(9000, rel=0.02)


def test_concentrazione_su_conto_totale():
    rm = PortfolioRiskManager(max_concentration=0.20)
    rm.update_position('AAA', 10, 400)   # 4.000
    rm.update_position('BBB', 10, 300)   # 3.000
    rm.set_cash(93_000)
    ok, violations = rm.check_portfolio_limits()
    assert ok, violations


def test_broker_simulato(universe):
    b = SimulatedBroker(initial_cash=10_000, data=universe, start_index=300, commission=0, slippage=0)
    b.connect()
    from execution.orders import Order
    o = b.place_order(Order('AAA', OrderSide.BUY, 10))
    assert o.status == OrderStatus.FILLED
    assert b.get_positions() == {'AAA': 10}
    assert b.place_order(Order('AAA', OrderSide.SELL, 11)).status == OrderStatus.REJECTED
    assert b.get_account_balance() == pytest.approx(10_000)


def test_alpaca_senza_chiavi_non_crasha():
    b = get_broker('alpaca', api_key='', secret_key='', host='x', initial_cash=1)
    assert isinstance(b, AlpacaBroker)
    assert b.connect() is False
    assert b.get_positions() == {} and b.get_account_balance() == 0.0


def test_alpaca_ordine_attende_esecuzione_e_dati():
    """Con alpaca-py un ordine a mercato torna 'accepted': il broker deve attendere il 'filled'."""
    pytest.importorskip('alpaca')
    from types import SimpleNamespace
    import pandas as pd
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.data.requests import StockBarsRequest
    from execution.orders import Order

    class FakeTrading:
        def __init__(self):
            self.calls = 0
        def submit_order(self, req):
            assert isinstance(req, MarketOrderRequest) and req.qty == 7
            return SimpleNamespace(id='abc', status=SimpleNamespace(value='accepted'),
                                   filled_avg_price=None, filled_qty=None, filled_at=None)
        def get_order_by_id(self, oid):
            self.calls += 1
            done = self.calls >= 2
            return SimpleNamespace(id=oid, status=SimpleNamespace(value='filled' if done else 'new'),
                                   filled_avg_price='101.5' if done else None,
                                   filled_qty='7' if done else '0', filled_at=None)

    class FakeData:
        def get_stock_bars(self, req):
            assert isinstance(req, StockBarsRequest) and req.start is not None
            idx = pd.MultiIndex.from_product([['AAA'], pd.bdate_range('2024-01-01', periods=10, tz='UTC')],
                                             names=['symbol', 'timestamp'])
            df = pd.DataFrame({c: range(10) for c in ['open', 'high', 'low', 'close', 'volume',
                                                       'trade_count', 'vwap']}, index=idx, dtype=float)
            return SimpleNamespace(df=df)

    b = AlpacaBroker('k', 's', fill_timeout=5)
    b.trading, b.data, b.is_connected = FakeTrading(), FakeData(), True
    o = b.place_order(Order('AAA', OrderSide.BUY, 7))
    assert o.status == OrderStatus.FILLED and o.filled_price == 101.5 and o.filled_quantity == 7
    bars = b.get_historical_data('AAA', 5)
    assert list(bars.columns) == ['open', 'high', 'low', 'close', 'volume'] and len(bars) == 5
