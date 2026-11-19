# tests/test_backtest.py
import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestEngine
from tests.synthetic import make_prices


def _signals_random(data, seed=0):
    rng = np.random.default_rng(seed)
    return {s: pd.Series(rng.uniform(-1, 1, len(df)), index=df.index) for s, df in data.items()}


def test_contabilita_quadra(universe):
    """Equity finale = capitale iniziale + somma dei P&L dei trade (prima cassa ed equity si mescolavano)."""
    bt = BacktestEngine(initial_capital=100_000, max_positions=3)
    res = bt.run(universe, _signals_random(universe), stop_loss=0.05, trailing_stop=0.05, max_holding_days=20)
    assert res.total_trades > 20
    assert res.equity_curve.iloc[-1] == pytest.approx(100_000 + sum(t.pnl for t in res.trades), abs=1e-6)


def test_contabilita_quadra_con_short(universe):
    bt = BacktestEngine(initial_capital=100_000, max_positions=4, allow_short=True)
    res = bt.run(universe, _signals_random(universe, 1), stop_loss=0.05, max_holding_days=15)
    assert any(t.side == 'sell' for t in res.trades)
    assert res.equity_curve.iloc[-1] == pytest.approx(100_000 + sum(t.pnl for t in res.trades), abs=1e-6)


def test_esecuzione_alla_barra_successiva():
    """Il segnale del giorno t si esegue all'apertura di t+1, non alla chiusura di t."""
    df = make_prices(n=50, seed=2)
    sig = pd.Series(0.0, index=df.index)
    t = df.index[10]
    sig.loc[t] = 0.9
    bt = BacktestEngine(slippage=0.0, commission=0.0)
    res = bt.run({'X': df}, {'X': sig}, max_holding_days=3)
    trade = res.trades[0]
    assert trade.entry_date == df.index[11]
    assert trade.entry_price == pytest.approx(df.loc[df.index[11], 'open'])


def test_nessun_trade_sotto_soglia(universe):
    signals = {s: pd.Series(0.2, index=df.index) for s, df in universe.items()}
    res = BacktestEngine(entry_threshold=0.3).run(universe, signals)
    assert res.total_trades == 0
    assert res.total_return == pytest.approx(0.0)


def test_stop_loss_scatta():
    idx = pd.bdate_range('2023-01-02', periods=30)
    close = np.r_[np.full(5, 100.0), np.linspace(100, 80, 25)]
    df = pd.DataFrame({'open': close, 'high': close, 'low': close, 'close': close, 'volume': 1e6}, index=idx)
    sig = pd.Series(0.0, index=idx)
    sig.iloc[1] = 1.0
    res = BacktestEngine(slippage=0, commission=0).run({'X': df}, {'X': sig}, stop_loss=0.05)
    assert res.trades[0].exit_reason == 'stop_loss'
    assert res.trades[0].pnl_percent < -4


def test_mesi_e_anni(universe):
    res = BacktestEngine().run(universe, _signals_random(universe))
    assert len(res.monthly_returns) > 12 and len(res.yearly_returns) >= 1
