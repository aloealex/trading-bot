# tests/test_signals.py
import numpy as np
import pandas as pd

from signals.ensemble import SignalEnsemble
from signals.raschke import RaschkePatterns
from signals.clenow import ClenowMomentum
from tests.synthetic import make_prices


def test_gradimento_deterministico(prices):
    a = SignalEnsemble().calculate(prices, 'X')
    b = SignalEnsemble().calculate(prices, 'X')
    assert a.gradimento == b.gradimento
    assert -1 <= a.gradimento <= 1


def test_dati_insufficienti():
    r = SignalEnsemble().calculate(make_prices(n=100)['close'])
    assert r.gradimento == 0.0 and r.confidence == 0.0


def test_peso_rischio_non_invertito():
    """Un titolo meno volatile (a parità di trend) deve avere un contributo di rischio migliore."""
    calmo = make_prices(n=400, seed=3, vol=0.008)['close']
    nervoso = make_prices(n=400, seed=3, vol=0.04)['close']
    e = SignalEnsemble()
    assert e.weights['risk'] > 0
    assert e.calculate(calmo).components['risk_score'] > e.calculate(nervoso).components['risk_score']


def test_vecchia_config_peso_negativo_corretta():
    e = SignalEnsemble(weights={'momentum': 0.4, 'raschke': 0.25, 'risk': -0.2, 'mean_reversion': 0.15})
    assert e.weights['risk'] == 0.2


def test_divergenza_raschke_puo_scattare():
    """Prima non poteva mai attivarsi (prezzo confrontato con un massimo che lo includeva)."""
    r = RaschkePatterns()
    found = set()
    for seed in range(40):
        p = make_prices(n=300, seed=seed)['close']
        for end in range(60, 300, 7):
            found.add(r.divergence(p.iloc[:end])[0])
    assert {-1, 1} <= found


def test_clenow_trend_positivo():
    idx = pd.bdate_range('2022-01-03', periods=200)
    up = pd.Series(100 * np.exp(np.linspace(0, 0.3, 200)), index=idx)
    assert ClenowMomentum().calculate(up) > 0
    assert ClenowMomentum().calculate(1 / up) < 0


def test_signal_series_causale(prices):
    e = SignalEnsemble(mc_simulations=300)
    series = e.signal_series(prices, start=380)
    t = series.index[5]
    expected = e.calculate(prices.loc[:t]).gradimento
    assert series.loc[t] == expected
    # modificare il futuro non cambia il passato
    future_changed = prices.copy()
    future_changed.loc[future_changed.index > t] *= 2
    assert e.signal_series(future_changed, start=380).loc[t] == series.loc[t]
