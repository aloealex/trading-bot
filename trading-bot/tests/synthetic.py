# tests/synthetic.py
"""Generatore di prezzi sintetici (OHLCV) per test offline, riproducibile."""
import numpy as np
import pandas as pd


def make_prices(n: int = 800, seed: int = 0, drift: float = 0.0004, vol: float = 0.018,
                start: str = '2020-01-02', start_price: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # regimi di trend alternati, per avere segnali di momentum in entrambe le direzioni
    regime = np.repeat(rng.choice([-1.0, 0.0, 1.0], size=n // 60 + 1), 60)[:n]
    rets = rng.normal(drift + regime * 0.0015, vol, n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = close * np.exp(rng.normal(0, vol / 3, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume}, index=idx)


def make_universe(symbols=('AAA', 'BBB', 'CCC', 'DDD', 'EEE'), n: int = 800) -> dict:
    return {s: make_prices(n=n, seed=i + 1, start_price=50 + 30 * i) for i, s in enumerate(symbols)}


def write_csv_dir(directory: str, symbols=('AAA', 'BBB', 'CCC', 'DDD', 'EEE'), n: int = 800) -> dict:
    import os
    os.makedirs(directory, exist_ok=True)
    data = make_universe(symbols, n)
    for s, df in data.items():
        df.rename_axis('date').to_csv(os.path.join(directory, f'{s}.csv'))
    return data
