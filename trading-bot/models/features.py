# models/features.py
"""
Feature per il modello di machine learning.

Unica fonte di verità usata sia in addestramento sia in predizione: prima i due
percorsi calcolavano feature diverse (e in addestramento alcune erano calcolate
sull'intera serie, cioè con informazioni dal futuro).

Ogni feature alla riga t usa SOLO dati fino a t (rolling/ewm all'indietro).
"""
import numpy as np
import pandas as pd

FEATURE_NAMES = [
    'momentum_5', 'momentum_10', 'momentum_20', 'momentum_50', 'momentum_125',
    'price_vs_sma_20', 'price_vs_sma_50', 'price_vs_sma_200',
    'volatility_20', 'volatility_50', 'volatility_ratio',
    'rsi_14', 'efficiency_ratio_20',
    'clenow_score', 'raschke_osc', 'raschke_slow',
    'macd_diff', 'bb_position', 'atr_pct',
    'range_10', 'range_20', 'drawdown_252',
    'return_skew_50', 'volume_ratio_20',
]

TARGET_HORIZON = 5  # giorni


def _rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(gain.notna())


def _rolling_clenow(close: pd.Series, window: int = 125) -> pd.Series:
    """Momentum di Clenow (pendenza annualizzata del log-prezzo * R²) su finestra mobile."""
    y = np.log(close)
    x = pd.Series(np.arange(len(close), dtype=float), index=close.index)
    mean_x = x.rolling(window).mean()
    mean_y = y.rolling(window).mean()
    cov = (x * y).rolling(window).mean() - mean_x * mean_y
    var_x = (x * x).rolling(window).mean() - mean_x ** 2
    var_y = (y * y).rolling(window).mean() - mean_y ** 2
    slope = cov / var_x
    r2 = (cov ** 2) / (var_x * var_y)
    annualized = (np.exp(slope * 252) - 1) * 100
    return (annualized * r2).clip(-100, 100)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcola la matrice delle feature (una riga per barra) da un DataFrame OHLCV."""
    close = df['close'].astype(float)
    high = df['high'].astype(float) if 'high' in df.columns else close
    low = df['low'].astype(float) if 'low' in df.columns else close
    returns = close.pct_change()
    f = pd.DataFrame(index=df.index)

    for p in (5, 10, 20, 50, 125):
        f[f'momentum_{p}'] = close / close.shift(p) - 1
    for p in (20, 50, 200):
        f[f'price_vs_sma_{p}'] = close / close.rolling(p).mean() - 1
    f['volatility_20'] = returns.rolling(20).std() * np.sqrt(252)
    f['volatility_50'] = returns.rolling(50).std() * np.sqrt(252)
    f['volatility_ratio'] = f['volatility_20'] / f['volatility_50']
    f['rsi_14'] = _rsi_wilder(close, 14)
    f['efficiency_ratio_20'] = (close - close.shift(20)).abs() / close.diff().abs().rolling(20).sum()
    f['clenow_score'] = _rolling_clenow(close, 125)

    osc = close.rolling(3).mean() - close.rolling(10).mean()
    f['raschke_osc'] = osc / close
    f['raschke_slow'] = osc.rolling(16).mean() / close

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    f['macd_diff'] = (macd - macd.ewm(span=9, adjust=False).mean()) / close
    mid, std = close.rolling(20).mean(), close.rolling(20).std()
    f['bb_position'] = (close - (mid - 2 * std)) / (4 * std)

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    f['atr_pct'] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / close
    for p in (10, 20):
        f[f'range_{p}'] = (close.rolling(p).max() - close.rolling(p).min()) / close.rolling(p).mean()
    f['drawdown_252'] = close / close.rolling(252, min_periods=20).max() - 1
    f['return_skew_50'] = returns.rolling(50).skew()
    if 'volume' in df.columns and df['volume'].fillna(0).sum() > 0:
        vol = df['volume'].astype(float)
        f['volume_ratio_20'] = vol / vol.rolling(20).mean()
    else:
        f['volume_ratio_20'] = 1.0

    return f[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)


def build_target(df: pd.DataFrame, horizon: int = TARGET_HORIZON) -> pd.Series:
    """Rendimento futuro a `horizon` giorni (solo per l'addestramento)."""
    close = df['close'].astype(float)
    return close.shift(-horizon) / close - 1


def build_dataset(data: dict, horizon: int = TARGET_HORIZON) -> pd.DataFrame:
    """
    Unisce più simboli in un unico dataset ordinato per data, con colonne
    FEATURE_NAMES + 'target' + 'symbol'. Le righe senza target (ultime `horizon`) sono escluse.
    """
    frames = []
    for symbol, df in data.items():
        if df is None or df.empty:
            continue
        f = build_features(df)
        f['target'] = build_target(df, horizon)
        f['symbol'] = symbol
        frames.append(f.dropna())
    if not frames:
        return pd.DataFrame(columns=FEATURE_NAMES + ['target', 'symbol'])
    out = pd.concat(frames)
    return out.sort_index(kind='stable')
