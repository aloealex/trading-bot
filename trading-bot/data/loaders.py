# data/loaders.py
"""
Caricamento dei prezzi storici per backtest e addestramento.

- da yfinance (richiede connessione internet)
- da una cartella di CSV: un file per simbolo, es. dati/AAPL.csv, con colonne
  date, open, high, low, close, volume (maiuscole/minuscole indifferenti)
"""
import os
import logging
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

OHLCV = ['open', 'high', 'low', 'close', 'volume']


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
    if 'adj_close' in df.columns and 'close' in df.columns:
        # usa il prezzo rettificato, riscalando anche open/high/low
        factor = df['adj_close'] / df['close']
        for col in ('open', 'high', 'low'):
            if col in df.columns:
                df[col] = df[col] * factor
        df['close'] = df['adj_close']
    for col in ('date', 'datetime', 'timestamp'):
        if col in df.columns:
            df = df.set_index(col)
            break
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='last')]
    if 'close' not in df.columns:
        raise ValueError("colonna 'close' mancante")
    for col in OHLCV:
        if col not in df.columns:
            df[col] = df['close'] if col != 'volume' else 0.0
    return df[OHLCV].astype(float).dropna(subset=['close'])


def load_csv_dir(directory: str, symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Carica tutti i CSV di una cartella (o solo quelli dei simboli indicati)."""
    data = {}
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Cartella dati non trovata: {directory}")
    for name in sorted(os.listdir(directory)):
        if not name.lower().endswith('.csv'):
            continue
        symbol = os.path.splitext(name)[0].upper()
        if symbols and symbol not in [s.upper() for s in symbols]:
            continue
        try:
            data[symbol] = _normalize(pd.read_csv(os.path.join(directory, name)))
        except Exception as e:
            logger.warning(f"CSV ignorato {name}: {e}")
    return data


def load_yfinance(symbols: List[str], years: float = 5) -> Dict[str, pd.DataFrame]:
    """Scarica lo storico giornaliero (rettificato) da yfinance."""
    import yfinance as yf

    data = {}
    period = f"{int(round(years))}y" if years >= 1 else f"{int(years * 12)}mo"
    for symbol in symbols:
        try:
            df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
            if df.empty:
                logger.warning(f"Nessun dato yfinance per {symbol}")
                continue
            data[symbol] = _normalize(df.reset_index())
        except Exception as e:
            logger.warning(f"Errore download {symbol}: {e}")
    return data


def load_prices(symbols: List[str], csv_dir: Optional[str] = None, years: float = 5) -> Dict[str, pd.DataFrame]:
    """CSV se è indicata una cartella, altrimenti yfinance."""
    if csv_dir:
        return load_csv_dir(csv_dir, symbols)
    return load_yfinance(symbols, years)
