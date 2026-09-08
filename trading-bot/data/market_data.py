# data/market_data.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time
import logging

logger = logging.getLogger(__name__)

class MarketData:
    """
    Gestione dei dati di mercato da multiple fonti.
    """
    
    def __init__(self, 
                 primary_source: str = 'yfinance',
                 cache_enabled: bool = True,
                 cache_duration: int = 60):  # secondi
        self.primary_source = primary_source
        self.cache_enabled = cache_enabled
        self.cache_duration = cache_duration
        self._cache = {}
        self._cache_time = {}
        
        self._initialize_sources()
    
    def _initialize_sources(self):
        """Inizializza le fonti dati"""
        self.sources = {}
        
        if self.primary_source == 'yfinance':
            try:
                import yfinance as yf
                self.sources['yfinance'] = yf
                logger.info("yfinance caricato con successo")
            except ImportError:
                logger.error("yfinance non disponibile")
        
        # Altre fonti dati...
    
    def get_historical_data(self, 
                           symbol: str,
                           lookback_days: int = 252,
                           interval: str = '1d') -> pd.DataFrame:
        """
        Recupera i dati storici.
        """
        cache_key = f"{symbol}_{lookback_days}_{interval}"
        
        # Controlla la cache
        if self.cache_enabled and cache_key in self._cache:
            cache_age = time.time() - self._cache_time.get(cache_key, 0)
            if cache_age < self.cache_duration:
                logger.debug(f"Cache hit per {symbol}")
                return self._cache[cache_key].copy()
        
        logger.debug(f"Recupero dati per {symbol}")
        
        if self.primary_source == 'yfinance' and 'yfinance' in self.sources:
            yf = self.sources['yfinance']
            
            try:
                ticker = yf.Ticker(symbol)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=lookback_days * 2)
                
                df = ticker.history(start=start_date, end=end_date, interval=interval)
                
                if df.empty:
                    logger.warning(f"Nessun dato per {symbol}")
                    return pd.DataFrame()
                
                # Standardizza le colonne
                df = df.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                # Aggiungi info aggiuntive
                df['symbol'] = symbol
                
                # Cache
                if self.cache_enabled:
                    self._cache[cache_key] = df.copy()
                    self._cache_time[cache_key] = time.time()
                
                return df
                
            except Exception as e:
                logger.error(f"Errore nel recupero dati per {symbol}: {e}")
                return pd.DataFrame()
        
        return pd.DataFrame()
    
    def get_live_price(self, symbol: str) -> float:
        """
        Recupera il prezzo corrente.
        """
        try:
            if self.primary_source == 'yfinance' and 'yfinance' in self.sources:
                yf = self.sources['yfinance']
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='1d', interval='1m')
                if not data.empty:
                    return data['Close'].iloc[-1]
        except Exception as e:
            logger.error(f"Errore nel recupero prezzo per {symbol}: {e}")
        
        return 0.0
    
    def get_batch_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Recupera i prezzi correnti per più simboli"""
        prices = {}
        for symbol in symbols:
            prices[symbol] = self.get_live_price(symbol)
        return prices
    
    def get_historical_batch(self, 
                            symbols: List[str],
                            lookback_days: int = 252,
                            interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """Recupera dati storici per multipli simboli"""
        data = {}
        for symbol in symbols:
            df = self.get_historical_data(symbol, lookback_days, interval)
            if not df.empty:
                data[symbol] = df
        return data
    
    def get_market_snapshot(self, symbols: List[str]) -> pd.DataFrame:
        """
        Recupera uno snapshot di mercato per i simboli.
        """
        snapshot = []
        
        for symbol in symbols:
            df = self.get_historical_data(symbol, 5, '1d')
            if df.empty:
                continue
            
            current_price = df['close'].iloc[-1]
            prev_close = df['close'].iloc[-2] if len(df) > 1 else current_price
            
            snapshot.append({
                'symbol': symbol,
                'price': current_price,
                'change': (current_price - prev_close) / prev_close * 100,
                'volume': df['volume'].iloc[-1] if 'volume' in df.columns else 0,
                'timestamp': datetime.now()
            })
        
        return pd.DataFrame(snapshot)
    
    def clear_cache(self):
        """Pulisce la cache"""
        self._cache = {}
        self._cache_time = {}

class MarketDataProvider:
    """
    Provider di dati di mercato con supporto per multiple fonti.
    """
    
    def __init__(self):
        self.providers = {}
        self.default_provider = 'yfinance'
    
    def register_provider(self, name: str, provider) -> None:
        """Registra un provider di dati"""
        self.providers[name] = provider
    
    def get_data(self, 
                symbol: str,
                provider: Optional[str] = None,
                **kwargs) -> pd.DataFrame:
        """
        Recupera dati da un provider specifico.
        """
        provider_name = provider or self.default_provider
        
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} non trovato")
        
        return self.providers[provider_name].get_historical_data(symbol, **kwargs)