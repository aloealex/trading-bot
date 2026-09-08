# data/database.py
import pandas as pd
import sqlite3
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class Database:
    """
    Gestione del database per i dati storici.
    """
    
    def __init__(self, db_path: str = 'trading_data.db'):
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self):
        """Inizializza il database e crea le tabelle"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Tabella prezzi storici
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    UNIQUE(symbol, date)
                )
            ''')
            
            # Tabella segnali
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    signal REAL,
                    confidence REAL,
                    components TEXT,
                    UNIQUE(symbol, date)
                )
            ''')
            
            # Tabella trades
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    entry_date TEXT NOT NULL,
                    exit_date TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    quantity INTEGER,
                    side TEXT,
                    pnl REAL,
                    pnl_percent REAL,
                    exit_reason TEXT
                )
            ''')
            
            # Tabella performance giornaliera
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_performance (
                    date TEXT PRIMARY KEY,
                    capital REAL,
                    pnl REAL,
                    pnl_percent REAL,
                    positions INTEGER,
                    trades INTEGER
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info(f"Database inizializzato: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Errore nell'inizializzazione del database: {e}")
    
    def save_prices(self, df: pd.DataFrame) -> int:
        """
        Salva i prezzi nel database.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Prepara i dati
            df = df.copy()
            df['date'] = df.index.strftime('%Y-%m-%d')
            
            # Salva in batch
            df.to_sql('prices', conn, if_exists='append', index=False,
                     method='multi')
            
            conn.close()
            return len(df)
            
        except Exception as e:
            logger.error(f"Errore nel salvataggio dei prezzi: {e}")
            return 0
    
    def get_prices(self, 
                  symbol: str,
                  start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Recupera i prezzi dal database.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = "SELECT * FROM prices WHERE symbol = ?"
            params = [symbol]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date.strftime('%Y-%m-%d'))
            if end_date:
                query += " AND date <= ?"
                params.append(end_date.strftime('%Y-%m-%d'))
            
            query += " ORDER BY date"
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                df = df.sort_index()
            
            return df
            
        except Exception as e:
            logger.error(f"Errore nel recupero dei prezzi: {e}")
            return pd.DataFrame()
    
    def save_signal(self, symbol: str, date: datetime, 
                   signal: float, confidence: float, 
                   components: Dict[str, float]) -> None:
        """Salva un segnale nel database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            import json
            components_json = json.dumps(components)
            
            cursor.execute('''
                INSERT OR REPLACE INTO signals 
                (symbol, date, signal, confidence, components)
                VALUES (?, ?, ?, ?, ?)
            ''', (symbol, date.strftime('%Y-%m-%d'), signal, confidence, components_json))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Errore nel salvataggio del segnale: {e}")
    
    def save_trade(self, trade_data: Dict) -> None:
        """Salva un trade nel database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades 
                (symbol, entry_date, exit_date, entry_price, exit_price,
                 quantity, side, pnl, pnl_percent, exit_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data.get('symbol'),
                trade_data.get('entry_date'),
                trade_data.get('exit_date'),
                trade_data.get('entry_price'),
                trade_data.get('exit_price'),
                trade_data.get('quantity'),
                trade_data.get('side'),
                trade_data.get('pnl'),
                trade_data.get('pnl_percent'),
                trade_data.get('exit_reason')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Errore nel salvataggio del trade: {e}")
    
    def get_trades(self, 
                  symbol: Optional[str] = None,
                  start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Recupera i trades dal database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = "SELECT * FROM trades WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            if start_date:
                query += " AND entry_date >= ?"
                params.append(start_date.strftime('%Y-%m-%d'))
            if end_date:
                query += " AND entry_date <= ?"
                params.append(end_date.strftime('%Y-%m-%d'))
            
            query += " ORDER BY entry_date"
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Errore nel recupero dei trades: {e}")
            return pd.DataFrame()