# config/symbols.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class SymbolConfig:
    """Configurazione per un singolo simbolo"""
    symbol: str
    name: str = ''
    sector: str = ''
    industry: str = ''
    market_cap: float = 0.0
    
    # Parametri specifici
    momentum_window: Optional[int] = None
    volatility_window: Optional[int] = None
    buy_threshold: Optional[float] = None
    sell_threshold: Optional[float] = None
    max_position_size: Optional[float] = None
    
    # Filtri
    min_volume: int = 100000
    min_price: float = 1.0
    max_price: float = 10000.0
    min_atr: float = 0.01
    max_atr: float = 100.0
    
    # Esclusione
    enabled: bool = True
    reason_disabled: str = ''

class Watchlist:
    """
    Gestisce la watchlist dei simboli.
    """
    
    def __init__(self):
        self.symbols: Dict[str, SymbolConfig] = {}
    
    def add_symbol(self, symbol: str, **kwargs) -> None:
        """Aggiunge un simbolo alla watchlist"""
        self.symbols[symbol] = SymbolConfig(symbol=symbol, **kwargs)
    
    def remove_symbol(self, symbol: str) -> bool:
        """Rimuove un simbolo dalla watchlist"""
        if symbol in self.symbols:
            del self.symbols[symbol]
            return True
        return False
    
    def get_enabled_symbols(self) -> List[str]:
        """Restituisce la lista dei simboli abilitati"""
        return [s for s, cfg in self.symbols.items() if cfg.enabled]
    
    def get_symbol_config(self, symbol: str) -> Optional[SymbolConfig]:
        """Restituisce la configurazione di un simbolo"""
        return self.symbols.get(symbol)
    
    def to_list(self) -> List[str]:
        """Restituisce la lista di tutti i simboli"""
        return list(self.symbols.keys())
    
    def from_list(self, symbols: List[str]) -> None:
        """Carica una lista di simboli"""
        for symbol in symbols:
            self.add_symbol(symbol)
    
    @classmethod
    def default_watchlist(cls) -> 'Watchlist':
        """Crea una watchlist predefinita"""
        watchlist = cls()
        
        # Tech stocks
        tech_stocks = [
            ('AAPL', 'Apple Inc.', 'Technology', 'Consumer Electronics'),
            ('MSFT', 'Microsoft Corp.', 'Technology', 'Software'),
            ('GOOGL', 'Alphabet Inc.', 'Technology', 'Internet'),
            ('AMZN', 'Amazon.com Inc.', 'Consumer Cyclical', 'E-commerce'),
            ('NVDA', 'NVIDIA Corp.', 'Technology', 'Semiconductors'),
            ('META', 'Meta Platforms', 'Technology', 'Social Media'),
            ('TSLA', 'Tesla Inc.', 'Consumer Cyclical', 'Automotive'),
            ('AMD', 'Advanced Micro Devices', 'Technology', 'Semiconductors'),
            ('INTC', 'Intel Corp.', 'Technology', 'Semiconductors'),
            ('NFLX', 'Netflix Inc.', 'Technology', 'Entertainment')
        ]
        
        for symbol, name, sector, industry in tech_stocks:
            watchlist.add_symbol(
                symbol=symbol,
                name=name,
                sector=sector,
                industry=industry
            )
        
        return watchlist