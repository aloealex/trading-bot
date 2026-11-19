# config/settings.py
from dataclasses import dataclass, field, fields
from typing import List, Dict, Optional, Any
import json
import os
import re

@dataclass
class BrokerSettings:
    """Configurazione del broker"""
    broker_type: str = 'alpaca'  # 'alpaca', 'ib', 'sim' (simulato, nessun ordine reale)
    paper: bool = True           # Alpaca: conto paper (finto). Metti False SOLO per il conto reale
    data_feed: str = 'iex'       # Alpaca: 'iex' gratuito, 'sip' a pagamento
    host: str = '127.0.0.1'
    port: int = 7497
    client_id: int = 1
    account_id: str = ''
    
    # Alpaca
    alpaca_api_key: str = field(default_factory=lambda: os.getenv('ALPACA_API_KEY', ''))
    alpaca_secret_key: str = field(default_factory=lambda: os.getenv('ALPACA_SECRET_KEY', ''))
    alpaca_base_url: str = 'https://paper-api.alpaca.markets'
    
    # Interactive Brokers
    ib_host: str = '127.0.0.1'
    ib_port: int = 7497
    ib_client_id: int = 1

@dataclass
class TradingSettings:
    """Parametri di trading"""
    initial_capital: float = 100000.0
    max_positions: int = 10
    max_position_size: float = 0.20  # 20% del capitale per posizione
    min_position_size: float = 0.01   # 1% del capitale per posizione
    position_size: float = 0.10       # quota base per posizione (x |gradimento|), live e backtest
    
    # Risk management
    kelly_fraction: float = 0.25
    volatility_target: float = 0.15
    max_drawdown: float = 0.25
    max_holding_days: int = 30
    
    # Thresholds
    buy_threshold: float = 0.3
    sell_threshold: float = -0.3
    neutral_threshold: float = 0.1
    strong_buy_threshold: float = 0.6
    strong_sell_threshold: float = -0.6
    
    # Stop loss e take profit
    stop_loss: float = 0.05    # 5%
    take_profit: float = 0.15   # 15%
    trailing_stop: float = 0.03 # 3%
    
    # Commissioni e slippage
    commission: float = 0.001   # 0.1%
    slippage: float = 0.0005    # 0.05%

@dataclass
class SignalSettings:
    """Parametri degli algoritmi di segnale"""
    # Clenow
    momentum_window: int = 125
    volatility_window: int = 20
    min_observations: int = 30
    
    # Raschke
    ema_fast: int = 3
    ema_slow: int = 10
    divergence_window: int = 10
    smoothing: int = 16
    
    # Brandimarte (Monte Carlo)
    mc_simulations: int = 10000
    mc_horizon: int = 252
    mc_confidence: float = 0.95
    mc_use_bootstrap: bool = True
    
    # Ensemble weights
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        'momentum': 0.40,
        'raschke': 0.25,
        'risk': -0.20,
        'mean_reversion': 0.15
    })

@dataclass
class ModelSettings:
    """Configurazione del modello IA"""
    use_ml: bool = False          # False = segnali diretti (Clenow/Raschke/Monte Carlo)
    model_type: str = 'xgboost'   # 'xgboost', 'neural', 'ensemble'
    lookback_days: int = 300      # barre di storico scaricate (servono almeno 200 + margine)
    retrain_frequency: int = 30  # giorni
    sequence_length: int = 20
    
    features: List[str] = field(default_factory=lambda: [
        'momentum_10d', 'momentum_20d', 'momentum_50d', 'momentum_125d',
        'price_vs_sma_20', 'price_vs_sma_50', 'price_vs_sma_200',
        'volatility_20d', 'volatility_50d', 'volatility_ratio',
        'rsi_14', 'efficiency_ratio',
        'clenow_score', 'raschke_signal',
        'risk_volatility', 'risk_var_95', 'risk_sharpe', 'risk_drawdown',
        'higher_high', 'lower_low',
        'range_10d', 'range_20d',
        'return_skew', 'return_kurtosis', 'current_drawdown'
    ])
    
    # XGBoost
    xgboost_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0
    })
    
    # Neural Network
    neural_params: Dict[str, Any] = field(default_factory=lambda: {
        'hidden_dims': [512, 256, 128],
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'batch_size': 256,
        'epochs': 100
    })
    
    # Model paths
    model_path: str = './saved_models/xgboost'
    use_gpu: bool = False

@dataclass
class NotificationSettings:
    """Configurazione notifiche"""
    telegram_enabled: bool = False
    telegram_bot_token: str = ''
    telegram_chat_id: str = ''
    
    email_enabled: bool = False
    email_smtp_server: str = 'smtp.gmail.com'
    email_smtp_port: int = 587
    email_username: str = ''
    email_password: str = ''
    email_recipient: str = ''
    
    alerts_on_trade: bool = True
    alerts_on_error: bool = True
    alerts_daily_summary: bool = True

@dataclass
class Settings:
    """Container principale delle configurazioni"""
    broker: BrokerSettings = field(default_factory=BrokerSettings)
    trading: TradingSettings = field(default_factory=TradingSettings)
    signals: SignalSettings = field(default_factory=SignalSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    notifications: NotificationSettings = field(default_factory=NotificationSettings)
    
    symbols: List[str] = field(default_factory=lambda: [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
        'META', 'TSLA', 'AMD', 'INTC', 'NFLX'
    ])
    
    update_interval: int = 300  # secondi
    log_level: str = 'INFO'
    close_on_shutdown: bool = False
    state_file: str = './state/bot_state.json'  # letto dalla dashboard
    
    def to_dict(self) -> Dict:
        """Converte le impostazioni in un dizionario"""
        return {
            'broker': self.broker.__dict__,
            'trading': self.trading.__dict__,
            'signals': self.signals.__dict__,
            'model': self.model.__dict__,
            'notifications': self.notifications.__dict__,
            'symbols': self.symbols,
            'update_interval': self.update_interval,
            'log_level': self.log_level,
            'close_on_shutdown': self.close_on_shutdown
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Settings':
        """Crea impostazioni da un dizionario annidato (stessa struttura di to_dict)."""
        settings = cls()
        sections = {'broker': BrokerSettings, 'trading': TradingSettings, 'signals': SignalSettings,
                    'model': ModelSettings, 'notifications': NotificationSettings}
        for name, klass in sections.items():
            if isinstance(data.get(name), dict):
                setattr(settings, name, _build(klass, data[name]))
        for key in ('symbols', 'update_interval', 'log_level', 'close_on_shutdown', 'state_file'):
            if key in data:
                setattr(settings, key, data[key])
        return settings

    @classmethod
    def from_json(cls, path: str) -> 'Settings':
        """
        Carica config/production.json. Accetta sia il formato "piatto" storico
        (initial_capital, buy_threshold, ... al primo livello) sia quello annidato.
        I valori "${NOME}" vengono sostituiti con le variabili d'ambiente (file .env).
        """
        with open(path, encoding='utf-8') as f:
            raw = _expand_env(json.load(f))

        if isinstance(raw.get('broker'), dict):  # formato annidato
            return cls.from_dict(raw)

        s = cls()
        # broker
        if 'broker' in raw:
            s.broker.broker_type = raw['broker']
        bc = raw.get('broker_config', {})
        s.broker.paper = bc.get('paper', s.broker.paper)
        s.broker.alpaca_api_key = bc.get('api_key') or s.broker.alpaca_api_key
        s.broker.alpaca_secret_key = bc.get('secret_key') or s.broker.alpaca_secret_key
        s.broker.data_feed = bc.get('data_feed', s.broker.data_feed)
        s.broker.ib_host = bc.get('host', s.broker.ib_host)
        s.broker.ib_port = int(bc.get('port', s.broker.ib_port))
        s.broker.ib_client_id = int(bc.get('client_id', s.broker.ib_client_id))
        # trading: tutte le chiavi con lo stesso nome del campo
        _apply(s.trading, raw)
        _apply(s.signals, raw)
        # modello
        _apply(s.model, raw)
        # generali
        for key in ('symbols', 'update_interval', 'close_on_shutdown', 'state_file'):
            if key in raw:
                setattr(s, key, raw[key])
        s.log_level = raw.get('logging', {}).get('level', s.log_level)
        # notifiche
        notif = raw.get('notifications', {})
        tg, em = notif.get('telegram', {}), notif.get('email', {})
        n = s.notifications
        n.telegram_enabled = bool(tg.get('enabled', n.telegram_enabled))
        n.telegram_bot_token = tg.get('bot_token', n.telegram_bot_token)
        n.telegram_chat_id = tg.get('chat_id', n.telegram_chat_id)
        n.email_enabled = bool(em.get('enabled', n.email_enabled))
        n.email_smtp_server = em.get('smtp_server', n.email_smtp_server)
        n.email_smtp_port = int(em.get('smtp_port', n.email_smtp_port))
        n.email_username = em.get('username', n.email_username)
        n.email_password = em.get('password', n.email_password)
        n.email_recipient = em.get('recipient', n.email_recipient)
        return s


def _build(klass, values: Dict):
    """Crea un dataclass ignorando le chiavi sconosciute (prima un refuso mandava in crash)."""
    names = {f.name for f in fields(klass)}
    return klass(**{k: v for k, v in values.items() if k in names})


def _apply(obj, values: Dict):
    for f in fields(obj):
        if f.name in values:
            setattr(obj, f.name, values[f.name])


_ENV = re.compile(r'\$\{([A-Za-z0-9_]+)\}')


def _expand_env(value):
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, str):
        return _ENV.sub(lambda m: os.getenv(m.group(1), ''), value)
    return value
