# execution/broker.py
"""
Adattatori broker.

- AlpacaBroker    -> libreria ufficiale `alpaca-py` (la vecchia `alpaca-trade-api` è deprecata)
- IBKRBroker      -> `ib_insync` (richiede TWS o IB Gateway in esecuzione)
- SimulatedBroker -> broker finto, in memoria, per test e prove a secco senza conto reale
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta, timezone
import time
import logging

import pandas as pd

from .orders import Order, OrderSide, OrderType, OrderStatus

logger = logging.getLogger(__name__)

OHLCV = ['open', 'high', 'low', 'close', 'volume']


class BrokerInterface(ABC):
    """Interfaccia astratta per la connessione al broker."""

    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def get_historical_data(self, symbol: str, lookback: int, interval: str = '1d') -> pd.DataFrame:
        """Restituisce un DataFrame OHLCV (colonne minuscole) con circa `lookback` barre."""
        pass

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        pass

    @abstractmethod
    def place_order(self, order: Order) -> Order:
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        pass

    @abstractmethod
    def get_open_orders(self) -> List[Order]:
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, int]:
        pass

    @abstractmethod
    def get_account_balance(self) -> float:
        """Valore totale del conto (equity)."""
        pass

    def get_cash(self) -> float:
        """Liquidità disponibile. Di default coincide con l'equity."""
        return self.get_account_balance()

    def is_market_open(self) -> bool:
        """True se il mercato è aperto. Di default si assume aperto."""
        return True


# --------------------------------------------------------------------------- #
# Alpaca
# --------------------------------------------------------------------------- #
class AlpacaBroker(BrokerInterface):
    """Implementazione per Alpaca con `alpaca-py`."""

    _TIMEFRAMES = {'1d': ('Day', 1), '1h': ('Hour', 1), '5m': ('Minute', 5), '1m': ('Minute', 1)}

    def __init__(self, api_key: str, secret_key: str, paper: bool = True,
                 data_feed: str = 'iex', fill_timeout: float = 10.0):
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        # 'iex' è il feed gratuito; 'sip' richiede l'abbonamento dati a pagamento
        self.data_feed = data_feed
        self.fill_timeout = fill_timeout
        self.trading = None
        self.data = None
        self.is_connected = False

    def connect(self) -> bool:
        if not self.api_key or not self.secret_key:
            logger.error("Chiavi Alpaca mancanti: imposta ALPACA_API_KEY e ALPACA_SECRET_KEY nel file .env")
            return False
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient

            self.trading = TradingClient(self.api_key, self.secret_key, paper=self.paper)
            self.data = StockHistoricalDataClient(self.api_key, self.secret_key)
            self.trading.get_account()  # verifica reale delle credenziali
            self.is_connected = True
            logger.info(f"Connesso ad Alpaca ({'paper' if self.paper else 'LIVE'})")
            return True
        except Exception as e:
            logger.error(f"Errore connessione ad Alpaca: {e}")
            return False

    def disconnect(self):
        self.is_connected = False
        logger.info("Disconnesso da Alpaca")

    def get_historical_data(self, symbol: str, lookback: int, interval: str = '1d') -> pd.DataFrame:
        if not self.is_connected:
            return pd.DataFrame()
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            from alpaca.data.enums import DataFeed, Adjustment

            unit_name, amount = self._TIMEFRAMES.get(interval, ('Day', 1))
            timeframe = TimeFrame(amount, getattr(TimeFrameUnit, unit_name))

            # Serve una data di inizio: senza, Alpaca restituisce solo le barre più recenti.
            # ~1.6 giorni di calendario per ogni giorno di borsa (weekend e festivi).
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=int(lookback * 1.6) + 10)

            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                adjustment=Adjustment.ALL,  # prezzi corretti per split e dividendi
                feed=DataFeed(self.data_feed),
            )
            bars = self.data.get_stock_bars(request).df
            if bars.empty:
                return pd.DataFrame()

            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.xs(symbol, level='symbol')
            bars = bars[OHLCV].astype(float)
            return bars.iloc[-lookback:]
        except Exception as e:
            logger.error(f"Errore recupero dati per {symbol}: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> float:
        if not self.is_connected:
            return 0.0
        try:
            from alpaca.data.requests import StockLatestTradeRequest
            from alpaca.data.enums import DataFeed

            req = StockLatestTradeRequest(symbol_or_symbols=symbol, feed=DataFeed(self.data_feed))
            return float(self.data.get_stock_latest_trade(req)[symbol].price)
        except Exception as e:
            logger.warning(f"Prezzo corrente non disponibile per {symbol}: {e}")
            return 0.0

    def place_order(self, order: Order) -> Order:
        if not self.is_connected:
            order.status = OrderStatus.REJECTED
            return order
        try:
            from alpaca.trading.requests import (MarketOrderRequest, LimitOrderRequest,
                                                 StopOrderRequest, StopLimitOrderRequest)
            from alpaca.trading.enums import OrderSide as ASide, TimeInForce

            common = dict(
                symbol=order.symbol,
                qty=order.quantity,
                side=ASide.BUY if order.side == OrderSide.BUY else ASide.SELL,
                time_in_force=TimeInForce.DAY,
                client_order_id=order.order_id,
            )
            limit_price = order.limit_price or order.price
            if order.order_type == OrderType.LIMIT and limit_price:
                req = LimitOrderRequest(limit_price=limit_price, **common)
            elif order.order_type == OrderType.STOP and order.stop_price:
                req = StopOrderRequest(stop_price=order.stop_price, **common)
            elif order.order_type == OrderType.STOP_LIMIT and order.stop_price and limit_price:
                req = StopLimitOrderRequest(stop_price=order.stop_price, limit_price=limit_price, **common)
            else:
                req = MarketOrderRequest(**common)

            resp = self.trading.submit_order(req)
            order.broker_order_id = str(resp.id)

            # Un ordine a mercato torna quasi sempre "accepted"/"new": aspettiamo l'esito.
            # Prima il bot lo trattava come non eseguito e non registrava mai l'acquisto.
            deadline = time.time() + self.fill_timeout
            while True:
                status = getattr(resp.status, 'value', str(resp.status))
                if status in ('filled', 'canceled', 'expired', 'rejected') or time.time() > deadline:
                    break
                time.sleep(0.5)
                resp = self.trading.get_order_by_id(resp.id)

            self._apply_status(order, resp)
            return order
        except Exception as e:
            logger.error(f"Errore invio ordine {order.symbol}: {e}")
            order.status = OrderStatus.REJECTED
            return order

    @staticmethod
    def _apply_status(order: Order, resp) -> None:
        status = getattr(resp.status, 'value', str(resp.status))
        if status == 'filled':
            order.status = OrderStatus.FILLED
        elif status == 'partially_filled':
            order.status = OrderStatus.PARTIALLY_FILLED
        elif status in ('canceled', 'expired'):
            order.status = OrderStatus.CANCELLED
        elif status == 'rejected':
            order.status = OrderStatus.REJECTED
        else:
            order.status = OrderStatus.PENDING
        if resp.filled_avg_price is not None:
            order.filled_price = float(resp.filled_avg_price)
        if resp.filled_qty is not None:
            order.filled_quantity = int(float(resp.filled_qty))
        order.filled_at = resp.filled_at

    def cancel_order(self, order_id: str) -> bool:
        try:
            self.trading.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            logger.error(f"Errore annullamento ordine {order_id}: {e}")
            return False

    def get_open_orders(self) -> List[Order]:
        if not self.is_connected:
            return []
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            result = []
            for o in self.trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN)):
                side = OrderSide.BUY if getattr(o.side, 'value', o.side) == 'buy' else OrderSide.SELL
                order = Order(symbol=o.symbol, side=side, quantity=int(float(o.qty or 0)))
                order.broker_order_id = str(o.id)
                result.append(order)
            return result
        except Exception as e:
            logger.error(f"Errore lettura ordini aperti: {e}")
            return []

    def get_positions(self) -> Dict[str, int]:
        if not self.is_connected:
            return {}
        try:
            return {p.symbol: int(float(p.qty)) for p in self.trading.get_all_positions()}
        except Exception as e:
            logger.error(f"Errore lettura posizioni: {e}")
            return {}

    def get_account_balance(self) -> float:
        if not self.is_connected:
            return 0.0
        try:
            return float(self.trading.get_account().equity)
        except Exception as e:
            logger.error(f"Errore lettura conto: {e}")
            return 0.0

    def get_cash(self) -> float:
        if not self.is_connected:
            return 0.0
        try:
            return float(self.trading.get_account().cash)
        except Exception:
            return 0.0

    def is_market_open(self) -> bool:
        # A mercato chiuso un ordine a mercato resta in coda fino all'apertura:
        # il bot in quel caso non invia ordini.
        try:
            return bool(self.trading.get_clock().is_open)
        except Exception:
            return False


# --------------------------------------------------------------------------- #
# Interactive Brokers
# --------------------------------------------------------------------------- #
class IBKRBroker(BrokerInterface):
    """Implementazione per Interactive Brokers (TWS / IB Gateway)."""

    # ib_insync vuole stringhe tipo '1 day', non '1d'
    _BAR_SIZE = {'1d': '1 day', '1h': '1 hour', '5m': '5 mins', '1m': '1 min'}

    def __init__(self, host: str = '127.0.0.1', port: int = 7497, client_id: int = 1,
                 fill_timeout: float = 10.0):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.fill_timeout = fill_timeout
        self.ib = None
        self.is_connected = False

    def connect(self) -> bool:
        try:
            from ib_insync import IB
            self.ib = IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.is_connected = True
            logger.info("Connesso a Interactive Brokers")
            return True
        except Exception as e:
            logger.error(f"Errore connessione a IBKR ({self.host}:{self.port}): {e}")
            return False

    def disconnect(self):
        if self.ib and self.is_connected:
            self.ib.disconnect()
            self.is_connected = False
            logger.info("Disconnesso da Interactive Brokers")

    def _create_contract(self, symbol: str):
        from ib_insync import Stock
        return Stock(symbol, 'SMART', 'USD')

    def get_historical_data(self, symbol: str, lookback: int, interval: str = '1d') -> pd.DataFrame:
        if not self.is_connected:
            return pd.DataFrame()
        try:
            from ib_insync import util

            contract = self._create_contract(symbol)
            if interval == '1d':
                years = max(1, int(lookback / 252) + 1)
                duration = f'{years} Y'
            else:
                duration = '1 M'

            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=self._BAR_SIZE.get(interval, '1 day'),
                whatToShow='ADJUSTED_LAST' if interval == '1d' else 'TRADES',
                useRTH=True,
                formatDate=1,
            )
            df = util.df(bars)
            if df is None or df.empty:
                return pd.DataFrame()
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            return df[OHLCV].astype(float).iloc[-lookback:]
        except Exception as e:
            logger.error(f"Errore recupero dati per {symbol}: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> float:
        if not self.is_connected:
            return 0.0
        try:
            import math
            contract = self._create_contract(symbol)
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(1.0)
            price = ticker.marketPrice()
            if price is None or math.isnan(price):
                price = ticker.close
            self.ib.cancelMktData(contract)
            return float(price) if price and not math.isnan(price) else 0.0
        except Exception as e:
            logger.warning(f"Prezzo corrente non disponibile per {symbol}: {e}")
            return 0.0

    def place_order(self, order: Order) -> Order:
        if not self.is_connected:
            order.status = OrderStatus.REJECTED
            return order
        try:
            from ib_insync import MarketOrder, LimitOrder, StopOrder

            contract = self._create_contract(order.symbol)
            action = 'BUY' if order.side == OrderSide.BUY else 'SELL'  # IB vuole maiuscolo
            limit_price = order.limit_price or order.price

            if order.order_type == OrderType.LIMIT and limit_price:
                ib_order = LimitOrder(action, order.quantity, limit_price)
            elif order.order_type == OrderType.STOP and order.stop_price:
                ib_order = StopOrder(action, order.quantity, order.stop_price)
            else:
                ib_order = MarketOrder(action, order.quantity)

            trade = self.ib.placeOrder(contract, ib_order)
            deadline = time.time() + self.fill_timeout
            while not trade.isDone() and time.time() < deadline:
                self.ib.sleep(0.5)

            status = trade.orderStatus.status
            if status == 'Filled':
                order.status = OrderStatus.FILLED
                order.filled_price = float(trade.orderStatus.avgFillPrice)
                order.filled_quantity = int(trade.orderStatus.filled)
                order.filled_at = datetime.now()
            elif status in ('Cancelled', 'ApiCancelled', 'Inactive'):
                order.status = OrderStatus.REJECTED
            else:
                order.status = OrderStatus.PENDING

            order.broker_order_id = str(trade.order.orderId)
            return order
        except Exception as e:
            logger.error(f"Errore invio ordine {order.symbol}: {e}")
            order.status = OrderStatus.REJECTED
            return order

    def cancel_order(self, order_id: str) -> bool:
        try:
            for trade in self.ib.openTrades():
                if str(trade.order.orderId) == str(order_id):
                    self.ib.cancelOrder(trade.order)
                    return True
            return False
        except Exception:
            return False

    def get_open_orders(self) -> List[Order]:
        if not self.is_connected:
            return []
        result = []
        for trade in self.ib.openTrades():
            side = OrderSide.BUY if trade.order.action == 'BUY' else OrderSide.SELL
            order = Order(symbol=trade.contract.symbol, side=side, quantity=int(trade.order.totalQuantity))
            order.broker_order_id = str(trade.order.orderId)
            result.append(order)
        return result

    def get_positions(self) -> Dict[str, int]:
        if not self.is_connected:
            return {}
        try:
            return {p.contract.symbol: int(p.position) for p in self.ib.positions() if p.position}
        except Exception as e:
            logger.error(f"Errore lettura posizioni: {e}")
            return {}

    def _account_value(self, tag: str) -> float:
        for item in self.ib.accountSummary():
            if item.tag == tag and item.currency in ('USD', 'BASE', ''):
                return float(item.value)
        return 0.0

    def get_account_balance(self) -> float:
        if not self.is_connected:
            return 0.0
        try:
            return self._account_value('NetLiquidation')
        except Exception:
            return 0.0

    def get_cash(self) -> float:
        if not self.is_connected:
            return 0.0
        try:
            return self._account_value('AvailableFunds')
        except Exception:
            return 0.0


# --------------------------------------------------------------------------- #
# Broker simulato
# --------------------------------------------------------------------------- #
class SimulatedBroker(BrokerInterface):
    """
    Broker in memoria: esegue gli ordini a mercato all'ultimo prezzo (+ slippage).
    Serve per provare il ciclo completo del bot senza conto reale.

    I dati arrivano da `data` (dict simbolo -> DataFrame OHLCV) oppure, se non
    forniti, da yfinance tramite data.market_data.MarketData.
    Con `advance()` si sposta in avanti di una barra il "giorno corrente".
    """

    def __init__(self, initial_cash: float = 100000.0,
                 data: Optional[Dict[str, pd.DataFrame]] = None,
                 slippage: float = 0.0005, commission: float = 0.001,
                 start_index: Optional[int] = None):
        self.cash = float(initial_cash)
        self.positions: Dict[str, int] = {}
        self.slippage = slippage
        self.commission = commission
        self.data = data
        self._market = None
        self.is_connected = False
        self.filled_orders: List[Order] = []
        # indice della barra "di oggi" per ciascun simbolo (solo con dati forniti)
        self._cursor = start_index

    def connect(self) -> bool:
        if self.data is None:
            from data.market_data import MarketData
            self._market = MarketData()
        self.is_connected = True
        logger.info("Connesso al broker SIMULATO (nessun ordine reale)")
        return True

    def disconnect(self):
        self.is_connected = False

    def advance(self, bars: int = 1):
        if self.data is not None and self._cursor is not None:
            self._cursor += bars

    def _frame(self, symbol: str, lookback: int = 400) -> pd.DataFrame:
        if self.data is not None:
            df = self.data.get(symbol)
            if df is None:
                return pd.DataFrame()
            end = len(df) if self._cursor is None else min(self._cursor, len(df))
            return df.iloc[max(0, end - lookback):end]
        return self._market.get_historical_data(symbol, lookback)

    def get_historical_data(self, symbol: str, lookback: int, interval: str = '1d') -> pd.DataFrame:
        df = self._frame(symbol, lookback)
        return df[OHLCV].copy() if not df.empty else df

    def get_current_price(self, symbol: str) -> float:
        df = self._frame(symbol, 1)
        return float(df['close'].iloc[-1]) if not df.empty else 0.0

    def place_order(self, order: Order) -> Order:
        price = self.get_current_price(order.symbol)
        if not self.is_connected or price <= 0 or order.quantity <= 0:
            order.status = OrderStatus.REJECTED
            return order

        if order.side == OrderSide.BUY:
            fill = price * (1 + self.slippage)
            cost = fill * order.quantity * (1 + self.commission)
            if cost > self.cash:
                order.status = OrderStatus.REJECTED
                order.notes = 'liquidità insufficiente'
                return order
            self.cash -= cost
            self.positions[order.symbol] = self.positions.get(order.symbol, 0) + order.quantity
        else:
            held = self.positions.get(order.symbol, 0)
            if order.quantity > held:
                order.status = OrderStatus.REJECTED
                order.notes = 'vendita allo scoperto non supportata'
                return order
            fill = price * (1 - self.slippage)
            self.cash += fill * order.quantity * (1 - self.commission)
            self.positions[order.symbol] = held - order.quantity
            if self.positions[order.symbol] == 0:
                del self.positions[order.symbol]

        order.status = OrderStatus.FILLED
        order.filled_price = fill
        order.filled_quantity = order.quantity
        order.filled_at = datetime.now()
        self.filled_orders.append(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        return False

    def get_open_orders(self) -> List[Order]:
        return []

    def get_positions(self) -> Dict[str, int]:
        return dict(self.positions)

    def get_account_balance(self) -> float:
        value = self.cash
        for symbol, qty in self.positions.items():
            value += qty * self.get_current_price(symbol)
        return value

    def get_cash(self) -> float:
        return self.cash


def get_broker(broker_type: str = 'alpaca', **kwargs) -> BrokerInterface:
    """
    Factory per creare il broker appropriato.
    Gli argomenti non pertinenti al broker scelto vengono ignorati.
    """
    broker_type = (broker_type or 'alpaca').lower()

    def pick(*names):
        return {k: kwargs[k] for k in names if k in kwargs and kwargs[k] is not None}

    if broker_type == 'alpaca':
        return AlpacaBroker(**pick('api_key', 'secret_key', 'paper', 'data_feed', 'fill_timeout'))
    if broker_type in ('ib', 'ibkr'):
        return IBKRBroker(**pick('host', 'port', 'client_id', 'fill_timeout'))
    if broker_type in ('sim', 'simulated', 'paper_sim'):
        return SimulatedBroker(**pick('initial_cash', 'data', 'slippage', 'commission', 'start_index'))
    raise ValueError(f"Broker non supportato: {broker_type} (usa 'alpaca', 'ib' o 'sim')")
