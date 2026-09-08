# execution/broker.py
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import pandas as pd
from datetime import datetime
import logging

from .orders import Order, OrderSide, OrderType, OrderStatus

logger = logging.getLogger(__name__)

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
        pass


class AlpacaBroker(BrokerInterface):
    """Implementazione per Alpaca."""
    
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = 'https://paper-api.alpaca.markets' if paper else 'https://api.alpaca.markets'
        self.client = None
        self.is_connected = False
    
    def connect(self) -> bool:
        try:
            import alpaca_trade_api as tradeapi
            self.client = tradeapi.REST(
                key_id=self.api_key,
                secret_key=self.secret_key,
                base_url=self.base_url
            )
            self.is_connected = True
            logger.info("Connesso ad Alpaca")
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
            timeframe_map = {
                '1d': '1Day',
                '1h': '1Hour',
                '5m': '5Min',
                '1m': '1Min'
            }
            
            bars = self.client.get_bars(
                symbol,
                timeframe_map.get(interval, '1Day'),
                limit=lookback
            ).df
            
            if bars.empty:
                return pd.DataFrame()
            
            bars = bars.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            return bars
            
        except Exception as e:
            logger.error(f"Errore recupero dati per {symbol}: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> float:
        if not self.is_connected:
            return 0.0
        
        try:
            quote = self.client.get_last_quote(symbol)
            return (quote.bid + quote.ask) / 2
        except:
            try:
                # Fallback con ultimo trade
                trade = self.client.get_last_trade(symbol)
                return trade.price
            except:
                return 0.0
    
    def place_order(self, order: Order) -> Order:
        if not self.is_connected:
            order.status = OrderStatus.REJECTED
            return order
        
        try:
            side = 'buy' if order.side == OrderSide.BUY else 'sell'
            order_type = order.order_type.value
            
            params = {
                'symbol': order.symbol,
                'qty': order.quantity,
                'side': side,
                'type': order_type,
                'time_in_force': 'day'
            }
            
            if order_type == 'limit' and order.limit_price:
                params['limit_price'] = order.limit_price
            elif order_type == 'stop' and order.stop_price:
                params['stop_price'] = order.stop_price
            elif order_type == 'stop_limit' and order.stop_price and order.limit_price:
                params['stop_price'] = order.stop_price
                params['limit_price'] = order.limit_price
            
            resp = self.client.submit_order(**params)
            
            if resp.status == 'filled':
                order.status = OrderStatus.FILLED
                order.filled_price = float(resp.filled_avg_price)
                order.filled_at = resp.filled_at
            elif resp.status in ['accepted', 'new']:
                order.status = OrderStatus.PENDING
            else:
                order.status = OrderStatus.REJECTED
            
            order.order_id = resp.id
            return order
            
        except Exception as e:
            logger.error(f"Errore invio ordine: {e}")
            order.status = OrderStatus.REJECTED
            return order
    
    def cancel_order(self, order_id: str) -> bool:
        try:
            self.client.cancel_order(order_id)
            return True
        except:
            return False
    
    def get_open_orders(self) -> List[Order]:
        if not self.is_connected:
            return []
        
        try:
            orders = []
            for order in self.client.list_orders():
                # Crea un oggetto Order da Alpaca
                # Implementazione semplificata
                pass
            return orders
        except:
            return []
    
    def get_positions(self) -> Dict[str, int]:
        if not self.is_connected:
            return {}
        
        positions = {}
        try:
            for pos in self.client.list_positions():
                positions[pos.symbol] = int(pos.qty)
        except:
            pass
        
        return positions
    
    def get_account_balance(self) -> float:
        if not self.is_connected:
            return 0.0
        
        try:
            account = self.client.get_account()
            return float(account.equity)
        except:
            return 0.0


class IBKRBroker(BrokerInterface):
    """Implementazione per Interactive Brokers."""
    
    def __init__(self, host: str = '127.0.0.1', port: int = 7497, client_id: int = 1):
        self.host = host
        self.port = port
        self.client_id = client_id
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
            logger.error(f"Errore connessione a IBKR: {e}")
            return False
    
    def disconnect(self):
        if self.ib and self.is_connected:
            self.ib.disconnect()
            self.is_connected = False
            logger.info("Disconnesso da Interactive Brokers")
    
    def _create_contract(self, symbol: str):
        """Crea un contratto Stock per IBKR."""
        from ib_insync import Stock
        return Stock(symbol, 'SMART', 'USD')
    
    def get_historical_data(self, symbol: str, lookback: int, interval: str = '1d') -> pd.DataFrame:
        if not self.is_connected:
            return pd.DataFrame()
        
        try:
            from ib_insync import util
            
            contract = self._create_contract(symbol)
            
            duration_map = {
                '1d': '1 Y',
                '1h': '1 M',
                '5m': '1 M',
                '1m': '1 M'
            }
            
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration_map.get(interval, '1 Y'),
                barSizeSetting=interval,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            
            df = util.df(bars)
            if df.empty:
                return pd.DataFrame()
            
            df = df.set_index('date')
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"Errore recupero dati per {symbol}: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> float:
        if not self.is_connected:
            return 0.0
        
        try:
            contract = self._create_contract(symbol)
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(0.5)
            return ticker.last if ticker.last else ticker.close
        except:
            return 0.0
    
    def place_order(self, order: Order) -> Order:
        if not self.is_connected:
            order.status = OrderStatus.REJECTED
            return order
        
        try:
            from ib_insync import MarketOrder, LimitOrder, StopOrder
            
            contract = self._create_contract(order.symbol)
            
            order_type_map = {
                OrderType.MARKET: MarketOrder,
                OrderType.LIMIT: LimitOrder,
                OrderType.STOP: StopOrder
            }
            
            order_class = order_type_map.get(order.order_type, MarketOrder)
            
            if order.order_type == OrderType.LIMIT and order.price:
                ib_order = order_class(order.side.value, order.quantity, order.price)
            elif order.order_type == OrderType.STOP and order.stop_price:
                ib_order = order_class(order.side.value, order.quantity, order.stop_price)
            else:
                ib_order = order_class(order.side.value, order.quantity)
            
            trade = self.ib.placeOrder(contract, ib_order)
            self.ib.sleep(0.5)
            
            if trade.orderStatus.status == 'Filled':
                order.status = OrderStatus.FILLED
                order.filled_price = trade.orderStatus.avgFillPrice
                order.filled_at = trade.orderStatus.lastFillDate
            elif trade.orderStatus.status == 'Submitted':
                order.status = OrderStatus.PENDING
            else:
                order.status = OrderStatus.REJECTED
            
            order.order_id = trade.order.orderId
            return order
            
        except Exception as e:
            logger.error(f"Errore invio ordine: {e}")
            order.status = OrderStatus.REJECTED
            return order
    
    def cancel_order(self, order_id: str) -> bool:
        try:
            self.ib.cancelOrder(order_id)
            return True
        except:
            return False
    
    def get_open_orders(self) -> List[Order]:
        if not self.is_connected:
            return []
        
        try:
            orders = []
            for order in self.ib.orders():
                # Crea un oggetto Order da IBKR
                # Implementazione semplificata
                pass
            return orders
        except:
            return []
    
    def get_positions(self) -> Dict[str, int]:
        if not self.is_connected:
            return {}
        
        positions = {}
        try:
            for pos in self.ib.positions():
                positions[pos.contract.symbol] = pos.position
        except:
            pass
        
        return positions
    
    def get_account_balance(self) -> float:
        if not self.is_connected:
            return 0.0
        
        try:
            for item in self.ib.accountSummary():
                if item.tag == 'NetLiquidation':
                    return float(item.value)
            return 0.0
        except:
            return 0.0


def get_broker(broker_type: str = 'alpaca', **kwargs):
    """Factory per creare il broker appropriato."""
    if broker_type == 'alpaca':
        return AlpacaBroker(**kwargs)
    elif broker_type == 'ib':
        return IBKRBroker(**kwargs)
    else:
        raise ValueError(f"Broker non supportato: {broker_type}")