# execution/orders.py
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict
from datetime import datetime
import uuid

class OrderType(Enum):
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'
    STOP_LIMIT = 'stop_limit'

class OrderSide(Enum):
    BUY = 'buy'
    SELL = 'sell'

class OrderStatus(Enum):
    PENDING = 'pending'
    FILLED = 'filled'
    CANCELLED = 'cancelled'
    REJECTED = 'rejected'
    PARTIALLY_FILLED = 'partially_filled'

@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    stop_price: Optional[float] = None
    limit_price: Optional[float] = None
    time_in_force: str = 'day'
    placed_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: Optional[int] = None
    status: OrderStatus = OrderStatus.PENDING
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    client_order_id: Optional[str] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'price': self.price,
            'stop_price': self.stop_price,
            'limit_price': self.limit_price,
            'time_in_force': self.time_in_force,
            'placed_at': self.placed_at.isoformat(),
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'filled_price': self.filled_price,
            'filled_quantity': self.filled_quantity,
            'status': self.status.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Order':
        order = cls(
            symbol=data['symbol'],
            side=OrderSide(data['side']),
            quantity=data['quantity'],
            order_type=OrderType(data['order_type'])
        )
        order.price = data.get('price')
        order.stop_price = data.get('stop_price')
        order.limit_price = data.get('limit_price')
        order.time_in_force = data.get('time_in_force', 'day')
        order.order_id = data.get('order_id', str(uuid.uuid4()))
        order.status = OrderStatus(data.get('status', 'pending'))
        return order


class OrderManager:
    """Gestione degli ordini."""
    
    def __init__(self, max_open_orders: int = 20):
        self.max_open_orders = max_open_orders
        self.open_orders: Dict[str, Order] = {}
        self.closed_orders: List[Order] = []
        self.order_counter = 0
    
    def create_order(self,
                    symbol: str,
                    side: OrderSide,
                    quantity: int,
                    order_type: OrderType = OrderType.MARKET,
                    price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    limit_price: Optional[float] = None,
                    client_order_id: Optional[str] = None) -> Order:
        """Crea un nuovo ordine."""
        if len(self.open_orders) >= self.max_open_orders:
            raise ValueError(f"Massimo {self.max_open_orders} ordini aperti raggiunto")
        
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            stop_price=stop_price,
            limit_price=limit_price,
            client_order_id=client_order_id
        )
        
        self.order_counter += 1
        self.open_orders[order.order_id] = order
        
        return order
    
    def update_order(self,
                    order_id: str,
                    status: OrderStatus,
                    filled_price: Optional[float] = None,
                    filled_quantity: Optional[int] = None) -> Optional[Order]:
        """Aggiorna lo stato di un ordine."""
        if order_id not in self.open_orders:
            return None
        
        order = self.open_orders[order_id]
        order.status = status
        
        if status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
            if filled_price:
                order.filled_price = filled_price
            if filled_quantity:
                order.filled_quantity = filled_quantity
            order.filled_at = datetime.now()
        
        if status == OrderStatus.FILLED:
            self.closed_orders.append(order)
            del self.open_orders[order_id]
        
        return order
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Restituisce un ordine per ID."""
        if order_id in self.open_orders:
            return self.open_orders[order_id]
        for order in self.closed_orders:
            if order.order_id == order_id:
                return order
        return None
    
    def get_open_orders(self) -> List[Order]:
        """Restituisce tutti gli ordini aperti."""
        return list(self.open_orders.values())
    
    def get_open_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Restituisce gli ordini aperti per un simbolo."""
        return [o for o in self.open_orders.values() if o.symbol == symbol]
    
    def cancel_order(self, order_id: str) -> bool:
        """Annulla un ordine aperto."""
        if order_id in self.open_orders:
            order = self.open_orders[order_id]
            order.status = OrderStatus.CANCELLED
            self.closed_orders.append(order)
            del self.open_orders[order_id]
            return True
        return False
    
    def cancel_all_orders(self) -> int:
        """Annulla tutti gli ordini aperti."""
        count = 0
        for order_id in list(self.open_orders.keys()):
            if self.cancel_order(order_id):
                count += 1
        return count
    
    def get_positions(self) -> Dict[str, int]:
        """Calcola le posizioni attuali dagli ordini aperti e chiusi."""
        positions = {}
        
        # Ordini aperti
        for order in self.open_orders.values():
            if order.status == OrderStatus.FILLED:
                if order.symbol not in positions:
                    positions[order.symbol] = 0
                if order.side == OrderSide.BUY:
                    positions[order.symbol] += order.quantity
                else:
                    positions[order.symbol] -= order.quantity
        
        # Ordini chiusi (ultimi)
        for order in self.closed_orders[-50:]:  # Ultimi 50
            if order.status == OrderStatus.FILLED:
                if order.symbol not in positions:
                    positions[order.symbol] = 0
                if order.side == OrderSide.BUY:
                    positions[order.symbol] += order.quantity
                else:
                    positions[order.symbol] -= order.quantity
        
        return {k: v for k, v in positions.items() if v != 0}
    
    def get_order_history(self, symbol: Optional[str] = None) -> List[Order]:
        """Restituisce la cronologia degli ordini."""
        if symbol:
            return [o for o in self.closed_orders if o.symbol == symbol]
        return self.closed_orders.copy()
    
    def get_summary(self) -> Dict:
        """Restituisce un riassunto degli ordini."""
        return {
            'open_orders': len(self.open_orders),
            'closed_orders': len(self.closed_orders),
            'total_orders': len(self.open_orders) + len(self.closed_orders),
            'filled_orders': sum(1 for o in self.closed_orders if o.status == OrderStatus.FILLED),
            'cancelled_orders': sum(1 for o in self.closed_orders if o.status == OrderStatus.CANCELLED),
            'rejected_orders': sum(1 for o in self.closed_orders if o.status == OrderStatus.REJECTED)
        }