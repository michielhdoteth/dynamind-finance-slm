"""
Limit Order Book (LOB) Simulator

Realistic limit order book simulation for market microstructure modeling.
Supports order placement, execution, and realistic market dynamics.
"""

import numpy as np
import heapq
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types"""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides"""

    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Order representation"""

    order_id: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    trader_id: str = ""
    time_in_force: str = "GTC"  # GTC, IOC, FOK, DAY

    def __post_init__(self):
        if self.side == OrderSide.BUY:
            self.priority = (
                -self.price if self.price else 0
            )  # Higher price = higher priority
        else:
            self.priority = (
                self.price if self.price else float("in")
            )  # Lower price = higher priority


@dataclass
class Trade:
    """Trade representation"""

    trade_id: str
    buy_order_id: str
    sell_order_id: str
    price: float
    quantity: int
    timestamp: float
    buyer_id: str
    seller_id: str


class LimitOrderBook:
    """
    Realistic Limit Order Book simulator.

    Features:
    - Price-time priority for limit orders
    - Market order execution with realistic slippage
    - Order cancellation and modification
    - Market impact modeling
    - Realistic spread dynamics
    """

    def __init__(
        self,
        tick_size: float = 0.01,
        min_quantity: int = 1,
        max_quantity: int = 10000,
        market_impact_factor: float = 0.001,
        base_spread: float = 0.02,
    ):
        """
        Initialize LOB simulator.

        Args:
            tick_size: Minimum price increment
            min_quantity: Minimum order quantity
            max_quantity: Maximum order quantity
            market_impact_factor: Market impact coefficient
            base_spread: Base bid-ask spread
        """
        self.tick_size = tick_size
        self.min_quantity = min_quantity
        self.max_quantity = max_quantity
        self.market_impact_factor = market_impact_factor
        self.base_spread = base_spread

        # Order books
        self.bid_book = []  # Max heap for buy orders (negative prices for max heap)
        self.ask_book = []  # Min heap for sell orders
        self.orders = {}  # Order ID to Order mapping

        # Market state
        self.mid_price = 100.0
        self.last_trade_price = 100.0
        self.current_spread = base_spread

        # Trade history
        self.trades = deque(maxlen=1000)
        self.order_counter = 0

        # Statistics
        self.total_volume = 0
        self.total_trades = 0

        logger.info(
            f"LOB Simulator initialized with tick_size={tick_size}, base_spread={base_spread}"
        )

    def get_best_bid(self) -> Optional[float]:
        """Get best bid price."""
        if not self.bid_book:
            return None
        return -self.bid_book[0][0]  # Negate back to positive

    def get_best_ask(self) -> Optional[float]:
        """Get best ask price."""
        if not self.ask_book:
            return None
        return self.ask_book[0][0]

    def get_mid_price(self) -> float:
        """Get mid price."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid is None and best_ask is None:
            return self.mid_price
        elif best_bid is None:
            self.mid_price = best_ask
            return best_ask
        elif best_ask is None:
            self.mid_price = best_bid
            return best_bid
        else:
            self.mid_price = (best_bid + best_ask) / 2
            return self.mid_price

    def get_spread(self) -> float:
        """Get current bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid is None or best_ask is None:
            return self.current_spread
        else:
            self.current_spread = best_ask - best_bid
            return self.current_spread

    def get_market_depth(self, levels: int = 5) -> Dict[str, List[Tuple[float, int]]]:
        """
        Get market depth at multiple price levels.

        Args:
            levels: Number of price levels to return

        Returns:
            Dictionary with bid and ask depth
        """
        bid_depth = []
        ask_depth = []

        # Get bid levels
        temp_bid_book = [
            (price, qty, order_id) for price, qty, order_id in self.bid_book
        ]
        bid_prices = {}
        for price, qty, order_id in temp_bid_book:
            if -price not in bid_prices:
                bid_prices[-price] = 0
            bid_prices[-price] += qty

        for price in sorted(bid_prices.keys(), reverse=True)[:levels]:
            bid_depth.append((price, bid_prices[price]))

        # Get ask levels
        temp_ask_book = list(self.ask_book)
        ask_prices = {}
        for price, qty, order_id in temp_ask_book:
            if price not in ask_prices:
                ask_prices[price] = 0
            ask_prices[price] += qty

        for price in sorted(ask_prices.keys())[:levels]:
            ask_depth.append((price, ask_prices[price]))

        return {"bids": bid_depth, "asks": ask_depth}

    def place_order(self, order: Order) -> List[Trade]:
        """
        Place an order in the LOB.

        Args:
            order: Order to place

        Returns:
            List of executed trades
        """
        if not self._validate_order(order):
            return []

        self.order_counter += 1
        if not order.order_id:
            order.order_id = f"order_{self.order_counter}"

        self.orders[order.order_id] = order

        trades = []

        if order.order_type == OrderType.MARKET:
            trades = self._execute_market_order(order)
        elif order.order_type == OrderType.LIMIT:
            trades = self._execute_limit_order(order)
        elif order.order_type == OrderType.STOP:
            # Store stop order for later execution
            self._handle_stop_order(order)
        elif order.order_type == OrderType.STOP_LIMIT:
            # Store stop-limit order for later execution
            self._handle_stop_limit_order(order)

        # Update market statistics
        self._update_market_stats(trades)

        return trades

    def _validate_order(self, order: Order) -> bool:
        """Validate order parameters."""
        if order.quantity < self.min_quantity or order.quantity > self.max_quantity:
            return False

        if order.order_type == OrderType.LIMIT and order.price is None:
            return False

        if (
            order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]
            and order.stop_price is None
        ):
            return False

        return True

    def _execute_market_order(self, order: Order) -> List[Trade]:
        """Execute a market order with realistic slippage."""
        trades = []
        remaining_quantity = order.quantity

        if order.side == OrderSide.BUY:
            # Market buy: consume from ask book
            while remaining_quantity > 0 and self.ask_book:
                price, available_qty, order_id = heapq.heappop(self.ask_book)

                # Calculate execution quantity
                exec_qty = min(remaining_quantity, available_qty)

                # Apply market impact
                impacted_price = self._apply_market_impact(
                    price, exec_qty, OrderSide.BUY
                )

                # Create trade
                trade = Trade(
                    trade_id=f"trade_{self.total_trades + 1}",
                    buy_order_id=order.order_id,
                    sell_order_id=order_id,
                    price=impacted_price,
                    quantity=exec_qty,
                    timestamp=time.time(),
                    buyer_id=order.trader_id,
                    seller_id=self.orders[order_id].trader_id
                    if order_id in self.orders
                    else "",
                )
                trades.append(trade)

                # Update quantities
                remaining_quantity -= exec_qty
                if available_qty > exec_qty:
                    # Put back remaining quantity
                    heapq.heappush(
                        self.ask_book, (price, available_qty - exec_qty, order_id)
                    )
                else:
                    # Remove fully executed order
                    if order_id in self.orders:
                        del self.orders[order_id]

        else:  # SELL
            # Market sell: consume from bid book
            while remaining_quantity > 0 and self.bid_book:
                price, available_qty, order_id = heapq.heappop(self.bid_book)

                # Calculate execution quantity
                exec_qty = min(remaining_quantity, available_qty)

                # Apply market impact
                impacted_price = self._apply_market_impact(
                    -price, exec_qty, OrderSide.SELL
                )

                # Create trade
                trade = Trade(
                    trade_id=f"trade_{self.total_trades + 1}",
                    buy_order_id=order_id,
                    sell_order_id=order.order_id,
                    price=impacted_price,
                    quantity=exec_qty,
                    timestamp=time.time(),
                    buyer_id=self.orders[order_id].trader_id
                    if order_id in self.orders
                    else "",
                    seller_id=order.trader_id,
                )
                trades.append(trade)

                # Update quantities
                remaining_quantity -= exec_qty
                if available_qty > exec_qty:
                    # Put back remaining quantity
                    heapq.heappush(
                        self.bid_book, (price, available_qty - exec_qty, order_id)
                    )
                else:
                    # Remove fully executed order
                    if order_id in self.orders:
                        del self.orders[order_id]

        return trades

    def _execute_limit_order(self, order: Order) -> List[Trade]:
        """Execute a limit order."""
        trades = []
        remaining_quantity = order.quantity

        if order.side == OrderSide.BUY:
            # Check if order can be executed immediately
            while (
                remaining_quantity > 0
                and self.ask_book
                and order.price >= self.ask_book[0][0]
            ):
                price, available_qty, order_id = heapq.heappop(self.ask_book)
                exec_qty = min(remaining_quantity, available_qty)

                # Create trade at limit price
                trade = Trade(
                    trade_id=f"trade_{self.total_trades + 1}",
                    buy_order_id=order.order_id,
                    sell_order_id=order_id,
                    price=min(order.price, price),
                    quantity=exec_qty,
                    timestamp=time.time(),
                    buyer_id=order.trader_id,
                    seller_id=self.orders[order_id].trader_id
                    if order_id in self.orders
                    else "",
                )
                trades.append(trade)

                # Update quantities
                remaining_quantity -= exec_qty
                if available_qty > exec_qty:
                    heapq.heappush(
                        self.ask_book, (price, available_qty - exec_qty, order_id)
                    )
                else:
                    if order_id in self.orders:
                        del self.orders[order_id]

            # Add remaining quantity to bid book
            if remaining_quantity > 0:
                heapq.heappush(
                    self.bid_book, (-order.price, remaining_quantity, order.order_id)
                )
                order.quantity = remaining_quantity

        else:  # SELL
            # Check if order can be executed immediately
            while (
                remaining_quantity > 0
                and self.bid_book
                and -self.bid_book[0][0] >= order.price
            ):
                price, available_qty, order_id = heapq.heappop(self.bid_book)
                exec_qty = min(remaining_quantity, available_qty)

                # Create trade at limit price
                trade = Trade(
                    trade_id=f"trade_{self.total_trades + 1}",
                    buy_order_id=order_id,
                    sell_order_id=order.order_id,
                    price=max(order.price, -price),
                    quantity=exec_qty,
                    timestamp=time.time(),
                    buyer_id=self.orders[order_id].trader_id
                    if order_id in self.orders
                    else "",
                    seller_id=order.trader_id,
                )
                trades.append(trade)

                # Update quantities
                remaining_quantity -= exec_qty
                if available_qty > exec_qty:
                    heapq.heappush(
                        self.bid_book, (price, available_qty - exec_qty, order_id)
                    )
                else:
                    if order_id in self.orders:
                        del self.orders[order_id]

            # Add remaining quantity to ask book
            if remaining_quantity > 0:
                heapq.heappush(
                    self.ask_book, (order.price, remaining_quantity, order.order_id)
                )
                order.quantity = remaining_quantity

        return trades

    def _apply_market_impact(
        self, price: float, quantity: int, side: OrderSide
    ) -> float:
        """Apply realistic market impact to execution price."""
        # Simple linear market impact model
        impact = self.market_impact_factor * np.sqrt(quantity)

        if side == OrderSide.BUY:
            # Buy orders push price up
            return price * (1 + impact)
        else:
            # Sell orders push price down
            return price * (1 - impact)

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.

        Args:
            order_id: ID of order to cancel

        Returns:
            True if order was cancelled, False if not found
        """
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]

        # Remove from appropriate order book
        if order.side == OrderSide.BUY:
            # Search bid book
            for i, (price, qty, oid) in enumerate(self.bid_book):
                if oid == order_id:
                    del self.bid_book[i]
                    heapq.heapify(self.bid_book)
                    break
        else:
            # Search ask book
            for i, (price, qty, oid) in enumerate(self.ask_book):
                if oid == order_id:
                    del self.ask_book[i]
                    heapq.heapify(self.ask_book)
                    break

        del self.orders[order_id]
        return True

    def _handle_stop_order(self, order: Order):
        """Handle stop order (simplified - triggers at market price)."""
        # In a full implementation, this would monitor price movements
        # and trigger when stop price is reached
        pass

    def _handle_stop_limit_order(self, order: Order):
        """Handle stop-limit order (simplified)."""
        # In a full implementation, this would monitor price movements
        # and place limit order when stop price is reached
        pass

    def _update_market_stats(self, trades: List[Trade]):
        """Update market statistics after trades."""
        for trade in trades:
            self.trades.append(trade)
            self.last_trade_price = trade.price
            self.total_volume += trade.quantity
            self.total_trades += 1

    def get_order_book_state(self) -> Dict[str, Any]:
        """Get current order book state."""
        return {
            "best_bid": self.get_best_bid(),
            "best_ask": self.get_best_ask(),
            "mid_price": self.get_mid_price(),
            "spread": self.get_spread(),
            "bid_volume": sum(qty for _, qty, _ in self.bid_book),
            "ask_volume": sum(qty for _, qty, _ in self.ask_book),
            "total_orders": len(self.orders),
            "last_trade_price": self.last_trade_price,
            "total_volume": self.total_volume,
            "total_trades": self.total_trades,
        }

    def add_liquidity(self, num_orders: int = 10, price_range: float = 0.5):
        """
        Add liquidity to the order book for testing.

        Args:
            num_orders: Number of orders to add on each side
            price_range: Price range around mid price
        """
        mid_price = self.get_mid_price()

        # Add buy orders
        for i in range(num_orders):
            price_offset = np.random.uniform(-price_range, 0)
            price = mid_price + price_offset
            price = round(price / self.tick_size) * self.tick_size

            quantity = np.random.randint(100, 1000)

            order = Order(
                order_id=f"liquidity_bid_{i}",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=price,
                trader_id="liquidity_provider",
            )

            self.place_order(order)

        # Add sell orders
        for i in range(num_orders):
            price_offset = np.random.uniform(0, price_range)
            price = mid_price + price_offset
            price = round(price / self.tick_size) * self.tick_size

            quantity = np.random.randint(100, 1000)

            order = Order(
                order_id=f"liquidity_ask_{i}",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=price,
                trader_id="liquidity_provider",
            )

            self.place_order(order)

        logger.info(f"Added {2 * num_orders} liquidity orders")
