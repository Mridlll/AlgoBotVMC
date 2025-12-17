"""Abstract base class for exchange implementations."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any


class OrderSide(str, Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"


class PositionSide(str, Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"


class OrderStatus(str, Enum):
    """Order status."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Candle:
    """OHLCV candle data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass
class Order:
    """Order information."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    size: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    avg_fill_price: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reduce_only: bool = False
    client_order_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "size": self.size,
            "price": self.price,
            "status": self.status.value,
            "filled_size": self.filled_size,
            "avg_fill_price": self.avg_fill_price,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
        }


@dataclass
class Position:
    """Position information."""
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    mark_price: float
    liquidation_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    leverage: float = 1.0
    margin: float = 0.0
    created_at: Optional[datetime] = None

    @property
    def notional_value(self) -> float:
        """Calculate notional value of position."""
        return self.size * self.mark_price

    @property
    def pnl_percent(self) -> float:
        """Calculate PnL percentage."""
        if self.entry_price == 0:
            return 0.0
        if self.side == PositionSide.LONG:
            return ((self.mark_price - self.entry_price) / self.entry_price) * 100 * self.leverage
        else:
            return ((self.entry_price - self.mark_price) / self.entry_price) * 100 * self.leverage

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "size": self.size,
            "entry_price": self.entry_price,
            "mark_price": self.mark_price,
            "liquidation_price": self.liquidation_price,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "leverage": self.leverage,
            "pnl_percent": self.pnl_percent,
        }


@dataclass
class AccountBalance:
    """Account balance information."""
    total_balance: float
    available_balance: float
    used_margin: float
    unrealized_pnl: float = 0.0
    currency: str = "USD"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_balance": self.total_balance,
            "available_balance": self.available_balance,
            "used_margin": self.used_margin,
            "unrealized_pnl": self.unrealized_pnl,
            "currency": self.currency,
        }


@dataclass
class SymbolInfo:
    """Trading symbol information."""
    symbol: str
    base_asset: str
    quote_asset: str
    tick_size: float
    lot_size: float
    min_size: float
    max_size: float
    max_leverage: float = 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "base_asset": self.base_asset,
            "quote_asset": self.quote_asset,
            "tick_size": self.tick_size,
            "lot_size": self.lot_size,
            "min_size": self.min_size,
            "max_size": self.max_size,
            "max_leverage": self.max_leverage,
        }


class BaseExchange(ABC):
    """Abstract base class for exchange implementations."""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """
        Initialize exchange client.

        Args:
            api_key: API key for authentication
            api_secret: API secret for signing requests
            testnet: Whether to use testnet
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

    @property
    @abstractmethod
    def name(self) -> str:
        """Exchange name."""
        pass

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the exchange.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the exchange."""
        pass

    # Market Data Methods

    @abstractmethod
    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Candle]:
        """
        Fetch OHLCV candles for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., "BTC")
            timeframe: Candle timeframe (e.g., "4h")
            limit: Number of candles to fetch
            start_time: Start time for historical data
            end_time: End time for historical data

        Returns:
            List of Candle objects
        """
        pass

    async def get_candles_multiple(
        self,
        symbol: str,
        timeframes: List[str],
        limit: int = 100
    ) -> Dict[str, List[Candle]]:
        """
        Fetch OHLCV candles for multiple timeframes in parallel.

        Handles custom timeframes by aggregating from smaller intervals:
        - 3m: aggregated from 1m candles
        - 10m: aggregated from 5m candles
        - 8h: aggregated from 4h candles

        Args:
            symbol: Trading pair symbol (e.g., "BTC")
            timeframes: List of timeframes (e.g., ["5m", "15m", "4h", "1d"])
            limit: Number of candles to fetch per timeframe

        Returns:
            Dictionary mapping timeframe to list of Candle objects
        """
        # Mapping for custom timeframes to base timeframe and aggregation factor
        custom_tf_map = {
            "3m": ("1m", 3),
            "10m": ("5m", 2),
            "8h": ("4h", 2),
        }

        results: Dict[str, List[Candle]] = {}
        tasks = []
        tf_mapping = []  # Track which task corresponds to which target timeframe

        for tf in timeframes:
            if tf in custom_tf_map:
                base_tf, factor = custom_tf_map[tf]
                # Fetch more candles to aggregate
                tasks.append(self.get_candles(symbol, base_tf, limit=limit * factor + factor))
                tf_mapping.append((tf, base_tf, factor))
            else:
                tasks.append(self.get_candles(symbol, tf, limit=limit))
                tf_mapping.append((tf, tf, 1))

        # Fetch all timeframes in parallel
        fetched = await asyncio.gather(*tasks, return_exceptions=True)

        for i, candles in enumerate(fetched):
            target_tf, base_tf, factor = tf_mapping[i]

            if isinstance(candles, Exception):
                # Log error but continue with other timeframes
                results[target_tf] = []
                continue

            if factor > 1:
                # Aggregate candles for custom timeframes
                results[target_tf] = self._aggregate_candles(candles, factor)[:limit]
            else:
                results[target_tf] = candles

        return results

    def _aggregate_candles(self, candles: List[Candle], factor: int) -> List[Candle]:
        """
        Aggregate smaller timeframe candles into larger ones.

        Args:
            candles: List of candles to aggregate
            factor: Number of candles to combine

        Returns:
            List of aggregated candles
        """
        if not candles or factor <= 1:
            return candles

        aggregated = []
        # Process from oldest to newest
        sorted_candles = sorted(candles, key=lambda c: c.timestamp)

        for i in range(0, len(sorted_candles) - factor + 1, factor):
            group = sorted_candles[i:i + factor]
            if len(group) < factor:
                break

            agg_candle = Candle(
                timestamp=group[0].timestamp,
                open=group[0].open,
                high=max(c.high for c in group),
                low=min(c.low for c in group),
                close=group[-1].close,
                volume=sum(c.volume for c in group)
            )
            aggregated.append(agg_candle)

        return aggregated

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker data for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Ticker data including price, volume, etc.
        """
        pass

    @abstractmethod
    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """
        Get trading symbol information.

        Args:
            symbol: Trading pair symbol

        Returns:
            SymbolInfo object
        """
        pass

    # Account Methods

    @abstractmethod
    async def get_balance(self) -> AccountBalance:
        """
        Get account balance.

        Returns:
            AccountBalance object
        """
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """
        Get all open positions.

        Returns:
            List of Position objects
        """
        pass

    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Position object or None if no position
        """
        pass

    # Order Methods

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        size: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        reduce_only: bool = False,
        client_order_id: Optional[str] = None
    ) -> Order:
        """
        Place a new order.

        Args:
            symbol: Trading pair symbol
            side: Buy or sell
            order_type: Market or limit
            size: Order size
            price: Limit price (required for limit orders)
            stop_loss: Stop loss price
            take_profit: Take profit price
            reduce_only: Only reduce position
            client_order_id: Custom order ID

        Returns:
            Order object
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Order ID to cancel
            symbol: Trading pair symbol

        Returns:
            True if cancelled successfully
        """
        pass

    @abstractmethod
    async def get_order(self, order_id: str, symbol: str) -> Optional[Order]:
        """
        Get order details.

        Args:
            order_id: Order ID
            symbol: Trading pair symbol

        Returns:
            Order object or None
        """
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get all open orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of Order objects
        """
        pass

    # Position Management

    @abstractmethod
    async def close_position(
        self,
        symbol: str,
        size: Optional[float] = None
    ) -> Order:
        """
        Close a position.

        Args:
            symbol: Trading pair symbol
            size: Size to close (None = close entire position)

        Returns:
            Order object for the closing trade
        """
        pass

    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: float) -> bool:
        """
        Set leverage for a symbol.

        Args:
            symbol: Trading pair symbol
            leverage: Leverage value

        Returns:
            True if successful
        """
        pass

    # Utility Methods

    def format_symbol(self, asset: str) -> str:
        """
        Format asset name to exchange symbol format.

        Args:
            asset: Asset name (e.g., "BTC")

        Returns:
            Formatted symbol for the exchange
        """
        return asset

    async def is_connected(self) -> bool:
        """
        Check if connected to exchange.

        Returns:
            True if connected
        """
        try:
            await self.get_balance()
            return True
        except Exception:
            return False
