"""Bitunix exchange implementation."""

import asyncio
import hashlib
import hmac
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import json

import aiohttp
import pandas as pd

from .base import (
    BaseExchange, Candle, Order, Position, AccountBalance, SymbolInfo,
    OrderSide, OrderType, PositionSide, OrderStatus
)
from utils.logger import get_logger

logger = get_logger("bitunix")


class BitunixExchange(BaseExchange):
    """
    Bitunix exchange client.

    Implements REST API for Bitunix futures trading.
    API Documentation: https://openapidoc.bitunix.com/
    """

    # API endpoints
    MAINNET_URL = "https://fapi.bitunix.com"
    TESTNET_URL = "https://fapi.bitunix.com"  # Update if testnet URL differs

    # Timeframe mapping
    TIMEFRAME_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1H",
        "2h": "2H",
        "4h": "4H",
        "6h": "6H",
        "12h": "12H",
        "1d": "1D",
        "1w": "1W",
    }

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True
    ):
        """
        Initialize Bitunix client.

        Args:
            api_key: API key for authentication
            api_secret: API secret for signing requests
            testnet: Use testnet if True
        """
        super().__init__(api_key, api_secret, testnet)
        self.base_url = self.TESTNET_URL if testnet else self.MAINNET_URL
        self._session: Optional[aiohttp.ClientSession] = None
        self._connected = False

    @property
    def name(self) -> str:
        return "bitunix"

    async def connect(self) -> bool:
        """Establish connection to Bitunix."""
        try:
            self._session = aiohttp.ClientSession()
            self._connected = True

            # Test connection
            await self.get_balance()

            logger.info(f"Connected to Bitunix {'testnet' if self.testnet else 'mainnet'}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Bitunix: {e}")
            if self._session:
                await self._session.close()
            return False

    async def disconnect(self) -> None:
        """Disconnect from Bitunix."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
        logger.info("Disconnected from Bitunix")

    def _generate_signature(self, params: Dict[str, Any], timestamp: int, nonce: str) -> str:
        """
        Generate request signature.

        Signature format: HMAC-SHA256 of sorted params + timestamp + nonce
        """
        # Sort params alphabetically and create query string
        sorted_params = sorted(params.items())
        query_string = '&'.join(f"{k}={v}" for k, v in sorted_params)

        # Create sign string
        sign_string = f"{query_string}&timestamp={timestamp}&nonce={nonce}"

        # Generate HMAC-SHA256 signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            sign_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return signature

    def _get_headers(self, params: Dict[str, Any] = None) -> Dict[str, str]:
        """Generate authenticated headers."""
        timestamp = int(time.time() * 1000)
        nonce = str(uuid.uuid4()).replace('-', '')[:32]

        params = params or {}
        signature = self._generate_signature(params, timestamp, nonce)

        return {
            "api-key": self.api_key,
            "timestamp": str(timestamp),
            "nonce": nonce,
            "sign": signature,
            "Content-Type": "application/json"
        }

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Dict[str, Any] = None,
        signed: bool = True
    ) -> Dict[str, Any]:
        """Make API request."""
        if not self._session:
            raise RuntimeError("Not connected to Bitunix")

        url = f"{self.base_url}{endpoint}"
        params = params or {}

        headers = self._get_headers(params) if signed else {"Content-Type": "application/json"}

        try:
            if method.upper() == "GET":
                async with self._session.get(url, params=params, headers=headers) as response:
                    data = await response.json()
            else:
                async with self._session.post(url, json=params, headers=headers) as response:
                    data = await response.json()

            if data.get('code') != 0:
                raise Exception(f"API error: {data.get('msg', 'Unknown error')}")

            return data.get('data', data)

        except aiohttp.ClientError as e:
            logger.error(f"Request failed: {e}")
            raise

    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Candle]:
        """Fetch OHLCV candles."""
        try:
            interval = self.TIMEFRAME_MAP.get(timeframe.lower(), "4H")

            params = {
                "symbol": self.format_symbol(symbol),
                "interval": interval,
                "limit": min(limit, 1000)  # API limit
            }

            if start_time:
                params["startTime"] = int(start_time.timestamp() * 1000)
            if end_time:
                params["endTime"] = int(end_time.timestamp() * 1000)

            data = await self._request("GET", "/api/v1/futures/market/kline", params, signed=False)

            candles = []
            for c in data:
                candle = Candle(
                    timestamp=datetime.fromtimestamp(c[0] / 1000),
                    open=float(c[1]),
                    high=float(c[2]),
                    low=float(c[3]),
                    close=float(c[4]),
                    volume=float(c[5])
                )
                candles.append(candle)

            candles.sort(key=lambda x: x.timestamp)
            return candles[-limit:]

        except Exception as e:
            logger.error(f"Failed to fetch candles for {symbol}: {e}")
            raise

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker data."""
        try:
            params = {"symbol": self.format_symbol(symbol)}
            data = await self._request("GET", "/api/v1/futures/market/ticker", params, signed=False)

            return {
                "symbol": symbol,
                "price": float(data.get('lastPrice', 0)),
                "bid": float(data.get('bidPrice', 0)),
                "ask": float(data.get('askPrice', 0)),
                "volume": float(data.get('volume', 0)),
                "high": float(data.get('highPrice', 0)),
                "low": float(data.get('lowPrice', 0)),
            }

        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            raise

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """Get trading symbol information."""
        try:
            data = await self._request("GET", "/api/v1/futures/market/tradingPairs", signed=False)

            formatted_symbol = self.format_symbol(symbol)

            for pair in data:
                if pair.get('symbol') == formatted_symbol:
                    return SymbolInfo(
                        symbol=symbol,
                        base_asset=pair.get('baseCurrency', symbol),
                        quote_asset=pair.get('quoteCurrency', 'USDT'),
                        tick_size=float(pair.get('tickSize', 0.01)),
                        lot_size=float(pair.get('stepSize', 0.001)),
                        min_size=float(pair.get('minQty', 0.001)),
                        max_size=float(pair.get('maxQty', 100000)),
                        max_leverage=float(pair.get('maxLeverage', 100))
                    )

            raise ValueError(f"Symbol {symbol} not found")

        except Exception as e:
            logger.error(f"Failed to get symbol info for {symbol}: {e}")
            raise

    async def get_balance(self) -> AccountBalance:
        """Get account balance."""
        try:
            data = await self._request("GET", "/api/v1/futures/account/balance")

            return AccountBalance(
                total_balance=float(data.get('totalWalletBalance', 0)),
                available_balance=float(data.get('availableBalance', 0)),
                used_margin=float(data.get('totalMarginBalance', 0)) - float(data.get('availableBalance', 0)),
                unrealized_pnl=float(data.get('unrealizedProfit', 0)),
                currency="USDT"
            )

        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            raise

    async def get_positions(self) -> List[Position]:
        """Get all open positions."""
        try:
            data = await self._request("GET", "/api/v1/futures/account/positions")

            positions = []
            for pos in data:
                size = float(pos.get('positionAmt', 0))
                if size == 0:
                    continue

                side = PositionSide.LONG if size > 0 else PositionSide.SHORT

                positions.append(Position(
                    symbol=pos.get('symbol', '').replace('USDT', ''),
                    side=side,
                    size=abs(size),
                    entry_price=float(pos.get('entryPrice', 0)),
                    mark_price=float(pos.get('markPrice', 0)),
                    liquidation_price=float(pos.get('liquidationPrice', 0)) if pos.get('liquidationPrice') else None,
                    unrealized_pnl=float(pos.get('unrealizedProfit', 0)),
                    realized_pnl=0.0,
                    leverage=float(pos.get('leverage', 1)),
                    margin=float(pos.get('isolatedMargin', 0))
                ))

            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        positions = await self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

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
        """Place a new order."""
        try:
            params = {
                "symbol": self.format_symbol(symbol),
                "side": "BUY" if side == OrderSide.BUY else "SELL",
                "type": "MARKET" if order_type == OrderType.MARKET else "LIMIT",
                "quantity": str(size),
                "reduceOnly": reduce_only
            }

            if order_type == OrderType.LIMIT and price:
                params["price"] = str(price)
                params["timeInForce"] = "GTC"

            if client_order_id:
                params["newClientOrderId"] = client_order_id

            data = await self._request("POST", "/api/v1/futures/trade/placeOrder", params)

            order = Order(
                order_id=str(data.get('orderId', '')),
                symbol=symbol,
                side=side,
                order_type=order_type,
                size=size,
                price=price,
                status=OrderStatus.FILLED if order_type == OrderType.MARKET else OrderStatus.OPEN,
                filled_size=float(data.get('executedQty', 0)),
                avg_fill_price=float(data.get('avgPrice', price or 0)),
                created_at=datetime.utcnow(),
                stop_loss=stop_loss,
                take_profit=take_profit,
                reduce_only=reduce_only,
                client_order_id=client_order_id
            )

            # Place TP/SL orders if specified
            if stop_loss:
                await self._place_stop_loss(symbol, size, stop_loss, side == OrderSide.BUY)
            if take_profit:
                await self._place_take_profit(symbol, size, take_profit, side == OrderSide.BUY)

            return order

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise

    async def _place_stop_loss(
        self,
        symbol: str,
        size: float,
        stop_price: float,
        is_long: bool
    ) -> None:
        """Place a stop loss order."""
        try:
            params = {
                "symbol": self.format_symbol(symbol),
                "side": "SELL" if is_long else "BUY",
                "type": "STOP_MARKET",
                "quantity": str(size),
                "stopPrice": str(stop_price),
                "reduceOnly": True
            }
            await self._request("POST", "/api/v1/futures/trade/placeOrder", params)
        except Exception as e:
            logger.warning(f"Failed to place stop loss: {e}")

    async def _place_take_profit(
        self,
        symbol: str,
        size: float,
        tp_price: float,
        is_long: bool
    ) -> None:
        """Place a take profit order."""
        try:
            params = {
                "symbol": self.format_symbol(symbol),
                "side": "SELL" if is_long else "BUY",
                "type": "TAKE_PROFIT_MARKET",
                "quantity": str(size),
                "stopPrice": str(tp_price),
                "reduceOnly": True
            }
            await self._request("POST", "/api/v1/futures/trade/placeOrder", params)
        except Exception as e:
            logger.warning(f"Failed to place take profit: {e}")

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order."""
        try:
            params = {
                "symbol": self.format_symbol(symbol),
                "orderId": order_id
            }
            await self._request("POST", "/api/v1/futures/trade/cancelOrder", params)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def get_order(self, order_id: str, symbol: str) -> Optional[Order]:
        """Get order details."""
        try:
            params = {
                "symbol": self.format_symbol(symbol),
                "orderId": order_id
            }
            data = await self._request("GET", "/api/v1/futures/trade/orderDetail", params)

            if not data:
                return None

            status_map = {
                "NEW": OrderStatus.OPEN,
                "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
                "FILLED": OrderStatus.FILLED,
                "CANCELED": OrderStatus.CANCELLED,
                "REJECTED": OrderStatus.REJECTED,
            }

            return Order(
                order_id=order_id,
                symbol=symbol,
                side=OrderSide.BUY if data.get('side') == 'BUY' else OrderSide.SELL,
                order_type=OrderType.MARKET if data.get('type') == 'MARKET' else OrderType.LIMIT,
                size=float(data.get('origQty', 0)),
                price=float(data.get('price', 0)) if data.get('price') else None,
                status=status_map.get(data.get('status'), OrderStatus.PENDING),
                filled_size=float(data.get('executedQty', 0)),
                avg_fill_price=float(data.get('avgPrice', 0)) if data.get('avgPrice') else None,
            )

        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders."""
        try:
            params = {}
            if symbol:
                params["symbol"] = self.format_symbol(symbol)

            data = await self._request("GET", "/api/v1/futures/trade/pendingOrders", params)

            orders = []
            for order_data in data:
                orders.append(Order(
                    order_id=str(order_data.get('orderId', '')),
                    symbol=order_data.get('symbol', '').replace('USDT', ''),
                    side=OrderSide.BUY if order_data.get('side') == 'BUY' else OrderSide.SELL,
                    order_type=OrderType.MARKET if order_data.get('type') == 'MARKET' else OrderType.LIMIT,
                    size=float(order_data.get('origQty', 0)),
                    price=float(order_data.get('price', 0)) if order_data.get('price') else None,
                    status=OrderStatus.OPEN,
                ))

            return orders

        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            raise

    async def close_position(
        self,
        symbol: str,
        size: Optional[float] = None
    ) -> Order:
        """Close a position."""
        position = await self.get_position(symbol)

        if not position:
            raise ValueError(f"No position found for {symbol}")

        close_size = size or position.size
        close_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY

        return await self.place_order(
            symbol=symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            size=close_size,
            reduce_only=True
        )

    async def set_leverage(self, symbol: str, leverage: float) -> bool:
        """Set leverage for a symbol."""
        try:
            params = {
                "symbol": self.format_symbol(symbol),
                "leverage": int(leverage)
            }
            await self._request("POST", "/api/v1/futures/account/setLeverage", params)
            return True
        except Exception as e:
            logger.error(f"Failed to set leverage for {symbol}: {e}")
            return False

    def format_symbol(self, asset: str) -> str:
        """Format asset to Bitunix symbol."""
        # Bitunix uses BTCUSDT format
        asset = asset.upper()
        if not asset.endswith('USDT'):
            return f"{asset}USDT"
        return asset
