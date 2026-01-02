"""Hyperliquid exchange implementation."""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import json

import pandas as pd

from .base import (
    BaseExchange, Candle, Order, Position, AccountBalance, SymbolInfo,
    OrderSide, OrderType, PositionSide, OrderStatus
)
from utils.logger import get_logger

logger = get_logger("hyperliquid")


class HyperliquidExchange(BaseExchange):
    """
    Hyperliquid exchange client.

    Uses the official hyperliquid-python-sdk for trading operations.
    """

    # Timeframe mapping
    TIMEFRAME_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "12h": "12h",
        "1d": "1d",
        "1w": "1w",
    }

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        wallet_address: Optional[str] = None,
        account_address: Optional[str] = None,
        testnet: bool = True
    ):
        """
        Initialize Hyperliquid client.

        Hyperliquid uses a two-wallet architecture:
        - API Wallet: Generated at https://app.hyperliquid.xyz/API
          Used for signing transactions (private key required)
        - Main Wallet: Your main account that holds the funds
          Used for querying balance/positions

        Args:
            api_key: API key (not used for Hyperliquid, kept for compatibility)
            api_secret: Private key for the API wallet (for signing)
            wallet_address: API wallet address (derived from private key)
            account_address: Main wallet address where funds are held
                            If not provided, falls back to wallet_address
            testnet: Use testnet if True
        """
        super().__init__(api_key, api_secret, testnet)
        self.wallet_address = wallet_address
        self.account_address = account_address  # Main wallet for queries
        self._exchange = None
        self._info = None
        self._connected = False
        self._signing_address = None  # Will be set from private key

        # Cache for szDecimals per asset (prevents repeated API calls)
        # CRITICAL: Different assets have different precision requirements
        self._sz_decimals_cache: Dict[str, int] = {}

    @property
    def name(self) -> str:
        return "hyperliquid"

    async def connect(self) -> bool:
        """Establish connection to Hyperliquid."""
        try:
            from hyperliquid.info import Info
            from hyperliquid.exchange import Exchange
            from hyperliquid.utils import constants

            # Select API URL based on testnet setting
            base_url = constants.TESTNET_API_URL if self.testnet else constants.MAINNET_API_URL

            # Initialize Info client for reading data
            self._info = Info(base_url, skip_ws=True)

            # Initialize Exchange client for trading
            if self.api_secret:
                from eth_account import Account
                # Convert private key to LocalAccount object
                wallet = Account.from_key(self.api_secret)

                # Store the signing address (derived from private key)
                self._signing_address = wallet.address

                # For Hyperliquid with API wallet trading on behalf of main account:
                # - wallet: The API wallet that signs transactions
                # - account_address: The main account to trade on behalf of
                # If using main wallet directly, account_address can be None
                target_account = self.account_address if self.account_address else None

                self._exchange = Exchange(
                    wallet,
                    base_url,
                    account_address=target_account
                )

                logger.info(f"Signing Wallet: {self._signing_address}")
                if self.account_address:
                    logger.info(f"Trading on behalf of: {self.account_address}")
                    logger.info("NOTE: API wallet must be authorized at https://app.hyperliquid.xyz/API")

            self._connected = True
            logger.info(f"Connected to Hyperliquid {'testnet' if self.testnet else 'mainnet'}")
            return True

        except ImportError:
            logger.error("hyperliquid-python-sdk not installed. Run: pip install hyperliquid-python-sdk")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Hyperliquid: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Hyperliquid."""
        self._exchange = None
        self._info = None
        self._connected = False
        logger.info("Disconnected from Hyperliquid")

    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Candle]:
        """Fetch OHLCV candles."""
        if not self._info:
            raise RuntimeError("Not connected to Hyperliquid")

        try:
            # Map timeframe
            interval = self.TIMEFRAME_MAP.get(timeframe.lower(), "4h")

            # Calculate time range
            if end_time is None:
                end_time = datetime.utcnow()
            if start_time is None:
                # Estimate start time based on limit and timeframe
                hours_per_candle = self._timeframe_to_hours(timeframe)
                start_time = end_time - timedelta(hours=hours_per_candle * limit)

            # Convert to timestamps
            start_ts = int(start_time.timestamp() * 1000)
            end_ts = int(end_time.timestamp() * 1000)

            # Fetch candles using info API
            candle_data = self._info.candles_snapshot(
                name=symbol,
                interval=interval,
                startTime=start_ts,
                endTime=end_ts
            )

            candles = []
            for c in candle_data:
                candle = Candle(
                    timestamp=datetime.fromtimestamp(c['t'] / 1000),
                    open=float(c['o']),
                    high=float(c['h']),
                    low=float(c['l']),
                    close=float(c['c']),
                    volume=float(c['v'])
                )
                candles.append(candle)

            # Sort by timestamp and limit
            candles.sort(key=lambda x: x.timestamp)
            return candles[-limit:]

        except Exception as e:
            logger.error(f"Failed to fetch candles for {symbol}: {e}")
            raise

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker data."""
        if not self._info:
            raise RuntimeError("Not connected to Hyperliquid")

        try:
            # Get all mids (mid prices)
            all_mids = self._info.all_mids()

            if symbol in all_mids:
                mid_price = float(all_mids[symbol])
                return {
                    "symbol": symbol,
                    "price": mid_price,
                    "bid": mid_price,  # Approximation
                    "ask": mid_price,  # Approximation
                }

            raise ValueError(f"Symbol {symbol} not found")

        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            raise

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """Get trading symbol information."""
        if not self._info:
            raise RuntimeError("Not connected to Hyperliquid")

        try:
            # Get metadata for all assets
            meta = self._info.meta()

            for asset_info in meta.get('universe', []):
                if asset_info.get('name') == symbol:
                    return SymbolInfo(
                        symbol=symbol,
                        base_asset=symbol,
                        quote_asset="USD",
                        tick_size=float(10 ** -asset_info.get('szDecimals', 4)),
                        lot_size=float(10 ** -asset_info.get('szDecimals', 4)),
                        min_size=float(10 ** -asset_info.get('szDecimals', 4)),
                        max_size=1000000.0,  # Hyperliquid doesn't specify max
                        max_leverage=float(asset_info.get('maxLeverage', 50))
                    )

            raise ValueError(f"Symbol {symbol} not found")

        except Exception as e:
            logger.error(f"Failed to get symbol info for {symbol}: {e}")
            raise

    @property
    def _query_address(self) -> str:
        """Get the address to use for querying balance/positions.

        Priority: account_address (main wallet) > wallet_address > signing_address
        """
        if self.account_address:
            return self.account_address
        if self.wallet_address:
            return self.wallet_address
        if self._signing_address:
            return self._signing_address
        raise RuntimeError("No wallet address configured")

    async def get_balance(self) -> AccountBalance:
        """Get account balance."""
        if not self._info:
            raise RuntimeError("Not connected")

        address = self._query_address

        try:
            # Get user state
            user_state = self._info.user_state(address)

            margin_summary = user_state.get('marginSummary', {})

            return AccountBalance(
                total_balance=float(margin_summary.get('accountValue', 0)),
                available_balance=float(user_state.get('withdrawable', 0)),  # Fixed: use withdrawable, not marginUsed
                used_margin=float(margin_summary.get('totalMarginUsed', 0)),
                unrealized_pnl=float(margin_summary.get('totalUnrealizedPnl', 0)),
                currency="USD"
            )

        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            raise

    async def get_positions(self) -> List[Position]:
        """Get all open positions."""
        if not self._info:
            raise RuntimeError("Not connected")

        address = self._query_address

        try:
            user_state = self._info.user_state(address)
            positions = []

            for pos in user_state.get('assetPositions', []):
                position_data = pos.get('position', {})
                size = float(position_data.get('szi', 0))

                if size == 0:
                    continue

                side = PositionSide.LONG if size > 0 else PositionSide.SHORT

                positions.append(Position(
                    symbol=position_data.get('coin', ''),
                    side=side,
                    size=abs(size),
                    entry_price=float(position_data.get('entryPx', 0)),
                    mark_price=float(position_data.get('positionValue', 0)) / abs(size) if size != 0 else 0,
                    liquidation_price=float(position_data.get('liquidationPx', 0)) if position_data.get('liquidationPx') else None,
                    unrealized_pnl=float(position_data.get('unrealizedPnl', 0)),
                    realized_pnl=0.0,  # Not directly available
                    leverage=float(position_data.get('leverage', {}).get('value', 1)),
                    margin=float(position_data.get('marginUsed', 0))
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
        if not self._exchange:
            raise RuntimeError("Exchange client not initialized. Check API credentials.")

        try:
            is_buy = side == OrderSide.BUY

            # For market orders, we need to get current price and add slippage
            if order_type == OrderType.MARKET:
                ticker = await self.get_ticker(symbol)
                # Use slippage for market orders (IOC with price buffer)
                slippage = 0.005  # 0.5% slippage
                if is_buy:
                    price = ticker['price'] * (1 + slippage)
                else:
                    price = ticker['price'] * (1 - slippage)

            # Ensure we have a price
            if price is None:
                raise ValueError("Price is required for limit orders")

            # Round price to avoid tick size errors (5 significant figures)
            price = self._round_price(price)

            # Get correct size decimals for this asset and round accordingly
            sz_decimals = await self._get_sz_decimals(symbol)
            size = self._round_size(size, sz_decimals)
            logger.info(f"Order size rounded to {sz_decimals} decimals: {size}")

            # Check minimum notional value ($10 on Hyperliquid)
            notional = size * price
            MIN_NOTIONAL = 11.0  # $11 minimum to be safe
            if notional < MIN_NOTIONAL:
                raise ValueError(f"Order notional ${notional:.2f} below minimum ${MIN_NOTIONAL}. Size: {size}, Price: {price}")

            # Build order type for Hyperliquid SDK
            # Market orders use IOC (Immediate or Cancel), limit orders use GTC (Good Till Cancel)
            if order_type == OrderType.MARKET:
                hl_order_type = {"limit": {"tif": "Ioc"}}
            else:
                hl_order_type = {"limit": {"tif": "Gtc"}}

            # Place the order using positional arguments
            # Signature: order(name, is_buy, sz, limit_px, order_type, reduce_only, cloid, builder)
            logger.info(f"Placing order: {symbol} {'BUY' if is_buy else 'SELL'} size={size} price={price} notional=${size*price:.2f}")
            result = self._exchange.order(
                symbol,          # name: str
                is_buy,          # is_buy: bool
                size,            # sz: float
                price,           # limit_px: float
                hl_order_type,   # order_type: OrderType dict
                reduce_only      # reduce_only: bool
            )

            if result.get('status') == 'ok':
                order_result = result.get('response', {}).get('data', {}).get('statuses', [{}])[0]

                # Check for errors in the order result
                if 'error' in order_result:
                    raise Exception(f"Order rejected: {order_result['error']}")

                # Extract order ID - can be in 'resting' (limit) or 'filled' (market)
                order_id = ""
                if 'resting' in order_result:
                    order_id = str(order_result['resting'].get('oid', ''))
                elif 'filled' in order_result:
                    order_id = str(order_result['filled'].get('oid', ''))

                # If no order ID, something went wrong
                if not order_id:
                    logger.warning(f"Order response missing order ID: {order_result}")
                    raise Exception(f"Order accepted but no order ID returned: {order_result}")

                # Determine fill status
                is_filled = 'filled' in order_result
                fill_price = float(order_result.get('filled', {}).get('avgPx', price)) if is_filled else price

                # Handle TP/SL orders separately if specified
                if stop_loss:
                    await self._place_stop_loss(symbol, size, stop_loss, is_buy)
                if take_profit:
                    await self._place_take_profit(symbol, size, take_profit, is_buy)

                return Order(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    size=size,
                    price=price,
                    status=OrderStatus.FILLED if is_filled else OrderStatus.OPEN,
                    filled_size=size if is_filled else 0,
                    avg_fill_price=fill_price,
                    created_at=datetime.utcnow(),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reduce_only=reduce_only,
                    client_order_id=client_order_id
                )
            else:
                raise Exception(f"Order failed: {result}")

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
            # Round price to avoid float_to_wire rounding errors
            stop_price = self._round_price(stop_price)
            hl_order_type = {
                "trigger": {
                    "isMarket": True,
                    "triggerPx": stop_price,
                    "tpsl": "sl"
                }
            }
            self._exchange.order(
                symbol,              # name
                not is_long,         # is_buy (opposite side to close)
                size,                # sz
                stop_price,          # limit_px
                hl_order_type,       # order_type
                True                 # reduce_only
            )
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
            # Round price to avoid float_to_wire rounding errors
            tp_price = self._round_price(tp_price)
            hl_order_type = {
                "trigger": {
                    "isMarket": True,
                    "triggerPx": tp_price,
                    "tpsl": "tp"
                }
            }
            self._exchange.order(
                symbol,              # name
                not is_long,         # is_buy
                size,                # sz
                tp_price,            # limit_px
                hl_order_type,       # order_type
                True                 # reduce_only
            )
        except Exception as e:
            logger.warning(f"Failed to place take profit: {e}")

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order."""
        if not self._exchange:
            raise RuntimeError("Exchange client not initialized")

        try:
            # cancel(name: str, oid: int)
            result = self._exchange.cancel(symbol, int(order_id))
            success = result.get('status') == 'ok'
            if success:
                logger.info(f"Order {order_id} cancelled successfully")
            else:
                logger.warning(f"Cancel order returned: {result}")
            return success
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def get_order(self, order_id: str, symbol: str) -> Optional[Order]:
        """Get order details."""
        if not self._info:
            raise RuntimeError("Not connected")

        address = self._query_address

        try:
            # Get order status from user state
            user_state = self._info.user_state(address)

            for order_status in user_state.get('openOrders', []):
                if str(order_status.get('oid')) == order_id:
                    # CRITICAL FIX: filled_size = origSz - sz (original minus remaining)
                    # Previous: sz - origSz was BACKWARDS (gave negative values!)
                    orig_size = float(order_status.get('origSz', 0))
                    remaining_size = float(order_status.get('sz', 0))
                    filled = max(0, orig_size - remaining_size)  # origSz - sz

                    return Order(
                        order_id=order_id,
                        symbol=order_status.get('coin', symbol),
                        side=OrderSide.BUY if order_status.get('side') == 'B' else OrderSide.SELL,
                        order_type=OrderType.LIMIT,
                        size=orig_size,
                        price=float(order_status.get('limitPx', 0)),
                        status=OrderStatus.OPEN if remaining_size > 0 else OrderStatus.FILLED,
                        filled_size=filled,
                    )

            return None

        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders."""
        if not self._info:
            raise RuntimeError("Not connected")

        address = self._query_address

        try:
            user_state = self._info.user_state(address)
            orders = []

            for order_status in user_state.get('openOrders', []):
                if symbol and order_status.get('coin') != symbol:
                    continue

                orders.append(Order(
                    order_id=str(order_status.get('oid', '')),
                    symbol=order_status.get('coin', ''),
                    side=OrderSide.BUY if order_status.get('side') == 'B' else OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    size=float(order_status.get('sz', 0)),
                    price=float(order_status.get('limitPx', 0)),
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
        if not self._exchange:
            raise RuntimeError("Exchange client not initialized")

        try:
            result = self._exchange.update_leverage(
                leverage=int(leverage),
                name=symbol,  # SDK uses 'name' not 'coin'
                is_cross=True
            )
            return result.get('status') == 'ok'
        except Exception as e:
            logger.error(f"Failed to set leverage for {symbol}: {e}")
            return False

    def format_symbol(self, asset: str) -> str:
        """Format asset to Hyperliquid symbol."""
        # Hyperliquid uses just the asset name (e.g., "BTC", "ETH")
        return asset.upper()

    @staticmethod
    def _timeframe_to_hours(timeframe: str) -> float:
        """Convert timeframe string to hours."""
        mapping = {
            "1m": 1/60, "5m": 5/60, "15m": 15/60, "30m": 30/60,
            "1h": 1, "2h": 2, "4h": 4, "6h": 6, "12h": 12,
            "1d": 24, "1w": 168
        }
        return mapping.get(timeframe.lower(), 4)

    @staticmethod
    def _round_price(price: float, significant_figures: int = 5) -> float:
        """
        Round price to significant figures to avoid Hyperliquid tick size errors.

        Hyperliquid requires prices to be divisible by tick size.
        Using 5 significant figures is generally safe.
        """
        if price == 0:
            return 0.0
        import math
        magnitude = math.floor(math.log10(abs(price)))
        factor = 10 ** (significant_figures - 1 - magnitude)
        return round(price * factor) / factor

    @staticmethod
    def _round_size(size: float, sz_decimals: int = 5) -> float:
        """Round size to the asset's decimal precision."""
        factor = 10 ** sz_decimals
        return round(size * factor) / factor

    async def _get_sz_decimals(self, symbol: str) -> int:
        """Get the size decimal precision for an asset.

        Different assets have different precision requirements:
        - BTC: 5 decimals (0.00001)
        - ETH: 4 decimals (0.0001)
        - SOL: 2 decimals (0.01)

        Uses caching to avoid repeated API calls (CRITICAL for order placement).
        """
        # Check cache first (CRITICAL: prevents repeated metadata fetches)
        if symbol in self._sz_decimals_cache:
            return self._sz_decimals_cache[symbol]

        if not self._info:
            return 5  # Default fallback

        try:
            meta = self._info.meta()
            # Cache ALL assets at once to minimize future calls
            for asset_info in meta.get('universe', []):
                asset_name = asset_info.get('name')
                decimals = asset_info.get('szDecimals', 5)
                if asset_name:
                    self._sz_decimals_cache[asset_name] = decimals

            # Log what we found
            if symbol in self._sz_decimals_cache:
                logger.info(f"Cached szDecimals: {symbol}={self._sz_decimals_cache[symbol]}")
                return self._sz_decimals_cache[symbol]

            logger.warning(f"{symbol} not found in metadata, using default 5")
            return 5  # Default if not found
        except Exception as e:
            logger.warning(f"Failed to get szDecimals for {symbol}: {e}")
            return 5
