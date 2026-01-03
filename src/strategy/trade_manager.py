"""Trade execution and position management."""

import asyncio
import math
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from exchanges.base import (
    BaseExchange, Order, Position, OrderSide, OrderType, PositionSide, OrderStatus
)
from strategy.signals import Signal, SignalType
from strategy.risk import RiskManager, RiskParams, StopLossMethod, TakeProfitMethod
from utils.logger import get_logger


class OrderNotFilledException(Exception):
    """Raised when an order fails to fill within timeout."""
    pass

logger = get_logger("trade_manager")


class TradeStatus(str, Enum):
    """Trade status."""
    PENDING = "pending"
    OPEN = "open"
    PARTIAL_TP = "partial_tp"
    CLOSED = "closed"
    CANCELLED = "cancelled"


@dataclass
class Trade:
    """Active trade information."""
    trade_id: str
    symbol: str
    signal_type: SignalType
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    status: TradeStatus = TradeStatus.PENDING
    entry_order_id: Optional[str] = None
    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_percent: float = 0.0
    partial_closed_size: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_long(self) -> bool:
        return self.signal_type == SignalType.LONG

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "entry_price": self.entry_price,
            "size": self.size,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "status": self.status.value,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "pnl_percent": self.pnl_percent,
        }


class TradeManager:
    """
    Manages trade execution and active positions.

    Handles:
    - Signal to order conversion
    - Position tracking
    - TP/SL management
    - Partial exits
    - Trailing stops
    """

    def __init__(
        self,
        exchange: BaseExchange,
        risk_manager: RiskManager,
        tp_method: TakeProfitMethod = TakeProfitMethod.FIXED_RR,
        sl_method: StopLossMethod = StopLossMethod.SWING,
        max_positions: int = 2,
        max_positions_per_asset: int = 1,
        max_portfolio_heat_percent: float = 12.0
    ):
        """
        Initialize trade manager.

        Args:
            exchange: Exchange client
            risk_manager: Risk manager instance
            tp_method: Take profit method
            sl_method: Stop loss method
            max_positions: Maximum total positions
            max_positions_per_asset: Maximum positions per asset
            max_portfolio_heat_percent: Maximum total risk across all positions (% of account)
        """
        self.exchange = exchange
        self.risk_manager = risk_manager
        self.tp_method = tp_method
        self.sl_method = sl_method
        self.max_positions = max_positions
        self.max_positions_per_asset = max_positions_per_asset
        self.max_portfolio_heat_percent = max_portfolio_heat_percent

        self._active_trades: Dict[str, Trade] = {}
        self._trade_history: List[Trade] = []
        self._trade_counter = 0

        # CRITICAL: Lock for thread-safe access to _active_trades
        # Prevents race conditions when multiple async operations modify trades
        self._trades_lock = asyncio.Lock()

    @property
    def active_trades(self) -> Dict[str, 'Trade']:
        """Thread-safe access to active trades (read-only snapshot)."""
        return dict(self._active_trades)

    async def add_trade(self, trade: 'Trade') -> None:
        """Thread-safe add trade."""
        async with self._trades_lock:
            self._active_trades[trade.trade_id] = trade

    async def remove_trade(self, trade_id: str) -> Optional['Trade']:
        """Thread-safe remove trade."""
        async with self._trades_lock:
            return self._active_trades.pop(trade_id, None)

    async def get_trade(self, trade_id: str) -> Optional['Trade']:
        """Thread-safe get trade."""
        async with self._trades_lock:
            return self._active_trades.get(trade_id)

    def calculate_trade_heat(self, trade: 'Trade') -> float:
        """
        Calculate the risk (heat) of a single trade in USD.

        Heat = potential loss if stop loss is hit.
        """
        if trade.is_long:
            risk_per_unit = trade.entry_price - trade.stop_loss
        else:
            risk_per_unit = trade.stop_loss - trade.entry_price

        return abs(risk_per_unit * trade.size)

    def get_current_heat(self, account_balance: float) -> Tuple[float, float]:
        """
        Calculate current portfolio heat.

        Returns:
            Tuple of (heat_usd, heat_percent)
        """
        total_heat_usd = 0.0
        for trade in self._active_trades.values():
            total_heat_usd += self.calculate_trade_heat(trade)

        heat_percent = (total_heat_usd / account_balance * 100) if account_balance > 0 else 0.0
        return total_heat_usd, heat_percent

    def get_available_heat(self, account_balance: float) -> Tuple[float, float]:
        """
        Calculate available heat budget.

        Returns:
            Tuple of (available_heat_usd, available_heat_percent)
        """
        _, current_heat_percent = self.get_current_heat(account_balance)
        available_percent = max(0, self.max_portfolio_heat_percent - current_heat_percent)
        available_usd = account_balance * (available_percent / 100)
        return available_usd, available_percent

    def _generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        self._trade_counter += 1
        return f"trade_{self._trade_counter}_{int(datetime.utcnow().timestamp())}"

    async def _wait_for_fill(
        self,
        order_id: str,
        symbol: str,
        timeout_seconds: int = 30,
        poll_interval: float = 1.0,
        expected_size: float = 0.0,
        is_buy: bool = True
    ) -> Optional[Order]:
        """
        Wait for an order to be filled.

        Args:
            order_id: The order ID to wait for
            symbol: Trading symbol
            timeout_seconds: Maximum time to wait for fill
            poll_interval: Time between status checks
            expected_size: Expected position size change (for fallback detection)
            is_buy: True if buy order, False if sell

        Returns:
            Filled Order object, or None if timeout
        """
        max_attempts = int(timeout_seconds / poll_interval)

        # Get initial position for fallback detection
        initial_position = await self.exchange.get_position(symbol)
        initial_size = initial_position.size if initial_position else 0.0
        logger.debug(f"Initial position for {symbol}: {initial_size}")

        for attempt in range(max_attempts):
            try:
                order = await self.exchange.get_order(order_id, symbol)

                if order is None:
                    # Order not in openOrders - may have filled instantly
                    # Use position-based fallback detection
                    if expected_size > 0:
                        current_position = await self.exchange.get_position(symbol)
                        current_size = current_position.size if current_position else 0.0

                        # Calculate expected size change
                        if is_buy:
                            expected_new_size = initial_size + expected_size
                        else:
                            expected_new_size = initial_size - expected_size

                        # Check if position changed by expected amount (5% tolerance)
                        size_diff = abs(abs(current_size) - abs(expected_new_size))
                        if size_diff <= expected_size * 0.05 or abs(current_size - initial_size) >= expected_size * 0.95:
                            # Position changed as expected - order filled
                            current_price = await self.exchange.get_ticker(symbol)
                            fill_price = current_price.last_price if current_price else 0.0

                            logger.info(f"Order {order_id} detected as filled via position change: {initial_size} -> {current_size}")

                            # Return synthetic filled order
                            return Order(
                                order_id=order_id,
                                symbol=symbol,
                                side=OrderSide.BUY if is_buy else OrderSide.SELL,
                                order_type=OrderType.MARKET,
                                size=expected_size,
                                price=fill_price,
                                status=OrderStatus.FILLED,
                                filled_size=expected_size,
                                avg_fill_price=fill_price
                            )

                    logger.warning(f"Order {order_id} not found on attempt {attempt + 1}")
                    await asyncio.sleep(poll_interval)
                    continue

                if order.status == OrderStatus.FILLED:
                    logger.debug(f"Order {order_id} filled at {order.avg_fill_price}")
                    return order

                if order.status == OrderStatus.PARTIALLY_FILLED:
                    logger.info(f"Order {order_id} partially filled: {order.filled_size}/{order.size}")
                    # For market orders, partial fills should complete quickly
                    await asyncio.sleep(poll_interval)
                    continue

                if order.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
                    logger.warning(f"Order {order_id} was {order.status.value}")
                    return None

                # Status is PENDING or OPEN, keep waiting
                await asyncio.sleep(poll_interval)

            except Exception as e:
                logger.warning(f"Error checking order status: {e}")
                await asyncio.sleep(poll_interval)

        logger.warning(f"Order {order_id} did not fill within {timeout_seconds}s")
        return None

    async def can_open_trade(self, symbol: str, account_balance: Optional[float] = None) -> Tuple[bool, float]:
        """
        Check if we can open a new trade and return available heat.

        Args:
            symbol: Trading symbol
            account_balance: Account balance for heat calculation (optional)

        Returns:
            Tuple of (can_trade: bool, available_heat_percent: float)
        """
        # Check total positions
        if len(self._active_trades) >= self.max_positions:
            logger.warning(f"Max positions ({self.max_positions}) reached")
            return False, 0.0

        # Check positions per asset
        asset_trades = [t for t in self._active_trades.values() if t.symbol == symbol]
        if len(asset_trades) >= self.max_positions_per_asset:
            logger.warning(f"Max positions per asset ({self.max_positions_per_asset}) reached for {symbol}")
            return False, 0.0

        # Check existing exchange positions
        try:
            position = await self.exchange.get_position(symbol)
            if position and abs(position.size) > 0:
                logger.warning(f"Existing position found for {symbol}")
                return False, 0.0
        except Exception as e:
            logger.warning(f"Failed to check existing position: {e}")

        # Check portfolio heat
        if account_balance and account_balance > 0:
            _, current_heat = self.get_current_heat(account_balance)
            available_heat = self.max_portfolio_heat_percent - current_heat

            if available_heat <= 0:
                logger.warning(f"Portfolio heat limit reached: {current_heat:.1f}% >= {self.max_portfolio_heat_percent}%")
                return False, 0.0

            logger.debug(f"Portfolio heat: {current_heat:.1f}%, available: {available_heat:.1f}%")
            return True, available_heat

        return True, self.max_portfolio_heat_percent

    async def execute_signal(
        self,
        signal: Signal,
        df: pd.DataFrame,
        risk_percent: Optional[float] = None,
        risk_reward: Optional[float] = None
    ) -> Optional[Trade]:
        """
        Execute a trading signal.

        Args:
            signal: Signal to execute
            df: DataFrame with OHLC data for SL calculation
            risk_percent: Override risk percentage
            risk_reward: Override risk:reward ratio

        Returns:
            Trade object if executed successfully
        """
        symbol = signal.metadata.get('symbol', 'BTC')

        try:
            # Get account balance FIRST (needed for heat check)
            balance = await self.exchange.get_balance()

            # CRITICAL: Validate balance before proceeding
            if balance is None or balance.available_balance <= 0:
                logger.error(f"Invalid balance for {symbol}: {balance}")
                return None

            # Check if we can open (includes heat check)
            can_trade, available_heat = await self.can_open_trade(symbol, balance.available_balance)
            if not can_trade:
                return None

            # Cap risk_percent based on available heat budget
            requested_risk = risk_percent or self.risk_manager.default_risk_percent
            effective_risk = min(requested_risk, available_heat)

            if effective_risk < requested_risk:
                logger.info(f"Risk capped by heat budget: {requested_risk}% -> {effective_risk:.1f}% (available heat: {available_heat:.1f}%)")

            # Calculate risk parameters with heat-capped risk
            is_long = signal.signal_type == SignalType.LONG
            risk_params = self.risk_manager.calculate_full_risk_params(
                account_balance=balance.available_balance,
                entry_price=signal.entry_price,
                is_long=is_long,
                df=df,
                sl_method=self.sl_method,
                risk_percent=effective_risk,
                risk_reward=risk_reward
            )

            # CRITICAL: Validate position size before placing order (check for NaN too!)
            if risk_params.position_size <= 0 or math.isnan(risk_params.position_size):
                logger.error(f"Invalid position size for {symbol}: {risk_params.position_size}")
                return None

            # Validate SL/TP are not NaN
            if math.isnan(risk_params.stop_loss) or math.isnan(risk_params.take_profit):
                logger.error(f"Invalid SL/TP for {symbol}: SL={risk_params.stop_loss}, TP={risk_params.take_profit}")
                return None

            # Check minimum notional value ($10 on Hyperliquid)
            notional = risk_params.position_size * signal.entry_price
            MIN_NOTIONAL = 11.0
            if notional < MIN_NOTIONAL:
                logger.warning(f"Position notional ${notional:.2f} below minimum ${MIN_NOTIONAL} for {symbol}")
                return None

            # Create trade object with metadata including timeframe for exit signal checking
            trade = Trade(
                trade_id=self._generate_trade_id(),
                symbol=symbol,
                signal_type=signal.signal_type,
                entry_price=signal.entry_price,
                size=risk_params.position_size,
                stop_loss=risk_params.stop_loss,
                take_profit=risk_params.take_profit,
                metadata={
                    "signal": signal.to_dict(),
                    "risk_params": {
                        "risk_amount": risk_params.risk_amount,
                        "risk_reward": risk_params.risk_reward
                    },
                    # Copy key signal metadata to top level for easy access
                    "symbol": signal.metadata.get('symbol', symbol),
                    "timeframe": signal.metadata.get('timeframe', '5m'),
                    "strategy": signal.metadata.get('strategy', ''),
                }
            )

            # Place entry order
            order_side = OrderSide.BUY if is_long else OrderSide.SELL
            logger.info(f"Placing entry order: {symbol} {order_side.value} size={risk_params.position_size:.6f} SL={risk_params.stop_loss:.2f} TP={risk_params.take_profit:.2f}")

            entry_order = await self.exchange.place_order(
                symbol=symbol,
                side=order_side,
                order_type=OrderType.MARKET,
                size=risk_params.position_size,
                stop_loss=risk_params.stop_loss,
                take_profit=risk_params.take_profit
            )

            trade.entry_order_id = entry_order.order_id

            # Wait for fill confirmation with timeout
            filled_order = await self._wait_for_fill(
                order_id=entry_order.order_id,
                symbol=symbol,
                timeout_seconds=30,
                poll_interval=1.0,
                expected_size=risk_params.position_size,
                is_buy=is_long
            )

            if filled_order is None:
                # Order didn't fill - cancel and abort
                logger.warning(f"Order {entry_order.order_id} did not fill, cancelling...")
                try:
                    await self.exchange.cancel_order(entry_order.order_id, symbol)
                except Exception as cancel_err:
                    logger.error(f"Failed to cancel unfilled order: {cancel_err}")
                raise OrderNotFilledException(f"Order {entry_order.order_id} did not fill within timeout")

            # Use actual fill price from exchange
            actual_fill_price = filled_order.avg_fill_price or signal.entry_price
            trade.entry_price = actual_fill_price

            # BACKTEST-ALIGNED: Do NOT recalculate TP based on fill price
            # Previous behavior adjusted TP when fill differed from signal price
            # Backtest uses original signal TP without adjustment
            # The original TP from risk_params is already set in trade.take_profit

            # Update trade with order info
            trade.status = TradeStatus.OPEN
            trade.opened_at = datetime.utcnow()

            # Store SL/TP order IDs if the exchange returns them
            if hasattr(filled_order, 'stop_loss_order_id'):
                trade.sl_order_id = filled_order.stop_loss_order_id
            if hasattr(filled_order, 'take_profit_order_id'):
                trade.tp_order_id = filled_order.take_profit_order_id

            # Store active trade
            self._active_trades[trade.trade_id] = trade

            logger.info(
                f"Opened {signal.signal_type.value} trade: {symbol} @ {actual_fill_price} "
                f"(requested: {signal.entry_price}), "
                f"Size: {risk_params.position_size}, SL: {trade.stop_loss}, TP: {trade.take_profit}"
            )

            return trade

        except Exception as e:
            logger.error(f"Failed to execute signal: {e}")
            raise

    async def close_trade(
        self,
        trade_id: str,
        exit_price: Optional[float] = None,
        size: Optional[float] = None
    ) -> Optional[Trade]:
        """
        Close a trade.

        Args:
            trade_id: Trade ID to close
            exit_price: Exit price (uses market if None)
            size: Size to close (closes full position if None)

        Returns:
            Updated Trade object
        """
        if trade_id not in self._active_trades:
            logger.warning(f"Trade {trade_id} not found")
            return None

        trade = self._active_trades[trade_id]

        try:
            close_size = size or trade.size

            # Close position on exchange
            close_order = await self.exchange.close_position(
                symbol=trade.symbol,
                size=close_size
            )

            # Update trade
            trade.exit_price = exit_price or close_order.avg_fill_price or trade.entry_price
            trade.closed_at = datetime.utcnow()

            # Calculate PnL
            if trade.is_long:
                trade.pnl = (trade.exit_price - trade.entry_price) * close_size
            else:
                trade.pnl = (trade.entry_price - trade.exit_price) * close_size

            trade.pnl_percent = (trade.pnl / (trade.entry_price * close_size)) * 100

            # Update status
            if close_size >= trade.size:
                trade.status = TradeStatus.CLOSED
                del self._active_trades[trade_id]
                self._trade_history.append(trade)
            else:
                trade.partial_closed_size += close_size
                trade.status = TradeStatus.PARTIAL_TP

            logger.info(
                f"Closed trade {trade_id}: {trade.symbol} @ {trade.exit_price}, "
                f"PnL: {trade.pnl:.2f} ({trade.pnl_percent:.2f}%)"
            )

            return trade

        except Exception as e:
            logger.error(f"Failed to close trade {trade_id}: {e}")
            raise

    async def update_stops(self, trade_id: str, stop_loss: float) -> bool:
        """
        Update stop loss for a trade.

        Args:
            trade_id: Trade ID
            stop_loss: New stop loss price

        Returns:
            True if successful
        """
        if trade_id not in self._active_trades:
            return False

        trade = self._active_trades[trade_id]

        try:
            # Cancel existing SL order if any
            if trade.sl_order_id:
                await self.exchange.cancel_order(trade.sl_order_id, trade.symbol)

            # Place new SL order
            sl_side = OrderSide.SELL if trade.is_long else OrderSide.BUY

            # Note: This is a simplified approach
            # Real implementation would use exchange-specific SL order types
            trade.stop_loss = stop_loss

            logger.info(f"Updated stop loss for {trade_id} to {stop_loss}")
            return True

        except Exception as e:
            logger.error(f"Failed to update stops for {trade_id}: {e}")
            return False

    async def check_and_update_trailing_stops(
        self,
        current_prices: Dict[str, float]
    ) -> List[str]:
        """
        Check and update trailing stops for all active trades.

        Args:
            current_prices: Dict of symbol -> current price

        Returns:
            List of trade IDs that were updated
        """
        if self.tp_method != TakeProfitMethod.TRAILING:
            return []

        updated = []

        for trade_id, trade in self._active_trades.items():
            if trade.symbol not in current_prices:
                continue

            current_price = current_prices[trade.symbol]

            new_stop = self.risk_manager.calculate_trailing_stop(
                current_price=current_price,
                entry_price=trade.entry_price,
                stop_loss_price=trade.stop_loss,
                current_stop=trade.stop_loss,
                is_long=trade.is_long
            )

            if new_stop != trade.stop_loss:
                await self.update_stops(trade_id, new_stop)
                updated.append(trade_id)

        return updated

    async def check_exit_conditions(
        self,
        trade_id: str,
        current_price: float,
        opposite_signal: bool = False
    ) -> Optional[str]:
        """
        Check if trade should be closed.

        Args:
            trade_id: Trade ID to check
            current_price: Current market price
            opposite_signal: True if opposite signal detected

        Returns:
            Reason for exit if should close, None otherwise
        """
        if trade_id not in self._active_trades:
            return None

        trade = self._active_trades[trade_id]

        # Check stop loss
        if trade.is_long and current_price <= trade.stop_loss:
            return "stop_loss"
        if not trade.is_long and current_price >= trade.stop_loss:
            return "stop_loss"

        # Check take profit
        if trade.is_long and current_price >= trade.take_profit:
            return "take_profit"
        if not trade.is_long and current_price <= trade.take_profit:
            return "take_profit"

        # Check oscillator exit
        if self.tp_method == TakeProfitMethod.OSCILLATOR and opposite_signal:
            return "oscillator_exit"

        return None


    @property
    def trade_history(self) -> List[Trade]:
        """Get trade history."""
        return self._trade_history.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get trading statistics."""
        if not self._trade_history:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_pnl": 0,
            }

        total = len(self._trade_history)
        winners = [t for t in self._trade_history if t.pnl > 0]
        losers = [t for t in self._trade_history if t.pnl < 0]

        total_pnl = sum(t.pnl for t in self._trade_history)
        avg_pnl = total_pnl / total if total > 0 else 0

        return {
            "total_trades": total,
            "winning_trades": len(winners),
            "losing_trades": len(losers),
            "win_rate": len(winners) / total * 100 if total > 0 else 0,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "best_trade": max((t.pnl for t in self._trade_history), default=0),
            "worst_trade": min((t.pnl for t in self._trade_history), default=0),
        }
