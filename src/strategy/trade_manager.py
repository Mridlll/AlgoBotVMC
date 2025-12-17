"""Trade execution and position management."""

import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from exchanges.base import (
    BaseExchange, Order, Position, OrderSide, OrderType, PositionSide
)
from strategy.signals import Signal, SignalType
from strategy.risk import RiskManager, RiskParams, StopLossMethod, TakeProfitMethod
from utils.logger import get_logger

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
        max_positions_per_asset: int = 1
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
        """
        self.exchange = exchange
        self.risk_manager = risk_manager
        self.tp_method = tp_method
        self.sl_method = sl_method
        self.max_positions = max_positions
        self.max_positions_per_asset = max_positions_per_asset

        self._active_trades: Dict[str, Trade] = {}
        self._trade_history: List[Trade] = []
        self._trade_counter = 0

    def _generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        self._trade_counter += 1
        return f"trade_{self._trade_counter}_{int(datetime.utcnow().timestamp())}"

    async def can_open_trade(self, symbol: str) -> bool:
        """
        Check if we can open a new trade.

        Args:
            symbol: Trading symbol

        Returns:
            True if we can open a trade
        """
        # Check total positions
        if len(self._active_trades) >= self.max_positions:
            logger.warning(f"Max positions ({self.max_positions}) reached")
            return False

        # Check positions per asset
        asset_trades = [t for t in self._active_trades.values() if t.symbol == symbol]
        if len(asset_trades) >= self.max_positions_per_asset:
            logger.warning(f"Max positions per asset ({self.max_positions_per_asset}) reached for {symbol}")
            return False

        # Check existing exchange positions
        try:
            position = await self.exchange.get_position(symbol)
            if position and position.size > 0:
                logger.warning(f"Existing position found for {symbol}")
                return False
        except Exception as e:
            logger.warning(f"Failed to check existing position: {e}")

        return True

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

        # Check if we can open
        if not await self.can_open_trade(symbol):
            return None

        try:
            # Get account balance
            balance = await self.exchange.get_balance()

            # Calculate risk parameters
            is_long = signal.signal_type == SignalType.LONG
            risk_params = self.risk_manager.calculate_full_risk_params(
                account_balance=balance.available_balance,
                entry_price=signal.entry_price,
                is_long=is_long,
                df=df,
                sl_method=self.sl_method,
                risk_percent=risk_percent,
                risk_reward=risk_reward
            )

            # Create trade object
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
                    }
                }
            )

            # Place entry order
            order_side = OrderSide.BUY if is_long else OrderSide.SELL

            entry_order = await self.exchange.place_order(
                symbol=symbol,
                side=order_side,
                order_type=OrderType.MARKET,
                size=risk_params.position_size,
                stop_loss=risk_params.stop_loss,
                take_profit=risk_params.take_profit
            )

            # Update trade with order info
            trade.entry_order_id = entry_order.order_id
            trade.status = TradeStatus.OPEN
            trade.opened_at = datetime.utcnow()

            # Store active trade
            self._active_trades[trade.trade_id] = trade

            logger.info(
                f"Opened {signal.signal_type.value} trade: {symbol} @ {signal.entry_price}, "
                f"Size: {risk_params.position_size}, SL: {risk_params.stop_loss}, TP: {risk_params.take_profit}"
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
    def active_trades(self) -> Dict[str, Trade]:
        """Get all active trades."""
        return self._active_trades.copy()

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
