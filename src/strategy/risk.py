"""Risk management and position sizing."""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from utils.helpers import round_to_tick


class StopLossMethod(str, Enum):
    """Stop loss calculation methods."""
    SWING = "swing"
    ATR = "atr"
    FIXED_PERCENT = "fixed_percent"


class TakeProfitMethod(str, Enum):
    """Take profit methods."""
    FIXED_RR = "fixed_rr"
    OSCILLATOR = "oscillator"
    PARTIAL = "partial"
    TRAILING = "trailing"


@dataclass
class RiskParams:
    """Risk calculation parameters."""
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_amount: float
    risk_reward: float


class RiskManager:
    """
    Risk management for position sizing and SL/TP calculation.

    Handles:
    - Position sizing based on risk percentage
    - Stop loss calculation (swing, ATR, fixed)
    - Take profit calculation (fixed RR, oscillator, partial, trailing)
    """

    def __init__(
        self,
        default_risk_percent: float = 3.0,
        default_leverage: float = 1.0,
        default_rr: float = 2.0,
        swing_lookback: int = 5,
        swing_buffer_percent: float = 0.5,
        atr_period: int = 14,
        atr_multiplier: float = 1.5,
        fixed_sl_percent: float = 2.0
    ):
        """
        Initialize risk manager.

        Args:
            default_risk_percent: Default risk per trade (%)
            default_leverage: Default leverage
            default_rr: Default risk:reward ratio
            swing_lookback: Candles to look back for swing points
            swing_buffer_percent: Buffer below/above swing levels (%)
            atr_period: ATR calculation period
            atr_multiplier: ATR multiplier for SL distance
            fixed_sl_percent: Fixed SL distance (%)
        """
        self.default_risk_percent = default_risk_percent
        self.default_leverage = default_leverage
        self.default_rr = default_rr
        self.swing_lookback = swing_lookback
        self.swing_buffer_percent = swing_buffer_percent
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.fixed_sl_percent = fixed_sl_percent

    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        risk_percent: Optional[float] = None,
        leverage: Optional[float] = None,
        tick_size: float = 0.0001
    ) -> float:
        """
        Calculate position size based on risk.

        Args:
            account_balance: Total account balance
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            risk_percent: Risk percentage (uses default if None)
            leverage: Leverage (uses default if None)
            tick_size: Minimum size increment

        Returns:
            Position size in base currency
        """
        risk = (risk_percent or self.default_risk_percent) / 100
        lev = leverage or self.default_leverage

        # Risk amount in USD
        risk_amount = account_balance * risk

        # Price difference (risk per unit)
        price_diff = abs(entry_price - stop_loss_price)

        if price_diff == 0:
            return 0.0

        # Position size = risk_amount / price_diff
        # With leverage: effective position can be larger
        position_size = risk_amount / price_diff

        # Round to tick size
        if tick_size > 0:
            position_size = round_to_tick(position_size, tick_size)

        return position_size

    def calculate_stop_loss(
        self,
        entry_price: float,
        is_long: bool,
        df: pd.DataFrame,
        method: StopLossMethod = StopLossMethod.SWING,
        buffer_percent: Optional[float] = None
    ) -> float:
        """
        Calculate stop loss price.

        Args:
            entry_price: Entry price
            is_long: True for long positions
            df: DataFrame with OHLC data
            method: Stop loss calculation method
            buffer_percent: Override buffer percentage

        Returns:
            Stop loss price
        """
        buffer = (buffer_percent or self.swing_buffer_percent) / 100

        if method == StopLossMethod.SWING:
            return self._swing_stop_loss(entry_price, is_long, df, buffer)
        elif method == StopLossMethod.ATR:
            return self._atr_stop_loss(entry_price, is_long, df)
        elif method == StopLossMethod.FIXED_PERCENT:
            return self._fixed_stop_loss(entry_price, is_long)
        else:
            return self._swing_stop_loss(entry_price, is_long, df, buffer)

    def _swing_stop_loss(
        self,
        entry_price: float,
        is_long: bool,
        df: pd.DataFrame,
        buffer: float
    ) -> float:
        """Calculate swing-based stop loss."""
        lookback = min(self.swing_lookback, len(df))

        if is_long:
            # Stop below recent swing low
            swing_low = df['low'].iloc[-lookback:].min()
            sl = swing_low * (1 - buffer)
        else:
            # Stop above recent swing high
            swing_high = df['high'].iloc[-lookback:].max()
            sl = swing_high * (1 + buffer)

        return sl

    def _atr_stop_loss(
        self,
        entry_price: float,
        is_long: bool,
        df: pd.DataFrame
    ) -> float:
        """Calculate ATR-based stop loss."""
        atr = self._calculate_atr(df, self.atr_period)
        atr_distance = atr * self.atr_multiplier

        if is_long:
            return entry_price - atr_distance
        else:
            return entry_price + atr_distance

    def _fixed_stop_loss(
        self,
        entry_price: float,
        is_long: bool
    ) -> float:
        """Calculate fixed percentage stop loss."""
        sl_distance = entry_price * (self.fixed_sl_percent / 100)

        if is_long:
            return entry_price - sl_distance
        else:
            return entry_price + sl_distance

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss_price: float,
        is_long: bool,
        risk_reward: Optional[float] = None
    ) -> float:
        """
        Calculate take profit price based on risk:reward ratio.

        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            is_long: True for long positions
            risk_reward: Risk:reward ratio (uses default if None)

        Returns:
            Take profit price
        """
        rr = risk_reward or self.default_rr
        sl_distance = abs(entry_price - stop_loss_price)
        tp_distance = sl_distance * rr

        if is_long:
            return entry_price + tp_distance
        else:
            return entry_price - tp_distance

    def calculate_partial_tp_levels(
        self,
        entry_price: float,
        stop_loss_price: float,
        is_long: bool,
        first_tp_rr: float = 1.0,
        final_tp_rr: float = 2.0
    ) -> Tuple[float, float]:
        """
        Calculate partial take profit levels.

        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            is_long: True for long positions
            first_tp_rr: R:R for first partial TP
            final_tp_rr: R:R for final TP

        Returns:
            Tuple of (first_tp_price, final_tp_price)
        """
        sl_distance = abs(entry_price - stop_loss_price)

        first_tp_distance = sl_distance * first_tp_rr
        final_tp_distance = sl_distance * final_tp_rr

        if is_long:
            first_tp = entry_price + first_tp_distance
            final_tp = entry_price + final_tp_distance
        else:
            first_tp = entry_price - first_tp_distance
            final_tp = entry_price - final_tp_distance

        return first_tp, final_tp

    def calculate_trailing_stop(
        self,
        current_price: float,
        entry_price: float,
        stop_loss_price: float,
        current_stop: float,
        is_long: bool,
        trailing_distance_percent: float = 1.0,
        activation_rr: float = 1.5
    ) -> float:
        """
        Calculate trailing stop price.

        Args:
            current_price: Current market price
            entry_price: Original entry price
            stop_loss_price: Original stop loss price
            current_stop: Current stop loss level
            is_long: True for long positions
            trailing_distance_percent: Trail distance (%)
            activation_rr: R:R level to activate trailing

        Returns:
            New trailing stop price
        """
        sl_distance = abs(entry_price - stop_loss_price)
        activation_distance = sl_distance * activation_rr
        trail_distance = current_price * (trailing_distance_percent / 100)

        if is_long:
            # Check if trailing should activate
            profit_distance = current_price - entry_price
            if profit_distance < activation_distance:
                return current_stop  # Not activated yet

            # Calculate new trailing stop
            new_stop = current_price - trail_distance

            # Only move stop up, never down
            return max(new_stop, current_stop)
        else:
            # Short position
            profit_distance = entry_price - current_price
            if profit_distance < activation_distance:
                return current_stop

            new_stop = current_price + trail_distance
            return min(new_stop, current_stop)

    def calculate_full_risk_params(
        self,
        account_balance: float,
        entry_price: float,
        is_long: bool,
        df: pd.DataFrame,
        sl_method: StopLossMethod = StopLossMethod.SWING,
        risk_percent: Optional[float] = None,
        risk_reward: Optional[float] = None,
        tick_size: float = 0.0001
    ) -> RiskParams:
        """
        Calculate all risk parameters for a trade.

        Args:
            account_balance: Account balance
            entry_price: Entry price
            is_long: True for long
            df: DataFrame with OHLC data
            sl_method: Stop loss method
            risk_percent: Risk percentage
            risk_reward: Risk:reward ratio
            tick_size: Minimum size increment

        Returns:
            RiskParams with all calculated values
        """
        # Calculate stop loss
        stop_loss = self.calculate_stop_loss(entry_price, is_long, df, sl_method)

        # Calculate take profit
        rr = risk_reward or self.default_rr
        take_profit = self.calculate_take_profit(entry_price, stop_loss, is_long, rr)

        # Calculate position size
        position_size = self.calculate_position_size(
            account_balance, entry_price, stop_loss,
            risk_percent, tick_size=tick_size
        )

        # Calculate risk amount
        risk = (risk_percent or self.default_risk_percent) / 100
        risk_amount = account_balance * risk

        return RiskParams(
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            risk_amount=risk_amount,
            risk_reward=rr
        )

    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int) -> float:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]

        return atr

    def find_swing_low(self, df: pd.DataFrame, lookback: Optional[int] = None) -> float:
        """Find the lowest low in recent candles."""
        lb = lookback or self.swing_lookback
        return df['low'].iloc[-lb:].min()

    def find_swing_high(self, df: pd.DataFrame, lookback: Optional[int] = None) -> float:
        """Find the highest high in recent candles."""
        lb = lookback or self.swing_lookback
        return df['high'].iloc[-lb:].max()
