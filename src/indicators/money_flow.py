"""Money Flow (RSI+MFI) indicator implementation."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class MoneyFlowResult:
    """Money Flow calculation results."""
    mfi: pd.Series  # MFI values
    is_positive: pd.Series  # MFI > 0 (green)
    is_negative: pd.Series  # MFI < 0 (red)
    curving_up: pd.Series  # MFI is rising
    curving_down: pd.Series  # MFI is falling


class MoneyFlow:
    """
    Money Flow indicator (RSI+MFI Area).

    This indicator measures buying and selling pressure based on
    the relationship between candle body direction and range.

    Green (positive): Buyers in control
    Red (negative): Sellers in control

    Based on andreholanda73's MFI+RSI Area for TradingView.
    """

    def __init__(
        self,
        period: int = 60,
        multiplier: float = 150.0,
        y_pos: float = 2.5
    ):
        """
        Initialize Money Flow indicator.

        Args:
            period: SMA period for smoothing (default 60)
            multiplier: Multiplier for the MFI area (default 150)
            y_pos: Y-axis position offset (default 2.5)
        """
        self.period = period
        self.multiplier = multiplier
        self.y_pos = y_pos

    def calculate(self, df: pd.DataFrame) -> MoneyFlowResult:
        """
        Calculate Money Flow indicator values.

        Args:
            df: DataFrame with OHLC data (open, high, low, close)

        Returns:
            MoneyFlowResult with all calculated values
        """
        # Calculate raw MFI
        # MFI = SMA(((Close - Open) / (High - Low)) * Multiplier, Period) - Y_Pos

        # Handle division by zero (doji candles where high == low)
        range_val = df['high'] - df['low']
        range_val = range_val.replace(0, np.nan)

        body = df['close'] - df['open']
        raw_mfi = (body / range_val) * self.multiplier

        # Fill NaN with 0 (flat candles have no directional pressure)
        raw_mfi = raw_mfi.fillna(0)

        # Apply SMA smoothing and offset
        mfi = self._sma(raw_mfi, self.period) - self.y_pos

        # Determine direction
        is_positive = mfi > 0
        is_negative = mfi < 0

        # Determine if curving (changing direction)
        mfi_diff = mfi.diff()
        mfi_diff_prev = mfi_diff.shift(1)

        # Curving up: MFI was falling (or flat) and now rising
        # More specifically: current derivative > previous derivative
        curving_up = mfi_diff > mfi_diff_prev

        # Curving down: MFI was rising (or flat) and now falling
        curving_down = mfi_diff < mfi_diff_prev

        return MoneyFlowResult(
            mfi=mfi,
            is_positive=is_positive,
            is_negative=is_negative,
            curving_up=curving_up,
            curving_down=curving_down
        )

    def calculate_single(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray
    ) -> Tuple[float, bool, bool]:
        """
        Calculate Money Flow for the most recent value only.

        Args:
            opens: Array of open prices
            highs: Array of high prices
            lows: Array of low prices
            closes: Array of close prices

        Returns:
            Tuple of (mfi_value, is_curving_up, is_curving_down)
        """
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes
        })

        result = self.calculate(df)

        return (
            result.mfi.iloc[-1],
            result.curving_up.iloc[-1],
            result.curving_down.iloc[-1]
        )

    def is_curving_up(
        self,
        mfi_current: float,
        mfi_prev: float,
        mfi_prev2: float
    ) -> bool:
        """
        Check if MFI is curving up (forming a bottom).

        The MFI is curving up when:
        - It was falling (mfi_prev < mfi_prev2)
        - And now rising or flat (mfi_current >= mfi_prev)

        Args:
            mfi_current: Current MFI value
            mfi_prev: Previous MFI value
            mfi_prev2: MFI value 2 bars ago

        Returns:
            True if MFI is curving up
        """
        was_falling = mfi_prev < mfi_prev2
        now_rising = mfi_current >= mfi_prev
        return was_falling and now_rising

    def is_curving_down(
        self,
        mfi_current: float,
        mfi_prev: float,
        mfi_prev2: float
    ) -> bool:
        """
        Check if MFI is curving down (forming a top).

        The MFI is curving down when:
        - It was rising (mfi_prev > mfi_prev2)
        - And now falling or flat (mfi_current <= mfi_prev)

        Args:
            mfi_current: Current MFI value
            mfi_prev: Previous MFI value
            mfi_prev2: MFI value 2 bars ago

        Returns:
            True if MFI is curving down
        """
        was_rising = mfi_prev > mfi_prev2
        now_falling = mfi_current <= mfi_prev
        return was_rising and now_falling

    def is_turning_bullish(
        self,
        mfi_values: pd.Series,
        lookback: int = 3
    ) -> bool:
        """
        Check if MFI is showing bullish momentum shift.

        Bullish when:
        - MFI is negative (red area)
        - But starting to curve up (sellers losing control)

        Args:
            mfi_values: Series of MFI values
            lookback: Number of bars to check

        Returns:
            True if showing bullish momentum shift
        """
        if len(mfi_values) < lookback + 1:
            return False

        recent = mfi_values.iloc[-lookback:]
        current = mfi_values.iloc[-1]

        # Must be negative (red wave)
        if current >= 0:
            return False

        # Check if curving up (rising from a low)
        min_idx = recent.idxmin()
        current_idx = mfi_values.index[-1]

        # If minimum was not at the end, we're curving up
        return min_idx != current_idx and current > recent.min()

    def is_turning_bearish(
        self,
        mfi_values: pd.Series,
        lookback: int = 3
    ) -> bool:
        """
        Check if MFI is showing bearish momentum shift.

        Bearish when:
        - MFI is positive (green area)
        - But starting to curve down (buyers losing control)

        Args:
            mfi_values: Series of MFI values
            lookback: Number of bars to check

        Returns:
            True if showing bearish momentum shift
        """
        if len(mfi_values) < lookback + 1:
            return False

        recent = mfi_values.iloc[-lookback:]
        current = mfi_values.iloc[-1]

        # Must be positive (green wave)
        if current <= 0:
            return False

        # Check if curving down (falling from a high)
        max_idx = recent.idxmax()
        current_idx = mfi_values.index[-1]

        # If maximum was not at the end, we're curving down
        return max_idx != current_idx and current < recent.max()

    @staticmethod
    def _sma(series: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return series.rolling(window=period, min_periods=1).mean()


def calculate_money_flow(
    df: pd.DataFrame,
    period: int = 60,
    multiplier: float = 150.0,
    y_pos: float = 2.5
) -> pd.Series:
    """
    Convenience function to calculate Money Flow values.

    Args:
        df: DataFrame with OHLC data
        period: SMA period
        multiplier: MFI multiplier
        y_pos: Y-axis offset

    Returns:
        Series of MFI values
    """
    mf = MoneyFlow(period, multiplier, y_pos)
    result = mf.calculate(df)
    return result.mfi
