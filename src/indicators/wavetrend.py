"""WaveTrend Oscillator implementation."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class WaveTrendResult:
    """WaveTrend calculation results."""
    wt1: pd.Series  # Fast wave (EMA of CI)
    wt2: pd.Series  # Slow wave (SMA of WT1)
    momentum: pd.Series  # WT1 - WT2 (NOT real VWAP - use VWAPCalculator for that)

    # Crosses
    cross: pd.Series  # Any cross
    cross_up: pd.Series  # Bullish cross (wt1 crosses above wt2)
    cross_down: pd.Series  # Bearish cross (wt1 crosses below wt2)

    # Overbought/Oversold
    oversold: pd.Series
    overbought: pd.Series


class WaveTrend:
    """
    WaveTrend Oscillator indicator.

    The WaveTrend oscillator is a momentum indicator that identifies
    overbought and oversold conditions using a smoothed channel
    of the typical price.

    Based on LazyBear's WaveTrend Oscillator for TradingView.
    """

    def __init__(
        self,
        channel_len: int = 9,
        average_len: int = 12,
        ma_len: int = 3,
        overbought_1: int = 53,
        overbought_2: int = 60,
        oversold_1: int = -53,
        oversold_2: int = -60
    ):
        """
        Initialize WaveTrend indicator.

        Args:
            channel_len: Channel length for EMA (default 9)
            average_len: Average length for WT1 EMA (default 12)
            ma_len: MA length for WT2 SMA (default 3)
            overbought_1: First overbought level (default 53)
            overbought_2: Second overbought level - anchor for shorts (default 60)
            oversold_1: First oversold level (default -53)
            oversold_2: Second oversold level - anchor for longs (default -60)
        """
        self.channel_len = channel_len
        self.average_len = average_len
        self.ma_len = ma_len
        self.overbought_1 = overbought_1
        self.overbought_2 = overbought_2
        self.oversold_1 = oversold_1
        self.oversold_2 = oversold_2

    def calculate(
        self,
        df: pd.DataFrame,
        src_col: str = 'hlc3'
    ) -> WaveTrendResult:
        """
        Calculate WaveTrend indicator values.

        Args:
            df: DataFrame with OHLC data
            src_col: Source column name or 'hlc3' for typical price

        Returns:
            WaveTrendResult with all calculated values
        """
        # Get source data
        if src_col == 'hlc3':
            src = (df['high'] + df['low'] + df['close']) / 3
        elif src_col in df.columns:
            src = df[src_col]
        else:
            raise ValueError(f"Source column '{src_col}' not found in DataFrame")

        # Calculate WaveTrend
        # ESA = EMA(Source, Channel Length)
        esa = self._ema(src, self.channel_len)

        # D = EMA(|Source - ESA|, Channel Length)
        d = self._ema(abs(src - esa), self.channel_len)

        # CI = (Source - ESA) / (0.015 * D)
        # Handle division by zero
        ci = pd.Series(np.where(d != 0, (src - esa) / (0.015 * d), 0), index=src.index)

        # WT1 = EMA(CI, Average Length)
        wt1 = self._ema(ci, self.average_len)

        # WT2 = SMA(WT1, MA Length)
        wt2 = self._sma(wt1, self.ma_len)

        # Momentum = WT1 - WT2 (Note: This is NOT real VWAP)
        momentum = wt1 - wt2

        # Crosses
        cross_up = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
        cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
        cross = cross_up | cross_down

        # Overbought/Oversold (using wt2 as the main wave)
        oversold = wt2 <= self.oversold_1
        overbought = wt2 >= self.overbought_1

        return WaveTrendResult(
            wt1=wt1,
            wt2=wt2,
            momentum=momentum,
            cross=cross,
            cross_up=cross_up,
            cross_down=cross_down,
            oversold=oversold,
            overbought=overbought
        )

    def calculate_single(
        self,
        prices: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calculate WaveTrend for the most recent value only.

        Args:
            prices: Array of hlc3 prices (most recent last)

        Returns:
            Tuple of (wt1, wt2, momentum) for the latest bar
        """
        df = pd.DataFrame({'hlc3': prices})
        df['high'] = prices
        df['low'] = prices
        df['close'] = prices

        result = self.calculate(df, src_col='hlc3')

        return (
            result.wt1.iloc[-1],
            result.wt2.iloc[-1],
            result.momentum.iloc[-1]
        )

    def is_anchor_long(self, wt2: float) -> bool:
        """
        Check if WT2 is at anchor level for longs.

        Args:
            wt2: Current WT2 value

        Returns:
            True if below oversold_2 level (-60)
        """
        return wt2 <= self.oversold_2

    def is_anchor_short(self, wt2: float) -> bool:
        """
        Check if WT2 is at anchor level for shorts.

        Args:
            wt2: Current WT2 value

        Returns:
            True if above overbought_2 level (60)
        """
        return wt2 >= self.overbought_2

    def momentum_crossed_up(self, momentum_current: float, momentum_prev: float) -> bool:
        """
        Check if momentum (WT1-WT2) crossed above 0.

        Args:
            momentum_current: Current momentum value
            momentum_prev: Previous momentum value

        Returns:
            True if crossed up through 0
        """
        return momentum_current > 0 and momentum_prev <= 0

    def momentum_crossed_down(self, momentum_current: float, momentum_prev: float) -> bool:
        """
        Check if momentum (WT1-WT2) crossed below 0.

        Args:
            momentum_current: Current momentum value
            momentum_prev: Previous momentum value

        Returns:
            True if crossed down through 0
        """
        return momentum_current < 0 and momentum_prev >= 0

    @staticmethod
    def _ema(series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _sma(series: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return series.rolling(window=period).mean()


def calculate_wavetrend(
    df: pd.DataFrame,
    channel_len: int = 9,
    average_len: int = 12,
    ma_len: int = 3
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Convenience function to calculate WaveTrend values.

    Args:
        df: DataFrame with OHLC data
        channel_len: Channel length
        average_len: Average length
        ma_len: MA length

    Returns:
        Tuple of (wt1, wt2, momentum) Series
    """
    wt = WaveTrend(channel_len, average_len, ma_len)
    result = wt.calculate(df)
    return result.wt1, result.wt2, result.momentum
