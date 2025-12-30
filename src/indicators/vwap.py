"""Real VWAP (Volume Weighted Average Price) indicator implementation.

This is the TRUE VWAP, not to be confused with the 'vwap' field in WaveTrendResult
which is actually wt1 - wt2 (momentum difference).

True VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
where Typical Price = (High + Low + Close) / 3
"""

import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class VWAPResult:
    """VWAP calculation results."""
    vwap: pd.Series           # True VWAP values
    price_vs_vwap: pd.Series  # Close - VWAP (positive = above VWAP)
    curving_up: pd.Series     # VWAP direction changing from down to up
    curving_down: pd.Series   # VWAP direction changing from up to down
    bands_upper: pd.Series    # VWAP + 2 standard deviations
    bands_lower: pd.Series    # VWAP - 2 standard deviations


class VWAPCalculator:
    """
    Volume Weighted Average Price Calculator.

    VWAP is a trading benchmark that gives the average price a security
    has traded at throughout the day, based on both volume and price.

    Features:
    - Daily reset at 00:00 UTC (matches TradingView)
    - Curving detection for trend reversals
    - Standard deviation bands for overbought/oversold levels

    Usage:
        calculator = VWAPCalculator()
        result = calculator.calculate(df)

        # Check if price is above VWAP with curving confirmation
        if result.price_vs_vwap.iloc[-1] > 0 and result.curving_up.iloc[-1]:
            # Bullish signal
            pass
    """

    def __init__(
        self,
        reset_period: str = "daily",
        band_multiplier: float = 2.0
    ):
        """
        Initialize VWAP calculator.

        Args:
            reset_period: When to reset cumulative values ("daily" or "session")
                         Default "daily" resets at 00:00 UTC to match TradingView
            band_multiplier: Standard deviation multiplier for bands (default 2.0)
        """
        self.reset_period = reset_period
        self.band_multiplier = band_multiplier

    def calculate(self, df: pd.DataFrame) -> VWAPResult:
        """
        Calculate VWAP with daily reset.

        Args:
            df: DataFrame with OHLCV data (open, high, low, close, volume)
                Index should be DatetimeIndex for proper daily reset

        Returns:
            VWAPResult with all calculated values
        """
        # Validate required columns
        required = ['high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Calculate typical price
        typical_price = (df['high'] + df['low'] + df['close']) / 3

        # Calculate price * volume
        pv = typical_price * df['volume']

        # Get day boundaries for reset
        if isinstance(df.index, pd.DatetimeIndex):
            # Group by date for daily reset
            day_groups = df.index.date
        else:
            # If no datetime index, treat entire series as one session
            day_groups = np.zeros(len(df))

        # Calculate cumulative values with daily reset
        cum_pv = self._cumsum_with_reset(pv, day_groups)
        cum_vol = self._cumsum_with_reset(df['volume'], day_groups)

        # Calculate VWAP (handle zero volume)
        vwap = cum_pv / cum_vol.replace(0, np.nan)
        vwap = vwap.ffill()  # Forward fill for zero volume bars

        # Calculate standard deviation bands
        # Variance = cumsum((price - vwap)^2 * volume) / cumsum(volume)
        squared_diff = ((typical_price - vwap) ** 2) * df['volume']
        cum_sq_diff = self._cumsum_with_reset(squared_diff, day_groups)
        variance = cum_sq_diff / cum_vol.replace(0, np.nan)
        std_dev = np.sqrt(variance.fillna(0))

        bands_upper = vwap + (self.band_multiplier * std_dev)
        bands_lower = vwap - (self.band_multiplier * std_dev)

        # Calculate price vs VWAP
        price_vs_vwap = df['close'] - vwap

        # Calculate curving (direction change)
        vwap_diff = vwap.diff()
        vwap_diff_prev = vwap_diff.shift(1)

        # Curving up: was falling (negative diff) and now rising (positive diff)
        # Or acceleration is positive (second derivative > 0)
        curving_up = (vwap_diff > 0) & (vwap_diff_prev <= 0)

        # Curving down: was rising (positive diff) and now falling (negative diff)
        curving_down = (vwap_diff < 0) & (vwap_diff_prev >= 0)

        return VWAPResult(
            vwap=vwap,
            price_vs_vwap=price_vs_vwap,
            curving_up=curving_up,
            curving_down=curving_down,
            bands_upper=bands_upper,
            bands_lower=bands_lower
        )

    def _cumsum_with_reset(
        self,
        series: pd.Series,
        groups: np.ndarray
    ) -> pd.Series:
        """
        Calculate cumulative sum that resets at group boundaries.

        Args:
            series: Values to cumsum
            groups: Group labels (resets on change)

        Returns:
            Cumulative sum with resets
        """
        # Create a DataFrame for grouped cumsum
        temp_df = pd.DataFrame({
            'value': series.values,
            'group': groups
        }, index=series.index)

        # Cumulative sum within each group
        return temp_df.groupby('group')['value'].cumsum()

    def calculate_single(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> tuple:
        """
        Calculate VWAP for the most recent value only.

        Args:
            highs: Array of high prices
            lows: Array of low prices
            closes: Array of close prices
            volumes: Array of volumes
            timestamps: Optional array of timestamps for daily reset

        Returns:
            Tuple of (vwap_value, price_vs_vwap, is_curving_up, is_curving_down)
        """
        # Build DataFrame
        df = pd.DataFrame({
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })

        if timestamps is not None:
            df.index = pd.DatetimeIndex(timestamps)

        result = self.calculate(df)

        return (
            result.vwap.iloc[-1],
            result.price_vs_vwap.iloc[-1],
            result.curving_up.iloc[-1],
            result.curving_down.iloc[-1]
        )

    def is_curving_up(
        self,
        vwap_current: float,
        vwap_prev: float,
        vwap_prev2: float
    ) -> bool:
        """
        Check if VWAP is curving up (forming a bottom).

        The VWAP is curving up when:
        - It was falling (vwap_prev < vwap_prev2)
        - And now rising or flat (vwap_current >= vwap_prev)

        Args:
            vwap_current: Current VWAP value
            vwap_prev: Previous VWAP value
            vwap_prev2: VWAP value 2 bars ago

        Returns:
            True if VWAP is curving up
        """
        was_falling = vwap_prev < vwap_prev2
        now_rising = vwap_current >= vwap_prev
        return was_falling and now_rising

    def is_curving_down(
        self,
        vwap_current: float,
        vwap_prev: float,
        vwap_prev2: float
    ) -> bool:
        """
        Check if VWAP is curving down (forming a top).

        The VWAP is curving down when:
        - It was rising (vwap_prev > vwap_prev2)
        - And now falling or flat (vwap_current <= vwap_prev)

        Args:
            vwap_current: Current VWAP value
            vwap_prev: Previous VWAP value
            vwap_prev2: VWAP value 2 bars ago

        Returns:
            True if VWAP is curving down
        """
        was_rising = vwap_prev > vwap_prev2
        now_falling = vwap_current <= vwap_prev
        return was_rising and now_falling

    def is_price_above_vwap(self, close: float, vwap: float) -> bool:
        """Check if price is above VWAP (bullish)."""
        return close > vwap

    def is_price_below_vwap(self, close: float, vwap: float) -> bool:
        """Check if price is below VWAP (bearish)."""
        return close < vwap

    def get_vwap_zone(
        self,
        close: float,
        vwap: float,
        upper_band: float,
        lower_band: float
    ) -> str:
        """
        Determine which VWAP zone the price is in.

        Args:
            close: Current close price
            vwap: Current VWAP value
            upper_band: Upper standard deviation band
            lower_band: Lower standard deviation band

        Returns:
            Zone name: "above_upper", "above_vwap", "below_vwap", "below_lower"
        """
        if close >= upper_band:
            return "above_upper"  # Overbought
        elif close >= vwap:
            return "above_vwap"   # Bullish
        elif close >= lower_band:
            return "below_vwap"   # Bearish
        else:
            return "below_lower"  # Oversold


def calculate_vwap(
    df: pd.DataFrame,
    reset_period: str = "daily",
    band_multiplier: float = 2.0
) -> pd.Series:
    """
    Convenience function to calculate VWAP values.

    Args:
        df: DataFrame with OHLCV data
        reset_period: Reset period ("daily" or "session")
        band_multiplier: Standard deviation multiplier for bands

    Returns:
        Series of VWAP values
    """
    calculator = VWAPCalculator(reset_period, band_multiplier)
    result = calculator.calculate(df)
    return result.vwap
