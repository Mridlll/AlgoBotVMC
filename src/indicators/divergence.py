"""Divergence Detection between Price and Indicators.

Detects regular divergences which signal potential reversals:
- Bullish Regular: Price makes lower low, indicator makes higher low (potential reversal UP)
- Bearish Regular: Price makes higher high, indicator makes lower high (potential reversal DOWN)

Hidden divergences (continuation patterns) are NOT included per user preference.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class SwingPoint:
    """Represents a swing high or low point."""
    index: int           # Bar index in the series
    price: float         # Price value at this point
    indicator: float     # Indicator value at this point
    timestamp: pd.Timestamp  # Timestamp of the bar


@dataclass
class Divergence:
    """Represents a detected divergence."""
    type: str            # "bullish_regular" or "bearish_regular"
    start_idx: int       # Index of first swing point
    end_idx: int         # Index of second swing point
    strength: float      # Divergence strength (0-1)
    price_swing1: float  # First price swing point
    price_swing2: float  # Second price swing point
    ind_swing1: float    # First indicator swing point
    ind_swing2: float    # Second indicator swing point


@dataclass
class DivergenceResult:
    """Results from divergence detection."""
    bullish_regular: pd.Series   # Boolean mask where bullish divergence detected
    bearish_regular: pd.Series   # Boolean mask where bearish divergence detected
    divergences: List[Divergence]  # List of all detected divergences
    strength: pd.Series          # Divergence strength at each bar (0 if none)


class DivergenceDetector:
    """
    Detects divergences between price and an oscillator indicator.

    Divergence occurs when price and indicator move in opposite directions,
    signaling a potential reversal.

    Algorithm:
    1. Find swing highs/lows in price using local extrema detection
    2. Find swing highs/lows in indicator at same locations
    3. Compare consecutive swing points:
       - Bullish: Price LL (lower low) + Indicator HL (higher low)
       - Bearish: Price HH (higher high) + Indicator LH (lower high)

    Usage:
        detector = DivergenceDetector(lookback=5, min_swing_distance=3)
        result = detector.detect(price_series, indicator_series, timestamps)

        if result.bullish_regular.iloc[-1]:
            # Bullish divergence detected at current bar
            pass
    """

    def __init__(
        self,
        lookback: int = 5,
        min_swing_distance: int = 3,
        max_swing_distance: int = 50,
        min_price_diff_pct: float = 0.1,
        min_indicator_diff_pct: float = 5.0
    ):
        """
        Initialize divergence detector.

        Args:
            lookback: Bars on each side to confirm a swing point
            min_swing_distance: Minimum bars between swing points for valid divergence
            max_swing_distance: Maximum bars between swing points
            min_price_diff_pct: Minimum % difference in price swings to be significant
            min_indicator_diff_pct: Minimum % difference in indicator swings
        """
        self.lookback = lookback
        self.min_swing_distance = min_swing_distance
        self.max_swing_distance = max_swing_distance
        self.min_price_diff_pct = min_price_diff_pct
        self.min_indicator_diff_pct = min_indicator_diff_pct

    def detect(
        self,
        price: pd.Series,
        indicator: pd.Series,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> DivergenceResult:
        """
        Detect divergences between price and indicator.

        Args:
            price: Price series (typically close prices or HA close)
            indicator: Oscillator series (WT2, MFI, etc.)
            timestamps: Optional timestamps for the data

        Returns:
            DivergenceResult with boolean masks and divergence details
        """
        if timestamps is None:
            timestamps = price.index if isinstance(price.index, pd.DatetimeIndex) else None

        n = len(price)

        # Initialize result arrays
        bullish = pd.Series(False, index=price.index)
        bearish = pd.Series(False, index=price.index)
        strength = pd.Series(0.0, index=price.index)
        divergences: List[Divergence] = []

        # Find swing lows for bullish divergence
        swing_lows = self._find_swing_lows(price, indicator, timestamps)

        # Find swing highs for bearish divergence
        swing_highs = self._find_swing_highs(price, indicator, timestamps)

        # Detect bullish regular divergences (price LL, indicator HL)
        for i in range(1, len(swing_lows)):
            prev_swing = swing_lows[i - 1]
            curr_swing = swing_lows[i]

            # Check distance constraints
            distance = curr_swing.index - prev_swing.index
            if distance < self.min_swing_distance or distance > self.max_swing_distance:
                continue

            # Check for divergence: Price Lower Low, Indicator Higher Low
            price_lower = curr_swing.price < prev_swing.price
            indicator_higher = curr_swing.indicator > prev_swing.indicator

            if price_lower and indicator_higher:
                # Calculate strength
                price_diff_pct = abs(curr_swing.price - prev_swing.price) / prev_swing.price * 100
                indicator_diff = abs(curr_swing.indicator - prev_swing.indicator)

                # Skip if changes are too small
                if price_diff_pct < self.min_price_diff_pct:
                    continue

                # Strength based on magnitude of divergence
                strength_val = min(1.0, (price_diff_pct / 5.0) * (indicator_diff / 10.0))

                div = Divergence(
                    type="bullish_regular",
                    start_idx=prev_swing.index,
                    end_idx=curr_swing.index,
                    strength=strength_val,
                    price_swing1=prev_swing.price,
                    price_swing2=curr_swing.price,
                    ind_swing1=prev_swing.indicator,
                    ind_swing2=curr_swing.indicator
                )
                divergences.append(div)

                # Mark the end bar as having divergence
                bullish.iloc[curr_swing.index] = True
                strength.iloc[curr_swing.index] = max(strength.iloc[curr_swing.index], strength_val)

        # Detect bearish regular divergences (price HH, indicator LH)
        for i in range(1, len(swing_highs)):
            prev_swing = swing_highs[i - 1]
            curr_swing = swing_highs[i]

            # Check distance constraints
            distance = curr_swing.index - prev_swing.index
            if distance < self.min_swing_distance or distance > self.max_swing_distance:
                continue

            # Check for divergence: Price Higher High, Indicator Lower High
            price_higher = curr_swing.price > prev_swing.price
            indicator_lower = curr_swing.indicator < prev_swing.indicator

            if price_higher and indicator_lower:
                # Calculate strength
                price_diff_pct = abs(curr_swing.price - prev_swing.price) / prev_swing.price * 100
                indicator_diff = abs(curr_swing.indicator - prev_swing.indicator)

                # Skip if changes are too small
                if price_diff_pct < self.min_price_diff_pct:
                    continue

                # Strength based on magnitude of divergence
                strength_val = min(1.0, (price_diff_pct / 5.0) * (indicator_diff / 10.0))

                div = Divergence(
                    type="bearish_regular",
                    start_idx=prev_swing.index,
                    end_idx=curr_swing.index,
                    strength=strength_val,
                    price_swing1=prev_swing.price,
                    price_swing2=curr_swing.price,
                    ind_swing1=prev_swing.indicator,
                    ind_swing2=curr_swing.indicator
                )
                divergences.append(div)

                # Mark the end bar as having divergence
                bearish.iloc[curr_swing.index] = True
                strength.iloc[curr_swing.index] = max(strength.iloc[curr_swing.index], strength_val)

        return DivergenceResult(
            bullish_regular=bullish,
            bearish_regular=bearish,
            divergences=divergences,
            strength=strength
        )

    def _find_swing_lows(
        self,
        price: pd.Series,
        indicator: pd.Series,
        timestamps: Optional[pd.DatetimeIndex]
    ) -> List[SwingPoint]:
        """Find swing low points (local minima) in price."""
        swing_lows: List[SwingPoint] = []
        n = len(price)

        for i in range(self.lookback, n - self.lookback):
            # Check if this is a swing low in price
            window_start = i - self.lookback
            window_end = i + self.lookback + 1

            window_prices = price.iloc[window_start:window_end]
            current_price = price.iloc[i]

            # Is this the minimum in the window?
            if current_price == window_prices.min():
                ts = timestamps[i] if timestamps is not None else None
                swing_lows.append(SwingPoint(
                    index=i,
                    price=current_price,
                    indicator=indicator.iloc[i],
                    timestamp=ts
                ))

        return swing_lows

    def _find_swing_highs(
        self,
        price: pd.Series,
        indicator: pd.Series,
        timestamps: Optional[pd.DatetimeIndex]
    ) -> List[SwingPoint]:
        """Find swing high points (local maxima) in price."""
        swing_highs: List[SwingPoint] = []
        n = len(price)

        for i in range(self.lookback, n - self.lookback):
            # Check if this is a swing high in price
            window_start = i - self.lookback
            window_end = i + self.lookback + 1

            window_prices = price.iloc[window_start:window_end]
            current_price = price.iloc[i]

            # Is this the maximum in the window?
            if current_price == window_prices.max():
                ts = timestamps[i] if timestamps is not None else None
                swing_highs.append(SwingPoint(
                    index=i,
                    price=current_price,
                    indicator=indicator.iloc[i],
                    timestamp=ts
                ))

        return swing_highs

    def detect_single(
        self,
        price: pd.Series,
        indicator: pd.Series,
        check_bullish: bool = True,
        check_bearish: bool = True
    ) -> Tuple[bool, bool, float]:
        """
        Check for divergence at the most recent bar.

        Args:
            price: Price series
            indicator: Indicator series
            check_bullish: Whether to check for bullish divergence
            check_bearish: Whether to check for bearish divergence

        Returns:
            Tuple of (is_bullish, is_bearish, strength)
        """
        result = self.detect(price, indicator)

        is_bullish = check_bullish and result.bullish_regular.iloc[-1]
        is_bearish = check_bearish and result.bearish_regular.iloc[-1]
        strength_val = result.strength.iloc[-1]

        return is_bullish, is_bearish, strength_val

    def get_recent_divergence(
        self,
        price: pd.Series,
        indicator: pd.Series,
        lookback_bars: int = 10
    ) -> Optional[Divergence]:
        """
        Get the most recent divergence within lookback window.

        Args:
            price: Price series
            indicator: Indicator series
            lookback_bars: How many bars back to look for divergence

        Returns:
            Most recent Divergence or None
        """
        result = self.detect(price, indicator)

        if not result.divergences:
            return None

        # Filter to recent divergences
        min_idx = len(price) - lookback_bars
        recent = [d for d in result.divergences if d.end_idx >= min_idx]

        if not recent:
            return None

        # Return the most recent one
        return max(recent, key=lambda d: d.end_idx)


def detect_divergence(
    price: pd.Series,
    indicator: pd.Series,
    lookback: int = 5,
    min_swing_distance: int = 3
) -> DivergenceResult:
    """
    Convenience function to detect divergences.

    Args:
        price: Price series
        indicator: Indicator series (WT2, MFI, etc.)
        lookback: Bars on each side for swing detection
        min_swing_distance: Minimum bars between swings

    Returns:
        DivergenceResult with detection results
    """
    detector = DivergenceDetector(lookback=lookback, min_swing_distance=min_swing_distance)
    return detector.detect(price, indicator)
