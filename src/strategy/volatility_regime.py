"""
Volatility Regime Detector for Adaptive Mode Selection.

Analyzes market conditions using WT2 range to determine:
- TRENDING: High volatility, wide WT2 swings -> Use ENHANCED mode (stricter filters)
- RANGING: Low volatility, tight WT2 range -> Use SIMPLE mode (more signals)
- TRANSITIONAL: Medium volatility -> Continue current mode (avoid whipsawing)
"""

import pandas as pd
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, List
from datetime import datetime


class VolatilityRegime(str, Enum):
    """Market volatility regime classification."""
    TRENDING = "trending"
    RANGING = "ranging"
    TRANSITIONAL = "transitional"


@dataclass
class RegimeAnalysis:
    """Result of volatility regime analysis."""
    regime: VolatilityRegime
    wt2_range: float
    wt2_std: float
    recommended_mode: str  # "simple", "enhanced", or "continue"
    confidence: float  # 0-1 confidence in regime classification
    timestamp: Optional[datetime] = None

    def to_dict(self):
        return {
            "regime": self.regime.value,
            "wt2_range": self.wt2_range,
            "wt2_std": self.wt2_std,
            "recommended_mode": self.recommended_mode,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


class VolatilityRegimeDetector:
    """
    Detect market volatility regime for adaptive mode selection.

    Strategy:
    - Analyze WT2 oscillator range over lookback period
    - High range (>trending_threshold) = Trending market = Use ENHANCED mode
    - Low range (<ranging_threshold) = Ranging market = Use SIMPLE mode
    - Medium range = Transitional = Continue current mode

    ENHANCED mode is better for trending:
    - Stricter 4-step confirmation reduces false signals
    - Captures larger moves with better R:R

    SIMPLE mode is better for ranging:
    - More signals to capture small oscillations
    - Quick WT cross entries work well in choppy markets
    """

    def __init__(
        self,
        lookback: int = 20,
        trending_threshold: float = 80,
        ranging_threshold: float = 30,
        min_regime_bars: int = 5,
        use_std_confirmation: bool = True
    ):
        """
        Initialize volatility regime detector.

        Args:
            lookback: Number of bars to analyze for regime detection
            trending_threshold: WT2 range above this = TRENDING
            ranging_threshold: WT2 range below this = RANGING
            min_regime_bars: Minimum bars before allowing regime switch (hysteresis)
            use_std_confirmation: Use standard deviation as secondary confirmation
        """
        self.lookback = lookback
        self.trending_threshold = trending_threshold
        self.ranging_threshold = ranging_threshold
        self.min_regime_bars = min_regime_bars
        self.use_std_confirmation = use_std_confirmation

        # State tracking
        self._current_regime = VolatilityRegime.TRANSITIONAL
        self._bars_in_regime = 0
        self._regime_history: List[Tuple[datetime, VolatilityRegime]] = []

    def reset(self):
        """Reset detector state."""
        self._current_regime = VolatilityRegime.TRANSITIONAL
        self._bars_in_regime = 0
        self._regime_history = []

    def analyze(
        self,
        wt2_series: pd.Series,
        timestamp: Optional[datetime] = None
    ) -> RegimeAnalysis:
        """
        Analyze WT2 series to determine current market regime.

        Args:
            wt2_series: Series of WT2 values (most recent values at end)
            timestamp: Current bar timestamp (optional)

        Returns:
            RegimeAnalysis with regime classification and recommendation
        """
        if len(wt2_series) < self.lookback:
            return RegimeAnalysis(
                regime=VolatilityRegime.TRANSITIONAL,
                wt2_range=0,
                wt2_std=0,
                recommended_mode="continue",
                confidence=0.0,
                timestamp=timestamp
            )

        # Get recent WT2 values
        recent_wt2 = wt2_series.iloc[-self.lookback:]

        # Calculate range (max - min)
        wt2_range = recent_wt2.max() - recent_wt2.min()

        # Calculate standard deviation for confirmation
        wt2_std = recent_wt2.std()

        # Determine raw regime
        raw_regime, confidence = self._classify_regime(wt2_range, wt2_std)

        # Apply hysteresis to prevent whipsawing
        final_regime = self._apply_hysteresis(raw_regime)

        # Determine recommended mode
        if final_regime == VolatilityRegime.TRENDING:
            recommended_mode = "enhanced"
        elif final_regime == VolatilityRegime.RANGING:
            recommended_mode = "simple"
        else:
            recommended_mode = "continue"

        # Update state
        if final_regime != self._current_regime:
            self._current_regime = final_regime
            self._bars_in_regime = 0
        else:
            self._bars_in_regime += 1

        if timestamp:
            self._regime_history.append((timestamp, final_regime))

        return RegimeAnalysis(
            regime=final_regime,
            wt2_range=wt2_range,
            wt2_std=wt2_std,
            recommended_mode=recommended_mode,
            confidence=confidence,
            timestamp=timestamp
        )

    def _classify_regime(
        self,
        wt2_range: float,
        wt2_std: float
    ) -> Tuple[VolatilityRegime, float]:
        """
        Classify regime based on WT2 range and standard deviation.

        Returns:
            Tuple of (regime, confidence)
        """
        # Primary classification based on range
        if wt2_range > self.trending_threshold:
            regime = VolatilityRegime.TRENDING
            # Confidence increases with range
            confidence = min(1.0, wt2_range / (self.trending_threshold * 1.5))
        elif wt2_range < self.ranging_threshold:
            regime = VolatilityRegime.RANGING
            # Confidence increases as range decreases
            confidence = min(1.0, (self.ranging_threshold - wt2_range) / self.ranging_threshold + 0.5)
        else:
            regime = VolatilityRegime.TRANSITIONAL
            # Lower confidence in transitional zone
            confidence = 0.5

        # Secondary confirmation using standard deviation
        if self.use_std_confirmation:
            # High std (>25) suggests trending, low std (<15) suggests ranging
            if wt2_std > 25 and regime != VolatilityRegime.TRENDING:
                regime = VolatilityRegime.TRANSITIONAL
                confidence *= 0.7  # Reduce confidence due to mixed signals
            elif wt2_std < 15 and regime != VolatilityRegime.RANGING:
                regime = VolatilityRegime.TRANSITIONAL
                confidence *= 0.7

        return regime, confidence

    def _apply_hysteresis(self, raw_regime: VolatilityRegime) -> VolatilityRegime:
        """
        Apply hysteresis to prevent rapid regime switching.

        Only switch regime if:
        1. We've been in current regime for min_regime_bars
        2. The raw regime is different from current
        """
        # Always allow initial classification
        if self._bars_in_regime == 0:
            return raw_regime

        # If we haven't been in current regime long enough, don't switch
        if self._bars_in_regime < self.min_regime_bars:
            return self._current_regime

        # Otherwise, allow switch to new regime
        return raw_regime

    def get_mode_for_signal(
        self,
        wt2_series: pd.Series,
        current_mode: str,
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Get the recommended signal mode based on current regime.

        Args:
            wt2_series: Series of WT2 values
            current_mode: Current signal mode ("simple" or "enhanced")
            timestamp: Current bar timestamp

        Returns:
            Recommended mode: "simple", "enhanced", or current_mode if transitional
        """
        analysis = self.analyze(wt2_series, timestamp)

        if analysis.recommended_mode == "continue":
            return current_mode
        return analysis.recommended_mode

    def get_regime_stats(self) -> dict:
        """Get statistics about regime changes."""
        if not self._regime_history:
            return {
                "total_regime_changes": 0,
                "time_in_trending": 0,
                "time_in_ranging": 0,
                "time_in_transitional": 0,
            }

        regime_counts = {
            VolatilityRegime.TRENDING: 0,
            VolatilityRegime.RANGING: 0,
            VolatilityRegime.TRANSITIONAL: 0,
        }

        prev_regime = None
        regime_changes = 0

        for _, regime in self._regime_history:
            regime_counts[regime] += 1
            if prev_regime is not None and regime != prev_regime:
                regime_changes += 1
            prev_regime = regime

        total = len(self._regime_history)
        return {
            "total_regime_changes": regime_changes,
            "time_in_trending": regime_counts[VolatilityRegime.TRENDING] / total * 100,
            "time_in_ranging": regime_counts[VolatilityRegime.RANGING] / total * 100,
            "time_in_transitional": regime_counts[VolatilityRegime.TRANSITIONAL] / total * 100,
        }


def calculate_regime_zones(
    df: pd.DataFrame,
    wt2_series: pd.Series,
    lookback: int = 20,
    trending_threshold: float = 80,
    ranging_threshold: float = 30
) -> pd.DataFrame:
    """
    Calculate regime zones for an entire DataFrame.

    Useful for visualization and analysis.

    Args:
        df: Source DataFrame with price data
        wt2_series: WT2 indicator values
        lookback: Lookback period for regime detection
        trending_threshold: Threshold for trending regime
        ranging_threshold: Threshold for ranging regime

    Returns:
        DataFrame with regime columns added
    """
    detector = VolatilityRegimeDetector(
        lookback=lookback,
        trending_threshold=trending_threshold,
        ranging_threshold=ranging_threshold
    )

    regimes = []
    ranges = []
    modes = []

    for i in range(len(wt2_series)):
        if i < lookback:
            regimes.append("transitional")
            ranges.append(0)
            modes.append("enhanced")
        else:
            analysis = detector.analyze(
                wt2_series.iloc[:i+1],
                timestamp=df.index[i] if hasattr(df, 'index') else None
            )
            regimes.append(analysis.regime.value)
            ranges.append(analysis.wt2_range)
            modes.append(analysis.recommended_mode)

    result = df.copy()
    result['regime'] = regimes
    result['wt2_range'] = ranges
    result['recommended_mode'] = modes

    return result
