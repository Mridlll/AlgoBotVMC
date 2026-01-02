"""
Market Regime Detector for VMC Trading Bot.

Price-based regime detection using ADX, ATR%, and Bollinger Band width.
Filters trades based on market conditions - strategy is mean-reversion
that works best in RANGING markets (98% of profits from ranging regime).

Usage:
    detector = RegimeDetector()
    can_trade, multiplier = detector.check_regime_filter(df, timestamp)
"""

import pandas as pd
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, List
from datetime import datetime


class MarketRegime(str, Enum):
    """Market regime classification based on price action."""
    RANGING = "ranging"              # Low volatility, mean-reverting - BEST for strategy
    TRENDING_UP = "trending_up"      # Strong uptrend - reduce or skip
    TRENDING_DOWN = "trending_down"  # Strong downtrend - reduce or skip
    CHOPPY = "choppy"                # High volatility, no direction - skip
    TRANSITIONAL = "transitional"    # Between regimes - reduce size


class RegimeAction(str, Enum):
    """Recommended trading action based on regime."""
    FULL = "full"       # Full position size - ranging market (ideal)
    REDUCE = "reduce"   # Reduced position size - weak trend/transition
    SKIP = "skip"       # Skip trading - strong trend or chop


@dataclass
class RegimeResult:
    """Result of regime detection analysis."""
    regime: MarketRegime
    action: RegimeAction
    confidence: float           # 0-1 confidence in classification
    position_multiplier: float  # 0.0 (skip) to 1.0 (full)

    # Underlying metrics for logging/debugging
    adx: float                  # Average Directional Index (trend strength)
    atr_percent: float          # ATR as % of price (volatility)
    bb_width: float             # Bollinger Band width as % of price
    trend_direction: float      # +1 = up, -1 = down, 0 = neutral

    timestamp: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "regime": self.regime.value,
            "action": self.action.value,
            "confidence": round(self.confidence, 3),
            "position_multiplier": round(self.position_multiplier, 2),
            "adx": round(self.adx, 2),
            "atr_percent": round(self.atr_percent, 4),
            "bb_width": round(self.bb_width, 4),
            "trend_direction": round(self.trend_direction, 2),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


class RegimeDetector:
    """
    Price-based market regime detector for VMC Trading Bot.

    Detects market conditions and recommends trading actions:
    - RANGING: Low ADX (<25), moderate volatility -> FULL SIZE (best for mean-reversion)
    - TRENDING: High ADX (>25) + directional move -> SKIP or REDUCE
    - CHOPPY: High volatility, no direction -> SKIP
    - TRANSITIONAL: Between regimes -> Continue with reduced size

    Based on analysis showing 98% of strategy profits come from ranging markets.
    """

    def __init__(
        self,
        # ADX parameters (trend strength)
        adx_period: int = 14,
        adx_trending_threshold: float = 25.0,     # Above = trending
        adx_strong_trend_threshold: float = 40.0,  # Above = strong trend (skip)

        # ATR parameters (volatility)
        atr_period: int = 14,
        atr_high_threshold: float = 0.03,   # >3% = high volatility
        atr_low_threshold: float = 0.01,    # <1% = low volatility (ideal)

        # Bollinger Band parameters (ranging detection)
        bb_period: int = 20,
        bb_std: float = 2.0,
        bb_narrow_threshold: float = 0.03,  # <3% width = narrow (ranging)
        bb_wide_threshold: float = 0.08,    # >8% width = wide (volatile)

        # Trend direction parameters
        trend_lookback: int = 20,
        trend_threshold: float = 0.02,  # >2% move = trending

        # Hysteresis to prevent whipsawing
        min_regime_bars: int = 5,

        # Position sizing when reducing
        reduced_position_multiplier: float = 0.5
    ):
        self.adx_period = adx_period
        self.adx_trending_threshold = adx_trending_threshold
        self.adx_strong_trend_threshold = adx_strong_trend_threshold

        self.atr_period = atr_period
        self.atr_high_threshold = atr_high_threshold
        self.atr_low_threshold = atr_low_threshold

        self.bb_period = bb_period
        self.bb_std = bb_std
        self.bb_narrow_threshold = bb_narrow_threshold
        self.bb_wide_threshold = bb_wide_threshold

        self.trend_lookback = trend_lookback
        self.trend_threshold = trend_threshold

        self.min_regime_bars = min_regime_bars
        self.reduced_position_multiplier = reduced_position_multiplier

        # State tracking
        self._current_regime = MarketRegime.TRANSITIONAL
        self._bars_in_regime = 0
        self._regime_history: List[Tuple[datetime, RegimeResult]] = []

    def reset(self) -> None:
        """Reset detector state."""
        self._current_regime = MarketRegime.TRANSITIONAL
        self._bars_in_regime = 0
        self._regime_history = []

    def detect(
        self,
        df: pd.DataFrame,
        timestamp: Optional[datetime] = None
    ) -> RegimeResult:
        """
        Detect current market regime from OHLC data.

        Args:
            df: DataFrame with OHLC data (requires 'open', 'high', 'low', 'close')
            timestamp: Current bar timestamp for logging

        Returns:
            RegimeResult with regime classification and trading recommendation
        """
        # Calculate indicators
        adx = self._calculate_adx(df)
        atr_percent = self._calculate_atr_percent(df)
        bb_width = self._calculate_bb_width(df)
        trend_direction = self._calculate_trend_direction(df)

        # Classify regime
        raw_regime, confidence = self._classify_regime(
            adx, atr_percent, bb_width, trend_direction
        )

        # Apply hysteresis
        final_regime = self._apply_hysteresis(raw_regime)

        # Determine action and position multiplier
        action, multiplier = self._get_action(final_regime, confidence)

        # Build result
        result = RegimeResult(
            regime=final_regime,
            action=action,
            confidence=confidence,
            position_multiplier=multiplier,
            adx=adx,
            atr_percent=atr_percent,
            bb_width=bb_width,
            trend_direction=trend_direction,
            timestamp=timestamp
        )

        # Update state
        if final_regime != self._current_regime:
            self._current_regime = final_regime
            self._bars_in_regime = 0
        else:
            self._bars_in_regime += 1

        if timestamp:
            self._regime_history.append((timestamp, result))

        return result

    def check_regime_filter(
        self,
        df: pd.DataFrame,
        timestamp: Optional[datetime] = None
    ) -> Tuple[bool, float]:
        """
        Check regime filter - follows _check_time_filter() pattern.

        Args:
            df: DataFrame with OHLC data
            timestamp: Current bar timestamp

        Returns:
            Tuple of (can_trade, position_multiplier):
            - (True, 1.0) = ranging market, full position
            - (True, 0.5) = transitional, reduced position
            - (False, 0.0) = trending/choppy, skip trade
        """
        result = self.detect(df, timestamp)

        if result.action == RegimeAction.SKIP:
            return (False, 0.0)
        elif result.action == RegimeAction.REDUCE:
            return (True, result.position_multiplier)
        else:  # FULL
            return (True, 1.0)

    def _calculate_adx(self, df: pd.DataFrame) -> float:
        """Calculate Average Directional Index (trend strength)."""
        if len(df) < self.adx_period + 1:
            return 0.0

        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.adx_period).mean()

        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        # Smoothed DM
        plus_di = 100 * (plus_dm.rolling(window=self.adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=self.adx_period).mean() / atr)

        # ADX
        di_sum = plus_di + minus_di
        di_sum = di_sum.replace(0, 1)  # Avoid division by zero
        dx = 100 * abs(plus_di - minus_di) / di_sum
        adx = dx.rolling(window=self.adx_period).mean()

        return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0.0

    def _calculate_atr_percent(self, df: pd.DataFrame) -> float:
        """Calculate ATR as percentage of price (normalized volatility)."""
        if len(df) < self.atr_period + 1:
            return 0.0

        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()

        current_atr = atr.iloc[-1]
        current_price = close.iloc[-1]

        if pd.isna(current_atr) or current_price <= 0:
            return 0.0

        return current_atr / current_price

    def _calculate_bb_width(self, df: pd.DataFrame) -> float:
        """Calculate Bollinger Band width as percentage of price."""
        if len(df) < self.bb_period:
            return 0.0

        close = df['close']
        sma = close.rolling(window=self.bb_period).mean()
        std = close.rolling(window=self.bb_period).std()

        upper_band = sma + (std * self.bb_std)
        lower_band = sma - (std * self.bb_std)

        current_sma = sma.iloc[-1]
        if pd.isna(current_sma) or current_sma <= 0:
            return 0.0

        width = (upper_band.iloc[-1] - lower_band.iloc[-1]) / current_sma

        return width if not pd.isna(width) else 0.0

    def _calculate_trend_direction(self, df: pd.DataFrame) -> float:
        """
        Calculate trend direction.

        Returns:
            +1 for uptrend, -1 for downtrend, 0 for neutral
        """
        if len(df) < self.trend_lookback:
            return 0.0

        close = df['close']
        lookback_price = close.iloc[-self.trend_lookback]
        current_price = close.iloc[-1]

        if lookback_price <= 0:
            return 0.0

        price_change = (current_price / lookback_price) - 1

        if price_change > self.trend_threshold:
            return 1.0
        elif price_change < -self.trend_threshold:
            return -1.0
        else:
            return 0.0

    def _classify_regime(
        self,
        adx: float,
        atr_percent: float,
        bb_width: float,
        trend_direction: float
    ) -> Tuple[MarketRegime, float]:
        """
        Classify market regime based on indicators.

        Returns:
            Tuple of (regime, confidence)
        """
        # Strong trend detection (highest priority - SKIP)
        if adx > self.adx_strong_trend_threshold:
            if trend_direction > 0:
                return MarketRegime.TRENDING_UP, min(1.0, adx / 50)
            elif trend_direction < 0:
                return MarketRegime.TRENDING_DOWN, min(1.0, adx / 50)
            # Strong ADX but no clear direction - choppy
            return MarketRegime.CHOPPY, min(1.0, adx / 50)

        # High volatility chop (SKIP)
        if atr_percent > self.atr_high_threshold and bb_width > self.bb_wide_threshold:
            if abs(trend_direction) < 0.5:  # No clear direction
                return MarketRegime.CHOPPY, min(1.0, atr_percent / 0.05)

        # Moderate trend (REDUCE)
        if adx > self.adx_trending_threshold:
            if trend_direction > 0:
                return MarketRegime.TRENDING_UP, min(1.0, adx / 40)
            elif trend_direction < 0:
                return MarketRegime.TRENDING_DOWN, min(1.0, adx / 40)

        # RANGING detection (BEST for strategy - FULL)
        if adx < self.adx_trending_threshold and bb_width < self.bb_narrow_threshold:
            # Low trend strength + narrow bands = ideal ranging
            confidence = 1.0 - (adx / self.adx_trending_threshold)
            return MarketRegime.RANGING, min(1.0, confidence)

        # Low volatility ranging
        if atr_percent < self.atr_low_threshold and adx < self.adx_trending_threshold:
            return MarketRegime.RANGING, 0.8

        # Moderate conditions - ranging but not ideal
        if adx < self.adx_trending_threshold:
            return MarketRegime.RANGING, 0.6

        # Default: transitional
        return MarketRegime.TRANSITIONAL, 0.5

    def _apply_hysteresis(self, raw_regime: MarketRegime) -> MarketRegime:
        """Apply hysteresis to prevent rapid regime switching."""
        # If we just started, accept raw regime
        if self._bars_in_regime == 0:
            return raw_regime

        # If haven't been in current regime long enough, stay in it
        if self._bars_in_regime < self.min_regime_bars:
            # Exception: if raw regime is more favorable, switch faster
            if raw_regime == MarketRegime.RANGING and self._current_regime != MarketRegime.RANGING:
                # Allow faster switch TO ranging (since that's where we want to trade)
                if self._bars_in_regime >= 2:
                    return raw_regime
            return self._current_regime

        return raw_regime

    def _get_action(
        self,
        regime: MarketRegime,
        confidence: float
    ) -> Tuple[RegimeAction, float]:
        """
        Determine trading action based on regime.

        Returns:
            Tuple of (action, position_multiplier)
        """
        if regime == MarketRegime.RANGING:
            return RegimeAction.FULL, 1.0

        elif regime == MarketRegime.TRENDING_UP:
            if confidence > 0.8:  # Strong trend
                return RegimeAction.SKIP, 0.0
            else:  # Moderate trend
                return RegimeAction.REDUCE, self.reduced_position_multiplier

        elif regime == MarketRegime.TRENDING_DOWN:
            if confidence > 0.8:
                return RegimeAction.SKIP, 0.0
            else:
                return RegimeAction.REDUCE, self.reduced_position_multiplier

        elif regime == MarketRegime.CHOPPY:
            return RegimeAction.SKIP, 0.0

        else:  # TRANSITIONAL
            return RegimeAction.REDUCE, self.reduced_position_multiplier

    def get_regime_stats(self) -> dict:
        """Get statistics about regime distribution."""
        if not self._regime_history:
            return {
                "total_bars": 0,
                "regime_distribution": {},
                "regime_changes": 0
            }

        regime_counts = {}
        prev_regime = None
        regime_changes = 0

        for _, result in self._regime_history:
            regime_name = result.regime.value
            regime_counts[regime_name] = regime_counts.get(regime_name, 0) + 1

            if prev_regime is not None and result.regime != prev_regime:
                regime_changes += 1
            prev_regime = result.regime

        total = len(self._regime_history)
        return {
            "total_bars": total,
            "regime_distribution": {
                k: round(v / total * 100, 1)
                for k, v in regime_counts.items()
            },
            "regime_changes": regime_changes,
            "avg_bars_per_regime": round(total / max(regime_changes, 1), 1)
        }
