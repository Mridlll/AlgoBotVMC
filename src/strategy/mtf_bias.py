"""
Multi-Timeframe Bias Calculator for VMC Strategy.

Based on Discord feedback from atmk:
- Use HTF (4H and higher) to determine market BIAS
- Check MFI direction (red=bearish, green=bullish, curving direction)
- Check WT position (above/below zero)
- This determines if we should look for LONGS or SHORTS on LTF
"""

import pandas as pd
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

from src.strategy.signals import Bias, SignalType


@dataclass
class HTFAnalysis:
    """Analysis results from a single higher timeframe."""
    timeframe: str
    timestamp: datetime
    wt2: float
    mfi: float
    mfi_prev: float
    vwap: float

    # Derived values
    wt_position: str       # "oversold", "overbought", "neutral"
    mfi_direction: str     # "bullish", "bearish", "neutral"
    mfi_curving: str       # "up", "down", "flat"
    bias: Bias


@dataclass
class MTFBiasResult:
    """Combined bias result from all higher timeframes."""
    overall_bias: Bias
    confidence: float           # 0-1 confidence score
    bullish_count: int
    bearish_count: int
    neutral_count: int
    htf_analyses: List[HTFAnalysis]
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_bias": self.overall_bias.value,
            "confidence": self.confidence,
            "bullish_count": self.bullish_count,
            "bearish_count": self.bearish_count,
            "neutral_count": self.neutral_count,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp)
        }


class MTFBiasCalculator:
    """
    Calculate market bias from higher timeframes.

    Strategy (from Discord feedback):
    1. Look at HTF (4H and higher) for money flow direction
       - Green (MFI > 0) = bullish
       - Red (MFI < 0) = bearish
       - Curving up = bullish momentum
       - Curving down = bearish momentum

    2. Look at wave trends position
       - WT2 < 0 (oversold zone) = bullish bias
       - WT2 > 0 (overbought zone) = bearish bias

    3. Combine signals with weighting for final bias
    """

    # Timeframe weights (higher TF = more weight)
    DEFAULT_WEIGHTS = {
        "1d": 4,
        "12h": 3,
        "8h": 2,
        "4h": 1,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, int]] = None,
        mfi_lookback: int = 3,
        wt_neutral_zone: float = 20,  # WT2 between -20 and +20 is neutral
        require_mfi_confirm: bool = True
    ):
        """
        Initialize MTF bias calculator.

        Args:
            weights: Timeframe weights (higher = more important)
            mfi_lookback: Bars to check for MFI curve direction
            wt_neutral_zone: WT2 values within this range are neutral
            require_mfi_confirm: Require MFI to confirm WT direction
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.mfi_lookback = mfi_lookback
        self.wt_neutral_zone = wt_neutral_zone
        self.require_mfi_confirm = require_mfi_confirm

    def analyze_single_timeframe(
        self,
        timeframe: str,
        timestamp: datetime,
        wt2: float,
        mfi: float,
        mfi_prev: float,
        vwap: float = 0.0
    ) -> HTFAnalysis:
        """
        Analyze a single higher timeframe for bias.

        Args:
            timeframe: Timeframe string (e.g., "4h", "1d")
            timestamp: Current bar timestamp
            wt2: WaveTrend 2 value
            mfi: Current Money Flow value
            mfi_prev: Previous Money Flow value
            vwap: VWAP value (optional)

        Returns:
            HTFAnalysis with bias determination
        """
        # Determine WT position
        if wt2 <= -self.wt_neutral_zone:
            wt_position = "oversold"
        elif wt2 >= self.wt_neutral_zone:
            wt_position = "overbought"
        else:
            wt_position = "neutral"

        # Determine MFI direction (color)
        if mfi > 0:
            mfi_direction = "bullish"  # Green
        elif mfi < 0:
            mfi_direction = "bearish"  # Red
        else:
            mfi_direction = "neutral"

        # Determine MFI curve direction
        if mfi > mfi_prev:
            mfi_curving = "up"
        elif mfi < mfi_prev:
            mfi_curving = "down"
        else:
            mfi_curving = "flat"

        # Determine bias for this timeframe
        bias = self._determine_bias(wt_position, mfi_direction, mfi_curving)

        return HTFAnalysis(
            timeframe=timeframe,
            timestamp=timestamp,
            wt2=wt2,
            mfi=mfi,
            mfi_prev=mfi_prev,
            vwap=vwap,
            wt_position=wt_position,
            mfi_direction=mfi_direction,
            mfi_curving=mfi_curving,
            bias=bias
        )

    def _determine_bias(
        self,
        wt_position: str,
        mfi_direction: str,
        mfi_curving: str
    ) -> Bias:
        """
        Determine bias from WT position and MFI direction.

        Logic:
        - Oversold WT + Bullish/Up MFI = BULLISH
        - Overbought WT + Bearish/Down MFI = BEARISH
        - Mixed signals or neutral = NEUTRAL
        """
        # Strong bullish: oversold + green MFI curving up
        if wt_position == "oversold":
            if mfi_direction == "bullish" or mfi_curving == "up":
                return Bias.BULLISH
            elif not self.require_mfi_confirm:
                return Bias.BULLISH  # WT alone is enough

        # Strong bearish: overbought + red MFI curving down
        if wt_position == "overbought":
            if mfi_direction == "bearish" or mfi_curving == "down":
                return Bias.BEARISH
            elif not self.require_mfi_confirm:
                return Bias.BEARISH  # WT alone is enough

        # Neutral zone - check MFI direction only
        if wt_position == "neutral":
            if mfi_direction == "bullish" and mfi_curving == "up":
                return Bias.BULLISH
            elif mfi_direction == "bearish" and mfi_curving == "down":
                return Bias.BEARISH

        return Bias.NEUTRAL

    def calculate_combined_bias(
        self,
        htf_analyses: List[HTFAnalysis]
    ) -> MTFBiasResult:
        """
        Calculate combined bias from multiple HTF analyses using weighted voting.

        Args:
            htf_analyses: List of HTFAnalysis from different timeframes

        Returns:
            MTFBiasResult with overall bias and confidence
        """
        if not htf_analyses:
            return MTFBiasResult(
                overall_bias=Bias.NEUTRAL,
                confidence=0.0,
                bullish_count=0,
                bearish_count=0,
                neutral_count=0,
                htf_analyses=[],
                timestamp=datetime.now()
            )

        bullish_score = 0
        bearish_score = 0
        neutral_count = 0
        total_weight = 0

        for analysis in htf_analyses:
            weight = self.weights.get(analysis.timeframe, 1)
            total_weight += weight

            if analysis.bias == Bias.BULLISH:
                bullish_score += weight
            elif analysis.bias == Bias.BEARISH:
                bearish_score += weight
            else:
                neutral_count += 1

        # Determine overall bias
        if bullish_score > bearish_score:
            overall_bias = Bias.BULLISH
            confidence = bullish_score / total_weight if total_weight > 0 else 0
        elif bearish_score > bullish_score:
            overall_bias = Bias.BEARISH
            confidence = bearish_score / total_weight if total_weight > 0 else 0
        else:
            overall_bias = Bias.NEUTRAL
            confidence = 0.5  # Equal scores

        # Count biases
        bullish_count = sum(1 for a in htf_analyses if a.bias == Bias.BULLISH)
        bearish_count = sum(1 for a in htf_analyses if a.bias == Bias.BEARISH)

        return MTFBiasResult(
            overall_bias=overall_bias,
            confidence=confidence,
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            neutral_count=neutral_count,
            htf_analyses=htf_analyses,
            timestamp=htf_analyses[0].timestamp if htf_analyses else datetime.now()
        )

    def should_take_signal(
        self,
        signal_type: SignalType,
        bias_result: MTFBiasResult,
        min_confidence: float = 0.5
    ) -> Tuple[bool, str]:
        """
        Determine if a signal should be taken based on HTF bias.

        Args:
            signal_type: The signal type (LONG or SHORT)
            bias_result: The calculated HTF bias
            min_confidence: Minimum confidence required

        Returns:
            Tuple of (should_take, reason)
        """
        # Check confidence
        if bias_result.confidence < min_confidence:
            return False, f"Low confidence ({bias_result.confidence:.1%} < {min_confidence:.1%})"

        # Check alignment
        if signal_type == SignalType.LONG:
            if bias_result.overall_bias == Bias.BULLISH:
                return True, f"LONG aligned with BULLISH bias ({bias_result.confidence:.1%})"
            elif bias_result.overall_bias == Bias.BEARISH:
                return False, "LONG conflicts with BEARISH HTF bias"
            else:
                return False, "LONG skipped - HTF bias is NEUTRAL"

        elif signal_type == SignalType.SHORT:
            if bias_result.overall_bias == Bias.BEARISH:
                return True, f"SHORT aligned with BEARISH bias ({bias_result.confidence:.1%})"
            elif bias_result.overall_bias == Bias.BULLISH:
                return False, "SHORT conflicts with BULLISH HTF bias"
            else:
                return False, "SHORT skipped - HTF bias is NEUTRAL"

        return False, "Unknown signal type"


def is_mfi_curving_up(mfi_series: pd.Series, lookback: int = 3) -> bool:
    """Check if MFI is curving up (was falling, now rising)."""
    if len(mfi_series) < lookback + 1:
        return False

    current = mfi_series.iloc[-1]
    prev = mfi_series.iloc[-2]
    prev2 = mfi_series.iloc[-3] if len(mfi_series) > 2 else prev

    was_falling = prev < prev2
    now_rising = current > prev

    return was_falling and now_rising


def is_mfi_curving_down(mfi_series: pd.Series, lookback: int = 3) -> bool:
    """Check if MFI is curving down (was rising, now falling)."""
    if len(mfi_series) < lookback + 1:
        return False

    current = mfi_series.iloc[-1]
    prev = mfi_series.iloc[-2]
    prev2 = mfi_series.iloc[-3] if len(mfi_series) > 2 else prev

    was_rising = prev > prev2
    now_falling = current < prev

    return was_rising and now_falling


def is_vwap_curving_up(vwap_series: pd.Series, lookback: int = 3) -> bool:
    """Check if VWAP is curving up."""
    if len(vwap_series) < lookback + 1:
        return False

    current = vwap_series.iloc[-1]
    prev = vwap_series.iloc[-2]
    prev2 = vwap_series.iloc[-3] if len(vwap_series) > 2 else prev

    was_falling = prev < prev2
    now_rising = current > prev

    return was_falling and now_rising


def is_vwap_curving_down(vwap_series: pd.Series, lookback: int = 3) -> bool:
    """Check if VWAP is curving down."""
    if len(vwap_series) < lookback + 1:
        return False

    current = vwap_series.iloc[-1]
    prev = vwap_series.iloc[-2]
    prev2 = vwap_series.iloc[-3] if len(vwap_series) > 2 else prev

    was_rising = prev > prev2
    now_falling = current < prev

    return was_rising and now_falling
