"""Multi-timeframe signal coordination for VMC strategy."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import pandas as pd

from strategy.signals import Signal, SignalType, SignalDetector
from indicators import WaveTrend, MoneyFlow, HeikinAshi
from indicators.wavetrend import WaveTrendResult
from indicators.money_flow import MoneyFlowResult


class BiasDirection(str, Enum):
    """Higher timeframe bias direction."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class TimeframeSignal:
    """Signal from a specific timeframe."""
    timeframe: str
    signal: Optional[Signal]
    wt2: float  # Current WT2 value for bias calculation
    is_entry_tf: bool  # True if this is an entry timeframe
    signal_strength: float = 1.0  # Confidence/strength of signal


@dataclass
class BiasResult:
    """Result of HTF bias calculation."""
    direction: BiasDirection
    confidence: float  # 0-1 confidence score
    bullish_score: float
    bearish_score: float
    timeframe_biases: Dict[str, str]  # {timeframe: "bullish"/"bearish"}
    aligned_count: int
    total_count: int


@dataclass
class MultiTimeframeSignal:
    """Combined signal from multiple timeframes."""
    entry_signals: List[TimeframeSignal]
    bias_result: BiasResult
    primary_signal: Optional[Signal]  # The main entry signal to execute
    is_aligned: bool  # Whether entry aligns with HTF bias
    alignment_percent: float  # Percentage of bias TFs aligned
    overall_confidence: float  # Combined confidence score
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_signal": self.primary_signal.to_dict() if self.primary_signal else None,
            "bias_direction": self.bias_result.direction.value,
            "bias_confidence": self.bias_result.confidence,
            "is_aligned": self.is_aligned,
            "alignment_percent": self.alignment_percent,
            "overall_confidence": self.overall_confidence,
            "entry_timeframes": [s.timeframe for s in self.entry_signals if s.signal],
            "timeframe_biases": self.bias_result.timeframe_biases,
        }


class MultiTimeframeCoordinator:
    """
    Coordinates signal detection across multiple timeframes.

    Entry timeframes (scalping): 3m, 5m, 10m, 15m - for VMC entry signals
    Bias timeframes (HTF): 4h, 8h, 12h, 1D - for trend direction confirmation

    Bias Hierarchy: 1D > 12h > 8h > 4h (higher timeframes have more weight)
    Bias is determined by WT2 position: WT2 < 0 = bullish zone, WT2 > 0 = bearish zone
    """

    # Default bias weights (higher = more important)
    DEFAULT_BIAS_WEIGHTS = {
        "1d": 4,
        "12h": 3,
        "8h": 2,
        "4h": 1,
    }

    def __init__(
        self,
        entry_timeframes: List[str],
        bias_timeframes: List[str],
        bias_weights: Optional[Dict[str, int]] = None,
        require_bias_alignment: bool = True,
        min_bias_aligned: int = 2,
        entry_on_any_timeframe: bool = True,
        anchor_level_long: float = -60,
        anchor_level_short: float = 60,
    ):
        """
        Initialize multi-timeframe coordinator.

        Args:
            entry_timeframes: List of entry timeframes (e.g., ["5m", "15m"])
            bias_timeframes: List of bias timeframes (e.g., ["4h", "12h", "1d"])
            bias_weights: Optional custom weights for bias calculation
            require_bias_alignment: Whether to require HTF alignment
            min_bias_aligned: Minimum bias TFs that must agree
            entry_on_any_timeframe: True=signal on ANY entry TF, False=ALL must agree
            anchor_level_long: WT2 level for long anchor
            anchor_level_short: WT2 level for short anchor
        """
        self.entry_timeframes = [tf.lower() for tf in entry_timeframes]
        self.bias_timeframes = [tf.lower() for tf in bias_timeframes]
        self.bias_weights = bias_weights or self.DEFAULT_BIAS_WEIGHTS
        self.require_bias_alignment = require_bias_alignment
        self.min_bias_aligned = min_bias_aligned
        self.entry_on_any_timeframe = entry_on_any_timeframe

        # Create signal detectors for each entry timeframe
        self.detectors: Dict[str, SignalDetector] = {}
        for tf in self.entry_timeframes:
            self.detectors[tf] = SignalDetector(
                anchor_level_long=anchor_level_long,
                anchor_level_short=anchor_level_short,
            )

        # Indicator calculators
        self.wavetrend = WaveTrend()
        self.money_flow = MoneyFlow()
        self.heikin_ashi = HeikinAshi()

    def reset(self) -> None:
        """Reset all detectors."""
        for detector in self.detectors.values():
            detector.reset()

    def calculate_bias(
        self,
        htf_data: Dict[str, Tuple[WaveTrendResult, MoneyFlowResult]]
    ) -> BiasResult:
        """
        Calculate overall bias from higher timeframe indicators.

        Bias is determined by WT2 position:
        - WT2 < 0: Bullish zone (price likely to go up)
        - WT2 > 0: Bearish zone (price likely to go down)

        Hierarchy: 1D > 12h > 8h > 4h (higher timeframes carry more weight)

        Args:
            htf_data: Dict mapping timeframe to (WaveTrendResult, MoneyFlowResult)

        Returns:
            BiasResult with direction and confidence
        """
        bullish_score = 0.0
        bearish_score = 0.0
        total_weight = 0
        timeframe_biases = {}

        for tf, (wt_result, _) in htf_data.items():
            weight = self.bias_weights.get(tf.lower(), 1)
            total_weight += weight

            # Get current WT2 value
            wt2 = wt_result.wt2.iloc[-1] if len(wt_result.wt2) > 0 else 0

            # WT2 < 0 = bullish zone, WT2 > 0 = bearish zone
            if wt2 < 0:
                bullish_score += weight
                timeframe_biases[tf] = "bullish"
            else:
                bearish_score += weight
                timeframe_biases[tf] = "bearish"

        # Determine direction
        if total_weight == 0:
            return BiasResult(
                direction=BiasDirection.NEUTRAL,
                confidence=0.0,
                bullish_score=0,
                bearish_score=0,
                timeframe_biases=timeframe_biases,
                aligned_count=0,
                total_count=0,
            )

        # Count aligned timeframes
        bullish_count = sum(1 for bias in timeframe_biases.values() if bias == "bullish")
        bearish_count = len(timeframe_biases) - bullish_count

        if bullish_score > bearish_score:
            confidence = bullish_score / total_weight
            direction = BiasDirection.BULLISH
            aligned_count = bullish_count
        elif bearish_score > bullish_score:
            confidence = bearish_score / total_weight
            direction = BiasDirection.BEARISH
            aligned_count = bearish_count
        else:
            # Split decision = neutral
            direction = BiasDirection.NEUTRAL
            confidence = 0.5
            aligned_count = 0

        return BiasResult(
            direction=direction,
            confidence=confidence,
            bullish_score=bullish_score,
            bearish_score=bearish_score,
            timeframe_biases=timeframe_biases,
            aligned_count=aligned_count,
            total_count=len(timeframe_biases),
        )

    def process_candles(
        self,
        entry_candles: Dict[str, pd.DataFrame],
        bias_candles: Dict[str, pd.DataFrame],
    ) -> Optional[MultiTimeframeSignal]:
        """
        Process candles from all timeframes and generate signals.

        Args:
            entry_candles: Dict mapping entry timeframe to DataFrame with OHLCV
            bias_candles: Dict mapping bias timeframe to DataFrame with OHLCV

        Returns:
            MultiTimeframeSignal if conditions are met, None otherwise
        """
        # Calculate indicators for bias timeframes
        htf_data: Dict[str, Tuple[WaveTrendResult, MoneyFlowResult]] = {}
        for tf, df in bias_candles.items():
            if df is None or df.empty:
                continue
            ha_df = self.heikin_ashi.convert(df)
            wt_result = self.wavetrend.calculate(ha_df)
            mf_result = self.money_flow.calculate(df)
            htf_data[tf] = (wt_result, mf_result)

        # Calculate bias
        bias_result = self.calculate_bias(htf_data)

        # Process entry timeframes
        entry_signals: List[TimeframeSignal] = []
        for tf, df in entry_candles.items():
            if df is None or df.empty:
                continue

            # Get or create detector for this timeframe
            detector = self.detectors.get(tf)
            if detector is None:
                detector = SignalDetector()
                self.detectors[tf] = detector

            # Calculate indicators
            ha_df = self.heikin_ashi.convert(df)
            wt_result = self.wavetrend.calculate(ha_df)
            mf_result = self.money_flow.calculate(df)

            # Get latest bar data
            timestamp = df.index[-1] if hasattr(df.index, '__getitem__') else datetime.now()
            close_price = df['close'].iloc[-1]
            wt2 = wt_result.wt2.iloc[-1]

            # Process through signal detector
            signal = detector.process_bar(
                timestamp=timestamp,
                close_price=close_price,
                wt_result=wt_result,
                mf_result=mf_result,
            )

            entry_signals.append(TimeframeSignal(
                timeframe=tf,
                signal=signal,
                wt2=wt2,
                is_entry_tf=True,
                signal_strength=1.0 if signal else 0.0,
            ))

        # Determine primary signal
        signals_with_entry = [s for s in entry_signals if s.signal is not None]

        if not signals_with_entry:
            return None

        # Select primary signal based on entry_on_any_timeframe setting
        if self.entry_on_any_timeframe:
            # Use first available signal
            primary_signal = signals_with_entry[0].signal
        else:
            # Require all entry TFs to have signals (not typically used)
            if len(signals_with_entry) != len(entry_signals):
                return None
            primary_signal = signals_with_entry[0].signal

        # Check alignment
        is_aligned = self._check_alignment(primary_signal.signal_type, bias_result)
        alignment_percent = bias_result.aligned_count / bias_result.total_count * 100 if bias_result.total_count > 0 else 0

        # Calculate overall confidence
        overall_confidence = self._calculate_confidence(
            primary_signal=primary_signal,
            bias_result=bias_result,
            is_aligned=is_aligned,
        )

        # If alignment is required but not met, return None
        if self.require_bias_alignment and not is_aligned:
            return None

        return MultiTimeframeSignal(
            entry_signals=entry_signals,
            bias_result=bias_result,
            primary_signal=primary_signal,
            is_aligned=is_aligned,
            alignment_percent=alignment_percent,
            overall_confidence=overall_confidence,
            metadata={
                "entry_tf": signals_with_entry[0].timeframe,
                "bias_direction": bias_result.direction.value,
            }
        )

    def _check_alignment(
        self,
        signal_type: SignalType,
        bias_result: BiasResult
    ) -> bool:
        """
        Check if entry signal aligns with HTF bias.

        Args:
            signal_type: LONG or SHORT
            bias_result: Bias calculation result

        Returns:
            True if aligned (or bias is neutral)
        """
        # Check minimum aligned requirement
        if bias_result.aligned_count < self.min_bias_aligned:
            return False

        if bias_result.direction == BiasDirection.NEUTRAL:
            return False  # No entries when neutral

        if signal_type == SignalType.LONG:
            return bias_result.direction == BiasDirection.BULLISH
        else:
            return bias_result.direction == BiasDirection.BEARISH

    def _calculate_confidence(
        self,
        primary_signal: Signal,
        bias_result: BiasResult,
        is_aligned: bool,
    ) -> float:
        """
        Calculate overall confidence score for the signal.

        Factors:
        - Bias confidence (how strong is HTF agreement)
        - Alignment (signal matches bias)
        - Signal quality (anchor/trigger levels)

        Args:
            primary_signal: The entry signal
            bias_result: Bias calculation result
            is_aligned: Whether signal aligns with bias

        Returns:
            Confidence score 0-1
        """
        if not is_aligned:
            return 0.0

        # Start with bias confidence (50% weight)
        confidence = bias_result.confidence * 0.5

        # Add alignment bonus (30% weight)
        alignment_score = bias_result.aligned_count / max(bias_result.total_count, 1)
        confidence += alignment_score * 0.3

        # Signal quality (20% weight)
        # Better anchor levels = higher confidence
        if primary_signal.signal_type == SignalType.LONG:
            anchor_quality = min(abs(primary_signal.anchor_wave.wt2_value) / 70, 1.0)
        else:
            anchor_quality = min(abs(primary_signal.anchor_wave.wt2_value) / 70, 1.0)
        confidence += anchor_quality * 0.2

        return min(confidence, 1.0)

    def get_state(self) -> Dict[str, Any]:
        """Get current state of all detectors."""
        return {
            tf: detector.get_current_state()
            for tf, detector in self.detectors.items()
        }
