"""Multi-Timeframe Divergence Scanner.

Scans 8 standard timeframes for divergences and aggregates signals
with higher timeframe confirmation.

Timeframe Weights (higher TF = stronger signal):
- 5m:  0.15
- 15m: 0.25
- 30m: 0.35
- 1h:  0.50
- 4h:  0.70
- 8h:  0.80
- 12h: 0.90
- 1D:  1.00
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Import from parent modules
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from indicators import (
    WaveTrend, MoneyFlow, VWAPCalculator,
    DivergenceDetector, Divergence
)
from indicators.heikin_ashi import convert_to_heikin_ashi


class Bias(str, Enum):
    """Market bias direction."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


# Standard timeframes to scan
STANDARD_TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h", "8h", "12h", "1D"]

# Timeframe weights (higher = stronger signal)
TIMEFRAME_WEIGHTS = {
    "5m": 0.15,
    "15m": 0.25,
    "30m": 0.35,
    "1h": 0.50,
    "4h": 0.70,
    "8h": 0.80,
    "12h": 0.90,
    "1D": 1.00,
}

# HTF confirmation timeframes
HTF_CONFIRMATION_TFS = ["4h", "8h", "12h", "1D"]


@dataclass
class MTFDivergenceSignal:
    """A divergence signal from a specific timeframe."""
    timeframe: str
    divergence_type: str  # "bullish_regular" or "bearish_regular"
    strength: float       # Divergence strength (0-1)
    weight: float         # Timeframe weight
    weighted_strength: float  # strength * weight
    htf_confirmed: bool   # Whether HTF confirms the signal
    start_bar: int
    end_bar: int
    price_swing1: float
    price_swing2: float
    ind_swing1: float
    ind_swing2: float
    timestamp: Optional[datetime] = None


@dataclass
class MTFScanResult:
    """Results from multi-timeframe divergence scan."""
    signals: List[MTFDivergenceSignal]
    strongest_signal: Optional[MTFDivergenceSignal]
    overall_bias: Bias
    confidence: float     # Weighted average confidence
    htf_bias: Bias        # What HTF indicators suggest
    bullish_count: int    # Number of bullish divergences
    bearish_count: int    # Number of bearish divergences
    timestamp: datetime = field(default_factory=datetime.now)


class MTFDivergenceScanner:
    """
    Multi-Timeframe Divergence Scanner.

    Scans multiple timeframes for divergences between price and WT2 indicator,
    weights signals by timeframe, and confirms with HTF VWAP/MFI curving.

    Algorithm:
    1. For each timeframe:
       a. Convert to Heikin Ashi
       b. Calculate WaveTrend, MFI, VWAP
       c. Detect divergences (price vs WT2)
       d. Record signals with TF weight

    2. For HTF confirmation:
       a. Check if VWAP is curving in signal direction
       b. Check if MFI is curving in signal direction
       c. Signal is "confirmed" if either HTF indicator aligns

    3. Aggregate:
       a. Calculate weighted strength for each signal
       b. Determine overall bias based on weighted votes
       c. Return strongest signal with HTF confirmation

    Usage:
        scanner = MTFDivergenceScanner()
        result = scanner.scan(tf_data_dict)

        if result.strongest_signal and result.strongest_signal.htf_confirmed:
            # High confidence signal
            if result.overall_bias == Bias.BULLISH:
                # Consider long entry
                pass
    """

    def __init__(
        self,
        timeframes: Optional[List[str]] = None,
        divergence_lookback: int = 5,
        min_swing_distance: int = 3,
        require_htf_confirmation: bool = True
    ):
        """
        Initialize the scanner.

        Args:
            timeframes: List of timeframes to scan (defaults to STANDARD_TIMEFRAMES)
            divergence_lookback: Bars on each side for swing detection
            min_swing_distance: Minimum bars between swings
            require_htf_confirmation: Only return signals with HTF confirmation
        """
        self.timeframes = timeframes or STANDARD_TIMEFRAMES
        self.divergence_lookback = divergence_lookback
        self.min_swing_distance = min_swing_distance
        self.require_htf_confirmation = require_htf_confirmation

        # Initialize indicators
        self.wavetrend = WaveTrend()
        self.money_flow = MoneyFlow()
        self.vwap_calc = VWAPCalculator()
        self.divergence_detector = DivergenceDetector(
            lookback=divergence_lookback,
            min_swing_distance=min_swing_distance
        )

    def scan(
        self,
        tf_data: Dict[str, pd.DataFrame],
        lookback_bars: int = 20
    ) -> MTFScanResult:
        """
        Scan all timeframes for divergences.

        Args:
            tf_data: Dict of timeframe -> OHLCV DataFrame
            lookback_bars: How many bars back to look for recent divergences

        Returns:
            MTFScanResult with aggregated signals
        """
        signals: List[MTFDivergenceSignal] = []
        htf_indicators = {}  # Store HTF indicator results

        # First pass: Calculate all indicators and store HTF data
        tf_results = {}
        for tf in self.timeframes:
            if tf not in tf_data:
                continue

            df = tf_data[tf]
            if len(df) < 50:  # Need minimum data
                continue

            # Convert to Heikin Ashi for indicators
            ha_df = convert_to_heikin_ashi(df)

            # Calculate indicators
            wt_result = self.wavetrend.calculate(ha_df)
            mf_result = self.money_flow.calculate(ha_df)
            vwap_result = self.vwap_calc.calculate(df)

            tf_results[tf] = {
                'df': df,
                'ha_df': ha_df,
                'wt': wt_result,
                'mf': mf_result,
                'vwap': vwap_result
            }

            # Store HTF indicators for confirmation
            if tf in HTF_CONFIRMATION_TFS:
                htf_indicators[tf] = {
                    'vwap_curving_up': vwap_result.curving_up.iloc[-1] if len(vwap_result.curving_up) > 0 else False,
                    'vwap_curving_down': vwap_result.curving_down.iloc[-1] if len(vwap_result.curving_down) > 0 else False,
                    'mfi_curving_up': mf_result.curving_up.iloc[-1] if len(mf_result.curving_up) > 0 else False,
                    'mfi_curving_down': mf_result.curving_down.iloc[-1] if len(mf_result.curving_down) > 0 else False,
                }

        # Calculate HTF bias
        htf_bias = self._calculate_htf_bias(htf_indicators)

        # Second pass: Detect divergences and create signals
        for tf, data in tf_results.items():
            ha_df = data['ha_df']
            wt_result = data['wt']

            # Detect divergences using HA close vs WT2
            div_result = self.divergence_detector.detect(
                price=ha_df['close'],
                indicator=wt_result.wt2
            )

            # Get recent divergences
            min_idx = len(ha_df) - lookback_bars
            for div in div_result.divergences:
                if div.end_idx < min_idx:
                    continue  # Skip old divergences

                # Check if this divergence is confirmed by HTF
                is_bullish = div.type == "bullish_regular"
                htf_confirmed = self._check_htf_confirmation(
                    htf_indicators, is_bullish
                )

                weight = TIMEFRAME_WEIGHTS.get(tf, 0.5)
                weighted_strength = div.strength * weight

                # Get timestamp if available
                ts = None
                if isinstance(ha_df.index, pd.DatetimeIndex) and div.end_idx < len(ha_df):
                    ts = ha_df.index[div.end_idx]

                signal = MTFDivergenceSignal(
                    timeframe=tf,
                    divergence_type=div.type,
                    strength=div.strength,
                    weight=weight,
                    weighted_strength=weighted_strength,
                    htf_confirmed=htf_confirmed,
                    start_bar=div.start_idx,
                    end_bar=div.end_idx,
                    price_swing1=div.price_swing1,
                    price_swing2=div.price_swing2,
                    ind_swing1=div.ind_swing1,
                    ind_swing2=div.ind_swing2,
                    timestamp=ts
                )
                signals.append(signal)

        # Filter by HTF confirmation if required
        if self.require_htf_confirmation:
            confirmed_signals = [s for s in signals if s.htf_confirmed]
        else:
            confirmed_signals = signals

        # Calculate aggregates
        bullish_count = sum(1 for s in signals if "bullish" in s.divergence_type)
        bearish_count = sum(1 for s in signals if "bearish" in s.divergence_type)

        # Find strongest signal
        strongest = None
        if confirmed_signals:
            strongest = max(confirmed_signals, key=lambda s: s.weighted_strength)

        # Calculate overall bias
        if not signals:
            overall_bias = Bias.NEUTRAL
            confidence = 0.0
        else:
            bullish_weight = sum(s.weighted_strength for s in signals if "bullish" in s.divergence_type)
            bearish_weight = sum(s.weighted_strength for s in signals if "bearish" in s.divergence_type)

            if bullish_weight > bearish_weight * 1.2:
                overall_bias = Bias.BULLISH
                confidence = bullish_weight / (bullish_weight + bearish_weight) if (bullish_weight + bearish_weight) > 0 else 0
            elif bearish_weight > bullish_weight * 1.2:
                overall_bias = Bias.BEARISH
                confidence = bearish_weight / (bullish_weight + bearish_weight) if (bullish_weight + bearish_weight) > 0 else 0
            else:
                overall_bias = Bias.NEUTRAL
                confidence = 0.5

        return MTFScanResult(
            signals=signals,
            strongest_signal=strongest,
            overall_bias=overall_bias,
            confidence=confidence,
            htf_bias=htf_bias,
            bullish_count=bullish_count,
            bearish_count=bearish_count
        )

    def _calculate_htf_bias(
        self,
        htf_indicators: Dict[str, Dict[str, bool]]
    ) -> Bias:
        """Calculate overall HTF bias based on VWAP/MFI curving."""
        bullish_votes = 0
        bearish_votes = 0

        for tf, indicators in htf_indicators.items():
            weight = TIMEFRAME_WEIGHTS.get(tf, 0.5)

            # VWAP curving
            if indicators.get('vwap_curving_up'):
                bullish_votes += weight
            elif indicators.get('vwap_curving_down'):
                bearish_votes += weight

            # MFI curving
            if indicators.get('mfi_curving_up'):
                bullish_votes += weight * 0.5  # MFI has less weight than VWAP
            elif indicators.get('mfi_curving_down'):
                bearish_votes += weight * 0.5

        if bullish_votes > bearish_votes * 1.2:
            return Bias.BULLISH
        elif bearish_votes > bullish_votes * 1.2:
            return Bias.BEARISH
        else:
            return Bias.NEUTRAL

    def _check_htf_confirmation(
        self,
        htf_indicators: Dict[str, Dict[str, bool]],
        is_bullish: bool
    ) -> bool:
        """
        Check if HTF indicators confirm the divergence signal.

        Args:
            htf_indicators: Dict of HTF indicator states
            is_bullish: Whether checking for bullish confirmation

        Returns:
            True if at least one HTF indicator confirms
        """
        if not htf_indicators:
            return False  # No HTF data available

        for tf, indicators in htf_indicators.items():
            if is_bullish:
                # For bullish signal, check if VWAP or MFI is curving up
                if indicators.get('vwap_curving_up') or indicators.get('mfi_curving_up'):
                    return True
            else:
                # For bearish signal, check if VWAP or MFI is curving down
                if indicators.get('vwap_curving_down') or indicators.get('mfi_curving_down'):
                    return True

        return False

    def get_signal_summary(self, result: MTFScanResult) -> str:
        """Get a human-readable summary of the scan result."""
        lines = [
            f"MTF Divergence Scan Summary:",
            f"  Overall Bias: {result.overall_bias.value.upper()}",
            f"  Confidence: {result.confidence:.1%}",
            f"  HTF Bias: {result.htf_bias.value.upper()}",
            f"  Bullish Divergences: {result.bullish_count}",
            f"  Bearish Divergences: {result.bearish_count}",
        ]

        if result.strongest_signal:
            s = result.strongest_signal
            lines.append(f"")
            lines.append(f"  Strongest Signal:")
            lines.append(f"    Timeframe: {s.timeframe}")
            lines.append(f"    Type: {s.divergence_type}")
            lines.append(f"    Strength: {s.strength:.2f}")
            lines.append(f"    Weighted Strength: {s.weighted_strength:.2f}")
            lines.append(f"    HTF Confirmed: {s.htf_confirmed}")

        return "\n".join(lines)


def scan_for_divergences(
    tf_data: Dict[str, pd.DataFrame],
    require_htf_confirmation: bool = True
) -> MTFScanResult:
    """
    Convenience function to scan for divergences.

    Args:
        tf_data: Dict of timeframe -> OHLCV DataFrame
        require_htf_confirmation: Only return signals with HTF confirmation

    Returns:
        MTFScanResult with aggregated signals
    """
    scanner = MTFDivergenceScanner(require_htf_confirmation=require_htf_confirmation)
    return scanner.scan(tf_data)
