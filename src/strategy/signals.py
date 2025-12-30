"""Signal detection state machine for VMC strategy."""

import pandas as pd
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime

from indicators import WaveTrend, MoneyFlow, HeikinAshi
from indicators.wavetrend import WaveTrendResult
from indicators.money_flow import MoneyFlowResult

# Import SignalMode from config to avoid duplication
from config.config import SignalMode


class SignalType(str, Enum):
    """Type of trading signal."""
    LONG = "long"
    SHORT = "short"


class Bias(str, Enum):
    """Market bias from higher timeframe analysis."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SignalState(Enum):
    """State machine states for signal detection."""
    IDLE = auto()                    # No setup detected
    ANCHOR_DETECTED = auto()         # Anchor wave found
    TRIGGER_DETECTED = auto()        # Trigger wave found
    AWAITING_MFI = auto()            # Waiting for MFI confirmation
    AWAITING_VWAP = auto()           # Waiting for VWAP cross
    SIGNAL_READY = auto()            # All conditions met


@dataclass
class AnchorWave:
    """Information about detected anchor wave."""
    timestamp: datetime
    wt2_value: float
    bar_index: int
    signal_type: SignalType


@dataclass
class TriggerWave:
    """Information about detected trigger wave."""
    timestamp: datetime
    wt2_value: float
    bar_index: int
    has_cross: bool  # For shorts: green dot (cross down)


@dataclass
class Signal:
    """Complete trading signal with all details."""
    signal_type: SignalType
    timestamp: datetime
    entry_price: float
    anchor_wave: AnchorWave
    trigger_wave: TriggerWave
    wt1: float
    wt2: float
    vwap: float
    mfi: float
    confidence: float = 1.0  # 0-1 confidence score
    timeframe: str = ""  # Timeframe that generated this signal (for MTF mode)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_type": self.signal_type.value,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            "entry_price": self.entry_price,
            "timeframe": self.timeframe,
            "wt1": self.wt1,
            "wt2": self.wt2,
            "vwap": self.vwap,
            "mfi": self.mfi,
            "confidence": self.confidence,
            "anchor_wt2": self.anchor_wave.wt2_value,
            "trigger_wt2": self.trigger_wave.wt2_value,
        }


class SignalDetector:
    """
    State machine for detecting VMC trading signals.

    Long Entry Sequence:
    1. Anchor wave: wt2 < -60
    2. Trigger wave: Subsequent smaller dip in wt2
    3. Money flow (red wave): Curving up
    4. VWAP: Crosses above 0
    5. Execute long

    Short Entry Sequence:
    1. Anchor wave: wt2 > 60
    2. Trigger wave: Subsequent smaller peak + WT cross down
    3. Money flow (green wave): Curving down
    4. VWAP: Crosses below 0
    5. Execute short
    """

    def __init__(
        self,
        anchor_level_long: float = -60,
        anchor_level_short: float = 60,
        trigger_lookback: int = 20,
        mfi_lookback: int = 3,
        timeframe: str = ""
    ):
        """
        Initialize signal detector.

        Args:
            anchor_level_long: WT2 level for long anchor (default -60)
            anchor_level_short: WT2 level for short anchor (default 60)
            trigger_lookback: Bars to look for trigger after anchor
            mfi_lookback: Bars to check for MFI curve
            timeframe: Timeframe this detector is for (for MTF mode)
        """
        self.anchor_level_long = anchor_level_long
        self.anchor_level_short = anchor_level_short
        self.trigger_lookback = trigger_lookback
        self.mfi_lookback = mfi_lookback
        self.timeframe = timeframe

        # State tracking
        self._long_state = SignalState.IDLE
        self._short_state = SignalState.IDLE
        self._long_anchor: Optional[AnchorWave] = None
        self._short_anchor: Optional[AnchorWave] = None
        self._long_trigger: Optional[TriggerWave] = None
        self._short_trigger: Optional[TriggerWave] = None

        # History tracking
        self._bar_index = 0
        self._signals_history: List[Signal] = []

    def reset(self) -> None:
        """Reset all state."""
        self._long_state = SignalState.IDLE
        self._short_state = SignalState.IDLE
        self._long_anchor = None
        self._short_anchor = None
        self._long_trigger = None
        self._short_trigger = None
        self._bar_index = 0

    def process_bar(
        self,
        timestamp: datetime,
        close_price: float,
        wt_result: WaveTrendResult,
        mf_result: MoneyFlowResult,
        bar_idx: Optional[int] = None
    ) -> Optional[Signal]:
        """
        Process a single bar and check for signals.

        Args:
            timestamp: Bar timestamp
            close_price: Close price for entry
            wt_result: WaveTrend calculation results
            mf_result: MoneyFlow calculation results
            bar_idx: Optional bar index (uses internal counter if None)

        Returns:
            Signal object if signal is ready, None otherwise
        """
        if bar_idx is not None:
            self._bar_index = bar_idx
        else:
            self._bar_index += 1

        idx = self._bar_index

        # Get current values
        wt1 = wt_result.wt1.iloc[-1]
        wt2 = wt_result.wt2.iloc[-1]
        vwap = wt_result.vwap.iloc[-1]
        mfi = mf_result.mfi.iloc[-1]

        # Get previous values for cross detection
        vwap_prev = wt_result.vwap.iloc[-2] if len(wt_result.vwap) > 1 else 0
        cross_down = wt_result.cross_down.iloc[-1] if len(wt_result.cross_down) > 0 else False

        # Check for MFI curve using last few bars
        mfi_curving_up = self._is_mfi_curving_up(mf_result.mfi)
        mfi_curving_down = self._is_mfi_curving_down(mf_result.mfi)

        # Process long setup
        long_signal = self._process_long_setup(
            timestamp, close_price, idx,
            wt1, wt2, vwap, vwap_prev, mfi,
            mfi_curving_up
        )
        if long_signal:
            self._signals_history.append(long_signal)
            return long_signal

        # Process short setup
        short_signal = self._process_short_setup(
            timestamp, close_price, idx,
            wt1, wt2, vwap, vwap_prev, mfi,
            mfi_curving_down, cross_down
        )
        if short_signal:
            self._signals_history.append(short_signal)
            return short_signal

        return None

    def _process_long_setup(
        self,
        timestamp: datetime,
        close_price: float,
        idx: int,
        wt1: float,
        wt2: float,
        vwap: float,
        vwap_prev: float,
        mfi: float,
        mfi_curving_up: bool
    ) -> Optional[Signal]:
        """Process long signal state machine."""

        # Step 1: Look for anchor wave
        if self._long_state == SignalState.IDLE:
            if wt2 <= self.anchor_level_long:
                self._long_anchor = AnchorWave(
                    timestamp=timestamp,
                    wt2_value=wt2,
                    bar_index=idx,
                    signal_type=SignalType.LONG
                )
                self._long_state = SignalState.ANCHOR_DETECTED
            return None

        # Step 2: Look for trigger wave (smaller dip)
        if self._long_state == SignalState.ANCHOR_DETECTED:
            # Check if anchor is still valid (not too old)
            if idx - self._long_anchor.bar_index > self.trigger_lookback:
                self._long_state = SignalState.IDLE
                self._long_anchor = None
                return None

            # Look for trigger: wt2 dips again but less extreme than anchor
            if wt2 < 0 and wt2 > self._long_anchor.wt2_value:
                # Found a higher low (trigger wave)
                self._long_trigger = TriggerWave(
                    timestamp=timestamp,
                    wt2_value=wt2,
                    bar_index=idx,
                    has_cross=False
                )
                self._long_state = SignalState.TRIGGER_DETECTED
            return None

        # Step 3: Wait for MFI to curve up (MFI must be negative)
        if self._long_state == SignalState.TRIGGER_DETECTED:
            if mfi_curving_up and mfi < 0:
                self._long_state = SignalState.AWAITING_VWAP
            return None

        # Step 4: Wait for VWAP to cross above 0
        if self._long_state == SignalState.AWAITING_VWAP:
            if vwap > 0 and vwap_prev <= 0:
                # Signal ready!
                signal = Signal(
                    signal_type=SignalType.LONG,
                    timestamp=timestamp,
                    entry_price=close_price,
                    anchor_wave=self._long_anchor,
                    trigger_wave=self._long_trigger,
                    wt1=wt1,
                    wt2=wt2,
                    vwap=vwap,
                    mfi=mfi,
                    timeframe=self.timeframe
                )

                # Reset state
                self._long_state = SignalState.IDLE
                self._long_anchor = None
                self._long_trigger = None

                return signal

        return None

    def _process_short_setup(
        self,
        timestamp: datetime,
        close_price: float,
        idx: int,
        wt1: float,
        wt2: float,
        vwap: float,
        vwap_prev: float,
        mfi: float,
        mfi_curving_down: bool,
        wt_cross_down: bool
    ) -> Optional[Signal]:
        """Process short signal state machine."""

        # Step 1: Look for anchor wave
        if self._short_state == SignalState.IDLE:
            if wt2 >= self.anchor_level_short:
                self._short_anchor = AnchorWave(
                    timestamp=timestamp,
                    wt2_value=wt2,
                    bar_index=idx,
                    signal_type=SignalType.SHORT
                )
                self._short_state = SignalState.ANCHOR_DETECTED
            return None

        # Step 2: Look for trigger wave (smaller peak + cross down)
        if self._short_state == SignalState.ANCHOR_DETECTED:
            # Check if anchor is still valid
            if idx - self._short_anchor.bar_index > self.trigger_lookback:
                self._short_state = SignalState.IDLE
                self._short_anchor = None
                return None

            # Look for trigger: wt2 peaks again but less extreme than anchor
            # Plus need a cross down (green dot in indicator)
            if wt2 > 0 and wt2 < self._short_anchor.wt2_value and wt_cross_down:
                self._short_trigger = TriggerWave(
                    timestamp=timestamp,
                    wt2_value=wt2,
                    bar_index=idx,
                    has_cross=True
                )
                self._short_state = SignalState.TRIGGER_DETECTED
            return None

        # Step 3: Wait for MFI to curve down (MFI must be positive)
        if self._short_state == SignalState.TRIGGER_DETECTED:
            if mfi_curving_down and mfi > 0:
                self._short_state = SignalState.AWAITING_VWAP
            return None

        # Step 4: Wait for VWAP to cross below 0
        if self._short_state == SignalState.AWAITING_VWAP:
            if vwap < 0 and vwap_prev >= 0:
                # Signal ready!
                signal = Signal(
                    signal_type=SignalType.SHORT,
                    timestamp=timestamp,
                    entry_price=close_price,
                    anchor_wave=self._short_anchor,
                    trigger_wave=self._short_trigger,
                    wt1=wt1,
                    wt2=wt2,
                    vwap=vwap,
                    mfi=mfi,
                    timeframe=self.timeframe
                )

                # Reset state
                self._short_state = SignalState.IDLE
                self._short_anchor = None
                self._short_trigger = None

                return signal

        return None

    def _is_mfi_curving_up(self, mfi_series: pd.Series, lookback: int = None) -> bool:
        """Check if MFI is curving up."""
        lookback = lookback or self.mfi_lookback
        if len(mfi_series) < lookback + 1:
            return False

        recent = mfi_series.iloc[-lookback:]
        current = mfi_series.iloc[-1]
        prev = mfi_series.iloc[-2]
        prev2 = mfi_series.iloc[-3] if len(mfi_series) > 2 else prev

        # Was falling, now rising
        was_falling = prev < prev2
        now_rising = current > prev

        return was_falling and now_rising

    def _is_mfi_curving_down(self, mfi_series: pd.Series, lookback: int = None) -> bool:
        """Check if MFI is curving down."""
        lookback = lookback or self.mfi_lookback
        if len(mfi_series) < lookback + 1:
            return False

        recent = mfi_series.iloc[-lookback:]
        current = mfi_series.iloc[-1]
        prev = mfi_series.iloc[-2]
        prev2 = mfi_series.iloc[-3] if len(mfi_series) > 2 else prev

        # Was rising, now falling
        was_rising = prev > prev2
        now_falling = current < prev

        return was_rising and now_falling

    def get_current_state(self) -> Dict[str, Any]:
        """Get current state of both long and short setups."""
        return {
            "long_state": self._long_state.name,
            "short_state": self._short_state.name,
            "long_anchor": self._long_anchor.wt2_value if self._long_anchor else None,
            "short_anchor": self._short_anchor.wt2_value if self._short_anchor else None,
            "long_trigger": self._long_trigger.wt2_value if self._long_trigger else None,
            "short_trigger": self._short_trigger.wt2_value if self._short_trigger else None,
            "bar_index": self._bar_index,
        }

    @property
    def signals_history(self) -> List[Signal]:
        """Get history of all generated signals."""
        return self._signals_history.copy()


class SimpleSignalDetector:
    """
    Simple VMC signal detector - matches original TradingView indicator.

    Original VMC Signal Logic:
    - BUY: WT1 crosses above WT2 while WT2 is in oversold zone (< -53)
    - SELL: WT1 crosses below WT2 while WT2 is in overbought zone (> +53)

    This is much simpler than the enhanced 4-step state machine.
    """

    def __init__(
        self,
        oversold_level: float = -53,
        overbought_level: float = 53,
        timeframe: str = ""
    ):
        """
        Initialize simple signal detector.

        Args:
            oversold_level: WT2 level for buy signals (default -53, original VMC)
            overbought_level: WT2 level for sell signals (default +53, original VMC)
            timeframe: Timeframe identifier for logging
        """
        self.oversold_level = oversold_level
        self.overbought_level = overbought_level
        self.timeframe = timeframe

        # Track previous values for cross detection
        self._prev_wt1: Optional[float] = None
        self._prev_wt2: Optional[float] = None
        self._signals_history: List[Signal] = []

    def reset(self) -> None:
        """Reset detector state."""
        self._prev_wt1 = None
        self._prev_wt2 = None

    def process_bar(
        self,
        timestamp: datetime,
        close_price: float,
        wt1: float,
        wt2: float,
        mfi: float = 0.0,
        vwap: float = 0.0
    ) -> Optional[Signal]:
        """
        Process a single bar and check for simple VMC signals.

        Args:
            timestamp: Bar timestamp
            close_price: Close price for entry
            wt1: WaveTrend 1 (fast line)
            wt2: WaveTrend 2 (slow line)
            mfi: Money Flow (optional, for signal metadata)
            vwap: VWAP (optional, for signal metadata)

        Returns:
            Signal if conditions met, None otherwise
        """
        signal = None

        # Need previous values for cross detection
        if self._prev_wt1 is not None and self._prev_wt2 is not None:
            # Detect WT crosses
            # Note: Uses >= and <= to match Pine Script's wtCrossUp = (wt2 - wt1 <= 0)
            # which includes the equality case (wt1 == wt2)
            cross_up = self._prev_wt1 <= self._prev_wt2 and wt1 >= wt2
            cross_down = self._prev_wt1 >= self._prev_wt2 and wt1 <= wt2

            # BUY: Cross up while oversold
            if cross_up and wt2 <= self.oversold_level:
                signal = Signal(
                    signal_type=SignalType.LONG,
                    timestamp=timestamp,
                    entry_price=close_price,
                    anchor_wave=AnchorWave(
                        timestamp=timestamp,
                        wt2_value=wt2,
                        bar_index=0,
                        signal_type=SignalType.LONG
                    ),
                    trigger_wave=TriggerWave(
                        timestamp=timestamp,
                        wt2_value=wt2,
                        bar_index=0,
                        has_cross=True
                    ),
                    wt1=wt1,
                    wt2=wt2,
                    vwap=vwap,
                    mfi=mfi,
                    timeframe=self.timeframe,
                    metadata={"mode": "simple", "cross_type": "up"}
                )

            # SELL: Cross down while overbought
            elif cross_down and wt2 >= self.overbought_level:
                signal = Signal(
                    signal_type=SignalType.SHORT,
                    timestamp=timestamp,
                    entry_price=close_price,
                    anchor_wave=AnchorWave(
                        timestamp=timestamp,
                        wt2_value=wt2,
                        bar_index=0,
                        signal_type=SignalType.SHORT
                    ),
                    trigger_wave=TriggerWave(
                        timestamp=timestamp,
                        wt2_value=wt2,
                        bar_index=0,
                        has_cross=True
                    ),
                    wt1=wt1,
                    wt2=wt2,
                    vwap=vwap,
                    mfi=mfi,
                    timeframe=self.timeframe,
                    metadata={"mode": "simple", "cross_type": "down"}
                )

        # Store current values for next bar
        self._prev_wt1 = wt1
        self._prev_wt2 = wt2

        if signal:
            self._signals_history.append(signal)

        return signal

    def get_current_state(self) -> Dict[str, Any]:
        """Get current state of the detector."""
        return {
            "mode": "simple",
            "oversold_level": self.oversold_level,
            "overbought_level": self.overbought_level,
            "prev_wt1": self._prev_wt1,
            "prev_wt2": self._prev_wt2,
            "signals_count": len(self._signals_history)
        }

    @property
    def signals_history(self) -> List[Signal]:
        """Get history of all generated signals."""
        return self._signals_history.copy()


@dataclass
class DCASignal:
    """Dollar-cost averaging signal for adding to position."""
    signal_type: SignalType
    timestamp: datetime
    entry_price: float
    reason: str
    position_multiplier: float = 0.5  # Add 50% of original position


class MTFSignalDetector:
    """
    Multi-Timeframe Signal Detector.

    Based on Discord feedback (atmk's strategy):
    1. Use HTF (4H+) to determine BIAS (bullish/bearish)
    2. Use LTF (3-30min) to find entry signals
    3. Only take signals that align with HTF bias
    4. Optional DCA when VWAP curves in direction

    Entry Logic (LTF):
    - First entry: Anchor wave + Trigger wave (like enhanced mode)
    - Second entry (DCA): When VWAP curves in trade direction
    """

    def __init__(
        self,
        anchor_level_long: float = -60,
        anchor_level_short: float = 60,
        trigger_lookback: int = 20,
        mfi_lookback: int = 3,
        enable_dca: bool = True,
        dca_on_vwap_curve: bool = True,
        timeframe: str = ""
    ):
        """
        Initialize MTF signal detector.

        Args:
            anchor_level_long: WT2 level for long anchor
            anchor_level_short: WT2 level for short anchor
            trigger_lookback: Bars to look for trigger after anchor
            mfi_lookback: Bars to check for MFI curve
            enable_dca: Enable DCA entries
            dca_on_vwap_curve: Trigger DCA when VWAP curves
            timeframe: Timeframe identifier
        """
        self.anchor_level_long = anchor_level_long
        self.anchor_level_short = anchor_level_short
        self.trigger_lookback = trigger_lookback
        self.mfi_lookback = mfi_lookback
        self.enable_dca = enable_dca
        self.dca_on_vwap_curve = dca_on_vwap_curve
        self.timeframe = timeframe

        # Internal enhanced detector for LTF signals
        self._enhanced_detector = SignalDetector(
            anchor_level_long=anchor_level_long,
            anchor_level_short=anchor_level_short,
            trigger_lookback=trigger_lookback,
            mfi_lookback=mfi_lookback,
            timeframe=timeframe
        )

        # Track VWAP for DCA
        self._prev_vwap: Optional[float] = None
        self._prev_prev_vwap: Optional[float] = None

        # Track active position for DCA
        self._active_position: Optional[SignalType] = None
        self._dca_taken: bool = False

        self._signals_history: List[Signal] = []
        self._dca_history: List[DCASignal] = []

    def reset(self) -> None:
        """Reset detector state."""
        self._enhanced_detector.reset()
        self._prev_vwap = None
        self._prev_prev_vwap = None
        self._active_position = None
        self._dca_taken = False

    def set_active_position(self, position_type: Optional[SignalType]) -> None:
        """Set the current active position for DCA tracking."""
        self._active_position = position_type
        self._dca_taken = False  # Reset DCA flag on new position

    def clear_position(self) -> None:
        """Clear the active position (when trade closes)."""
        self._active_position = None
        self._dca_taken = False

    def process_bar(
        self,
        timestamp: datetime,
        close_price: float,
        wt_result,  # WaveTrendResult
        mf_result,  # MoneyFlowResult
        htf_bias: Bias,
        bar_idx: Optional[int] = None
    ) -> Optional[Signal]:
        """
        Process a bar for MTF signal detection.

        Args:
            timestamp: Bar timestamp
            close_price: Close price for entry
            wt_result: WaveTrend calculation results
            mf_result: MoneyFlow calculation results
            htf_bias: Bias from higher timeframe analysis
            bar_idx: Optional bar index

        Returns:
            Signal if conditions met and aligned with HTF bias, None otherwise
        """
        # Get LTF signal from enhanced detector
        ltf_signal = self._enhanced_detector.process_bar(
            timestamp=timestamp,
            close_price=close_price,
            wt_result=wt_result,
            mf_result=mf_result,
            bar_idx=bar_idx
        )

        # If we have a signal, check HTF alignment
        if ltf_signal:
            aligned = self._check_htf_alignment(ltf_signal.signal_type, htf_bias)

            if aligned:
                # Add MTF metadata
                ltf_signal.metadata["mode"] = "mtf"
                ltf_signal.metadata["htf_bias"] = htf_bias.value
                ltf_signal.metadata["aligned"] = True
                self._signals_history.append(ltf_signal)
                return ltf_signal
            else:
                # Signal rejected due to HTF bias conflict
                return None

        return None

    def check_dca_opportunity(
        self,
        timestamp: datetime,
        close_price: float,
        vwap: float
    ) -> Optional[DCASignal]:
        """
        Check for DCA opportunity based on VWAP curve.

        Args:
            timestamp: Bar timestamp
            close_price: Current close price
            vwap: Current VWAP value

        Returns:
            DCASignal if DCA conditions met, None otherwise
        """
        if not self.enable_dca or not self.dca_on_vwap_curve:
            return None

        if self._active_position is None or self._dca_taken:
            return None

        dca_signal = None

        # Check VWAP curve
        if self._prev_vwap is not None and self._prev_prev_vwap is not None:
            vwap_curving_up = (
                self._prev_vwap < self._prev_prev_vwap and  # Was falling
                vwap > self._prev_vwap  # Now rising
            )
            vwap_curving_down = (
                self._prev_vwap > self._prev_prev_vwap and  # Was rising
                vwap < self._prev_vwap  # Now falling
            )

            # DCA for longs: VWAP curving up
            if self._active_position == SignalType.LONG and vwap_curving_up:
                dca_signal = DCASignal(
                    signal_type=SignalType.LONG,
                    timestamp=timestamp,
                    entry_price=close_price,
                    reason="VWAP curving up - DCA long",
                    position_multiplier=0.5
                )
                self._dca_taken = True
                self._dca_history.append(dca_signal)

            # DCA for shorts: VWAP curving down
            elif self._active_position == SignalType.SHORT and vwap_curving_down:
                dca_signal = DCASignal(
                    signal_type=SignalType.SHORT,
                    timestamp=timestamp,
                    entry_price=close_price,
                    reason="VWAP curving down - DCA short",
                    position_multiplier=0.5
                )
                self._dca_taken = True
                self._dca_history.append(dca_signal)

        # Update VWAP history
        self._prev_prev_vwap = self._prev_vwap
        self._prev_vwap = vwap

        return dca_signal

    def _check_htf_alignment(self, signal_type: SignalType, htf_bias: Bias) -> bool:
        """
        Check if signal aligns with HTF bias.

        Args:
            signal_type: The signal type (LONG or SHORT)
            htf_bias: The HTF bias

        Returns:
            True if aligned, False otherwise
        """
        if signal_type == SignalType.LONG:
            return htf_bias == Bias.BULLISH
        elif signal_type == SignalType.SHORT:
            return htf_bias == Bias.BEARISH
        return False

    def get_current_state(self) -> Dict[str, Any]:
        """Get current state of the detector."""
        return {
            "mode": "mtf",
            "enhanced_state": self._enhanced_detector.get_current_state(),
            "active_position": self._active_position.value if self._active_position else None,
            "dca_taken": self._dca_taken,
            "signals_count": len(self._signals_history),
            "dca_count": len(self._dca_history)
        }

    @property
    def signals_history(self) -> List[Signal]:
        """Get history of all generated signals."""
        return self._signals_history.copy()

    @property
    def dca_history(self) -> List[DCASignal]:
        """Get history of all DCA signals."""
        return self._dca_history.copy()


@dataclass
class HTFBiasResult:
    """Result of higher timeframe bias analysis."""
    bias: Bias
    wt2_value: float
    mfi_value: float
    mfi_curving_up: bool
    mfi_curving_down: bool
    confidence: float  # 0-1, how strong the bias signal is
    timestamp: datetime


def determine_htf_bias(
    wt2: float,
    mfi: float,
    prev_mfi: float,
    prev_prev_mfi: float,
    timestamp: datetime,
    wt2_threshold: float = 30.0,
    require_mfi_confirm: bool = True
) -> HTFBiasResult:
    """
    Determine market bias from higher timeframe indicators.

    V4 HTF Bias Logic:
    - BULLISH: WT2 < -threshold AND (MFI curving up OR no MFI confirmation required)
    - BEARISH: WT2 > +threshold AND (MFI curving down OR no MFI confirmation required)
    - NEUTRAL: Neither condition met

    Args:
        wt2: Current WT2 value from HTF
        mfi: Current MFI value from HTF
        prev_mfi: Previous MFI value
        prev_prev_mfi: MFI value 2 bars ago (for curve detection)
        timestamp: Current bar timestamp
        wt2_threshold: WT2 level for bias (default 30)
        require_mfi_confirm: Whether to require MFI confirmation

    Returns:
        HTFBiasResult with bias direction and confidence
    """
    # Detect MFI curve direction
    mfi_curving_up = prev_mfi < prev_prev_mfi and mfi > prev_mfi  # Was falling, now rising
    mfi_curving_down = prev_mfi > prev_prev_mfi and mfi < prev_mfi  # Was rising, now falling

    # Calculate confidence based on how extreme WT2 is
    if wt2 < -wt2_threshold:
        wt2_strength = min(abs(wt2) / 60, 1.0)  # Normalize to 0-1
    elif wt2 > wt2_threshold:
        wt2_strength = min(wt2 / 60, 1.0)
    else:
        wt2_strength = 0.0

    # Determine bias
    bias = Bias.NEUTRAL
    confidence = 0.0

    # BULLISH: WT2 in oversold zone + optionally MFI curving up
    if wt2 < -wt2_threshold:
        if require_mfi_confirm:
            if mfi_curving_up:
                bias = Bias.BULLISH
                confidence = wt2_strength * 1.0  # Full confidence with MFI confirm
            else:
                bias = Bias.BULLISH
                confidence = wt2_strength * 0.6  # Reduced confidence without MFI
        else:
            bias = Bias.BULLISH
            confidence = wt2_strength

    # BEARISH: WT2 in overbought zone + optionally MFI curving down
    elif wt2 > wt2_threshold:
        if require_mfi_confirm:
            if mfi_curving_down:
                bias = Bias.BEARISH
                confidence = wt2_strength * 1.0
            else:
                bias = Bias.BEARISH
                confidence = wt2_strength * 0.6
        else:
            bias = Bias.BEARISH
            confidence = wt2_strength

    return HTFBiasResult(
        bias=bias,
        wt2_value=wt2,
        mfi_value=mfi,
        mfi_curving_up=mfi_curving_up,
        mfi_curving_down=mfi_curving_down,
        confidence=confidence,
        timestamp=timestamp
    )


class V4SignalDetector:
    """
    V4 Multi-Timeframe Signal Detector with Direction Filtering.

    Uses 4H timeframe for bias determination and 1H for entry signals.
    Supports long-only, short-only, or both directions.

    Entry Logic:
    1. Get HTF (4H) bias: BULLISH, BEARISH, or NEUTRAL
    2. On LTF (1H), look for ENHANCED or SIMPLE signals
    3. Filter signals based on HTF bias and direction_filter setting
    """

    def __init__(
        self,
        signal_mode: SignalMode = SignalMode.ENHANCED,
        anchor_level_long: float = -60,
        anchor_level_short: float = 60,
        trigger_lookback: int = 20,
        simple_oversold: float = -53,
        simple_overbought: float = 53,
        direction_filter: str = "both",  # "both", "long_only", "short_only"
        bias_wt2_threshold: float = 30.0,
        require_mfi_confirm: bool = True,
        allow_neutral_trades: bool = False,
        timeframe: str = "1h"
    ):
        """
        Initialize V4 signal detector.

        Args:
            signal_mode: LTF signal detection mode (SIMPLE or ENHANCED)
            anchor_level_long: WT2 level for long anchor (ENHANCED mode)
            anchor_level_short: WT2 level for short anchor (ENHANCED mode)
            trigger_lookback: Bars to look for trigger after anchor
            simple_oversold: WT2 level for buy signals (SIMPLE mode)
            simple_overbought: WT2 level for sell signals (SIMPLE mode)
            direction_filter: Filter trades by direction
            bias_wt2_threshold: WT2 threshold for HTF bias
            require_mfi_confirm: Require MFI confirmation for HTF bias
            allow_neutral_trades: Allow trades when HTF bias is neutral
            timeframe: LTF timeframe identifier
        """
        self.signal_mode = signal_mode
        self.direction_filter = direction_filter
        self.bias_wt2_threshold = bias_wt2_threshold
        self.require_mfi_confirm = require_mfi_confirm
        self.allow_neutral_trades = allow_neutral_trades
        self.timeframe = timeframe

        # Initialize LTF signal detector based on mode
        if signal_mode == SignalMode.SIMPLE:
            self._ltf_detector = SimpleSignalDetector(
                oversold_level=simple_oversold,
                overbought_level=simple_overbought,
                timeframe=timeframe
            )
        else:
            self._ltf_detector = SignalDetector(
                anchor_level_long=anchor_level_long,
                anchor_level_short=anchor_level_short,
                trigger_lookback=trigger_lookback,
                timeframe=timeframe
            )

        # Track HTF bias history
        self._htf_bias_history: List[HTFBiasResult] = []
        self._signals_history: List[Signal] = []
        self._filtered_signals: int = 0

    def reset(self) -> None:
        """Reset detector state."""
        self._ltf_detector.reset()
        self._htf_bias_history = []
        self._signals_history = []
        self._filtered_signals = 0

    def update_htf_bias(
        self,
        wt2: float,
        mfi: float,
        prev_mfi: float,
        prev_prev_mfi: float,
        timestamp: datetime
    ) -> HTFBiasResult:
        """
        Update and store HTF bias.

        Call this method with 4H indicator values before processing 1H bars.

        Returns:
            HTFBiasResult with current bias
        """
        bias_result = determine_htf_bias(
            wt2=wt2,
            mfi=mfi,
            prev_mfi=prev_mfi,
            prev_prev_mfi=prev_prev_mfi,
            timestamp=timestamp,
            wt2_threshold=self.bias_wt2_threshold,
            require_mfi_confirm=self.require_mfi_confirm
        )
        self._htf_bias_history.append(bias_result)
        return bias_result

    def get_current_htf_bias(self) -> Optional[HTFBiasResult]:
        """Get the most recent HTF bias."""
        if self._htf_bias_history:
            return self._htf_bias_history[-1]
        return None

    def process_ltf_bar(
        self,
        timestamp: datetime,
        close_price: float,
        wt1: float,
        wt2: float,
        mfi: float,
        vwap: float,
        wt_result=None,
        mf_result=None,
        bar_idx: Optional[int] = None
    ) -> Optional[Signal]:
        """
        Process LTF bar and check for signals aligned with HTF bias.

        Args:
            timestamp: Bar timestamp
            close_price: Close price for entry
            wt1, wt2: WaveTrend values
            mfi, vwap: MFI and VWAP values
            wt_result: Full WaveTrend result (for ENHANCED mode)
            mf_result: Full MoneyFlow result (for ENHANCED mode)
            bar_idx: Optional bar index

        Returns:
            Signal if conditions met and aligned, None otherwise
        """
        # Get LTF signal
        if self.signal_mode == SignalMode.SIMPLE:
            signal = self._ltf_detector.process_bar(
                timestamp=timestamp,
                close_price=close_price,
                wt1=wt1,
                wt2=wt2,
                mfi=mfi,
                vwap=vwap
            )
        else:
            if wt_result is None or mf_result is None:
                return None
            signal = self._ltf_detector.process_bar(
                timestamp=timestamp,
                close_price=close_price,
                wt_result=wt_result,
                mf_result=mf_result,
                bar_idx=bar_idx
            )

        if signal is None:
            return None

        # Apply direction filter first
        if not self._passes_direction_filter(signal.signal_type):
            self._filtered_signals += 1
            return None

        # Apply HTF bias filter
        htf_bias = self.get_current_htf_bias()
        if htf_bias is None:
            # No HTF bias available yet
            if not self.allow_neutral_trades:
                self._filtered_signals += 1
                return None
        else:
            if not self._passes_htf_bias_filter(signal.signal_type, htf_bias):
                self._filtered_signals += 1
                return None

        # Signal passed all filters
        signal.metadata["mode"] = "mtf_v4"
        signal.metadata["direction_filter"] = self.direction_filter
        if htf_bias:
            signal.metadata["htf_bias"] = htf_bias.bias.value
            signal.metadata["htf_confidence"] = htf_bias.confidence

        self._signals_history.append(signal)
        return signal

    def _passes_direction_filter(self, signal_type: SignalType) -> bool:
        """Check if signal passes direction filter."""
        if self.direction_filter == "both":
            return True
        elif self.direction_filter == "long_only":
            return signal_type == SignalType.LONG
        elif self.direction_filter == "short_only":
            return signal_type == SignalType.SHORT
        return True

    def _passes_htf_bias_filter(self, signal_type: SignalType, htf_bias: HTFBiasResult) -> bool:
        """Check if signal aligns with HTF bias."""
        if htf_bias.bias == Bias.NEUTRAL:
            return self.allow_neutral_trades

        if signal_type == SignalType.LONG:
            return htf_bias.bias == Bias.BULLISH
        elif signal_type == SignalType.SHORT:
            return htf_bias.bias == Bias.BEARISH

        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            "mode": "mtf_v4",
            "ltf_signal_mode": self.signal_mode.value if hasattr(self.signal_mode, 'value') else str(self.signal_mode),
            "direction_filter": self.direction_filter,
            "total_signals": len(self._signals_history),
            "filtered_signals": self._filtered_signals,
            "htf_bias_updates": len(self._htf_bias_history)
        }

    @property
    def signals_history(self) -> List[Signal]:
        """Get history of all signals that passed filters."""
        return self._signals_history.copy()
