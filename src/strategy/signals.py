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


class SignalType(str, Enum):
    """Type of trading signal."""
    LONG = "long"
    SHORT = "short"


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
