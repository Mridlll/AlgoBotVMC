"""Partial Exit System for scaling out of positions.

Implements a two-stage exit strategy:
- Exit 50% at 1R (1:1 risk-reward)
- Move stop loss to breakeven after first exit
- Exit remaining 50% at 2R (2:1 risk-reward) or on opposite signal

This reduces risk by locking in partial profits while allowing
the remainder to run for larger gains.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ExitReason(str, Enum):
    """Reason for exiting a position or partial position."""
    PARTIAL_1R = "partial_1r"        # First partial at 1R target
    PARTIAL_2R = "partial_2r"        # Second partial at 2R target
    FULL_SIGNAL = "full_signal"      # Opposite signal exit
    STOP_LOSS = "stop_loss"          # Stop loss hit
    BREAKEVEN = "breakeven"          # Breakeven stop hit
    VWAP_CURVE = "vwap_curve"        # VWAP curved against position
    MFI_CURVE = "mfi_curve"          # MFI curved against position
    END_OF_DATA = "end_of_data"      # Backtest ended


@dataclass
class ExitLeg:
    """Represents a partial exit from a position."""
    exit_time: datetime
    exit_price: float
    size_closed: float       # Absolute size closed
    size_percent: float      # Percentage of original position
    pnl: float               # PnL for this leg
    pnl_percent: float       # Return percentage
    exit_reason: ExitReason
    remaining_size: float    # Size still open after this exit


@dataclass
class PartialExitPosition:
    """
    A position with partial exit tracking.

    Tracks the full lifecycle of a position including:
    - Original entry
    - Multiple partial exits
    - Final exit
    - Aggregated PnL
    """
    # Entry info
    entry_time: datetime
    entry_price: float
    original_size: float
    is_long: bool
    stop_loss: float
    take_profit_1r: float    # 1:1 R:R target
    take_profit_2r: float    # 2:1 R:R target

    # Current state
    remaining_size: float = field(default=0.0)
    current_stop_loss: float = field(default=0.0)
    exit_legs: List[ExitLeg] = field(default_factory=list)

    # Flags
    first_partial_done: bool = False
    position_closed: bool = False

    def __post_init__(self):
        """Initialize remaining size and current stop."""
        if self.remaining_size == 0.0:
            self.remaining_size = self.original_size
        if self.current_stop_loss == 0.0:
            self.current_stop_loss = self.stop_loss

    @property
    def total_pnl(self) -> float:
        """Total PnL from all exit legs."""
        return sum(leg.pnl for leg in self.exit_legs)

    @property
    def total_pnl_percent(self) -> float:
        """Weighted average PnL percentage."""
        if not self.exit_legs:
            return 0.0
        total_size = sum(leg.size_closed for leg in self.exit_legs)
        if total_size == 0:
            return 0.0
        return sum(leg.pnl_percent * leg.size_closed for leg in self.exit_legs) / total_size

    @property
    def final_exit_time(self) -> Optional[datetime]:
        """Time of final exit."""
        if self.exit_legs:
            return self.exit_legs[-1].exit_time
        return None

    @property
    def avg_exit_price(self) -> float:
        """Volume-weighted average exit price."""
        if not self.exit_legs:
            return 0.0
        total_value = sum(leg.exit_price * leg.size_closed for leg in self.exit_legs)
        total_size = sum(leg.size_closed for leg in self.exit_legs)
        return total_value / total_size if total_size > 0 else 0.0


class PartialExitManager:
    """
    Manages partial exits for positions.

    Default strategy (50/50 at 1R/2R):
    1. Enter with 100% position
    2. When price reaches 1R target: exit 50%, move SL to breakeven
    3. When price reaches 2R target: exit remaining 50%
    4. If opposite signal: exit all remaining
    5. If breakeven SL hit: exit all remaining

    Usage:
        manager = PartialExitManager()

        # Create position
        position = manager.create_position(
            entry_time=timestamp,
            entry_price=100.0,
            size=1.0,
            is_long=True,
            stop_loss=95.0
        )

        # On each bar, check for exits
        exit_legs = manager.check_exits(
            position=position,
            current_price=105.0,
            current_time=timestamp
        )

        if position.position_closed:
            # Record final trade
            pass
    """

    def __init__(
        self,
        first_partial_pct: float = 0.50,
        first_partial_rr: float = 1.0,
        second_partial_pct: float = 0.50,
        second_partial_rr: float = 2.0,
        move_sl_to_breakeven: bool = True
    ):
        """
        Initialize partial exit manager.

        Args:
            first_partial_pct: Percentage to exit at first target (default 50%)
            first_partial_rr: R:R ratio for first target (default 1.0)
            second_partial_pct: Percentage to exit at second target (default 50%)
            second_partial_rr: R:R ratio for second target (default 2.0)
            move_sl_to_breakeven: Move SL to breakeven after first partial
        """
        self.first_partial_pct = first_partial_pct
        self.first_partial_rr = first_partial_rr
        self.second_partial_pct = second_partial_pct
        self.second_partial_rr = second_partial_rr
        self.move_sl_to_breakeven = move_sl_to_breakeven

    def create_position(
        self,
        entry_time: datetime,
        entry_price: float,
        size: float,
        is_long: bool,
        stop_loss: float,
        take_profit: Optional[float] = None
    ) -> PartialExitPosition:
        """
        Create a new position with partial exit targets.

        Args:
            entry_time: Entry timestamp
            entry_price: Entry price
            size: Position size (in base currency or contracts)
            is_long: True for long, False for short
            stop_loss: Stop loss price
            take_profit: Optional fixed take profit (overrides R:R calculation)

        Returns:
            PartialExitPosition ready for exit management
        """
        # Calculate R (risk per share)
        risk_per_share = abs(entry_price - stop_loss)

        # Calculate targets based on R:R ratios
        if is_long:
            tp_1r = entry_price + (risk_per_share * self.first_partial_rr)
            tp_2r = entry_price + (risk_per_share * self.second_partial_rr)
        else:
            tp_1r = entry_price - (risk_per_share * self.first_partial_rr)
            tp_2r = entry_price - (risk_per_share * self.second_partial_rr)

        return PartialExitPosition(
            entry_time=entry_time,
            entry_price=entry_price,
            original_size=size,
            is_long=is_long,
            stop_loss=stop_loss,
            take_profit_1r=tp_1r,
            take_profit_2r=take_profit if take_profit else tp_2r,
            remaining_size=size,
            current_stop_loss=stop_loss
        )

    def check_exits(
        self,
        position: PartialExitPosition,
        current_price: float,
        current_high: float,
        current_low: float,
        current_time: datetime,
        opposite_signal: bool = False,
        vwap_curves_against: bool = False,
        mfi_curves_against: bool = False
    ) -> List[ExitLeg]:
        """
        Check for and execute any triggered exits.

        Args:
            position: The position to check
            current_price: Current close price
            current_high: Current bar high
            current_low: Current bar low
            current_time: Current timestamp
            opposite_signal: True if opposite signal appeared
            vwap_curves_against: True if VWAP is curving against position
            mfi_curves_against: True if MFI is curving against position

        Returns:
            List of ExitLeg objects for any exits that occurred
        """
        if position.position_closed or position.remaining_size <= 0:
            return []

        exit_legs = []

        # Check stop loss first (always full remaining)
        sl_hit = self._check_stop_loss_hit(
            position, current_high, current_low
        )
        if sl_hit:
            leg = self._execute_full_exit(
                position, position.current_stop_loss, current_time,
                ExitReason.BREAKEVEN if position.first_partial_done else ExitReason.STOP_LOSS
            )
            exit_legs.append(leg)
            return exit_legs

        # Check for opposite signal (exit all remaining)
        if opposite_signal:
            leg = self._execute_full_exit(
                position, current_price, current_time, ExitReason.FULL_SIGNAL
            )
            exit_legs.append(leg)
            return exit_legs

        # Check first partial target (1R)
        if not position.first_partial_done:
            target_hit = self._check_target_hit(
                position, position.take_profit_1r, current_high, current_low
            )
            if target_hit:
                leg = self._execute_partial_exit(
                    position, position.take_profit_1r, current_time,
                    self.first_partial_pct, ExitReason.PARTIAL_1R
                )
                exit_legs.append(leg)
                position.first_partial_done = True

                # Move stop loss to breakeven
                if self.move_sl_to_breakeven:
                    position.current_stop_loss = position.entry_price

        # Check second partial target (2R) if first is done
        if position.first_partial_done and position.remaining_size > 0:
            target_hit = self._check_target_hit(
                position, position.take_profit_2r, current_high, current_low
            )
            if target_hit:
                leg = self._execute_full_exit(
                    position, position.take_profit_2r, current_time,
                    ExitReason.PARTIAL_2R
                )
                exit_legs.append(leg)

        # Optional: Check for indicator curve exits (use remaining 25% each)
        if position.remaining_size > 0 and position.first_partial_done:
            if vwap_curves_against:
                size_to_close = position.remaining_size * 0.5  # Close half of remaining
                if size_to_close > 0:
                    leg = self._execute_sized_exit(
                        position, current_price, current_time,
                        size_to_close, ExitReason.VWAP_CURVE
                    )
                    exit_legs.append(leg)

            elif mfi_curves_against and position.remaining_size > 0:
                size_to_close = position.remaining_size * 0.5
                if size_to_close > 0:
                    leg = self._execute_sized_exit(
                        position, current_price, current_time,
                        size_to_close, ExitReason.MFI_CURVE
                    )
                    exit_legs.append(leg)

        return exit_legs

    def _check_stop_loss_hit(
        self,
        position: PartialExitPosition,
        high: float,
        low: float
    ) -> bool:
        """Check if stop loss was hit during this bar."""
        if position.is_long:
            return low <= position.current_stop_loss
        else:
            return high >= position.current_stop_loss

    def _check_target_hit(
        self,
        position: PartialExitPosition,
        target: float,
        high: float,
        low: float
    ) -> bool:
        """Check if target was hit during this bar."""
        if position.is_long:
            return high >= target
        else:
            return low <= target

    def _calculate_pnl(
        self,
        position: PartialExitPosition,
        exit_price: float,
        size: float
    ) -> Tuple[float, float]:
        """Calculate PnL for a given exit."""
        if position.is_long:
            pnl = (exit_price - position.entry_price) * size
        else:
            pnl = (position.entry_price - exit_price) * size

        pnl_percent = (pnl / (position.entry_price * size)) * 100 if size > 0 else 0

        return pnl, pnl_percent

    def _execute_partial_exit(
        self,
        position: PartialExitPosition,
        exit_price: float,
        exit_time: datetime,
        exit_pct: float,
        reason: ExitReason
    ) -> ExitLeg:
        """Execute a partial exit."""
        size_to_close = position.original_size * exit_pct
        size_to_close = min(size_to_close, position.remaining_size)

        pnl, pnl_pct = self._calculate_pnl(position, exit_price, size_to_close)
        position.remaining_size -= size_to_close

        leg = ExitLeg(
            exit_time=exit_time,
            exit_price=exit_price,
            size_closed=size_to_close,
            size_percent=exit_pct * 100,
            pnl=pnl,
            pnl_percent=pnl_pct,
            exit_reason=reason,
            remaining_size=position.remaining_size
        )
        position.exit_legs.append(leg)

        if position.remaining_size <= 0:
            position.position_closed = True

        return leg

    def _execute_full_exit(
        self,
        position: PartialExitPosition,
        exit_price: float,
        exit_time: datetime,
        reason: ExitReason
    ) -> ExitLeg:
        """Exit all remaining position."""
        size_to_close = position.remaining_size
        pnl, pnl_pct = self._calculate_pnl(position, exit_price, size_to_close)
        position.remaining_size = 0

        leg = ExitLeg(
            exit_time=exit_time,
            exit_price=exit_price,
            size_closed=size_to_close,
            size_percent=100.0,  # Closing 100% of remaining
            pnl=pnl,
            pnl_percent=pnl_pct,
            exit_reason=reason,
            remaining_size=0
        )
        position.exit_legs.append(leg)
        position.position_closed = True

        return leg

    def _execute_sized_exit(
        self,
        position: PartialExitPosition,
        exit_price: float,
        exit_time: datetime,
        size: float,
        reason: ExitReason
    ) -> ExitLeg:
        """Exit a specific size."""
        size_to_close = min(size, position.remaining_size)
        pnl, pnl_pct = self._calculate_pnl(position, exit_price, size_to_close)
        position.remaining_size -= size_to_close

        pct_of_original = (size_to_close / position.original_size) * 100

        leg = ExitLeg(
            exit_time=exit_time,
            exit_price=exit_price,
            size_closed=size_to_close,
            size_percent=pct_of_original,
            pnl=pnl,
            pnl_percent=pnl_pct,
            exit_reason=reason,
            remaining_size=position.remaining_size
        )
        position.exit_legs.append(leg)

        if position.remaining_size <= 0:
            position.position_closed = True

        return leg

    def force_close(
        self,
        position: PartialExitPosition,
        exit_price: float,
        exit_time: datetime,
        reason: ExitReason = ExitReason.END_OF_DATA
    ) -> Optional[ExitLeg]:
        """Force close any remaining position."""
        if position.position_closed or position.remaining_size <= 0:
            return None

        return self._execute_full_exit(position, exit_price, exit_time, reason)
