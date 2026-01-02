"""Backtesting engine for VMC strategy."""

import sys
from pathlib import Path

# Add project root and src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from indicators import HeikinAshi, WaveTrend, MoneyFlow, VWAPCalculator
from indicators.heikin_ashi import convert_to_heikin_ashi
from strategy.signals import (
    SignalDetector, Signal, SignalType, SimpleSignalDetector, Bias,
    V4SignalDetector, HTFBiasResult, determine_htf_bias
)
from strategy.risk import RiskManager, StopLossMethod
from utils.logger import get_logger

# Import config enums
try:
    from config.config import TakeProfitMethod, OscillatorExitMode, TimeFilterMode, SignalMode, DirectionFilter
except ImportError:
    # Fallback for when running standalone
    from enum import Enum
    class TakeProfitMethod(str, Enum):
        FIXED_RR = "fixed_rr"
        OSCILLATOR = "oscillator"
    class OscillatorExitMode(str, Enum):
        FULL_SIGNAL = "full_signal"
        WT_CROSS = "wt_cross"
        FIRST_REVERSAL = "first_reversal"
    class TimeFilterMode(str, Enum):
        SKIP = "skip"
        REDUCE = "reduce"
    class SignalMode(str, Enum):
        SIMPLE = "simple"
        ENHANCED = "enhanced"
        MTF = "mtf"
        MTF_V4 = "mtf_v4"
    class DirectionFilter(str, Enum):
        BOTH = "both"
        LONG_ONLY = "long_only"
        SHORT_ONLY = "short_only"

logger = get_logger("backtest")


@dataclass
class BacktestTrade:
    """Backtested trade information."""
    entry_time: datetime
    exit_time: Optional[datetime]
    signal_type: SignalType
    entry_price: float
    exit_price: Optional[float]
    size: float
    stop_loss: float
    take_profit: float
    pnl: float = 0.0
    pnl_percent: float = 0.0
    exit_reason: str = ""
    # Indicator values at entry for client verification
    wt1_entry: float = 0.0
    wt2_entry: float = 0.0
    mfi_entry: float = 0.0
    vwap_entry: float = 0.0

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary for DataFrame export."""
        return {
            "entry_time": self.entry_time.strftime("%Y-%m-%d %H:%M") if self.entry_time else None,
            "exit_time": self.exit_time.strftime("%Y-%m-%d %H:%M") if self.exit_time else None,
            "signal_type": self.signal_type.value if self.signal_type else None,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "size": self.size,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "pnl": self.pnl,
            "pnl_percent": self.pnl_percent,
            "exit_reason": self.exit_reason,
            "is_winner": self.is_winner,
            "wt1_entry": self.wt1_entry,
            "wt2_entry": self.wt2_entry,
            "mfi_entry": self.mfi_entry,
            "vwap_entry": self.vwap_entry,
        }


@dataclass
class BacktestResult:
    """Backtesting results."""
    initial_balance: float
    final_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_percent: float
    max_drawdown: float
    max_drawdown_percent: float
    sharpe_ratio: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float  # in bars
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_balance": self.initial_balance,
            "final_balance": self.final_balance,
            "total_pnl": self.total_pnl,
            "total_pnl_percent": self.total_pnl_percent,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_percent": self.max_drawdown_percent,
            "sharpe_ratio": self.sharpe_ratio,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
        }

    def export_trades_to_csv(self, filepath: str) -> None:
        """Export all trades to CSV file for verification."""
        if not self.trades:
            return

        rows = []
        running_balance = self.initial_balance
        for i, t in enumerate(self.trades, 1):
            running_balance += t.pnl
            rows.append({
                "trade_num": i,
                "entry_time": t.entry_time.strftime("%Y-%m-%d %H:%M") if t.entry_time else "",
                "exit_time": t.exit_time.strftime("%Y-%m-%d %H:%M") if t.exit_time else "",
                "direction": "LONG" if t.signal_type == SignalType.LONG else "SHORT",
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2) if t.exit_price else "",
                "stop_loss": round(t.stop_loss, 2),
                "take_profit": round(t.take_profit, 2),
                "size": round(t.size, 6),
                "pnl": round(t.pnl, 2),
                "pnl_percent": round(t.pnl_percent, 2),
                "exit_reason": t.exit_reason,
                "result": "WIN" if t.pnl > 0 else "LOSS",
                "running_balance": round(running_balance, 2),
                # Indicator values at entry for client verification
                "wt1_entry": round(t.wt1_entry, 2),
                "wt2_entry": round(t.wt2_entry, 2),
                "mfi_entry": round(t.mfi_entry, 2),
                "vwap_entry": round(t.vwap_entry, 2)
            })

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)

    def get_trades_summary(self) -> str:
        """Get a text summary of trades for quick verification."""
        if not self.trades:
            return "No trades"

        lines = [
            f"Total Trades: {len(self.trades)}",
            f"Winners: {self.winning_trades} ({self.win_rate:.1f}%)",
            f"Losers: {self.losing_trades}",
            f"",
            f"First 5 trades:",
        ]

        for i, t in enumerate(self.trades[:5], 1):
            direction = "LONG" if t.signal_type == SignalType.LONG else "SHORT"
            result = "WIN" if t.pnl > 0 else "LOSS"
            lines.append(f"  {i}. {t.entry_time.strftime('%Y-%m-%d')} {direction} ${t.pnl:+.2f} ({result}) - {t.exit_reason}")

        if len(self.trades) > 5:
            lines.append(f"  ... and {len(self.trades) - 5} more trades")

        return "\n".join(lines)


class BacktestEngine:
    """
    Backtesting engine for VMC strategy.

    Simulates trading on historical data with realistic
    position sizing and risk management.
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        risk_percent: float = 3.0,
        risk_reward: float = 2.0,
        commission_percent: float = 0.06,
        sl_method: StopLossMethod = StopLossMethod.SWING,
        swing_lookback: int = 5,
        anchor_level_long: float = -60,
        anchor_level_short: float = 60,
        tp_method: TakeProfitMethod = TakeProfitMethod.FIXED_RR,
        oscillator_mode: OscillatorExitMode = OscillatorExitMode.WT_CROSS,
        # Time filter parameters (v3)
        time_filter_enabled: bool = False,
        time_filter_mode: TimeFilterMode = TimeFilterMode.SKIP,
        avoid_us_market_hours: bool = True,
        market_hours_start_utc: int = 14,
        market_hours_end_utc: int = 21,
        trade_weekends: bool = True,
        reduced_position_multiplier: float = 0.5,
        # Signal mode parameters (v3 rework)
        signal_mode: SignalMode = SignalMode.ENHANCED,
        simple_oversold: float = -53,
        simple_overbought: float = 53,
        # Slippage parameter (realistic execution costs)
        slippage_percent: float = 0.0,
        # No stop loss mode (exit only on opposite signal)
        use_stop_loss: bool = True,
        # Trailing stop loss parameters
        use_trailing_sl: bool = False,
        trailing_activation_rr: float = 1.0,
        trailing_distance_percent: float = 1.0,
        # Dynamic ATR stop loss parameters (only used with DYNAMIC_ATR sl_method)
        dynamic_sl_base_mult: float = 2.0,
        dynamic_sl_high_vol_mult: float = 3.0,
        dynamic_sl_low_vol_mult: float = 1.5,
        volatility_lookback: int = 20,
        # Regime filter (v4) - filters trades based on market regime
        regime_detector: Optional[Any] = None,
        # Direction filter (v5) - filter by trade direction
        direction_filter: str = "both"  # "both", "long_only", "short_only"
    ):
        """
        Initialize backtest engine.

        Args:
            initial_balance: Starting balance
            risk_percent: Risk per trade (%)
            risk_reward: Risk:Reward ratio
            commission_percent: Commission per trade (%)
            sl_method: Stop loss method
            swing_lookback: Candles for swing detection
            anchor_level_long: WT2 level for long anchor
            anchor_level_short: WT2 level for short anchor
            tp_method: Take profit method (fixed_rr or oscillator)
            oscillator_mode: Exit mode when tp_method is oscillator
            time_filter_enabled: Enable time-based trading filter (v3)
            time_filter_mode: skip=no trades, reduce=smaller position
            avoid_us_market_hours: Skip/reduce during US market hours
            market_hours_start_utc: Market hours start (UTC, default 14 = 9:30 AM EST)
            market_hours_end_utc: Market hours end (UTC, default 21 = 4:00 PM EST)
            trade_weekends: Allow trading on weekends
            reduced_position_multiplier: Position size multiplier when mode=reduce
            signal_mode: Signal detection mode (simple, enhanced, mtf)
            simple_oversold: WT2 level for LONG signals in simple mode (original: -53)
            simple_overbought: WT2 level for SHORT signals in simple mode (original: +53)
            slippage_percent: Slippage per trade (%) - applied on both entry and exit (default: 0.0)
            use_stop_loss: Enable stop loss (default: True). When False, only exit on opposite signal.
            use_trailing_sl: Enable trailing stop loss (default: False)
            trailing_activation_rr: R:R level to activate trailing stop (default: 1.0)
            trailing_distance_percent: Trailing distance as % of price (default: 1.0%)
            dynamic_sl_base_mult: Base ATR multiplier for normal volatility (default: 2.0)
            dynamic_sl_high_vol_mult: ATR multiplier for high volatility (default: 3.0)
            dynamic_sl_low_vol_mult: ATR multiplier for low volatility (default: 1.5)
            volatility_lookback: Bars for volatility regime detection (default: 20)
            direction_filter: Trade direction filter ('both', 'long_only', 'short_only')
        """
        self.initial_balance = initial_balance
        self.risk_percent = risk_percent
        self.risk_reward = risk_reward
        self.commission_percent = commission_percent
        self.sl_method = sl_method
        self.swing_lookback = swing_lookback
        self.tp_method = tp_method
        self.oscillator_mode = oscillator_mode

        # Time filter settings (v3)
        self.time_filter_enabled = time_filter_enabled
        self.time_filter_mode = time_filter_mode
        self.avoid_us_market_hours = avoid_us_market_hours
        self.market_hours_start_utc = market_hours_start_utc
        self.market_hours_end_utc = market_hours_end_utc
        self.trade_weekends = trade_weekends
        self.reduced_position_multiplier = reduced_position_multiplier

        # Signal mode settings (v3 rework)
        self.signal_mode = signal_mode
        self.simple_oversold = simple_oversold
        self.simple_overbought = simple_overbought

        # Slippage setting
        self.slippage_percent = slippage_percent

        # Stop loss setting
        self.use_stop_loss = use_stop_loss

        # Trailing stop loss settings
        self.use_trailing_sl = use_trailing_sl
        self.trailing_activation_rr = trailing_activation_rr
        self.trailing_distance_percent = trailing_distance_percent

        # Dynamic ATR settings
        self.dynamic_sl_base_mult = dynamic_sl_base_mult
        self.dynamic_sl_high_vol_mult = dynamic_sl_high_vol_mult
        self.dynamic_sl_low_vol_mult = dynamic_sl_low_vol_mult
        self.volatility_lookback = volatility_lookback

        # Regime filter (v4)
        self.regime_detector = regime_detector

        # Direction filter (v5)
        self.direction_filter = direction_filter

        # Initialize components
        self.heikin_ashi = HeikinAshi()
        self.wavetrend = WaveTrend(
            overbought_2=int(anchor_level_short),
            oversold_2=int(anchor_level_long)
        )
        self.money_flow = MoneyFlow()
        self.vwap_calc = VWAPCalculator()  # Real VWAP (not WT momentum diff)

        # Initialize signal detector based on mode
        if signal_mode == SignalMode.SIMPLE:
            self.signal_detector = SimpleSignalDetector(
                oversold_level=simple_oversold,
                overbought_level=simple_overbought
            )
            logger.info(f"Using SIMPLE signal mode (oversold={simple_oversold}, overbought={simple_overbought})")
        elif signal_mode == SignalMode.MTF:
            # MTF mode requires multi-timeframe data - use enhanced as fallback for single TF
            self.signal_detector = SignalDetector(
                anchor_level_long=anchor_level_long,
                anchor_level_short=anchor_level_short,
                trigger_lookback=20
            )
            logger.info("Using MTF signal mode (requires multi-timeframe data, falling back to enhanced for single TF)")
        else:  # ENHANCED (default)
            self.signal_detector = SignalDetector(
                anchor_level_long=anchor_level_long,
                anchor_level_short=anchor_level_short,
                trigger_lookback=20
            )
            logger.info(f"Using ENHANCED signal mode (anchor={anchor_level_long}/{anchor_level_short})")

        self.risk_manager = RiskManager(
            default_risk_percent=risk_percent,
            default_rr=risk_reward,
            swing_lookback=swing_lookback,
            # Dynamic ATR parameters
            dynamic_atr_base_mult=dynamic_sl_base_mult,
            dynamic_atr_high_vol_mult=dynamic_sl_high_vol_mult,
            dynamic_atr_low_vol_mult=dynamic_sl_low_vol_mult,
            volatility_lookback=volatility_lookback
        )

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            df: DataFrame with OHLCV data (must have: open, high, low, close, volume)

        Returns:
            BacktestResult with performance metrics
        """
        tp_method_name = self.tp_method.value if hasattr(self.tp_method, 'value') else str(self.tp_method)
        osc_mode_name = self.oscillator_mode.value if hasattr(self.oscillator_mode, 'value') else str(self.oscillator_mode)
        signal_mode_name = self.signal_mode.value if hasattr(self.signal_mode, 'value') else str(self.signal_mode)
        logger.info(f"Running backtest on {len(df)} candles (Signal: {signal_mode_name}, TP: {tp_method_name}, Osc: {osc_mode_name})...")

        # Reset signal detector (only if it has reset method)
        if hasattr(self.signal_detector, 'reset'):
            self.signal_detector.reset()

        # Convert to Heikin Ashi
        ha_df = convert_to_heikin_ashi(df)

        # Calculate indicators on full dataset
        wt_result = self.wavetrend.calculate(ha_df)
        mf_result = self.money_flow.calculate(ha_df)
        vwap_result = self.vwap_calc.calculate(df)  # Use raw OHLCV for volume data

        # Trading state
        balance = self.initial_balance
        equity_curve = [balance]
        trades: List[BacktestTrade] = []
        active_trade: Optional[BacktestTrade] = None
        initial_stop_loss: float = 0.0  # Track initial SL for trailing stop calculation

        # Previous indicator values for oscillator exit detection
        prev_wt1 = 0.0
        prev_wt2 = 0.0
        prev_mfi = 0.0
        prev_vwap = 0.0

        # Process each bar
        for i in range(100, len(df)):  # Start after warmup period
            current_bar = df.iloc[i]
            current_price = current_bar['close']
            current_high = current_bar['high']
            current_low = current_bar['low']
            current_time = df.index[i]

            # Get current indicator values
            wt1 = wt_result.wt1.iloc[i]
            wt2 = wt_result.wt2.iloc[i]
            mfi = mf_result.mfi.iloc[i]
            vwap = vwap_result.vwap.iloc[i]  # Use REAL VWAP, not WT momentum

            # Create subset DataFrames for signal detection
            wt_subset = type(wt_result)(
                wt1=wt_result.wt1.iloc[:i+1],
                wt2=wt_result.wt2.iloc[:i+1],
                momentum=wt_result.momentum.iloc[:i+1],
                cross=wt_result.cross.iloc[:i+1],
                cross_up=wt_result.cross_up.iloc[:i+1],
                cross_down=wt_result.cross_down.iloc[:i+1],
                oversold=wt_result.oversold.iloc[:i+1],
                overbought=wt_result.overbought.iloc[:i+1]
            )

            mf_subset = type(mf_result)(
                mfi=mf_result.mfi.iloc[:i+1],
                is_positive=mf_result.is_positive.iloc[:i+1],
                is_negative=mf_result.is_negative.iloc[:i+1],
                curving_up=mf_result.curving_up.iloc[:i+1],
                curving_down=mf_result.curving_down.iloc[:i+1]
            )

            # Run signal detection based on mode
            if self.signal_mode == SignalMode.SIMPLE:
                # Simple mode uses direct WT values (handles prev internally)
                signal = self.signal_detector.process_bar(
                    timestamp=current_time,
                    close_price=current_price,
                    wt1=wt1,
                    wt2=wt2,
                    mfi=mfi,
                    vwap=vwap
                )
            else:
                # Enhanced/MTF mode uses full indicator results
                signal = self.signal_detector.process_bar(
                    timestamp=current_time,
                    close_price=current_price,
                    wt_result=wt_subset,
                    mf_result=mf_subset,
                    bar_idx=i
                )

            # Check active trade exit conditions
            if active_trade:
                exit_reason = None

                # Check stop loss first (price-based) - only if enabled
                is_long = active_trade.signal_type == SignalType.LONG

                # Update trailing stop if enabled
                if self.use_trailing_sl and self.use_stop_loss and initial_stop_loss != 0:
                    new_sl = self.risk_manager.calculate_trailing_stop(
                        current_price=current_price,
                        entry_price=active_trade.entry_price,
                        stop_loss_price=initial_stop_loss,  # Use initial SL for R:R calculation
                        current_stop=active_trade.stop_loss,
                        is_long=is_long,
                        trailing_distance_percent=self.trailing_distance_percent,
                        activation_rr=self.trailing_activation_rr
                    )
                    # Only update if new stop is better (higher for longs, lower for shorts)
                    if is_long and new_sl > active_trade.stop_loss:
                        active_trade.stop_loss = new_sl
                    elif not is_long and new_sl < active_trade.stop_loss:
                        active_trade.stop_loss = new_sl

                if self.use_stop_loss:
                    if is_long:
                        if current_low <= active_trade.stop_loss:
                            exit_reason = "stop_loss"
                    else:
                        if current_high >= active_trade.stop_loss:
                            exit_reason = "stop_loss"

                # Check take profit based on method
                if not exit_reason:
                    if self.tp_method == TakeProfitMethod.FIXED_RR:
                        # Fixed R:R - check price-based take profit
                        if is_long:
                            if current_high >= active_trade.take_profit:
                                exit_reason = "take_profit"
                        else:
                            if current_low <= active_trade.take_profit:
                                exit_reason = "take_profit"
                    elif self.tp_method == TakeProfitMethod.OSCILLATOR:
                        # Oscillator-based exit
                        exit_reason = self._check_oscillator_exit(
                            trade=active_trade,
                            wt1=wt1,
                            wt2=wt2,
                            prev_wt1=prev_wt1,
                            prev_wt2=prev_wt2,
                            mfi=mfi,
                            prev_mfi=prev_mfi,
                            vwap=vwap,
                            prev_vwap=prev_vwap,
                            current_signal=signal,
                            mode=self.oscillator_mode
                        )

                if exit_reason:
                    # Close trade
                    active_trade = self._close_trade(
                        active_trade, current_price, current_time, exit_reason
                    )

                    # Apply commission
                    commission = (active_trade.exit_price * active_trade.size) * (self.commission_percent / 100)
                    active_trade.pnl -= commission

                    # Update balance
                    balance += active_trade.pnl
                    trades.append(active_trade)
                    active_trade = None

            # Check for new signals (only if no active trade)
            if not active_trade and signal:
                # Check direction filter (v5) - before time/regime filters
                if self.direction_filter == "long_only" and signal.signal_type != SignalType.LONG:
                    signal = None  # Skip non-long signals
                elif self.direction_filter == "short_only" and signal.signal_type != SignalType.SHORT:
                    signal = None  # Skip non-short signals

            if not active_trade and signal:
                # Check time filter (v3)
                can_trade, position_multiplier = self._check_time_filter(current_time)

                # Check regime filter (v4) - after time filter, before trade
                if can_trade and self.regime_detector is not None:
                    regime_can_trade, regime_mult = self.regime_detector.check_regime_filter(
                        df.iloc[:i+1], current_time
                    )
                    if not regime_can_trade:
                        can_trade = False
                    else:
                        position_multiplier *= regime_mult

                if can_trade:
                    # Open new trade
                    active_trade = self._open_trade(
                        signal, balance, df.iloc[:i+1], current_time,
                        position_multiplier=position_multiplier
                    )
                    # Store initial stop loss for trailing stop calculation
                    initial_stop_loss = active_trade.stop_loss

                    # Apply entry commission
                    commission = (active_trade.entry_price * active_trade.size) * (self.commission_percent / 100)
                    balance -= commission

            # Update equity curve
            if active_trade:
                unrealized_pnl = self._calculate_unrealized_pnl(active_trade, current_price)
                equity_curve.append(balance + unrealized_pnl)
            else:
                equity_curve.append(balance)

            # Store current values as previous for next iteration
            prev_wt1 = wt1
            prev_wt2 = wt2
            prev_mfi = mfi
            prev_vwap = vwap

        # Close any remaining trade at last price
        if active_trade:
            last_price = df.iloc[-1]['close']
            last_time = df.index[-1]
            active_trade = self._close_trade(active_trade, last_price, last_time, "end_of_data")
            balance += active_trade.pnl
            trades.append(active_trade)

        # Calculate results
        return self._calculate_results(trades, equity_curve)

    def run_mtf(
        self,
        ltf_data: pd.DataFrame,
        htf_data: Dict[str, pd.DataFrame],
        ltf_timeframe: str = "15m",
        htf_timeframes: Optional[List[str]] = None,
        min_confidence: float = 0.5
    ) -> BacktestResult:
        """
        Run MTF backtest with proper bar synchronization.

        Uses higher timeframes to determine market bias and only takes
        signals that align with the HTF direction.

        Args:
            ltf_data: DataFrame with LTF OHLCV data (entry timeframe)
            htf_data: Dict of timeframe -> DataFrame for each HTF
            ltf_timeframe: LTF timeframe string (e.g., '15m')
            htf_timeframes: List of HTF timeframes to use (defaults to keys in htf_data)
            min_confidence: Minimum HTF bias confidence required (0-1)

        Returns:
            BacktestResult with performance metrics and MTF-specific info
        """
        from backtest.mtf_sync import MTFSynchronizer

        if htf_timeframes is None:
            htf_timeframes = list(htf_data.keys())

        logger.info(f"Running MTF backtest: LTF={ltf_timeframe}, HTF={htf_timeframes}, min_conf={min_confidence}")

        # Create synchronizer
        synchronizer = MTFSynchronizer(
            ltf_data=ltf_data,
            htf_data=htf_data,
            ltf_timeframe=ltf_timeframe,
            htf_timeframes=htf_timeframes,
            wt_overbought=int(abs(self.wavetrend.overbought_2)),
            wt_oversold=int(-abs(self.wavetrend.oversold_2))
        )

        # Log sync stats
        stats = synchronizer.get_sync_stats()
        logger.info(f"MTF Sync stats: {stats}")

        # Reset signal detector
        if hasattr(self.signal_detector, 'reset'):
            self.signal_detector.reset()

        # Convert to Heikin Ashi
        ha_df = convert_to_heikin_ashi(ltf_data)

        # Calculate LTF indicators
        wt_result = self.wavetrend.calculate(ha_df)
        mf_result = self.money_flow.calculate(ha_df)
        vwap_result = self.vwap_calc.calculate(ltf_data)  # Use raw OHLCV for volume data

        # Trading state
        balance = self.initial_balance
        equity_curve = [balance]
        trades: List[BacktestTrade] = []
        active_trade: Optional[BacktestTrade] = None
        initial_stop_loss: float = 0.0  # Track initial SL for trailing stop calculation

        # MTF-specific tracking
        signals_filtered = 0
        signals_aligned = 0
        signals_neutral = 0

        # Previous indicator values
        prev_wt1 = 0.0
        prev_wt2 = 0.0
        prev_mfi = 0.0
        prev_vwap = 0.0

        # Process each bar
        for i in range(100, len(ltf_data)):
            current_bar = ltf_data.iloc[i]
            current_price = current_bar['close']
            current_high = current_bar['high']
            current_low = current_bar['low']
            current_time = ltf_data.index[i]

            # Get current indicator values
            wt1 = wt_result.wt1.iloc[i]
            wt2 = wt_result.wt2.iloc[i]
            mfi = mf_result.mfi.iloc[i]
            vwap = vwap_result.vwap.iloc[i]  # Use REAL VWAP, not WT momentum

            # Create indicator subsets
            wt_subset = type(wt_result)(
                wt1=wt_result.wt1.iloc[:i+1],
                wt2=wt_result.wt2.iloc[:i+1],
                momentum=wt_result.momentum.iloc[:i+1],
                cross=wt_result.cross.iloc[:i+1],
                cross_up=wt_result.cross_up.iloc[:i+1],
                cross_down=wt_result.cross_down.iloc[:i+1],
                oversold=wt_result.oversold.iloc[:i+1],
                overbought=wt_result.overbought.iloc[:i+1]
            )

            mf_subset = type(mf_result)(
                mfi=mf_result.mfi.iloc[:i+1],
                is_positive=mf_result.is_positive.iloc[:i+1],
                is_negative=mf_result.is_negative.iloc[:i+1],
                curving_up=mf_result.curving_up.iloc[:i+1],
                curving_down=mf_result.curving_down.iloc[:i+1]
            )

            # Get LTF signal using ENHANCED mode detection
            signal = self.signal_detector.process_bar(
                timestamp=current_time,
                close_price=current_price,
                wt_result=wt_subset,
                mf_result=mf_subset,
                bar_idx=i
            )

            # Check active trade exit conditions
            if active_trade:
                exit_reason = None

                is_long = active_trade.signal_type == SignalType.LONG

                # Update trailing stop if enabled
                if self.use_trailing_sl and self.use_stop_loss and initial_stop_loss != 0:
                    new_sl = self.risk_manager.calculate_trailing_stop(
                        current_price=current_price,
                        entry_price=active_trade.entry_price,
                        stop_loss_price=initial_stop_loss,
                        current_stop=active_trade.stop_loss,
                        is_long=is_long,
                        trailing_distance_percent=self.trailing_distance_percent,
                        activation_rr=self.trailing_activation_rr
                    )
                    if is_long and new_sl > active_trade.stop_loss:
                        active_trade.stop_loss = new_sl
                    elif not is_long and new_sl < active_trade.stop_loss:
                        active_trade.stop_loss = new_sl

                if self.use_stop_loss:
                    if is_long:
                        if current_low <= active_trade.stop_loss:
                            exit_reason = "stop_loss"
                    else:
                        if current_high >= active_trade.stop_loss:
                            exit_reason = "stop_loss"

                if not exit_reason:
                    if self.tp_method == TakeProfitMethod.FIXED_RR:
                        if is_long:
                            if current_high >= active_trade.take_profit:
                                exit_reason = "take_profit"
                        else:
                            if current_low <= active_trade.take_profit:
                                exit_reason = "take_profit"
                    elif self.tp_method == TakeProfitMethod.OSCILLATOR:
                        exit_reason = self._check_oscillator_exit(
                            trade=active_trade,
                            wt1=wt1, wt2=wt2,
                            prev_wt1=prev_wt1, prev_wt2=prev_wt2,
                            mfi=mfi, prev_mfi=prev_mfi,
                            vwap=vwap, prev_vwap=prev_vwap,
                            current_signal=signal,
                            mode=self.oscillator_mode
                        )

                if exit_reason:
                    active_trade = self._close_trade(
                        active_trade, current_price, current_time, exit_reason
                    )
                    commission = (active_trade.exit_price * active_trade.size) * (self.commission_percent / 100)
                    active_trade.pnl -= commission
                    balance += active_trade.pnl
                    trades.append(active_trade)
                    active_trade = None

            # Check for new signals with MTF filter
            if not active_trade and signal:
                # Get HTF bias
                should_take, reason, bias_result = synchronizer.should_take_signal(
                    signal_type=signal.signal_type,
                    ltf_timestamp=current_time,
                    min_confidence=min_confidence
                )

                if should_take:
                    signals_aligned += 1

                    # Check time filter
                    can_trade, position_multiplier = self._check_time_filter(current_time)

                    if can_trade:
                        active_trade = self._open_trade(
                            signal, balance, ltf_data.iloc[:i+1], current_time,
                            position_multiplier=position_multiplier
                        )
                        initial_stop_loss = active_trade.stop_loss
                        commission = (active_trade.entry_price * active_trade.size) * (self.commission_percent / 100)
                        balance -= commission
                else:
                    if bias_result.overall_bias == Bias.NEUTRAL:
                        signals_neutral += 1
                    else:
                        signals_filtered += 1

            # Update equity curve
            if active_trade:
                unrealized_pnl = self._calculate_unrealized_pnl(active_trade, current_price)
                equity_curve.append(balance + unrealized_pnl)
            else:
                equity_curve.append(balance)

            prev_wt1 = wt1
            prev_wt2 = wt2
            prev_mfi = mfi
            prev_vwap = vwap

        # Close remaining trade
        if active_trade:
            last_price = ltf_data.iloc[-1]['close']
            last_time = ltf_data.index[-1]
            active_trade = self._close_trade(active_trade, last_price, last_time, "end_of_data")
            balance += active_trade.pnl
            trades.append(active_trade)

        # Log MTF filter stats
        total_signals = signals_aligned + signals_filtered + signals_neutral
        logger.info(f"MTF Filter: {signals_aligned} aligned, {signals_filtered} filtered, "
                   f"{signals_neutral} neutral / {total_signals} total LTF signals")

        result = self._calculate_results(trades, equity_curve)
        return result

    def run_adaptive(
        self,
        df: pd.DataFrame,
        regime_lookback: int = 20,
        trending_threshold: float = 80,
        ranging_threshold: float = 30
    ) -> BacktestResult:
        """
        Run backtest with adaptive mode switching.

        Automatically switches between SIMPLE and ENHANCED modes based on
        market volatility regime detection.

        Args:
            df: DataFrame with OHLCV data
            regime_lookback: Bars to analyze for regime detection
            trending_threshold: WT2 range above this = TRENDING = ENHANCED mode
            ranging_threshold: WT2 range below this = RANGING = SIMPLE mode

        Returns:
            BacktestResult with performance metrics
        """
        from strategy.volatility_regime import VolatilityRegimeDetector, VolatilityRegime

        logger.info(f"Running ADAPTIVE backtest on {len(df)} candles "
                   f"(trending>{trending_threshold}, ranging<{ranging_threshold})")

        # Initialize regime detector
        regime_detector = VolatilityRegimeDetector(
            lookback=regime_lookback,
            trending_threshold=trending_threshold,
            ranging_threshold=ranging_threshold,
            min_regime_bars=5
        )

        # Initialize both signal detectors
        simple_detector = SimpleSignalDetector(
            oversold_level=self.simple_oversold,
            overbought_level=self.simple_overbought
        )
        enhanced_detector = SignalDetector(
            anchor_level_long=self.wavetrend.oversold_2,
            anchor_level_short=self.wavetrend.overbought_2,
            trigger_lookback=20
        )

        # Convert to Heikin Ashi and calculate indicators
        ha_df = convert_to_heikin_ashi(df)
        wt_result = self.wavetrend.calculate(ha_df)
        mf_result = self.money_flow.calculate(ha_df)
        vwap_result = self.vwap_calc.calculate(df)  # Use raw OHLCV for volume data

        # Trading state
        balance = self.initial_balance
        equity_curve = [balance]
        trades: List[BacktestTrade] = []
        active_trade: Optional[BacktestTrade] = None
        initial_stop_loss: float = 0.0  # Track initial SL for trailing stop calculation

        # Adaptive mode tracking
        current_mode = "enhanced"  # Start with enhanced
        mode_switches = 0
        bars_in_simple = 0
        bars_in_enhanced = 0
        trades_in_simple = 0
        trades_in_enhanced = 0

        # Previous values
        prev_wt1 = 0.0
        prev_wt2 = 0.0
        prev_mfi = 0.0
        prev_vwap = 0.0

        for i in range(100, len(df)):
            current_bar = df.iloc[i]
            current_price = current_bar['close']
            current_high = current_bar['high']
            current_low = current_bar['low']
            current_time = df.index[i]

            wt1 = wt_result.wt1.iloc[i]
            wt2 = wt_result.wt2.iloc[i]
            mfi = mf_result.mfi.iloc[i]
            vwap = vwap_result.vwap.iloc[i]  # Use REAL VWAP, not WT momentum

            # Analyze regime and potentially switch mode (only when not in trade)
            if not active_trade:
                analysis = regime_detector.analyze(
                    wt_result.wt2.iloc[:i+1],
                    timestamp=current_time
                )

                new_mode = regime_detector.get_mode_for_signal(
                    wt_result.wt2.iloc[:i+1],
                    current_mode,
                    timestamp=current_time
                )

                if new_mode != current_mode:
                    mode_switches += 1
                    current_mode = new_mode

            # Track bars in each mode
            if current_mode == "simple":
                bars_in_simple += 1
            else:
                bars_in_enhanced += 1

            # Get signal based on current mode
            if current_mode == "simple":
                signal = simple_detector.process_bar(
                    timestamp=current_time,
                    close_price=current_price,
                    wt1=wt1,
                    wt2=wt2,
                    mfi=mfi,
                    vwap=vwap
                )
            else:
                wt_subset = type(wt_result)(
                    wt1=wt_result.wt1.iloc[:i+1],
                    wt2=wt_result.wt2.iloc[:i+1],
                    momentum=wt_result.momentum.iloc[:i+1],
                    cross=wt_result.cross.iloc[:i+1],
                    cross_up=wt_result.cross_up.iloc[:i+1],
                    cross_down=wt_result.cross_down.iloc[:i+1],
                    oversold=wt_result.oversold.iloc[:i+1],
                    overbought=wt_result.overbought.iloc[:i+1]
                )
                mf_subset = type(mf_result)(
                    mfi=mf_result.mfi.iloc[:i+1],
                    is_positive=mf_result.is_positive.iloc[:i+1],
                    is_negative=mf_result.is_negative.iloc[:i+1],
                    curving_up=mf_result.curving_up.iloc[:i+1],
                    curving_down=mf_result.curving_down.iloc[:i+1]
                )
                signal = enhanced_detector.process_bar(
                    timestamp=current_time,
                    close_price=current_price,
                    wt_result=wt_subset,
                    mf_result=mf_subset,
                    bar_idx=i
                )

            # Check active trade exit
            if active_trade:
                exit_reason = None
                is_long = active_trade.signal_type == SignalType.LONG

                # Update trailing stop if enabled
                if self.use_trailing_sl and self.use_stop_loss and initial_stop_loss != 0:
                    new_sl = self.risk_manager.calculate_trailing_stop(
                        current_price=current_price,
                        entry_price=active_trade.entry_price,
                        stop_loss_price=initial_stop_loss,
                        current_stop=active_trade.stop_loss,
                        is_long=is_long,
                        trailing_distance_percent=self.trailing_distance_percent,
                        activation_rr=self.trailing_activation_rr
                    )
                    if is_long and new_sl > active_trade.stop_loss:
                        active_trade.stop_loss = new_sl
                    elif not is_long and new_sl < active_trade.stop_loss:
                        active_trade.stop_loss = new_sl

                if self.use_stop_loss:
                    if is_long:
                        if current_low <= active_trade.stop_loss:
                            exit_reason = "stop_loss"
                    else:
                        if current_high >= active_trade.stop_loss:
                            exit_reason = "stop_loss"

                if not exit_reason:
                    if self.tp_method == TakeProfitMethod.FIXED_RR:
                        if is_long:
                            if current_high >= active_trade.take_profit:
                                exit_reason = "take_profit"
                        else:
                            if current_low <= active_trade.take_profit:
                                exit_reason = "take_profit"
                    elif self.tp_method == TakeProfitMethod.OSCILLATOR:
                        exit_reason = self._check_oscillator_exit(
                            trade=active_trade,
                            wt1=wt1, wt2=wt2,
                            prev_wt1=prev_wt1, prev_wt2=prev_wt2,
                            mfi=mfi, prev_mfi=prev_mfi,
                            vwap=vwap, prev_vwap=prev_vwap,
                            current_signal=signal,
                            mode=self.oscillator_mode
                        )

                if exit_reason:
                    active_trade = self._close_trade(
                        active_trade, current_price, current_time, exit_reason
                    )
                    commission = (active_trade.exit_price * active_trade.size) * (self.commission_percent / 100)
                    active_trade.pnl -= commission
                    balance += active_trade.pnl
                    trades.append(active_trade)
                    active_trade = None

            # Check for new signals
            if not active_trade and signal:
                can_trade, position_multiplier = self._check_time_filter(current_time)

                if can_trade:
                    active_trade = self._open_trade(
                        signal, balance, df.iloc[:i+1], current_time,
                        position_multiplier=position_multiplier
                    )
                    initial_stop_loss = active_trade.stop_loss
                    commission = (active_trade.entry_price * active_trade.size) * (self.commission_percent / 100)
                    balance -= commission

                    # Track which mode generated this trade
                    if current_mode == "simple":
                        trades_in_simple += 1
                    else:
                        trades_in_enhanced += 1

            # Update equity curve
            if active_trade:
                unrealized_pnl = self._calculate_unrealized_pnl(active_trade, current_price)
                equity_curve.append(balance + unrealized_pnl)
            else:
                equity_curve.append(balance)

            prev_wt1 = wt1
            prev_wt2 = wt2
            prev_mfi = mfi
            prev_vwap = vwap

        # Close remaining trade
        if active_trade:
            last_price = df.iloc[-1]['close']
            last_time = df.index[-1]
            active_trade = self._close_trade(active_trade, last_price, last_time, "end_of_data")
            balance += active_trade.pnl
            trades.append(active_trade)

        # Log adaptive mode stats
        total_bars = bars_in_simple + bars_in_enhanced
        logger.info(f"Adaptive Mode Stats: {mode_switches} switches | "
                   f"Simple: {bars_in_simple/total_bars*100:.1f}% time, {trades_in_simple} trades | "
                   f"Enhanced: {bars_in_enhanced/total_bars*100:.1f}% time, {trades_in_enhanced} trades")

        return self._calculate_results(trades, equity_curve)

    def run_mtf_v4(
        self,
        htf_data: pd.DataFrame,
        ltf_data: pd.DataFrame,
        htf_timeframe: str = "4h",
        ltf_timeframe: str = "1h",
        ltf_signal_mode: SignalMode = SignalMode.SIMPLE,
        direction_filter: str = "both",
        bias_wt2_threshold: float = 30.0,
        require_mfi_confirm: bool = True,
        allow_neutral_trades: bool = False
    ) -> BacktestResult:
        """
        Run V4 Multi-Timeframe backtest with 4H bias + configurable LTF entry.

        This is the primary V4 backtest method that:
        1. Uses HTF (default 4H) data to determine market bias
        2. Uses LTF (default 1H) data for entry signals
        3. Filters entries based on HTF bias alignment
        4. Supports direction filtering (long-only, short-only, or both)

        Args:
            htf_data: DataFrame with HTF OHLCV data (for bias)
            ltf_data: DataFrame with LTF OHLCV data (for entry)
            htf_timeframe: HTF timeframe string (e.g., '4h')
            ltf_timeframe: LTF timeframe string (e.g., '1h', '30m', '15m')
            ltf_signal_mode: Signal mode for LTF entries (SIMPLE or ENHANCED)
            direction_filter: Trade direction filter ('both', 'long_only', 'short_only')
            bias_wt2_threshold: WT2 threshold for HTF bias detection
            require_mfi_confirm: Require MFI confirmation for HTF bias
            allow_neutral_trades: Allow trades when HTF bias is neutral

        Returns:
            BacktestResult with performance metrics and V4-specific info
        """
        logger.info(f"Running V4 MTF backtest: HTF={htf_timeframe}, LTF={ltf_timeframe}, "
                   f"LTF_Mode={ltf_signal_mode.value if hasattr(ltf_signal_mode, 'value') else ltf_signal_mode}, "
                   f"Direction={direction_filter}")

        # Initialize V4 signal detector
        v4_detector = V4SignalDetector(
            signal_mode=ltf_signal_mode,
            anchor_level_long=self.wavetrend.oversold_2,
            anchor_level_short=self.wavetrend.overbought_2,
            trigger_lookback=20,
            simple_oversold=self.simple_oversold,
            simple_overbought=self.simple_overbought,
            direction_filter=direction_filter,
            bias_wt2_threshold=bias_wt2_threshold,
            require_mfi_confirm=require_mfi_confirm,
            allow_neutral_trades=allow_neutral_trades,
            timeframe=ltf_timeframe
        )

        # Convert both HTF and LTF data to Heikin Ashi
        htf_ha = convert_to_heikin_ashi(htf_data)
        ltf_ha = convert_to_heikin_ashi(ltf_data)

        # Calculate indicators for both timeframes
        htf_wt = self.wavetrend.calculate(htf_ha)
        htf_mf = self.money_flow.calculate(htf_ha)
        ltf_wt = self.wavetrend.calculate(ltf_ha)
        ltf_mf = self.money_flow.calculate(ltf_ha)
        htf_vwap = self.vwap_calc.calculate(htf_data)  # HTF VWAP
        ltf_vwap = self.vwap_calc.calculate(ltf_data)  # LTF VWAP

        # Build HTF timestamp -> index mapping
        htf_timestamps = htf_data.index.tolist()
        htf_index_map = {ts: i for i, ts in enumerate(htf_timestamps)}

        # Trading state
        balance = self.initial_balance
        equity_curve = [balance]
        trades: List[BacktestTrade] = []
        active_trade: Optional[BacktestTrade] = None
        initial_stop_loss: float = 0.0  # Track initial SL for trailing stop calculation

        # V4 tracking stats
        htf_bias_updates = 0
        ltf_signals_generated = 0
        signals_filtered_by_direction = 0
        signals_filtered_by_bias = 0
        signals_taken = 0

        # Previous indicator values for exit detection
        prev_wt1 = 0.0
        prev_wt2 = 0.0
        prev_mfi = 0.0
        prev_vwap = 0.0

        # Track last HTF bar processed
        last_htf_idx = -1

        # Process each LTF bar
        for i in range(100, len(ltf_data)):
            current_bar = ltf_data.iloc[i]
            current_price = current_bar['close']
            current_high = current_bar['high']
            current_low = current_bar['low']
            current_time = ltf_data.index[i]

            # Find corresponding HTF bar (most recent HTF bar <= current LTF time)
            htf_idx = self._find_htf_bar_for_ltf(current_time, htf_timestamps)

            # Update HTF bias if we have a new HTF bar
            if htf_idx is not None and htf_idx > last_htf_idx and htf_idx >= 2:
                last_htf_idx = htf_idx
                htf_bias_updates += 1

                # Get HTF indicator values
                htf_wt2 = htf_wt.wt2.iloc[htf_idx]
                htf_mfi = htf_mf.mfi.iloc[htf_idx]
                htf_prev_mfi = htf_mf.mfi.iloc[htf_idx - 1]
                htf_prev_prev_mfi = htf_mf.mfi.iloc[htf_idx - 2]

                # Update V4 detector with new HTF bias
                v4_detector.update_htf_bias(
                    wt2=htf_wt2,
                    mfi=htf_mfi,
                    prev_mfi=htf_prev_mfi,
                    prev_prev_mfi=htf_prev_prev_mfi,
                    timestamp=htf_timestamps[htf_idx]
                )

            # Get LTF indicator values
            ltf_wt1 = ltf_wt.wt1.iloc[i]
            ltf_wt2 = ltf_wt.wt2.iloc[i]
            ltf_mfi = ltf_mf.mfi.iloc[i]
            ltf_vwap = ltf_wt.vwap.iloc[i]

            # Create LTF indicator subsets for ENHANCED mode
            ltf_wt_subset = type(ltf_wt)(
                wt1=ltf_wt.wt1.iloc[:i+1],
                wt2=ltf_wt.wt2.iloc[:i+1],
                vwap=ltf_wt.vwap.iloc[:i+1],
                cross=ltf_wt.cross.iloc[:i+1],
                cross_up=ltf_wt.cross_up.iloc[:i+1],
                cross_down=ltf_wt.cross_down.iloc[:i+1],
                oversold=ltf_wt.oversold.iloc[:i+1],
                overbought=ltf_wt.overbought.iloc[:i+1]
            )

            ltf_mf_subset = type(ltf_mf)(
                mfi=ltf_mf.mfi.iloc[:i+1],
                is_positive=ltf_mf.is_positive.iloc[:i+1],
                is_negative=ltf_mf.is_negative.iloc[:i+1],
                curving_up=ltf_mf.curving_up.iloc[:i+1],
                curving_down=ltf_mf.curving_down.iloc[:i+1]
            )

            # Get filtered signal from V4 detector
            signal = v4_detector.process_ltf_bar(
                timestamp=current_time,
                close_price=current_price,
                wt1=ltf_wt1,
                wt2=ltf_wt2,
                mfi=ltf_mfi,
                vwap=ltf_vwap,
                wt_result=ltf_wt_subset,
                mf_result=ltf_mf_subset,
                bar_idx=i
            )

            # Check active trade exit conditions
            if active_trade:
                exit_reason = None
                is_long = active_trade.signal_type == SignalType.LONG

                # Update trailing stop if enabled
                if self.use_trailing_sl and self.use_stop_loss and initial_stop_loss != 0:
                    new_sl = self.risk_manager.calculate_trailing_stop(
                        current_price=current_price,
                        entry_price=active_trade.entry_price,
                        stop_loss_price=initial_stop_loss,
                        current_stop=active_trade.stop_loss,
                        is_long=is_long,
                        trailing_distance_percent=self.trailing_distance_percent,
                        activation_rr=self.trailing_activation_rr
                    )
                    if is_long and new_sl > active_trade.stop_loss:
                        active_trade.stop_loss = new_sl
                    elif not is_long and new_sl < active_trade.stop_loss:
                        active_trade.stop_loss = new_sl

                # Check stop loss first (only if enabled)
                if self.use_stop_loss:
                    if is_long:
                        if current_low <= active_trade.stop_loss:
                            exit_reason = "stop_loss"
                    else:
                        if current_high >= active_trade.stop_loss:
                            exit_reason = "stop_loss"

                # Check take profit
                if not exit_reason:
                    if self.tp_method == TakeProfitMethod.FIXED_RR:
                        if is_long:
                            if current_high >= active_trade.take_profit:
                                exit_reason = "take_profit"
                        else:
                            if current_low <= active_trade.take_profit:
                                exit_reason = "take_profit"
                    elif self.tp_method == TakeProfitMethod.OSCILLATOR:
                        exit_reason = self._check_oscillator_exit(
                            trade=active_trade,
                            wt1=ltf_wt1, wt2=ltf_wt2,
                            prev_wt1=prev_wt1, prev_wt2=prev_wt2,
                            mfi=ltf_mfi, prev_mfi=prev_mfi,
                            vwap=ltf_vwap, prev_vwap=prev_vwap,
                            current_signal=signal,
                            mode=self.oscillator_mode
                        )

                if exit_reason:
                    active_trade = self._close_trade(
                        active_trade, current_price, current_time, exit_reason
                    )
                    commission = (active_trade.exit_price * active_trade.size) * (self.commission_percent / 100)
                    active_trade.pnl -= commission
                    balance += active_trade.pnl
                    trades.append(active_trade)
                    active_trade = None

            # Check for new signals (signal is already filtered by V4 detector)
            if not active_trade and signal:
                signals_taken += 1

                # Check time filter
                can_trade, position_multiplier = self._check_time_filter(current_time)

                if can_trade:
                    active_trade = self._open_trade(
                        signal, balance, ltf_data.iloc[:i+1], current_time,
                        position_multiplier=position_multiplier
                    )
                    commission = (active_trade.entry_price * active_trade.size) * (self.commission_percent / 100)
                    balance -= commission
                    # Store initial stop loss for trailing stop calculation
                    initial_stop_loss = active_trade.stop_loss

            # Update equity curve
            if active_trade:
                unrealized_pnl = self._calculate_unrealized_pnl(active_trade, current_price)
                equity_curve.append(balance + unrealized_pnl)
            else:
                equity_curve.append(balance)

            # Store previous values
            prev_wt1 = ltf_wt1
            prev_wt2 = ltf_wt2
            prev_mfi = ltf_mfi
            prev_vwap = ltf_vwap

        # Close remaining trade
        if active_trade:
            last_price = ltf_data.iloc[-1]['close']
            last_time = ltf_data.index[-1]
            active_trade = self._close_trade(active_trade, last_price, last_time, "end_of_data")
            balance += active_trade.pnl
            trades.append(active_trade)

        # Get V4 detector stats
        v4_stats = v4_detector.get_stats()

        # Log V4 filter stats
        logger.info(f"V4 MTF Stats: HTF bias updates={htf_bias_updates} | "
                   f"Signals taken={signals_taken} | Filtered={v4_stats['filtered_signals']} | "
                   f"Direction filter={direction_filter}")

        return self._calculate_results(trades, equity_curve)

    def _find_htf_bar_for_ltf(
        self,
        ltf_timestamp: datetime,
        htf_timestamps: List[datetime]
    ) -> Optional[int]:
        """
        Find the most recent HTF bar that is <= LTF timestamp.

        Args:
            ltf_timestamp: Current LTF bar timestamp
            htf_timestamps: List of HTF timestamps

        Returns:
            Index of the HTF bar, or None if no valid bar found
        """
        # Binary search for efficiency
        left, right = 0, len(htf_timestamps) - 1
        result = None

        while left <= right:
            mid = (left + right) // 2
            if htf_timestamps[mid] <= ltf_timestamp:
                result = mid
                left = mid + 1
            else:
                right = mid - 1

        return result

    def _open_trade(
        self,
        signal: Signal,
        balance: float,
        df: pd.DataFrame,
        current_time: datetime,
        position_multiplier: float = 1.0
    ) -> BacktestTrade:
        """Open a new trade based on signal.

        Args:
            signal: Trade signal
            balance: Current account balance
            df: Price data for stop loss calculation
            current_time: Entry timestamp
            position_multiplier: Size multiplier (1.0=full, 0.5=half for time filter)
        """
        is_long = signal.signal_type == SignalType.LONG
        entry_price = signal.entry_price

        # Apply slippage (buying at higher price for long, lower for short)
        if self.slippage_percent > 0:
            slippage_factor = self.slippage_percent / 100
            if is_long:
                entry_price = entry_price * (1 + slippage_factor)  # Worse fill for longs
            else:
                entry_price = entry_price * (1 - slippage_factor)  # Worse fill for shorts

        # Calculate stop loss
        stop_loss = self.risk_manager.calculate_stop_loss(
            entry_price=entry_price,
            is_long=is_long,
            df=df,
            method=self.sl_method
        )

        # Calculate take profit
        take_profit = self.risk_manager.calculate_take_profit(
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            is_long=is_long,
            risk_reward=self.risk_reward
        )

        # Calculate position size (with time filter multiplier)
        size = self.risk_manager.calculate_position_size(
            account_balance=balance,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            risk_percent=self.risk_percent
        )

        # Apply position multiplier (v3 time filter)
        size = size * position_multiplier

        return BacktestTrade(
            entry_time=current_time,
            exit_time=None,
            signal_type=signal.signal_type,
            entry_price=entry_price,
            exit_price=None,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            wt1_entry=signal.wt1,
            wt2_entry=signal.wt2,
            mfi_entry=signal.mfi,
            vwap_entry=signal.vwap
        )

    def _check_exit(
        self,
        trade: BacktestTrade,
        current_price: float,
        high: float,
        low: float
    ) -> Optional[str]:
        """Check if trade should be closed."""
        is_long = trade.signal_type == SignalType.LONG

        if is_long:
            # Check stop loss (use low for worst-case) - only if enabled
            if self.use_stop_loss and low <= trade.stop_loss:
                return "stop_loss"
            # Check take profit (use high for best-case)
            if high >= trade.take_profit:
                return "take_profit"
        else:
            # Short position
            if self.use_stop_loss and high >= trade.stop_loss:
                return "stop_loss"
            if low <= trade.take_profit:
                return "take_profit"

        return None

    def _check_oscillator_exit(
        self,
        trade: BacktestTrade,
        wt1: float,
        wt2: float,
        prev_wt1: float,
        prev_wt2: float,
        mfi: float,
        prev_mfi: float,
        vwap: float,
        prev_vwap: float,
        current_signal: Optional[Signal],
        mode: OscillatorExitMode
    ) -> Optional[str]:
        """
        Check oscillator-based exit conditions.

        Args:
            trade: Active trade to check
            wt1, wt2: Current WaveTrend values
            prev_wt1, prev_wt2: Previous bar WaveTrend values
            mfi: Current Money Flow value
            prev_mfi: Previous bar Money Flow value
            vwap: Current VWAP (wt1 - wt2)
            prev_vwap: Previous bar VWAP
            current_signal: Current detected signal (if any)
            mode: Oscillator exit mode

        Returns:
            Exit reason string or None if no exit
        """
        is_long = trade.signal_type == SignalType.LONG

        if mode == OscillatorExitMode.FULL_SIGNAL:
            # Exit on complete opposite VMC signal
            if current_signal:
                if is_long and current_signal.signal_type == SignalType.SHORT:
                    return "osc_full_signal"
                elif not is_long and current_signal.signal_type == SignalType.LONG:
                    return "osc_full_signal"

        elif mode == OscillatorExitMode.WT_CROSS:
            # Exit when WT1 crosses WT2 in opposite direction
            if is_long:
                # Bearish cross: WT1 crosses below WT2
                if prev_wt1 >= prev_wt2 and wt1 < wt2:
                    return "osc_wt_cross"
            else:
                # Bullish cross: WT1 crosses above WT2
                if prev_wt1 <= prev_wt2 and wt1 > wt2:
                    return "osc_wt_cross"

        elif mode == OscillatorExitMode.FIRST_REVERSAL:
            # Exit on first sign of reversal (most aggressive)
            if is_long:
                # Any bearish sign exits the long
                wt_cross_down = prev_wt1 >= prev_wt2 and wt1 < wt2
                mfi_curving_down = mfi < prev_mfi and prev_mfi > 0
                vwap_cross_down = prev_vwap >= 0 and vwap < 0
                if wt_cross_down or mfi_curving_down or vwap_cross_down:
                    return "osc_first_reversal"
            else:
                # Any bullish sign exits the short
                wt_cross_up = prev_wt1 <= prev_wt2 and wt1 > wt2
                mfi_curving_up = mfi > prev_mfi and prev_mfi < 0
                vwap_cross_up = prev_vwap <= 0 and vwap > 0
                if wt_cross_up or mfi_curving_up or vwap_cross_up:
                    return "osc_first_reversal"

        return None

    def _close_trade(
        self,
        trade: BacktestTrade,
        exit_price: float,
        exit_time: datetime,
        reason: str
    ) -> BacktestTrade:
        """Close a trade and calculate PnL."""
        is_long = trade.signal_type == SignalType.LONG

        # Determine actual exit price based on reason
        if reason == "stop_loss":
            actual_exit = trade.stop_loss
        elif reason == "take_profit":
            actual_exit = trade.take_profit
        else:
            actual_exit = exit_price

        # Apply slippage (selling at lower price for long, higher for short)
        if self.slippage_percent > 0:
            slippage_factor = self.slippage_percent / 100
            if is_long:
                actual_exit = actual_exit * (1 - slippage_factor)  # Worse fill for long exit
            else:
                actual_exit = actual_exit * (1 + slippage_factor)  # Worse fill for short exit

        trade.exit_price = actual_exit
        trade.exit_time = exit_time
        trade.exit_reason = reason

        # Calculate PnL
        if is_long:
            trade.pnl = (trade.exit_price - trade.entry_price) * trade.size
        else:
            trade.pnl = (trade.entry_price - trade.exit_price) * trade.size

        trade.pnl_percent = (trade.pnl / (trade.entry_price * trade.size)) * 100

        return trade

    def _calculate_unrealized_pnl(self, trade: BacktestTrade, current_price: float) -> float:
        """Calculate unrealized PnL for open trade."""
        if trade.signal_type == SignalType.LONG:
            return (current_price - trade.entry_price) * trade.size
        else:
            return (trade.entry_price - current_price) * trade.size

    def _calculate_results(
        self,
        trades: List[BacktestTrade],
        equity_curve: List[float]
    ) -> BacktestResult:
        """Calculate backtest performance metrics."""
        if not trades:
            return BacktestResult(
                initial_balance=self.initial_balance,
                final_balance=self.initial_balance,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                total_pnl=0,
                total_pnl_percent=0,
                max_drawdown=0,
                max_drawdown_percent=0,
                sharpe_ratio=0,
                profit_factor=0,
                avg_win=0,
                avg_loss=0,
                largest_win=0,
                largest_loss=0,
                avg_trade_duration=0,
                trades=[],
                equity_curve=equity_curve
            )

        # Basic stats
        total_trades = len(trades)
        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if not t.is_winner]

        winning_trades = len(winners)
        losing_trades = len(losers)
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

        # PnL stats
        total_pnl = sum(t.pnl for t in trades)
        total_pnl_percent = (total_pnl / self.initial_balance) * 100
        final_balance = self.initial_balance + total_pnl

        # Win/Loss stats
        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0

        avg_win = gross_profit / len(winners) if winners else 0
        avg_loss = gross_loss / len(losers) if losers else 0
        largest_win = max((t.pnl for t in winners), default=0)
        largest_loss = min((t.pnl for t in losers), default=0)

        # Profit factor
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Drawdown
        max_dd, max_dd_pct = self._calculate_max_drawdown(equity_curve)

        # Sharpe ratio (simplified)
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 1 and returns.std() > 0 else 0

        return BacktestResult(
            initial_balance=self.initial_balance,
            final_balance=final_balance,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            max_drawdown=max_dd,
            max_drawdown_percent=max_dd_pct,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade_duration=0,
            trades=trades,
            equity_curve=equity_curve
        )

    def _calculate_max_drawdown(
        self,
        equity_curve: List[float]
    ) -> Tuple[float, float]:
        """Calculate maximum drawdown."""
        peak = equity_curve[0]
        max_dd = 0
        max_dd_pct = 0

        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            dd_pct = (dd / peak) * 100 if peak > 0 else 0

            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct

        return max_dd, max_dd_pct

    def _check_time_filter(self, timestamp: datetime) -> Tuple[bool, float]:
        """
        Check if trading is allowed at the given timestamp.

        Args:
            timestamp: Current bar timestamp (assumed UTC)

        Returns:
            Tuple of (can_trade, position_multiplier):
            - (True, 1.0) = fully allowed
            - (False, 0.0) = blocked (skip mode)
            - (True, multiplier) = allowed with reduced size
        """
        if not self.time_filter_enabled:
            return (True, 1.0)

        hour = timestamp.hour
        day_of_week = timestamp.weekday()  # 0=Monday, 6=Sunday

        # Check weekends
        is_weekend = day_of_week >= 5  # Saturday=5, Sunday=6

        if is_weekend:
            if not self.trade_weekends:
                if self.time_filter_mode == TimeFilterMode.SKIP:
                    return (False, 0.0)
                else:
                    return (True, self.reduced_position_multiplier)
            # Weekend trading allowed
            return (True, 1.0)

        # Check US market hours (weekdays only)
        if self.avoid_us_market_hours:
            is_market_hours = self.market_hours_start_utc <= hour < self.market_hours_end_utc

            if is_market_hours:
                if self.time_filter_mode == TimeFilterMode.SKIP:
                    return (False, 0.0)
                else:
                    return (True, self.reduced_position_multiplier)

        # Off hours - fully allowed
        return (True, 1.0)
