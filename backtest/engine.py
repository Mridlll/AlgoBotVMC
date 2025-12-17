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

from indicators import HeikinAshi, WaveTrend, MoneyFlow
from indicators.heikin_ashi import convert_to_heikin_ashi
from strategy.signals import SignalDetector, Signal, SignalType
from strategy.risk import RiskManager, StopLossMethod
from utils.logger import get_logger

# Import config enums
try:
    from config.config import TakeProfitMethod, OscillatorExitMode
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

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0


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
        oscillator_mode: OscillatorExitMode = OscillatorExitMode.WT_CROSS
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
        """
        self.initial_balance = initial_balance
        self.risk_percent = risk_percent
        self.risk_reward = risk_reward
        self.commission_percent = commission_percent
        self.sl_method = sl_method
        self.swing_lookback = swing_lookback
        self.tp_method = tp_method
        self.oscillator_mode = oscillator_mode

        # Initialize components
        self.heikin_ashi = HeikinAshi()
        self.wavetrend = WaveTrend(
            overbought_2=int(anchor_level_short),
            oversold_2=int(anchor_level_long)
        )
        self.money_flow = MoneyFlow()
        self.signal_detector = SignalDetector(
            anchor_level_long=anchor_level_long,
            anchor_level_short=anchor_level_short,
            trigger_lookback=20
        )
        self.risk_manager = RiskManager(
            default_risk_percent=risk_percent,
            default_rr=risk_reward,
            swing_lookback=swing_lookback
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
        logger.info(f"Running backtest on {len(df)} candles (TP method: {tp_method_name}, Osc mode: {osc_mode_name})...")

        # Reset signal detector
        self.signal_detector.reset()

        # Convert to Heikin Ashi
        ha_df = convert_to_heikin_ashi(df)

        # Calculate indicators on full dataset
        wt_result = self.wavetrend.calculate(ha_df)
        mf_result = self.money_flow.calculate(ha_df)

        # Trading state
        balance = self.initial_balance
        equity_curve = [balance]
        trades: List[BacktestTrade] = []
        active_trade: Optional[BacktestTrade] = None

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
            vwap = wt_result.vwap.iloc[i]

            # Create subset DataFrames for signal detection
            wt_subset = type(wt_result)(
                wt1=wt_result.wt1.iloc[:i+1],
                wt2=wt_result.wt2.iloc[:i+1],
                vwap=wt_result.vwap.iloc[:i+1],
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

            # Always run signal detection (needed for oscillator exits too)
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

                # Always check stop loss first (price-based)
                is_long = active_trade.signal_type == SignalType.LONG
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
                    commission = abs(active_trade.pnl) * (self.commission_percent / 100)
                    active_trade.pnl -= commission

                    # Update balance
                    balance += active_trade.pnl
                    trades.append(active_trade)
                    active_trade = None

            # Check for new signals (only if no active trade)
            if not active_trade and signal:
                # Open new trade
                active_trade = self._open_trade(
                    signal, balance, df.iloc[:i+1], current_time
                )

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

    def _open_trade(
        self,
        signal: Signal,
        balance: float,
        df: pd.DataFrame,
        current_time: datetime
    ) -> BacktestTrade:
        """Open a new trade based on signal."""
        is_long = signal.signal_type == SignalType.LONG
        entry_price = signal.entry_price

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

        # Calculate position size
        size = self.risk_manager.calculate_position_size(
            account_balance=balance,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            risk_percent=self.risk_percent
        )

        return BacktestTrade(
            entry_time=current_time,
            exit_time=None,
            signal_type=signal.signal_type,
            entry_price=entry_price,
            exit_price=None,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit
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
            # Check stop loss (use low for worst-case)
            if low <= trade.stop_loss:
                return "stop_loss"
            # Check take profit (use high for best-case)
            if high >= trade.take_profit:
                return "take_profit"
        else:
            # Short position
            if high >= trade.stop_loss:
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
        # Determine actual exit price based on reason
        if reason == "stop_loss":
            actual_exit = trade.stop_loss
        elif reason == "take_profit":
            actual_exit = trade.take_profit
        else:
            actual_exit = exit_price

        trade.exit_price = actual_exit
        trade.exit_time = exit_time
        trade.exit_reason = reason

        # Calculate PnL
        if trade.signal_type == SignalType.LONG:
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
