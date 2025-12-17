#!/usr/bin/env python3
"""
Detailed VMC Strategy Backtest Runner

Tests BTC, ETH, SOL across multiple timeframes with comprehensive logging.
"""

import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum

from indicators import HeikinAshi, WaveTrend, MoneyFlow
from indicators.heikin_ashi import convert_to_heikin_ashi
from strategy.signals import SignalDetector, Signal, SignalType, SignalState
from strategy.risk import RiskManager, StopLossMethod

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-5s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class DetailedTrade:
    """Trade with full details."""
    asset: str
    timeframe: str
    signal_type: str
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    anchor_wt2: float
    trigger_wt2: float
    mfi_at_entry: float
    vwap_at_entry: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl: float = 0.0
    pnl_percent: float = 0.0
    bars_held: int = 0


def generate_realistic_data(
    asset: str,
    timeframe: str,
    num_candles: int = 2000,
    base_price: float = 50000.0,
    volatility: float = 0.015
) -> pd.DataFrame:
    """
    Generate realistic OHLCV data with trending behavior.

    Different assets have different characteristics:
    - BTC: Lower volatility, trending
    - ETH: Medium volatility
    - SOL: Higher volatility, more signals
    """
    np.random.seed(hash(f"{asset}_{timeframe}") % 2**32)

    # Asset-specific parameters
    asset_params = {
        'BTC': {'base_price': 98000.0, 'vol': 0.012, 'trend_strength': 0.001},
        'ETH': {'base_price': 3800.0, 'vol': 0.018, 'trend_strength': 0.0015},
        'SOL': {'base_price': 220.0, 'vol': 0.025, 'trend_strength': 0.002},
    }

    params = asset_params.get(asset, asset_params['BTC'])
    base_price = params['base_price']
    volatility = params['vol']
    trend_strength = params['trend_strength']

    # Timeframe to hours mapping
    tf_hours = {
        '5m': 5/60, '15m': 15/60, '30m': 0.5, '1h': 1,
        '4h': 4, '8h': 8, '12h': 12, '1d': 24
    }
    hours = tf_hours.get(timeframe, 4)

    # Generate timestamps
    timestamps = pd.date_range(
        start=datetime(2025, 1, 1),
        periods=num_candles,
        freq=f'{int(hours*60)}min' if hours < 1 else f'{int(hours)}h'
    )

    # Generate price movement with trends and reversals
    prices = [base_price]
    trend = 0

    for i in range(1, num_candles):
        # Trend changes occasionally
        if np.random.random() < 0.02:
            trend = np.random.choice([-1, 0, 1]) * trend_strength

        # Random walk with trend
        change = np.random.randn() * volatility + trend
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)

    prices = np.array(prices)

    # Generate OHLC from prices
    intrabar_vol = volatility * 0.5
    high = prices * (1 + np.abs(np.random.randn(num_candles)) * intrabar_vol)
    low = prices * (1 - np.abs(np.random.randn(num_candles)) * intrabar_vol)
    open_prices = np.roll(prices, 1)
    open_prices[0] = base_price

    # Ensure OHLC consistency
    for i in range(num_candles):
        high[i] = max(high[i], prices[i], open_prices[i])
        low[i] = min(low[i], prices[i], open_prices[i])

    # Generate volume (higher on trend days)
    base_vol = {'BTC': 1e9, 'ETH': 5e8, 'SOL': 2e8}.get(asset, 1e8)
    volume = base_vol * (0.5 + np.random.random(num_candles))

    df = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    }, index=timestamps)
    df.index.name = 'timestamp'

    return df


class DetailedBacktester:
    """Backtest engine with comprehensive logging."""

    def __init__(
        self,
        initial_balance: float = 10000.0,
        risk_percent: float = 3.0,
        risk_reward: float = 2.0,
        commission_percent: float = 0.06
    ):
        self.initial_balance = initial_balance
        self.risk_percent = risk_percent
        self.risk_reward = risk_reward
        self.commission_percent = commission_percent

        # Initialize indicators
        self.wavetrend = WaveTrend(
            channel_len=9,
            average_len=12,
            ma_len=3,
            overbought_2=60,
            oversold_2=-60
        )
        self.money_flow = MoneyFlow()
        self.risk_manager = RiskManager(
            default_risk_percent=risk_percent,
            default_rr=risk_reward,
            swing_lookback=5
        )

    def run_backtest(
        self,
        df: pd.DataFrame,
        asset: str,
        timeframe: str,
        verbose: bool = True
    ) -> Dict:
        """Run backtest with detailed logging."""

        if verbose:
            logger.info("=" * 70)
            logger.info(f"BACKTEST: {asset} on {timeframe}")
            logger.info(f"Period: {df.index[0]} to {df.index[-1]}")
            logger.info(f"Candles: {len(df)}")
            logger.info("=" * 70)

        # Reset signal detector
        signal_detector = SignalDetector(
            anchor_level_long=-60,
            anchor_level_short=60,
            trigger_lookback=20,
            timeframe=timeframe
        )

        # Convert to Heikin Ashi
        ha_df = convert_to_heikin_ashi(df)

        # Calculate indicators on full dataset
        wt_result = self.wavetrend.calculate(ha_df)
        mf_result = self.money_flow.calculate(ha_df)

        # Trading state
        balance = self.initial_balance
        equity_curve = [balance]
        trades: List[DetailedTrade] = []
        active_trade: Optional[DetailedTrade] = None

        # State tracking for logging
        prev_long_state = SignalState.IDLE
        prev_short_state = SignalState.IDLE

        # Signal detection logs
        signal_log = []
        state_transitions = []

        # Process each bar
        for i in range(100, len(df)):
            current_bar = df.iloc[i]
            current_time = df.index[i]
            current_price = current_bar['close']
            current_high = current_bar['high']
            current_low = current_bar['low']

            wt1 = wt_result.wt1.iloc[i]
            wt2 = wt_result.wt2.iloc[i]
            vwap = wt_result.vwap.iloc[i]
            mfi = mf_result.mfi.iloc[i]

            # Check active trade exit conditions
            if active_trade:
                exit_reason = self._check_exit(active_trade, current_high, current_low)

                if exit_reason:
                    # Close trade
                    active_trade = self._close_trade(
                        active_trade, current_time, exit_reason
                    )
                    active_trade.bars_held = i - trades[-1].bars_held if trades else 0

                    # Apply commission
                    commission = abs(active_trade.pnl) * (self.commission_percent / 100)
                    active_trade.pnl -= commission

                    # Update balance
                    balance += active_trade.pnl
                    trades.append(active_trade)

                    if verbose:
                        pnl_str = f"+${active_trade.pnl:.2f}" if active_trade.pnl > 0 else f"-${abs(active_trade.pnl):.2f}"
                        logger.info(
                            f"[{asset}] TRADE CLOSED: {active_trade.signal_type} "
                            f"@ ${active_trade.exit_price:.2f} | "
                            f"{exit_reason.upper()} | PnL: {pnl_str} ({active_trade.pnl_percent:+.2f}%)"
                        )

                    active_trade = None

            # Check for new signals
            if not active_trade:
                # Create subset for signal detection
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

                # Get current state before processing
                current_state = signal_detector.get_current_state()

                signal = signal_detector.process_bar(
                    timestamp=current_time,
                    close_price=current_price,
                    wt_result=wt_subset,
                    mf_result=mf_subset,
                    bar_idx=i
                )

                # Log state transitions
                new_state = signal_detector.get_current_state()

                if new_state['long_state'] != prev_long_state.name:
                    if verbose and new_state['long_state'] != 'IDLE':
                        logger.info(
                            f"[{asset}] LONG State: {prev_long_state.name} -> {new_state['long_state']} "
                            f"| WT2={wt2:.1f} | MFI={mfi:.2f} | VWAP={vwap:.2f}"
                        )
                    state_transitions.append({
                        'time': current_time,
                        'asset': asset,
                        'direction': 'LONG',
                        'from_state': prev_long_state.name,
                        'to_state': new_state['long_state'],
                        'wt2': wt2,
                        'mfi': mfi,
                        'vwap': vwap
                    })
                    prev_long_state = SignalState[new_state['long_state']]

                if new_state['short_state'] != prev_short_state.name:
                    if verbose and new_state['short_state'] != 'IDLE':
                        logger.info(
                            f"[{asset}] SHORT State: {prev_short_state.name} -> {new_state['short_state']} "
                            f"| WT2={wt2:.1f} | MFI={mfi:.2f} | VWAP={vwap:.2f}"
                        )
                    state_transitions.append({
                        'time': current_time,
                        'asset': asset,
                        'direction': 'SHORT',
                        'from_state': prev_short_state.name,
                        'to_state': new_state['short_state'],
                        'wt2': wt2,
                        'mfi': mfi,
                        'vwap': vwap
                    })
                    prev_short_state = SignalState[new_state['short_state']]

                if signal:
                    # Calculate risk parameters
                    is_long = signal.signal_type == SignalType.LONG
                    stop_loss = self.risk_manager.calculate_stop_loss(
                        entry_price=current_price,
                        is_long=is_long,
                        df=df.iloc[:i+1],
                        method=StopLossMethod.SWING
                    )
                    take_profit = self.risk_manager.calculate_take_profit(
                        entry_price=current_price,
                        stop_loss_price=stop_loss,
                        is_long=is_long,
                        risk_reward=self.risk_reward
                    )
                    size = self.risk_manager.calculate_position_size(
                        account_balance=balance,
                        entry_price=current_price,
                        stop_loss_price=stop_loss,
                        risk_percent=self.risk_percent
                    )

                    active_trade = DetailedTrade(
                        asset=asset,
                        timeframe=timeframe,
                        signal_type=signal.signal_type.value.upper(),
                        entry_time=current_time,
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        anchor_wt2=signal.anchor_wave.wt2_value,
                        trigger_wt2=signal.trigger_wave.wt2_value,
                        mfi_at_entry=mfi,
                        vwap_at_entry=vwap,
                        bars_held=i
                    )

                    # Apply entry commission
                    commission = (current_price * size) * (self.commission_percent / 100)
                    balance -= commission

                    signal_log.append({
                        'time': current_time,
                        'asset': asset,
                        'timeframe': timeframe,
                        'signal': signal.signal_type.value,
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'anchor_wt2': signal.anchor_wave.wt2_value,
                        'trigger_wt2': signal.trigger_wave.wt2_value,
                        'mfi': mfi,
                        'vwap': vwap
                    })

                    if verbose:
                        sl_dist = abs(current_price - stop_loss) / current_price * 100
                        tp_dist = abs(take_profit - current_price) / current_price * 100
                        logger.info("")
                        logger.info(f"{'='*60}")
                        logger.info(f"[{asset}] SIGNAL: {signal.signal_type.value.upper()} @ ${current_price:.2f}")
                        logger.info(f"  Anchor WT2:  {signal.anchor_wave.wt2_value:.1f}")
                        logger.info(f"  Trigger WT2: {signal.trigger_wave.wt2_value:.1f}")
                        logger.info(f"  MFI:         {mfi:.2f} ({'negative' if mfi < 0 else 'positive'})")
                        logger.info(f"  VWAP:        {vwap:.2f} ({'crossed above 0' if is_long else 'crossed below 0'})")
                        logger.info(f"  Stop Loss:   ${stop_loss:.2f} ({sl_dist:.2f}%)")
                        logger.info(f"  Take Profit: ${take_profit:.2f} ({tp_dist:.2f}%)")
                        logger.info(f"  Position:    {size:.6f} {asset}")
                        logger.info(f"{'='*60}")
                        logger.info("")

            # Update equity curve
            if active_trade:
                unrealized_pnl = self._calculate_unrealized_pnl(active_trade, current_price)
                equity_curve.append(balance + unrealized_pnl)
            else:
                equity_curve.append(balance)

        # Close any remaining trade
        if active_trade:
            last_price = df.iloc[-1]['close']
            last_time = df.index[-1]
            active_trade.exit_time = last_time
            active_trade.exit_price = last_price
            active_trade.exit_reason = "end_of_data"

            if active_trade.signal_type == 'LONG':
                active_trade.pnl = (last_price - active_trade.entry_price) * \
                    self.risk_manager.calculate_position_size(
                        balance, active_trade.entry_price, active_trade.stop_loss, self.risk_percent
                    )
            else:
                active_trade.pnl = (active_trade.entry_price - last_price) * \
                    self.risk_manager.calculate_position_size(
                        balance, active_trade.entry_price, active_trade.stop_loss, self.risk_percent
                    )

            active_trade.pnl_percent = (active_trade.pnl / (active_trade.entry_price * 1)) * 100
            balance += active_trade.pnl
            trades.append(active_trade)

        # Calculate results
        results = self._calculate_results(trades, equity_curve, asset, timeframe)
        results['signal_log'] = signal_log
        results['state_transitions'] = state_transitions

        return results

    def _check_exit(self, trade: DetailedTrade, high: float, low: float) -> Optional[str]:
        """Check exit conditions."""
        if trade.signal_type == 'LONG':
            if low <= trade.stop_loss:
                return "stop_loss"
            if high >= trade.take_profit:
                return "take_profit"
        else:
            if high >= trade.stop_loss:
                return "stop_loss"
            if low <= trade.take_profit:
                return "take_profit"
        return None

    def _close_trade(self, trade: DetailedTrade, exit_time: datetime, reason: str) -> DetailedTrade:
        """Close trade and calculate PnL."""
        if reason == "stop_loss":
            trade.exit_price = trade.stop_loss
        elif reason == "take_profit":
            trade.exit_price = trade.take_profit

        trade.exit_time = exit_time
        trade.exit_reason = reason

        size = self.risk_manager.calculate_position_size(
            self.initial_balance, trade.entry_price, trade.stop_loss, self.risk_percent
        )

        if trade.signal_type == 'LONG':
            trade.pnl = (trade.exit_price - trade.entry_price) * size
        else:
            trade.pnl = (trade.entry_price - trade.exit_price) * size

        trade.pnl_percent = (trade.pnl / (trade.entry_price * size)) * 100

        return trade

    def _calculate_unrealized_pnl(self, trade: DetailedTrade, current_price: float) -> float:
        """Calculate unrealized PnL."""
        size = self.risk_manager.calculate_position_size(
            self.initial_balance, trade.entry_price, trade.stop_loss, self.risk_percent
        )

        if trade.signal_type == 'LONG':
            return (current_price - trade.entry_price) * size
        else:
            return (trade.entry_price - current_price) * size

    def _calculate_results(
        self,
        trades: List[DetailedTrade],
        equity_curve: List[float],
        asset: str,
        timeframe: str
    ) -> Dict:
        """Calculate performance metrics."""
        if not trades:
            return {
                'asset': asset,
                'timeframe': timeframe,
                'initial_balance': self.initial_balance,
                'final_balance': self.initial_balance,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_pnl_percent': 0,
                'max_drawdown_percent': 0,
                'profit_factor': 0,
                'trades': [],
                'equity_curve': equity_curve
            }

        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]

        total_pnl = sum(t.pnl for t in trades)
        final_balance = self.initial_balance + total_pnl

        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 1

        # Max drawdown
        peak = equity_curve[0]
        max_dd_pct = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd_pct = (peak - equity) / peak * 100 if peak > 0 else 0
            max_dd_pct = max(max_dd_pct, dd_pct)

        return {
            'asset': asset,
            'timeframe': timeframe,
            'initial_balance': self.initial_balance,
            'final_balance': final_balance,
            'total_trades': len(trades),
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'win_rate': len(winners) / len(trades) * 100 if trades else 0,
            'total_pnl': total_pnl,
            'total_pnl_percent': (total_pnl / self.initial_balance) * 100,
            'max_drawdown_percent': max_dd_pct,
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
            'avg_win': gross_profit / len(winners) if winners else 0,
            'avg_loss': gross_loss / len(losers) if losers else 0,
            'trades': trades,
            'equity_curve': equity_curve
        }


def print_results_summary(all_results: List[Dict]):
    """Print comprehensive results summary."""
    print("\n")
    print("=" * 90)
    print("                         BACKTEST RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Asset':<6} {'TF':<6} {'Trades':>7} {'Win%':>7} {'PnL':>12} {'PnL%':>8} {'MaxDD%':>8} {'PF':>6}")
    print("-" * 90)

    for result in all_results:
        pnl_str = f"${result['total_pnl']:,.2f}"
        print(
            f"{result['asset']:<6} "
            f"{result['timeframe']:<6} "
            f"{result['total_trades']:>7} "
            f"{result['win_rate']:>6.1f}% "
            f"{pnl_str:>12} "
            f"{result['total_pnl_percent']:>+7.2f}% "
            f"{result['max_drawdown_percent']:>7.2f}% "
            f"{result['profit_factor']:>6.2f}"
        )

    print("-" * 90)

    # Totals
    total_trades = sum(r['total_trades'] for r in all_results)
    total_pnl = sum(r['total_pnl'] for r in all_results)
    avg_win_rate = sum(r['win_rate'] for r in all_results if r['total_trades'] > 0) / len([r for r in all_results if r['total_trades'] > 0]) if any(r['total_trades'] > 0 for r in all_results) else 0

    print(f"{'TOTAL':<6} {'ALL':<6} {total_trades:>7} {avg_win_rate:>6.1f}% ${total_pnl:>11,.2f}")
    print("=" * 90)


def print_trade_log(trades: List[DetailedTrade], asset: str, timeframe: str):
    """Print detailed trade log."""
    if not trades:
        return

    print(f"\n{'='*80}")
    print(f"TRADE LOG: {asset} {timeframe}")
    print(f"{'='*80}")
    print(f"{'#':>3} {'Type':<6} {'Entry Time':<20} {'Entry':>12} {'Exit':>12} {'P&L':>12} {'Reason':<12}")
    print("-" * 80)

    for i, trade in enumerate(trades, 1):
        entry_time_str = trade.entry_time.strftime('%Y-%m-%d %H:%M') if hasattr(trade.entry_time, 'strftime') else str(trade.entry_time)[:16]
        pnl_str = f"${trade.pnl:+,.2f}"
        exit_price = f"${trade.exit_price:,.2f}" if trade.exit_price else "OPEN"
        print(
            f"{i:>3} "
            f"{trade.signal_type:<6} "
            f"{entry_time_str:<20} "
            f"${trade.entry_price:>11,.2f} "
            f"{exit_price:>12} "
            f"{pnl_str:>12} "
            f"{trade.exit_reason:<12}"
        )

    print("-" * 80)


def main():
    """Run comprehensive backtests."""
    # Test configuration
    assets = ['BTC', 'ETH', 'SOL']
    timeframes = ['15m', '1h', '4h']  # Multiple timeframes

    # Initialize backtester
    backtester = DetailedBacktester(
        initial_balance=10000.0,
        risk_percent=3.0,
        risk_reward=2.0,
        commission_percent=0.06
    )

    all_results = []

    print("\n" + "=" * 70)
    print("         VMC TRADING BOT - DETAILED BACKTEST")
    print("         Assets: BTC, ETH, SOL")
    print("         Timeframes: 15m, 1h, 4h")
    print("         Period: 2025-01-01 to 2025-12-17")
    print("=" * 70 + "\n")

    for asset in assets:
        for timeframe in timeframes:
            # Generate data for this asset/timeframe
            logger.info(f"\nGenerating data for {asset} {timeframe}...")

            # Adjust candle count based on timeframe
            candle_counts = {'5m': 5000, '15m': 3000, '1h': 2000, '4h': 1500}
            num_candles = candle_counts.get(timeframe, 2000)

            df = generate_realistic_data(asset, timeframe, num_candles=num_candles)

            # Run backtest
            results = backtester.run_backtest(df, asset, timeframe, verbose=True)
            all_results.append(results)

            # Print trade log for this asset/timeframe
            if results['trades']:
                print_trade_log(results['trades'], asset, timeframe)

    # Print summary
    print_results_summary(all_results)

    # Print state machine analysis
    print("\n" + "=" * 70)
    print("STATE MACHINE TRANSITION ANALYSIS")
    print("=" * 70)

    for result in all_results:
        transitions = result.get('state_transitions', [])
        if transitions:
            long_anchors = len([t for t in transitions if t['direction'] == 'LONG' and t['to_state'] == 'ANCHOR_DETECTED'])
            long_triggers = len([t for t in transitions if t['direction'] == 'LONG' and t['to_state'] == 'TRIGGER_DETECTED'])
            long_vwap = len([t for t in transitions if t['direction'] == 'LONG' and t['to_state'] == 'AWAITING_VWAP'])

            short_anchors = len([t for t in transitions if t['direction'] == 'SHORT' and t['to_state'] == 'ANCHOR_DETECTED'])
            short_triggers = len([t for t in transitions if t['direction'] == 'SHORT' and t['to_state'] == 'TRIGGER_DETECTED'])
            short_vwap = len([t for t in transitions if t['direction'] == 'SHORT' and t['to_state'] == 'AWAITING_VWAP'])

            print(f"\n{result['asset']} {result['timeframe']}:")
            print(f"  LONG:  Anchors={long_anchors:>3} -> Triggers={long_triggers:>3} -> VWAP={long_vwap:>3} -> Signals={result['winning_trades'] + result['losing_trades'] - short_vwap}")
            print(f"  SHORT: Anchors={short_anchors:>3} -> Triggers={short_triggers:>3} -> VWAP={short_vwap:>3}")

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
