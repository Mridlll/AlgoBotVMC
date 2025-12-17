#!/usr/bin/env python3
"""
VMC Trading Bot - Client Backtest Report
=========================================

This script runs comprehensive backtests and generates a professional report.
Data Source: Real historical data from Binance public API (no API key required)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
import requests
import time

from indicators import HeikinAshi, WaveTrend, MoneyFlow
from indicators.heikin_ashi import convert_to_heikin_ashi
from strategy.signals import SignalDetector, Signal, SignalType, SignalState
from strategy.risk import RiskManager, StopLossMethod


def fetch_binance_data(symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
    """
    Fetch real historical data from Binance public API.
    No API key required for public market data.
    """
    # Map our intervals to Binance format
    interval_map = {
        '3m': '3m', '5m': '5m', '10m': '15m',  # 10m not available, use 15m
        '15m': '15m', '30m': '30m', '1h': '1h',
        '4h': '4h', '8h': '8h', '12h': '12h', '1d': '1d'
    }

    binance_interval = interval_map.get(interval, interval)

    # Binance futures API for perp data
    url = f"https://fapi.binance.com/fapi/v1/klines"
    params = {
        'symbol': f"{symbol}USDT",
        'interval': binance_interval,
        'limit': limit
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        return df[['open', 'high', 'low', 'close', 'volume']]

    except Exception as e:
        print(f"  Warning: Could not fetch {symbol} {interval} from Binance: {e}")
        return None


@dataclass
class TradeResult:
    """Single trade result."""
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_percent: float
    exit_reason: str


class BacktestEngine:
    """Simplified backtest engine for client report."""

    def __init__(self, initial_balance: float = 10000.0, risk_percent: float = 3.0):
        self.initial_balance = initial_balance
        self.risk_percent = risk_percent
        self.risk_reward = 2.0

        self.wavetrend = WaveTrend(
            channel_len=9, average_len=12, ma_len=3,
            overbought_2=60, oversold_2=-60
        )
        self.money_flow = MoneyFlow()
        self.risk_manager = RiskManager(
            default_risk_percent=risk_percent,
            default_rr=self.risk_reward,
            swing_lookback=5
        )

    def run(self, df: pd.DataFrame, asset: str, timeframe: str) -> Dict:
        """Run backtest and return results."""
        if df is None or len(df) < 150:
            return None

        signal_detector = SignalDetector(
            anchor_level_long=-60,
            anchor_level_short=60,
            trigger_lookback=20
        )

        ha_df = convert_to_heikin_ashi(df)
        wt_result = self.wavetrend.calculate(ha_df)
        mf_result = self.money_flow.calculate(ha_df)

        balance = self.initial_balance
        trades: List[TradeResult] = []
        active_trade = None

        # Track state transitions
        anchors_long = 0
        triggers_long = 0
        vwap_long = 0
        anchors_short = 0
        triggers_short = 0
        vwap_short = 0

        prev_long_state = "IDLE"
        prev_short_state = "IDLE"

        for i in range(100, len(df)):
            current_bar = df.iloc[i]
            current_time = df.index[i]
            current_price = current_bar['close']
            current_high = current_bar['high']
            current_low = current_bar['low']

            # Check exit
            if active_trade:
                exit_reason = None
                exit_price = None

                if active_trade['direction'] == 'LONG':
                    if current_low <= active_trade['stop_loss']:
                        exit_reason = 'Stop Loss'
                        exit_price = active_trade['stop_loss']
                    elif current_high >= active_trade['take_profit']:
                        exit_reason = 'Take Profit'
                        exit_price = active_trade['take_profit']
                else:
                    if current_high >= active_trade['stop_loss']:
                        exit_reason = 'Stop Loss'
                        exit_price = active_trade['stop_loss']
                    elif current_low <= active_trade['take_profit']:
                        exit_reason = 'Take Profit'
                        exit_price = active_trade['take_profit']

                if exit_reason:
                    if active_trade['direction'] == 'LONG':
                        pnl = (exit_price - active_trade['entry_price']) * active_trade['size']
                    else:
                        pnl = (active_trade['entry_price'] - exit_price) * active_trade['size']

                    pnl_pct = (pnl / (active_trade['entry_price'] * active_trade['size'])) * 100

                    trades.append(TradeResult(
                        entry_time=active_trade['entry_time'],
                        exit_time=current_time,
                        direction=active_trade['direction'],
                        entry_price=active_trade['entry_price'],
                        exit_price=exit_price,
                        pnl=pnl,
                        pnl_percent=pnl_pct,
                        exit_reason=exit_reason
                    ))

                    balance += pnl
                    active_trade = None

            # Check for new signals
            if not active_trade:
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

                # Track state transitions
                state_before = signal_detector.get_current_state()

                signal = signal_detector.process_bar(
                    timestamp=current_time,
                    close_price=current_price,
                    wt_result=wt_subset,
                    mf_result=mf_subset,
                    bar_idx=i
                )

                state_after = signal_detector.get_current_state()

                # Count transitions
                if state_after['long_state'] != prev_long_state:
                    if state_after['long_state'] == 'ANCHOR_DETECTED':
                        anchors_long += 1
                    elif state_after['long_state'] == 'TRIGGER_DETECTED':
                        triggers_long += 1
                    elif state_after['long_state'] == 'AWAITING_VWAP':
                        vwap_long += 1
                    prev_long_state = state_after['long_state']

                if state_after['short_state'] != prev_short_state:
                    if state_after['short_state'] == 'ANCHOR_DETECTED':
                        anchors_short += 1
                    elif state_after['short_state'] == 'TRIGGER_DETECTED':
                        triggers_short += 1
                    elif state_after['short_state'] == 'AWAITING_VWAP':
                        vwap_short += 1
                    prev_short_state = state_after['short_state']

                if signal:
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

                    active_trade = {
                        'entry_time': current_time,
                        'direction': 'LONG' if is_long else 'SHORT',
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'size': size
                    }

        # Calculate metrics
        if not trades:
            return {
                'asset': asset,
                'timeframe': timeframe,
                'data_source': 'Binance Futures',
                'period_start': df.index[0],
                'period_end': df.index[-1],
                'candles': len(df),
                'total_trades': 0,
                'long_trades': 0,
                'short_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_pnl_percent': 0,
                'max_drawdown': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'state_flow': {
                    'long': {'anchors': anchors_long, 'triggers': triggers_long, 'vwap': vwap_long},
                    'short': {'anchors': anchors_short, 'triggers': triggers_short, 'vwap': vwap_short}
                },
                'trades': []
            }

        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]
        long_trades = [t for t in trades if t.direction == 'LONG']
        short_trades = [t for t in trades if t.direction == 'SHORT']

        total_pnl = sum(t.pnl for t in trades)
        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0.01

        return {
            'asset': asset,
            'timeframe': timeframe,
            'data_source': 'Binance Futures',
            'period_start': df.index[0],
            'period_end': df.index[-1],
            'candles': len(df),
            'total_trades': len(trades),
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'win_rate': len(winners) / len(trades) * 100,
            'total_pnl': total_pnl,
            'total_pnl_percent': (total_pnl / self.initial_balance) * 100,
            'profit_factor': gross_profit / gross_loss,
            'avg_win': gross_profit / len(winners) if winners else 0,
            'avg_loss': gross_loss / len(losers) if losers else 0,
            'state_flow': {
                'long': {'anchors': anchors_long, 'triggers': triggers_long, 'vwap': vwap_long},
                'short': {'anchors': anchors_short, 'triggers': triggers_short, 'vwap': vwap_short}
            },
            'trades': trades
        }


def print_client_report(results: List[Dict]):
    """Print professional client-facing report."""

    print("\n")
    print("=" * 80)
    print("                    VMC TRADING BOT - BACKTEST REPORT")
    print("=" * 80)
    print()

    # Data source info
    valid_results = [r for r in results if r is not None and r['total_trades'] > 0]
    if valid_results:
        r = valid_results[0]
        print(f"  Data Source:    {r['data_source']} (Real Historical Data)")
        print(f"  Test Period:    {r['period_start'].strftime('%Y-%m-%d')} to {r['period_end'].strftime('%Y-%m-%d')}")
        print(f"  Initial Capital: $10,000")
        print(f"  Risk Per Trade:  3%")
        print(f"  Risk:Reward:     1:2")
        print()

    print("-" * 80)
    print("                           PERFORMANCE BY ASSET & TIMEFRAME")
    print("-" * 80)
    print()
    print(f"  {'Asset':<6} {'Timeframe':<10} {'Trades':>8} {'Win Rate':>10} {'Net P&L':>14} {'Return':>10} {'PF':>8}")
    print(f"  {'-'*6} {'-'*10} {'-'*8} {'-'*10} {'-'*14} {'-'*10} {'-'*8}")

    for r in results:
        if r is None:
            continue

        pnl_str = f"${r['total_pnl']:+,.2f}" if r['total_pnl'] != 0 else "$0.00"
        ret_str = f"{r['total_pnl_percent']:+.2f}%" if r['total_pnl_percent'] != 0 else "0.00%"
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] > 0 else "N/A"

        # Color coding hint
        status = "✓" if r['total_pnl'] > 0 else "✗" if r['total_pnl'] < 0 else "-"

        print(f"  {r['asset']:<6} {r['timeframe']:<10} {r['total_trades']:>8} {r['win_rate']:>9.1f}% {pnl_str:>14} {ret_str:>10} {pf_str:>8}")

    print()
    print("-" * 80)
    print("                              STRATEGY EXECUTION FLOW")
    print("-" * 80)
    print()
    print("  The VMC strategy follows a 4-step signal detection process:")
    print()
    print("  Step 1: ANCHOR WAVE    - WT2 reaches extreme level (-60 for longs, +60 for shorts)")
    print("  Step 2: TRIGGER WAVE   - Price makes a higher low (long) or lower high (short)")
    print("  Step 3: MFI CONFIRM    - Money Flow curves in trade direction")
    print("  Step 4: VWAP CROSS     - VWAP crosses zero (final confirmation)")
    print()
    print(f"  {'Asset':<6} {'TF':<6} {'Direction':<10} {'Anchors':>10} {'Triggers':>10} {'VWAP Wait':>10} {'Signals':>10}")
    print(f"  {'-'*6} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for r in results:
        if r is None or 'state_flow' not in r:
            continue

        sf = r['state_flow']
        long_signals = len([t for t in r.get('trades', []) if t.direction == 'LONG'])
        short_signals = len([t for t in r.get('trades', []) if t.direction == 'SHORT'])

        print(f"  {r['asset']:<6} {r['timeframe']:<6} {'LONG':<10} {sf['long']['anchors']:>10} {sf['long']['triggers']:>10} {sf['long']['vwap']:>10} {long_signals:>10}")
        print(f"  {'':<6} {'':<6} {'SHORT':<10} {sf['short']['anchors']:>10} {sf['short']['triggers']:>10} {sf['short']['vwap']:>10} {short_signals:>10}")

    print()
    print("-" * 80)
    print("                                  TRADE DETAILS")
    print("-" * 80)

    for r in results:
        if r is None or not r.get('trades'):
            continue

        print(f"\n  {r['asset']} {r['timeframe']} - {len(r['trades'])} trades")
        print(f"  {'#':>4} {'Date':<12} {'Dir':<6} {'Entry':>12} {'Exit':>12} {'P&L':>12} {'Result':<12}")
        print(f"  {'-'*4} {'-'*12} {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

        for i, t in enumerate(r['trades'][:10], 1):  # Show first 10 trades
            date_str = t.entry_time.strftime('%Y-%m-%d')
            pnl_str = f"${t.pnl:+,.2f}"
            result = "WIN" if t.pnl > 0 else "LOSS"

            print(f"  {i:>4} {date_str:<12} {t.direction:<6} ${t.entry_price:>11,.2f} ${t.exit_price:>11,.2f} {pnl_str:>12} {t.exit_reason:<12}")

        if len(r['trades']) > 10:
            print(f"  ... and {len(r['trades']) - 10} more trades")

    # Summary
    print()
    print("=" * 80)
    print("                                    SUMMARY")
    print("=" * 80)
    print()

    total_trades = sum(r['total_trades'] for r in results if r)
    total_pnl = sum(r['total_pnl'] for r in results if r)

    profitable = [r for r in results if r and r['total_pnl'] > 0]
    unprofitable = [r for r in results if r and r['total_pnl'] < 0]

    print(f"  Total Configurations Tested: {len([r for r in results if r])}")
    print(f"  Profitable Configurations:   {len(profitable)}")
    print(f"  Total Trades Executed:       {total_trades}")
    print(f"  Combined Net P&L:            ${total_pnl:+,.2f}")
    print()

    if profitable:
        print("  BEST PERFORMING CONFIGURATIONS:")
        for r in sorted(profitable, key=lambda x: x['total_pnl'], reverse=True)[:3]:
            print(f"    - {r['asset']} {r['timeframe']}: {r['win_rate']:.1f}% win rate, ${r['total_pnl']:+,.2f} profit")

    print()
    print("=" * 80)
    print("  Note: Past performance does not guarantee future results.")
    print("  This backtest uses real historical data from Binance Futures.")
    print("=" * 80)
    print()


def main():
    """Run client backtest report."""

    print("\n" + "=" * 60)
    print("  VMC TRADING BOT - FETCHING REAL MARKET DATA")
    print("=" * 60)

    # Test configuration
    assets = ['BTC', 'ETH', 'SOL']
    timeframes = ['5m', '15m', '1h', '4h']  # Include lower TFs

    engine = BacktestEngine(initial_balance=10000.0, risk_percent=3.0)
    all_results = []

    for asset in assets:
        for tf in timeframes:
            print(f"\n  Fetching {asset} {tf} data from Binance...")

            # Fetch real data
            df = fetch_binance_data(asset, tf, limit=1000)

            if df is not None and len(df) > 150:
                print(f"  Running backtest on {len(df)} candles...")
                result = engine.run(df, asset, tf)
                all_results.append(result)

                if result and result['total_trades'] > 0:
                    print(f"  Result: {result['total_trades']} trades, {result['win_rate']:.1f}% win rate, ${result['total_pnl']:+,.2f}")
                else:
                    print(f"  Result: No trades generated")
            else:
                print(f"  Skipped: Insufficient data")
                all_results.append(None)

            time.sleep(0.2)  # Rate limiting

    # Print report
    print_client_report(all_results)


if __name__ == "__main__":
    main()
