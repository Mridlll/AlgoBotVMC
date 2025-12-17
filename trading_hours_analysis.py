#!/usr/bin/env python3
"""
VMC Trading Bot - Trading Hours Analysis
=========================================
Analyzes strategy performance during US TradFi hours vs off-hours/weekends.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
from typing import List, Dict, Optional, Tuple
import pytz

from indicators import HeikinAshi, WaveTrend, MoneyFlow
from indicators.heikin_ashi import convert_to_heikin_ashi
from strategy.signals import SignalDetector, SignalType


# US Eastern timezone
ET = pytz.timezone('US/Eastern')
UTC = pytz.UTC


def is_us_trading_hours(dt: datetime) -> bool:
    """Check if datetime is during US TradFi trading hours (9:30 AM - 4:00 PM ET, Mon-Fri)."""
    # Convert to ET
    if dt.tzinfo is None:
        dt = UTC.localize(dt)
    dt_et = dt.astimezone(ET)

    # Check if weekday (0=Monday, 6=Sunday)
    if dt_et.weekday() >= 5:  # Saturday or Sunday
        return False

    # Check time (9:30 AM to 4:00 PM ET)
    market_open = dt_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = dt_et.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= dt_et <= market_close


def is_weekend(dt: datetime) -> bool:
    """Check if datetime is during weekend."""
    if dt.tzinfo is None:
        dt = UTC.localize(dt)
    dt_et = dt.astimezone(ET)
    return dt_et.weekday() >= 5


def get_session_type(dt: datetime) -> str:
    """Classify the trading session."""
    if dt.tzinfo is None:
        dt = UTC.localize(dt)
    dt_et = dt.astimezone(ET)

    if dt_et.weekday() >= 5:
        return "WEEKEND"

    market_open = dt_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = dt_et.replace(hour=16, minute=0, second=0, microsecond=0)

    if market_open <= dt_et <= market_close:
        return "US_HOURS"
    else:
        return "OFF_HOURS"


def fetch_binance_data(symbol: str, interval: str, limit: int = 1500) -> Optional[pd.DataFrame]:
    """Fetch real data from Binance Futures API."""
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        'symbol': f"{symbol}USDT",
        'interval': interval,
        'limit': limit
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code != 200:
            return None

        data = response.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"Error fetching {symbol} {interval}: {e}")
        return None


def run_backtest_with_sessions(df: pd.DataFrame, initial_balance: float = 10000.0,
                                risk_percent: float = 3.0, risk_reward: float = 2.0) -> Dict:
    """Run backtest and categorize trades by session type."""
    if df is None or len(df) < 150:
        return None

    wavetrend = WaveTrend(channel_len=9, average_len=12, ma_len=3, overbought_2=60, oversold_2=-60)
    money_flow = MoneyFlow()
    signal_detector = SignalDetector(anchor_level_long=-60, anchor_level_short=60, trigger_lookback=20)

    ha_df = convert_to_heikin_ashi(df)
    wt_result = wavetrend.calculate(ha_df)
    mf_result = money_flow.calculate(ha_df)

    balance = initial_balance
    trades = []
    active_trade = None

    for i in range(100, len(df)):
        current_bar = df.iloc[i]
        current_time = df.index[i]
        current_price = current_bar['close']
        current_high = current_bar['high']
        current_low = current_bar['low']

        # Check exits
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

                balance += pnl

                # Determine session type at entry time
                session = get_session_type(active_trade['entry_time'])

                trades.append({
                    'entry_time': active_trade['entry_time'],
                    'exit_time': current_time,
                    'direction': active_trade['direction'],
                    'entry_price': active_trade['entry_price'],
                    'stop_loss': active_trade['stop_loss'],
                    'take_profit': active_trade['take_profit'],
                    'exit_price': exit_price,
                    'size': active_trade['size'],
                    'risk_amount': active_trade['risk_amount'],
                    'pnl': pnl,
                    'exit_reason': exit_reason,
                    'balance_after': balance,
                    'session': session
                })

                active_trade = None

        # Check for signals
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

            signal = signal_detector.process_bar(
                timestamp=current_time,
                close_price=current_price,
                wt_result=wt_subset,
                mf_result=mf_subset,
                bar_idx=i
            )

            if signal:
                is_long = signal.signal_type == SignalType.LONG
                direction = 'LONG' if is_long else 'SHORT'

                swing_lookback = 5
                if is_long:
                    recent_lows = df['low'].iloc[max(0, i-swing_lookback):i+1]
                    stop_loss = recent_lows.min() * 0.995
                else:
                    recent_highs = df['high'].iloc[max(0, i-swing_lookback):i+1]
                    stop_loss = recent_highs.max() * 1.005

                risk_distance = abs(current_price - stop_loss)
                reward_distance = risk_distance * risk_reward

                if is_long:
                    take_profit = current_price + reward_distance
                else:
                    take_profit = current_price - reward_distance

                risk_amount = balance * (risk_percent / 100)
                size = risk_amount / risk_distance if risk_distance > 0 else 0

                active_trade = {
                    'entry_time': current_time,
                    'direction': direction,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'size': size,
                    'risk_amount': risk_amount
                }

    return {
        'trades': trades,
        'final_balance': balance,
        'total_pnl': balance - initial_balance
    }


def analyze_by_session(trades: List[Dict]) -> Dict:
    """Analyze trades grouped by session type."""
    sessions = {'US_HOURS': [], 'OFF_HOURS': [], 'WEEKEND': []}

    for t in trades:
        sessions[t['session']].append(t)

    results = {}
    for session, session_trades in sessions.items():
        if not session_trades:
            results[session] = {
                'trades': 0,
                'winners': 0,
                'losers': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'profit_factor': 0
            }
            continue

        winners = [t for t in session_trades if t['pnl'] > 0]
        losers = [t for t in session_trades if t['pnl'] <= 0]
        total_pnl = sum(t['pnl'] for t in session_trades)
        gross_profit = sum(t['pnl'] for t in winners) if winners else 0
        gross_loss = abs(sum(t['pnl'] for t in losers)) if losers else 0.01

        results[session] = {
            'trades': len(session_trades),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners) / len(session_trades) * 100 if session_trades else 0,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(session_trades) if session_trades else 0,
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else 0,
            'trade_list': session_trades
        }

    return results


def generate_report(all_results: Dict, output_path: str):
    """Generate comprehensive markdown report."""

    report = []
    report.append("# VMC Trading Bot - Trading Hours Analysis Report")
    report.append("")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    report.append("")
    report.append("**Data Source:** Binance Futures API (Real Historical Data)")
    report.append("")
    report.append("---")
    report.append("")

    # Session definitions
    report.append("## 1. Session Definitions")
    report.append("")
    report.append("| Session | Description | Time (ET) |")
    report.append("|---------|-------------|-----------|")
    report.append("| **US_HOURS** | US TradFi Market Hours | Mon-Fri 9:30 AM - 4:00 PM ET |")
    report.append("| **OFF_HOURS** | Weekday Off-Market Hours | Mon-Fri outside 9:30-4:00 ET |")
    report.append("| **WEEKEND** | Weekend Trading | Saturday & Sunday |")
    report.append("")

    # Test configuration
    report.append("## 2. Test Configuration")
    report.append("")
    report.append("| Parameter | Value |")
    report.append("|-----------|-------|")
    report.append("| Test Period | ~30 Days |")
    report.append("| Initial Balance | $10,000.00 |")
    report.append("| Risk Per Trade | 3% |")
    report.append("| Risk:Reward | 1:2 |")
    report.append("| Assets | BTC, ETH, SOL |")
    report.append("| Timeframes | 5m, 15m, 1h, 4h |")
    report.append("")
    report.append("---")
    report.append("")

    # Summary by session
    report.append("## 3. Performance by Session Type")
    report.append("")

    # Aggregate across all assets/timeframes
    all_us = {'trades': 0, 'winners': 0, 'pnl': 0}
    all_off = {'trades': 0, 'winners': 0, 'pnl': 0}
    all_wknd = {'trades': 0, 'winners': 0, 'pnl': 0}

    for asset, tf_results in all_results.items():
        for tf, data in tf_results.items():
            if data is None or 'sessions' not in data:
                continue
            sessions = data['sessions']

            if 'US_HOURS' in sessions:
                all_us['trades'] += sessions['US_HOURS']['trades']
                all_us['winners'] += sessions['US_HOURS']['winners']
                all_us['pnl'] += sessions['US_HOURS']['total_pnl']

            if 'OFF_HOURS' in sessions:
                all_off['trades'] += sessions['OFF_HOURS']['trades']
                all_off['winners'] += sessions['OFF_HOURS']['winners']
                all_off['pnl'] += sessions['OFF_HOURS']['total_pnl']

            if 'WEEKEND' in sessions:
                all_wknd['trades'] += sessions['WEEKEND']['trades']
                all_wknd['winners'] += sessions['WEEKEND']['winners']
                all_wknd['pnl'] += sessions['WEEKEND']['total_pnl']

    report.append("### Overall Session Performance (All Assets/Timeframes)")
    report.append("")
    report.append("| Session | Trades | Winners | Win Rate | Net P&L | Avg P&L/Trade |")
    report.append("|---------|--------|---------|----------|---------|---------------|")

    for name, data in [("US_HOURS", all_us), ("OFF_HOURS", all_off), ("WEEKEND", all_wknd)]:
        if data['trades'] > 0:
            wr = data['winners'] / data['trades'] * 100
            avg = data['pnl'] / data['trades']
            report.append(f"| {name} | {data['trades']} | {data['winners']} | {wr:.1f}% | ${data['pnl']:+,.2f} | ${avg:+,.2f} |")
        else:
            report.append(f"| {name} | 0 | 0 | N/A | $0.00 | N/A |")

    report.append("")
    report.append("---")
    report.append("")

    # Detailed results by asset/timeframe
    report.append("## 4. Detailed Results by Asset & Timeframe")
    report.append("")

    for asset in ['BTC', 'ETH', 'SOL']:
        if asset not in all_results:
            continue

        report.append(f"### {asset}")
        report.append("")

        for tf in ['5m', '15m', '1h', '4h']:
            if tf not in all_results[asset] or all_results[asset][tf] is None:
                continue

            data = all_results[asset][tf]

            report.append(f"#### {asset} {tf}")
            report.append("")
            report.append(f"**Period:** {data['period_start']} to {data['period_end']}")
            report.append(f"**Total Candles:** {data['candles']}")
            report.append("")

            report.append("| Session | Trades | Win Rate | Net P&L | Profit Factor |")
            report.append("|---------|--------|----------|---------|---------------|")

            for session in ['US_HOURS', 'OFF_HOURS', 'WEEKEND']:
                if session in data['sessions']:
                    s = data['sessions'][session]
                    wr = f"{s['win_rate']:.1f}%" if s['trades'] > 0 else "N/A"
                    pf = f"{s['profit_factor']:.2f}" if s['profit_factor'] > 0 else "N/A"
                    report.append(f"| {session} | {s['trades']} | {wr} | ${s['total_pnl']:+,.2f} | {pf} |")

            # Trade details
            if data['sessions']:
                report.append("")
                report.append("<details>")
                report.append(f"<summary>View Trade Details ({sum(s['trades'] for s in data['sessions'].values())} trades)</summary>")
                report.append("")
                report.append("| # | Date | Time (ET) | Session | Dir | Entry | Exit | P&L | Result |")
                report.append("|---|------|-----------|---------|-----|-------|------|-----|--------|")

                all_trades = []
                for session, s_data in data['sessions'].items():
                    if 'trade_list' in s_data:
                        all_trades.extend(s_data['trade_list'])

                all_trades.sort(key=lambda x: x['entry_time'])

                for i, t in enumerate(all_trades[:25], 1):
                    entry_et = t['entry_time'].astimezone(ET) if t['entry_time'].tzinfo else UTC.localize(t['entry_time']).astimezone(ET)
                    date_str = entry_et.strftime('%Y-%m-%d')
                    time_str = entry_et.strftime('%H:%M')
                    result = "WIN" if t['pnl'] > 0 else "LOSS"
                    report.append(f"| {i} | {date_str} | {time_str} | {t['session']} | {t['direction']} | ${t['entry_price']:,.2f} | ${t['exit_price']:,.2f} | ${t['pnl']:+,.2f} | {result} |")

                if len(all_trades) > 25:
                    report.append(f"| ... | *{len(all_trades) - 25} more trades* | | | | | | |")

                report.append("")
                report.append("</details>")

            report.append("")

        report.append("---")
        report.append("")

    # Best configurations
    report.append("## 5. Best Performing Configurations")
    report.append("")

    configs = []
    for asset, tf_results in all_results.items():
        for tf, data in tf_results.items():
            if data is None or 'sessions' not in data:
                continue
            for session, s_data in data['sessions'].items():
                if s_data['trades'] > 0:
                    configs.append({
                        'config': f"{asset} {tf}",
                        'session': session,
                        'trades': s_data['trades'],
                        'win_rate': s_data['win_rate'],
                        'pnl': s_data['total_pnl'],
                        'pf': s_data['profit_factor']
                    })

    # Sort by P&L
    profitable = sorted([c for c in configs if c['pnl'] > 0], key=lambda x: x['pnl'], reverse=True)

    if profitable:
        report.append("### Top Profitable Configurations")
        report.append("")
        report.append("| Rank | Config | Session | Trades | Win Rate | Net P&L | PF |")
        report.append("|------|--------|---------|--------|----------|---------|-----|")

        for i, c in enumerate(profitable[:10], 1):
            report.append(f"| {i} | {c['config']} | {c['session']} | {c['trades']} | {c['win_rate']:.1f}% | ${c['pnl']:+,.2f} | {c['pf']:.2f} |")

        report.append("")

    # Key insights
    report.append("## 6. Key Insights")
    report.append("")

    # Determine best session
    best_session = max([(n, d) for n, d in [("US_HOURS", all_us), ("OFF_HOURS", all_off), ("WEEKEND", all_wknd)] if d['trades'] > 0],
                       key=lambda x: x[1]['pnl'] if x[1]['trades'] > 0 else -float('inf'), default=None)

    if best_session:
        report.append(f"- **Best Overall Session:** {best_session[0]} with ${best_session[1]['pnl']:+,.2f} total P&L")

    # US Hours vs Off Hours comparison
    if all_us['trades'] > 0 and all_off['trades'] > 0:
        us_wr = all_us['winners'] / all_us['trades'] * 100
        off_wr = all_off['winners'] / all_off['trades'] * 100

        if us_wr > off_wr:
            report.append(f"- **US Hours outperform Off Hours:** {us_wr:.1f}% vs {off_wr:.1f}% win rate")
        else:
            report.append(f"- **Off Hours outperform US Hours:** {off_wr:.1f}% vs {us_wr:.1f}% win rate")

    # Weekend analysis
    if all_wknd['trades'] > 0:
        wknd_wr = all_wknd['winners'] / all_wknd['trades'] * 100
        report.append(f"- **Weekend Performance:** {all_wknd['trades']} trades, {wknd_wr:.1f}% win rate, ${all_wknd['pnl']:+,.2f}")
    else:
        report.append("- **Weekend:** Limited or no trading signals during test period")

    report.append("")
    report.append("---")
    report.append("")
    report.append("## Disclaimer")
    report.append("")
    report.append("*This analysis uses real historical data from Binance Futures API. Trading hours are based on US Eastern Time. Past performance does not guarantee future results.*")
    report.append("")
    report.append("---")
    report.append("")
    report.append("*Report generated by VMC Trading Bot v1.0*")

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    return '\n'.join(report)


def main():
    print("\n" + "="*70)
    print("  VMC TRADING BOT - TRADING HOURS ANALYSIS")
    print("="*70)
    print()
    print("  Analyzing performance during:")
    print("  - US TradFi Hours (Mon-Fri 9:30 AM - 4:00 PM ET)")
    print("  - Off-Market Hours (Weekday evenings/nights)")
    print("  - Weekends (Saturday & Sunday)")
    print()

    # Configuration - fetch enough data for ~30 days
    assets = ['BTC', 'ETH', 'SOL']
    timeframes = {
        '5m': 8640,    # 30 days * 24 * 12 = 8640 candles
        '15m': 2880,   # 30 days * 24 * 4 = 2880 candles
        '1h': 720,     # 30 days * 24 = 720 candles
        '4h': 180,     # 30 days * 6 = 180 candles
    }

    all_results = {}

    for asset in assets:
        print(f"\nProcessing {asset}...")
        all_results[asset] = {}

        for tf, limit in timeframes.items():
            # Binance limit is 1500 per request, so cap it
            actual_limit = min(limit, 1500)
            print(f"  Fetching {tf} data ({actual_limit} candles)...")

            df = fetch_binance_data(asset, tf, limit=actual_limit)

            if df is not None and len(df) > 150:
                print(f"  Running backtest on {len(df)} candles...")
                result = run_backtest_with_sessions(df, initial_balance=10000.0, risk_percent=3.0, risk_reward=2.0)

                if result and result['trades']:
                    sessions = analyze_by_session(result['trades'])

                    all_results[asset][tf] = {
                        'period_start': df.index[0].strftime('%Y-%m-%d'),
                        'period_end': df.index[-1].strftime('%Y-%m-%d'),
                        'candles': len(df),
                        'total_trades': len(result['trades']),
                        'total_pnl': result['total_pnl'],
                        'sessions': sessions
                    }

                    # Print summary
                    print(f"    Total: {len(result['trades'])} trades, ${result['total_pnl']:+,.2f}")
                    for session, s_data in sessions.items():
                        if s_data['trades'] > 0:
                            print(f"    {session}: {s_data['trades']} trades, {s_data['win_rate']:.1f}% WR, ${s_data['total_pnl']:+,.2f}")
                else:
                    print(f"    -> No trades generated")
                    all_results[asset][tf] = None
            else:
                print(f"    -> Insufficient data")
                all_results[asset][tf] = None

            time.sleep(0.3)  # Rate limiting

    # Generate report
    output_path = project_root / "TRADING_HOURS_REPORT.md"
    print(f"\nGenerating report: {output_path}")

    generate_report(all_results, str(output_path))

    print("\n" + "="*70)
    print("  REPORT GENERATED SUCCESSFULLY!")
    print("="*70)
    print(f"\n  Output file: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()
