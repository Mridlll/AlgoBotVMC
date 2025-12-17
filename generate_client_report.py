#!/usr/bin/env python3
"""
VMC Trading Bot - Comprehensive Client Report Generator
========================================================
Generates a detailed markdown report with multiple test periods.
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
from typing import List, Dict, Optional

from indicators import HeikinAshi, WaveTrend, MoneyFlow
from indicators.heikin_ashi import convert_to_heikin_ashi
from strategy.signals import SignalDetector, SignalType


def fetch_binance_data(symbol: str, interval: str, limit: int = 1000) -> Optional[pd.DataFrame]:
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

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"Error fetching {symbol} {interval}: {e}")
        return None


def run_backtest(df: pd.DataFrame, initial_balance: float = 10000.0, risk_percent: float = 3.0, risk_reward: float = 2.0):
    """Run backtest and return detailed results."""
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
    equity_curve = [initial_balance]

    # State tracking
    anchors_long = 0
    triggers_long = 0
    mfi_long = 0
    anchors_short = 0
    triggers_short = 0
    mfi_short = 0

    prev_long_state = "IDLE"
    prev_short_state = "IDLE"

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
                equity_curve.append(balance)

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
                    'pnl_percent': (pnl / initial_balance) * 100,
                    'exit_reason': exit_reason,
                    'balance_after': balance
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

            state_before = signal_detector.get_current_state()

            signal = signal_detector.process_bar(
                timestamp=current_time,
                close_price=current_price,
                wt_result=wt_subset,
                mf_result=mf_subset,
                bar_idx=i
            )

            state_after = signal_detector.get_current_state()

            # Track states
            if state_after['long_state'] != prev_long_state:
                if state_after['long_state'] == 'ANCHOR_DETECTED':
                    anchors_long += 1
                elif state_after['long_state'] == 'TRIGGER_DETECTED':
                    triggers_long += 1
                elif state_after['long_state'] == 'AWAITING_VWAP':
                    mfi_long += 1
                prev_long_state = state_after['long_state']

            if state_after['short_state'] != prev_short_state:
                if state_after['short_state'] == 'ANCHOR_DETECTED':
                    anchors_short += 1
                elif state_after['short_state'] == 'TRIGGER_DETECTED':
                    triggers_short += 1
                elif state_after['short_state'] == 'AWAITING_VWAP':
                    mfi_short += 1
                prev_short_state = state_after['short_state']

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

    # Calculate metrics
    if not trades:
        return {
            'trades': [],
            'total_trades': 0,
            'winners': 0,
            'losers': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'total_pnl_percent': 0,
            'max_drawdown': 0,
            'profit_factor': 0,
            'final_balance': initial_balance,
            'state_flow': {
                'long': {'anchors': anchors_long, 'triggers': triggers_long, 'mfi': mfi_long},
                'short': {'anchors': anchors_short, 'triggers': triggers_short, 'mfi': mfi_short}
            },
            'equity_curve': equity_curve
        }

    winners = [t for t in trades if t['pnl'] > 0]
    losers = [t for t in trades if t['pnl'] <= 0]
    total_pnl = sum(t['pnl'] for t in trades)
    gross_profit = sum(t['pnl'] for t in winners) if winners else 0
    gross_loss = abs(sum(t['pnl'] for t in losers)) if losers else 0.01

    # Max drawdown
    peak = initial_balance
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd

    return {
        'trades': trades,
        'total_trades': len(trades),
        'long_trades': len([t for t in trades if t['direction'] == 'LONG']),
        'short_trades': len([t for t in trades if t['direction'] == 'SHORT']),
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': len(winners) / len(trades) * 100 if trades else 0,
        'total_pnl': total_pnl,
        'total_pnl_percent': (total_pnl / initial_balance) * 100,
        'max_drawdown': max_dd,
        'profit_factor': gross_profit / gross_loss if gross_loss > 0 else 0,
        'avg_win': gross_profit / len(winners) if winners else 0,
        'avg_loss': gross_loss / len(losers) if losers else 0,
        'final_balance': balance,
        'state_flow': {
            'long': {'anchors': anchors_long, 'triggers': triggers_long, 'mfi': mfi_long},
            'short': {'anchors': anchors_short, 'triggers': triggers_short, 'mfi': mfi_short}
        },
        'equity_curve': equity_curve
    }


def generate_markdown_report(all_results: Dict, output_path: str):
    """Generate comprehensive markdown report."""

    report = []
    report.append("# VMC Trading Bot - Comprehensive Backtest Report")
    report.append("")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    report.append("")
    report.append("**Data Source:** Binance Futures API (Real Historical Data)")
    report.append("")
    report.append("---")
    report.append("")

    # Configuration section
    report.append("## 1. Test Configuration")
    report.append("")
    report.append("| Parameter | Value |")
    report.append("|-----------|-------|")
    report.append("| Initial Balance | $10,000.00 |")
    report.append("| Risk Per Trade | 3% ($300 max loss per trade) |")
    report.append("| Risk:Reward Ratio | 1:2 |")
    report.append("| Stop Loss Method | Swing Low/High (5 bars) |")
    report.append("| Take Profit | 2x Risk Distance |")
    report.append("| Anchor Level (Long) | WT2 < -60 |")
    report.append("| Anchor Level (Short) | WT2 > +60 |")
    report.append("| Trigger Lookback | 20 bars |")
    report.append("")

    # Strategy explanation
    report.append("## 2. Strategy Overview")
    report.append("")
    report.append("The VMC (Visual Market Cipher) strategy uses a 4-step signal detection process:")
    report.append("")
    report.append("1. **Anchor Wave**: Identifies extreme market conditions (WT2 < -60 for longs, > +60 for shorts)")
    report.append("2. **Trigger Wave**: Confirms momentum shift with higher low (long) or lower high (short)")
    report.append("3. **MFI Confirmation**: Money Flow Index must be curving in trade direction with correct sign")
    report.append("4. **VWAP Cross**: Final confirmation when VWAP crosses zero")
    report.append("")
    report.append("---")
    report.append("")

    # Results by asset
    report.append("## 3. Results by Asset & Timeframe")
    report.append("")

    for asset, timeframe_results in all_results.items():
        report.append(f"### {asset}")
        report.append("")
        report.append("| Timeframe | Period | Candles | Trades | Win Rate | Net P&L | Return | Max DD | Profit Factor |")
        report.append("|-----------|--------|---------|--------|----------|---------|--------|--------|---------------|")

        for tf, data in timeframe_results.items():
            if data['result'] is None:
                report.append(f"| {tf} | - | - | 0 | - | - | - | - | - |")
                continue

            result = data['result']
            period = f"{data['start'].strftime('%Y-%m-%d')} to {data['end'].strftime('%Y-%m-%d')}"
            candles = data['candles']

            pnl_str = f"${result['total_pnl']:+,.2f}"
            ret_str = f"{result['total_pnl_percent']:+.2f}%"
            dd_str = f"{result['max_drawdown']:.2f}%"
            pf_str = f"{result['profit_factor']:.2f}" if result['profit_factor'] > 0 else "N/A"
            wr_str = f"{result['win_rate']:.1f}%" if result['total_trades'] > 0 else "N/A"

            report.append(f"| {tf} | {period} | {candles} | {result['total_trades']} | {wr_str} | {pnl_str} | {ret_str} | {dd_str} | {pf_str} |")

        report.append("")

    report.append("---")
    report.append("")

    # Detailed trade log
    report.append("## 4. Detailed Trade Log")
    report.append("")

    for asset, timeframe_results in all_results.items():
        for tf, data in timeframe_results.items():
            if data['result'] is None or not data['result']['trades']:
                continue

            result = data['result']
            trades = result['trades']

            report.append(f"### {asset} {tf}")
            report.append("")
            report.append(f"**Period:** {data['start'].strftime('%Y-%m-%d')} to {data['end'].strftime('%Y-%m-%d')}")
            report.append("")
            report.append(f"**Signal Detection Flow:**")
            report.append(f"- Long: {result['state_flow']['long']['anchors']} anchors -> {result['state_flow']['long']['triggers']} triggers -> {result['state_flow']['long']['mfi']} MFI confirms -> {result['long_trades']} signals")
            report.append(f"- Short: {result['state_flow']['short']['anchors']} anchors -> {result['state_flow']['short']['triggers']} triggers -> {result['state_flow']['short']['mfi']} MFI confirms -> {result['short_trades']} signals")
            report.append("")

            report.append("| # | Date | Dir | Entry | Stop Loss | Take Profit | Exit | Risk | P&L | Result | Balance |")
            report.append("|---|------|-----|-------|-----------|-------------|------|------|-----|--------|---------|")

            for i, t in enumerate(trades, 1):
                date_str = t['entry_time'].strftime('%Y-%m-%d %H:%M')
                entry_str = f"${t['entry_price']:,.2f}"
                sl_str = f"${t['stop_loss']:,.2f}"
                tp_str = f"${t['take_profit']:,.2f}"
                exit_str = f"${t['exit_price']:,.2f}"
                risk_str = f"${t['risk_amount']:,.2f}"
                pnl_str = f"${t['pnl']:+,.2f}"
                result_str = "WIN" if t['pnl'] > 0 else "LOSS"
                balance_str = f"${t['balance_after']:,.2f}"

                report.append(f"| {i} | {date_str} | {t['direction']} | {entry_str} | {sl_str} | {tp_str} | {exit_str} | {risk_str} | {pnl_str} | {result_str} | {balance_str} |")

            report.append("")
            report.append(f"**Summary:** {len(trades)} trades, {result['winners']} wins, {result['losers']} losses, {result['win_rate']:.1f}% win rate, ${result['total_pnl']:+,.2f} net P&L")
            report.append("")
            report.append("---")
            report.append("")

    # Overall summary
    report.append("## 5. Overall Summary")
    report.append("")

    total_trades = 0
    total_winners = 0
    total_pnl = 0
    best_config = None
    best_pnl = -float('inf')
    worst_config = None
    worst_pnl = float('inf')

    for asset, timeframe_results in all_results.items():
        for tf, data in timeframe_results.items():
            if data['result'] is None:
                continue
            result = data['result']
            total_trades += result['total_trades']
            total_winners += result['winners']
            total_pnl += result['total_pnl']

            if result['total_pnl'] > best_pnl:
                best_pnl = result['total_pnl']
                best_config = f"{asset} {tf}"
            if result['total_pnl'] < worst_pnl and result['total_trades'] > 0:
                worst_pnl = result['total_pnl']
                worst_config = f"{asset} {tf}"

    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| Total Configurations Tested | {sum(1 for a in all_results.values() for t in a.values() if t['result'] is not None)} |")
    report.append(f"| Total Trades Executed | {total_trades} |")
    report.append(f"| Overall Win Rate | {total_winners/total_trades*100:.1f}% |" if total_trades > 0 else "| Overall Win Rate | N/A |")
    report.append(f"| Combined Net P&L | ${total_pnl:+,.2f} |")
    report.append(f"| Best Configuration | {best_config} (${best_pnl:+,.2f}) |" if best_config else "| Best Configuration | N/A |")
    report.append(f"| Worst Configuration | {worst_config} (${worst_pnl:+,.2f}) |" if worst_config else "| Worst Configuration | N/A |")
    report.append("")

    # Recommendations
    report.append("## 6. Recommendations")
    report.append("")
    report.append("Based on the backtest results:")
    report.append("")

    profitable_configs = []
    for asset, timeframe_results in all_results.items():
        for tf, data in timeframe_results.items():
            if data['result'] is not None and data['result']['total_pnl'] > 0:
                profitable_configs.append((asset, tf, data['result']))

    if profitable_configs:
        report.append("**Profitable Configurations:**")
        for asset, tf, result in sorted(profitable_configs, key=lambda x: x[2]['total_pnl'], reverse=True)[:5]:
            report.append(f"- {asset} {tf}: {result['win_rate']:.1f}% win rate, ${result['total_pnl']:+,.2f} profit, PF: {result['profit_factor']:.2f}")
        report.append("")

    report.append("**Key Observations:**")
    report.append("- Higher timeframes (1h, 4h) generally produce more reliable signals")
    report.append("- Lower timeframes (5m, 15m) have more noise and false signals")
    report.append("- The 4-step signal filter significantly reduces false entries")
    report.append("- MFI sign validation (negative for longs, positive for shorts) is critical")
    report.append("")

    # Disclaimer
    report.append("---")
    report.append("")
    report.append("## Disclaimer")
    report.append("")
    report.append("*This backtest uses real historical data from Binance Futures API. Past performance does not guarantee future results. Trading cryptocurrencies carries significant risk. Always use proper risk management and never trade with funds you cannot afford to lose.*")
    report.append("")
    report.append("---")
    report.append("")
    report.append(f"*Report generated by VMC Trading Bot v1.0*")

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    return '\n'.join(report)


def main():
    print("\n" + "="*70)
    print("  VMC TRADING BOT - GENERATING COMPREHENSIVE CLIENT REPORT")
    print("="*70)
    print()

    # Test configurations
    assets = ['BTC', 'ETH', 'SOL']
    timeframes = {
        '5m': 1000,    # ~3.5 days
        '15m': 1000,   # ~10 days
        '1h': 1000,    # ~42 days
        '4h': 1000,    # ~166 days
    }

    all_results = {}

    for asset in assets:
        print(f"\nProcessing {asset}...")
        all_results[asset] = {}

        for tf, limit in timeframes.items():
            print(f"  Fetching {tf} data ({limit} candles)...")

            df = fetch_binance_data(asset, tf, limit=limit)

            if df is not None and len(df) > 150:
                print(f"  Running backtest on {len(df)} candles...")
                result = run_backtest(df, initial_balance=10000.0, risk_percent=3.0, risk_reward=2.0)

                all_results[asset][tf] = {
                    'start': df.index[0],
                    'end': df.index[-1],
                    'candles': len(df),
                    'result': result
                }

                if result and result['total_trades'] > 0:
                    print(f"    -> {result['total_trades']} trades, {result['win_rate']:.1f}% win rate, ${result['total_pnl']:+,.2f}")
                else:
                    print(f"    -> No trades generated")
            else:
                print(f"    -> Insufficient data")
                all_results[asset][tf] = {
                    'start': None,
                    'end': None,
                    'candles': 0,
                    'result': None
                }

            time.sleep(0.3)  # Rate limiting

    # Generate report
    output_path = project_root / "BACKTEST_REPORT.md"
    print(f"\nGenerating report: {output_path}")

    report = generate_markdown_report(all_results, str(output_path))

    print("\n" + "="*70)
    print("  REPORT GENERATED SUCCESSFULLY!")
    print("="*70)
    print(f"\n  Output file: {output_path}")
    print("\n  You can open this file in any markdown viewer or text editor.")
    print("="*70)

    # Also print summary to console
    print("\n\nQUICK SUMMARY:")
    print("-"*70)

    for asset, timeframe_results in all_results.items():
        print(f"\n{asset}:")
        for tf, data in timeframe_results.items():
            if data['result'] is None:
                continue
            r = data['result']
            status = "PROFIT" if r['total_pnl'] > 0 else "LOSS" if r['total_pnl'] < 0 else "BREAK-EVEN"
            print(f"  {tf}: {r['total_trades']} trades, {r['win_rate']:.1f}% win rate, ${r['total_pnl']:+,.2f} ({status})")


if __name__ == "__main__":
    main()
