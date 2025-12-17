#!/usr/bin/env python3
"""
VMC Trading Bot - Market Hours Analysis
Segments backtests by:
1. US Market Hours (9:30 AM - 4:00 PM EST, Mon-Fri)
2. Off Hours (outside market hours, Mon-Fri)
3. Weekends (Saturday & Sunday)
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass

from config import TakeProfitMethod, OscillatorExitMode
from backtest.engine import BacktestEngine


ASSETS = ["BTC", "ETH", "SOL"]
TIMEFRAMES = ["15m", "30m", "1h"]  # Focus on profitable timeframes
STRATEGIES = [
    ("Fixed R:R", TakeProfitMethod.FIXED_RR, OscillatorExitMode.WT_CROSS),
    ("Full Signal", TakeProfitMethod.OSCILLATOR, OscillatorExitMode.FULL_SIGNAL),
    ("WT Cross", TakeProfitMethod.OSCILLATOR, OscillatorExitMode.WT_CROSS),
    ("1st Reversal", TakeProfitMethod.OSCILLATOR, OscillatorExitMode.FIRST_REVERSAL),
]

CACHE_DIR = project_root / "data" / "binance_cache"


@dataclass
class Result:
    asset: str
    timeframe: str
    period: str
    strategy: str
    trades: int
    win_rate: float
    pnl: float
    pnl_pct: float
    max_dd: float
    candles: int


def load_cached(asset: str, timeframe: str) -> pd.DataFrame:
    path = CACHE_DIR / f"{asset.lower()}_{timeframe}.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=['timestamp'], index_col='timestamp')
        return df
    return None


def filter_by_market_hours(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into market hours, off hours, and weekends.

    US Market Hours: 9:30 AM - 4:00 PM EST (14:30 - 21:00 UTC)
    Off Hours: Outside market hours on weekdays
    Weekends: Saturday and Sunday
    """
    # Convert to UTC if not already (Binance data is in UTC)
    df = df.copy()

    # Get hour and day of week
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek  # 0=Monday, 6=Sunday

    # US Market hours in UTC: 14:30 - 21:00 (9:30 AM - 4:00 PM EST)
    # Simplified to 14:00 - 21:00 for easier filtering
    market_hours_mask = (
        (df['dayofweek'] < 5) &  # Monday to Friday
        (df['hour'] >= 14) &
        (df['hour'] < 21)
    )

    # Off hours: Weekdays outside market hours
    off_hours_mask = (
        (df['dayofweek'] < 5) &  # Monday to Friday
        ~((df['hour'] >= 14) & (df['hour'] < 21))
    )

    # Weekends
    weekend_mask = df['dayofweek'] >= 5  # Saturday and Sunday

    # Filter and drop helper columns
    market_hours_df = df[market_hours_mask].drop(columns=['hour', 'dayofweek'])
    off_hours_df = df[off_hours_mask].drop(columns=['hour', 'dayofweek'])
    weekend_df = df[weekend_mask].drop(columns=['hour', 'dayofweek'])

    return market_hours_df, off_hours_df, weekend_df


def run_backtest(df: pd.DataFrame, tp_method, osc_mode) -> dict:
    """Run backtest and return results."""
    if df is None or len(df) < 100:
        return None

    try:
        engine = BacktestEngine(
            initial_balance=10000,
            risk_percent=3.0,
            risk_reward=2.0,
            commission_percent=0.06,
            anchor_level_long=-60,
            anchor_level_short=60,
            tp_method=tp_method,
            oscillator_mode=osc_mode
        )
        result = engine.run(df)
        return {
            'trades': result.total_trades,
            'win_rate': result.win_rate,
            'pnl': result.total_pnl,
            'pnl_pct': result.total_pnl_percent,
            'max_dd': result.max_drawdown_percent,
            'candles': len(df)
        }
    except Exception as e:
        return None


def run_market_hours_analysis():
    print("\n" + "=" * 80)
    print("  VMC Trading Bot - Market Hours Analysis")
    print("=" * 80)
    print("\n  Segmenting backtests by:")
    print("    - US Market Hours: 9:30 AM - 4:00 PM EST (Mon-Fri)")
    print("    - Off Hours: Outside market hours (Mon-Fri)")
    print("    - Weekends: Saturday & Sunday")
    print()

    results = []

    for asset in ASSETS:
        for tf in TIMEFRAMES:
            df = load_cached(asset, tf)
            if df is None or len(df) < 200:
                continue

            # Split by market hours
            market_df, off_hours_df, weekend_df = filter_by_market_hours(df)

            periods = [
                ("Market Hours", market_df),
                ("Off Hours", off_hours_df),
                ("Weekends", weekend_df)
            ]

            print(f"  {asset}/{tf}: Full={len(df)}, Market={len(market_df)}, Off={len(off_hours_df)}, Weekend={len(weekend_df)}")

            for period_name, period_df in periods:
                if len(period_df) < 100:
                    continue

                for strategy_name, tp_method, osc_mode in STRATEGIES:
                    res = run_backtest(period_df, tp_method, osc_mode)
                    if res:
                        results.append(Result(
                            asset=asset,
                            timeframe=tf,
                            period=period_name,
                            strategy=strategy_name,
                            trades=res['trades'],
                            win_rate=res['win_rate'],
                            pnl=res['pnl'],
                            pnl_pct=res['pnl_pct'],
                            max_dd=res['max_dd'],
                            candles=res['candles']
                        ))

    return results


def print_report(results: List[Result]):
    periods = ["Market Hours", "Off Hours", "Weekends"]

    for period in periods:
        period_results = [r for r in results if r.period == period]
        if not period_results:
            continue

        print(f"\n" + "=" * 90)
        print(f"  {period.upper()}")
        print("=" * 90)

        for asset in ASSETS:
            asset_results = [r for r in period_results if r.asset == asset]
            if not asset_results:
                continue

            print(f"\n  {asset}:")
            print(f"  {'TF':<6} | {'Strategy':<12} | {'Candles':>7} | {'Trades':>6} | {'Win%':>6} | {'PnL$':>10} | {'MaxDD%':>7}")
            print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*7}")

            for tf in TIMEFRAMES:
                tf_results = [r for r in asset_results if r.timeframe == tf]
                if not tf_results:
                    continue

                best_pnl = max(r.pnl for r in tf_results)

                for i, r in enumerate(tf_results):
                    tf_label = tf if i == 0 else ""
                    candles_label = str(r.candles) if i == 0 else ""
                    marker = "*" if r.pnl == best_pnl else " "
                    print(f"  {tf_label:<6} | {r.strategy:<12} | {candles_label:>7} | {r.trades:>6} | {r.win_rate:>5.1f}% | {r.pnl:>9,.0f}{marker} | {r.max_dd:>6.1f}%")

    # Summary comparison
    print(f"\n" + "=" * 90)
    print("  PERIOD COMPARISON SUMMARY")
    print("=" * 90)

    print(f"\n  {'Period':<15} | {'Total Trades':>12} | {'Avg Win%':>10} | {'Total PnL':>12} | {'Avg MaxDD':>10}")
    print(f"  {'-'*15}-+-{'-'*12}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}")

    for period in periods:
        period_results = [r for r in results if r.period == period]
        if not period_results:
            continue

        total_trades = sum(r.trades for r in period_results)
        avg_wr = sum(r.win_rate for r in period_results) / len(period_results)
        total_pnl = sum(r.pnl for r in period_results)
        avg_dd = sum(r.max_dd for r in period_results) / len(period_results)

        print(f"  {period:<15} | {total_trades:>12,} | {avg_wr:>9.1f}% | ${total_pnl:>11,.0f} | {avg_dd:>9.1f}%")

    # Best by period and strategy
    print(f"\n" + "=" * 90)
    print("  STRATEGY PERFORMANCE BY PERIOD")
    print("=" * 90)

    for strategy_name, _, _ in STRATEGIES:
        print(f"\n  {strategy_name}:")
        print(f"    {'Period':<15} | {'Total PnL':>12} | {'Avg Win%':>10} | {'Avg MaxDD':>10}")
        print(f"    {'-'*15}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}")

        for period in periods:
            strat_period = [r for r in results if r.strategy == strategy_name and r.period == period]
            if not strat_period:
                continue

            total_pnl = sum(r.pnl for r in strat_period)
            avg_wr = sum(r.win_rate for r in strat_period) / len(strat_period)
            avg_dd = sum(r.max_dd for r in strat_period) / len(strat_period)

            print(f"    {period:<15} | ${total_pnl:>11,.0f} | {avg_wr:>9.1f}% | {avg_dd:>9.1f}%")

    # Top performers by period
    print(f"\n" + "=" * 90)
    print("  TOP 5 PROFITABLE SETUPS BY PERIOD")
    print("=" * 90)

    for period in periods:
        period_results = [r for r in results if r.period == period and r.pnl > 0]
        period_results.sort(key=lambda x: x.pnl, reverse=True)

        print(f"\n  {period}:")
        if not period_results:
            print("    No profitable setups")
            continue

        print(f"    {'Asset':<5} | {'TF':<5} | {'Strategy':<12} | {'PnL':>10} | {'Win%':>6} | {'MaxDD':>7}")
        print(f"    {'-'*5}-+-{'-'*5}-+-{'-'*12}-+-{'-'*10}-+-{'-'*6}-+-{'-'*7}")

        for r in period_results[:5]:
            print(f"    {r.asset:<5} | {r.timeframe:<5} | {r.strategy:<12} | ${r.pnl:>9,.0f} | {r.win_rate:>5.1f}% | {r.max_dd:>6.1f}%")


def main():
    start = time.time()
    results = run_market_hours_analysis()

    if results:
        print_report(results)

        # Save to file
        report_path = project_root / "output" / "market_hours_analysis.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w') as f:
            f.write("Market Hours Analysis\n")
            f.write("=" * 50 + "\n\n")

            for period in ["Market Hours", "Off Hours", "Weekends"]:
                f.write(f"\n{period}:\n")
                period_results = [r for r in results if r.period == period]
                for r in sorted(period_results, key=lambda x: x.pnl, reverse=True):
                    f.write(f"  {r.asset}/{r.timeframe} {r.strategy}: WR={r.win_rate:.1f}%, PnL=${r.pnl:,.0f}\n")

        print(f"\n  [Completed in {time.time()-start:.1f}s]")
        print(f"  [Report: {report_path}]")
    else:
        print("\n  No results generated.")


if __name__ == "__main__":
    main()
