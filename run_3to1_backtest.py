#!/usr/bin/env python3
"""
VMC Trading Bot - 3:1 Risk:Reward Backtest
Uses cached Binance data, just changes R:R ratio
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass

from config import TakeProfitMethod, OscillatorExitMode
from backtest.engine import BacktestEngine


ASSETS = ["BTC", "ETH", "SOL"]
TIMEFRAMES = ["15m", "30m", "1h", "4h"]  # Skip noisy 3m, 5m
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
    strategy: str
    trades: int
    win_rate: float
    pnl: float
    pnl_pct: float
    max_dd: float


def load_cached(asset: str, timeframe: str) -> pd.DataFrame:
    path = CACHE_DIR / f"{asset.lower()}_{timeframe}.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=['timestamp'], index_col='timestamp')
    return None


def run_backtest_3to1():
    print("\n" + "=" * 70)
    print("  VMC Trading Bot - 3:1 Risk:Reward Backtest")
    print("=" * 70)
    print("\n  Using cached Binance data")
    print("  Risk:Reward = 3:1 (vs previous 2:1)")
    print()

    results = []

    for asset in ASSETS:
        for tf in TIMEFRAMES:
            df = load_cached(asset, tf)
            if df is None or len(df) < 100:
                continue

            print(f"  {asset}/{tf} ({len(df)} candles):", end=" ")

            win_rates = []
            for strategy_name, tp_method, osc_mode in STRATEGIES:
                try:
                    engine = BacktestEngine(
                        initial_balance=10000,
                        risk_percent=3.0,
                        risk_reward=3.0,  # 3:1 R:R
                        commission_percent=0.06,
                        anchor_level_long=-60,
                        anchor_level_short=60,
                        tp_method=tp_method,
                        oscillator_mode=osc_mode
                    )

                    result = engine.run(df)
                    results.append(Result(
                        asset=asset,
                        timeframe=tf,
                        strategy=strategy_name,
                        trades=result.total_trades,
                        win_rate=result.win_rate,
                        pnl=result.total_pnl,
                        pnl_pct=result.total_pnl_percent,
                        max_dd=result.max_drawdown_percent
                    ))
                    win_rates.append(f"{result.win_rate:.0f}%")
                except:
                    win_rates.append("ERR")

            print(" | ".join(win_rates))

    return results


def print_report(results: List[Result]):
    print("\n" + "=" * 90)
    print("  3:1 R:R BACKTEST RESULTS")
    print("=" * 90)

    for asset in ASSETS:
        asset_results = [r for r in results if r.asset == asset]
        if not asset_results:
            continue

        print(f"\n  {asset}:")
        print(f"  {'TF':<6} | {'Strategy':<12} | {'Trades':>6} | {'Win%':>6} | {'PnL$':>10} | {'PnL%':>7} | {'MaxDD%':>7}")
        print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*7}-+-{'-'*7}")

        for tf in TIMEFRAMES:
            tf_results = [r for r in asset_results if r.timeframe == tf]
            if not tf_results:
                continue

            best_pnl = max(r.pnl for r in tf_results)

            for i, r in enumerate(tf_results):
                tf_label = tf if i == 0 else ""
                marker = "*" if r.pnl == best_pnl else " "
                print(f"  {tf_label:<6} | {r.strategy:<12} | {r.trades:>6} | {r.win_rate:>5.1f}% | {r.pnl:>9,.0f}{marker} | {r.pnl_pct:>6.1f}% | {r.max_dd:>6.1f}%")

    # Summary
    print(f"\n" + "=" * 90)
    print("  STRATEGY SUMMARY (3:1 R:R)")
    print("=" * 90)

    for strategy_name, _, _ in STRATEGIES:
        strat_results = [r for r in results if r.strategy == strategy_name]
        if not strat_results:
            continue

        total_pnl = sum(r.pnl for r in strat_results)
        avg_wr = sum(r.win_rate for r in strat_results) / len(strat_results)
        avg_dd = sum(r.max_dd for r in strat_results) / len(strat_results)

        print(f"\n  {strategy_name}:")
        print(f"    Total PnL: ${total_pnl:,.0f}")
        print(f"    Avg Win Rate: {avg_wr:.1f}%")
        print(f"    Avg Max DD: {avg_dd:.1f}%")


def main():
    start = time.time()
    results = run_backtest_3to1()

    if results:
        print_report(results)

        # Save to file
        report_path = project_root / "output" / "backtest_3to1_rr.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w') as f:
            f.write("3:1 Risk:Reward Backtest Results\n")
            f.write("=" * 50 + "\n\n")
            for r in results:
                f.write(f"{r.asset}/{r.timeframe} - {r.strategy}: WR={r.win_rate:.1f}%, PnL=${r.pnl:,.0f}, DD={r.max_dd:.1f}%\n")

        print(f"\n  [Completed in {time.time()-start:.1f}s]")
        print(f"  [Report: {report_path}]")


if __name__ == "__main__":
    main()
