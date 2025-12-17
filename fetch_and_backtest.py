#!/usr/bin/env python3
"""
VMC Trading Bot - Data Fetcher & Full Comparison
=================================================
1. Downloads all data from Binance (reliable, no rate limit issues)
2. Caches locally
3. Runs all backtests offline from cache
4. Minimal output to avoid filling context
"""

import sys
from pathlib import Path

# Fix Windows encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

from config import load_config, TakeProfitMethod, OscillatorExitMode
from backtest.engine import BacktestEngine, BacktestResult


# ============================================================================
# CONFIGURATION
# ============================================================================

ASSETS = ["BTC", "ETH", "SOL"]
TIMEFRAMES = ["3m", "5m", "15m", "30m", "1h", "4h"]
STRATEGIES = [
    ("Fixed R:R", TakeProfitMethod.FIXED_RR, OscillatorExitMode.WT_CROSS),
    ("Full Signal", TakeProfitMethod.OSCILLATOR, OscillatorExitMode.FULL_SIGNAL),
    ("WT Cross", TakeProfitMethod.OSCILLATOR, OscillatorExitMode.WT_CROSS),
    ("1st Reversal", TakeProfitMethod.OSCILLATOR, OscillatorExitMode.FIRST_REVERSAL),
]

# 3 months of data
DAYS_OF_DATA = 90

# Cache directory
CACHE_DIR = project_root / "data" / "binance_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ComparisonResult:
    """Stores key metrics for comparison."""
    asset: str
    timeframe: str
    strategy: str
    total_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    max_drawdown: float
    profit_factor: float
    sharpe: float
    candles: int


# ============================================================================
# DATA FETCHING (Binance Futures API)
# ============================================================================

def get_binance_interval(timeframe: str) -> str:
    """Convert timeframe to Binance interval format."""
    return timeframe  # Already matches: 3m, 5m, 15m, 30m, 1h, 4h


def fetch_binance_data(symbol: str, interval: str, days: int = 90) -> Optional[pd.DataFrame]:
    """
    Fetch historical data from Binance Futures API.
    Fetches in chunks to get full history.
    """
    url = "https://fapi.binance.com/fapi/v1/klines"

    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    all_data = []
    current_start = start_time

    while current_start < end_time:
        params = {
            'symbol': f"{symbol}USDT",
            'interval': interval,
            'startTime': current_start,
            'endTime': end_time,
            'limit': 1500  # Max allowed
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code != 200:
                print(f"    Error {response.status_code}: {response.text[:100]}")
                break

            data = response.json()
            if not data:
                break

            all_data.extend(data)

            # Move to next batch
            last_time = data[-1][0]
            if last_time <= current_start:
                break
            current_start = last_time + 1

            time.sleep(0.1)  # Small delay to be nice to API

        except Exception as e:
            print(f"    Exception: {e}")
            break

    if not all_data:
        return None

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    df = df[['open', 'high', 'low', 'close', 'volume']]

    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    df.sort_index(inplace=True)

    return df


def get_cache_path(asset: str, timeframe: str) -> Path:
    """Get cache file path."""
    return CACHE_DIR / f"{asset.lower()}_{timeframe}.csv"


def save_to_cache(df: pd.DataFrame, asset: str, timeframe: str) -> None:
    """Save data to cache."""
    path = get_cache_path(asset, timeframe)
    df.to_csv(path)


def load_from_cache(asset: str, timeframe: str) -> Optional[pd.DataFrame]:
    """Load data from cache."""
    path = get_cache_path(asset, timeframe)
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path, parse_dates=['timestamp'], index_col='timestamp')
        return df
    except Exception:
        return None


def fetch_all_data() -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    Fetch all data for all assets and timeframes.
    Returns dict of (asset, timeframe) -> DataFrame
    """
    print("\n" + "=" * 60)
    print("  STEP 1: FETCHING DATA FROM BINANCE")
    print("=" * 60)

    data_store = {}
    total = len(ASSETS) * len(TIMEFRAMES)
    count = 0

    for asset in ASSETS:
        for tf in TIMEFRAMES:
            count += 1
            key = (asset, tf)

            # Check cache first
            cached = load_from_cache(asset, tf)
            if cached is not None and len(cached) > 100:
                # Check if cache is recent (within last day)
                if cached.index.max() > datetime.now() - timedelta(days=1):
                    data_store[key] = cached
                    print(f"  [{count}/{total}] {asset}/{tf}: Loaded from cache ({len(cached)} candles)")
                    continue

            # Fetch from Binance
            print(f"  [{count}/{total}] {asset}/{tf}: Fetching...", end=" ", flush=True)
            df = fetch_binance_data(asset, tf, DAYS_OF_DATA)

            if df is not None and len(df) > 100:
                save_to_cache(df, asset, tf)
                data_store[key] = df
                print(f"{len(df)} candles")
            else:
                print("FAILED")

            time.sleep(0.2)  # Rate limiting

    print(f"\n  Total datasets loaded: {len(data_store)}/{total}")
    return data_store


# ============================================================================
# BACKTESTING
# ============================================================================

def run_all_backtests(data_store: Dict[Tuple[str, str], pd.DataFrame]) -> List[ComparisonResult]:
    """Run all backtests on cached data."""

    print("\n" + "=" * 60)
    print("  STEP 2: RUNNING BACKTESTS")
    print("=" * 60)

    # Load config
    config_path = str(project_root / "config" / "config.yaml")
    try:
        config = load_config(config_path)
    except:
        print("  Error loading config, using defaults")
        # Use defaults
        class DefaultConfig:
            class backtest:
                initial_balance = 10000.0
                commission_percent = 0.06
            class trading:
                risk_percent = 3.0
            class take_profit:
                risk_reward = 2.0
            class indicators:
                wt_oversold_2 = -60
                wt_overbought_2 = 60
        config = DefaultConfig()

    results = []
    total_tests = len(data_store) * len(STRATEGIES)
    count = 0

    for (asset, tf), df in data_store.items():
        print(f"\n  {asset}/{tf} ({len(df)} candles):", end=" ")

        strategy_results = []
        for strategy_name, tp_method, osc_mode in STRATEGIES:
            count += 1

            try:
                engine = BacktestEngine(
                    initial_balance=config.backtest.initial_balance,
                    risk_percent=config.trading.risk_percent,
                    risk_reward=config.take_profit.risk_reward,
                    commission_percent=config.backtest.commission_percent,
                    anchor_level_long=config.indicators.wt_oversold_2,
                    anchor_level_short=config.indicators.wt_overbought_2,
                    tp_method=tp_method,
                    oscillator_mode=osc_mode
                )

                result = engine.run(df)

                comp_result = ComparisonResult(
                    asset=asset,
                    timeframe=tf,
                    strategy=strategy_name,
                    total_trades=result.total_trades,
                    win_rate=result.win_rate,
                    total_pnl=result.total_pnl,
                    total_pnl_pct=result.total_pnl_percent,
                    max_drawdown=result.max_drawdown_percent,
                    profit_factor=min(result.profit_factor, 999.99),
                    sharpe=result.sharpe_ratio,
                    candles=len(df)
                )
                results.append(comp_result)
                strategy_results.append(f"{result.win_rate:.0f}%")

            except Exception as e:
                strategy_results.append("ERR")

        print(" | ".join(strategy_results))

    print(f"\n  Completed {len(results)} backtests")
    return results


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(results: List[ComparisonResult]) -> str:
    """Generate comprehensive comparison report."""

    lines = []
    lines.append("=" * 100)
    lines.append("  VMC TRADING BOT - FULL COMPARISON REPORT")
    lines.append("  3 Months Data | 3 Assets | 6 Timeframes | 4 Exit Strategies")
    lines.append("=" * 100)

    # Group by asset
    for asset in ASSETS:
        asset_results = [r for r in results if r.asset == asset]
        if not asset_results:
            continue

        lines.append(f"\n{'='*100}")
        lines.append(f"  {asset} RESULTS")
        lines.append(f"{'='*100}")

        # Table header
        lines.append(f"\n  {'TF':<6} | {'Candles':>7} | {'Strategy':<12} | {'Trades':>6} | {'Win%':>6} | {'PnL$':>10} | {'PnL%':>7} | {'MaxDD%':>7} | {'PF':>6}")
        lines.append(f"  {'-'*6}-+-{'-'*7}-+-{'-'*12}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}")

        for tf in TIMEFRAMES:
            tf_results = [r for r in asset_results if r.timeframe == tf]
            if not tf_results:
                continue

            # Find best strategy for this timeframe (by PnL)
            best_pnl = max(r.total_pnl for r in tf_results)

            for i, r in enumerate(tf_results):
                tf_label = tf if i == 0 else ""
                candles_label = str(r.candles) if i == 0 else ""
                pnl_marker = "*" if r.total_pnl == best_pnl else " "

                lines.append(
                    f"  {tf_label:<6} | {candles_label:>7} | {r.strategy:<12} | {r.total_trades:>6} | "
                    f"{r.win_rate:>5.1f}% | {r.total_pnl:>9,.0f}{pnl_marker} | {r.total_pnl_pct:>6.1f}% | "
                    f"{r.max_drawdown:>6.1f}% | {r.profit_factor:>6.2f}"
                )

            lines.append(f"  {'-'*6}-+-{'-'*7}-+-{'-'*12}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}")

    # Summary: Best strategy per asset/timeframe
    lines.append(f"\n{'='*100}")
    lines.append("  BEST STRATEGY BY ASSET/TIMEFRAME (by Total PnL)")
    lines.append(f"{'='*100}")
    lines.append(f"\n  {'Asset':<6} | {'3m':<12} | {'5m':<12} | {'15m':<12} | {'30m':<12} | {'1h':<12} | {'4h':<12}")
    lines.append(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for asset in ASSETS:
        row = [asset]
        for tf in TIMEFRAMES:
            tf_results = [r for r in results if r.asset == asset and r.timeframe == tf]
            if tf_results:
                best = max(tf_results, key=lambda x: x.total_pnl)
                row.append(best.strategy)
            else:
                row.append("N/A")
        lines.append(f"  {row[0]:<6} | {row[1]:<12} | {row[2]:<12} | {row[3]:<12} | {row[4]:<12} | {row[5]:<12} | {row[6]:<12}")

    # Overall strategy performance
    lines.append(f"\n{'='*100}")
    lines.append("  OVERALL STRATEGY PERFORMANCE (Aggregated)")
    lines.append(f"{'='*100}")

    for strategy_name, _, _ in STRATEGIES:
        strategy_results = [r for r in results if r.strategy == strategy_name]
        if not strategy_results:
            continue

        total_trades = sum(r.total_trades for r in strategy_results)
        avg_win_rate = sum(r.win_rate for r in strategy_results) / len(strategy_results)
        total_pnl = sum(r.total_pnl for r in strategy_results)
        avg_dd = sum(r.max_drawdown for r in strategy_results) / len(strategy_results)

        # Count times this strategy was best
        wins_as_best = 0
        for asset in ASSETS:
            for tf in TIMEFRAMES:
                tf_results = [r for r in results if r.asset == asset and r.timeframe == tf]
                if tf_results:
                    best_pnl = max(r.total_pnl for r in tf_results)
                    if any(r.strategy == strategy_name and r.total_pnl == best_pnl for r in tf_results):
                        wins_as_best += 1

        lines.append(f"\n  {strategy_name}:")
        lines.append(f"    Total Trades: {total_trades}")
        lines.append(f"    Avg Win Rate: {avg_win_rate:.1f}%")
        lines.append(f"    Total PnL: ${total_pnl:,.0f}")
        lines.append(f"    Avg Max DD: {avg_dd:.1f}%")
        lines.append(f"    Times Best: {wins_as_best}/{len(ASSETS)*len(TIMEFRAMES)}")

    lines.append(f"\n{'='*100}")
    lines.append("  RECOMMENDATIONS")
    lines.append(f"{'='*100}")
    lines.append("""
  * in PnL column indicates best strategy for that timeframe

  Strategy Selection Guide:
  - Fixed R:R: Predictable exits, good for ranging markets
  - Full Signal: Rides trends longer, higher reward potential
  - WT Cross: Balanced approach, exits on momentum shift
  - 1st Reversal: Fastest exits, most conservative

  Note: Results based on 3 months historical data. Past performance
  does not guarantee future results.
    """)

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("  VMC Trading Bot - Full Comparison")
    print("  Binance Data | 3 Months | All Timeframes")
    print("=" * 60)

    start_time = time.time()

    # Step 1: Fetch all data
    data_store = fetch_all_data()

    if not data_store:
        print("\nERROR: No data fetched. Check internet connection.")
        return

    # Step 2: Run all backtests
    results = run_all_backtests(data_store)

    if not results:
        print("\nERROR: No backtest results generated.")
        return

    # Step 3: Generate report
    report = generate_report(results)

    # Save report
    report_path = project_root / "output" / "full_comparison_binance.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    elapsed = time.time() - start_time

    # Print summary only (not full report to save context)
    print("\n" + "=" * 60)
    print("  COMPLETED")
    print("=" * 60)
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Data: {len(data_store)} asset/timeframe combinations")
    print(f"  Backtests: {len(results)} total")
    print(f"  Report: {report_path}")
    print("\n  To view full report:")
    print(f"  cat {report_path}")


if __name__ == "__main__":
    main()
