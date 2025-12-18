#!/usr/bin/env python3
"""
VMC Trading Bot - V3.1 Walk-Forward Validation
===============================================
Comprehensive walk-forward test of all V3.1 strategy variants on recent data
with time-based filtering analysis.

Tests:
- Signal modes: SIMPLE, ENHANCED (60), ENHANCED (70)
- Exit strategies: FIXED_RR, FULL_SIGNAL, WT_CROSS, FIRST_REVERSAL
- Time filters: All hours, Off-market only, Weekends only, NY hours only
- Assets: BTC, ETH, SOL
- Timeframes: 30m, 1h, 4h

Usage:
    python run_v31_walkforward.py              # Full test (all variants)
    python run_v31_walkforward.py --days 45    # Last 45 days
    python run_v31_walkforward.py --quick      # Quick test (best configs only)
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd

# Fix encoding for Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from config import TakeProfitMethod, OscillatorExitMode
from config.config import SignalMode
from backtest.engine import BacktestEngine
from src.strategy.signals import SignalType

# Data cache directory
CACHE_DIR = project_root / "data" / "binance_cache"
OUTPUT_DIR = project_root / "output" / "walkforward_results"

# ============================================================================
# CONFIGURATION VARIANTS
# ============================================================================

SIGNAL_MODES = {
    "SIMPLE": {
        "signal_mode": SignalMode.SIMPLE,
        "anchor_long": -60,
        "anchor_short": 60,
        "simple_oversold": -53,
        "simple_overbought": 53,
    },
    "ENHANCED_60": {
        "signal_mode": SignalMode.ENHANCED,
        "anchor_long": -60,
        "anchor_short": 60,
        "simple_oversold": -53,
        "simple_overbought": 53,
    },
    "ENHANCED_70": {
        "signal_mode": SignalMode.ENHANCED,
        "anchor_long": -70,
        "anchor_short": 70,
        "simple_oversold": -53,
        "simple_overbought": 53,
    },
}

EXIT_STRATEGIES = {
    "FIXED_RR": {
        "tp_method": TakeProfitMethod.FIXED_RR,
        "osc_mode": OscillatorExitMode.WT_CROSS,
    },
    "FULL_SIGNAL": {
        "tp_method": TakeProfitMethod.OSCILLATOR,
        "osc_mode": OscillatorExitMode.FULL_SIGNAL,
    },
    "WT_CROSS": {
        "tp_method": TakeProfitMethod.OSCILLATOR,
        "osc_mode": OscillatorExitMode.WT_CROSS,
    },
    "FIRST_REVERSAL": {
        "tp_method": TakeProfitMethod.OSCILLATOR,
        "osc_mode": OscillatorExitMode.FIRST_REVERSAL,
    },
}

# Time filter configurations
TIME_FILTERS = {
    "ALL_HOURS": {
        "description": "All trading hours (no filter)",
        "time_filter_mode": "skip",
        "avoid_us_market_hours": False,
        "trade_weekends": True,
    },
    "OFF_MARKET": {
        "description": "Off-market hours only (avoid NY 9:30-4pm)",
        "time_filter_mode": "skip",
        "avoid_us_market_hours": True,
        "trade_weekends": True,
    },
    "WEEKENDS_ONLY": {
        "description": "Weekends only (Sat-Sun)",
        "time_filter_mode": "skip",
        "avoid_us_market_hours": False,
        "trade_weekends": True,
        "weekends_only": True,
    },
    "NY_HOURS_ONLY": {
        "description": "NY market hours only (9:30am-4pm EST)",
        "time_filter_mode": "skip",
        "avoid_us_market_hours": False,
        "trade_weekends": False,
        "ny_hours_only": True,
    },
}

ASSETS = ["BTC", "ETH", "SOL"]
TIMEFRAMES = ["30m", "1h", "4h"]

# All profitable configs from 1-year backtest (14 total)
PROFITABLE_CONFIGS = [
    # Top 10 from annual backtest
    ("BTC", "4h", "SIMPLE", "FULL_SIGNAL"),       # #1: +$5,856
    ("ETH", "30m", "ENHANCED_60", "FULL_SIGNAL"), # #2: +$5,410
    ("BTC", "4h", "ENHANCED_60", "FIXED_RR"),     # #3: +$5,104 (RECOMMENDED)
    ("BTC", "4h", "ENHANCED_60", "FULL_SIGNAL"),  # #4: +$3,658
    ("SOL", "4h", "ENHANCED_60", "FIXED_RR"),     # #5: +$2,617
    ("BTC", "4h", "ENHANCED_70", "FIXED_RR"),     # #6: +$2,256
    ("BTC", "4h", "ENHANCED_60", "WT_CROSS"),     # #7: +$2,097
    ("BTC", "4h", "ENHANCED_60", "FIRST_REVERSAL"), # #8: +$926
    ("BTC", "4h", "ENHANCED_70", "WT_CROSS"),     # #9: +$767
    ("ETH", "4h", "ENHANCED_70", "FIRST_REVERSAL"), # #10: +$749
    # Additional profitable configs
    ("ETH", "4h", "ENHANCED_70", "WT_CROSS"),     # +$496
    ("SOL", "4h", "ENHANCED_70", "FIXED_RR"),     # +$386
    ("BTC", "4h", "SIMPLE", "WT_CROSS"),          # +$563
    ("BTC", "4h", "SIMPLE", "FIRST_REVERSAL"),    # +$161
]

# Best configs from 1-year backtest (for quick mode) - deployed strategy
BEST_CONFIGS = [
    ("BTC", "4h", "ENHANCED_60", "FIXED_RR"),
    ("ETH", "30m", "ENHANCED_60", "FULL_SIGNAL"),
    ("SOL", "4h", "ENHANCED_60", "FIXED_RR"),
]


@dataclass
class TestResult:
    """Single test result."""
    asset: str
    timeframe: str
    signal_mode: str
    exit_strategy: str
    time_filter: str
    trades: int
    wins: int
    losses: int
    win_rate: float
    pnl: float
    pnl_percent: float
    max_drawdown: float
    profit_factor: float
    long_trades: int
    short_trades: int
    error: Optional[str] = None


def load_data(asset: str, timeframe: str, days: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Load cached data, optionally filtered to last N days."""
    path = CACHE_DIR / f"{asset.lower()}_{timeframe}.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path, parse_dates=['timestamp'], index_col='timestamp')

    if days and len(df) > 0:
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df.index >= cutoff]

    return df


def filter_by_time(df: pd.DataFrame, time_filter: str) -> pd.DataFrame:
    """Apply time-based filtering to data.

    Time filters:
    - ALL_HOURS: No filtering
    - OFF_MARKET: Exclude NY market hours (9:30am-4pm EST = 14:30-21:00 UTC)
    - WEEKENDS_ONLY: Only Saturday and Sunday
    - NY_HOURS_ONLY: Only NY market hours on weekdays
    """
    if time_filter == "ALL_HOURS":
        return df

    df = df.copy()

    if time_filter == "OFF_MARKET":
        # Exclude NY market hours (14:30-21:00 UTC on weekdays)
        # Keep: weekends OR (weekdays AND outside 14:30-21:00 UTC)
        is_weekend = df.index.dayofweek >= 5
        is_ny_hours = (df.index.hour >= 14) & (df.index.hour < 21)
        df = df[is_weekend | ~is_ny_hours]
    elif time_filter == "WEEKENDS_ONLY":
        # Saturday = 5, Sunday = 6
        df = df[df.index.dayofweek >= 5]
    elif time_filter == "NY_HOURS_ONLY":
        # NY hours: 14:00-21:00 UTC (9:30am-4pm EST)
        # Weekdays only
        df = df[(df.index.dayofweek < 5) &
                (df.index.hour >= 14) &
                (df.index.hour < 21)]

    return df


def run_single_test(
    asset: str,
    timeframe: str,
    signal_cfg: Dict,
    exit_cfg: Dict,
    time_filter: str,
    df: pd.DataFrame,
) -> TestResult:
    """Run a single backtest configuration."""
    try:
        # Apply time filter to data
        filtered_df = filter_by_time(df, time_filter)

        if len(filtered_df) < 50:
            return TestResult(
                asset=asset,
                timeframe=timeframe,
                signal_mode=signal_cfg.get("name", "unknown"),
                exit_strategy=exit_cfg.get("name", "unknown"),
                time_filter=time_filter,
                trades=0, wins=0, losses=0, win_rate=0,
                pnl=0, pnl_percent=0, max_drawdown=0, profit_factor=0,
                long_trades=0, short_trades=0,
                error="Insufficient data after time filter"
            )

        # Get time filter settings
        tf_cfg = TIME_FILTERS.get(time_filter, TIME_FILTERS["ALL_HOURS"])

        engine = BacktestEngine(
            initial_balance=10000,
            risk_percent=1.5,
            commission_percent=0.06,
            risk_reward=2.0,
            signal_mode=signal_cfg["signal_mode"],
            simple_oversold=signal_cfg["simple_oversold"],
            simple_overbought=signal_cfg["simple_overbought"],
            anchor_level_long=signal_cfg["anchor_long"],
            anchor_level_short=signal_cfg["anchor_short"],
            tp_method=exit_cfg["tp_method"],
            oscillator_mode=exit_cfg["osc_mode"],
            time_filter_mode=tf_cfg.get("time_filter_mode", "skip"),
            avoid_us_market_hours=tf_cfg.get("avoid_us_market_hours", False),
            trade_weekends=tf_cfg.get("trade_weekends", True),
        )

        result = engine.run(filtered_df)

        return TestResult(
            asset=asset,
            timeframe=timeframe,
            signal_mode=signal_cfg.get("name", "unknown"),
            exit_strategy=exit_cfg.get("name", "unknown"),
            time_filter=time_filter,
            trades=result.total_trades,
            wins=result.winning_trades,
            losses=result.losing_trades,
            win_rate=result.win_rate,
            pnl=result.total_pnl,
            pnl_percent=result.total_pnl_percent,
            max_drawdown=result.max_drawdown_percent,
            profit_factor=result.profit_factor,
            long_trades=sum(1 for t in result.trades if t.signal_type == SignalType.LONG),
            short_trades=sum(1 for t in result.trades if t.signal_type == SignalType.SHORT),
        )

    except Exception as e:
        return TestResult(
            asset=asset,
            timeframe=timeframe,
            signal_mode=signal_cfg.get("name", "unknown"),
            exit_strategy=exit_cfg.get("name", "unknown"),
            time_filter=time_filter,
            trades=0, wins=0, losses=0, win_rate=0,
            pnl=0, pnl_percent=0, max_drawdown=0, profit_factor=0,
            long_trades=0, short_trades=0,
            error=str(e)
        )


def run_full_test(days: int = 45, quick: bool = False, profitable_only: bool = False) -> List[TestResult]:
    """Run all test configurations.

    Args:
        days: Number of days to test
        quick: If True, test only BEST_CONFIGS (3 deployed configs)
        profitable_only: If True, test only PROFITABLE_CONFIGS (14 configs from annual backtest)
    """
    results = []
    total_tests = 0

    if quick:
        configs = BEST_CONFIGS
        time_filters = ["ALL_HOURS", "OFF_MARKET"]
    elif profitable_only:
        configs = PROFITABLE_CONFIGS
        time_filters = list(TIME_FILTERS.keys())
    else:
        configs = [
            (asset, tf, sig, exit)
            for asset in ASSETS
            for tf in TIMEFRAMES
            for sig in SIGNAL_MODES.keys()
            for exit in EXIT_STRATEGIES.keys()
        ]
        time_filters = list(TIME_FILTERS.keys())

    # Pre-load all data
    print("\n  Loading data...")
    data_cache = {}
    for asset in ASSETS:
        for tf in TIMEFRAMES:
            key = f"{asset}_{tf}"
            df = load_data(asset, tf, days)
            if df is not None:
                data_cache[key] = df
                print(f"    {key}: {len(df)} candles")

    print(f"\n  Running {len(configs)} configurations x {len(list(time_filters))} time filters...")
    print(f"  Total tests: {len(configs) * len(list(time_filters))}")

    for asset, tf, sig_name, exit_name in configs:
        key = f"{asset}_{tf}"
        df = data_cache.get(key)

        if df is None:
            print(f"    [SKIP] {asset}/{tf} - No data")
            continue

        sig_cfg = SIGNAL_MODES[sig_name].copy()
        sig_cfg["name"] = sig_name
        exit_cfg = EXIT_STRATEGIES[exit_name].copy()
        exit_cfg["name"] = exit_name

        for time_filter in time_filters:
            total_tests += 1
            result = run_single_test(asset, tf, sig_cfg, exit_cfg, time_filter, df)
            results.append(result)

            # Progress indicator
            if total_tests % 10 == 0:
                print(f"    Completed {total_tests} tests...")

    return results


def analyze_results(results: List[TestResult], days: int) -> Dict[str, Any]:
    """Analyze and summarize test results."""
    # Filter out errors
    valid = [r for r in results if r.error is None and r.trades > 0]

    if not valid:
        return {"error": "No valid results"}

    # Overall summary
    total_pnl = sum(r.pnl for r in valid)
    total_trades = sum(r.trades for r in valid)

    # Best by PnL
    best_pnl = max(valid, key=lambda r: r.pnl)

    # Best by profit factor (min 5 trades)
    valid_pf = [r for r in valid if r.trades >= 5]
    best_pf = max(valid_pf, key=lambda r: r.profit_factor) if valid_pf else None

    # Best by risk-adjusted (PnL / MaxDD)
    valid_risk = [r for r in valid if r.max_drawdown > 0]
    best_risk = max(valid_risk, key=lambda r: r.pnl / r.max_drawdown) if valid_risk else None

    # Group by signal mode
    by_signal = {}
    for sig in SIGNAL_MODES.keys():
        sig_results = [r for r in valid if r.signal_mode == sig]
        if sig_results:
            by_signal[sig] = {
                "pnl": sum(r.pnl for r in sig_results),
                "trades": sum(r.trades for r in sig_results),
                "avg_win_rate": sum(r.win_rate for r in sig_results) / len(sig_results),
                "configs": len(sig_results),
            }

    # Group by exit strategy
    by_exit = {}
    for exit in EXIT_STRATEGIES.keys():
        exit_results = [r for r in valid if r.exit_strategy == exit]
        if exit_results:
            by_exit[exit] = {
                "pnl": sum(r.pnl for r in exit_results),
                "trades": sum(r.trades for r in exit_results),
                "avg_win_rate": sum(r.win_rate for r in exit_results) / len(exit_results),
                "configs": len(exit_results),
            }

    # Group by time filter
    by_time = {}
    for tf in TIME_FILTERS.keys():
        tf_results = [r for r in valid if r.time_filter == tf]
        if tf_results:
            by_time[tf] = {
                "pnl": sum(r.pnl for r in tf_results),
                "trades": sum(r.trades for r in tf_results),
                "avg_win_rate": sum(r.win_rate for r in tf_results) / len(tf_results),
                "configs": len(tf_results),
            }

    # Group by asset
    by_asset = {}
    for asset in ASSETS:
        asset_results = [r for r in valid if r.asset == asset]
        if asset_results:
            best_asset = max(asset_results, key=lambda r: r.pnl)
            by_asset[asset] = {
                "pnl": sum(r.pnl for r in asset_results),
                "trades": sum(r.trades for r in asset_results),
                "best_config": f"{best_asset.timeframe} {best_asset.signal_mode} {best_asset.exit_strategy}",
                "best_pnl": best_asset.pnl,
            }

    return {
        "total_configs_tested": len(results),
        "valid_results": len(valid),
        "total_pnl": total_pnl,
        "total_trades": total_trades,
        "best_pnl": best_pnl,
        "best_profit_factor": best_pf,
        "best_risk_adjusted": best_risk,
        "by_signal_mode": by_signal,
        "by_exit_strategy": by_exit,
        "by_time_filter": by_time,
        "by_asset": by_asset,
    }


def print_report(results: List[TestResult], analysis: Dict, days: int):
    """Print comprehensive report."""
    print(f"""
================================================================================
  VMC V3.1 WALK-FORWARD VALIDATION REPORT
================================================================================
  Test Period: Last {days} days
  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
  Total Configurations Tested: {analysis.get('total_configs_tested', 0)}
  Valid Results: {analysis.get('valid_results', 0)}

================================================================================
  OVERALL SUMMARY
================================================================================
  Total PnL (all configs): ${analysis.get('total_pnl', 0):,.2f}
  Total Trades: {analysis.get('total_trades', 0)}
""")

    # Best configurations
    if analysis.get('best_pnl'):
        r = analysis['best_pnl']
        print(f"""
  BEST BY PnL:
    Config: {r.asset}/{r.timeframe} {r.signal_mode} + {r.exit_strategy} ({r.time_filter})
    PnL: ${r.pnl:,.2f} ({r.pnl_percent:.1f}%)
    Trades: {r.trades} | Win Rate: {r.win_rate:.1f}% | Max DD: {r.max_drawdown:.1f}%
""")

    if analysis.get('best_profit_factor'):
        r = analysis['best_profit_factor']
        print(f"""  BEST BY PROFIT FACTOR:
    Config: {r.asset}/{r.timeframe} {r.signal_mode} + {r.exit_strategy} ({r.time_filter})
    Profit Factor: {r.profit_factor:.2f}
    PnL: ${r.pnl:,.2f} | Trades: {r.trades} | Win Rate: {r.win_rate:.1f}%
""")

    if analysis.get('best_risk_adjusted'):
        r = analysis['best_risk_adjusted']
        risk_ratio = r.pnl / r.max_drawdown if r.max_drawdown > 0 else 0
        print(f"""  BEST RISK-ADJUSTED (PnL/MaxDD):
    Config: {r.asset}/{r.timeframe} {r.signal_mode} + {r.exit_strategy} ({r.time_filter})
    Risk Ratio: {risk_ratio:.2f}
    PnL: ${r.pnl:,.2f} | Max DD: {r.max_drawdown:.1f}%
""")

    # By Signal Mode
    print("""
================================================================================
  RESULTS BY SIGNAL MODE
================================================================================""")
    for mode, data in analysis.get('by_signal_mode', {}).items():
        print(f"  {mode}:")
        print(f"    PnL: ${data['pnl']:,.2f} | Trades: {data['trades']} | Avg Win Rate: {data['avg_win_rate']:.1f}%")

    # By Exit Strategy
    print("""
================================================================================
  RESULTS BY EXIT STRATEGY
================================================================================""")
    for exit_s, data in analysis.get('by_exit_strategy', {}).items():
        print(f"  {exit_s}:")
        print(f"    PnL: ${data['pnl']:,.2f} | Trades: {data['trades']} | Avg Win Rate: {data['avg_win_rate']:.1f}%")

    # By Time Filter
    print("""
================================================================================
  RESULTS BY TIME FILTER
================================================================================""")
    for tf, data in analysis.get('by_time_filter', {}).items():
        desc = TIME_FILTERS.get(tf, {}).get('description', tf)
        print(f"  {tf} ({desc}):")
        print(f"    PnL: ${data['pnl']:,.2f} | Trades: {data['trades']} | Avg Win Rate: {data['avg_win_rate']:.1f}%")

    # By Asset
    print("""
================================================================================
  BEST CONFIG PER ASSET
================================================================================""")
    for asset, data in analysis.get('by_asset', {}).items():
        print(f"  {asset}:")
        print(f"    Best: {data['best_config']} = ${data['best_pnl']:,.2f}")
        print(f"    Total PnL: ${data['pnl']:,.2f} | Trades: {data['trades']}")

    # Top 10 configurations
    valid = [r for r in results if r.error is None and r.trades > 0]
    top_10 = sorted(valid, key=lambda r: r.pnl, reverse=True)[:10]

    print("""
================================================================================
  TOP 10 CONFIGURATIONS
================================================================================
  Rank | Asset | TF  | Signal      | Exit          | Filter     | PnL      | Trades | Win% | PF
  -----|-------|-----|-------------|---------------|------------|----------|--------|------|-----""")
    for i, r in enumerate(top_10, 1):
        print(f"  {i:4} | {r.asset:5} | {r.timeframe:3} | {r.signal_mode:11} | {r.exit_strategy:13} | {r.time_filter:10} | ${r.pnl:7,.0f} | {r.trades:6} | {r.win_rate:4.1f} | {r.profit_factor:.2f}")

    print("""
================================================================================
  RECOMMENDATIONS
================================================================================""")

    # Find best overall
    if top_10:
        best = top_10[0]
        print(f"""
  PRIMARY CONFIG (Best PnL):
    {best.asset}/{best.timeframe} with {best.signal_mode} + {best.exit_strategy}
    Time Filter: {best.time_filter}
    Expected: ${best.pnl:,.2f} ({best.pnl_percent:.1f}%) over {days} days

  DEPLOYMENT READY: {"YES" if best.pnl > 0 and best.profit_factor > 1.0 else "REVIEW NEEDED"}
""")


def save_results(results: List[TestResult], days: int):
    """Save results to CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in results:
        rows.append({
            "asset": r.asset,
            "timeframe": r.timeframe,
            "signal_mode": r.signal_mode,
            "exit_strategy": r.exit_strategy,
            "time_filter": r.time_filter,
            "trades": r.trades,
            "wins": r.wins,
            "losses": r.losses,
            "win_rate": r.win_rate,
            "pnl": r.pnl,
            "pnl_percent": r.pnl_percent,
            "max_drawdown": r.max_drawdown,
            "profit_factor": r.profit_factor,
            "long_trades": r.long_trades,
            "short_trades": r.short_trades,
            "error": r.error,
        })

    df = pd.DataFrame(rows)
    filename = f"v31_walkforward_{days}d_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(OUTPUT_DIR / filename, index=False)
    print(f"\n  Results saved to: {OUTPUT_DIR / filename}")


def main():
    parser = argparse.ArgumentParser(description="V3.1 Walk-Forward Validation")
    parser.add_argument("--days", type=int, default=45, help="Number of days to test (default: 45)")
    parser.add_argument("--quick", action="store_true", help="Quick test (3 deployed configs only)")
    parser.add_argument("--profitable", action="store_true", help="Test all 14 profitable configs from annual backtest")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to CSV")
    args = parser.parse_args()

    # Determine mode
    if args.quick:
        mode_desc = "Quick (3 deployed configs)"
    elif args.profitable:
        mode_desc = "Profitable (14 configs from annual backtest)"
    else:
        mode_desc = "Full (all 108 variants)"

    print(f"""
================================================================================
  VMC Trading Bot - V3.1 Walk-Forward Validation
================================================================================
  Mode: {mode_desc}
  Period: Last {args.days} days
""")

    # Run tests
    results = run_full_test(days=args.days, quick=args.quick, profitable_only=args.profitable)

    # Analyze
    analysis = analyze_results(results, args.days)

    # Print report
    print_report(results, analysis, args.days)

    # Save results
    if not args.no_save:
        save_results(results, args.days)


if __name__ == "__main__":
    main()
