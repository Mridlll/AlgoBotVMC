#!/usr/bin/env python3
"""
VMC Trading Bot - Strategy Validation Script
=============================================
Run this before deploying to production to validate the V3.1 strategy
on recent data.

Usage:
    python validate_strategy.py              # Run all V3.1 configs
    python validate_strategy.py --asset BTC  # Run only BTC
    python validate_strategy.py --days 30    # Use last 30 days of data
"""

import sys
from pathlib import Path
import argparse

# Fix encoding for Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd

from config import TakeProfitMethod, OscillatorExitMode
from config.config import SignalMode
from backtest.engine import BacktestEngine
from src.strategy.signals import SignalType

# Data cache directory
CACHE_DIR = project_root / "data" / "binance_cache"

# V3.1 Optimal Configurations (from 1-year backtest)
V31_CONFIGS = {
    "BTC": {
        "timeframe": "4h",
        "signal_mode": SignalMode.ENHANCED,
        "tp_method": TakeProfitMethod.FIXED_RR,
        "osc_mode": OscillatorExitMode.WT_CROSS,  # Not used for FIXED_RR
        "risk_reward": 2.0,
        "anchor_long": -60,
        "anchor_short": 60,
    },
    "ETH": {
        "timeframe": "30m",
        "signal_mode": SignalMode.ENHANCED,
        "tp_method": TakeProfitMethod.OSCILLATOR,
        "osc_mode": OscillatorExitMode.FULL_SIGNAL,
        "risk_reward": 2.0,
        "anchor_long": -60,
        "anchor_short": 60,
    },
    "SOL": {
        "timeframe": "4h",
        "signal_mode": SignalMode.ENHANCED,
        "tp_method": TakeProfitMethod.FIXED_RR,
        "osc_mode": OscillatorExitMode.WT_CROSS,
        "risk_reward": 2.0,
        "anchor_long": -60,
        "anchor_short": 60,
    },
}


def load_data(asset: str, timeframe: str, days: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Load cached Binance data, optionally filtered to last N days."""
    path = CACHE_DIR / f"{asset.lower()}_{timeframe}.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path, parse_dates=['timestamp'], index_col='timestamp')

    # Filter to last N days if specified
    if days and len(df) > 0:
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df.index >= cutoff]

    return df


def run_validation(asset: str, config: Dict[str, Any], days: int = 90) -> Dict[str, Any]:
    """Run validation backtest for a single asset."""
    exit_name = 'FIXED_RR' if config['tp_method'] == TakeProfitMethod.FIXED_RR else config['osc_mode'].value
    print(f"\n  Testing {asset} ({config['timeframe']}, {config['signal_mode'].value}, {exit_name})...")

    try:
        # Load data
        df = load_data(asset, config['timeframe'], days)
        if df is None or len(df) < 100:
            return {
                "asset": asset,
                "error": f"Data not found at {CACHE_DIR}/{asset.lower()}_{config['timeframe']}.csv. Run fetch_and_backtest.py first.",
            }

        engine = BacktestEngine(
            initial_balance=10000,
            risk_percent=1.5,
            commission_percent=0.06,
            signal_mode=config['signal_mode'],
            tp_method=config['tp_method'],
            oscillator_mode=config['osc_mode'],
            risk_reward=config['risk_reward'],
            anchor_level_long=config['anchor_long'],
            anchor_level_short=config['anchor_short'],
        )

        result = engine.run(df)

        return {
            "asset": asset,
            "timeframe": config['timeframe'],
            "mode": config['signal_mode'].value,
            "exit": config['tp_method'].value if config['tp_method'] == TakeProfitMethod.FIXED_RR else config['osc_mode'].value,
            "trades": result.total_trades,
            "wins": result.winning_trades,
            "losses": result.losing_trades,
            "win_rate": result.win_rate,
            "pnl": result.total_pnl,
            "pnl_percent": result.total_pnl_percent,
            "max_drawdown": result.max_drawdown_percent,
            "profit_factor": result.profit_factor,
            "long_trades": sum(1 for t in result.trades if t.signal_type == SignalType.LONG),
            "short_trades": sum(1 for t in result.trades if t.signal_type == SignalType.SHORT),
        }

    except Exception as e:
        return {
            "asset": asset,
            "error": str(e),
        }


def print_results(results: List[Dict[str, Any]], days: int):
    """Print validation results."""
    print(f"""
================================================================================
  V3.1 STRATEGY VALIDATION RESULTS
================================================================================
  Test Period: Last {days} days
  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
""")

    total_pnl = 0
    total_trades = 0

    for r in results:
        if "error" in r:
            print(f"""
  {r['asset']}: ERROR - {r['error']}
""")
            continue

        total_pnl += r['pnl']
        total_trades += r['trades']

        status = "PASS" if r['pnl'] > 0 else "REVIEW"

        print(f"""  {r['asset']} ({r['timeframe']}) - {status}
    Mode: {r['mode']} + {r['exit']}
    Trades: {r['trades']} (L: {r['long_trades']}, S: {r['short_trades']})
    Win Rate: {r['win_rate']:.1f}%
    PnL: ${r['pnl']:,.2f} ({r['pnl_percent']:.1f}%)
    Max DD: {r['max_drawdown']:.1f}%
    Profit Factor: {r['profit_factor']:.2f}
""")

    print(f"""--------------------------------------------------------------------------------
  COMBINED RESULTS
--------------------------------------------------------------------------------
    Total Trades: {total_trades}
    Total PnL: ${total_pnl:,.2f}

  Recommendation: {"READY FOR TESTNET" if total_pnl > 0 else "REVIEW BEFORE DEPLOYMENT"}
================================================================================
""")


def main():
    parser = argparse.ArgumentParser(description="Validate V3.1 strategy on recent data")
    parser.add_argument("--asset", choices=["BTC", "ETH", "SOL"], help="Test specific asset only")
    parser.add_argument("--days", type=int, default=90, help="Number of days to test (default: 90)")
    args = parser.parse_args()

    print("""
================================================================================
  VMC Trading Bot - V3.1 Strategy Validation
================================================================================
""")

    # Determine which assets to test
    if args.asset:
        assets = [args.asset]
    else:
        assets = ["BTC", "ETH", "SOL"]

    print(f"  Validating: {', '.join(assets)}")
    print(f"  Period: Last {args.days} days")
    print(f"  Loading data and running backtests...")

    # Run validations
    results = []
    for asset in assets:
        config = V31_CONFIGS[asset]
        result = run_validation(asset, config, args.days)
        results.append(result)

    # Print results
    print_results(results, args.days)


if __name__ == "__main__":
    main()
