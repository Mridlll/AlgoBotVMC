#!/usr/bin/env python3
"""
VMC Trading Bot - Canonical Backtesting Script
==============================================
This is the OFFICIAL backtesting script for the VMC Trading Bot.

Use this script for:
- Testing any signal mode (SIMPLE, ENHANCED, ENHANCED_70, MTF)
- Testing any exit strategy (FIXED_RR, FULL_SIGNAL, WT_CROSS, FIRST_REVERSAL)
- Generating trade logs for verification
- Pre-deployment strategy validation

For comprehensive comparisons across all configs, use run_master_comparison.py instead.

Usage:
    python run_backtest.py                           # Interactive mode
    python run_backtest.py --mode SIMPLE --exit FULL_SIGNAL --asset SOL --tf 30m
    python run_backtest.py --preset best             # Best performing config
    python run_backtest.py --preset safe             # Best risk-adjusted config

Presets:
    best  - SIMPLE + FULL_SIGNAL + SOL/30m (highest profit)
    safe  - ENHANCED_70 + WT_CROSS + SOL/30m (best risk-adjusted)
    btc   - SIMPLE + FULL_SIGNAL + BTC/30m (best for BTC)
    eth   - SIMPLE + FULL_SIGNAL + ETH/30m (best for ETH)
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

import pandas as pd
from datetime import datetime
from typing import Optional

from config import TakeProfitMethod, OscillatorExitMode
from config.config import SignalMode
from backtest.engine import BacktestEngine
from src.strategy.signals import SignalType

# ============================================================================
# CONFIGURATION
# ============================================================================

CACHE_DIR = project_root / "data" / "binance_cache"
OUTPUT_DIR = project_root / "output" / "backtest_results"

# Signal mode configurations
SIGNAL_CONFIGS = {
    "SIMPLE": {
        "signal_mode": SignalMode.SIMPLE,
        "simple_oversold": -53,
        "simple_overbought": 53,
        "anchor_long": -60,  # Not used in SIMPLE, but required param
        "anchor_short": 60,
    },
    "ENHANCED": {
        "signal_mode": SignalMode.ENHANCED,
        "simple_oversold": -53,
        "simple_overbought": 53,
        "anchor_long": -60,
        "anchor_short": 60,
    },
    "ENHANCED_70": {
        "signal_mode": SignalMode.ENHANCED,
        "simple_oversold": -53,
        "simple_overbought": 53,
        "anchor_long": -70,
        "anchor_short": 70,
    },
    "MTF": {
        "signal_mode": SignalMode.MTF,
        "simple_oversold": -53,
        "simple_overbought": 53,
        "anchor_long": -70,
        "anchor_short": 70,
    },
}

# Exit strategy configurations
EXIT_CONFIGS = {
    "FIXED_RR": {
        "tp_method": TakeProfitMethod.FIXED_RR,
        "osc_mode": OscillatorExitMode.WT_CROSS,  # Not used for FIXED_RR
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

# Presets (proven configurations)
PRESETS = {
    "best": {
        "mode": "SIMPLE",
        "exit": "FULL_SIGNAL",
        "asset": "SOL",
        "tf": "30m",
        "rr": 2.0,
        "description": "Highest profit: $31,187 (+311.9%)",
    },
    "safe": {
        "mode": "ENHANCED_70",
        "exit": "WT_CROSS",
        "asset": "SOL",
        "tf": "30m",
        "rr": 2.0,
        "description": "Best risk-adjusted: PF 2.37, 10.4% DD",
    },
    "btc": {
        "mode": "SIMPLE",
        "exit": "FULL_SIGNAL",
        "asset": "BTC",
        "tf": "30m",
        "rr": 2.0,
        "description": "Best for BTC: $7,863 (+78.6%)",
    },
    "eth": {
        "mode": "SIMPLE",
        "exit": "FULL_SIGNAL",
        "asset": "ETH",
        "tf": "30m",
        "rr": 2.0,
        "description": "Best for ETH: $12,332 (+123.3%)",
    },
    "conservative": {
        "mode": "ENHANCED_70",
        "exit": "FIRST_REVERSAL",
        "asset": "ETH",
        "tf": "30m",
        "rr": 2.0,
        "description": "Most consistent: 50% win rate, 9.5% DD",
    },
}


def load_data(asset: str, timeframe: str) -> Optional[pd.DataFrame]:
    """Load cached Binance data."""
    path = CACHE_DIR / f"{asset.lower()}_{timeframe}.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=['timestamp'], index_col='timestamp')
        return df
    print(f"  [ERROR] Data not found: {path}")
    print(f"  Run 'python fetch_and_backtest.py' to download data first.")
    return None


def run_backtest(
    mode: str,
    exit_strategy: str,
    asset: str,
    timeframe: str,
    risk_reward: float = 2.0,
    initial_balance: float = 10000,
    risk_percent: float = 3.0,
    save_trades: bool = True,
) -> dict:
    """
    Run a single backtest with specified configuration.

    Args:
        mode: Signal mode (SIMPLE, ENHANCED, ENHANCED_70, MTF)
        exit_strategy: Exit strategy (FIXED_RR, FULL_SIGNAL, WT_CROSS, FIRST_REVERSAL)
        asset: Trading asset (BTC, ETH, SOL)
        timeframe: Candle timeframe (15m, 30m, 1h)
        risk_reward: Risk:reward ratio for FIXED_RR exits
        initial_balance: Starting balance
        risk_percent: Risk per trade (%)
        save_trades: Whether to save trade log to CSV

    Returns:
        dict with backtest results
    """
    # Load data
    df = load_data(asset, timeframe)
    if df is None or len(df) < 100:
        return {"error": "Insufficient data"}

    # Get configurations
    signal_cfg = SIGNAL_CONFIGS.get(mode)
    exit_cfg = EXIT_CONFIGS.get(exit_strategy)

    if not signal_cfg or not exit_cfg:
        return {"error": f"Invalid config: mode={mode}, exit={exit_strategy}"}

    # Create engine with EXPLICIT signal_mode
    engine = BacktestEngine(
        initial_balance=initial_balance,
        risk_percent=risk_percent,
        risk_reward=risk_reward,
        commission_percent=0.06,
        # Signal mode configuration
        signal_mode=signal_cfg["signal_mode"],
        simple_oversold=signal_cfg["simple_oversold"],
        simple_overbought=signal_cfg["simple_overbought"],
        anchor_level_long=signal_cfg["anchor_long"],
        anchor_level_short=signal_cfg["anchor_short"],
        # Exit strategy configuration
        tp_method=exit_cfg["tp_method"],
        oscillator_mode=exit_cfg["osc_mode"],
    )

    # Run backtest
    result = engine.run(df)

    # Save trade log
    if save_trades and result.trades:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        trades_df = pd.DataFrame([t.to_dict() for t in result.trades])
        filename = f"{mode}_{exit_strategy}_{asset}_{timeframe}_RR{risk_reward}_trades.csv"
        trades_df.to_csv(OUTPUT_DIR / filename, index=False)

    return {
        "mode": mode,
        "exit": exit_strategy,
        "asset": asset,
        "timeframe": timeframe,
        "risk_reward": risk_reward,
        "trades": result.total_trades,
        "wins": result.winning_trades,
        "losses": result.losing_trades,
        "win_rate": result.win_rate,
        "pnl": result.total_pnl,
        "pnl_percent": result.total_pnl_percent,
        "max_drawdown": result.max_drawdown_percent,
        "profit_factor": result.profit_factor,
        "final_balance": result.final_balance,
        "long_trades": sum(1 for t in result.trades if t.signal_type == SignalType.LONG),
        "short_trades": sum(1 for t in result.trades if t.signal_type == SignalType.SHORT),
    }


def print_result(result: dict):
    """Pretty print backtest result."""
    if "error" in result:
        print(f"\n  [ERROR] {result['error']}")
        return

    print(f"""
================================================================================
  BACKTEST RESULT
================================================================================
  Configuration:
    Signal Mode:    {result['mode']}
    Exit Strategy:  {result['exit']}
    Asset:          {result['asset']}/{result['timeframe']}
    Risk:Reward:    {result['risk_reward']}:1

  Performance:
    Total Trades:   {result['trades']} (Long: {result['long_trades']}, Short: {result['short_trades']})
    Win Rate:       {result['win_rate']:.1f}%
    Profit Factor:  {result['profit_factor']:.2f}

  Financials:
    Starting:       $10,000
    Final:          ${result['final_balance']:,.2f}
    Total PnL:      ${result['pnl']:,.2f} ({result['pnl_percent']:.1f}%)
    Max Drawdown:   {result['max_drawdown']:.1f}%

  Trade Log:       output/backtest_results/{result['mode']}_{result['exit']}_{result['asset']}_{result['timeframe']}_RR{result['risk_reward']}_trades.csv
================================================================================
""")


def interactive_mode():
    """Interactive configuration selection."""
    print("""
================================================================================
  VMC Trading Bot - Backtest Configuration
================================================================================
""")

    # Signal mode selection
    print("  Signal Modes:")
    print("    1. SIMPLE      - Original VMC (WT cross at ±53)")
    print("    2. ENHANCED    - 4-step state machine (±60)")
    print("    3. ENHANCED_70 - Stricter 4-step (±70, best risk-adjusted)")
    print("    4. MTF         - Multi-timeframe (same as ENHANCED_70 in single-TF)")
    mode_choice = input("\n  Select signal mode [1-4, default=1]: ").strip() or "1"
    modes = ["SIMPLE", "ENHANCED", "ENHANCED_70", "MTF"]
    mode = modes[int(mode_choice) - 1] if mode_choice.isdigit() else "SIMPLE"

    # Exit strategy selection
    print("\n  Exit Strategies:")
    print("    1. FULL_SIGNAL    - Wait for opposite VMC signal (best profit)")
    print("    2. WT_CROSS       - Exit on WT1/WT2 cross")
    print("    3. FIRST_REVERSAL - Exit on first reversal sign")
    print("    4. FIXED_RR       - Exit at fixed R:R target")
    exit_choice = input("\n  Select exit strategy [1-4, default=1]: ").strip() or "1"
    exits = ["FULL_SIGNAL", "WT_CROSS", "FIRST_REVERSAL", "FIXED_RR"]
    exit_strategy = exits[int(exit_choice) - 1] if exit_choice.isdigit() else "FULL_SIGNAL"

    # Asset selection
    print("\n  Assets:")
    print("    1. SOL - Best performer ($31K profit on SIMPLE+FULL_SIGNAL)")
    print("    2. ETH - Second best ($12K profit)")
    print("    3. BTC - Third best ($7.8K profit)")
    asset_choice = input("\n  Select asset [1-3, default=1]: ").strip() or "1"
    assets = ["SOL", "ETH", "BTC"]
    asset = assets[int(asset_choice) - 1] if asset_choice.isdigit() else "SOL"

    # Timeframe selection
    print("\n  Timeframes:")
    print("    1. 30m - Best overall (82% of configs profitable)")
    print("    2. 1h  - Good for SOL")
    print("    3. 15m - More noise, lower success rate")
    tf_choice = input("\n  Select timeframe [1-3, default=1]: ").strip() or "1"
    timeframes = ["30m", "1h", "15m"]
    timeframe = timeframes[int(tf_choice) - 1] if tf_choice.isdigit() else "30m"

    # R:R ratio (only matters for FIXED_RR)
    rr = 2.0
    if exit_strategy == "FIXED_RR":
        rr_input = input("\n  Risk:Reward ratio [1.5/2.0/2.5, default=2.0]: ").strip() or "2.0"
        rr = float(rr_input) if rr_input else 2.0

    return mode, exit_strategy, asset, timeframe, rr


def main():
    parser = argparse.ArgumentParser(description="VMC Trading Bot Backtester")
    parser.add_argument("--mode", choices=SIGNAL_CONFIGS.keys(), help="Signal mode")
    parser.add_argument("--exit", choices=EXIT_CONFIGS.keys(), help="Exit strategy")
    parser.add_argument("--asset", choices=["BTC", "ETH", "SOL"], help="Trading asset")
    parser.add_argument("--tf", choices=["15m", "30m", "1h"], help="Timeframe")
    parser.add_argument("--rr", type=float, default=2.0, help="Risk:Reward ratio")
    parser.add_argument("--preset", choices=PRESETS.keys(), help="Use preset configuration")
    parser.add_argument("--no-save", action="store_true", help="Don't save trade log")

    args = parser.parse_args()

    print("""
================================================================================
  VMC Trading Bot - Canonical Backtester
================================================================================
""")

    # Handle preset
    if args.preset:
        preset = PRESETS[args.preset]
        print(f"  Using preset: {args.preset}")
        print(f"  Description: {preset['description']}")
        mode = preset["mode"]
        exit_strategy = preset["exit"]
        asset = preset["asset"]
        timeframe = preset["tf"]
        rr = preset["rr"]
    elif args.mode and args.exit and args.asset and args.tf:
        mode = args.mode
        exit_strategy = args.exit
        asset = args.asset
        timeframe = args.tf
        rr = args.rr
    else:
        mode, exit_strategy, asset, timeframe, rr = interactive_mode()

    print(f"\n  Running: {mode} + {exit_strategy} on {asset}/{timeframe}")
    print("  Loading data...")

    result = run_backtest(
        mode=mode,
        exit_strategy=exit_strategy,
        asset=asset,
        timeframe=timeframe,
        risk_reward=rr,
        save_trades=not args.no_save,
    )

    print_result(result)


if __name__ == "__main__":
    main()
