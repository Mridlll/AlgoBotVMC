#!/usr/bin/env python3
"""
VMC Trading Bot - Exit Strategy Comparison Backtest
====================================================
Compares 4 exit strategies:
1. Fixed R:R (v1) - Exit at fixed risk:reward ratio
2. Oscillator - Full Signal - Exit on complete opposite VMC signal
3. Oscillator - WT Cross - Exit when WT1 crosses WT2 opposite direction
4. Oscillator - First Reversal - Exit on first reversal sign
"""

import sys
from pathlib import Path

# Fix Windows encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add project root and src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import asyncio
from datetime import datetime
from typing import Dict, List, Any

from config import Config, load_config, TakeProfitMethod, OscillatorExitMode
from backtest.engine import BacktestEngine, BacktestResult
from backtest.data_loader import DataLoader
from exchanges.hyperliquid import HyperliquidExchange
from utils.logger import setup_logger, get_logger

setup_logger(log_level="INFO")
logger = get_logger("comparison_backtest")


async def run_comparison_backtest(
    config_path: str = None,
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-17"
) -> Dict[str, Dict[str, BacktestResult]]:
    """
    Run comparison backtest with all 4 exit strategies.

    Returns:
        Dict mapping asset -> strategy -> BacktestResult
    """
    # Default to config in project root
    if config_path is None:
        config_path = str(project_root / "config" / "config.yaml")
    # Load config
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        return {}

    # Define strategies to test
    strategies = [
        ("Fixed R:R (v1)", TakeProfitMethod.FIXED_RR, OscillatorExitMode.WT_CROSS),
        ("Oscillator - Full Signal", TakeProfitMethod.OSCILLATOR, OscillatorExitMode.FULL_SIGNAL),
        ("Oscillator - WT Cross", TakeProfitMethod.OSCILLATOR, OscillatorExitMode.WT_CROSS),
        ("Oscillator - First Reversal", TakeProfitMethod.OSCILLATOR, OscillatorExitMode.FIRST_REVERSAL),
    ]

    # Create exchange for data fetching
    exchange = HyperliquidExchange(
        api_key="",
        api_secret="",
        testnet=True
    )
    await exchange.connect()

    # Data loader
    data_loader = DataLoader()
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Results storage
    all_results: Dict[str, Dict[str, BacktestResult]] = {}

    # Only test BTC for now due to rate limiting
    assets_to_test = ["BTC"]  # config.trading.assets
    for asset in assets_to_test:
        logger.info(f"\n{'='*60}")
        logger.info(f"BACKTESTING {asset}")
        logger.info(f"{'='*60}")

        # Load data (cached if available)
        try:
            df = data_loader.load_cached(asset, config.trading.timeframe, start_dt, end_dt)
        except Exception as e:
            logger.warning(f"Cache load failed for {asset}: {e}")
            df = None

        if df is None or len(df) < 100:
            logger.info(f"Fetching {asset} data from exchange...")
            try:
                df = await data_loader.load_from_exchange(
                    exchange=exchange,
                    symbol=asset,
                    timeframe=config.trading.timeframe,
                    start_date=start_dt,
                    end_date=end_dt
                )
                if df is not None and len(df) > 0:
                    data_loader.save_to_cache(df, asset, config.trading.timeframe)
            except Exception as e:
                logger.error(f"Failed to fetch data for {asset}: {e}")
                continue

        if df is None:
            logger.warning(f"Skipping {asset} - no data available")
            continue

        logger.info(f"Loaded {len(df)} candles for {asset}")

        # Skip if no data
        if df is None or len(df) < 200:
            logger.warning(f"Skipping {asset} - insufficient data ({len(df) if df is not None else 0} candles)")
            continue

        all_results[asset] = {}

        # Run each strategy
        for strategy_name, tp_method, osc_mode in strategies:
            logger.info(f"\nRunning: {strategy_name}...")

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
                all_results[asset][strategy_name] = result
            except Exception as e:
                logger.error(f"Error running {strategy_name} for {asset}: {e}")
                continue

    await exchange.disconnect()
    return all_results


def print_comparison_report(results: Dict[str, Dict[str, BacktestResult]]) -> None:
    """Print comparison report for all strategies."""

    print("\n" + "="*80)
    print("  VMC TRADING BOT - EXIT STRATEGY COMPARISON REPORT")
    print("="*80)

    for asset, strategy_results in results.items():
        print(f"\n{'='*80}")
        print(f"  {asset} COMPARISON")
        print(f"{'='*80}")

        # Table header
        headers = ["Metric", "Fixed R:R", "Full Signal", "WT Cross", "First Reversal"]
        strategy_order = [
            "Fixed R:R (v1)",
            "Oscillator - Full Signal",
            "Oscillator - WT Cross",
            "Oscillator - First Reversal"
        ]

        # Collect results in order
        ordered_results = [strategy_results.get(s) for s in strategy_order]

        # Print comparison table
        print(f"\n  {'Metric':<20} | {'Fixed R:R':>12} | {'Full Signal':>12} | {'WT Cross':>12} | {'1st Reversal':>12}")
        print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

        # Metrics to compare
        metrics = [
            ("Total Trades", lambda r: f"{r.total_trades}", lambda r: r.total_trades),
            ("Win Rate (%)", lambda r: f"{r.win_rate:.1f}", lambda r: r.win_rate),
            ("Total PnL ($)", lambda r: f"{r.total_pnl:,.0f}", lambda r: r.total_pnl),
            ("PnL (%)", lambda r: f"{r.total_pnl_percent:.1f}", lambda r: r.total_pnl_percent),
            ("Max Drawdown (%)", lambda r: f"{r.max_drawdown_percent:.1f}", lambda r: -r.max_drawdown_percent),  # Negative is better
            ("Profit Factor", lambda r: f"{r.profit_factor:.2f}" if r.profit_factor < float('inf') else "inf", lambda r: r.profit_factor if r.profit_factor < float('inf') else 0),
            ("Sharpe Ratio", lambda r: f"{r.sharpe_ratio:.2f}", lambda r: r.sharpe_ratio),
            ("Avg Win ($)", lambda r: f"{r.avg_win:.0f}", lambda r: r.avg_win),
            ("Avg Loss ($)", lambda r: f"{r.avg_loss:.0f}", lambda r: -r.avg_loss),  # Smaller is better
        ]

        for metric_name, format_fn, sort_fn in metrics:
            values = []
            for r in ordered_results:
                if r:
                    values.append(format_fn(r))
                else:
                    values.append("N/A")

            # Find best value (highest sort_fn result)
            best_idx = -1
            best_val = float('-inf')
            for i, r in enumerate(ordered_results):
                if r:
                    val = sort_fn(r)
                    if val > best_val:
                        best_val = val
                        best_idx = i

            # Format with highlight for best
            formatted_values = []
            for i, v in enumerate(values):
                if i == best_idx and v != "N/A":
                    formatted_values.append(f"*{v}*")  # Highlight best
                else:
                    formatted_values.append(v)

            print(f"  {metric_name:<20} | {formatted_values[0]:>12} | {formatted_values[1]:>12} | {formatted_values[2]:>12} | {formatted_values[3]:>12}")

        # Exit reason breakdown
        print(f"\n  Exit Reason Breakdown:")
        print(f"  {'-'*70}")

        for strategy_name in strategy_order:
            result = strategy_results.get(strategy_name)
            if result and result.trades:
                exit_reasons = {}
                for trade in result.trades:
                    reason = trade.exit_reason
                    exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

                reasons_str = ", ".join([f"{k}: {v}" for k, v in sorted(exit_reasons.items())])
                print(f"  {strategy_name:<30}: {reasons_str}")

    # Overall recommendation
    print(f"\n{'='*80}")
    print("  RECOMMENDATIONS")
    print(f"{'='*80}")
    print("""
  * indicates best value for each metric

  Strategy Selection Guide:
  - Fixed R:R: Predictable exits, good for ranging markets
  - Full Signal: Longest trades, rides trends, may give back profits
  - WT Cross: Balanced approach, exits on momentum shift
  - First Reversal: Fastest exits, most conservative profit taking

  Choose based on:
  - Higher Win Rate: Better for consistency
  - Higher Total PnL: Better for overall returns
  - Lower Drawdown: Better for capital preservation
  - Higher Profit Factor: Better risk-adjusted returns
    """)


async def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("  VMC Trading Bot - Exit Strategy Comparison")
    print("="*60)
    print("\nThis will run 4 backtests for each asset:")
    print("  1. Fixed R:R (v1)")
    print("  2. Oscillator - Full Signal")
    print("  3. Oscillator - WT Cross")
    print("  4. Oscillator - First Reversal")
    print("\nStart: 2025-01-01")
    print("End:   2025-12-17")
    print()

    results = await run_comparison_backtest()

    if results:
        print_comparison_report(results)

        # Save report to file
        report_path = project_root / "output" / "comparison_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            # Redirect print output to file
            import io
            import contextlib
            string_io = io.StringIO()
            with contextlib.redirect_stdout(string_io):
                print_comparison_report(results)
            f.write(string_io.getvalue())
        print(f"\n[Report saved to: {report_path}]")
    else:
        print("\nNo results - check configuration and data availability.")


if __name__ == "__main__":
    asyncio.run(main())
