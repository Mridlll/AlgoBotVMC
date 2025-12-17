#!/usr/bin/env python3
"""
VMC Trading Bot - Full Multi-Timeframe Multi-Asset Comparison
==============================================================
Compares 4 exit strategies across multiple timeframes and assets.
Uses 3 months of data to reduce memory/context usage.
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
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from config import Config, load_config, TakeProfitMethod, OscillatorExitMode
from backtest.engine import BacktestEngine, BacktestResult
from backtest.data_loader import DataLoader
from exchanges.hyperliquid import HyperliquidExchange
from utils.logger import setup_logger, get_logger

# Minimal logging - only errors
setup_logger(log_level="WARNING")
logger = get_logger("full_comparison")


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


# Strategies to test
STRATEGIES = [
    ("Fixed R:R", TakeProfitMethod.FIXED_RR, OscillatorExitMode.WT_CROSS),
    ("Full Signal", TakeProfitMethod.OSCILLATOR, OscillatorExitMode.FULL_SIGNAL),
    ("WT Cross", TakeProfitMethod.OSCILLATOR, OscillatorExitMode.WT_CROSS),
    ("1st Reversal", TakeProfitMethod.OSCILLATOR, OscillatorExitMode.FIRST_REVERSAL),
]

# Timeframes to test
TIMEFRAMES = ["3m", "5m", "15m", "30m", "1h", "4h"]

# Assets to test
ASSETS = ["BTC", "ETH", "SOL"]


async def fetch_data_with_retry(
    data_loader: DataLoader,
    exchange: HyperliquidExchange,
    asset: str,
    timeframe: str,
    start_dt: datetime,
    end_dt: datetime,
    max_retries: int = 3
) -> Optional[Any]:
    """Fetch data with retry logic for rate limiting."""

    # Try cache first
    try:
        df = data_loader.load_cached(asset, timeframe, start_dt, end_dt)
        if df is not None and len(df) >= 50:
            return df
    except Exception:
        pass

    # Fetch from exchange with retries
    for attempt in range(max_retries):
        try:
            df = await data_loader.load_from_exchange(
                exchange=exchange,
                symbol=asset,
                timeframe=timeframe,
                start_date=start_dt,
                end_date=end_dt
            )
            if df is not None and len(df) > 0:
                data_loader.save_to_cache(df, asset, timeframe)
                return df
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait_time = 30 * (attempt + 1)
                print(f"  Rate limited, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"  Error: {e}")
                break

    return None


async def run_full_comparison() -> List[ComparisonResult]:
    """Run comparison across all timeframes and assets."""

    # Load config
    config_path = str(project_root / "config" / "config.yaml")
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_path}")
        return []

    # 3 months of data
    end_dt = datetime(2025, 12, 17)
    start_dt = end_dt - timedelta(days=90)

    print(f"\nData Range: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')} (3 months)")
    print(f"Assets: {', '.join(ASSETS)}")
    print(f"Timeframes: {', '.join(TIMEFRAMES)}")
    print(f"Strategies: {', '.join([s[0] for s in STRATEGIES])}")
    print()

    # Create exchange
    exchange = HyperliquidExchange(
        api_key="",
        api_secret="",
        testnet=True
    )
    await exchange.connect()

    data_loader = DataLoader()
    all_results: List[ComparisonResult] = []

    total_tests = len(ASSETS) * len(TIMEFRAMES) * len(STRATEGIES)
    completed = 0

    for asset in ASSETS:
        for timeframe in TIMEFRAMES:
            # Fetch data once per asset/timeframe combo
            print(f"[{asset}/{timeframe}] Fetching data...", end=" ", flush=True)

            df = await fetch_data_with_retry(
                data_loader, exchange, asset, timeframe, start_dt, end_dt
            )

            if df is None or len(df) < 100:
                print(f"Skipped (insufficient data)")
                completed += len(STRATEGIES)
                continue

            candles = len(df)
            print(f"{candles} candles", end=" | ", flush=True)

            # Run all 4 strategies
            strategy_results = []
            for strategy_name, tp_method, osc_mode in STRATEGIES:
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
                        timeframe=timeframe,
                        strategy=strategy_name,
                        total_trades=result.total_trades,
                        win_rate=result.win_rate,
                        total_pnl=result.total_pnl,
                        total_pnl_pct=result.total_pnl_percent,
                        max_drawdown=result.max_drawdown_percent,
                        profit_factor=result.profit_factor if result.profit_factor < 1000 else 999.99,
                        sharpe=result.sharpe_ratio,
                        candles=candles
                    )
                    all_results.append(comp_result)
                    strategy_results.append(f"{result.win_rate:.0f}%")

                except Exception as e:
                    strategy_results.append("ERR")

                completed += 1

            # Print compact results for this combo
            print(f"WR: {' / '.join(strategy_results)}")

            # Small delay between timeframes to avoid rate limiting
            await asyncio.sleep(1)

        # Delay between assets
        await asyncio.sleep(2)

    await exchange.disconnect()
    return all_results


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
        lines.append(f"\n  {'Timeframe':<10} | {'Candles':>8} | {'Strategy':<12} | {'Trades':>7} | {'Win%':>6} | {'PnL$':>10} | {'PnL%':>7} | {'MaxDD%':>7} | {'PF':>6} | {'Sharpe':>7}")
        lines.append(f"  {'-'*10}-+-{'-'*8}-+-{'-'*12}-+-{'-'*7}-+-{'-'*6}-+-{'-'*10}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}")

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
                    f"  {tf_label:<10} | {candles_label:>8} | {r.strategy:<12} | {r.total_trades:>7} | "
                    f"{r.win_rate:>5.1f}% | {r.total_pnl:>9,.0f}{pnl_marker} | {r.total_pnl_pct:>6.1f}% | "
                    f"{r.max_drawdown:>6.1f}% | {r.profit_factor:>6.2f} | {r.sharpe:>7.2f}"
                )

            lines.append(f"  {'-'*10}-+-{'-'*8}-+-{'-'*12}-+-{'-'*7}-+-{'-'*6}-+-{'-'*10}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}")

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
                row.append(f"{best.strategy}")
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
        wins_as_best = sum(1 for asset in ASSETS for tf in TIMEFRAMES
                         for r in results if r.asset == asset and r.timeframe == tf and r.strategy == strategy_name
                         and r.total_pnl == max(r2.total_pnl for r2 in results if r2.asset == asset and r2.timeframe == tf))

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

  Strategy Selection by Market Type:
  - Trending Markets: Full Signal or WT Cross (ride the trend)
  - Ranging Markets: Fixed R:R or 1st Reversal (quick exits)
  - High Volatility: 1st Reversal (protect profits quickly)
  - Low Volatility: Full Signal (wait for full moves)

  Timeframe Considerations:
  - Lower TFs (3m-15m): More noise, prefer quicker exits (1st Reversal, WT Cross)
  - Higher TFs (1h-4h): Cleaner signals, can use Full Signal

  Note: Results may vary with market conditions. This is historical analysis only.
    """)

    return "\n".join(lines)


async def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("  VMC Trading Bot - Full Multi-TF Comparison")
    print("=" * 60)

    start_time = time.time()
    results = await run_full_comparison()
    elapsed = time.time() - start_time

    if results:
        report = generate_report(results)
        print("\n" + report)

        # Save to file
        report_path = project_root / "output" / "full_comparison_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n[Report saved to: {report_path}]")
        print(f"[Completed in {elapsed:.1f}s]")
    else:
        print("\nNo results generated. Check errors above.")


if __name__ == "__main__":
    asyncio.run(main())
