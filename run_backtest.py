#!/usr/bin/env python3
"""
VMC Trading Bot V6 - Client Backtest Script

Simple script for clients to verify strategy performance on historical data.
Uses the same strategies and settings as the live bot.

Usage:
    python run_backtest.py                     # Test all 15 strategies (1 year)
    python run_backtest.py --asset BTC         # Test BTC strategies only
    python run_backtest.py --asset SOL --days 90   # Test SOL strategies (90 days)
"""

import sys
import asyncio
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd

from config.config import load_config, StrategyInstanceConfig
from backtest.engine import BacktestEngine, BacktestResult, SignalMode, TakeProfitMethod, OscillatorExitMode
from backtest.data_loader import DataLoader
from exchanges.hyperliquid import HyperliquidExchange
from utils.logger import setup_logger, get_logger

# Setup logging
setup_logger(log_level="INFO")
logger = get_logger("backtest")

# Constants
INITIAL_BALANCE = 10000.0
RISK_PERCENT = 3.0
COMMISSION = 0.06  # 0.06% (engine divides by 100)


@dataclass
class StrategyResult:
    """Result for a single strategy backtest."""
    strategy_name: str
    asset: str
    timeframe: str
    signal_mode: str
    time_filter: str
    total_trades: int
    winning_trades: int
    win_rate: float
    total_pnl: float
    pnl_percent: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    trades: List


def get_signal_mode_enum(mode_str: str, anchor_level: int) -> SignalMode:
    """Convert config signal mode to engine enum."""
    if mode_str == "simple":
        return SignalMode.SIMPLE
    elif mode_str == "enhanced":
        if anchor_level >= 70:
            return SignalMode.ENHANCED  # Use ENHANCED for 70
        else:
            return SignalMode.ENHANCED  # Use ENHANCED for 60 too
    return SignalMode.SIMPLE


async def load_historical_data(
    asset: str,
    timeframe: str,
    days: int,
    data_loader: DataLoader,
    exchange: Optional[HyperliquidExchange] = None
) -> Optional[pd.DataFrame]:
    """Load historical data from cache or exchange."""

    # Try cache first
    cache_dir = Path("data/binance_cache")
    cache_file = cache_dir / f"{asset.lower()}_{timeframe}.csv"

    if cache_file.exists():
        try:
            df = pd.read_csv(cache_file, parse_dates=['timestamp'], index_col='timestamp')
            # Filter to requested date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)

            if df.index.min() <= start_date:
                mask = (df.index >= start_date) & (df.index <= end_date)
                filtered_df = df[mask]
                if len(filtered_df) > 100:
                    logger.info(f"Loaded {len(filtered_df)} {timeframe} candles for {asset} from cache")
                    return filtered_df
        except Exception as e:
            logger.warning(f"Error loading cache for {asset}/{timeframe}: {e}")

    # Try loading from exchange if available
    if exchange:
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            df = await data_loader.load_from_exchange(
                exchange=exchange,
                symbol=asset,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            if len(df) > 100:
                # Save to cache
                data_loader.save_to_cache(df, asset, timeframe)
                return df
        except Exception as e:
            logger.warning(f"Error fetching from exchange for {asset}/{timeframe}: {e}")

    return None


def run_strategy_backtest(
    strategy_name: str,
    strategy_config: StrategyInstanceConfig,
    data: pd.DataFrame,
    anchor_level: int = 53
) -> Optional[StrategyResult]:
    """Run backtest for a single strategy."""

    signal_mode = get_signal_mode_enum(strategy_config.signal_mode, strategy_config.anchor_level)

    # Create engine with strategy settings
    engine = BacktestEngine(
        initial_balance=INITIAL_BALANCE,
        risk_percent=RISK_PERCENT,
        commission_percent=COMMISSION,
        signal_mode=signal_mode,
        tp_method=TakeProfitMethod.OSCILLATOR,  # V6 uses oscillator exit
        oscillator_mode=OscillatorExitMode.FULL_SIGNAL,
        simple_overbought=strategy_config.anchor_level,
        simple_oversold=-strategy_config.anchor_level,
        direction_filter=strategy_config.direction_filter
    )

    # NOTE: BacktestEngine already uses real VWAP from VWAPCalculator internally.
    # The use_vwap_confirmation config controls VWAP entry confirmation in the
    # signal detector, which stores VWAP in signal metadata for logging purposes.

    # Run backtest
    try:
        result = engine.run(data)

        time_filter_mode = "all_hours"
        if strategy_config.time_filter and strategy_config.time_filter.enabled:
            time_filter_mode = strategy_config.time_filter.mode.value if hasattr(strategy_config.time_filter.mode, 'value') else str(strategy_config.time_filter.mode)

        return StrategyResult(
            strategy_name=strategy_name,
            asset=strategy_config.asset,
            timeframe=strategy_config.timeframe,
            signal_mode=strategy_config.signal_mode,
            time_filter=time_filter_mode,
            total_trades=result.total_trades,
            winning_trades=result.winning_trades,
            win_rate=result.win_rate,
            total_pnl=result.total_pnl,
            pnl_percent=result.total_pnl_percent,
            max_drawdown=result.max_drawdown_percent,
            sharpe_ratio=result.sharpe_ratio,
            profit_factor=result.profit_factor,
            trades=result.trades
        )
    except Exception as e:
        logger.error(f"Error running backtest for {strategy_name}: {e}")
        return None


def filter_data_by_time(
    data: pd.DataFrame,
    time_filter_mode: str
) -> pd.DataFrame:
    """Filter data based on time filter mode."""
    if time_filter_mode == "all_hours" or not time_filter_mode:
        return data

    df = data.copy()

    if time_filter_mode == "ny_hours_only":
        # NY hours: 14:00-21:00 UTC on weekdays
        df['hour'] = df.index.hour
        df['weekday'] = df.index.weekday
        mask = (df['hour'] >= 14) & (df['hour'] < 21) & (df['weekday'] < 5)
        df = df[mask].drop(columns=['hour', 'weekday'])

    elif time_filter_mode == "weekends_only":
        # Weekends: Saturday (5) and Sunday (6)
        df['weekday'] = df.index.weekday
        mask = df['weekday'] >= 5
        df = df[mask].drop(columns=['weekday'])

    return df


def print_summary(results: List[StrategyResult], days: int):
    """Print backtest summary."""
    print("\n" + "=" * 70)
    print("VMC Trading Bot V6 - Backtest Results")
    print("=" * 70)
    print(f"Period: Last {days} days")
    print(f"Starting Capital: ${INITIAL_BALANCE:,.0f}")
    print(f"Risk Per Trade: {RISK_PERCENT}%")
    print("=" * 70)

    # Group by asset
    by_asset: Dict[str, List[StrategyResult]] = {}
    for r in results:
        if r.asset not in by_asset:
            by_asset[r.asset] = []
        by_asset[r.asset].append(r)

    # Asset summary
    print("\nRESULTS BY ASSET:")
    print("-" * 70)
    print(f"{'Asset':<8} {'PnL':>12} {'Win Rate':>10} {'Trades':>8} {'Sharpe':>8} {'Max DD':>10}")
    print("-" * 70)

    total_pnl = 0
    total_trades = 0

    for asset in ['BTC', 'ETH', 'SOL']:
        if asset not in by_asset:
            continue

        asset_results = by_asset[asset]
        asset_pnl = sum(r.total_pnl for r in asset_results)
        asset_trades = sum(r.total_trades for r in asset_results)
        asset_wins = sum(r.winning_trades for r in asset_results)
        asset_win_rate = (asset_wins / asset_trades * 100) if asset_trades > 0 else 0
        avg_sharpe = sum(r.sharpe_ratio for r in asset_results) / len(asset_results)
        max_dd = max(r.max_drawdown for r in asset_results)

        print(f"{asset:<8} ${asset_pnl:>10,.0f} {asset_win_rate:>9.1f}% {asset_trades:>8} {avg_sharpe:>8.2f} {max_dd:>9.1f}%")

        total_pnl += asset_pnl
        total_trades += asset_trades

    print("-" * 70)
    print(f"{'TOTAL':<8} ${total_pnl:>10,.0f} {'':>10} {total_trades:>8}")
    print("=" * 70)

    # Strategy details
    print("\nSTRATEGY DETAILS:")
    print("-" * 70)
    print(f"{'Strategy':<30} {'TF':>5} {'Mode':>10} {'Trades':>7} {'Win%':>7} {'PnL':>12}")
    print("-" * 70)

    for r in sorted(results, key=lambda x: x.total_pnl, reverse=True):
        mode = f"{r.signal_mode}"
        if "enhanced" in r.signal_mode.lower():
            mode = f"enh_{r.time_filter[:3]}"
        print(f"{r.strategy_name:<30} {r.timeframe:>5} {mode:>10} {r.total_trades:>7} {r.win_rate:>6.1f}% ${r.total_pnl:>10,.0f}")

    print("=" * 70)


def export_trades_csv(results: List[StrategyResult], output_dir: Path):
    """Export all trades to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_trades = []
    for r in results:
        for trade in r.trades:
            trade_dict = trade.to_dict() if hasattr(trade, 'to_dict') else {
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'signal_type': trade.signal_type.value if hasattr(trade.signal_type, 'value') else str(trade.signal_type),
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'pnl': trade.pnl,
                'pnl_percent': trade.pnl_percent,
                'exit_reason': trade.exit_reason,
            }
            trade_dict['strategy'] = r.strategy_name
            trade_dict['asset'] = r.asset
            trade_dict['timeframe'] = r.timeframe
            all_trades.append(trade_dict)

    if all_trades:
        df = pd.DataFrame(all_trades)
        output_file = output_dir / "backtest_trades.csv"
        df.to_csv(output_file, index=False)
        print(f"\nTrade log saved to: {output_file}")
    else:
        print("\nNo trades to export.")


async def main():
    parser = argparse.ArgumentParser(
        description='VMC Trading Bot V6 - Backtest Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_backtest.py                    # Test all strategies (1 year)
    python run_backtest.py --asset BTC        # Test BTC only
    python run_backtest.py --days 90          # Test last 90 days
    python run_backtest.py --asset SOL --days 180  # Test SOL (6 months)
        """
    )
    parser.add_argument(
        '--asset', '-a',
        choices=['BTC', 'ETH', 'SOL'],
        help='Test specific asset only'
    )
    parser.add_argument(
        '--days', '-d',
        type=int,
        default=365,
        help='Number of days to backtest (default: 365)'
    )
    parser.add_argument(
        '--config', '-c',
        default='config/config_v6_production.yaml',
        help='Config file path'
    )
    parser.add_argument(
        '--output', '-o',
        default='output',
        help='Output directory for trade logs'
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("VMC Trading Bot V6 - Backtest")
    print("=" * 70)

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}")
        print("Please run setup_wizard.py first or specify a valid config file.")
        sys.exit(1)

    config = load_config(str(config_path))
    strategies = config.get_enabled_strategies()

    if not strategies:
        print("Error: No V6 strategies found in config.")
        print("Make sure your config file has V6 strategies defined.")
        sys.exit(1)

    print(f"Loaded {len(strategies)} strategies from config")

    # Filter by asset if specified
    if args.asset:
        strategies = {k: v for k, v in strategies.items() if v.asset == args.asset}
        print(f"Filtering to {args.asset}: {len(strategies)} strategies")

    # Initialize data loader
    data_loader = DataLoader(cache_dir="data/cache")

    # Try to connect to exchange for fresh data (optional)
    exchange = None
    try:
        exchange = HyperliquidExchange(
            api_key="",
            api_secret="",
            testnet=True
        )
        await exchange.connect()
        print("Connected to exchange for data")
    except Exception as e:
        logger.warning(f"Could not connect to exchange: {e}")
        print("Using cached data only")

    # Get unique timeframes needed
    timeframes = set()
    assets = set()
    for strat in strategies.values():
        timeframes.add(strat.timeframe)
        assets.add(strat.asset)

    print(f"Assets: {sorted(assets)}")
    print(f"Timeframes: {sorted(timeframes)}")
    print(f"Period: {args.days} days")
    print("-" * 70)

    # Load data for each asset/timeframe
    data_cache: Dict[str, Dict[str, pd.DataFrame]] = {}

    for asset in assets:
        data_cache[asset] = {}
        for tf in timeframes:
            print(f"Loading {asset} {tf} data...", end=" ", flush=True)
            df = await load_historical_data(asset, tf, args.days, data_loader, exchange)
            if df is not None and len(df) > 100:
                data_cache[asset][tf] = df
                print(f"{len(df)} candles")
            else:
                print("Not enough data")

    # Disconnect exchange
    if exchange:
        await exchange.disconnect()

    # Run backtests
    print("\nRunning backtests...")
    print("-" * 70)

    results: List[StrategyResult] = []

    for strat_name, strat_config in strategies.items():
        asset = strat_config.asset
        tf = strat_config.timeframe

        if asset not in data_cache or tf not in data_cache[asset]:
            print(f"  {strat_name}: Skipped (no data)")
            continue

        data = data_cache[asset][tf]

        # Apply time filter to data
        time_filter_mode = "all_hours"
        if strat_config.time_filter and strat_config.time_filter.enabled:
            time_filter_mode = strat_config.time_filter.mode.value if hasattr(strat_config.time_filter.mode, 'value') else str(strat_config.time_filter.mode)

        filtered_data = filter_data_by_time(data, time_filter_mode)

        if len(filtered_data) < 100:
            print(f"  {strat_name}: Skipped (not enough filtered data)")
            continue

        print(f"  {strat_name}: Testing on {len(filtered_data)} candles...", end=" ", flush=True)

        result = run_strategy_backtest(strat_name, strat_config, filtered_data)
        if result:
            results.append(result)
            print(f"{result.total_trades} trades, ${result.total_pnl:,.0f}")
        else:
            print("Failed")

    # Print summary
    if results:
        print_summary(results, args.days)
        export_trades_csv(results, Path(args.output))
    else:
        print("\nNo results to display. Check data availability.")


if __name__ == "__main__":
    asyncio.run(main())
