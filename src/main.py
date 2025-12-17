"""Main entry point for VMC Trading Bot."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click

from config import Config, load_config, create_default_config
from core.bot import VMCBot
from utils.logger import setup_logger, get_logger
from notifications.discord import DiscordNotifier

logger = get_logger("main")


async def run_bot(config: Config, mode: str = "live") -> None:
    """
    Run the trading bot.

    Args:
        config: Bot configuration
        mode: Running mode ("live" or "paper")
    """
    # Initialize Discord notifier if enabled
    notifier = None
    if config.discord.enabled and config.discord.webhook_url:
        notifier = DiscordNotifier(
            webhook_url=config.discord.webhook_url,
            notify_on_signal=config.discord.notify_on_signal,
            notify_on_trade_open=config.discord.notify_on_trade_open,
            notify_on_trade_close=config.discord.notify_on_trade_close,
            notify_on_error=config.discord.notify_on_error
        )

    # Create bot
    bot = VMCBot(config)

    # Set notification callback
    if notifier:
        bot.set_notification_callback(notifier.notify)

    try:
        # Start bot
        await bot.start()

        # Calculate check interval based on timeframe
        timeframe_minutes = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "2h": 120, "4h": 240, "6h": 360,
            "12h": 720, "1d": 1440
        }
        interval = timeframe_minutes.get(config.trading.timeframe, 240) * 60  # seconds

        logger.info(f"Starting trading loop with {interval}s interval")

        # Run main loop
        await bot.run_loop(interval_seconds=interval)

    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    except Exception as e:
        logger.error(f"Bot error: {e}")
        raise
    finally:
        await bot.stop()
        if notifier:
            await notifier.close()


async def run_backtest(config: Config, start_date: str, end_date: str) -> None:
    """
    Run backtest.

    Args:
        config: Bot configuration
        start_date: Backtest start date
        end_date: Backtest end date
    """
    from datetime import datetime
    from backtest.engine import BacktestEngine
    from backtest.data_loader import DataLoader
    from backtest.metrics import calculate_metrics, plot_equity_curve
    from exchanges.hyperliquid import HyperliquidExchange
    from exchanges.bitunix import BitunixExchange
    from config import ExchangeName

    logger.info(f"Running backtest from {start_date} to {end_date}")

    # Create exchange client for data fetching
    if config.exchange.name == ExchangeName.HYPERLIQUID:
        exchange = HyperliquidExchange(
            api_key="",
            api_secret="",
            testnet=True
        )
    else:
        exchange = BitunixExchange(
            api_key="",
            api_secret="",
            testnet=True
        )

    # Connect to exchange
    await exchange.connect()

    # Load data
    data_loader = DataLoader()

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    for asset in config.trading.assets:
        logger.info(f"Backtesting {asset}...")

        # Try to load cached data first
        df = data_loader.load_cached(asset, config.trading.timeframe, start_dt, end_dt)

        if df is None:
            # Load from exchange
            df = await data_loader.load_from_exchange(
                exchange=exchange,
                symbol=asset,
                timeframe=config.trading.timeframe,
                start_date=start_dt,
                end_date=end_dt
            )
            # Cache the data
            data_loader.save_to_cache(df, asset, config.trading.timeframe)

        # Import take profit enums
        from config import TakeProfitMethod, OscillatorExitMode

        # Run backtest
        engine = BacktestEngine(
            initial_balance=config.backtest.initial_balance,
            risk_percent=config.trading.risk_percent,
            risk_reward=config.take_profit.risk_reward,
            commission_percent=config.backtest.commission_percent,
            anchor_level_long=config.indicators.wt_oversold_2,
            anchor_level_short=config.indicators.wt_overbought_2,
            tp_method=config.take_profit.method,
            oscillator_mode=config.take_profit.oscillator_mode
        )

        result = engine.run(df)

        # Print results
        print(f"\n{'='*50}")
        print(f"BACKTEST RESULTS: {asset}")
        print(f"{'='*50}")
        print(f"Initial Balance:   ${result.initial_balance:,.2f}")
        print(f"Final Balance:     ${result.final_balance:,.2f}")
        print(f"Total PnL:         ${result.total_pnl:,.2f} ({result.total_pnl_percent:.2f}%)")
        print(f"Max Drawdown:      {result.max_drawdown_percent:.2f}%")
        print(f"Total Trades:      {result.total_trades}")
        print(f"Win Rate:          {result.win_rate:.1f}%")
        print(f"Profit Factor:     {result.profit_factor:.2f}")
        print(f"Sharpe Ratio:      {result.sharpe_ratio:.2f}")
        print(f"Avg Win:           ${result.avg_win:.2f}")
        print(f"Avg Loss:          ${result.avg_loss:.2f}")

        # Plot equity curve
        try:
            plot_equity_curve(
                result.equity_curve,
                result.trades,
                title=f"{asset} Equity Curve",
                save_path=f"./output/{asset}_equity_curve.png"
            )
        except Exception as e:
            logger.warning(f"Could not plot equity curve: {e}")

    await exchange.disconnect()


@click.group()
def cli():
    """VMC Trading Bot CLI."""
    pass


@cli.command()
@click.option('--config', '-c', default='config/config.yaml', help='Path to config file')
@click.option('--mode', '-m', default='live', type=click.Choice(['live', 'paper']), help='Trading mode')
def run(config: str, mode: str):
    """Run the trading bot."""
    # Setup logging
    setup_logger(log_level="INFO", log_file="logs/bot.log")

    # Load config
    try:
        cfg = load_config(config)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config}")
        logger.info("Creating default config...")
        cfg = create_default_config(config)
        logger.info(f"Please edit {config} and restart")
        return

    # Run bot
    asyncio.run(run_bot(cfg, mode))


@cli.command()
@click.option('--config', '-c', default='config/config.yaml', help='Path to config file')
@click.option('--start', '-s', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end', '-e', required=True, help='End date (YYYY-MM-DD)')
def backtest(config: str, start: str, end: str):
    """Run backtest on historical data."""
    setup_logger(log_level="INFO")

    try:
        cfg = load_config(config)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config}")
        return

    asyncio.run(run_backtest(cfg, start, end))


@cli.command()
@click.option('--output', '-o', default='config/config.yaml', help='Output path')
def init(output: str):
    """Create default configuration file."""
    create_default_config(output)
    click.echo(f"Created default config at: {output}")
    click.echo("Please edit the config file with your API credentials and settings.")


if __name__ == "__main__":
    cli()
