#!/usr/bin/env python3
"""
VMC Trading Bot V6 - Production Runner

A production-grade wrapper for the VMC Trading Bot that provides:
- Automatic restart on crash with exponential backoff
- Graceful shutdown on SIGTERM/SIGINT (keeps positions open)
- Emergency position closing on crash
- Health monitoring and logging
- State persistence across restarts

Usage:
    python run_production.py --config config/config.yaml
    python run_production.py --config config/config.yaml --max-restarts 5
"""

import asyncio
import signal
import sys
import os
import time
import json
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config, load_config
from core.bot import VMCBot
from utils.logger import setup_logger, get_logger
from notifications.discord import DiscordNotifier

# Setup logging
setup_logger(log_level="INFO", log_file="logs/production.log")
logger = get_logger("production")


@dataclass
class RunnerState:
    """Persistent state for the production runner."""
    started_at: str
    restart_count: int
    last_crash_at: Optional[str]
    last_crash_reason: Optional[str]
    total_uptime_seconds: float
    successful_restarts: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RunnerState':
        return cls(**data)

    @classmethod
    def new(cls) -> 'RunnerState':
        return cls(
            started_at=datetime.now(timezone.utc).isoformat(),
            restart_count=0,
            last_crash_at=None,
            last_crash_reason=None,
            total_uptime_seconds=0,
            successful_restarts=0
        )


class Watchdog:
    """
    Watchdog to detect stuck/frozen bot and trigger restart.

    Monitors the heartbeat file and triggers callback if stale.
    """

    HEARTBEAT_FILE = Path("data/heartbeat.txt")

    def __init__(
        self,
        stale_threshold_seconds: int = 900,  # 15 minutes
        check_interval_seconds: int = 60,     # Check every minute
        startup_grace_seconds: int = 120,     # 2 minute grace period on startup
        on_stale_callback=None
    ):
        self.stale_threshold = stale_threshold_seconds
        self.check_interval = check_interval_seconds
        self.startup_grace = startup_grace_seconds
        self.on_stale_callback = on_stale_callback
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None

    def start(self):
        """Start the watchdog monitor thread."""
        self._running = True
        self._start_time = time.time()
        # Clear stale heartbeat file on startup
        if self.HEARTBEAT_FILE.exists():
            self.HEARTBEAT_FILE.unlink()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(f"Watchdog started (stale threshold: {self.stale_threshold}s, grace: {self.startup_grace}s)")

    def stop(self):
        """Stop the watchdog."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Watchdog stopped")

    def _monitor_loop(self):
        """Monitor loop running in background thread."""
        while self._running:
            try:
                if self._is_heartbeat_stale():
                    logger.warning("WATCHDOG: Heartbeat is stale - bot appears frozen!")
                    if self.on_stale_callback:
                        self.on_stale_callback()
            except Exception as e:
                logger.error(f"Watchdog error: {e}")

            time.sleep(self.check_interval)

    def _is_heartbeat_stale(self) -> bool:
        """Check if heartbeat file is stale."""
        # Check grace period - don't trigger during startup
        if self._start_time and (time.time() - self._start_time) < self.startup_grace:
            return False

        if not self.HEARTBEAT_FILE.exists():
            # No heartbeat file yet - bot might still be starting
            return False

        try:
            content = self.HEARTBEAT_FILE.read_text()
            lines = content.strip().split('\n')
            if not lines:
                return True

            # Parse timestamp from first line
            timestamp_str = lines[0]
            heartbeat_time = datetime.fromisoformat(timestamp_str)

            # Compare with current time
            now = datetime.now(timezone.utc)
            # Handle naive datetime from heartbeat
            if heartbeat_time.tzinfo is None:
                heartbeat_time = heartbeat_time.replace(tzinfo=timezone.utc)

            age_seconds = (now - heartbeat_time).total_seconds()

            if age_seconds > self.stale_threshold:
                logger.warning(f"Heartbeat age: {age_seconds:.0f}s (threshold: {self.stale_threshold}s)")
                return True

            return False

        except Exception as e:
            logger.warning(f"Error reading heartbeat: {e}")
            return False


class ProductionRunner:
    """
    Production-grade wrapper for VMC Trading Bot.

    Features:
    - Auto-restart on crash with exponential backoff
    - Graceful shutdown preserves positions
    - Crash handling closes positions for safety
    - State persistence for monitoring
    - Health logging
    """

    # Backoff schedule (seconds): 30s, 1m, 2m, 5m, 10m (then stays at 10m)
    BACKOFF_SCHEDULE = [30, 60, 120, 300, 600]

    # State file for persistence
    STATE_FILE = "data/runner_state.json"

    def __init__(
        self,
        config_path: str,
        max_restarts: int = 10,
        close_positions_on_crash: bool = True
    ):
        """
        Initialize the production runner.

        Args:
            config_path: Path to configuration file
            max_restarts: Maximum number of restarts before giving up
            close_positions_on_crash: Whether to close positions when bot crashes
        """
        self.config_path = config_path
        self.max_restarts = max_restarts
        self.close_positions_on_crash = close_positions_on_crash

        # Bot instance
        self.bot: Optional[VMCBot] = None
        self.config: Optional[Config] = None
        self.notifier: Optional[DiscordNotifier] = None

        # State
        self.state = self._load_state()
        self.shutdown_requested = False
        self.is_crashed = False
        self.start_time: Optional[float] = None

        # Watchdog for detecting frozen bot
        self.watchdog = Watchdog(
            stale_threshold_seconds=900,  # 15 minutes
            check_interval_seconds=60,
            on_stale_callback=self._handle_watchdog_timeout
        )

        # Setup signal handlers
        self._setup_signals()

    def _setup_signals(self):
        """Setup signal handlers for graceful shutdown."""
        # Handle SIGTERM (from systemd/docker)
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
        # Handle SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)

        # On Windows, also handle SIGBREAK
        if sys.platform == 'win32':
            signal.signal(signal.SIGBREAK, self._handle_shutdown_signal)

        logger.info("Signal handlers configured (SIGTERM, SIGINT)")

    def _handle_shutdown_signal(self, signum, frame):
        """Handle shutdown signals gracefully."""
        signal_name = signal.Signals(signum).name
        logger.info(f"Received {signal_name} signal - initiating graceful shutdown")
        self.shutdown_requested = True

        # If bot is running, stop the event loop
        if self.bot and self.bot._running:
            self.bot._running = False

    def _handle_watchdog_timeout(self):
        """Handle watchdog timeout - bot appears frozen."""
        logger.error("WATCHDOG TIMEOUT: Bot is frozen, forcing restart...")
        self.state.last_crash_reason = "Watchdog timeout - bot frozen"
        self.state.last_crash_at = datetime.now(timezone.utc).isoformat()
        self._save_state()
        self.is_crashed = True

        # Force stop the bot
        if self.bot and self.bot._running:
            self.bot._running = False

        # Force exit the process - will be restarted by external supervisor
        # This is necessary because the bot might be stuck in a blocking call
        logger.error("Forcing process exit for restart...")
        os._exit(1)

    def _load_state(self) -> RunnerState:
        """Load state from file or create new."""
        state_path = Path(self.STATE_FILE)
        if state_path.exists():
            try:
                with open(state_path, 'r') as f:
                    data = json.load(f)
                    state = RunnerState.from_dict(data)
                    logger.info(f"Loaded runner state: {state.restart_count} previous restarts")
                    return state
            except Exception as e:
                logger.warning(f"Could not load state file: {e}")

        return RunnerState.new()

    def _save_state(self):
        """Save state to file."""
        state_path = Path(self.STATE_FILE)
        state_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(state_path, 'w') as f:
                json.dump(self.state.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save state file: {e}")

    def _reset_state(self):
        """Reset state for a fresh start."""
        self.state = RunnerState.new()
        self._save_state()

    async def _close_all_positions(self):
        """Emergency close all positions on crash."""
        if not self.bot or not self.bot.exchange or not self.bot.trade_manager:
            return

        logger.warning("EMERGENCY: Closing all open positions due to crash")

        try:
            active_trades = list(self.bot.trade_manager._active_trades.values())

            for trade in active_trades:
                try:
                    logger.info(f"Closing position: {trade.symbol}")
                    await self.bot.exchange.close_position(trade.symbol)

                    # Notify
                    if self.notifier:
                        await self.notifier.notify("emergency_close", {
                            "symbol": trade.symbol,
                            "reason": "Bot crash",
                            "trade_id": trade.trade_id
                        })

                except Exception as e:
                    logger.error(f"Failed to close position {trade.symbol}: {e}")

            logger.info(f"Closed {len(active_trades)} positions")

        except Exception as e:
            logger.error(f"Error during emergency position closing: {e}")

    async def _run_bot_instance(self) -> bool:
        """
        Run a single bot instance.

        Returns:
            True if shutdown was graceful, False if crashed
        """
        self.start_time = time.time()

        try:
            # Load config
            self.config = load_config(self.config_path)

            # MAINNET WARNING
            if not self.config.exchange.testnet:
                logger.warning("=" * 60)
                logger.warning(">>> MAINNET MODE - REAL MONEY AT RISK! <<<")
                logger.warning("=" * 60)
                logger.warning("You are running on Hyperliquid MAINNET.")
                logger.warning("All trades will use REAL funds.")
                logger.warning("Make sure you understand the risks!")
                logger.warning("=" * 60)
            else:
                logger.info("Running on TESTNET (paper trading)")

            # Initialize Discord notifier
            if self.config.discord.enabled and self.config.discord.webhook_url:
                self.notifier = DiscordNotifier(
                    webhook_url=self.config.discord.webhook_url,
                    notify_on_signal=self.config.discord.notify_on_signal,
                    notify_on_trade_open=self.config.discord.notify_on_trade_open,
                    notify_on_trade_close=self.config.discord.notify_on_trade_close,
                    notify_on_error=self.config.discord.notify_on_error
                )

            # Create bot
            self.bot = VMCBot(self.config)

            if self.notifier:
                self.bot.set_notification_callback(self.notifier.notify)

            # Start bot
            await self.bot.start()

            # Calculate interval
            timeframe_minutes = {
                "1m": 1, "5m": 5, "15m": 15, "30m": 30,
                "1h": 60, "2h": 120, "4h": 240, "6h": 360,
                "12h": 720, "1d": 1440
            }
            interval = timeframe_minutes.get(self.config.trading.timeframe, 240) * 60

            logger.info(f"Bot started. Check interval: {interval}s")

            # Notify restart if applicable
            if self.state.restart_count > 0:
                await self._notify("bot_restarted", {
                    "restart_count": self.state.restart_count,
                    "last_crash_reason": self.state.last_crash_reason
                })

            # Run main loop
            await self.bot.run_loop(interval_seconds=interval)

            # If we get here, it was a graceful shutdown
            return True

        except asyncio.CancelledError:
            logger.info("Bot task cancelled")
            return True

        except Exception as e:
            logger.error(f"Bot crashed: {e}", exc_info=True)
            self.state.last_crash_at = datetime.utcnow().isoformat()
            self.state.last_crash_reason = str(e)
            self.is_crashed = True
            return False

        finally:
            # Update uptime
            if self.start_time:
                self.state.total_uptime_seconds += time.time() - self.start_time

            # Stop bot gracefully
            if self.bot:
                await self.bot.stop()

            # Close notifier
            if self.notifier:
                await self.notifier.close()
                self.notifier = None

    async def _notify(self, event_type: str, data: Dict[str, Any]):
        """Send notification via Discord."""
        if self.notifier:
            try:
                await self.notifier.notify(event_type, data)
            except Exception as e:
                logger.warning(f"Notification failed: {e}")

    async def run(self):
        """
        Run the production wrapper.

        Handles restarts and shutdown.
        """
        logger.info("=" * 60)
        logger.info("VMC Trading Bot V6 - Production Runner")
        logger.info("=" * 60)
        logger.info(f"Config: {self.config_path}")
        logger.info(f"Max restarts: {self.max_restarts}")
        logger.info(f"Close positions on crash: {self.close_positions_on_crash}")
        logger.info(f"Watchdog: enabled (15min timeout)")
        logger.info("=" * 60)

        # Start watchdog
        self.watchdog.start()

        while not self.shutdown_requested:
            # Check restart limit
            if self.state.restart_count >= self.max_restarts:
                logger.error(f"Max restarts ({self.max_restarts}) exceeded. Giving up.")
                await self._notify("bot_max_restarts", {
                    "restart_count": self.state.restart_count,
                    "max_restarts": self.max_restarts
                })
                break

            # Run bot
            self.is_crashed = False
            graceful = await self._run_bot_instance()

            if self.shutdown_requested:
                logger.info("Graceful shutdown completed. Positions preserved.")
                break

            if not graceful:
                # Crash handling
                self.state.restart_count += 1
                self._save_state()

                # Close positions on crash if enabled
                if self.close_positions_on_crash:
                    await self._close_all_positions()

                # Calculate backoff
                backoff_idx = min(self.state.restart_count - 1, len(self.BACKOFF_SCHEDULE) - 1)
                backoff = self.BACKOFF_SCHEDULE[backoff_idx]

                logger.warning(f"Crash #{self.state.restart_count}. Restarting in {backoff}s...")

                await self._notify("bot_crashed", {
                    "restart_count": self.state.restart_count,
                    "backoff_seconds": backoff,
                    "reason": self.state.last_crash_reason,
                    "positions_closed": self.close_positions_on_crash
                })

                # Wait with backoff
                await asyncio.sleep(backoff)

                # Increment successful restart counter if we make it past the wait
                if not self.shutdown_requested:
                    self.state.successful_restarts += 1
                    self._save_state()
                    logger.info(f"Restarting bot (attempt {self.state.restart_count + 1})...")

        # Stop watchdog
        self.watchdog.stop()

        # Final state save
        self._save_state()

        logger.info("=" * 60)
        logger.info("Production Runner Shutdown Summary")
        logger.info(f"  Total uptime: {self.state.total_uptime_seconds:.0f}s")
        logger.info(f"  Restart count: {self.state.restart_count}")
        logger.info(f"  Graceful shutdown: {not self.is_crashed}")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='VMC Trading Bot V6 - Production Runner')
    parser.add_argument(
        '--config', '-c',
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--max-restarts', '-m',
        type=int,
        default=10,
        help='Maximum number of automatic restarts (default: 10)'
    )
    parser.add_argument(
        '--no-close-positions',
        action='store_true',
        help='Do NOT close positions on crash (default: close them)'
    )
    parser.add_argument(
        '--reset-state',
        action='store_true',
        help='Reset the runner state (restart count, etc.)'
    )

    args = parser.parse_args()

    # Verify config exists
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        logger.info("Run 'python setup_wizard.py' to create configuration")
        sys.exit(1)

    # Create runner
    runner = ProductionRunner(
        config_path=args.config,
        max_restarts=args.max_restarts,
        close_positions_on_crash=not args.no_close_positions
    )

    # Reset state if requested
    if args.reset_state:
        runner._reset_state()
        logger.info("Runner state reset")

    # Run
    try:
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    sys.exit(0)


if __name__ == "__main__":
    main()
