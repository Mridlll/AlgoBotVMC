"""Main VMC Trading Bot orchestrator."""

import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import pandas as pd

from core.state import TradingState, BotState, AssetState
from config import Config, ExchangeName
from indicators import HeikinAshi, WaveTrend, MoneyFlow
from indicators.heikin_ashi import convert_to_heikin_ashi
from exchanges.base import BaseExchange, Candle, Position, OrderStatus
from exchanges.hyperliquid import HyperliquidExchange
from exchanges.bitunix import BitunixExchange
from strategy.signals import SignalDetector, Signal
from strategy.risk import RiskManager, StopLossMethod, TakeProfitMethod
from strategy.trade_manager import TradeManager, Trade, TradeStatus
from strategy.multi_timeframe import MultiTimeframeCoordinator, MultiTimeframeSignal
from strategy.v6_processor import V6SignalProcessor, V6ScanResult
from persistence.trade_store import TradeStore
from utils.logger import get_logger, setup_logger
from utils.daily_summary import DailySummaryLogger

logger = get_logger("vmc_bot")


class VMCBot:
    """
    Main VMC Trading Bot.

    Orchestrates all components:
    - Exchange connection
    - Data fetching
    - Indicator calculation
    - Signal detection
    - Trade execution
    """

    def __init__(self, config: Config):
        """
        Initialize the bot.

        Args:
            config: Bot configuration
        """
        self.config = config
        self.state = TradingState()

        # Components (initialized in start())
        self.exchange: Optional[BaseExchange] = None
        self.signal_detectors: Dict[str, SignalDetector] = {}
        self.risk_manager: Optional[RiskManager] = None
        self.trade_manager: Optional[TradeManager] = None
        self.trade_store: Optional[TradeStore] = None
        self._session_id: int = -1

        # Multi-timeframe coordinators (per asset)
        self.mtf_coordinators: Dict[str, MultiTimeframeCoordinator] = {}
        self.use_multi_timeframe = config.trading.use_multi_timeframe

        # V6 multi-strategy processor
        self.v6_processor: Optional[V6SignalProcessor] = None
        self.use_v6_strategies = config.has_v6_strategies()

        # Indicators
        self.heikin_ashi = HeikinAshi()
        self.wavetrend = WaveTrend(
            channel_len=config.indicators.wt_channel_len,
            average_len=config.indicators.wt_average_len,
            ma_len=config.indicators.wt_ma_len,
            overbought_2=config.indicators.wt_overbought_2,
            oversold_2=config.indicators.wt_oversold_2
        )
        self.money_flow = MoneyFlow(
            period=config.indicators.mfi_period,
            multiplier=config.indicators.mfi_multiplier,
            y_pos=config.indicators.mfi_y_pos
        )

        # Notification callback
        self._notification_callback = None

        # Daily summary logger
        self.daily_summary = DailySummaryLogger(log_dir="logs")

        # Running flag
        self._running = False

        # Heartbeat file for watchdog
        self._heartbeat_file = Path("data/heartbeat.txt")
        self._last_heartbeat = datetime.utcnow()

    def set_notification_callback(self, callback) -> None:
        """Set callback for notifications."""
        self._notification_callback = callback

    async def _notify(self, event_type: str, data: Dict[str, Any]) -> None:
        """Send notification if callback is set."""
        if self._notification_callback:
            try:
                await self._notification_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Notification failed: {e}")

    def _update_heartbeat(self, active_trades: int = 0, signals_found: int = 0) -> None:
        """Update heartbeat file for watchdog monitoring."""
        try:
            self._heartbeat_file.parent.mkdir(parents=True, exist_ok=True)
            now = datetime.utcnow()
            self._last_heartbeat = now
            self._heartbeat_file.write_text(
                f"{now.isoformat()}\n"
                f"active_trades={active_trades}\n"
                f"signals_found={signals_found}\n"
            )
        except Exception as e:
            logger.warning(f"Failed to update heartbeat: {e}")

    def _create_exchange(self) -> BaseExchange:
        """Create exchange client based on config."""
        if self.config.exchange.name == ExchangeName.HYPERLIQUID:
            return HyperliquidExchange(
                api_key=self.config.exchange.api_key,
                api_secret=self.config.exchange.api_secret,
                wallet_address=self.config.exchange.wallet_address,
                account_address=self.config.exchange.account_address,
                testnet=self.config.exchange.testnet
            )
        elif self.config.exchange.name == ExchangeName.BITUNIX:
            return BitunixExchange(
                api_key=self.config.exchange.api_key,
                api_secret=self.config.exchange.api_secret,
                testnet=self.config.exchange.testnet
            )
        else:
            raise ValueError(f"Unsupported exchange: {self.config.exchange.name}")

    async def start(self) -> None:
        """Start the trading bot."""
        logger.info("Starting VMC Trading Bot...")
        self.state.bot_state = BotState.STARTING

        try:
            # Create and connect to exchange
            self.exchange = self._create_exchange()
            connected = await self.exchange.connect()

            if not connected:
                raise RuntimeError("Failed to connect to exchange")

            logger.info(f"Connected to {self.exchange.name}")

            # Initialize risk manager
            self.risk_manager = RiskManager(
                default_risk_percent=self.config.trading.risk_percent,
                default_leverage=self.config.trading.leverage,
                default_rr=self.config.take_profit.risk_reward,
                swing_lookback=self.config.stop_loss.swing_lookback,
                swing_buffer_percent=self.config.stop_loss.buffer_percent,
                atr_period=self.config.stop_loss.atr_period,
                atr_multiplier=self.config.stop_loss.atr_multiplier
            )

            # Initialize trade manager
            self.trade_manager = TradeManager(
                exchange=self.exchange,
                risk_manager=self.risk_manager,
                tp_method=TakeProfitMethod(self.config.take_profit.method.value),
                sl_method=StopLossMethod(self.config.stop_loss.method.value),
                max_positions=self.config.trading.max_positions,
                max_positions_per_asset=self.config.trading.max_positions_per_asset
            )

            # Initialize trade persistence
            self.trade_store = TradeStore(db_path="data/trades.db")

            # Start new session and get initial balance
            balance = await self.exchange.get_balance()
            self._session_id = self.trade_store.start_session(balance.total_balance)

            # Load and recover any active trades from previous sessions
            await self._recover_trades_from_store()

            # Initialize V6 multi-strategy processor if strategies configured
            if self.use_v6_strategies:
                self.v6_processor = V6SignalProcessor(self.config)
                enabled_strategies = self.config.get_enabled_strategies()
                logger.info(f"V6 mode: {len(enabled_strategies)} strategies enabled")
                logger.info(self.v6_processor.get_strategy_summary())
            else:
                # Fallback: Initialize signal detectors for each asset (legacy mode)
                for asset in self.config.trading.assets:
                    if self.use_multi_timeframe:
                        # Use MTF coordinator for this asset
                        tf_config = self.config.trading.timeframes
                        self.mtf_coordinators[asset] = MultiTimeframeCoordinator(
                            entry_timeframes=tf_config.entry,
                            bias_timeframes=tf_config.bias,
                            bias_weights=tf_config.bias_weights,
                            require_bias_alignment=tf_config.require_bias_alignment,
                            min_bias_aligned=tf_config.min_bias_timeframes_aligned,
                            entry_on_any_timeframe=tf_config.entry_on_any_timeframe,
                            anchor_level_long=self.config.indicators.wt_oversold_2,
                            anchor_level_short=self.config.indicators.wt_overbought_2,
                        )
                        logger.info(f"MTF coordinator initialized for {asset}: entry={tf_config.entry}, bias={tf_config.bias}")
                    else:
                        # Use single timeframe detector
                        self.signal_detectors[asset] = SignalDetector(
                            anchor_level_long=self.config.indicators.wt_oversold_2,
                            anchor_level_short=self.config.indicators.wt_overbought_2,
                            timeframe=self.config.trading.timeframe
                        )

            # Set leverage for each asset
            for asset in self.config.trading.assets:
                symbol = self.exchange.format_symbol(asset)
                await self.exchange.set_leverage(symbol, self.config.trading.leverage)

            self.state.bot_state = BotState.RUNNING
            self.state.started_at = datetime.utcnow()
            self._running = True

            logger.info("VMC Trading Bot started successfully")
            await self._notify("bot_started", {"assets": self.config.trading.assets})

        except Exception as e:
            self.state.bot_state = BotState.ERROR
            self.state.last_error = str(e)
            logger.error(f"Failed to start bot: {e}")
            raise

    async def stop(self) -> None:
        """Stop the trading bot."""
        logger.info("Stopping VMC Trading Bot...")
        self._running = False
        self.state.bot_state = BotState.STOPPED

        # End trading session
        if self.trade_store and self._session_id > 0:
            try:
                balance = await self.exchange.get_balance() if self.exchange else None
                final_balance = balance.total_balance if balance else 0
                stats = self.trade_manager.get_stats() if self.trade_manager else {}

                self.trade_store.end_session(
                    session_id=self._session_id,
                    final_balance=final_balance,
                    trades_opened=stats.get("total_trades", 0),
                    trades_closed=stats.get("total_trades", 0),
                    total_pnl=stats.get("total_pnl", 0)
                )
            except Exception as e:
                logger.warning(f"Failed to end session: {e}")

            self.trade_store.close()

        if self.exchange:
            await self.exchange.disconnect()

        logger.info("VMC Trading Bot stopped")
        await self._notify("bot_stopped", {})

    async def _recover_trades_from_store(self) -> None:
        """
        Recover active trades from database on startup.

        This allows the bot to resume managing trades that were open
        when it was last stopped (e.g., crash recovery).
        """
        if not self.trade_store or not self.trade_manager:
            return

        try:
            active_trades = self.trade_store.load_active_trades()

            if not active_trades:
                logger.info("No active trades to recover from database")
                return

            logger.info(f"Recovering {len(active_trades)} trades from database...")

            for trade_dict in active_trades:
                try:
                    # Verify position still exists on exchange
                    symbol = trade_dict.get('symbol', '')
                    position = await self.exchange.get_position(symbol)

                    if not position or abs(position.size) == 0:
                        # Position was closed externally, mark as closed
                        logger.info(f"Trade {trade_dict['trade_id']} position no longer exists, marking closed")
                        self.trade_store.mark_trade_closed(
                            trade_id=trade_dict['trade_id'],
                            exit_price=trade_dict.get('entry_price', 0),  # Unknown exit
                            pnl=0,
                            pnl_percent=0,
                            reason="EXTERNAL_CLOSE"
                        )
                        continue

                    # Reconstruct Trade object
                    from strategy.signals import SignalType
                    trade = Trade(
                        trade_id=trade_dict['trade_id'],
                        symbol=trade_dict['symbol'],
                        signal_type=SignalType(trade_dict['signal_type']),
                        entry_price=trade_dict['entry_price'],
                        size=trade_dict.get('size', position.size),
                        stop_loss=trade_dict.get('stop_loss', 0),
                        take_profit=trade_dict.get('take_profit', 0),
                        status=TradeStatus(trade_dict['status']),
                        entry_order_id=trade_dict.get('entry_order_id'),
                        sl_order_id=trade_dict.get('sl_order_id'),
                        tp_order_id=trade_dict.get('tp_order_id'),
                        opened_at=datetime.fromisoformat(trade_dict['opened_at']) if trade_dict.get('opened_at') else None,
                        metadata=trade_dict.get('metadata', {})
                    )

                    # Add to active trades
                    self.trade_manager._active_trades[trade.trade_id] = trade
                    logger.info(f"Recovered trade {trade.trade_id}: {trade.symbol} {trade.signal_type.value}")

                except Exception as e:
                    logger.error(f"Failed to recover trade {trade_dict.get('trade_id')}: {e}")

            logger.info(f"Trade recovery complete. Active trades: {len(self.trade_manager._active_trades)}")

        except Exception as e:
            logger.error(f"Error during trade recovery: {e}")

    async def run_once(self) -> List[Signal]:
        """
        Run one iteration of signal checking.

        Returns:
            List of detected signals
        """
        if self.state.bot_state != BotState.RUNNING:
            return []

        signals = []

        # V6 multi-strategy mode
        if self.use_v6_strategies and self.v6_processor:
            for asset in self.config.trading.assets:
                try:
                    v6_result = await self._check_asset_v6(asset)
                    if v6_result and v6_result.has_signal:
                        # Use the best signal from V6 result
                        best = v6_result.best_signal
                        if best:
                            signals.append(best.signal)
                            logger.info(
                                f"V6 signal: {best.signal.signal_type.value} {asset} "
                                f"via {best.strategy_name} ({best.timeframe})"
                            )
                except Exception as e:
                    logger.error(f"Error checking {asset} (V6): {e}")
        else:
            # Legacy single-strategy mode
            for asset in self.config.trading.assets:
                try:
                    signal = await self._check_asset(asset)
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    logger.error(f"Error checking {asset}: {e}")

        return signals

    async def run_loop(self, interval_seconds: int = 60) -> None:
        """
        Run continuous trading loop.

        Args:
            interval_seconds: Seconds between checks
        """
        logger.info(f"Starting trading loop (interval: {interval_seconds}s)")

        # Reconcile positions on startup
        await self._reconcile_positions()

        # Track last reconciliation time
        last_reconcile = datetime.utcnow()
        reconcile_interval = 300  # 5 minutes

        # Track last exit check
        # BACKTEST-ALIGNED: Check only at bar closes (5 minute interval = smallest TF)
        # Previous: 30 seconds (intra-bar checks not aligned with backtest)
        # Now: 300 seconds matches 5m bar closes
        last_exit_check = datetime.utcnow()
        exit_check_interval = 300  # 5 minutes - aligned with smallest bar timeframe

        while self._running:
            try:
                now = datetime.utcnow()

                # Check exit conditions more frequently than signals
                if (now - last_exit_check).total_seconds() >= exit_check_interval:
                    await self._monitor_active_trades()
                    last_exit_check = now

                # Periodic position reconciliation
                if (now - last_reconcile).total_seconds() >= reconcile_interval:
                    await self._reconcile_positions()
                    last_reconcile = now

                # Check for daily summary generation (at midnight UTC)
                if self.daily_summary.should_generate(now):
                    await self._generate_daily_summary()

                # Check for new signals
                signals = await self.run_once()

                for signal in signals:
                    await self._handle_signal(signal)

                # Update heartbeat and log scan status
                active_count = len(self.trade_manager._active_trades) if self.trade_manager else 0
                self._update_heartbeat(active_trades=active_count, signals_found=len(signals))
                logger.info(f"Scan complete: {len(signals)} signals, {active_count} active trades")

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await self._notify("error", {"message": str(e)})

            await asyncio.sleep(min(interval_seconds, exit_check_interval))

    async def _monitor_active_trades(self) -> None:
        """
        Monitor active trades for exit conditions.

        Checks:
        1. SL/TP orders filled on the exchange
        2. Manual SL/TP price enforcement (backup)
        3. Oscillator exit signals (opposite WaveTrend signal)
        """
        if not self.trade_manager or not self.trade_manager.active_trades:
            return

        # active_trades is a Dict[str, Trade], iterate over values
        for trade in list(self.trade_manager.active_trades.values()):
            try:
                symbol = trade.symbol

                # Check if SL order was filled (order not in openOrders = filled)
                if trade.sl_order_id:
                    try:
                        sl_order = await self.exchange.get_order(trade.sl_order_id, symbol)
                        if sl_order and sl_order.status == OrderStatus.FILLED:
                            logger.info(f"SL hit for trade {trade.trade_id} on {symbol}")
                            exit_price = sl_order.avg_fill_price or trade.stop_loss
                            await self._close_trade(trade, exit_price, "SL_HIT")
                            continue
                    except Exception as e:
                        logger.debug(f"Error checking SL order: {e}")

                # Check if TP order was filled
                if trade.tp_order_id:
                    try:
                        tp_order = await self.exchange.get_order(trade.tp_order_id, symbol)
                        if tp_order and tp_order.status == OrderStatus.FILLED:
                            logger.info(f"TP hit for trade {trade.trade_id} on {symbol}")
                            exit_price = tp_order.avg_fill_price or trade.take_profit
                            await self._close_trade(trade, exit_price, "TP_HIT")
                            continue
                    except Exception as e:
                        logger.debug(f"Error checking TP order: {e}")

                # Get current price for SL/TP and oscillator checks
                current_price = None
                try:
                    ticker = await self.exchange.get_ticker(symbol)
                    if ticker:
                        current_price = ticker.get('mid', 0) or ticker.get('last', 0)
                except Exception as e:
                    logger.debug(f"Error getting ticker for {symbol}: {e}")

                if current_price and current_price > 0:
                    # Check SL - manual enforcement
                    if trade.is_long and current_price <= trade.stop_loss:
                        logger.warning(f"Manual SL trigger for {trade.trade_id} at {current_price}")
                        await self._close_trade(trade, current_price, "SL_HIT_MANUAL")
                        continue
                    elif not trade.is_long and current_price >= trade.stop_loss:
                        logger.warning(f"Manual SL trigger for {trade.trade_id} at {current_price}")
                        await self._close_trade(trade, current_price, "SL_HIT_MANUAL")
                        continue

                    # Check TP - manual enforcement
                    if trade.take_profit:
                        if trade.is_long and current_price >= trade.take_profit:
                            logger.warning(f"Manual TP trigger for {trade.trade_id} at {current_price}")
                            await self._close_trade(trade, current_price, "TP_HIT_MANUAL")
                            continue
                        elif not trade.is_long and current_price <= trade.take_profit:
                            logger.warning(f"Manual TP trigger for {trade.trade_id} at {current_price}")
                            await self._close_trade(trade, current_price, "TP_HIT_MANUAL")
                            continue

                # Check for oscillator exit signal (opposite WaveTrend signal)
                # This is the PRIMARY exit method for "full_signal" mode
                if await self._check_oscillator_exit(trade, current_price):
                    continue

            except Exception as e:
                logger.error(f"Error monitoring trade {trade.trade_id}: {e}")

    async def _check_oscillator_exit(self, trade: Trade, current_price: Optional[float]) -> bool:
        """
        Check if there's an opposite WaveTrend signal to exit the trade.

        For "full_signal" exit mode:
        - LONG positions exit on SHORT signal
        - SHORT positions exit on LONG signal

        Args:
            trade: Active trade to check
            current_price: Current price (for exit price if signal detected)

        Returns:
            True if trade was closed due to oscillator signal
        """
        try:
            # Extract asset from symbol (e.g., "BTC" from "BTC-USD-PERP")
            asset = trade.metadata.get('symbol', trade.symbol.split('-')[0].upper())

            # Get the strategy's timeframe from trade metadata
            timeframe = trade.metadata.get('timeframe', self.config.trading.timeframe)

            # Fetch candles for this asset
            candles = await self.exchange.get_candles(
                symbol=trade.symbol,
                timeframe=timeframe,
                limit=100
            )

            if len(candles) < 50:
                return False

            # Convert to DataFrame and Heikin Ashi
            df = self._candles_to_dataframe(candles)
            ha_df = convert_to_heikin_ashi(df)

            # Calculate WaveTrend
            wt_result = self.wavetrend.calculate(ha_df)

            # Get current and previous WaveTrend values
            wt1_current = float(wt_result.wt1.iloc[-1])
            wt2_current = float(wt_result.wt2.iloc[-1])
            wt1_prev = float(wt_result.wt1.iloc[-2])
            wt2_prev = float(wt_result.wt2.iloc[-2])

            # Check for opposite signal
            exit_price = current_price or float(df['close'].iloc[-1])

            if trade.is_long:
                # LONG exit: WT1 crosses below WT2 (bearish cross)
                crossed_down = wt1_prev >= wt2_prev and wt1_current < wt2_current
                if crossed_down:
                    logger.info(
                        f"Oscillator EXIT signal for LONG {trade.trade_id}: "
                        f"WT1 crossed below WT2 ({wt1_current:.1f} < {wt2_current:.1f})"
                    )
                    await self._close_trade(trade, exit_price, "OSCILLATOR_EXIT")
                    return True
            else:
                # SHORT exit: WT1 crosses above WT2 (bullish cross)
                crossed_up = wt1_prev <= wt2_prev and wt1_current > wt2_current
                if crossed_up:
                    logger.info(
                        f"Oscillator EXIT signal for SHORT {trade.trade_id}: "
                        f"WT1 crossed above WT2 ({wt1_current:.1f} > {wt2_current:.1f})"
                    )
                    await self._close_trade(trade, exit_price, "OSCILLATOR_EXIT")
                    return True

            return False

        except Exception as e:
            logger.debug(f"Error checking oscillator exit for {trade.trade_id}: {e}")
            return False

    async def _close_trade(self, trade: Trade, exit_price: float, reason: str) -> None:
        """
        Close a trade and update state.

        Args:
            trade: Trade to close
            exit_price: Exit price achieved
            reason: Reason for closing (SL_HIT, TP_HIT, etc.)
        """
        from strategy.trade_manager import TradeStatus

        try:
            # Cancel any remaining SL/TP orders
            if trade.sl_order_id:
                try:
                    await self.exchange.cancel_order(trade.sl_order_id, trade.symbol)
                except Exception:
                    pass  # Order may already be filled/cancelled

            if trade.tp_order_id:
                try:
                    await self.exchange.cancel_order(trade.tp_order_id, trade.symbol)
                except Exception:
                    pass

            # Close position on exchange if still open
            position = await self.exchange.get_position(trade.symbol)
            if position and abs(position.size) > 0:
                await self.exchange.close_position(trade.symbol)

            # Calculate PnL
            if trade.is_long:
                pnl = (exit_price - trade.entry_price) * trade.size
            else:
                pnl = (trade.entry_price - exit_price) * trade.size

            pnl_percent = (pnl / (trade.entry_price * trade.size)) * 100

            # Update trade record
            trade.exit_price = exit_price
            trade.closed_at = datetime.utcnow()
            trade.pnl = pnl
            trade.pnl_percent = pnl_percent
            trade.status = TradeStatus.CLOSED
            trade.metadata['exit_reason'] = reason

            # Remove from active trades (dict uses trade_id as key)
            if self.trade_manager and trade.trade_id in self.trade_manager._active_trades:
                del self.trade_manager._active_trades[trade.trade_id]
                self.trade_manager._trade_history.append(trade)

            # Persist trade closure to database
            if self.trade_store:
                self.trade_store.mark_trade_closed(
                    trade_id=trade.trade_id,
                    exit_price=exit_price,
                    pnl=pnl,
                    pnl_percent=pnl_percent,
                    reason=reason
                )

            logger.info(f"Trade {trade.trade_id} closed: {reason} @ {exit_price}, PnL: ${pnl:.2f}")

            # Record trade closed for daily summary
            self.daily_summary.record_trade_closed(
                symbol=trade.symbol,
                direction='LONG' if trade.is_long else 'SHORT',
                entry_price=trade.entry_price,
                exit_price=exit_price,
                pnl=pnl,
                exit_reason=reason
            )

            await self._notify("trade_closed", {
                "trade_id": trade.trade_id,
                "symbol": trade.symbol,
                "direction": "long" if trade.is_long else "short",
                "entry_price": trade.entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_percent": pnl_percent,
                "reason": reason
            })

        except Exception as e:
            logger.error(f"Error closing trade {trade.trade_id}: {e}")
            await self._notify("error", {"message": f"Failed to close trade: {e}"})

    async def _generate_daily_summary(self) -> None:
        """
        Generate and log daily performance summary.

        Called at midnight UTC to summarize the previous day's trading.
        """
        try:
            # Get current balance
            balance = await self.exchange.get_balance()
            current_balance = balance.total_balance if balance else 0.0

            # Get active positions for the summary
            active_positions = []
            if self.trade_manager:
                for trade in self.trade_manager._active_trades.values():
                    # Get current price for unrealized PnL
                    try:
                        ticker = await self.exchange.get_ticker(trade.symbol)
                        current_price = ticker.get('price', trade.entry_price)
                        if trade.is_long:
                            unrealized = (current_price - trade.entry_price) * trade.size
                        else:
                            unrealized = (trade.entry_price - current_price) * trade.size
                    except Exception:
                        unrealized = 0.0

                    active_positions.append({
                        'symbol': trade.symbol,
                        'direction': 'LONG' if trade.is_long else 'SHORT',
                        'entry_price': trade.entry_price,
                        'unrealized_pnl': unrealized
                    })

            # Generate summary
            summary = self.daily_summary.generate(
                current_balance=current_balance,
                active_positions=active_positions
            )

            # Log the formatted summary
            logger.info(f"\n{summary.format()}")

            # Send Discord notification
            await self._notify("daily_summary", summary.to_discord_embed())

        except Exception as e:
            logger.error(f"Error generating daily summary: {e}")

    async def _reconcile_positions(self) -> None:
        """
        Reconcile TradeManager state with actual exchange positions.

        This handles:
        1. Orphaned positions (positions on exchange without matching trade)
        2. Stale trades (trades in memory but position closed externally)
        3. Sync position sizes if they differ
        """
        from strategy.trade_manager import Trade, TradeStatus
        from strategy.signals import SignalType

        if not self.exchange or not self.trade_manager:
            return

        logger.debug("Reconciling positions with exchange...")

        try:
            # Get all positions from exchange
            exchange_positions = await self.exchange.get_positions()

            # Build lookup of exchange positions by symbol
            exchange_pos_by_symbol: Dict[str, Position] = {}
            for pos in exchange_positions:
                if pos and abs(pos.size) > 0:
                    exchange_pos_by_symbol[pos.symbol] = pos

            # Check for stale trades (trade exists but no position on exchange)
            # active_trades is Dict[str, Trade]
            for trade_id, trade in list(self.trade_manager._active_trades.items()):
                symbol = trade.symbol
                if symbol not in exchange_pos_by_symbol:
                    logger.warning(f"Trade {trade.trade_id} has no matching position on exchange - marking as closed")
                    trade.status = TradeStatus.CLOSED
                    trade.metadata['exit_reason'] = 'EXTERNAL_CLOSE'
                    trade.closed_at = datetime.utcnow()

                    # Move to history
                    del self.trade_manager._active_trades[trade_id]
                    self.trade_manager._trade_history.append(trade)

                    await self._notify("trade_closed", {
                        "trade_id": trade.trade_id,
                        "symbol": symbol,
                        "reason": "EXTERNAL_CLOSE"
                    })

            # Check for orphaned positions (position exists but no trade record)
            tracked_symbols = {t.symbol for t in self.trade_manager._active_trades.values()}
            for symbol, pos in exchange_pos_by_symbol.items():
                if symbol not in tracked_symbols:
                    logger.warning(f"Orphaned position found: {symbol} size={pos.size}")

                    # Create recovery trade to track this position
                    is_long = pos.size > 0
                    entry = pos.entry_price

                    # CRITICAL FIX: Set reasonable default SL/TP for recovered trades
                    # Uses 5% SL and 10% TP as safety defaults
                    # These can be adjusted manually but prevent unprotected positions
                    DEFAULT_SL_PCT = 0.05  # 5% stop loss
                    DEFAULT_TP_PCT = 0.10  # 10% take profit

                    if is_long:
                        default_sl = entry * (1 - DEFAULT_SL_PCT)
                        default_tp = entry * (1 + DEFAULT_TP_PCT)
                    else:
                        default_sl = entry * (1 + DEFAULT_SL_PCT)
                        default_tp = entry * (1 - DEFAULT_TP_PCT)

                    recovery_trade = Trade(
                        trade_id=f"recovery_{symbol}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                        symbol=symbol,
                        signal_type=SignalType.LONG if is_long else SignalType.SHORT,
                        entry_price=entry,
                        size=abs(pos.size),
                        stop_loss=default_sl,  # 5% default SL for safety
                        take_profit=default_tp,  # 10% default TP
                        status=TradeStatus.OPEN,
                        opened_at=datetime.utcnow(),
                        metadata={
                            'recovered': True,
                            'skip_sl_tp_orders': True,  # Don't place SL/TP orders (already on exchange or none)
                            'original_position': pos.to_dict()
                        }
                    )

                    self.trade_manager._active_trades[recovery_trade.trade_id] = recovery_trade
                    logger.info(f"Created recovery trade for orphaned position: {recovery_trade.trade_id}")

                    await self._notify("position_recovered", {
                        "trade_id": recovery_trade.trade_id,
                        "symbol": symbol,
                        "size": recovery_trade.size,
                        "direction": "long" if is_long else "short",
                        "entry_price": recovery_trade.entry_price
                    })

            logger.debug(f"Reconciliation complete. Active trades: {len(self.trade_manager._active_trades)}")

        except Exception as e:
            logger.error(f"Error reconciling positions: {e}")
            await self._notify("error", {"message": f"Position reconciliation failed: {e}"})

    async def _check_asset(self, asset: str) -> Optional[Signal]:
        """
        Check a single asset for signals.

        Args:
            asset: Asset symbol

        Returns:
            Signal if detected, None otherwise
        """
        if self.use_multi_timeframe:
            return await self._check_asset_mtf(asset)
        else:
            return await self._check_asset_single_tf(asset)

    async def _check_asset_single_tf(self, asset: str) -> Optional[Signal]:
        """
        Check asset using single timeframe mode.

        Args:
            asset: Asset symbol

        Returns:
            Signal if detected, None otherwise
        """
        symbol = self.exchange.format_symbol(asset)

        # Fetch candles
        candles = await self.exchange.get_candles(
            symbol=symbol,
            timeframe=self.config.trading.timeframe,
            limit=200  # Need enough for indicator calculation
        )

        if len(candles) < 100:
            logger.warning(f"Not enough candles for {asset}")
            return None

        # Convert to DataFrame
        df = self._candles_to_dataframe(candles)

        # Convert to Heikin Ashi
        ha_df = convert_to_heikin_ashi(df)

        # Calculate indicators
        wt_result = self.wavetrend.calculate(ha_df)
        mf_result = self.money_flow.calculate(ha_df)

        # Get signal detector for this asset
        detector = self.signal_detectors.get(asset)
        if not detector:
            return None

        # Process latest bar
        signal = detector.process_bar(
            timestamp=candles[-1].timestamp,
            close_price=candles[-1].close,
            wt_result=wt_result,
            mf_result=mf_result
        )

        if signal:
            signal.metadata['symbol'] = asset
            signal.metadata['timeframe'] = self.config.trading.timeframe
            self.state.record_signal(signal)

            logger.info(
                f"Signal detected: {signal.signal_type.value} {asset} @ {signal.entry_price}"
            )
            await self._notify("signal_detected", signal.to_dict())

        return signal

    async def _check_asset_v6(self, asset: str) -> Optional[V6ScanResult]:
        """
        Check asset using V6 multi-strategy mode.

        Scans all enabled strategies for this asset across different
        timeframes, applying time filters and VWAP confirmation as
        configured per strategy.

        Args:
            asset: Asset symbol

        Returns:
            V6ScanResult with any detected signals
        """
        if not self.v6_processor:
            return None

        symbol = self.exchange.format_symbol(asset)

        # Get required timeframes for this asset's strategies
        required_tfs = self.v6_processor.get_required_timeframes(asset)
        if not required_tfs:
            return None

        # Fetch candles for all required timeframes
        try:
            candles_by_tf_raw = await self.exchange.get_candles_multiple(
                symbol=symbol,
                timeframes=required_tfs,
                limit=200
            )
        except Exception as e:
            logger.error(f"Failed to fetch candles for {asset}: {e}")
            return None

        # Convert candles to DataFrames
        candles_by_tf: Dict[str, pd.DataFrame] = {}
        for tf in required_tfs:
            candles = candles_by_tf_raw.get(tf, [])
            if candles and len(candles) >= 50:
                candles_by_tf[tf] = self._candles_to_dataframe(candles)
            else:
                logger.debug(f"Not enough {tf} candles for {asset}: {len(candles) if candles else 0}")

        if not candles_by_tf:
            logger.warning(f"No valid candle data for {asset}")
            return None

        # Process through V6 processor
        result = await self.v6_processor.process_asset(asset, candles_by_tf)

        # Log scan results
        if result.strategies_scanned > 0:
            logger.debug(
                f"V6 scan {asset}: {result.strategies_scanned} strategies, "
                f"{result.strategies_with_signal} signals"
            )

        # Notify if signal detected
        if result.has_signal and result.best_signal:
            best = result.best_signal
            best.signal.metadata['symbol'] = asset
            self.state.record_signal(best.signal)
            await self._notify("signal_detected", {
                **best.signal.to_dict(),
                'v6_strategy': best.strategy_name,
                'v6_timeframe': best.timeframe,
                'vwap_confirmed': best.vwap_confirmed,
            })

        return result

    async def _check_asset_mtf(self, asset: str) -> Optional[Signal]:
        """
        Check asset using multi-timeframe mode.

        Args:
            asset: Asset symbol

        Returns:
            Signal if detected and aligned with HTF bias, None otherwise
        """
        symbol = self.exchange.format_symbol(asset)
        coordinator = self.mtf_coordinators.get(asset)

        if not coordinator:
            logger.warning(f"No MTF coordinator for {asset}")
            return None

        tf_config = self.config.trading.timeframes
        all_timeframes = list(set(tf_config.entry + tf_config.bias))

        # Fetch candles for all timeframes in parallel
        candles_by_tf = await self.exchange.get_candles_multiple(
            symbol=symbol,
            timeframes=all_timeframes,
            limit=200
        )

        # Convert candles to DataFrames
        entry_candles: Dict[str, pd.DataFrame] = {}
        bias_candles: Dict[str, pd.DataFrame] = {}

        for tf in tf_config.entry:
            candles = candles_by_tf.get(tf, [])
            if candles and len(candles) >= 50:
                entry_candles[tf] = self._candles_to_dataframe(candles)
            else:
                logger.debug(f"Not enough candles for {asset} {tf}")

        for tf in tf_config.bias:
            candles = candles_by_tf.get(tf, [])
            if candles and len(candles) >= 50:
                bias_candles[tf] = self._candles_to_dataframe(candles)
            else:
                logger.debug(f"Not enough candles for {asset} {tf}")

        if not entry_candles:
            logger.warning(f"No entry timeframe data for {asset}")
            return None

        # Process through MTF coordinator
        mtf_signal = coordinator.process_candles(entry_candles, bias_candles)

        if mtf_signal and mtf_signal.primary_signal:
            signal = mtf_signal.primary_signal
            signal.metadata['symbol'] = asset
            signal.metadata['mtf_aligned'] = mtf_signal.is_aligned
            signal.metadata['bias_direction'] = mtf_signal.bias_result.direction.value
            signal.metadata['alignment_percent'] = mtf_signal.alignment_percent
            signal.metadata['entry_timeframe'] = mtf_signal.metadata.get('entry_tf', '')

            self.state.record_signal(signal)

            logger.info(
                f"MTF Signal: {signal.signal_type.value} {asset} @ {signal.entry_price} "
                f"[bias={mtf_signal.bias_result.direction.value}, "
                f"aligned={mtf_signal.is_aligned}, "
                f"conf={mtf_signal.overall_confidence:.2f}]"
            )
            await self._notify("signal_detected", {
                **signal.to_dict(),
                "mtf_aligned": mtf_signal.is_aligned,
                "bias_direction": mtf_signal.bias_result.direction.value,
                "alignment_percent": mtf_signal.alignment_percent,
            })

            return signal

        return None

    async def _handle_signal(self, signal: Signal) -> Optional[Trade]:
        """
        Handle a detected signal.

        Args:
            signal: Signal to handle

        Returns:
            Trade if executed, None otherwise
        """
        asset = signal.metadata.get('symbol', 'BTC')
        symbol = self.exchange.format_symbol(asset)

        # Get fresh candles for risk calculation
        candles = await self.exchange.get_candles(
            symbol=symbol,
            timeframe=self.config.trading.timeframe,
            limit=50
        )
        df = self._candles_to_dataframe(candles)

        # Execute the signal
        try:
            trade = await self.trade_manager.execute_signal(
                signal=signal,
                df=df,
                risk_percent=self.config.trading.risk_percent,
                risk_reward=self.config.take_profit.risk_reward
            )

            if trade:
                self.state.total_trades += 1

                # Record trade opened for daily summary
                self.daily_summary.record_trade_opened(
                    symbol=trade.symbol,
                    direction='LONG' if trade.is_long else 'SHORT',
                    entry_price=trade.entry_price
                )

                # Persist trade to database
                if self.trade_store:
                    trade_dict = trade.to_dict()
                    trade_dict['size'] = trade.size
                    trade_dict['stop_loss'] = trade.stop_loss
                    trade_dict['take_profit'] = trade.take_profit
                    trade_dict['entry_order_id'] = trade.entry_order_id
                    trade_dict['sl_order_id'] = trade.sl_order_id
                    trade_dict['tp_order_id'] = trade.tp_order_id
                    trade_dict['metadata'] = trade.metadata
                    self.trade_store.save_trade(trade_dict)

                await self._notify("trade_opened", trade.to_dict())

            return trade

        except Exception as e:
            logger.error(f"Failed to execute signal: {e}")
            await self._notify("error", {"message": f"Trade execution failed: {e}"})
            return None

    def _candles_to_dataframe(self, candles: List[Candle]) -> pd.DataFrame:
        """Convert candles to DataFrame."""
        data = {
            'timestamp': [c.timestamp for c in candles],
            'open': [c.open for c in candles],
            'high': [c.high for c in candles],
            'low': [c.low for c in candles],
            'close': [c.close for c in candles],
            'volume': [c.volume for c in candles],
        }
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

    def get_status(self) -> Dict[str, Any]:
        """Get current bot status."""
        status = {
            "bot_state": self.state.bot_state.value,
            "started_at": self.state.started_at.isoformat() if self.state.started_at else None,
            "exchange": self.exchange.name if self.exchange else None,
            "assets": self.config.trading.assets,
            "total_signals": self.state.total_signals,
            "total_trades": self.state.total_trades,
            "active_trades": len(self.trade_manager.active_trades) if self.trade_manager else 0,
            "last_error": self.state.last_error,
            "multi_timeframe_mode": self.use_multi_timeframe,
        }

        if self.use_multi_timeframe:
            tf_config = self.config.trading.timeframes
            status["entry_timeframes"] = tf_config.entry
            status["bias_timeframes"] = tf_config.bias
            status["require_bias_alignment"] = tf_config.require_bias_alignment

        return status
