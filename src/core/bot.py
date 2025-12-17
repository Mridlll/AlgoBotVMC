"""Main VMC Trading Bot orchestrator."""

import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd

from core.state import TradingState, BotState, AssetState
from config import Config, ExchangeName
from indicators import HeikinAshi, WaveTrend, MoneyFlow
from indicators.heikin_ashi import convert_to_heikin_ashi
from exchanges.base import BaseExchange, Candle
from exchanges.hyperliquid import HyperliquidExchange
from exchanges.bitunix import BitunixExchange
from strategy.signals import SignalDetector, Signal
from strategy.risk import RiskManager, StopLossMethod, TakeProfitMethod
from strategy.trade_manager import TradeManager, Trade
from strategy.multi_timeframe import MultiTimeframeCoordinator, MultiTimeframeSignal
from utils.logger import get_logger, setup_logger

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

        # Multi-timeframe coordinators (per asset)
        self.mtf_coordinators: Dict[str, MultiTimeframeCoordinator] = {}
        self.use_multi_timeframe = config.trading.use_multi_timeframe

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

        # Running flag
        self._running = False

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

            # Initialize signal detectors for each asset
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

        if self.exchange:
            await self.exchange.disconnect()

        logger.info("VMC Trading Bot stopped")
        await self._notify("bot_stopped", {})

    async def run_once(self) -> List[Signal]:
        """
        Run one iteration of signal checking.

        Returns:
            List of detected signals
        """
        if self.state.bot_state != BotState.RUNNING:
            return []

        signals = []

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

        while self._running:
            try:
                signals = await self.run_once()

                for signal in signals:
                    await self._handle_signal(signal)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await self._notify("error", {"message": str(e)})

            await asyncio.sleep(interval_seconds)

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
