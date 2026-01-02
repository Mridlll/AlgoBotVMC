"""V6 Multi-Strategy Signal Processor.

Handles signal detection across all 15 V6 strategies:
- 5 strategies per asset (BTC, ETH, SOL)
- Multiple timeframes (5m, 15m, 30m, 1h, 4h)
- Time filters (NY hours, weekends, all hours)
- Signal modes (simple, enhanced with 53/60/70 anchor levels)
- Per-asset VWAP confirmation
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
from loguru import logger

from config.config import (
    Config,
    StrategyInstanceConfig,
    TimeFilterType,
)
from indicators.wavetrend import WaveTrend, WaveTrendResult
from indicators.money_flow import MoneyFlow, MoneyFlowResult
from indicators.heikin_ashi import convert_to_heikin_ashi
from indicators.vwap import VWAPCalculator
from strategy.signals import (
    SignalType, Signal, AnchorWave, TriggerWave,
    SignalDetector, SimpleSignalDetector  # Full state machine detectors
)
from utils.time_filter import should_trade_now
from typing import Union


@dataclass
class StrategySignal:
    """Signal from a specific V6 strategy instance."""
    strategy_name: str
    strategy_config: StrategyInstanceConfig
    signal: Signal
    timeframe: str
    signal_mode: str
    anchor_level: int
    vwap_confirmed: bool
    time_filter_passed: bool


@dataclass
class V6ScanResult:
    """Result of scanning all V6 strategies for an asset."""
    asset: str
    timestamp: datetime
    signals: List[StrategySignal] = field(default_factory=list)
    strategies_scanned: int = 0
    strategies_with_signal: int = 0
    strategies_filtered_by_time: int = 0
    strategies_filtered_by_vwap: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def has_signal(self) -> bool:
        return len(self.signals) > 0

    @property
    def best_signal(self) -> Optional[StrategySignal]:
        """Get the best signal (highest Sharpe from backtest metrics)."""
        if not self.signals:
            return None
        # Sort by sharpe ratio from strategy metrics if available
        signals_with_sharpe = [
            (s, s.strategy_config.metrics.sharpe if s.strategy_config.metrics else 0)
            for s in self.signals
        ]
        signals_with_sharpe.sort(key=lambda x: x[1] or 0, reverse=True)
        return signals_with_sharpe[0][0]


class V6SignalProcessor:
    """Processes signals for all V6 strategy configurations.

    This processor handles the multi-strategy approach of V6:
    - Scans multiple timeframes per asset
    - Applies time filters (NY hours, weekends)
    - Uses different signal modes (simple at 53, enhanced at 60/70)
    - Applies per-asset VWAP confirmation
    """

    def __init__(self, config: Config):
        """Initialize the V6 processor.

        Args:
            config: Bot configuration with V6 strategies
        """
        self.config = config
        self.wavetrend = WaveTrend(
            channel_len=config.indicators.wt_channel_len,
            average_len=config.indicators.wt_average_len,
            ma_len=config.indicators.wt_ma_len,
        )
        self.money_flow = MoneyFlow(
            period=config.indicators.mfi_period,
            multiplier=config.indicators.mfi_multiplier,
        )
        self.vwap_calc = VWAPCalculator()  # Real VWAP for entry confirmation

        # Cache for candle data by timeframe
        self._candle_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl_seconds = 30  # Refresh cache every 30s

        # Track signal state per strategy to avoid duplicates
        self._last_signal_bar: Dict[str, int] = {}

        # Per-strategy signal detectors (full state machine to match backtest)
        # CRITICAL: This aligns live bot with backtest signal detection
        self._signal_detectors: Dict[str, Union[SignalDetector, SimpleSignalDetector]] = {}
        self._init_signal_detectors()

    def _init_signal_detectors(self) -> None:
        """Initialize per-strategy signal detectors.

        Creates either SimpleSignalDetector or SignalDetector based on signal_mode:
        - simple: Uses SimpleSignalDetector with ±53 levels
        - enhanced: Uses full SignalDetector with 4-step state machine

        This matches the backtest engine behavior exactly.
        """
        strategies = self.config.get_enabled_strategies()

        for strat_name, strat_config in strategies.items():
            if strat_config.signal_mode == "simple":
                # Simple mode: single-bar cross detection with ±53 levels
                self._signal_detectors[strat_name] = SimpleSignalDetector(
                    oversold_level=-53,
                    overbought_level=53,
                    timeframe=strat_config.timeframe
                )
            else:
                # Enhanced mode: full 4-step state machine
                # Uses anchor_level (60 or 70) for entry zones
                anchor_level = strat_config.anchor_level or 60
                self._signal_detectors[strat_name] = SignalDetector(
                    anchor_level_long=-anchor_level,
                    anchor_level_short=anchor_level,
                    trigger_lookback=20,  # Matches backtest default
                    mfi_lookback=3,       # Matches backtest default
                    timeframe=strat_config.timeframe
                )

        logger.info(f"Initialized {len(self._signal_detectors)} signal detectors (backtest-aligned)")

    def get_required_timeframes(self, asset: Optional[str] = None) -> List[str]:
        """Get list of unique timeframes needed for V6 strategies.

        Args:
            asset: Optional asset to filter by

        Returns:
            List of unique timeframes
        """
        timeframes = set()
        strategies = self.config.get_strategies_for_asset(asset) if asset else self.config.get_enabled_strategies()

        for strat in strategies.values():
            if strat.enabled:
                timeframes.add(strat.timeframe)

        return sorted(list(timeframes))

    async def process_asset(
        self,
        asset: str,
        candles_by_tf: Dict[str, pd.DataFrame],
        current_time: Optional[datetime] = None,
    ) -> V6ScanResult:
        """Process all V6 strategies for an asset.

        Args:
            asset: Asset symbol (BTC, ETH, SOL)
            candles_by_tf: Dict of timeframe -> DataFrame with OHLCV data
            current_time: Current time for time filter (default: now UTC)

        Returns:
            V6ScanResult with any detected signals
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        result = V6ScanResult(asset=asset, timestamp=current_time)
        strategies = self.config.get_strategies_for_asset(asset)

        if not strategies:
            logger.debug(f"No V6 strategies configured for {asset}")
            return result

        for strat_name, strat_config in strategies.items():
            if not strat_config.enabled:
                continue

            result.strategies_scanned += 1

            try:
                signal = await self._process_strategy(
                    strat_name=strat_name,
                    strat_config=strat_config,
                    candles_by_tf=candles_by_tf,
                    current_time=current_time,
                )

                if signal:
                    result.signals.append(signal)
                    result.strategies_with_signal += 1

            except Exception as e:
                error_msg = f"Error processing {strat_name}: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)

        return result

    async def _process_strategy(
        self,
        strat_name: str,
        strat_config: StrategyInstanceConfig,
        candles_by_tf: Dict[str, pd.DataFrame],
        current_time: datetime,
    ) -> Optional[StrategySignal]:
        """Process a single V6 strategy using full state machine detection.

        BACKTEST-ALIGNED: Uses SignalDetector (4-step state machine for enhanced)
        or SimpleSignalDetector (single-bar for simple) to match backtest exactly.

        Args:
            strat_name: Strategy name/ID
            strat_config: Strategy configuration
            candles_by_tf: Candle data by timeframe
            current_time: Current time

        Returns:
            StrategySignal if detected, None otherwise
        """
        # Check time filter first (skip early if not in trading window)
        time_filter_passed = should_trade_now(strat_config.time_filter, current_time)
        if not time_filter_passed:
            logger.debug(f"{strat_name}: Skipped - time filter ({strat_config.time_filter.mode})")
            return None

        # Get candles for this strategy's timeframe
        tf = strat_config.timeframe
        df = candles_by_tf.get(tf)
        if df is None or len(df) < 100:
            logger.debug(f"{strat_name}: Not enough {tf} candles")
            return None

        # Convert to Heikin Ashi
        ha_df = convert_to_heikin_ashi(df)

        # Calculate WaveTrend indicators
        wt_result = self.wavetrend.calculate(ha_df)
        mf_result = self.money_flow.calculate(ha_df)

        # Get the appropriate signal detector for this strategy
        detector = self._signal_detectors.get(strat_name)
        if detector is None:
            logger.error(f"{strat_name}: No signal detector found - reinitializing")
            self._init_signal_detectors()
            detector = self._signal_detectors.get(strat_name)
            if detector is None:
                return None

        # Extract current values for signal detection
        bar_index = len(df) - 1
        current_price = float(df['close'].iloc[-1])
        current_ts = df.index[-1] if hasattr(df.index[-1], 'timestamp') else current_time

        # Detect signal using full state machine (BACKTEST-ALIGNED)
        # Different detectors have different interfaces
        signal = None
        if isinstance(detector, SimpleSignalDetector):
            # SimpleSignalDetector: single-bar cross detection
            wt1_val = float(wt_result.wt1.iloc[-1])
            wt2_val = float(wt_result.wt2.iloc[-1])
            mfi_val = float(mf_result.mfi.iloc[-1]) if hasattr(mf_result, 'mfi') else 0.0

            signal = detector.process_bar(
                timestamp=current_ts,
                close_price=current_price,
                wt1=wt1_val,
                wt2=wt2_val,
                mfi=mfi_val,
                vwap=0.0  # VWAP check done separately below
            )
        else:
            # SignalDetector: full 4-step state machine
            # Processes through: ANCHOR -> TRIGGER -> MFI -> MOMENTUM_CROSS
            signal = detector.process_bar(
                timestamp=current_ts,
                close_price=current_price,
                wt_result=wt_result,
                mf_result=mf_result,
                bar_idx=bar_index
            )

        if signal is None:
            return None

        signal_type = signal.signal_type

        # Check direction filter
        if not self._passes_direction_filter(signal_type, strat_config.direction_filter):
            logger.debug(f"{strat_name}: Signal {signal_type} filtered by direction {strat_config.direction_filter}")
            return None

        # VWAP confirmation (if enabled for this strategy)
        # Calculate real VWAP from raw OHLCV (not Heikin Ashi)
        vwap_confirmed = True
        real_vwap = None
        if strat_config.use_vwap_confirmation:
            try:
                vwap_result = self.vwap_calc.calculate(df)
                real_vwap = float(vwap_result.vwap.iloc[-1])
            except Exception as e:
                logger.warning(f"{strat_name}: Error calculating VWAP: {e}")
                real_vwap = None

            vwap_confirmed = self._check_vwap_confirmation(
                signal_type=signal_type,
                current_price=current_price,
                real_vwap=real_vwap,
            )
            if not vwap_confirmed:
                vwap_str = f"{real_vwap:.2f}" if real_vwap else "N/A"
                logger.debug(f"{strat_name}: Signal {signal_type} rejected by VWAP (price={current_price:.2f}, vwap={vwap_str})")
                return None

        # Avoid duplicate signals on same bar
        last_bar = self._last_signal_bar.get(strat_name, -1)
        if bar_index == last_bar:
            return None
        self._last_signal_bar[strat_name] = bar_index

        # Update signal metadata with strategy context
        signal.metadata.update({
            'strategy': strat_name,
            'asset': strat_config.asset,
            'timeframe': tf,
            'signal_mode': strat_config.signal_mode,
            'anchor_level': strat_config.anchor_level,
            'vwap_confirmed': vwap_confirmed,
            'detection_mode': 'state_machine',  # Mark as using full detector
        })

        logger.info(f"{strat_name}: Signal detected via state machine - {signal_type.value} @ {current_price:.2f}")

        return StrategySignal(
            strategy_name=strat_name,
            strategy_config=strat_config,
            signal=signal,
            timeframe=tf,
            signal_mode=strat_config.signal_mode,
            anchor_level=strat_config.anchor_level,
            vwap_confirmed=vwap_confirmed,
            time_filter_passed=time_filter_passed,
        )

    def _detect_signal(
        self,
        strat_config: StrategyInstanceConfig,
        wt_result: WaveTrendResult,
        mf_result: MoneyFlowResult,
    ) -> Optional[SignalType]:
        """Detect signal based on strategy mode.

        Simple mode: WT1 crosses WT2 while in oversold/overbought zone
        Enhanced mode: 4-step state machine (anchor -> trigger -> MFI -> VWAP)

        For V6, we simplify to check:
        - Simple: WT cross with zone check
        - Enhanced: WT cross with deeper zone (anchor_level instead of 53)
        """
        # Get WaveTrend values
        wt1 = wt_result.wt1
        wt2 = wt_result.wt2

        if hasattr(wt1, 'iloc'):
            wt1_current = float(wt1.iloc[-1])
            wt2_current = float(wt2.iloc[-1])
            wt1_prev = float(wt1.iloc[-2]) if len(wt1) > 1 else wt1_current
            wt2_prev = float(wt2.iloc[-2]) if len(wt2) > 1 else wt2_current
        else:
            wt1_current = float(wt1[-1])
            wt2_current = float(wt2[-1])
            wt1_prev = float(wt1[-2]) if len(wt1) > 1 else wt1_current
            wt2_prev = float(wt2[-2]) if len(wt2) > 1 else wt2_current

        # Determine thresholds based on mode
        anchor_level = strat_config.anchor_level
        if strat_config.signal_mode == "simple":
            # Simple mode uses standard 53 levels
            oversold = -53
            overbought = 53
        else:
            # Enhanced mode uses the anchor_level (60 or 70)
            oversold = -anchor_level
            overbought = anchor_level

        # Check for WT1 crossing WT2
        crossed_up = wt1_prev <= wt2_prev and wt1_current > wt2_current
        crossed_down = wt1_prev >= wt2_prev and wt1_current < wt2_current

        # Long signal: WT1 crosses above WT2 while in oversold zone
        if crossed_up and wt2_current < oversold:
            # MFI confirmation for enhanced mode
            if strat_config.signal_mode == "enhanced" and strat_config.use_mfi_confirmation:
                if not self._check_mfi_curving_up(mf_result):
                    return None
            return SignalType.LONG

        # Short signal: WT1 crosses below WT2 while in overbought zone
        if crossed_down and wt2_current > overbought:
            # MFI confirmation for enhanced mode
            if strat_config.signal_mode == "enhanced" and strat_config.use_mfi_confirmation:
                if not self._check_mfi_curving_down(mf_result):
                    return None
            return SignalType.SHORT

        return None

    def _check_mfi_curving_up(self, mf_result: MoneyFlowResult) -> bool:
        """Check if MFI is curving up (bullish confirmation)."""
        if not hasattr(mf_result, 'mfi') or mf_result.mfi is None:
            return True  # Skip check if MFI not available

        mfi = mf_result.mfi
        if hasattr(mfi, 'iloc'):
            if len(mfi) < 3:
                return True
            current = float(mfi.iloc[-1])
            prev1 = float(mfi.iloc[-2])
            prev2 = float(mfi.iloc[-3])
        else:
            if len(mfi) < 3:
                return True
            current = float(mfi[-1])
            prev1 = float(mfi[-2])
            prev2 = float(mfi[-3])

        # MFI curving up: decreasing slope becoming less negative or increasing
        return current > prev1 or (prev1 > prev2 and current >= prev1)

    def _check_mfi_curving_down(self, mf_result: MoneyFlowResult) -> bool:
        """Check if MFI is curving down (bearish confirmation)."""
        if not hasattr(mf_result, 'mfi') or mf_result.mfi is None:
            return True

        mfi = mf_result.mfi
        if hasattr(mfi, 'iloc'):
            if len(mfi) < 3:
                return True
            current = float(mfi.iloc[-1])
            prev1 = float(mfi.iloc[-2])
            prev2 = float(mfi.iloc[-3])
        else:
            if len(mfi) < 3:
                return True
            current = float(mfi[-1])
            prev1 = float(mfi[-2])
            prev2 = float(mfi[-3])

        # MFI curving down
        return current < prev1 or (prev1 < prev2 and current <= prev1)

    def _check_vwap_confirmation(
        self,
        signal_type: SignalType,
        current_price: float,
        real_vwap: Optional[float],
    ) -> bool:
        """Check VWAP confirmation for entry.

        Long: Price should be above VWAP (bullish positioning)
        Short: Price should be below VWAP (bearish positioning)

        This uses REAL VWAP from VWAPCalculator, not the WT momentum difference.

        Args:
            signal_type: Type of signal (LONG or SHORT)
            current_price: Current close price
            real_vwap: Real VWAP value from VWAPCalculator

        Returns:
            True if VWAP confirms the signal direction
        """
        # If VWAP couldn't be calculated, skip the check
        if real_vwap is None:
            return True

        # Long signals: price should be above VWAP
        if signal_type == SignalType.LONG:
            return current_price > real_vwap

        # Short signals: price should be below VWAP
        if signal_type == SignalType.SHORT:
            return current_price < real_vwap

        return True

    def _passes_direction_filter(self, signal_type: SignalType, direction_filter: str) -> bool:
        """Check if signal passes direction filter.

        Args:
            signal_type: Detected signal type
            direction_filter: Filter setting (both, long_only, short_only)

        Returns:
            True if signal passes filter
        """
        if direction_filter == "both":
            return True
        elif direction_filter == "long_only":
            return signal_type == SignalType.LONG
        elif direction_filter == "short_only":
            return signal_type == SignalType.SHORT
        return True

    def get_strategy_summary(self) -> str:
        """Get summary of configured V6 strategies."""
        strategies = self.config.get_enabled_strategies()
        if not strategies:
            return "No V6 strategies configured"

        by_asset: Dict[str, List[str]] = {}
        for name, strat in strategies.items():
            asset = strat.asset
            if asset not in by_asset:
                by_asset[asset] = []
            tf_mode = strat.time_filter.mode.value if strat.time_filter.enabled else "all"
            by_asset[asset].append(f"{strat.timeframe}_{strat.signal_mode}{strat.anchor_level}_{tf_mode}")

        lines = [f"V6 Strategies ({len(strategies)} total):"]
        for asset, strats in sorted(by_asset.items()):
            lines.append(f"  {asset}: {', '.join(strats)}")

        return "\n".join(lines)
