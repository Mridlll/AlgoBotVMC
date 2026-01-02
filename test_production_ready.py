#!/usr/bin/env python3
"""
VMC Trading Bot V6 - Production Readiness Test Suite
=====================================================

Comprehensive tests to verify the bot is ready for client deployment.
Tests cover:
1. Configuration loading and validation
2. Exchange connectivity
3. All 15 strategy configurations
4. Time filter logic
5. Signal detection across all assets
6. Trade execution simulation
"""

import sys
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple
import traceback

sys.path.insert(0, 'src')

# Test results tracking
TESTS_RUN = 0
TESTS_PASSED = 0
TESTS_FAILED = 0
FAILURES: List[str] = []


def test_result(name: str, passed: bool, details: str = ""):
    """Record test result."""
    global TESTS_RUN, TESTS_PASSED, TESTS_FAILED, FAILURES
    TESTS_RUN += 1
    status = "PASS" if passed else "FAIL"
    symbol = "[OK]" if passed else "[FAIL]"

    if passed:
        TESTS_PASSED += 1
        print(f"  {symbol} {name}")
    else:
        TESTS_FAILED += 1
        FAILURES.append(f"{name}: {details}")
        print(f"  {symbol} {name} - {details}")


def section(title: str):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


async def test_config_loading():
    """Test 1: Configuration loading and validation."""
    section("TEST 1: Configuration Loading")

    try:
        from config.config import load_config, Config

        # Load V6 production config
        config = load_config('config/config_v6_production.yaml')
        test_result("Config file loads", True)

        # Check it's a valid Config object
        test_result("Config is valid type", isinstance(config, Config))

        # Check exchange config
        test_result("Exchange configured", config.exchange.name.value == "hyperliquid")
        test_result("Testnet mode enabled", config.exchange.testnet == True)
        test_result("Wallet address set", len(config.exchange.wallet_address or "") > 10)
        test_result("Account address set", len(config.exchange.account_address or "") > 10)

        # Check trading config
        test_result("Assets configured", len(config.trading.assets) == 3)
        test_result("Assets are BTC/ETH/SOL", set(config.trading.assets) == {"BTC", "ETH", "SOL"})
        test_result("Risk percent valid", 0.1 <= config.trading.risk_percent <= 10)
        test_result("Leverage valid", 1 <= config.trading.leverage <= 10)

        # Check V6 strategies
        test_result("Has V6 strategies", config.has_v6_strategies())
        test_result("15 strategies configured", len(config.strategies) == 15)

        # Check strategy breakdown per asset
        btc_strats = config.get_strategies_for_asset("BTC")
        eth_strats = config.get_strategies_for_asset("ETH")
        sol_strats = config.get_strategies_for_asset("SOL")

        test_result("BTC has 5 strategies", len(btc_strats) == 5)
        test_result("ETH has 5 strategies", len(eth_strats) == 5)
        test_result("SOL has 5 strategies", len(sol_strats) == 5)

        return config

    except Exception as e:
        test_result("Config loading", False, str(e))
        traceback.print_exc()
        return None


async def test_exchange_connectivity(config):
    """Test 2: Exchange connectivity and data feeds."""
    section("TEST 2: Exchange Connectivity")

    if not config:
        test_result("Skipped - no config", False, "Config not loaded")
        return None

    try:
        from exchanges.hyperliquid import HyperliquidExchange

        exchange = HyperliquidExchange(
            api_key=config.exchange.api_key,
            api_secret=config.exchange.api_secret,
            wallet_address=config.exchange.wallet_address,
            account_address=config.exchange.account_address,
            testnet=config.exchange.testnet
        )

        # Connect
        connected = await exchange.connect()
        test_result("Exchange connects", connected)

        if not connected:
            return None

        # Test balance fetch
        balance = await exchange.get_balance()
        test_result("Can fetch balance", balance is not None)
        if balance:
            test_result(f"Balance: ${balance.total_balance:.2f}", balance.total_balance >= 0)

        # Test candle fetch for each asset
        for asset in ['BTC', 'ETH', 'SOL']:
            symbol = exchange.format_symbol(asset)
            candles = await exchange.get_candles(symbol, '5m', limit=100)
            has_candles = candles and len(candles) >= 50
            test_result(f"{asset} candle data ({len(candles) if candles else 0} bars)", has_candles)

        # Test multi-timeframe fetch
        candles_multi = await exchange.get_candles_multiple(
            symbol='BTC',
            timeframes=['5m', '15m', '30m', '1h', '4h'],
            limit=100
        )
        test_result("Multi-timeframe fetch works", len(candles_multi) == 5)

        return exchange

    except Exception as e:
        test_result("Exchange connectivity", False, str(e))
        traceback.print_exc()
        return None


async def test_strategy_configurations(config):
    """Test 3: Verify all 15 strategy configurations."""
    section("TEST 3: Strategy Configurations")

    if not config:
        test_result("Skipped - no config", False)
        return False

    try:
        from config.config import TimeFilterType

        all_valid = True
        strategies = config.get_enabled_strategies()

        # Expected configurations
        expected = {
            'BTC': {
                'count': 5,
                'timeframes': {'5m', '30m', '4h'},
                'vwap': True,  # BTC should have VWAP enabled
            },
            'ETH': {
                'count': 5,
                'timeframes': {'5m', '15m', '30m'},
                'vwap': False,  # ETH should NOT have VWAP
            },
            'SOL': {
                'count': 5,
                'timeframes': {'5m', '15m', '1h'},
                'vwap': False,  # SOL should NOT have VWAP
            }
        }

        for asset, exp in expected.items():
            strats = config.get_strategies_for_asset(asset)

            # Count check
            count_ok = len(strats) == exp['count']
            test_result(f"{asset}: {len(strats)} strategies", count_ok)
            if not count_ok:
                all_valid = False

            # Timeframe check
            actual_tfs = {s.timeframe for s in strats.values()}
            tf_ok = actual_tfs == exp['timeframes']
            test_result(f"{asset}: timeframes {actual_tfs}", tf_ok)
            if not tf_ok:
                all_valid = False

            # VWAP check
            vwap_settings = [s.use_vwap_confirmation for s in strats.values()]
            vwap_ok = all(v == exp['vwap'] for v in vwap_settings)
            test_result(f"{asset}: VWAP={'enabled' if exp['vwap'] else 'disabled'}", vwap_ok)
            if not vwap_ok:
                all_valid = False

        # Check signal modes and anchor levels
        signal_modes = set()
        anchor_levels = set()
        time_filters = set()

        for name, strat in strategies.items():
            signal_modes.add(strat.signal_mode)
            anchor_levels.add(strat.anchor_level)
            if strat.time_filter.enabled:
                time_filters.add(strat.time_filter.mode)

        test_result(f"Signal modes: {signal_modes}", 'simple' in signal_modes and 'enhanced' in signal_modes)
        test_result(f"Anchor levels: {anchor_levels}", anchor_levels >= {53, 60, 70})
        test_result(f"Time filters: {time_filters}", TimeFilterType.NY_HOURS_ONLY in time_filters)

        return all_valid

    except Exception as e:
        test_result("Strategy configurations", False, str(e))
        traceback.print_exc()
        return False


async def test_time_filters():
    """Test 4: Time filter logic."""
    section("TEST 4: Time Filter Logic")

    try:
        from utils.time_filter import (
            is_ny_market_hours,
            is_weekend,
            should_trade_now,
            NY_HOURS_START_UTC,
            NY_HOURS_END_UTC
        )
        from config.config import TimeFilterType, StrategyTimeFilterConfig

        # Test NY hours detection
        # NY hours are 14:00-21:00 UTC
        ny_open = datetime(2024, 1, 15, 15, 0, tzinfo=timezone.utc)  # Monday 3PM UTC - in NY hours
        ny_closed = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)  # Monday 10AM UTC - before NY

        test_result(f"15:00 UTC is NY hours", is_ny_market_hours(ny_open))
        test_result(f"10:00 UTC is NOT NY hours", not is_ny_market_hours(ny_closed))

        # Test weekend detection
        saturday = datetime(2024, 1, 13, 12, 0, tzinfo=timezone.utc)  # Saturday
        monday = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)  # Monday

        test_result("Saturday is weekend", is_weekend(saturday))
        test_result("Monday is NOT weekend", not is_weekend(monday))

        # Test should_trade_now with different configs
        ny_filter = StrategyTimeFilterConfig(enabled=True, mode=TimeFilterType.NY_HOURS_ONLY)
        weekend_filter = StrategyTimeFilterConfig(enabled=True, mode=TimeFilterType.WEEKENDS_ONLY)
        all_hours = StrategyTimeFilterConfig(enabled=False)

        # NY hours filter
        test_result("NY filter allows during NY hours", should_trade_now(ny_filter, ny_open))
        test_result("NY filter blocks outside NY hours", not should_trade_now(ny_filter, ny_closed))

        # Weekend filter
        test_result("Weekend filter allows on Saturday", should_trade_now(weekend_filter, saturday))
        test_result("Weekend filter blocks on Monday", not should_trade_now(weekend_filter, monday))

        # Disabled filter
        test_result("Disabled filter allows anytime", should_trade_now(all_hours, ny_closed))

        # Test current time
        now = datetime.now(timezone.utc)
        is_ny = is_ny_market_hours(now)
        is_wknd = is_weekend(now)
        print(f"\n  Current time: {now.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"  Is NY hours: {is_ny}")
        print(f"  Is weekend: {is_wknd}")

        return True

    except Exception as e:
        test_result("Time filter logic", False, str(e))
        traceback.print_exc()
        return False


async def test_v6_processor(config, exchange):
    """Test 5: V6 Signal Processor."""
    section("TEST 5: V6 Signal Processor")

    if not config or not exchange:
        test_result("Skipped - missing dependencies", False)
        return False

    try:
        from strategy.v6_processor import V6SignalProcessor
        import pandas as pd

        processor = V6SignalProcessor(config)
        test_result("V6 processor initializes", processor is not None)

        # Test strategy summary
        summary = processor.get_strategy_summary()
        test_result("Strategy summary generated", "15 total" in summary)
        print(f"\n{summary}\n")

        # Test each asset
        all_passed = True
        for asset in ['BTC', 'ETH', 'SOL']:
            # Get required timeframes
            tfs = processor.get_required_timeframes(asset)
            test_result(f"{asset}: timeframes identified ({len(tfs)})", len(tfs) >= 2)

            # Fetch candles
            symbol = exchange.format_symbol(asset)
            candles_raw = await exchange.get_candles_multiple(symbol, tfs, limit=200)

            # Convert to DataFrames
            candles_by_tf = {}
            for tf in tfs:
                candles = candles_raw.get(tf, [])
                if candles and len(candles) >= 50:
                    # Convert Candle objects to DataFrame
                    data = [{
                        'timestamp': c.timestamp,
                        'open': c.open,
                        'high': c.high,
                        'low': c.low,
                        'close': c.close,
                        'volume': c.volume
                    } for c in candles]
                    df = pd.DataFrame(data)
                    df.set_index('timestamp', inplace=True)
                    candles_by_tf[tf] = df

            test_result(f"{asset}: candle data ready ({len(candles_by_tf)} tfs)", len(candles_by_tf) >= 2)

            # Process asset
            result = await processor.process_asset(asset, candles_by_tf)

            strategies_ok = result.strategies_scanned == 5
            test_result(f"{asset}: 5 strategies scanned", strategies_ok,
                       f"got {result.strategies_scanned}")

            if not strategies_ok:
                all_passed = False

            # Log signal status
            if result.has_signal:
                best = result.best_signal
                print(f"    Signal: {best.signal.signal_type.value} via {best.strategy_name}")
            else:
                print(f"    No signal (normal - market conditions)")

            if result.errors:
                print(f"    Errors: {result.errors}")
                all_passed = False

        return all_passed

    except Exception as e:
        test_result("V6 processor", False, str(e))
        traceback.print_exc()
        return False


async def test_full_bot_cycle(config):
    """Test 6: Full bot initialization and signal cycle."""
    section("TEST 6: Full Bot Cycle")

    if not config:
        test_result("Skipped - no config", False)
        return False

    try:
        from core.bot import VMCBot

        bot = VMCBot(config)
        test_result("Bot instance created", bot is not None)
        test_result("V6 mode detected", bot.use_v6_strategies)

        # Start bot
        await bot.start()
        test_result("Bot starts successfully", bot.state.bot_state.value == "running")
        test_result("V6 processor initialized", bot.v6_processor is not None)
        test_result("Exchange connected", bot.exchange is not None)
        test_result("Trade manager ready", bot.trade_manager is not None)

        # Run signal check
        print("\n  Running signal check cycle...")
        signals = await bot.run_once()
        test_result("Signal check completes", True)
        test_result(f"Signals found: {len(signals)}", True)

        if signals:
            for sig in signals:
                strat = sig.metadata.get('strategy', 'unknown')
                print(f"    {sig.signal_type.value} @ {sig.entry_price:.2f} via {strat}")

        # Stop bot
        await bot.stop()
        test_result("Bot stops cleanly", True)

        return True

    except Exception as e:
        test_result("Full bot cycle", False, str(e))
        traceback.print_exc()
        return False


async def test_trade_execution_simulation(config, exchange):
    """Test 7: Trade execution simulation (no real trades)."""
    section("TEST 7: Trade Execution Simulation")

    if not config or not exchange:
        test_result("Skipped - missing dependencies", False)
        return False

    try:
        from strategy.risk import RiskManager
        from strategy.trade_manager import TradeManager, TakeProfitMethod, StopLossMethod

        # Initialize risk manager
        risk_manager = RiskManager(
            default_risk_percent=config.trading.risk_percent,
            default_leverage=config.trading.leverage,
            default_rr=config.take_profit.risk_reward,
            swing_lookback=config.stop_loss.swing_lookback,
            swing_buffer_percent=config.stop_loss.buffer_percent,
            atr_period=config.stop_loss.atr_period,
            atr_multiplier=config.stop_loss.atr_multiplier
        )
        test_result("Risk manager initializes", risk_manager is not None)

        # Initialize trade manager
        trade_manager = TradeManager(
            exchange=exchange,
            risk_manager=risk_manager,
            tp_method=TakeProfitMethod.OSCILLATOR,
            sl_method=StopLossMethod.ATR,
            max_positions=config.trading.max_positions,
            max_positions_per_asset=config.trading.max_positions_per_asset
        )
        test_result("Trade manager initializes", trade_manager is not None)

        # Get balance and calculate position size
        balance = await exchange.get_balance()
        test_result("Balance available", balance.total_balance > 0,
                   f"${balance.total_balance:.2f}")

        # Simulate position sizing for BTC
        btc_price = 0
        candles = await exchange.get_candles('BTC', '5m', limit=50)
        if candles:
            btc_price = candles[-1].close

        if btc_price > 0:
            # Calculate hypothetical position
            risk_amount = balance.total_balance * (config.trading.risk_percent / 100)
            test_result(f"Risk amount: ${risk_amount:.2f}", risk_amount > 0)

            # Assume 2% stop loss
            stop_distance = btc_price * 0.02
            position_size = risk_amount / stop_distance
            notional = position_size * btc_price

            test_result(f"Position size: {position_size:.6f} BTC", position_size > 0)
            test_result(f"Notional value: ${notional:.2f}", notional > 0)
            test_result("Position within limits", notional <= balance.total_balance * config.trading.leverage)

        # Check existing positions
        for asset in ['BTC', 'ETH', 'SOL']:
            symbol = exchange.format_symbol(asset)
            position = await exchange.get_position(symbol)
            has_position = position and abs(position.get('size', 0)) > 0
            test_result(f"{asset} position check", True,
                       "has position" if has_position else "no position")

        return True

    except Exception as e:
        test_result("Trade execution simulation", False, str(e))
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all production readiness tests."""
    print("\n" + "="*60)
    print(" VMC Trading Bot V6 - Production Readiness Tests")
    print(" " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)

    # Run tests
    config = await test_config_loading()
    exchange = await test_exchange_connectivity(config)
    await test_strategy_configurations(config)
    await test_time_filters()
    await test_v6_processor(config, exchange)
    await test_full_bot_cycle(config)
    await test_trade_execution_simulation(config, exchange)

    # Cleanup
    if exchange:
        try:
            await exchange.disconnect()
        except:
            pass

    # Print summary
    section("TEST SUMMARY")
    print(f"\n  Total tests: {TESTS_RUN}")
    print(f"  Passed: {TESTS_PASSED}")
    print(f"  Failed: {TESTS_FAILED}")

    if FAILURES:
        print(f"\n  Failures:")
        for f in FAILURES:
            print(f"    - {f}")

    success_rate = (TESTS_PASSED / TESTS_RUN * 100) if TESTS_RUN > 0 else 0
    print(f"\n  Success rate: {success_rate:.1f}%")

    if TESTS_FAILED == 0:
        print("\n  " + "="*50)
        print("  ALL TESTS PASSED - PRODUCTION READY!")
        print("  " + "="*50)
        return True
    else:
        print("\n  " + "="*50)
        print("  SOME TESTS FAILED - REVIEW REQUIRED")
        print("  " + "="*50)
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
