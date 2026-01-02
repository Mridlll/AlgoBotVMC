"""
Shared fixtures and test configuration for VMC Trading Bot test suite.

This module provides:
- Sample OHLCV data generators
- Mock Hyperliquid exchange fixtures
- Indicator instances
- Signal detector instances
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any, Optional
import asyncio

# Path setup for imports
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


# =============================================================================
# SAMPLE DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_ohlcv_data():
    """Generate 200 bars of realistic BTC-like OHLCV data."""
    np.random.seed(42)
    n = 200

    timestamps = pd.date_range(
        start=datetime.now(timezone.utc) - timedelta(hours=n * 4),
        periods=n,
        freq='4h'
    )

    # Generate trending price data
    trend = np.cumsum(np.random.randn(n) * 0.5) + 95000

    df = pd.DataFrame({
        'open': trend + np.random.randn(n) * 100,
        'high': trend + np.abs(np.random.randn(n)) * 200,
        'low': trend - np.abs(np.random.randn(n)) * 200,
        'close': trend + np.random.randn(n) * 100,
        'volume': np.random.rand(n) * 1000000
    }, index=timestamps)

    # Ensure high is highest and low is lowest
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


@pytest.fixture
def flat_price_data():
    """Flat prices for division-by-zero testing."""
    n = 100
    timestamps = pd.date_range(
        start=datetime.now(timezone.utc) - timedelta(hours=n * 4),
        periods=n,
        freq='4h'
    )

    return pd.DataFrame({
        'open': [100.0] * n,
        'high': [100.0] * n,
        'low': [100.0] * n,
        'close': [100.0] * n,
        'volume': [1000.0] * n
    }, index=timestamps)


@pytest.fixture
def oversold_data():
    """Data designed to produce WT2 < -53 (oversold condition)."""
    n = 100
    timestamps = pd.date_range(
        start=datetime.now(timezone.utc) - timedelta(hours=n * 4),
        periods=n,
        freq='4h'
    )

    # Sharp downtrend to trigger oversold
    trend = np.linspace(100000, 85000, n)

    return pd.DataFrame({
        'open': trend + 50,
        'high': trend + 100,
        'low': trend - 100,
        'close': trend - 50,
        'volume': np.random.rand(n) * 1000000
    }, index=timestamps)


@pytest.fixture
def overbought_data():
    """Data designed to produce WT2 > 53 (overbought condition)."""
    n = 100
    timestamps = pd.date_range(
        start=datetime.now(timezone.utc) - timedelta(hours=n * 4),
        periods=n,
        freq='4h'
    )

    # Sharp uptrend to trigger overbought
    trend = np.linspace(85000, 100000, n)

    return pd.DataFrame({
        'open': trend - 50,
        'high': trend + 100,
        'low': trend - 100,
        'close': trend + 50,
        'volume': np.random.rand(n) * 1000000
    }, index=timestamps)


# =============================================================================
# MOCK HYPERLIQUID EXCHANGE FIXTURES
# =============================================================================

@pytest.fixture
def mock_hyperliquid_info():
    """Mock Hyperliquid Info client."""
    mock = MagicMock()

    # Mock metadata - CRITICAL for szDecimals caching tests
    mock.meta.return_value = {
        'universe': [
            {'name': 'BTC', 'szDecimals': 5, 'maxLeverage': 100},
            {'name': 'ETH', 'szDecimals': 4, 'maxLeverage': 100},
            {'name': 'SOL', 'szDecimals': 2, 'maxLeverage': 50},
        ]
    }

    # Mock all_mids (prices)
    mock.all_mids.return_value = {
        'BTC': '95000.0',
        'ETH': '3500.0',
        'SOL': '180.0'
    }

    # Mock user_state - used for balance, positions, and orders
    mock.user_state.return_value = {
        'marginSummary': {
            'accountValue': '10000.0',
            'totalMarginUsed': '1000.0',
            'totalUnrealizedPnl': '50.0'
        },
        'withdrawable': '9000.0',
        'assetPositions': [],
        'openOrders': []
    }

    # Mock candles_snapshot
    candles = []
    base_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    for i in range(100, 0, -1):
        candles.append({
            't': base_time - (i * 4 * 3600 * 1000),  # 4h candles
            'o': str(95000 + i * 10),
            'h': str(95100 + i * 10),
            'l': str(94900 + i * 10),
            'c': str(95050 + i * 10),
            'v': '1000'
        })
    mock.candles_snapshot.return_value = candles

    return mock


@pytest.fixture
def mock_hyperliquid_exchange():
    """Mock Hyperliquid Exchange client."""
    mock = MagicMock()

    # Mock order placement - returns 'filled' for market orders
    mock.order.return_value = {
        'status': 'ok',
        'response': {
            'data': {
                'statuses': [{
                    'filled': {
                        'oid': '12345',
                        'avgPx': '95000.0'
                    }
                }]
            }
        }
    }

    # Mock cancel
    mock.cancel.return_value = {'status': 'ok'}

    # Mock leverage update
    mock.update_leverage.return_value = {'status': 'ok'}

    return mock


@pytest.fixture
def mock_exchange(mock_hyperliquid_info, mock_hyperliquid_exchange):
    """Full mock HyperliquidExchange for testing."""
    from exchanges.hyperliquid import HyperliquidExchange

    # Create exchange with test credentials
    exchange = HyperliquidExchange(
        api_key='test_key',
        api_secret='0x' + '1' * 64,  # Mock private key (64 hex chars)
        wallet_address='0x' + '2' * 40,
        account_address='0x' + '3' * 40,
        testnet=True
    )

    # Inject mocks
    exchange._info = mock_hyperliquid_info
    exchange._exchange = mock_hyperliquid_exchange
    exchange._connected = True
    exchange._signing_address = '0x' + '4' * 40
    exchange._sz_decimals_cache = {}  # Start with empty cache

    return exchange


@pytest.fixture
def mock_balance():
    """Mock account balance."""
    from exchanges.base import AccountBalance
    return AccountBalance(
        total_balance=10000.0,
        available_balance=9000.0,
        used_margin=1000.0,
        unrealized_pnl=50.0
    )


# =============================================================================
# INDICATOR FIXTURES
# =============================================================================

@pytest.fixture
def wavetrend():
    """WaveTrend indicator with default settings."""
    from indicators.wavetrend import WaveTrend
    return WaveTrend()


@pytest.fixture
def money_flow():
    """MoneyFlow indicator with default settings."""
    from indicators.money_flow import MoneyFlow
    return MoneyFlow()


@pytest.fixture
def vwap_calculator():
    """VWAP calculator."""
    from indicators.vwap import VWAPCalculator
    return VWAPCalculator()


# =============================================================================
# SIGNAL DETECTION FIXTURES
# =============================================================================

@pytest.fixture
def signal_detector():
    """SignalDetector with default settings."""
    from strategy.signals import SignalDetector
    return SignalDetector(
        anchor_level_long=-60,
        anchor_level_short=60,
        trigger_lookback=20,
        mfi_lookback=3
    )


@pytest.fixture
def simple_signal_detector():
    """SimpleSignalDetector for simple mode testing."""
    from strategy.signals import SimpleSignalDetector
    return SimpleSignalDetector(
        oversold_level=-53,
        overbought_level=53
    )


# =============================================================================
# TRADE MANAGER FIXTURES
# =============================================================================

@pytest.fixture
def mock_risk_manager():
    """Mock RiskManager."""
    from strategy.risk import RiskManager, RiskParams

    rm = MagicMock(spec=RiskManager)
    rm.calculate_full_risk_params.return_value = RiskParams(
        entry_price=95000.0,
        stop_loss=93000.0,
        take_profit=99000.0,
        position_size=0.001,
        risk_amount=30.0,
        risk_reward=2.0
    )
    return rm


@pytest.fixture
def trade_manager(mock_exchange, mock_risk_manager):
    """TradeManager with mocks."""
    from strategy.trade_manager import TradeManager

    return TradeManager(
        exchange=mock_exchange,
        risk_manager=mock_risk_manager,
        max_positions=2,
        max_positions_per_asset=1
    )


@pytest.fixture
def sample_signal():
    """Sample trading signal."""
    from strategy.signals import Signal, SignalType, AnchorWave, TriggerWave

    now = datetime.now(timezone.utc)
    return Signal(
        signal_type=SignalType.LONG,
        timestamp=now,
        entry_price=95000.0,
        anchor_wave=AnchorWave(now, -65, 100, SignalType.LONG),
        trigger_wave=TriggerWave(now, -55, 110, False),
        wt1=-30,
        wt2=-35,
        vwap=5,
        mfi=-2,
        metadata={'symbol': 'BTC', 'timeframe': '4h'}
    )


# =============================================================================
# CONFIG FIXTURES
# =============================================================================

@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = MagicMock()
    config.trading.assets = ['BTC', 'ETH']
    config.trading.risk_percent = 3.0
    config.trading.leverage = 10
    config.trading.max_positions = 2
    config.trading.max_positions_per_asset = 1
    config.indicators.wt_channel_len = 9
    config.indicators.wt_average_len = 12
    config.indicators.wt_ma_len = 3
    config.indicators.mfi_period = 60
    config.indicators.mfi_multiplier = 150.0

    return config
