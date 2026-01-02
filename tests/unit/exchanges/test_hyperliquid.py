"""
Tests for Hyperliquid exchange - verifies critical recent fixes.

CRITICAL TEST FILE: These tests validate the fixes made in db2cd8b commit:
1. szDecimals caching (lines 668-703)
2. filled_size calculation (lines 540-544)
3. Order parsing ('resting' vs 'filled')
4. Minimum notional validation ($11)
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from exchanges.hyperliquid import HyperliquidExchange
from exchanges.base import OrderSide, OrderType, OrderStatus


class TestSzDecimalsCache:
    """
    Test szDecimals caching (lines 668-703 in hyperliquid.py).

    CRITICAL: Different assets have different precision requirements:
    - BTC: 5 decimals (0.00001)
    - ETH: 4 decimals (0.0001)
    - SOL: 2 decimals (0.01)

    Cache prevents repeated API calls for metadata which would slow down
    order placement and potentially hit rate limits.
    """

    @pytest.mark.asyncio
    async def test_sz_decimals_cached_per_asset(self, mock_exchange):
        """Verify szDecimals are cached after first fetch."""
        # First call should fetch metadata
        decimals = await mock_exchange._get_sz_decimals('BTC')
        assert decimals == 5

        # Cache should be populated for ALL assets from a single API call
        assert 'BTC' in mock_exchange._sz_decimals_cache
        assert 'ETH' in mock_exchange._sz_decimals_cache
        assert 'SOL' in mock_exchange._sz_decimals_cache

        # Verify cache values match metadata
        assert mock_exchange._sz_decimals_cache['BTC'] == 5
        assert mock_exchange._sz_decimals_cache['ETH'] == 4
        assert mock_exchange._sz_decimals_cache['SOL'] == 2

    @pytest.mark.asyncio
    async def test_cache_prevents_repeated_api_calls(self, mock_exchange):
        """Second call should use cache, not call meta() again."""
        # First call populates cache
        await mock_exchange._get_sz_decimals('BTC')

        # Reset mock to track new calls
        mock_exchange._info.meta.reset_mock()

        # Second call should use cache
        decimals2 = await mock_exchange._get_sz_decimals('BTC')

        assert decimals2 == 5
        mock_exchange._info.meta.assert_not_called()

    @pytest.mark.asyncio
    async def test_sz_decimals_different_per_asset(self, mock_exchange):
        """Verify different assets return their correct precisions."""
        btc_decimals = await mock_exchange._get_sz_decimals('BTC')
        eth_decimals = await mock_exchange._get_sz_decimals('ETH')
        sol_decimals = await mock_exchange._get_sz_decimals('SOL')

        assert btc_decimals == 5
        assert eth_decimals == 4
        assert sol_decimals == 2

    @pytest.mark.asyncio
    async def test_sz_decimals_unknown_asset_default(self, mock_exchange):
        """Unknown assets should return default 5."""
        decimals = await mock_exchange._get_sz_decimals('UNKNOWN_ASSET')
        assert decimals == 5

    @pytest.mark.asyncio
    async def test_sz_decimals_no_info_client(self):
        """Should return default when Info client is not available."""
        exchange = HyperliquidExchange(
            api_key='test',
            api_secret='0x' + '1' * 64,
            testnet=True
        )
        exchange._info = None

        decimals = await exchange._get_sz_decimals('BTC')
        assert decimals == 5


class TestSizeRounding:
    """Test size rounding uses correct decimals."""

    def test_size_rounding_btc_5_decimals(self):
        """BTC rounds to 5 decimals."""
        size = HyperliquidExchange._round_size(0.123456789, 5)
        assert size == 0.12346  # Rounded to 5 decimals

    def test_size_rounding_eth_4_decimals(self):
        """ETH rounds to 4 decimals."""
        size = HyperliquidExchange._round_size(0.123456789, 4)
        assert size == 0.1235  # Rounded to 4 decimals

    def test_size_rounding_sol_2_decimals(self):
        """SOL rounds to 2 decimals."""
        size = HyperliquidExchange._round_size(1.23456, 2)
        assert size == 1.23  # Rounded to 2 decimals

    def test_size_rounding_zero(self):
        """Zero size remains zero."""
        size = HyperliquidExchange._round_size(0.0, 5)
        assert size == 0.0


class TestFilledSizeCalculation:
    """
    Test filled_size calculation (lines 540-544 in hyperliquid.py).

    CRITICAL FIX: filled_size = origSz - sz (was backwards before fix!)
    - origSz: Original order size
    - sz: Remaining size
    - filled = origSz - sz (NOT sz - origSz, which gave negative values!)
    """

    @pytest.mark.asyncio
    async def test_filled_size_calculation_correct(self, mock_exchange):
        """Verify filled_size = origSz - sz (not backwards)."""
        # Set up mock user_state with partial fill
        mock_exchange._info.user_state.return_value = {
            'marginSummary': {'accountValue': '10000'},
            'withdrawable': '9000',
            'openOrders': [{
                'oid': '12345',
                'coin': 'BTC',
                'side': 'B',
                'origSz': '1.0',  # Original: 1.0
                'sz': '0.3',      # Remaining: 0.3
                'limitPx': '95000'
            }],
            'assetPositions': []
        }

        order = await mock_exchange.get_order('12345', 'BTC')

        # CRITICAL: filled = 1.0 - 0.3 = 0.7 (not -0.7!)
        assert order is not None
        assert order.filled_size == 0.7
        assert order.size == 1.0

    @pytest.mark.asyncio
    async def test_filled_size_fully_filled(self, mock_exchange):
        """Fully filled order: origSz - 0 = origSz."""
        mock_exchange._info.user_state.return_value = {
            'marginSummary': {'accountValue': '10000'},
            'withdrawable': '9000',
            'openOrders': [{
                'oid': '12345',
                'coin': 'BTC',
                'side': 'B',
                'origSz': '0.5',
                'sz': '0',  # Fully filled, 0 remaining
                'limitPx': '95000'
            }],
            'assetPositions': []
        }

        order = await mock_exchange.get_order('12345', 'BTC')

        assert order is not None
        assert order.filled_size == 0.5
        assert order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_filled_size_never_negative(self, mock_exchange):
        """filled_size should never be negative even with bad data."""
        # Edge case: somehow sz > origSz (shouldn't happen but should be safe)
        mock_exchange._info.user_state.return_value = {
            'marginSummary': {'accountValue': '10000'},
            'withdrawable': '9000',
            'openOrders': [{
                'oid': '12345',
                'coin': 'BTC',
                'side': 'B',
                'origSz': '0.5',
                'sz': '1.0',  # More remaining than original (invalid state)
                'limitPx': '95000'
            }],
            'assetPositions': []
        }

        order = await mock_exchange.get_order('12345', 'BTC')

        # max(0, origSz - sz) ensures never negative
        assert order is not None
        assert order.filled_size >= 0

    @pytest.mark.asyncio
    async def test_filled_size_zero_for_unfilled(self, mock_exchange):
        """Unfilled order: origSz - origSz = 0."""
        mock_exchange._info.user_state.return_value = {
            'marginSummary': {'accountValue': '10000'},
            'withdrawable': '9000',
            'openOrders': [{
                'oid': '12345',
                'coin': 'BTC',
                'side': 'B',
                'origSz': '1.0',
                'sz': '1.0',  # Nothing filled yet
                'limitPx': '95000'
            }],
            'assetPositions': []
        }

        order = await mock_exchange.get_order('12345', 'BTC')

        assert order is not None
        assert order.filled_size == 0.0
        assert order.status == OrderStatus.OPEN


class TestOrderParsing:
    """Test order response parsing for 'resting' vs 'filled' responses."""

    @pytest.mark.asyncio
    async def test_market_order_filled_response(self, mock_exchange):
        """Market orders return 'filled' status."""
        mock_exchange._exchange.order.return_value = {
            'status': 'ok',
            'response': {
                'data': {
                    'statuses': [{
                        'filled': {
                            'oid': '12345',
                            'avgPx': '95050.5'
                        }
                    }]
                }
            }
        }

        # Mock get_ticker for market order price calculation
        mock_exchange._info.all_mids.return_value = {'BTC': '95000'}

        order = await mock_exchange.place_order(
            symbol='BTC',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=0.001
        )

        assert order.order_id == '12345'
        assert order.status == OrderStatus.FILLED
        assert order.avg_fill_price == 95050.5

    @pytest.mark.asyncio
    async def test_limit_order_resting_response(self, mock_exchange):
        """Limit orders return 'resting' status."""
        mock_exchange._exchange.order.return_value = {
            'status': 'ok',
            'response': {
                'data': {
                    'statuses': [{
                        'resting': {
                            'oid': '12346'
                        }
                    }]
                }
            }
        }

        order = await mock_exchange.place_order(
            symbol='BTC',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            size=0.001,
            price=94000.0
        )

        assert order.order_id == '12346'
        assert order.status == OrderStatus.OPEN

    @pytest.mark.asyncio
    async def test_order_error_response(self, mock_exchange):
        """Order errors should raise exception."""
        mock_exchange._exchange.order.return_value = {
            'status': 'ok',
            'response': {
                'data': {
                    'statuses': [{
                        'error': 'Insufficient margin'
                    }]
                }
            }
        }

        mock_exchange._info.all_mids.return_value = {'BTC': '95000'}

        with pytest.raises(Exception, match='Order rejected'):
            await mock_exchange.place_order(
                symbol='BTC',
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                size=0.001
            )

    @pytest.mark.asyncio
    async def test_order_missing_id_raises(self, mock_exchange):
        """Missing order ID should raise exception."""
        mock_exchange._exchange.order.return_value = {
            'status': 'ok',
            'response': {
                'data': {
                    'statuses': [{}]  # No 'resting' or 'filled'
                }
            }
        }

        mock_exchange._info.all_mids.return_value = {'BTC': '95000'}

        with pytest.raises(Exception, match='no order ID'):
            await mock_exchange.place_order(
                symbol='BTC',
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                size=0.001
            )


class TestMinimumNotional:
    """Test minimum notional validation ($11 minimum)."""

    @pytest.mark.asyncio
    async def test_order_below_minimum_notional_rejected(self, mock_exchange):
        """Orders below $11 notional should be rejected."""
        mock_exchange._info.all_mids.return_value = {'BTC': '95000'}

        # 0.0001 * 95000 = $9.50 < $11 minimum
        with pytest.raises(ValueError, match='below minimum'):
            await mock_exchange.place_order(
                symbol='BTC',
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                size=0.0001
            )

    @pytest.mark.asyncio
    async def test_order_at_minimum_notional_accepted(self, mock_exchange):
        """Orders at exactly $11 should be accepted."""
        mock_exchange._info.all_mids.return_value = {'BTC': '95000'}

        # Calculate size for exactly $11 notional
        # $11 / $95000 = 0.0001157... rounded to 5 decimals = 0.00012
        # 0.00012 * 95000 = $11.40 > $11 OK

        order = await mock_exchange.place_order(
            symbol='BTC',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=0.00012
        )

        assert order is not None
        assert order.order_id == '12345'

    @pytest.mark.asyncio
    async def test_order_above_minimum_notional_accepted(self, mock_exchange):
        """Orders above $11 notional should proceed."""
        mock_exchange._info.all_mids.return_value = {'BTC': '95000'}

        # 0.001 * 95000 = $95 > $11 OK
        order = await mock_exchange.place_order(
            symbol='BTC',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=0.001
        )

        assert order is not None
        assert order.order_id == '12345'


class TestPriceRounding:
    """Test price rounding to 5 significant figures."""

    def test_price_rounding_large_number(self):
        """Large prices round to 5 significant figures."""
        price = HyperliquidExchange._round_price(95123.456789)
        assert price == 95123  # 5 significant figures

    def test_price_rounding_medium_number(self):
        """Medium prices maintain precision."""
        price = HyperliquidExchange._round_price(1234.56789)
        assert price == 1234.6  # 5 significant figures

    def test_price_rounding_small_number(self):
        """Small prices preserve precision."""
        price = HyperliquidExchange._round_price(0.123456789)
        assert price == 0.12346  # 5 significant figures

    def test_price_rounding_zero(self):
        """Zero should remain zero."""
        price = HyperliquidExchange._round_price(0.0)
        assert price == 0.0

    def test_price_rounding_very_small(self):
        """Very small prices handle correctly."""
        price = HyperliquidExchange._round_price(0.00012345678)
        assert price == 0.00012346  # 5 significant figures


class TestOrderSides:
    """Test order side handling."""

    @pytest.mark.asyncio
    async def test_buy_order_side(self, mock_exchange):
        """BUY orders pass is_buy=True to SDK."""
        mock_exchange._info.all_mids.return_value = {'BTC': '95000'}

        await mock_exchange.place_order(
            symbol='BTC',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=0.001
        )

        # Verify the exchange.order was called with is_buy=True
        call_args = mock_exchange._exchange.order.call_args[0]
        assert call_args[1] is True  # is_buy parameter

    @pytest.mark.asyncio
    async def test_sell_order_side(self, mock_exchange):
        """SELL orders pass is_buy=False to SDK."""
        mock_exchange._info.all_mids.return_value = {'BTC': '95000'}

        await mock_exchange.place_order(
            symbol='BTC',
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            size=0.001
        )

        # Verify the exchange.order was called with is_buy=False
        call_args = mock_exchange._exchange.order.call_args[0]
        assert call_args[1] is False  # is_buy parameter


class TestGetOpenOrders:
    """Test get_open_orders functionality."""

    @pytest.mark.asyncio
    async def test_get_open_orders_empty(self, mock_exchange):
        """Returns empty list when no orders."""
        mock_exchange._info.user_state.return_value = {
            'marginSummary': {'accountValue': '10000'},
            'withdrawable': '9000',
            'openOrders': [],
            'assetPositions': []
        }

        orders = await mock_exchange.get_open_orders()
        assert orders == []

    @pytest.mark.asyncio
    async def test_get_open_orders_with_filter(self, mock_exchange):
        """Filter by symbol works correctly."""
        mock_exchange._info.user_state.return_value = {
            'marginSummary': {'accountValue': '10000'},
            'withdrawable': '9000',
            'openOrders': [
                {'oid': '1', 'coin': 'BTC', 'side': 'B', 'origSz': '0.1', 'sz': '0.1', 'limitPx': '95000'},
                {'oid': '2', 'coin': 'ETH', 'side': 'S', 'origSz': '1.0', 'sz': '1.0', 'limitPx': '3500'},
            ],
            'assetPositions': []
        }

        btc_orders = await mock_exchange.get_open_orders(symbol='BTC')
        assert len(btc_orders) == 1
        assert btc_orders[0].symbol == 'BTC'

        eth_orders = await mock_exchange.get_open_orders(symbol='ETH')
        assert len(eth_orders) == 1
        assert eth_orders[0].symbol == 'ETH'

        all_orders = await mock_exchange.get_open_orders()
        assert len(all_orders) == 2
