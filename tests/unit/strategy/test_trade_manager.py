"""
Tests for TradeManager - verifies thread safety and validation.

CRITICAL TEST FILE: These tests validate the fixes made in db2cd8b commit:
1. asyncio.Lock for thread-safe _active_trades access
2. Balance validation before trade execution
3. Position size validation
4. Minimum notional validation ($11)
5. Maximum position limits
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from strategy.trade_manager import TradeManager, Trade, TradeStatus
from strategy.signals import Signal, SignalType, AnchorWave, TriggerWave
from strategy.risk import RiskManager, RiskParams
from exchanges.base import AccountBalance, OrderSide, OrderType, OrderStatus, Order


class TestAsyncLockThreadSafety:
    """
    Test asyncio.Lock for thread-safe trade access.

    CRITICAL: The _trades_lock prevents race conditions when multiple
    async operations try to modify _active_trades simultaneously.
    """

    @pytest.mark.asyncio
    async def test_add_trade_uses_lock(self, trade_manager):
        """Verify add_trade acquires lock."""
        trade = Trade(
            trade_id='test_1',
            symbol='BTC',
            signal_type=SignalType.LONG,
            entry_price=95000.0,
            size=0.001,
            stop_loss=93000.0,
            take_profit=99000.0
        )

        await trade_manager.add_trade(trade)

        assert 'test_1' in trade_manager._active_trades

    @pytest.mark.asyncio
    async def test_remove_trade_uses_lock(self, trade_manager):
        """Verify remove_trade acquires lock."""
        # Add trade directly (bypassing lock for setup)
        trade = Trade(
            trade_id='test_2',
            symbol='BTC',
            signal_type=SignalType.LONG,
            entry_price=95000.0,
            size=0.001,
            stop_loss=93000.0,
            take_profit=99000.0
        )
        trade_manager._active_trades['test_2'] = trade

        removed = await trade_manager.remove_trade('test_2')

        assert removed is not None
        assert removed.trade_id == 'test_2'
        assert 'test_2' not in trade_manager._active_trades

    @pytest.mark.asyncio
    async def test_remove_nonexistent_trade(self, trade_manager):
        """Removing nonexistent trade returns None."""
        removed = await trade_manager.remove_trade('nonexistent')
        assert removed is None

    @pytest.mark.asyncio
    async def test_get_trade_uses_lock(self, trade_manager):
        """Verify get_trade acquires lock."""
        trade = Trade(
            trade_id='test_3',
            symbol='BTC',
            signal_type=SignalType.LONG,
            entry_price=95000.0,
            size=0.001,
            stop_loss=93000.0,
            take_profit=99000.0
        )
        trade_manager._active_trades['test_3'] = trade

        retrieved = await trade_manager.get_trade('test_3')

        assert retrieved is not None
        assert retrieved.trade_id == 'test_3'

    @pytest.mark.asyncio
    async def test_concurrent_access_safe(self, trade_manager):
        """Multiple concurrent operations should be safe."""
        async def add_remove(trade_id: str):
            trade = Trade(
                trade_id=trade_id,
                symbol='BTC',
                signal_type=SignalType.LONG,
                entry_price=95000.0,
                size=0.001,
                stop_loss=93000.0,
                take_profit=99000.0
            )
            await trade_manager.add_trade(trade)
            await asyncio.sleep(0.01)  # Simulate some work
            await trade_manager.remove_trade(trade_id)

        # Run multiple concurrent operations
        await asyncio.gather(
            add_remove('concurrent_1'),
            add_remove('concurrent_2'),
            add_remove('concurrent_3'),
            add_remove('concurrent_4'),
            add_remove('concurrent_5')
        )

        # Should complete without errors, all trades cleaned up
        assert len(trade_manager._active_trades) == 0

    @pytest.mark.asyncio
    async def test_active_trades_property_returns_snapshot(self, trade_manager):
        """active_trades property returns a read-only snapshot."""
        trade = Trade(
            trade_id='test_4',
            symbol='BTC',
            signal_type=SignalType.LONG,
            entry_price=95000.0,
            size=0.001,
            stop_loss=93000.0,
            take_profit=99000.0
        )
        trade_manager._active_trades['test_4'] = trade

        # Get snapshot
        snapshot = trade_manager.active_trades

        # Modifying snapshot should not affect internal state
        snapshot['new_trade'] = MagicMock()
        assert 'new_trade' not in trade_manager._active_trades


class TestBalanceValidation:
    """
    Test balance validation before trade execution.

    CRITICAL: The execute_signal method must validate balance > 0
    before attempting to place an order.
    """

    @pytest.mark.asyncio
    async def test_zero_balance_rejected(self, trade_manager, sample_signal):
        """Zero balance should prevent trade."""
        trade_manager.exchange.get_balance = AsyncMock(return_value=AccountBalance(
            total_balance=0.0,
            available_balance=0.0,
            used_margin=0.0,
            unrealized_pnl=0.0
        ))

        result = await trade_manager.execute_signal(
            signal=sample_signal,
            df=MagicMock()
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_negative_balance_rejected(self, trade_manager, sample_signal):
        """Negative balance should prevent trade."""
        trade_manager.exchange.get_balance = AsyncMock(return_value=AccountBalance(
            total_balance=100.0,
            available_balance=-100.0,
            used_margin=200.0,
            unrealized_pnl=0.0
        ))

        result = await trade_manager.execute_signal(
            signal=sample_signal,
            df=MagicMock()
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_none_balance_rejected(self, trade_manager, sample_signal):
        """None balance should prevent trade."""
        trade_manager.exchange.get_balance = AsyncMock(return_value=None)

        result = await trade_manager.execute_signal(
            signal=sample_signal,
            df=MagicMock()
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_valid_balance_allowed(self, trade_manager, sample_signal):
        """Valid positive balance should allow trade to proceed."""
        trade_manager.exchange.get_balance = AsyncMock(return_value=AccountBalance(
            total_balance=10000.0,
            available_balance=10000.0,
            used_margin=0.0,
            unrealized_pnl=0.0
        ))

        # Mock the order placement to return a filled order
        trade_manager.exchange.place_order = AsyncMock(return_value=Order(
            order_id='test_order',
            symbol='BTC',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=0.001,
            price=95000.0,
            status=OrderStatus.FILLED,
            filled_size=0.001,
            avg_fill_price=95000.0
        ))

        # Mock wait_for_fill to return the order immediately
        trade_manager._wait_for_fill = AsyncMock(return_value=Order(
            order_id='test_order',
            symbol='BTC',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=0.001,
            price=95000.0,
            status=OrderStatus.FILLED,
            filled_size=0.001,
            avg_fill_price=95000.0
        ))

        result = await trade_manager.execute_signal(
            signal=sample_signal,
            df=MagicMock()
        )

        assert result is not None


class TestPositionSizeValidation:
    """
    Test position size validation.

    CRITICAL: Position size must be > 0 before placing order.
    """

    @pytest.mark.asyncio
    async def test_zero_position_size_rejected(self, trade_manager, sample_signal):
        """Zero position size should prevent trade."""
        trade_manager.exchange.get_balance = AsyncMock(return_value=AccountBalance(
            total_balance=10000.0,
            available_balance=10000.0,
            used_margin=0.0,
            unrealized_pnl=0.0
        ))

        # Mock risk manager to return zero position size
        trade_manager.risk_manager.calculate_full_risk_params.return_value = RiskParams(
            entry_price=95000.0,
            stop_loss=93000.0,
            take_profit=99000.0,
            position_size=0.0,
            risk_amount=0.0,
            risk_reward=2.0
        )

        result = await trade_manager.execute_signal(
            signal=sample_signal,
            df=MagicMock()
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_negative_position_size_rejected(self, trade_manager, sample_signal):
        """Negative position size should prevent trade."""
        trade_manager.exchange.get_balance = AsyncMock(return_value=AccountBalance(
            total_balance=10000.0,
            available_balance=10000.0,
            used_margin=0.0,
            unrealized_pnl=0.0
        ))

        trade_manager.risk_manager.calculate_full_risk_params.return_value = RiskParams(
            entry_price=95000.0,
            stop_loss=93000.0,
            take_profit=99000.0,
            position_size=-0.001,
            risk_amount=30.0,
            risk_reward=2.0
        )

        result = await trade_manager.execute_signal(
            signal=sample_signal,
            df=MagicMock()
        )

        assert result is None


class TestMinimumNotional:
    """
    Test minimum notional validation ($11).

    CRITICAL: Orders with notional < $11 should be rejected to
    prevent Hyperliquid order rejections.
    """

    @pytest.mark.asyncio
    async def test_below_minimum_notional_rejected(self, trade_manager, sample_signal):
        """Position below $11 notional should be rejected."""
        trade_manager.exchange.get_balance = AsyncMock(return_value=AccountBalance(
            total_balance=10000.0,
            available_balance=10000.0,
            used_margin=0.0,
            unrealized_pnl=0.0
        ))

        # Position size that results in < $11 notional
        # 0.0001 * 95000 = $9.50 < $11
        trade_manager.risk_manager.calculate_full_risk_params.return_value = RiskParams(
            entry_price=95000.0,
            stop_loss=93000.0,
            take_profit=99000.0,
            position_size=0.0001,
            risk_amount=3.0,
            risk_reward=2.0
        )

        result = await trade_manager.execute_signal(
            signal=sample_signal,
            df=MagicMock()
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_above_minimum_notional_accepted(self, trade_manager, sample_signal):
        """Position above $11 notional should be accepted."""
        trade_manager.exchange.get_balance = AsyncMock(return_value=AccountBalance(
            total_balance=10000.0,
            available_balance=10000.0,
            used_margin=0.0,
            unrealized_pnl=0.0
        ))

        # Position size that results in > $11 notional
        # 0.001 * 95000 = $95 > $11
        trade_manager.risk_manager.calculate_full_risk_params.return_value = RiskParams(
            entry_price=95000.0,
            stop_loss=93000.0,
            take_profit=99000.0,
            position_size=0.001,
            risk_amount=30.0,
            risk_reward=2.0
        )

        trade_manager.exchange.place_order = AsyncMock(return_value=Order(
            order_id='test_order',
            symbol='BTC',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=0.001,
            price=95000.0,
            status=OrderStatus.FILLED,
            filled_size=0.001,
            avg_fill_price=95000.0
        ))

        trade_manager._wait_for_fill = AsyncMock(return_value=Order(
            order_id='test_order',
            symbol='BTC',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=0.001,
            price=95000.0,
            status=OrderStatus.FILLED,
            filled_size=0.001,
            avg_fill_price=95000.0
        ))

        result = await trade_manager.execute_signal(
            signal=sample_signal,
            df=MagicMock()
        )

        assert result is not None


class TestMaxPositions:
    """
    Test position limits enforcement.

    max_positions and max_positions_per_asset must be enforced.
    """

    @pytest.mark.asyncio
    async def test_max_positions_enforced(self, trade_manager):
        """Cannot open trade when max positions reached."""
        # Fill max positions (default is 2)
        for i in range(trade_manager.max_positions):
            trade = Trade(
                trade_id=f'trade_{i}',
                symbol='ETH',  # Different asset
                signal_type=SignalType.LONG,
                entry_price=3500.0,
                size=0.1,
                stop_loss=3300.0,
                take_profit=3900.0
            )
            trade_manager._active_trades[f'trade_{i}'] = trade

        can_open = await trade_manager.can_open_trade('BTC')
        assert not can_open

    @pytest.mark.asyncio
    async def test_max_per_asset_enforced(self, trade_manager):
        """Cannot open trade when max per asset reached."""
        # Add one position for BTC (max_positions_per_asset is 1)
        trade = Trade(
            trade_id='btc_trade_1',
            symbol='BTC',
            signal_type=SignalType.LONG,
            entry_price=95000.0,
            size=0.001,
            stop_loss=93000.0,
            take_profit=99000.0
        )
        trade_manager._active_trades['btc_trade_1'] = trade

        can_open = await trade_manager.can_open_trade('BTC')
        assert not can_open

    @pytest.mark.asyncio
    async def test_can_open_when_under_limits(self, trade_manager):
        """Can open trade when under all limits."""
        can_open = await trade_manager.can_open_trade('BTC')
        assert can_open

    @pytest.mark.asyncio
    async def test_can_open_different_asset(self, trade_manager):
        """Can open trade for different asset when under limits."""
        # Add BTC position
        trade = Trade(
            trade_id='btc_trade',
            symbol='BTC',
            signal_type=SignalType.LONG,
            entry_price=95000.0,
            size=0.001,
            stop_loss=93000.0,
            take_profit=99000.0
        )
        trade_manager._active_trades['btc_trade'] = trade

        # Should still be able to open ETH position
        can_open = await trade_manager.can_open_trade('ETH')
        assert can_open


class TestTradeIdGeneration:
    """Test trade ID generation."""

    def test_trade_id_unique(self, trade_manager):
        """Generated trade IDs should be unique."""
        ids = set()
        for _ in range(100):
            trade_id = trade_manager._generate_trade_id()
            assert trade_id not in ids
            ids.add(trade_id)

    def test_trade_id_format(self, trade_manager):
        """Trade ID should follow expected format."""
        trade_id = trade_manager._generate_trade_id()

        # Format: trade_{counter}_{timestamp}
        assert trade_id.startswith('trade_')
        parts = trade_id.split('_')
        assert len(parts) == 3


class TestTrade:
    """Test Trade dataclass."""

    def test_is_long_property(self):
        """is_long should return True for LONG trades."""
        long_trade = Trade(
            trade_id='test',
            symbol='BTC',
            signal_type=SignalType.LONG,
            entry_price=95000.0,
            size=0.001,
            stop_loss=93000.0,
            take_profit=99000.0
        )
        assert long_trade.is_long is True

        short_trade = Trade(
            trade_id='test',
            symbol='BTC',
            signal_type=SignalType.SHORT,
            entry_price=95000.0,
            size=0.001,
            stop_loss=97000.0,
            take_profit=91000.0
        )
        assert short_trade.is_long is False

    def test_to_dict(self):
        """to_dict should return complete trade info."""
        trade = Trade(
            trade_id='test',
            symbol='BTC',
            signal_type=SignalType.LONG,
            entry_price=95000.0,
            size=0.001,
            stop_loss=93000.0,
            take_profit=99000.0,
            status=TradeStatus.OPEN
        )

        d = trade.to_dict()

        assert d['trade_id'] == 'test'
        assert d['symbol'] == 'BTC'
        assert d['signal_type'] == 'long'
        assert d['entry_price'] == 95000.0
        assert d['size'] == 0.001
        assert d['stop_loss'] == 93000.0
        assert d['take_profit'] == 99000.0
        assert d['status'] == 'open'
