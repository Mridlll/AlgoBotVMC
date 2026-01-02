"""
Unit tests for WaveTrend oscillator.

Tests verify:
1. Calculation accuracy (EMA/SMA)
2. Division by zero handling
3. Cross detection logic
4. Overbought/oversold levels
"""

import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from indicators.wavetrend import WaveTrend, calculate_wavetrend, WaveTrendResult


class TestWaveTrendCalculation:
    """Test WaveTrend calculation accuracy."""

    def test_output_structure(self, sample_ohlcv_data, wavetrend):
        """Verify output has all expected fields."""
        result = wavetrend.calculate(sample_ohlcv_data)

        assert isinstance(result, WaveTrendResult)
        assert len(result.wt1) == len(sample_ohlcv_data)
        assert len(result.wt2) == len(sample_ohlcv_data)
        assert len(result.momentum) == len(sample_ohlcv_data)
        assert len(result.cross) == len(sample_ohlcv_data)
        assert len(result.cross_up) == len(sample_ohlcv_data)
        assert len(result.cross_down) == len(sample_ohlcv_data)
        assert len(result.oversold) == len(sample_ohlcv_data)
        assert len(result.overbought) == len(sample_ohlcv_data)

    def test_momentum_equals_wt1_minus_wt2(self, sample_ohlcv_data, wavetrend):
        """Verify momentum = WT1 - WT2."""
        result = wavetrend.calculate(sample_ohlcv_data)

        expected_momentum = result.wt1 - result.wt2

        # Compare only where not NaN
        valid_idx = ~result.momentum.isna()
        np.testing.assert_array_almost_equal(
            result.momentum[valid_idx].values,
            expected_momentum[valid_idx].values,
            decimal=10
        )

    def test_wt2_is_sma_of_wt1(self, sample_ohlcv_data, wavetrend):
        """Verify WT2 = SMA(WT1, ma_len)."""
        result = wavetrend.calculate(sample_ohlcv_data)

        # Manually calculate SMA of WT1
        expected_wt2 = result.wt1.rolling(window=wavetrend.ma_len).mean()

        # Compare after warmup period (skip first 20 bars)
        valid_idx = ~expected_wt2.isna()
        np.testing.assert_array_almost_equal(
            result.wt2[valid_idx].values,
            expected_wt2[valid_idx].values,
            decimal=10
        )

    def test_default_parameters(self):
        """Verify default WaveTrend parameters."""
        wt = WaveTrend()

        assert wt.channel_len == 9
        assert wt.average_len == 12
        assert wt.ma_len == 3
        assert wt.overbought_1 == 53
        assert wt.overbought_2 == 60
        assert wt.oversold_1 == -53
        assert wt.oversold_2 == -60


class TestDivisionByZeroHandling:
    """Test division by zero scenarios."""

    def test_flat_prices_no_inf(self, flat_price_data, wavetrend):
        """Flat prices should not produce Inf values."""
        result = wavetrend.calculate(flat_price_data)

        # Skip NaN values from warmup period
        valid_wt1 = result.wt1.dropna()
        valid_wt2 = result.wt2.dropna()
        valid_momentum = result.momentum.dropna()

        assert not np.isinf(valid_wt1).any()
        assert not np.isinf(valid_wt2).any()
        assert not np.isinf(valid_momentum).any()

    def test_flat_prices_produces_bounded_values(self, flat_price_data, wavetrend):
        """Flat prices should produce bounded WT values (not extreme)."""
        result = wavetrend.calculate(flat_price_data)

        # After warmup, values should be stable
        stable_wt1 = result.wt1.iloc[50:].dropna()

        # With flat prices, CI = 0 (when d = 0), so WT values should be near 0
        assert stable_wt1.abs().max() < 100  # Should be bounded


class TestCrossDetection:
    """Test WaveTrend cross detection logic."""

    def test_cross_exclusive(self, sample_ohlcv_data, wavetrend):
        """cross_up and cross_down should be mutually exclusive."""
        result = wavetrend.calculate(sample_ohlcv_data)

        # No bar should have both cross_up and cross_down
        both = result.cross_up & result.cross_down
        assert not both.any(), "cross_up and cross_down should be mutually exclusive"

    def test_cross_implies_direction(self, sample_ohlcv_data, wavetrend):
        """If cross is True, either cross_up or cross_down must be True."""
        result = wavetrend.calculate(sample_ohlcv_data)

        for i in range(1, len(result.cross)):
            if result.cross.iloc[i]:
                assert result.cross_up.iloc[i] or result.cross_down.iloc[i], \
                    f"Bar {i}: cross=True but neither cross_up nor cross_down"

    def test_cross_up_when_momentum_positive(self, sample_ohlcv_data, wavetrend):
        """cross_up should occur when momentum goes from negative to positive."""
        result = wavetrend.calculate(sample_ohlcv_data)

        for i in range(1, len(result.cross_up)):
            if result.cross_up.iloc[i]:
                # Momentum should be positive (WT1 > WT2)
                assert result.momentum.iloc[i] > 0 or np.isnan(result.momentum.iloc[i-1])

    def test_cross_down_when_momentum_negative(self, sample_ohlcv_data, wavetrend):
        """cross_down should occur when momentum goes from positive to negative."""
        result = wavetrend.calculate(sample_ohlcv_data)

        for i in range(1, len(result.cross_down)):
            if result.cross_down.iloc[i]:
                # Momentum should be negative (WT1 < WT2)
                assert result.momentum.iloc[i] < 0 or np.isnan(result.momentum.iloc[i-1])


class TestOverboughtOversoldLevels:
    """Test overbought/oversold level detection."""

    def test_oversold_when_below_threshold(self, oversold_data, wavetrend):
        """oversold should be True when WT2 <= oversold_1."""
        result = wavetrend.calculate(oversold_data)

        # Check consistency: oversold True iff WT2 <= -53
        for i in range(len(result.oversold)):
            if not np.isnan(result.wt2.iloc[i]):
                expected = result.wt2.iloc[i] <= wavetrend.oversold_1
                assert result.oversold.iloc[i] == expected, \
                    f"Bar {i}: WT2={result.wt2.iloc[i]}, oversold={result.oversold.iloc[i]}"

    def test_overbought_when_above_threshold(self, overbought_data, wavetrend):
        """overbought should be True when WT2 >= overbought_1."""
        result = wavetrend.calculate(overbought_data)

        # Check consistency: overbought True iff WT2 >= 53
        for i in range(len(result.overbought)):
            if not np.isnan(result.wt2.iloc[i]):
                expected = result.wt2.iloc[i] >= wavetrend.overbought_1
                assert result.overbought.iloc[i] == expected, \
                    f"Bar {i}: WT2={result.wt2.iloc[i]}, overbought={result.overbought.iloc[i]}"


class TestAnchorLevelMethods:
    """Test anchor level helper methods."""

    def test_is_anchor_long(self, wavetrend):
        """is_anchor_long should return True when WT2 <= -60."""
        # At anchor level
        assert wavetrend.is_anchor_long(-60)

        # Below anchor level
        assert wavetrend.is_anchor_long(-65)
        assert wavetrend.is_anchor_long(-100)

        # Above anchor level
        assert not wavetrend.is_anchor_long(-55)
        assert not wavetrend.is_anchor_long(0)
        assert not wavetrend.is_anchor_long(60)

    def test_is_anchor_short(self, wavetrend):
        """is_anchor_short should return True when WT2 >= 60."""
        # At anchor level
        assert wavetrend.is_anchor_short(60)

        # Above anchor level
        assert wavetrend.is_anchor_short(65)
        assert wavetrend.is_anchor_short(100)

        # Below anchor level
        assert not wavetrend.is_anchor_short(55)
        assert not wavetrend.is_anchor_short(0)
        assert not wavetrend.is_anchor_short(-60)


class TestMomentumCrossedMethods:
    """Test momentum crossed up/down methods."""

    def test_momentum_crossed_up(self, wavetrend):
        """momentum_crossed_up should detect positive crossing."""
        # Test with two sequential values
        momentum_prev = -5
        momentum_curr = 5

        # If method takes two args (current and previous)
        crossed = wavetrend.momentum_crossed_up(momentum_curr, momentum_prev)
        assert isinstance(crossed, bool)
        assert crossed is True  # Crossed from negative to positive

    def test_momentum_crossed_down(self, wavetrend):
        """momentum_crossed_down should detect negative crossing."""
        momentum_prev = 5
        momentum_curr = -5

        crossed = wavetrend.momentum_crossed_down(momentum_curr, momentum_prev)
        assert isinstance(crossed, bool)
        assert crossed is True  # Crossed from positive to negative


class TestCalculateWavetrendFunction:
    """Test the calculate_wavetrend convenience function."""

    def test_convenience_function_returns_tuple(self, sample_ohlcv_data):
        """calculate_wavetrend should return tuple of (wt1, wt2, momentum)."""
        # Using convenience function - returns tuple not WaveTrendResult
        result = calculate_wavetrend(sample_ohlcv_data)

        # Should be a tuple of 3 Series
        assert isinstance(result, tuple)
        assert len(result) == 3

        wt1, wt2, momentum = result
        assert isinstance(wt1, pd.Series)
        assert isinstance(wt2, pd.Series)
        assert isinstance(momentum, pd.Series)

    def test_convenience_function_matches_class(self, sample_ohlcv_data):
        """calculate_wavetrend values should match class method."""
        # Using convenience function
        wt1, wt2, momentum = calculate_wavetrend(sample_ohlcv_data)

        # Using class directly
        wt = WaveTrend()
        result2 = wt.calculate(sample_ohlcv_data)

        # Compare results
        np.testing.assert_array_almost_equal(
            wt1.values, result2.wt1.values
        )
        np.testing.assert_array_almost_equal(
            wt2.values, result2.wt2.values
        )

    def test_custom_parameters(self, sample_ohlcv_data):
        """calculate_wavetrend should accept custom parameters."""
        result = calculate_wavetrend(
            sample_ohlcv_data,
            channel_len=7,
            average_len=10,
            ma_len=5
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert len(result[0]) == len(sample_ohlcv_data)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimum_data_points(self, wavetrend):
        """Should handle minimum data gracefully."""
        # Very small dataset
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5]
        })

        result = wavetrend.calculate(df)

        # Should return result with same length as input
        assert len(result.wt1) == 3

        # With small data, wt2 will have NaN (needs 3 bars for SMA)
        # wt1 can be calculated but wt2 and momentum may have NaN
        assert len(result.wt2) == 3

    def test_index_preserved(self, sample_ohlcv_data, wavetrend):
        """Result index should match input index."""
        result = wavetrend.calculate(sample_ohlcv_data)

        pd.testing.assert_index_equal(
            result.wt1.index,
            sample_ohlcv_data.index
        )
        pd.testing.assert_index_equal(
            result.wt2.index,
            sample_ohlcv_data.index
        )

    def test_missing_column_raises(self, wavetrend):
        """Should raise error for missing required columns."""
        df = pd.DataFrame({
            'open': [100, 101],
            'high': [101, 102]
            # Missing 'low' and 'close'
        })

        with pytest.raises(Exception):
            wavetrend.calculate(df)
