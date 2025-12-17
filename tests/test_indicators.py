"""Unit tests for indicator calculations."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from indicators.heikin_ashi import HeikinAshi, convert_to_heikin_ashi
from indicators.wavetrend import WaveTrend, calculate_wavetrend
from indicators.money_flow import MoneyFlow, calculate_money_flow


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 200

    timestamps = pd.date_range(
        start=datetime.utcnow() - timedelta(hours=n * 4),
        periods=n,
        freq='4H'
    )

    # Generate trending price data
    trend = np.cumsum(np.random.randn(n) * 0.5) + 50000

    df = pd.DataFrame({
        'open': trend + np.random.randn(n) * 100,
        'high': trend + np.abs(np.random.randn(n)) * 200,
        'low': trend - np.abs(np.random.randn(n)) * 200,
        'close': trend + np.random.randn(n) * 100,
        'volume': np.random.rand(n) * 1000000
    }, index=timestamps)

    # Ensure high >= open, close and low <= open, close
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


class TestHeikinAshi:
    """Tests for Heikin Ashi converter."""

    def test_convert_dataframe(self, sample_ohlcv_data):
        """Test DataFrame conversion."""
        ha_df = convert_to_heikin_ashi(sample_ohlcv_data)

        assert len(ha_df) == len(sample_ohlcv_data)
        assert all(col in ha_df.columns for col in ['open', 'high', 'low', 'close'])

    def test_ha_high_always_highest(self, sample_ohlcv_data):
        """Test that HA high is always the maximum of high, open, close."""
        ha_df = convert_to_heikin_ashi(sample_ohlcv_data)

        for i in range(len(ha_df)):
            assert ha_df['high'].iloc[i] >= ha_df['open'].iloc[i]
            assert ha_df['high'].iloc[i] >= ha_df['close'].iloc[i]

    def test_ha_low_always_lowest(self, sample_ohlcv_data):
        """Test that HA low is always the minimum of low, open, close."""
        ha_df = convert_to_heikin_ashi(sample_ohlcv_data)

        for i in range(len(ha_df)):
            assert ha_df['low'].iloc[i] <= ha_df['open'].iloc[i]
            assert ha_df['low'].iloc[i] <= ha_df['close'].iloc[i]

    def test_ha_close_formula(self, sample_ohlcv_data):
        """Test HA close = (O+H+L+C)/4."""
        df = sample_ohlcv_data
        ha_df = convert_to_heikin_ashi(df)

        expected_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4

        np.testing.assert_array_almost_equal(
            ha_df['close'].values,
            expected_close.values,
            decimal=10
        )

    def test_single_candle_conversion(self):
        """Test single candle conversion."""
        ha = HeikinAshi()

        candle = ha.convert_single(
            open_price=100,
            high=105,
            low=95,
            close=102,
            volume=1000
        )

        # HA Close = (100 + 105 + 95 + 102) / 4 = 100.5
        assert candle.close == pytest.approx(100.5)

        # First candle HA Open = (100 + 102) / 2 = 101
        assert candle.open == pytest.approx(101)


class TestWaveTrend:
    """Tests for WaveTrend oscillator."""

    def test_wavetrend_output_shape(self, sample_ohlcv_data):
        """Test WaveTrend output has correct shape."""
        wt = WaveTrend()
        result = wt.calculate(sample_ohlcv_data)

        assert len(result.wt1) == len(sample_ohlcv_data)
        assert len(result.wt2) == len(sample_ohlcv_data)
        assert len(result.vwap) == len(sample_ohlcv_data)

    def test_wt_vwap_calculation(self, sample_ohlcv_data):
        """Test VWAP = WT1 - WT2."""
        wt = WaveTrend()
        result = wt.calculate(sample_ohlcv_data)

        expected_vwap = result.wt1 - result.wt2

        np.testing.assert_array_almost_equal(
            result.vwap.values,
            expected_vwap.values,
            decimal=10
        )

    def test_wt_cross_detection(self, sample_ohlcv_data):
        """Test cross detection logic."""
        wt = WaveTrend()
        result = wt.calculate(sample_ohlcv_data)

        # Cross should be True when either cross_up or cross_down
        for i in range(1, len(result.cross)):
            if result.cross.iloc[i]:
                assert result.cross_up.iloc[i] or result.cross_down.iloc[i]

    def test_overbought_oversold_levels(self, sample_ohlcv_data):
        """Test overbought/oversold detection."""
        wt = WaveTrend(overbought_1=53, oversold_1=-53)
        result = wt.calculate(sample_ohlcv_data)

        # Check oversold detection
        for i in range(len(result.wt2)):
            if result.wt2.iloc[i] <= -53:
                assert result.oversold.iloc[i]
            if result.wt2.iloc[i] >= 53:
                assert result.overbought.iloc[i]

    def test_anchor_level_detection(self, sample_ohlcv_data):
        """Test anchor level detection methods."""
        wt = WaveTrend(overbought_2=60, oversold_2=-60)

        assert wt.is_anchor_long(-65)
        assert not wt.is_anchor_long(-55)

        assert wt.is_anchor_short(65)
        assert not wt.is_anchor_short(55)

    def test_convenience_function(self, sample_ohlcv_data):
        """Test calculate_wavetrend convenience function."""
        wt1, wt2, vwap = calculate_wavetrend(sample_ohlcv_data)

        assert len(wt1) == len(sample_ohlcv_data)
        assert len(wt2) == len(sample_ohlcv_data)
        assert len(vwap) == len(sample_ohlcv_data)


class TestMoneyFlow:
    """Tests for Money Flow indicator."""

    def test_money_flow_output_shape(self, sample_ohlcv_data):
        """Test Money Flow output has correct shape."""
        mf = MoneyFlow()
        result = mf.calculate(sample_ohlcv_data)

        assert len(result.mfi) == len(sample_ohlcv_data)
        assert len(result.is_positive) == len(sample_ohlcv_data)
        assert len(result.is_negative) == len(sample_ohlcv_data)

    def test_positive_negative_exclusive(self, sample_ohlcv_data):
        """Test that positive and negative are mutually exclusive."""
        mf = MoneyFlow()
        result = mf.calculate(sample_ohlcv_data)

        for i in range(len(result.mfi)):
            # Cannot be both positive and negative
            assert not (result.is_positive.iloc[i] and result.is_negative.iloc[i])

    def test_mfi_value_range(self, sample_ohlcv_data):
        """Test MFI values are within reasonable range."""
        mf = MoneyFlow()
        result = mf.calculate(sample_ohlcv_data)

        # MFI should generally be within -100 to 100 range
        # (allowing some tolerance for extreme values)
        assert result.mfi.max() < 200
        assert result.mfi.min() > -200

    def test_curving_detection(self, sample_ohlcv_data):
        """Test MFI curving detection."""
        mf = MoneyFlow()

        # Create scenario where MFI curves up
        assert mf.is_curving_up(mfi_current=-5, mfi_prev=-10, mfi_prev2=-8)
        assert not mf.is_curving_up(mfi_current=-15, mfi_prev=-10, mfi_prev2=-8)

        # Create scenario where MFI curves down
        assert mf.is_curving_down(mfi_current=5, mfi_prev=10, mfi_prev2=8)
        assert not mf.is_curving_down(mfi_current=15, mfi_prev=10, mfi_prev2=8)

    def test_convenience_function(self, sample_ohlcv_data):
        """Test calculate_money_flow convenience function."""
        mfi = calculate_money_flow(sample_ohlcv_data)

        assert len(mfi) == len(sample_ohlcv_data)


class TestIndicatorIntegration:
    """Integration tests for indicator pipeline."""

    def test_full_pipeline(self, sample_ohlcv_data):
        """Test full indicator calculation pipeline."""
        # Convert to Heikin Ashi
        ha_df = convert_to_heikin_ashi(sample_ohlcv_data)

        # Calculate WaveTrend
        wt = WaveTrend()
        wt_result = wt.calculate(ha_df)

        # Calculate Money Flow
        mf = MoneyFlow()
        mf_result = mf.calculate(ha_df)

        # Verify all outputs are valid
        assert not wt_result.wt1.isna().all()
        assert not wt_result.wt2.isna().all()
        assert not mf_result.mfi.isna().all()

    def test_indicators_with_edge_cases(self):
        """Test indicators handle edge cases."""
        # Create DataFrame with some edge cases
        df = pd.DataFrame({
            'open': [100, 100, 100, 100, 100],  # Flat prices
            'high': [100, 100, 100, 100, 100],
            'low': [100, 100, 100, 100, 100],
            'close': [100, 100, 100, 100, 100],
            'volume': [1000, 1000, 1000, 1000, 1000]
        })

        # Should not raise errors
        ha_df = convert_to_heikin_ashi(df)
        assert len(ha_df) == 5

        mf = MoneyFlow()
        result = mf.calculate(df)
        assert len(result.mfi) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
