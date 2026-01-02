"""
Integration tests for data feed validation.

Tests verify:
1. Full indicator pipeline (OHLCV -> HA -> WT -> MFI)
2. VWAP daily reset logic
3. Data quality and handling edge cases
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from indicators.heikin_ashi import convert_to_heikin_ashi
from indicators.wavetrend import WaveTrend
from indicators.money_flow import MoneyFlow
from indicators.vwap import VWAPCalculator


class TestIndicatorPipeline:
    """Test the full indicator calculation pipeline."""

    def test_full_pipeline_produces_valid_output(self, sample_ohlcv_data):
        """Full pipeline: OHLCV -> HA -> WT -> MFI -> valid output."""
        # Step 1: Heikin Ashi conversion
        ha_df = convert_to_heikin_ashi(sample_ohlcv_data)

        assert len(ha_df) == len(sample_ohlcv_data)
        assert not ha_df['close'].isna().any()
        assert not ha_df['open'].isna().any()

        # Step 2: WaveTrend calculation
        wt = WaveTrend()
        wt_result = wt.calculate(ha_df)

        assert len(wt_result.wt1) == len(ha_df)
        # After warmup period (20 bars), should have valid values
        assert not wt_result.wt1.iloc[50:].isna().any()
        assert not wt_result.wt2.iloc[50:].isna().any()

        # Step 3: MFI calculation
        mf = MoneyFlow()
        mf_result = mf.calculate(ha_df)

        assert len(mf_result.mfi) == len(ha_df)
        # After 60-bar warmup, should have valid values
        assert not mf_result.mfi.iloc[65:].isna().any()

    def test_pipeline_handles_missing_data(self):
        """Pipeline should handle missing/NaN data gracefully."""
        n = 100
        timestamps = pd.date_range(
            start=datetime.now(timezone.utc) - timedelta(hours=n * 4),
            periods=n,
            freq='4h'
        )

        # Create data with some NaN values
        df = pd.DataFrame({
            'open': np.random.rand(n) * 100 + 90000,
            'high': np.random.rand(n) * 100 + 90100,
            'low': np.random.rand(n) * 100 + 89900,
            'close': np.random.rand(n) * 100 + 90000,
            'volume': np.random.rand(n) * 1000000
        }, index=timestamps)

        # Insert some NaN values
        df.loc[df.index[10], 'close'] = np.nan
        df.loc[df.index[20], 'volume'] = np.nan

        # Forward fill NaN values (standard preprocessing)
        df_clean = df.ffill().bfill()

        # Should handle without crashing
        ha_df = convert_to_heikin_ashi(df_clean)

        wt = WaveTrend()
        wt_result = wt.calculate(ha_df)

        # May have some NaN from warmup but shouldn't crash
        assert len(wt_result.wt1) == n

    def test_pipeline_output_alignment(self, sample_ohlcv_data):
        """All indicator outputs should be aligned (same index)."""
        ha_df = convert_to_heikin_ashi(sample_ohlcv_data)

        wt = WaveTrend()
        wt_result = wt.calculate(ha_df)

        mf = MoneyFlow()
        mf_result = mf.calculate(ha_df)

        # All should have same index
        pd.testing.assert_index_equal(wt_result.wt1.index, ha_df.index)
        pd.testing.assert_index_equal(mf_result.mfi.index, ha_df.index)


class TestVWAPDailyReset:
    """Test VWAP daily reset logic."""

    def test_vwap_resets_at_midnight_utc(self):
        """VWAP should reset at 00:00 UTC each day."""
        # Create data spanning multiple days
        n = 48  # 48 hours = 2 days at 1h intervals
        timestamps = pd.date_range(
            start='2025-01-01 00:00:00',
            periods=n,
            freq='1h',
            tz='UTC'
        )

        # Linearly increasing prices
        df = pd.DataFrame({
            'high': np.linspace(100, 150, n),
            'low': np.linspace(98, 148, n),
            'close': np.linspace(99, 149, n),
            'volume': [1000] * n
        }, index=timestamps)

        calc = VWAPCalculator()
        result = calc.calculate(df)

        # VWAP at first bar of each day should equal typical price
        # (since cumulative values reset)
        day1_first = 0
        day2_first = 24

        typical_price_day1 = (df['high'].iloc[day1_first] +
                             df['low'].iloc[day1_first] +
                             df['close'].iloc[day1_first]) / 3

        typical_price_day2 = (df['high'].iloc[day2_first] +
                             df['low'].iloc[day2_first] +
                             df['close'].iloc[day2_first]) / 3

        # First bar of each day: VWAP = typical price
        assert result.vwap.iloc[day1_first] == pytest.approx(typical_price_day1, rel=0.01)
        assert result.vwap.iloc[day2_first] == pytest.approx(typical_price_day2, rel=0.01)

    def test_vwap_handles_zero_volume(self):
        """Zero volume bars should not break VWAP calculation."""
        timestamps = pd.date_range(
            start='2025-01-01 00:00:00',
            periods=5,
            freq='1h',
            tz='UTC'
        )

        df = pd.DataFrame({
            'high': [100, 101, 102, 103, 104],
            'low': [99, 100, 101, 102, 103],
            'close': [99.5, 100.5, 101.5, 102.5, 103.5],
            'volume': [1000, 0, 1000, 0, 1000]  # Zero volume in some bars
        }, index=timestamps)

        calc = VWAPCalculator()
        result = calc.calculate(df)

        # Should forward-fill VWAP for zero volume bars
        assert not result.vwap.isna().any(), "VWAP should handle zero volume"

    def test_vwap_cumulative_within_day(self):
        """VWAP should be cumulative within a single day."""
        timestamps = pd.date_range(
            start='2025-01-01 00:00:00',
            periods=10,
            freq='1h',
            tz='UTC'
        )

        df = pd.DataFrame({
            'high': [100] * 10,
            'low': [100] * 10,
            'close': [100] * 10,
            'volume': [100] * 10
        }, index=timestamps)

        calc = VWAPCalculator()
        result = calc.calculate(df)

        # With constant prices and volume, VWAP should be constant
        assert result.vwap.std() == pytest.approx(0, abs=0.01)


class TestHeikinAshiConversion:
    """Test Heikin Ashi conversion."""

    def test_ha_close_formula(self, sample_ohlcv_data):
        """HA Close = (O + H + L + C) / 4."""
        ha_df = convert_to_heikin_ashi(sample_ohlcv_data)

        # HA Close should be (O + H + L + C) / 4 for each bar
        expected_close = (sample_ohlcv_data['open'] +
                         sample_ohlcv_data['high'] +
                         sample_ohlcv_data['low'] +
                         sample_ohlcv_data['close']) / 4

        np.testing.assert_array_almost_equal(
            ha_df['close'].values,
            expected_close.values,
            decimal=10
        )

    def test_ha_high_is_max(self, sample_ohlcv_data):
        """HA High should be max of (High, HA Open, HA Close)."""
        ha_df = convert_to_heikin_ashi(sample_ohlcv_data)

        for i in range(len(ha_df)):
            ha_high = ha_df['high'].iloc[i]
            raw_high = sample_ohlcv_data['high'].iloc[i]
            ha_open = ha_df['open'].iloc[i]
            ha_close = ha_df['close'].iloc[i]

            expected = max(raw_high, ha_open, ha_close)
            assert ha_high == pytest.approx(expected, rel=1e-10)

    def test_ha_low_is_min(self, sample_ohlcv_data):
        """HA Low should be min of (Low, HA Open, HA Close)."""
        ha_df = convert_to_heikin_ashi(sample_ohlcv_data)

        for i in range(len(ha_df)):
            ha_low = ha_df['low'].iloc[i]
            raw_low = sample_ohlcv_data['low'].iloc[i]
            ha_open = ha_df['open'].iloc[i]
            ha_close = ha_df['close'].iloc[i]

            expected = min(raw_low, ha_open, ha_close)
            assert ha_low == pytest.approx(expected, rel=1e-10)


class TestMoneyFlow:
    """Test MoneyFlow indicator."""

    def test_mfi_bounded(self, sample_ohlcv_data):
        """MFI values should be bounded."""
        mf = MoneyFlow()
        result = mf.calculate(sample_ohlcv_data)

        # After warmup, values should be reasonable
        valid_mfi = result.mfi.dropna()

        # MFI should be bounded (typically between -100 and 100)
        assert valid_mfi.max() < 200
        assert valid_mfi.min() > -200

    def test_doji_handling(self):
        """Doji candles (H == L) should be handled gracefully."""
        timestamps = pd.date_range(
            start='2025-01-01 00:00:00',
            periods=100,
            freq='1h',
            tz='UTC'
        )

        # Create data with doji candles
        df = pd.DataFrame({
            'open': [100] * 100,
            'high': [100] * 100,  # Same as low = doji
            'low': [100] * 100,
            'close': [100] * 100,
            'volume': [1000] * 100
        }, index=timestamps)

        mf = MoneyFlow()
        result = mf.calculate(df)

        # Should not crash, MFI should be 0 or NaN for doji
        assert len(result.mfi) == 100


class TestDataQuality:
    """Test data quality edge cases."""

    def test_handles_outliers(self, sample_ohlcv_data):
        """Pipeline should handle outlier values."""
        # Add an outlier
        df = sample_ohlcv_data.copy()
        df.loc[df.index[50], 'close'] = df['close'].iloc[50] * 10  # 10x spike

        ha_df = convert_to_heikin_ashi(df)
        wt = WaveTrend()
        result = wt.calculate(ha_df)

        # Should complete without error
        assert len(result.wt1) == len(df)

        # Values should still be reasonable (no Inf)
        assert not np.isinf(result.wt1.dropna()).any()

    def test_consistent_with_different_timeframes(self):
        """Same data at different timeframes should work."""
        for freq, n in [('5min', 288), ('15min', 96), ('1h', 24), ('4h', 50)]:
            timestamps = pd.date_range(
                start='2025-01-01 00:00:00',
                periods=n,
                freq=freq,
                tz='UTC'
            )

            df = pd.DataFrame({
                'open': np.random.rand(n) * 100 + 95000,
                'high': np.random.rand(n) * 100 + 95100,
                'low': np.random.rand(n) * 100 + 94900,
                'close': np.random.rand(n) * 100 + 95000,
                'volume': np.random.rand(n) * 1000000
            }, index=timestamps)

            # Fix high/low
            df['high'] = df[['open', 'high', 'close']].max(axis=1)
            df['low'] = df[['open', 'low', 'close']].min(axis=1)

            # Should work for all timeframes
            ha_df = convert_to_heikin_ashi(df)
            wt = WaveTrend()
            result = wt.calculate(ha_df)

            assert len(result.wt1) == n
