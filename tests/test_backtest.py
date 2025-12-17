"""Unit tests for backtesting engine."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from backtest.engine import BacktestEngine
from backtest.data_loader import DataLoader
from backtest.metrics import calculate_metrics, plot_equity_curve


def generate_2025_sample_data(num_candles: int = 2628) -> pd.DataFrame:
    """
    Generate sample BTC-like 4H OHLCV data for 2025.

    2628 candles = ~365 days * 24 hours / 4 hours

    Args:
        num_candles: Number of 4H candles to generate

    Returns:
        DataFrame with realistic price movements
    """
    np.random.seed(2025)  # Reproducible for 2025

    # Start from Jan 1, 2025
    start_time = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    timestamps = pd.date_range(
        start=start_time,
        periods=num_candles,
        freq='4h'
    )

    # Generate realistic BTC-like price movements
    # Start around $95,000 (realistic for 2025)
    base_price = 95000

    # Create trending + mean-reverting behavior
    trend = np.zeros(num_candles)

    # Add multiple market phases
    # Phase 1: Consolidation (Jan-Feb)
    phase1_end = int(num_candles * 0.17)
    trend[:phase1_end] = np.random.randn(phase1_end).cumsum() * 50

    # Phase 2: Uptrend (Mar-May)
    phase2_end = int(num_candles * 0.42)
    trend[phase1_end:phase2_end] = (
        trend[phase1_end-1] +
        np.linspace(0, 15000, phase2_end - phase1_end) +
        np.random.randn(phase2_end - phase1_end).cumsum() * 100
    )

    # Phase 3: Correction (Jun-Jul)
    phase3_end = int(num_candles * 0.58)
    trend[phase2_end:phase3_end] = (
        trend[phase2_end-1] +
        np.linspace(0, -8000, phase3_end - phase2_end) +
        np.random.randn(phase3_end - phase2_end).cumsum() * 80
    )

    # Phase 4: Recovery and new highs (Aug-Oct)
    phase4_end = int(num_candles * 0.83)
    trend[phase3_end:phase4_end] = (
        trend[phase3_end-1] +
        np.linspace(0, 20000, phase4_end - phase3_end) +
        np.random.randn(phase4_end - phase3_end).cumsum() * 120
    )

    # Phase 5: Year-end consolidation (Nov-Dec)
    trend[phase4_end:] = (
        trend[phase4_end-1] +
        np.random.randn(num_candles - phase4_end).cumsum() * 60
    )

    # Add volatility clusters
    volatility = np.ones(num_candles) * 0.015  # Base 1.5% per 4H

    # Higher volatility during trend changes
    for phase_point in [phase1_end, phase2_end, phase3_end, phase4_end]:
        start = max(0, phase_point - 50)
        end = min(num_candles, phase_point + 50)
        volatility[start:end] *= 1.5

    # Generate OHLC
    close_prices = base_price + trend

    # Ensure no negative prices
    close_prices = np.maximum(close_prices, 10000)

    # Generate realistic OHLC from close
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]

    # Add intrabar movement
    bar_range = close_prices * volatility

    high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.randn(num_candles)) * bar_range * 0.5
    low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.randn(num_candles)) * bar_range * 0.5

    # Volume correlated with volatility
    volume = (np.random.rand(num_candles) + 0.5) * volatility * close_prices * 100

    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=timestamps)

    df.index.name = 'timestamp'

    return df


class TestBacktestEngine:
    """Tests for BacktestEngine."""

    def test_backtest_with_sample_data(self):
        """Test backtest runs successfully with sample data."""
        # Generate 2025 sample data
        df = generate_2025_sample_data(1000)  # ~166 days

        # Create backtest engine
        engine = BacktestEngine(
            initial_balance=10000.0,
            risk_percent=3.0,
            risk_reward=2.0,
            commission_percent=0.06,
            anchor_level_long=-60,
            anchor_level_short=60
        )

        # Run backtest
        result = engine.run(df)

        # Basic assertions
        assert result.initial_balance == 10000.0
        assert result.total_trades >= 0
        assert len(result.equity_curve) > 0

        # Print results for visibility
        print(f"\n{'='*50}")
        print("BACKTEST RESULTS: Sample 2025 BTC Data")
        print(f"{'='*50}")
        print(f"Period:            {df.index[0].date()} to {df.index[-1].date()}")
        print(f"Candles:           {len(df)}")
        print(f"Initial Balance:   ${result.initial_balance:,.2f}")
        print(f"Final Balance:     ${result.final_balance:,.2f}")
        print(f"Total PnL:         ${result.total_pnl:,.2f} ({result.total_pnl_percent:.2f}%)")
        print(f"Max Drawdown:      {result.max_drawdown_percent:.2f}%")
        print(f"Total Trades:      {result.total_trades}")
        print(f"Win Rate:          {result.win_rate:.1f}%")
        print(f"Profit Factor:     {result.profit_factor:.2f}")
        print(f"Sharpe Ratio:      {result.sharpe_ratio:.2f}")

        if result.trades:
            print(f"\nTrade Details:")
            for i, trade in enumerate(result.trades[:5]):  # Show first 5 trades
                print(f"  Trade {i+1}: {trade.signal_type.value.upper()} "
                      f"Entry=${trade.entry_price:,.2f} "
                      f"Exit=${trade.exit_price:,.2f} "
                      f"PnL=${trade.pnl:,.2f} ({trade.exit_reason})")

    def test_backtest_full_year_2025(self):
        """Test full year 2025 backtest."""
        # Generate full year of 4H data
        df = generate_2025_sample_data(2628)

        engine = BacktestEngine(
            initial_balance=10000.0,
            risk_percent=3.0,
            risk_reward=2.0
        )

        result = engine.run(df)

        print(f"\n{'='*50}")
        print("FULL YEAR 2025 BACKTEST RESULTS")
        print(f"{'='*50}")
        print(f"Period:            Jan 1, 2025 - Dec 31, 2025")
        print(f"Initial Balance:   ${result.initial_balance:,.2f}")
        print(f"Final Balance:     ${result.final_balance:,.2f}")
        print(f"Total PnL:         ${result.total_pnl:,.2f} ({result.total_pnl_percent:.2f}%)")
        print(f"Max Drawdown:      {result.max_drawdown_percent:.2f}%")
        print(f"Total Trades:      {result.total_trades}")
        print(f"Win Rate:          {result.win_rate:.1f}%")
        print(f"Profit Factor:     {result.profit_factor:.2f}")

        # Calculate additional metrics
        if result.trades:
            metrics = calculate_metrics(result.equity_curve, result.trades)
            print(f"\nAdvanced Metrics:")
            print(f"Annualized Return: {metrics.annualized_return:.2f}%")
            print(f"Sortino Ratio:     {metrics.sortino_ratio:.2f}")
            print(f"Expectancy:        ${metrics.expectancy:.2f} per trade")

    def test_no_trades_scenario(self):
        """Test backtest with data that generates no signals."""
        # Create flat data with no volatility
        df = pd.DataFrame({
            'open': [100] * 200,
            'high': [101] * 200,
            'low': [99] * 200,
            'close': [100] * 200,
            'volume': [1000] * 200
        }, index=pd.date_range(start='2025-01-01', periods=200, freq='4h'))

        engine = BacktestEngine()
        result = engine.run(df)

        # With flat data, should have no trades
        assert result.total_trades == 0
        assert result.final_balance == result.initial_balance


class TestDataLoader:
    """Tests for DataLoader."""

    def test_generate_sample_data(self):
        """Test sample data generation."""
        df = DataLoader.generate_sample_data(num_candles=500)

        assert len(df) == 500
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        assert df['high'].max() > df['low'].min()

    def test_load_from_csv(self, tmp_path):
        """Test CSV loading."""
        # Create test CSV
        df = generate_2025_sample_data(100)
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path)

        # Load it back
        loader = DataLoader()
        loaded_df = loader.load_from_csv(str(csv_path))

        assert len(loaded_df) == len(df)
        assert all(col in loaded_df.columns for col in ['open', 'high', 'low', 'close'])


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
