"""Unit tests for signal detection."""

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

from indicators.wavetrend import WaveTrend
from indicators.money_flow import MoneyFlow
from indicators.heikin_ashi import convert_to_heikin_ashi
from strategy.signals import SignalDetector, SignalState, SignalType, Signal


@pytest.fixture
def sample_data_with_signals():
    """Generate sample data that should produce signals."""
    np.random.seed(42)
    n = 300

    timestamps = pd.date_range(
        start=datetime.utcnow() - timedelta(hours=n * 4),
        periods=n,
        freq='4H'
    )

    # Generate price data with some trends
    base = 50000
    trend = np.zeros(n)

    # Create a downtrend then uptrend (good for long signal)
    trend[:100] = np.linspace(0, -1000, 100)
    trend[100:150] = np.linspace(-1000, -500, 50)
    trend[150:] = np.linspace(-500, 500, 150)

    prices = base + trend + np.random.randn(n) * 50

    df = pd.DataFrame({
        'open': prices + np.random.randn(n) * 10,
        'high': prices + np.abs(np.random.randn(n)) * 50,
        'low': prices - np.abs(np.random.randn(n)) * 50,
        'close': prices,
        'volume': np.random.rand(n) * 1000000
    }, index=timestamps)

    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


class TestSignalDetector:
    """Tests for SignalDetector."""

    def test_initial_state(self):
        """Test detector starts in IDLE state."""
        detector = SignalDetector()

        state = detector.get_current_state()
        assert state['long_state'] == 'IDLE'
        assert state['short_state'] == 'IDLE'

    def test_reset(self):
        """Test reset clears state."""
        detector = SignalDetector()
        detector._long_state = SignalState.ANCHOR_DETECTED
        detector.reset()

        assert detector._long_state == SignalState.IDLE
        assert detector._short_state == SignalState.IDLE

    def test_anchor_detection_long(self):
        """Test long anchor wave detection."""
        detector = SignalDetector(anchor_level_long=-60)

        # Simulate WT2 going below -60
        assert detector._long_state == SignalState.IDLE

        # Check is_anchor_long
        from indicators.wavetrend import WaveTrend
        wt = WaveTrend(oversold_2=-60)
        assert wt.is_anchor_long(-65)
        assert not wt.is_anchor_long(-55)

    def test_anchor_detection_short(self):
        """Test short anchor wave detection."""
        detector = SignalDetector(anchor_level_short=60)

        from indicators.wavetrend import WaveTrend
        wt = WaveTrend(overbought_2=60)
        assert wt.is_anchor_short(65)
        assert not wt.is_anchor_short(55)

    def test_signal_history(self, sample_data_with_signals):
        """Test signal history is maintained."""
        detector = SignalDetector()

        # Process data
        ha_df = convert_to_heikin_ashi(sample_data_with_signals)
        wt = WaveTrend()
        mf = MoneyFlow()

        wt_result = wt.calculate(ha_df)
        mf_result = mf.calculate(ha_df)

        for i in range(100, len(ha_df)):
            # Create subset results for this bar
            wt_subset = type(wt_result)(
                wt1=wt_result.wt1.iloc[:i+1],
                wt2=wt_result.wt2.iloc[:i+1],
                momentum=wt_result.momentum.iloc[:i+1],
                cross=wt_result.cross.iloc[:i+1],
                cross_up=wt_result.cross_up.iloc[:i+1],
                cross_down=wt_result.cross_down.iloc[:i+1],
                oversold=wt_result.oversold.iloc[:i+1],
                overbought=wt_result.overbought.iloc[:i+1]
            )

            mf_subset = type(mf_result)(
                mfi=mf_result.mfi.iloc[:i+1],
                is_positive=mf_result.is_positive.iloc[:i+1],
                is_negative=mf_result.is_negative.iloc[:i+1],
                curving_up=mf_result.curving_up.iloc[:i+1],
                curving_down=mf_result.curving_down.iloc[:i+1]
            )

            signal = detector.process_bar(
                timestamp=ha_df.index[i],
                close_price=ha_df['close'].iloc[i],
                wt_result=wt_subset,
                mf_result=mf_subset,
                bar_idx=i
            )

        # History should contain any generated signals
        history = detector.signals_history
        # Even if no signals, history should be a list
        assert isinstance(history, list)


class TestSignalType:
    """Tests for Signal dataclass."""

    def test_signal_to_dict(self):
        """Test signal serialization."""
        from strategy.signals import AnchorWave, TriggerWave

        anchor = AnchorWave(
            timestamp=datetime.utcnow(),
            wt2_value=-65,
            bar_index=100,
            signal_type=SignalType.LONG
        )

        trigger = TriggerWave(
            timestamp=datetime.utcnow(),
            wt2_value=-50,
            bar_index=110,
            has_cross=False
        )

        signal = Signal(
            signal_type=SignalType.LONG,
            timestamp=datetime.utcnow(),
            entry_price=50000,
            anchor_wave=anchor,
            trigger_wave=trigger,
            wt1=-30,
            wt2=-25,
            vwap=5,
            mfi=-2
        )

        d = signal.to_dict()

        assert d['signal_type'] == 'long'
        assert d['entry_price'] == 50000
        assert d['wt1'] == -30
        assert d['anchor_wt2'] == -65


class TestSignalStates:
    """Tests for signal state transitions."""

    def test_state_enum_values(self):
        """Test SignalState enum has expected values."""
        assert SignalState.IDLE.value == 1
        assert SignalState.ANCHOR_DETECTED.value == 2
        assert SignalState.TRIGGER_DETECTED.value == 3
        assert SignalState.AWAITING_MFI.value == 4
        assert SignalState.AWAITING_MOMENTUM_CROSS.value == 5
        assert SignalState.SIGNAL_READY.value == 6

    def test_signal_type_enum(self):
        """Test SignalType enum values."""
        assert SignalType.LONG.value == "long"
        assert SignalType.SHORT.value == "short"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
