"""
Multi-Timeframe Synchronizer for VMC Backtesting.

Handles bar synchronization between LTF (entry timeframe) and HTF (bias timeframe).
Ensures that at any given LTF bar, we only use HTF data that would have been
available at that time (i.e., from CLOSED HTF bars).
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Generator, Tuple, Any
from dataclasses import dataclass

from indicators.wavetrend import WaveTrend, WaveTrendResult
from indicators.money_flow import MoneyFlow, MoneyFlowResult
from strategy.mtf_bias import MTFBiasCalculator, MTFBiasResult, HTFAnalysis
from strategy.signals import Bias, SignalType


# Timeframe to minutes mapping
TF_MINUTES = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
    "1D": 1440,
}


@dataclass
class HTFIndicators:
    """Pre-calculated indicators for a single HTF timeframe."""
    timeframe: str
    df: pd.DataFrame
    wt_result: WaveTrendResult
    mf_result: MoneyFlowResult


@dataclass
class SyncedBar:
    """A synchronized bar containing LTF data + HTF bias."""
    ltf_idx: int
    ltf_timestamp: datetime
    ltf_row: pd.Series
    htf_bias: MTFBiasResult
    htf_details: Dict[str, HTFAnalysis]


class MTFSynchronizer:
    """
    Synchronize LTF bars with HTF bias for backtesting.

    Key features:
    - Pre-calculates indicators for all HTF timeframes
    - Maps each LTF bar to the latest CLOSED HTF bar
    - Provides efficient iteration over synchronized bars
    """

    def __init__(
        self,
        ltf_data: pd.DataFrame,
        htf_data: Dict[str, pd.DataFrame],
        ltf_timeframe: str = "15m",
        htf_timeframes: Optional[List[str]] = None,
        wt_overbought: int = 60,
        wt_oversold: int = -60,
    ):
        """
        Initialize MTF synchronizer.

        Args:
            ltf_data: DataFrame with LTF OHLCV data (index must be datetime)
            htf_data: Dict of timeframe -> DataFrame for each HTF
            ltf_timeframe: LTF timeframe string (e.g., '15m')
            htf_timeframes: List of HTF timeframes to use (defaults to keys in htf_data)
            wt_overbought: WaveTrend overbought level
            wt_oversold: WaveTrend oversold level
        """
        self.ltf_data = ltf_data
        self.htf_data = htf_data
        self.ltf_timeframe = ltf_timeframe
        self.htf_timeframes = htf_timeframes or list(htf_data.keys())

        # Indicator parameters
        self.wt_overbought = wt_overbought
        self.wt_oversold = wt_oversold

        # Initialize indicators
        self.wt = WaveTrend(
            overbought_2=wt_overbought,
            oversold_2=wt_oversold
        )
        self.mf = MoneyFlow()

        # Bias calculator
        self.bias_calculator = MTFBiasCalculator()

        # Pre-calculate HTF indicators
        self.htf_indicators: Dict[str, HTFIndicators] = {}
        self._precalculate_htf_indicators()

        # Build timestamp mapping for each HTF
        self._htf_bar_indices: Dict[str, pd.DatetimeIndex] = {}
        for tf in self.htf_timeframes:
            if tf in self.htf_data:
                self._htf_bar_indices[tf] = self.htf_data[tf].index

    def _precalculate_htf_indicators(self) -> None:
        """Pre-calculate WaveTrend and MoneyFlow for all HTF data."""
        for tf in self.htf_timeframes:
            if tf not in self.htf_data:
                continue

            df = self.htf_data[tf]
            if len(df) < 100:
                continue

            try:
                wt_result = self.wt.calculate(df)
                mf_result = self.mf.calculate(df)

                self.htf_indicators[tf] = HTFIndicators(
                    timeframe=tf,
                    df=df,
                    wt_result=wt_result,
                    mf_result=mf_result
                )
            except Exception as e:
                print(f"Error calculating indicators for {tf}: {e}")

    def get_latest_closed_htf_bar(
        self,
        ltf_timestamp: datetime,
        htf_timeframe: str
    ) -> Optional[int]:
        """
        Get the index of the latest CLOSED HTF bar for a given LTF timestamp.

        Example: If LTF is 15m and HTF is 4h:
        - At 10:45, the latest closed 4h bar is 08:00 (not 12:00, which hasn't closed)
        - At 12:00, the latest closed 4h bar is 08:00 (12:00 just opened)
        - At 12:15, the latest closed 4h bar is 12:00

        Args:
            ltf_timestamp: The LTF bar timestamp
            htf_timeframe: The HTF timeframe string

        Returns:
            Index position of the latest closed HTF bar, or None if not available
        """
        if htf_timeframe not in self._htf_bar_indices:
            return None

        htf_index = self._htf_bar_indices[htf_timeframe]

        # Get HTF bar duration in minutes
        htf_minutes = TF_MINUTES.get(htf_timeframe, 240)

        # Find all HTF bars that CLOSED before or at the LTF timestamp
        # A bar at timestamp T closes at T + duration
        # So for a bar to be closed by LTF time X, we need: bar_time + duration <= X
        # Which means: bar_time <= X - duration

        # However, in typical OHLCV data, the timestamp is the OPEN time
        # So we want bars where: bar_open_time + duration <= ltf_timestamp
        # Equivalent to: bar_open_time <= ltf_timestamp - duration

        cutoff = ltf_timestamp - timedelta(minutes=htf_minutes)

        # Find bars that opened before or at the cutoff
        valid_bars = htf_index[htf_index <= cutoff]

        if len(valid_bars) == 0:
            return None

        # Return the index position of the latest valid bar
        latest_bar = valid_bars[-1]
        return htf_index.get_loc(latest_bar)

    def get_htf_indicators_at_bar(
        self,
        htf_timeframe: str,
        bar_idx: int
    ) -> Optional[Dict[str, float]]:
        """
        Get HTF indicator values at a specific bar index.

        Args:
            htf_timeframe: The HTF timeframe
            bar_idx: The bar index in HTF data

        Returns:
            Dict with wt2, mfi, mfi_prev, vwap values, or None if not available
        """
        if htf_timeframe not in self.htf_indicators:
            return None

        htf_ind = self.htf_indicators[htf_timeframe]

        if bar_idx < 0 or bar_idx >= len(htf_ind.df):
            return None

        try:
            return {
                "wt2": htf_ind.wt_result.wt2.iloc[bar_idx],
                "mfi": htf_ind.mf_result.mfi.iloc[bar_idx],
                "mfi_prev": htf_ind.mf_result.mfi.iloc[bar_idx - 1] if bar_idx > 0 else 0,
                "momentum": htf_ind.wt_result.momentum.iloc[bar_idx],
                "timestamp": htf_ind.df.index[bar_idx],
            }
        except Exception:
            return None

    def get_htf_bias_at_ltf_bar(
        self,
        ltf_timestamp: datetime
    ) -> MTFBiasResult:
        """
        Get combined HTF bias for a specific LTF bar timestamp.

        Args:
            ltf_timestamp: The LTF bar timestamp

        Returns:
            MTFBiasResult with overall bias and confidence
        """
        htf_analyses = []

        for tf in self.htf_timeframes:
            bar_idx = self.get_latest_closed_htf_bar(ltf_timestamp, tf)
            if bar_idx is None:
                continue

            indicators = self.get_htf_indicators_at_bar(tf, bar_idx)
            if indicators is None:
                continue

            # Analyze this HTF
            analysis = self.bias_calculator.analyze_single_timeframe(
                timeframe=tf,
                timestamp=indicators["timestamp"],
                wt2=indicators["wt2"],
                mfi=indicators["mfi"],
                mfi_prev=indicators["mfi_prev"],
                momentum=indicators["momentum"]
            )
            htf_analyses.append(analysis)

        # Combine into overall bias
        return self.bias_calculator.calculate_combined_bias(htf_analyses)

    def iterate_synced_bars(
        self,
        start_idx: int = 100,
        end_idx: Optional[int] = None
    ) -> Generator[SyncedBar, None, None]:
        """
        Iterate through LTF bars with synchronized HTF bias.

        Args:
            start_idx: Starting index in LTF data (default 100 for indicator warmup)
            end_idx: Ending index (default None = end of data)

        Yields:
            SyncedBar objects with LTF data and HTF bias
        """
        if end_idx is None:
            end_idx = len(self.ltf_data)

        for i in range(start_idx, end_idx):
            ltf_timestamp = self.ltf_data.index[i]
            ltf_row = self.ltf_data.iloc[i]

            # Get HTF bias
            htf_bias = self.get_htf_bias_at_ltf_bar(ltf_timestamp)

            # Get detailed HTF analyses
            htf_details = {}
            for analysis in htf_bias.htf_analyses:
                htf_details[analysis.timeframe] = analysis

            yield SyncedBar(
                ltf_idx=i,
                ltf_timestamp=ltf_timestamp,
                ltf_row=ltf_row,
                htf_bias=htf_bias,
                htf_details=htf_details
            )

    def should_take_signal(
        self,
        signal_type: SignalType,
        ltf_timestamp: datetime,
        min_confidence: float = 0.5
    ) -> Tuple[bool, str, MTFBiasResult]:
        """
        Check if a signal should be taken based on HTF bias.

        Args:
            signal_type: The signal type (LONG or SHORT)
            ltf_timestamp: The LTF bar timestamp
            min_confidence: Minimum confidence required

        Returns:
            Tuple of (should_take, reason, bias_result)
        """
        bias_result = self.get_htf_bias_at_ltf_bar(ltf_timestamp)

        should_take, reason = self.bias_calculator.should_take_signal(
            signal_type=signal_type,
            bias_result=bias_result,
            min_confidence=min_confidence
        )

        return should_take, reason, bias_result

    def get_sync_stats(self) -> Dict[str, Any]:
        """Get statistics about the synchronized data."""
        stats = {
            "ltf_bars": len(self.ltf_data),
            "ltf_timeframe": self.ltf_timeframe,
            "htf_timeframes": self.htf_timeframes,
            "htf_bars": {},
            "coverage": {},
        }

        for tf in self.htf_timeframes:
            if tf in self.htf_data:
                stats["htf_bars"][tf] = len(self.htf_data[tf])

            if tf in self.htf_indicators:
                # Check coverage at start and end of LTF data
                first_ltf = self.ltf_data.index[100] if len(self.ltf_data) > 100 else self.ltf_data.index[0]
                last_ltf = self.ltf_data.index[-1]

                first_htf_idx = self.get_latest_closed_htf_bar(first_ltf, tf)
                last_htf_idx = self.get_latest_closed_htf_bar(last_ltf, tf)

                stats["coverage"][tf] = {
                    "first_htf_available": first_htf_idx is not None,
                    "last_htf_available": last_htf_idx is not None,
                }

        return stats


def create_mtf_synchronizer(
    symbol: str,
    ltf: str = "15m",
    htf_list: List[str] = None,
    cache_dir: Optional[Path] = None
) -> Optional[MTFSynchronizer]:
    """
    Factory function to create an MTFSynchronizer from cached data.

    Args:
        symbol: Trading symbol (e.g., 'BTC', 'ETH', 'SOL')
        ltf: Lower timeframe for entries
        htf_list: List of higher timeframes for bias
        cache_dir: Cache directory (defaults to binance_cache)

    Returns:
        MTFSynchronizer instance or None if data not available
    """
    from backtest.data_loader import DataLoader, BINANCE_CACHE_DIR

    if htf_list is None:
        htf_list = ["4h", "12h", "1d"]

    if cache_dir is None:
        cache_dir = BINANCE_CACHE_DIR

    loader = DataLoader()
    data = loader.load_ltf_htf_pair(symbol, ltf, htf_list, cache_dir)

    if data["ltf"] is None:
        print(f"No LTF data available for {symbol}/{ltf}")
        return None

    if not data["htf"]:
        print(f"No HTF data available for {symbol}")
        return None

    return MTFSynchronizer(
        ltf_data=data["ltf"],
        htf_data=data["htf"],
        ltf_timeframe=ltf,
        htf_timeframes=list(data["htf"].keys())
    )
