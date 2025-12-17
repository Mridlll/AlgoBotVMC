"""Heikin Ashi candle conversion."""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class HACandle:
    """Heikin Ashi candle data."""
    timestamp: any
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def is_bullish(self) -> bool:
        """Check if candle is bullish (green)."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """Check if candle is bearish (red)."""
        return self.close < self.open

    @property
    def body_size(self) -> float:
        """Get candle body size."""
        return abs(self.close - self.open)


class HeikinAshi:
    """
    Heikin Ashi candle converter.

    Heikin Ashi candles use modified OHLC calculations that smooth
    price action and make trends easier to identify.
    """

    def __init__(self):
        """Initialize Heikin Ashi converter."""
        self._prev_ha_open: Optional[float] = None
        self._prev_ha_close: Optional[float] = None

    def reset(self) -> None:
        """Reset converter state."""
        self._prev_ha_open = None
        self._prev_ha_close = None

    def convert_single(
        self,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float = 0.0,
        timestamp: any = None
    ) -> HACandle:
        """
        Convert a single candle to Heikin Ashi.

        Args:
            open_price: Original open price
            high: Original high price
            low: Original low price
            close: Original close price
            volume: Volume
            timestamp: Candle timestamp

        Returns:
            HACandle object
        """
        # HA Close = (Open + High + Low + Close) / 4
        ha_close = (open_price + high + low + close) / 4

        # HA Open = (Previous HA Open + Previous HA Close) / 2
        if self._prev_ha_open is None:
            # First candle: HA Open = (Open + Close) / 2
            ha_open = (open_price + close) / 2
        else:
            ha_open = (self._prev_ha_open + self._prev_ha_close) / 2

        # HA High = Max(High, HA Open, HA Close)
        ha_high = max(high, ha_open, ha_close)

        # HA Low = Min(Low, HA Open, HA Close)
        ha_low = min(low, ha_open, ha_close)

        # Store for next candle
        self._prev_ha_open = ha_open
        self._prev_ha_close = ha_close

        return HACandle(
            timestamp=timestamp,
            open=ha_open,
            high=ha_high,
            low=ha_low,
            close=ha_close,
            volume=volume
        )

    def convert_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert a DataFrame of OHLCV data to Heikin Ashi.

        Args:
            df: DataFrame with columns: open, high, low, close, volume

        Returns:
            DataFrame with Heikin Ashi OHLCV values
        """
        ha_df = df.copy()

        # HA Close = (Open + High + Low + Close) / 4
        ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

        # HA Open = (Previous HA Open + Previous HA Close) / 2
        # First candle: HA Open = (Open + Close) / 2
        ha_open = np.zeros(len(df))
        ha_open[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2

        for i in range(1, len(df)):
            ha_open[i] = (ha_open[i-1] + ha_df['ha_close'].iloc[i-1]) / 2

        ha_df['ha_open'] = ha_open

        # HA High = Max(High, HA Open, HA Close)
        ha_df['ha_high'] = ha_df[['high', 'ha_open', 'ha_close']].max(axis=1)

        # HA Low = Min(Low, HA Open, HA Close)
        ha_df['ha_low'] = ha_df[['low', 'ha_open', 'ha_close']].min(axis=1)

        # Replace original OHLC with HA values
        result = pd.DataFrame({
            'open': ha_df['ha_open'],
            'high': ha_df['ha_high'],
            'low': ha_df['ha_low'],
            'close': ha_df['ha_close'],
            'volume': df['volume'] if 'volume' in df.columns else 0
        }, index=df.index)

        return result

    def convert_list(
        self,
        candles: List[Tuple[float, float, float, float, float]]
    ) -> List[HACandle]:
        """
        Convert a list of OHLCV tuples to Heikin Ashi candles.

        Args:
            candles: List of (open, high, low, close, volume) tuples

        Returns:
            List of HACandle objects
        """
        self.reset()
        ha_candles = []

        for i, (o, h, l, c, v) in enumerate(candles):
            ha_candle = self.convert_single(o, h, l, c, v, timestamp=i)
            ha_candles.append(ha_candle)

        return ha_candles


def convert_to_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to convert OHLCV DataFrame to Heikin Ashi.

    Args:
        df: DataFrame with open, high, low, close columns

    Returns:
        DataFrame with Heikin Ashi values
    """
    converter = HeikinAshi()
    return converter.convert_dataframe(df)
