"""Historical data loader for backtesting."""

import sys
from pathlib import Path as FilePath

# Add project root and src to path for imports
project_root = FilePath(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from pathlib import Path
import asyncio

from exchanges.base import BaseExchange, Candle
from exchanges.hyperliquid import HyperliquidExchange
from exchanges.bitunix import BitunixExchange
from utils.logger import get_logger

logger = get_logger("data_loader")

# Default binance cache directory
BINANCE_CACHE_DIR = project_root / "data" / "binance_cache"


class DataLoader:
    """
    Historical data loader for backtesting.

    Supports:
    - Loading from exchange APIs
    - Loading from CSV files
    - Saving data locally for faster access
    """

    def __init__(self, cache_dir: str = "./data/cache"):
        """
        Initialize data loader.

        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def load_from_exchange(
        self,
        exchange: BaseExchange,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load historical data from exchange.

        Args:
            exchange: Exchange client
            symbol: Trading symbol
            timeframe: Candle timeframe
            start_date: Start date
            end_date: End date (defaults to now)

        Returns:
            DataFrame with OHLCV data
        """
        if end_date is None:
            end_date = datetime.utcnow()

        logger.info(f"Loading {symbol} {timeframe} data from {start_date} to {end_date}")

        all_candles = []
        current_start = start_date

        # Fetch in batches
        while current_start < end_date:
            batch_end = min(
                current_start + timedelta(days=30),  # ~720 4H candles
                end_date
            )

            try:
                candles = await exchange.get_candles(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=1000,
                    start_time=current_start,
                    end_time=batch_end
                )

                all_candles.extend(candles)
                logger.info(f"Fetched {len(candles)} candles from {current_start}")

                if candles:
                    current_start = candles[-1].timestamp + timedelta(minutes=1)
                else:
                    current_start = batch_end

                # Rate limiting
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error fetching candles: {e}")
                current_start = batch_end

        # Convert to DataFrame
        df = self._candles_to_dataframe(all_candles)

        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]

        logger.info(f"Loaded {len(df)} total candles")
        return df

    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file.

        Expected columns: timestamp, open, high, low, close, volume

        Args:
            filepath: Path to CSV file

        Returns:
            DataFrame with OHLCV data
        """
        df = pd.read_csv(filepath)

        # Handle different timestamp formats
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
            df.set_index('timestamp', inplace=True)

        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        if 'volume' not in df.columns:
            df['volume'] = 0

        return df[['open', 'high', 'low', 'close', 'volume']]

    def save_to_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """
        Save data to CSV file.

        Args:
            df: DataFrame to save
            filepath: Output file path
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath)
        logger.info(f"Saved {len(df)} candles to {filepath}")

    def load_cached(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Load cached data if available.

        Args:
            symbol: Trading symbol
            timeframe: Candle timeframe
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame if cached data exists, None otherwise
        """
        cache_file = self._get_cache_path(symbol, timeframe)

        if not cache_file.exists():
            return None

        df = pd.read_csv(cache_file, parse_dates=['timestamp'], index_col='timestamp')

        # Check if cached data covers requested range
        if df.index.min() <= start_date and df.index.max() >= end_date:
            mask = (df.index >= start_date) & (df.index <= end_date)
            return df[mask]

        return None

    def save_to_cache(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> None:
        """
        Save data to cache.

        Args:
            df: DataFrame to cache
            symbol: Trading symbol
            timeframe: Candle timeframe
        """
        cache_file = self._get_cache_path(symbol, timeframe)

        # If cache exists, merge with new data
        if cache_file.exists():
            existing = pd.read_csv(cache_file, parse_dates=['timestamp'], index_col='timestamp')
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep='last')]
            df.sort_index(inplace=True)

        df.to_csv(cache_file)
        logger.info(f"Cached {len(df)} candles for {symbol} {timeframe}")

    def _get_cache_path(self, symbol: str, timeframe: str) -> Path:
        """Get cache file path for symbol/timeframe."""
        filename = f"{symbol.lower()}_{timeframe}.csv"
        return self.cache_dir / filename

    def _candles_to_dataframe(self, candles: List[Candle]) -> pd.DataFrame:
        """Convert candles to DataFrame."""
        if not candles:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        data = {
            'timestamp': [c.timestamp for c in candles],
            'open': [c.open for c in candles],
            'high': [c.high for c in candles],
            'low': [c.low for c in candles],
            'close': [c.close for c in candles],
            'volume': [c.volume for c in candles],
        }

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        return df

    def aggregate_to_higher_timeframe(
        self,
        df: pd.DataFrame,
        target_tf: str
    ) -> pd.DataFrame:
        """
        Aggregate lower timeframe data to higher timeframe.

        Args:
            df: Source DataFrame with OHLCV data (must have datetime index)
            target_tf: Target timeframe ('8h', '12h', '1d', etc.)

        Returns:
            DataFrame aggregated to target timeframe
        """
        # Map target timeframe to pandas resample rule (using lowercase for pandas 2.x)
        tf_map = {
            '8h': '8h',
            '12h': '12h',
            '1d': '1D',
            '1D': '1D',
            '2d': '2D',
            '1w': '1W',
        }

        rule = tf_map.get(target_tf, target_tf.upper())

        try:
            aggregated = df.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            logger.info(f"Aggregated {len(df)} bars to {len(aggregated)} {target_tf} bars")
            return aggregated

        except Exception as e:
            logger.error(f"Error aggregating to {target_tf}: {e}")
            return pd.DataFrame()

    def load_multiple_timeframes(
        self,
        symbol: str,
        timeframes: List[str],
        cache_dir: Optional[Path] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load cached data for multiple timeframes.

        Supports loading existing cached data and aggregating to higher
        timeframes (12h, 1d) from 4h data if not directly available.

        Args:
            symbol: Trading symbol (e.g., 'BTC', 'ETH', 'SOL')
            timeframes: List of timeframes to load (e.g., ['15m', '4h', '12h', '1d'])
            cache_dir: Cache directory (defaults to BINANCE_CACHE_DIR)

        Returns:
            Dict mapping timeframe to DataFrame
        """
        if cache_dir is None:
            cache_dir = BINANCE_CACHE_DIR

        data = {}
        base_tf = "4h"  # Base HTF data for aggregation

        for tf in timeframes:
            # Try direct load first
            path = cache_dir / f"{symbol.lower()}_{tf}.csv"

            if path.exists():
                try:
                    df = pd.read_csv(path, parse_dates=['timestamp'], index_col='timestamp')
                    data[tf] = df
                    logger.info(f"Loaded {len(df)} bars for {symbol}/{tf}")
                except Exception as e:
                    logger.error(f"Error loading {path}: {e}")

            elif tf in ['8h', '12h', '1d', '1D']:
                # Aggregate from 4h data
                base_path = cache_dir / f"{symbol.lower()}_{base_tf}.csv"

                if base_path.exists():
                    try:
                        base_df = pd.read_csv(base_path, parse_dates=['timestamp'], index_col='timestamp')
                        aggregated = self.aggregate_to_higher_timeframe(base_df, tf)

                        if not aggregated.empty:
                            data[tf] = aggregated
                            logger.info(f"Aggregated {symbol}/{base_tf} to {tf}: {len(aggregated)} bars")
                    except Exception as e:
                        logger.error(f"Error aggregating {symbol}/{tf}: {e}")
                else:
                    logger.warning(f"Cannot create {tf} data: no {base_tf} data available for {symbol}")
            else:
                logger.warning(f"No data available for {symbol}/{tf}")

        return data

    def load_ltf_htf_pair(
        self,
        symbol: str,
        ltf: str,
        htf_list: List[str],
        cache_dir: Optional[Path] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load LTF data and multiple HTF data for MTF backtesting.

        Args:
            symbol: Trading symbol
            ltf: Lower timeframe for entries (e.g., '15m', '30m')
            htf_list: List of higher timeframes for bias (e.g., ['4h', '12h', '1d'])
            cache_dir: Cache directory

        Returns:
            Dict with 'ltf' key and 'htf' dict containing each HTF DataFrame
        """
        all_tfs = [ltf] + htf_list
        data = self.load_multiple_timeframes(symbol, all_tfs, cache_dir)

        result = {
            'ltf': data.get(ltf),
            'htf': {tf: data.get(tf) for tf in htf_list if tf in data}
        }

        return result

    @staticmethod
    def generate_sample_data(
        num_candles: int = 1000,
        start_price: float = 50000.0,
        volatility: float = 0.02
    ) -> pd.DataFrame:
        """
        Generate sample OHLCV data for testing.

        Args:
            num_candles: Number of candles to generate
            start_price: Starting price
            volatility: Price volatility (0.02 = 2%)

        Returns:
            DataFrame with synthetic OHLCV data
        """
        np.random.seed(42)

        timestamps = pd.date_range(
            start=datetime.utcnow() - timedelta(hours=num_candles * 4),
            periods=num_candles,
            freq='4H'
        )

        # Generate random walk
        returns = np.random.randn(num_candles) * volatility
        prices = start_price * np.cumprod(1 + returns)

        # Generate OHLC from close prices
        data = {
            'close': prices,
            'open': np.roll(prices, 1),
            'high': prices * (1 + np.random.rand(num_candles) * volatility),
            'low': prices * (1 - np.random.rand(num_candles) * volatility),
            'volume': np.random.rand(num_candles) * 1000000,
        }

        data['open'][0] = start_price

        df = pd.DataFrame(data, index=timestamps)
        df.index.name = 'timestamp'

        return df
