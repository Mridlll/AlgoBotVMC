#!/usr/bin/env python3
"""
VMC Trading Bot V6 - Backtest GUI

Simple graphical interface for clients to run backtests.
Choose between cached data or fresh fetch from exchange.

Usage:
    python run_backtest_gui.py
"""

import sys
import asyncio
import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading

# Get script directory and change to it
SCRIPT_DIR = Path(__file__).parent.resolve()
os.chdir(SCRIPT_DIR)

# Add paths for imports
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "src"))

import pandas as pd

from config.config import load_config
from backtest.engine import BacktestEngine, SignalMode, TakeProfitMethod, OscillatorExitMode
from backtest.data_loader import DataLoader


class BacktestGUI:
    """Simple GUI for running backtests."""

    def __init__(self, root):
        self.root = root
        self.root.title("VMC Trading Bot V6 - Backtest")
        self.root.geometry("700x600")
        self.root.resizable(True, True)

        # Variables
        self.data_source = tk.StringVar(value="cache")
        self.asset = tk.StringVar(value="ALL")
        self.days = tk.StringVar(value="365")
        self.running = False

        self.create_widgets()

    def create_widgets(self):
        """Create all GUI widgets."""
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="VMC Trading Bot V6 - Backtest",
            font=('Helvetica', 16, 'bold')
        )
        title_label.pack(pady=(0, 20))

        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="15")
        settings_frame.pack(fill=tk.X, pady=(0, 15))

        # Data Source
        source_frame = ttk.Frame(settings_frame)
        source_frame.pack(fill=tk.X, pady=5)

        ttk.Label(source_frame, text="Data Source:", width=15).pack(side=tk.LEFT)

        source_options = ttk.Frame(source_frame)
        source_options.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Radiobutton(
            source_options,
            text="Use Cached Data (Faster)",
            variable=self.data_source,
            value="cache"
        ).pack(side=tk.LEFT, padx=10)

        ttk.Radiobutton(
            source_options,
            text="Fetch Fresh Data",
            variable=self.data_source,
            value="fetch"
        ).pack(side=tk.LEFT, padx=10)

        # Asset Selection
        asset_frame = ttk.Frame(settings_frame)
        asset_frame.pack(fill=tk.X, pady=5)

        ttk.Label(asset_frame, text="Asset:", width=15).pack(side=tk.LEFT)

        asset_combo = ttk.Combobox(
            asset_frame,
            textvariable=self.asset,
            values=["ALL", "BTC", "ETH", "SOL"],
            state="readonly",
            width=20
        )
        asset_combo.pack(side=tk.LEFT, padx=10)

        # Days
        days_frame = ttk.Frame(settings_frame)
        days_frame.pack(fill=tk.X, pady=5)

        ttk.Label(days_frame, text="Days:", width=15).pack(side=tk.LEFT)

        days_combo = ttk.Combobox(
            days_frame,
            textvariable=self.days,
            values=["30", "90", "180", "365", "730"],
            width=20
        )
        days_combo.pack(side=tk.LEFT, padx=10)

        ttk.Label(days_frame, text="(or type custom value)").pack(side=tk.LEFT)

        # Cache Status
        cache_frame = ttk.LabelFrame(main_frame, text="Cache Status", padding="10")
        cache_frame.pack(fill=tk.X, pady=(0, 15))

        self.cache_status = ttk.Label(cache_frame, text="Checking cache...")
        self.cache_status.pack(anchor=tk.W)

        self.update_cache_status()

        # Run Button
        self.run_button = ttk.Button(
            main_frame,
            text="Run Backtest",
            command=self.run_backtest,
            style='Accent.TButton'
        )
        self.run_button.pack(pady=10)

        # Progress
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(0, 10))

        # Output
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True)

        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            height=15,
            font=('Consolas', 9)
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def update_cache_status(self):
        """Check and display cache status."""
        cache_dirs = [
            Path("data/binance_cache"),
            Path("data/binance_cache_1year"),
            Path("data/cache")
        ]

        status_lines = []
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                files = list(cache_dir.glob("*.csv"))
                if files:
                    status_lines.append(f"{cache_dir}: {len(files)} files")
                    for f in files[:5]:  # Show first 5
                        size_mb = f.stat().st_size / (1024 * 1024)
                        status_lines.append(f"  - {f.name} ({size_mb:.1f} MB)")
                    if len(files) > 5:
                        status_lines.append(f"  ... and {len(files) - 5} more")

        if status_lines:
            self.cache_status.config(text="\n".join(status_lines))
        else:
            self.cache_status.config(text="No cached data found. Select 'Fetch Fresh Data' to download.")

    def log(self, message: str):
        """Add message to output."""
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        self.root.update_idletasks()

    def run_backtest(self):
        """Run the backtest in a separate thread."""
        if self.running:
            return

        self.running = True
        self.run_button.config(state='disabled')
        self.progress.start()
        self.output_text.delete(1.0, tk.END)

        # Run in thread to keep GUI responsive
        thread = threading.Thread(target=self._run_backtest_thread)
        thread.daemon = True
        thread.start()

    def _run_backtest_thread(self):
        """Backtest thread."""
        try:
            asyncio.run(self._run_backtest_async())
        except Exception as e:
            self.root.after(0, lambda: self.log(f"Error: {e}"))
        finally:
            self.root.after(0, self._backtest_complete)

    def _backtest_complete(self):
        """Called when backtest completes."""
        self.running = False
        self.run_button.config(state='normal')
        self.progress.stop()

    async def _run_backtest_async(self):
        """Async backtest execution."""
        from datetime import timezone

        self.root.after(0, lambda: self.log("=" * 60))
        self.root.after(0, lambda: self.log("VMC Trading Bot V6 - Backtest"))
        self.root.after(0, lambda: self.log("=" * 60))

        # Parse settings
        days = int(self.days.get())
        asset_filter = None if self.asset.get() == "ALL" else self.asset.get()
        use_cache = self.data_source.get() == "cache"

        self.root.after(0, lambda: self.log(f"Data Source: {'Cache' if use_cache else 'Fresh Fetch'}"))
        self.root.after(0, lambda: self.log(f"Asset: {self.asset.get()}"))
        self.root.after(0, lambda: self.log(f"Period: {days} days"))
        self.root.after(0, lambda: self.log("-" * 60))

        # Load config
        config_path = Path("config/config_v6_production.yaml")
        if not config_path.exists():
            self.root.after(0, lambda: self.log("Error: Config file not found!"))
            return

        config = load_config(str(config_path))
        strategies = config.get_enabled_strategies()

        if asset_filter:
            strategies = {k: v for k, v in strategies.items() if v.asset == asset_filter}

        self.root.after(0, lambda: self.log(f"Loaded {len(strategies)} strategies"))

        # Get unique timeframes/assets
        timeframes = set()
        assets = set()
        for strat in strategies.values():
            timeframes.add(strat.timeframe)
            assets.add(strat.asset)

        # Initialize data loader
        data_loader = DataLoader(cache_dir="data/cache")

        # Connect to exchange only if fetching
        exchange = None
        if not use_cache:
            try:
                from exchanges.hyperliquid import HyperliquidExchange
                exchange = HyperliquidExchange(
                    api_key="",
                    api_secret="",
                    testnet=True
                )
                await exchange.connect()
                self.root.after(0, lambda: self.log("Connected to exchange"))
            except Exception as e:
                self.root.after(0, lambda: self.log(f"Exchange connection failed: {e}"))
                self.root.after(0, lambda: self.log("Falling back to cache..."))
                use_cache = True

        # Load data
        data_cache: Dict[str, Dict[str, pd.DataFrame]] = {}

        for asset in assets:
            data_cache[asset] = {}
            for tf in timeframes:
                self.root.after(0, lambda a=asset, t=tf: self.log(f"Loading {a} {t}..."))

                df = await self._load_data(asset, tf, days, data_loader, exchange if not use_cache else None)

                if df is not None and len(df) > 100:
                    data_cache[asset][tf] = df
                    self.root.after(0, lambda a=asset, t=tf, n=len(df): self.log(f"  {a} {t}: {n} candles"))
                else:
                    self.root.after(0, lambda a=asset, t=tf: self.log(f"  {a} {t}: Not enough data"))

        # Disconnect exchange
        if exchange:
            await exchange.disconnect()

        # Run backtests
        self.root.after(0, lambda: self.log("-" * 60))
        self.root.after(0, lambda: self.log("Running backtests..."))

        results = []
        for strat_name, strat_config in strategies.items():
            asset = strat_config.asset
            tf = strat_config.timeframe

            if asset not in data_cache or tf not in data_cache[asset]:
                continue

            data = data_cache[asset][tf]

            # Run backtest
            result = self._run_single_backtest(strat_name, strat_config, data)
            if result:
                results.append(result)
                self.root.after(0, lambda n=strat_name, r=result:
                    self.log(f"  {n}: {r['trades']} trades, ${r['pnl']:,.0f}"))

        # Summary
        self.root.after(0, lambda: self.log("=" * 60))
        self.root.after(0, lambda: self.log("SUMMARY"))
        self.root.after(0, lambda: self.log("=" * 60))

        if results:
            total_pnl = sum(r['pnl'] for r in results)
            total_trades = sum(r['trades'] for r in results)
            avg_win_rate = sum(r['win_rate'] for r in results) / len(results)

            self.root.after(0, lambda: self.log(f"Total Strategies: {len(results)}"))
            self.root.after(0, lambda: self.log(f"Total Trades: {total_trades}"))
            self.root.after(0, lambda: self.log(f"Total PnL: ${total_pnl:,.0f}"))
            self.root.after(0, lambda: self.log(f"Avg Win Rate: {avg_win_rate:.1f}%"))
        else:
            self.root.after(0, lambda: self.log("No results to display"))

        self.root.after(0, lambda: self.log("=" * 60))
        self.root.after(0, lambda: self.log("Backtest complete!"))

    async def _load_data(
        self,
        asset: str,
        timeframe: str,
        days: int,
        data_loader: DataLoader,
        exchange = None
    ) -> Optional[pd.DataFrame]:
        """Load historical data from cache or exchange."""
        from datetime import timezone

        # Check cache first
        cache_locations = [
            Path("data/binance_cache") / f"{asset.lower()}_{timeframe}.csv",
            Path("data/binance_cache_1year") / f"{asset.lower()}_{timeframe}.csv",
            Path("data/cache") / f"{asset}_{timeframe}.csv",
            Path("data/cache") / f"{asset.lower()}_{timeframe}.csv",
        ]

        for cache_file in cache_locations:
            if cache_file.exists():
                try:
                    df = pd.read_csv(cache_file, parse_dates=['timestamp'], index_col='timestamp')

                    if len(df) > 100:
                        # Filter to requested date range
                        end_date = datetime.now(timezone.utc)
                        start_date = end_date - timedelta(days=days)

                        if df.index.tz is None:
                            df.index = df.index.tz_localize('UTC')

                        mask = (df.index >= start_date) & (df.index <= end_date)
                        filtered_df = df[mask]

                        if len(filtered_df) > 100:
                            return filtered_df
                        elif len(df) > 100:
                            return df
                except Exception:
                    continue

        # Fetch from exchange if provided
        if exchange:
            try:
                end_date = datetime.now(timezone.utc)
                start_date = end_date - timedelta(days=days)
                df = await data_loader.load_from_exchange(
                    exchange=exchange,
                    symbol=asset,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                if len(df) > 100:
                    data_loader.save_to_cache(df, asset, timeframe)
                    return df
            except Exception:
                pass

        return None

    def _run_single_backtest(self, strategy_name: str, config, data: pd.DataFrame) -> Optional[dict]:
        """Run backtest for a single strategy."""
        try:
            signal_mode = SignalMode.ENHANCED if config.signal_mode == "enhanced" else SignalMode.SIMPLE

            engine = BacktestEngine(
                initial_balance=10000.0,
                risk_percent=3.0,
                commission_percent=0.06,
                signal_mode=signal_mode,
                tp_method=TakeProfitMethod.OSCILLATOR,
                oscillator_mode=OscillatorExitMode.FULL_SIGNAL,
                simple_overbought=config.anchor_level,
                simple_oversold=-config.anchor_level,
                direction_filter=config.direction_filter
            )

            result = engine.run(data)

            return {
                'strategy': strategy_name,
                'trades': result.total_trades,
                'win_rate': result.win_rate,
                'pnl': result.total_pnl,
                'sharpe': result.sharpe_ratio
            }
        except Exception:
            return None


def main():
    """Launch the GUI."""
    root = tk.Tk()

    # Style
    style = ttk.Style()
    style.configure('Accent.TButton', font=('Helvetica', 11, 'bold'))

    app = BacktestGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
