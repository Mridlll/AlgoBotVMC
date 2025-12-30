"""
VMC Trading Bot V6 - FIXED Performance Evaluation
==================================================

V6 with bug fixes and improved exit logic:
1. Real VWAP (fixed - was reading WT momentum before)
2. Divergence detection for enhanced entries
3. Exit on: Opposite WT signal OR VWAP curves against OR Stop loss
4. No partial exits - clean comparison with V5
5. BTC uses BOTH directions (not SHORT_ONLY)
6. Fixed commission calculation
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json

# Core modules
from indicators import WaveTrend, MoneyFlow, VWAPCalculator, DivergenceDetector
from indicators.heikin_ashi import convert_to_heikin_ashi

# ============================================================================
# CONFIGURATION
# ============================================================================

ASSETS = ["BTC", "ETH", "SOL"]
TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h"]
SIGNAL_MODES = ["SIMPLE", "ENHANCED_60", "ENHANCED_70"]
TIME_FILTERS = ["ALL_HOURS", "NY_HOURS_ONLY", "WEEKENDS_ONLY"]

INITIAL_BALANCE = 10000
RISK_PER_TRADE = 3.0  # Percentage

# Per-asset VWAP settings: BTC benefits from VWAP filter, ETH/SOL don't
ASSET_VWAP_SETTINGS = {
    "BTC": True,   # VWAP entry confirmation ENABLED
    "ETH": False,  # VWAP entry confirmation DISABLED
    "SOL": False,  # VWAP entry confirmation DISABLED
}

# Output directory
OUTPUT_DIR = project_root / "output" / f"v6_perasset_vwap_{datetime.now().strftime('%Y%m%d_%H%M')}"


# ============================================================================
# V5-STYLE TIME FILTER (CORRECT APPROACH)
# ============================================================================

def apply_time_filter(df: pd.DataFrame, filter_name: str) -> pd.DataFrame:
    """
    Filter DataFrame to only include bars matching the time filter.
    This is the V5 approach - pre-filter data BEFORE backtest.
    """
    if filter_name == "ALL_HOURS":
        return df
    elif filter_name == "NY_HOURS_ONLY":
        # Weekdays (Mon-Fri) during NY hours (14:00-21:00 UTC)
        mask = (df.index.weekday < 5) & (df.index.hour >= 14) & (df.index.hour < 21)
        return df[mask]
    elif filter_name == "WEEKENDS_ONLY":
        # Saturday (5) and Sunday (6) only
        return df[df.index.weekday >= 5]
    return df


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(asset: str, timeframe: str) -> pd.DataFrame:
    """Load cached data for asset/timeframe."""
    cache_dirs = [
        project_root / "data" / "binance_cache_1year",
        project_root / "data" / "binance_cache"
    ]

    filename = f"{asset.lower()}_{timeframe}.csv"

    for cache_dir in cache_dirs:
        filepath = cache_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            df = df[~df.index.duplicated(keep='first')]
            return df

    raise FileNotFoundError(f"No data found for {asset} {timeframe}")


# ============================================================================
# V6 FIXED BACKTEST ENGINE
# ============================================================================

@dataclass
class V6Trade:
    """Trade with V6 features."""
    entry_time: datetime
    exit_time: Optional[datetime]
    direction: str  # "long" or "short"
    entry_price: float
    exit_price: float
    stop_loss: float
    size: float
    pnl: float
    pnl_percent: float
    exit_reason: str

    # V6 enhancements
    vwap_at_entry: float = 0.0
    vwap_confirmed: bool = False
    divergence_at_entry: bool = False
    divergence_strength: float = 0.0


@dataclass
class V6BacktestResult:
    """V6 backtest result with enhanced metrics."""
    config: str
    asset: str
    timeframe: str
    signal_mode: str
    time_filter: str

    # Core metrics
    total_pnl: float
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown_pct: float

    # Direction breakdown
    long_pnl: float = 0.0
    short_pnl: float = 0.0
    long_trades: int = 0
    short_trades: int = 0

    # V6 specific
    vwap_confirmed_trades: int = 0
    divergence_trades: int = 0
    vwap_exits: int = 0  # Exits triggered by VWAP curving against
    signal_exits: int = 0  # Exits triggered by opposite signal
    sl_exits: int = 0  # Stop loss exits

    # Trade list
    trades: List[V6Trade] = field(default_factory=list)


class V6FixedBacktestEngine:
    """
    V6 Fixed Backtest Engine with:
    - Real VWAP confirmation (fixed bug)
    - Divergence detection
    - Exit on: Opposite WT signal OR VWAP curves against OR Stop loss
    - No partial exits
    - Fixed commission calculation
    """

    def __init__(
        self,
        initial_balance: float = 10000,
        risk_percent: float = 3.0,
        signal_mode: str = "SIMPLE",
        anchor_long: int = -53,
        anchor_short: int = 53,
        direction_filter: str = "both",  # Always BOTH for all assets
        slippage_percent: float = 0.03,
        commission_percent: float = 0.06,
        use_vwap_confirmation: bool = True,
        use_divergence: bool = True,
        use_vwap_exit: bool = True,  # Exit when VWAP curves against
        vwap_exit_bars: int = 3  # Require N consecutive bars of VWAP curving against
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_percent = risk_percent / 100
        self.signal_mode = signal_mode
        self.anchor_long = anchor_long
        self.anchor_short = anchor_short
        self.direction_filter = direction_filter
        self.slippage_percent = slippage_percent / 100
        self.commission_percent = commission_percent / 100

        # V6 features
        self.use_vwap_confirmation = use_vwap_confirmation
        self.use_divergence = use_divergence
        self.use_vwap_exit = use_vwap_exit
        self.vwap_exit_bars = vwap_exit_bars  # Consecutive bars threshold

        # Indicators
        self.wt = WaveTrend()
        self.mfi = MoneyFlow()
        self.vwap_calc = VWAPCalculator()
        self.divergence_detector = DivergenceDetector(lookback=5, min_swing_distance=3)

    def run(self, df: pd.DataFrame) -> V6BacktestResult:
        """Run V6 backtest on data."""
        if len(df) < 100:
            return self._empty_result()

        # Convert to Heikin Ashi
        ha_df = convert_to_heikin_ashi(df)

        # Calculate indicators on HA data
        wt_result = self.wt.calculate(ha_df)
        mfi_result = self.mfi.calculate(ha_df)

        # Calculate VWAP on RAW data (correct volume) - FIXED BUG
        vwap_result = self.vwap_calc.calculate(df)

        # Calculate divergence on HA close vs WT2
        div_result = self.divergence_detector.detect(
            ha_df['close'],
            wt_result.wt2
        )

        # Trading loop
        trades: List[V6Trade] = []
        equity_curve = [self.initial_balance]
        self.balance = self.initial_balance

        # Position state
        in_position = False
        position_direction = None
        entry_price = 0.0
        stop_loss = 0.0
        position_size = 0.0
        trade_info = {}
        vwap_against_count = 0  # Track consecutive bars VWAP curves against

        # Exit counters
        vwap_exits = 0
        signal_exits = 0
        sl_exits = 0

        for i in range(100, len(df)):
            timestamp = df.index[i]
            close = df['close'].iloc[i]
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]

            wt1 = wt_result.wt1.iloc[i]
            wt2 = wt_result.wt2.iloc[i]
            mfi = mfi_result.mfi.iloc[i]

            # FIXED: Use real VWAP, not WT momentum
            vwap = vwap_result.vwap.iloc[i]
            vwap_curving_up = vwap_result.curving_up.iloc[i]
            vwap_curving_down = vwap_result.curving_down.iloc[i]

            bullish_div = div_result.bullish_regular.iloc[i]
            bearish_div = div_result.bearish_regular.iloc[i]
            div_strength = div_result.strength.iloc[i]

            # ============================================================
            # EXIT LOGIC: Opposite WT signal OR VWAP curves against OR SL
            # ============================================================
            if in_position:
                exit_triggered = False
                exit_reason = ""
                exit_price = close

                # 1. Check Stop Loss
                if position_direction == "long" and low <= stop_loss:
                    exit_triggered = True
                    exit_reason = "stop_loss"
                    exit_price = stop_loss
                    sl_exits += 1
                elif position_direction == "short" and high >= stop_loss:
                    exit_triggered = True
                    exit_reason = "stop_loss"
                    exit_price = stop_loss
                    sl_exits += 1

                # 2. Check VWAP curves against (if enabled and not already exiting)
                # Requires N consecutive bars of VWAP curving against
                if not exit_triggered and self.use_vwap_exit:
                    vwap_against_now = False
                    if position_direction == "long" and vwap_curving_down:
                        vwap_against_now = True
                    elif position_direction == "short" and vwap_curving_up:
                        vwap_against_now = True

                    if vwap_against_now:
                        vwap_against_count += 1
                        if vwap_against_count >= self.vwap_exit_bars:
                            exit_triggered = True
                            exit_reason = "vwap_curves_against"
                            vwap_exits += 1
                    else:
                        vwap_against_count = 0  # Reset if VWAP not curving against

                # 3. Check opposite WT signal (FULL_SIGNAL style)
                if not exit_triggered:
                    wt1_prev = wt_result.wt1.iloc[i - 1]
                    wt2_prev = wt_result.wt2.iloc[i - 1]

                    if position_direction == "long":
                        # Exit long on bearish cross from overbought
                        if wt1 < wt2 and wt1_prev >= wt2_prev and wt2 > self.anchor_short:
                            exit_triggered = True
                            exit_reason = "opposite_signal"
                            signal_exits += 1
                    else:  # short
                        # Exit short on bullish cross from oversold
                        if wt1 > wt2 and wt1_prev <= wt2_prev and wt2 < self.anchor_long:
                            exit_triggered = True
                            exit_reason = "opposite_signal"
                            signal_exits += 1

                # Execute exit
                if exit_triggered:
                    # Apply slippage on exit
                    if position_direction == "long":
                        exit_price = exit_price * (1 - self.slippage_percent)
                    else:
                        exit_price = exit_price * (1 + self.slippage_percent)

                    # Calculate PnL
                    if position_direction == "long":
                        pnl = (exit_price - entry_price) * position_size
                    else:
                        pnl = (entry_price - exit_price) * position_size

                    # FIXED: Commission based on notional value, not PnL
                    exit_commission = exit_price * position_size * self.commission_percent
                    net_pnl = pnl - exit_commission

                    pnl_percent = (net_pnl / (entry_price * position_size)) * 100

                    # Update balance
                    self.balance += net_pnl

                    # Record trade
                    trade = V6Trade(
                        entry_time=trade_info['entry_time'],
                        exit_time=timestamp,
                        direction=position_direction,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        stop_loss=stop_loss,
                        size=position_size,
                        pnl=net_pnl,
                        pnl_percent=pnl_percent,
                        exit_reason=exit_reason,
                        vwap_at_entry=trade_info['vwap'],
                        vwap_confirmed=trade_info['vwap_confirmed'],
                        divergence_at_entry=trade_info['divergence'],
                        divergence_strength=trade_info['div_strength']
                    )
                    trades.append(trade)

                    # Reset position
                    in_position = False
                    position_direction = None
                    entry_price = 0.0
                    stop_loss = 0.0
                    position_size = 0.0
                    trade_info = {}
                    vwap_against_count = 0  # Reset counter

            # ============================================================
            # ENTRY LOGIC: WT signal + VWAP confirmation + Divergence
            # ============================================================
            if not in_position:
                signal = self._detect_signal(wt1, wt2, i, wt_result)

                if signal is not None:
                    # Direction filter
                    if self.direction_filter == "long_only" and signal != "long":
                        signal = None
                    elif self.direction_filter == "short_only" and signal != "short":
                        signal = None

                    if signal is not None:
                        # V6: Check VWAP confirmation
                        vwap_confirmed = False
                        if self.use_vwap_confirmation:
                            if signal == "long" and close > vwap:
                                vwap_confirmed = True
                            elif signal == "short" and close < vwap:
                                vwap_confirmed = True

                            # Skip entry if VWAP doesn't confirm
                            if not vwap_confirmed:
                                signal = None

                        if signal is not None:
                            # V6: Check divergence (optional enhancement, not required)
                            has_divergence = False
                            if self.use_divergence:
                                if signal == "long" and bullish_div:
                                    has_divergence = True
                                elif signal == "short" and bearish_div:
                                    has_divergence = True

                            # Calculate position size and stop loss
                            entry_price = close * (1 + self.slippage_percent) if signal == "long" else close * (1 - self.slippage_percent)

                            # ATR-based stop loss
                            atr = self._calculate_atr(df, i, 14)
                            if signal == "long":
                                stop_loss = entry_price - (2 * atr)
                            else:
                                stop_loss = entry_price + (2 * atr)

                            risk_per_trade = self.balance * self.risk_percent
                            risk_per_unit = abs(entry_price - stop_loss)
                            position_size = risk_per_trade / risk_per_unit if risk_per_unit > 0 else 0

                            if position_size > 0:
                                # Enter position
                                in_position = True
                                position_direction = signal
                                vwap_against_count = 0  # Reset counter on new position

                                # Store trade info
                                trade_info = {
                                    'entry_time': timestamp,
                                    'vwap': vwap,
                                    'vwap_confirmed': vwap_confirmed,
                                    'divergence': has_divergence,
                                    'div_strength': div_strength
                                }

                                # Deduct entry commission
                                entry_commission = entry_price * position_size * self.commission_percent
                                self.balance -= entry_commission

            equity_curve.append(self.balance)

        # Force close any remaining position
        if in_position:
            final_price = df['close'].iloc[-1]
            if position_direction == "long":
                pnl = (final_price - entry_price) * position_size
            else:
                pnl = (entry_price - final_price) * position_size

            exit_commission = final_price * position_size * self.commission_percent
            net_pnl = pnl - exit_commission
            pnl_percent = (net_pnl / (entry_price * position_size)) * 100

            self.balance += net_pnl

            trade = V6Trade(
                entry_time=trade_info['entry_time'],
                exit_time=df.index[-1],
                direction=position_direction,
                entry_price=entry_price,
                exit_price=final_price,
                stop_loss=stop_loss,
                size=position_size,
                pnl=net_pnl,
                pnl_percent=pnl_percent,
                exit_reason="end_of_data",
                vwap_at_entry=trade_info['vwap'],
                vwap_confirmed=trade_info['vwap_confirmed'],
                divergence_at_entry=trade_info['divergence'],
                divergence_strength=trade_info['div_strength']
            )
            trades.append(trade)

        # Calculate result metrics
        return self._calculate_result(trades, equity_curve, vwap_exits, signal_exits, sl_exits)

    def _detect_signal(self, wt1: float, wt2: float, idx: int, wt_result) -> Optional[str]:
        """Detect trading signal based on signal mode."""
        if idx < 1:
            return None

        wt1_prev = wt_result.wt1.iloc[idx - 1]
        wt2_prev = wt_result.wt2.iloc[idx - 1]

        # WT cross detection
        wt1_crossed_above = wt1 > wt2 and wt1_prev <= wt2_prev
        wt1_crossed_below = wt1 < wt2 and wt1_prev >= wt2_prev

        if self.signal_mode == "SIMPLE":
            if wt1_crossed_above and wt2 < -53:
                return "long"
            elif wt1_crossed_below and wt2 > 53:
                return "short"
        else:
            if wt1_crossed_above and wt2 < self.anchor_long:
                return "long"
            elif wt1_crossed_below and wt2 > self.anchor_short:
                return "short"

        return None

    def _calculate_atr(self, df: pd.DataFrame, idx: int, period: int = 14) -> float:
        """Calculate ATR at given index."""
        if idx < period:
            return df['high'].iloc[:idx+1].max() - df['low'].iloc[:idx+1].min()

        tr_values = []
        for i in range(idx - period + 1, idx + 1):
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]
            prev_close = df['close'].iloc[i - 1]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)

        return np.mean(tr_values)

    def _calculate_result(
        self,
        trades: List[V6Trade],
        equity_curve: List[float],
        vwap_exits: int,
        signal_exits: int,
        sl_exits: int
    ) -> V6BacktestResult:
        """Calculate result metrics from trades."""
        if not trades:
            return self._empty_result()

        total_pnl = sum(t.pnl for t in trades)
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]

        win_rate = len(winners) / len(trades) * 100 if trades else 0

        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Sharpe ratio
        if len(trades) > 1:
            returns = [t.pnl_percent for t in trades]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        equity = np.array(equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        max_dd = drawdown.max()

        # Direction breakdown
        long_trades = [t for t in trades if t.direction == "long"]
        short_trades = [t for t in trades if t.direction == "short"]

        # V6 specific
        vwap_confirmed = len([t for t in trades if t.vwap_confirmed])
        divergence_trades = len([t for t in trades if t.divergence_at_entry])

        return V6BacktestResult(
            config="",  # Set by caller
            asset="",
            timeframe="",
            signal_mode="",
            time_filter="",
            total_pnl=total_pnl,
            total_trades=len(trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            long_pnl=sum(t.pnl for t in long_trades),
            short_pnl=sum(t.pnl for t in short_trades),
            long_trades=len(long_trades),
            short_trades=len(short_trades),
            vwap_confirmed_trades=vwap_confirmed,
            divergence_trades=divergence_trades,
            vwap_exits=vwap_exits,
            signal_exits=signal_exits,
            sl_exits=sl_exits,
            trades=trades
        )

    def _empty_result(self) -> V6BacktestResult:
        """Return empty result for insufficient data."""
        return V6BacktestResult(
            config="",
            asset="",
            timeframe="",
            signal_mode="",
            time_filter="",
            total_pnl=0,
            total_trades=0,
            win_rate=0,
            profit_factor=0,
            sharpe_ratio=0,
            max_drawdown_pct=0,
            trades=[]
        )


# ============================================================================
# MAIN EVALUATION RUNNER
# ============================================================================

def run_evaluation():
    """Run V6 fixed evaluation on all configurations."""
    print("=" * 80)
    print("VMC Trading Bot V6 - FIXED Evaluation")
    print("=" * 80)
    print(f"\nFeatures:")
    print("  - Real VWAP confirmation (bug fixed)")
    print("  - Divergence detection")
    print("  - Exit: Opposite WT signal OR VWAP curves against OR Stop loss")
    print("  - No partial exits")
    print("  - BTC: BOTH directions (not SHORT_ONLY)")
    print(f"\nConfigurations: {len(ASSETS) * len(TIMEFRAMES) * len(SIGNAL_MODES) * len(TIME_FILTERS)}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "trade_logs").mkdir(exist_ok=True)

    results = []
    total_configs = len(ASSETS) * len(TIMEFRAMES) * len(SIGNAL_MODES) * len(TIME_FILTERS)
    current = 0

    for asset in ASSETS:
        for tf in TIMEFRAMES:
            # Load data once per asset/timeframe
            try:
                df = load_data(asset, tf)
            except FileNotFoundError as e:
                print(f"  Skipping {asset} {tf}: {e}")
                continue

            for signal_mode in SIGNAL_MODES:
                for time_filter in TIME_FILTERS:
                    current += 1
                    config_id = f"{asset}_{tf}_{signal_mode}_{time_filter}"

                    # Apply time filter
                    filtered_df = apply_time_filter(df, time_filter)

                    if len(filtered_df) < 100:
                        print(f"  [{current}/{total_configs}] {config_id}: Insufficient data")
                        continue

                    # Set anchor levels based on signal mode
                    if signal_mode == "SIMPLE":
                        anchor_long, anchor_short = -53, 53
                    elif signal_mode == "ENHANCED_60":
                        anchor_long, anchor_short = -60, 60
                    else:  # ENHANCED_70
                        anchor_long, anchor_short = -70, 70

                    # Determine slippage based on timeframe
                    slippage = 0.01 if tf == "5m" else 0.03

                    # Create engine - ALL assets use BOTH directions
                    engine = V6FixedBacktestEngine(
                        initial_balance=INITIAL_BALANCE,
                        risk_percent=RISK_PER_TRADE,
                        signal_mode=signal_mode,
                        anchor_long=anchor_long,
                        anchor_short=anchor_short,
                        direction_filter="both",  # BOTH for all assets
                        slippage_percent=slippage,
                        commission_percent=0.06,
                        use_vwap_confirmation=ASSET_VWAP_SETTINGS.get(asset, True),  # Per-asset
                        use_divergence=True,
                        use_vwap_exit=True
                    )

                    # Run backtest
                    result = engine.run(filtered_df)

                    # Update result metadata
                    result.config = config_id
                    result.asset = asset
                    result.timeframe = tf
                    result.signal_mode = signal_mode
                    result.time_filter = time_filter

                    results.append(result)

                    # Print progress
                    status = "+" if result.total_pnl > 0 else "-"
                    print(f"  [{current}/{total_configs}] {config_id}: ${result.total_pnl:,.0f} ({result.total_trades} trades) [{status}]")

                    # Save trade log
                    if result.trades:
                        trade_df = pd.DataFrame([{
                            'entry_time': t.entry_time,
                            'exit_time': t.exit_time,
                            'direction': t.direction,
                            'entry_price': t.entry_price,
                            'exit_price': t.exit_price,
                            'stop_loss': t.stop_loss,
                            'size': t.size,
                            'pnl': t.pnl,
                            'pnl_percent': t.pnl_percent,
                            'exit_reason': t.exit_reason,
                            'vwap_at_entry': t.vwap_at_entry,
                            'vwap_confirmed': t.vwap_confirmed,
                            'divergence_at_entry': t.divergence_at_entry
                        } for t in result.trades])
                        trade_df.to_csv(OUTPUT_DIR / "trade_logs" / f"{config_id}_trades.csv", index=False)

    # Generate reports
    generate_reports(results)

    print(f"\n{'=' * 80}")
    print(f"Evaluation complete! Results saved to: {OUTPUT_DIR}")
    print(f"{'=' * 80}")


def generate_reports(results: List[V6BacktestResult]):
    """Generate summary reports."""
    # Summary CSV
    summary_data = [{
        'config': r.config,
        'asset': r.asset,
        'timeframe': r.timeframe,
        'signal_mode': r.signal_mode,
        'time_filter': r.time_filter,
        'total_pnl': r.total_pnl,
        'total_trades': r.total_trades,
        'win_rate': r.win_rate,
        'profit_factor': r.profit_factor,
        'sharpe_ratio': r.sharpe_ratio,
        'max_drawdown_pct': r.max_drawdown_pct,
        'long_pnl': r.long_pnl,
        'short_pnl': r.short_pnl,
        'long_trades': r.long_trades,
        'short_trades': r.short_trades,
        'vwap_confirmed_trades': r.vwap_confirmed_trades,
        'divergence_trades': r.divergence_trades,
        'vwap_exits': r.vwap_exits,
        'signal_exits': r.signal_exits,
        'sl_exits': r.sl_exits
    } for r in results]

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(OUTPUT_DIR / "summary_results.csv", index=False)

    # Generate markdown report
    profitable = [r for r in results if r.total_pnl > 0]
    total_pnl = sum(r.total_pnl for r in results)

    report = f"""# VMC Trading Bot V6 - FIXED Evaluation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Configurations Tested:** {len(results)}

## V6 Features Enabled

| Feature | Status |
|---------|--------|
| Real VWAP Confirmation (Fixed) | Enabled |
| Divergence Detection | Enabled |
| Exit: Opposite WT OR VWAP curves against | Enabled |
| Partial Exits | **Disabled** |
| BTC Direction Filter | **BOTH** |

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Configurations Tested | {len(results)} |
| Profitable | {len(profitable)} ({len(profitable)/len(results)*100:.1f}%) |
| Total Net PnL | ${total_pnl:,.0f} |
| Best Config PnL | ${max(r.total_pnl for r in results):,.0f} |
| Worst Config PnL | ${min(r.total_pnl for r in results):,.0f} |

## Top 10 Configurations

| # | Asset | TF | Signal | Filter | PnL | Trades | Win% | PF | VWAP Exits | Signal Exits |
|---|-------|-----|--------|--------|-----|--------|------|-----|------------|--------------|
"""

    top_10 = sorted(results, key=lambda x: x.total_pnl, reverse=True)[:10]
    for i, r in enumerate(top_10, 1):
        report += f"| {i} | {r.asset} | {r.timeframe} | {r.signal_mode} | {r.time_filter} | ${r.total_pnl:,.0f} | {r.total_trades} | {r.win_rate:.1f}% | {r.profit_factor:.2f} | {r.vwap_exits} | {r.signal_exits} |\n"

    # Per-asset breakdown
    report += "\n## Per-Asset Performance\n\n"

    for asset in ASSETS:
        asset_results = [r for r in results if r.asset == asset]
        if not asset_results:
            continue

        asset_pnl = sum(r.total_pnl for r in asset_results)
        asset_profitable = len([r for r in asset_results if r.total_pnl > 0])
        best = max(asset_results, key=lambda x: x.total_pnl)

        long_pnl = sum(r.long_pnl for r in asset_results)
        short_pnl = sum(r.short_pnl for r in asset_results)

        report += f"""### {asset}

| Metric | Value |
|--------|-------|
| Net PnL | ${asset_pnl:,.0f} |
| Profitable Configs | {asset_profitable}/{len(asset_results)} |
| Long PnL | ${long_pnl:,.0f} |
| Short PnL | ${short_pnl:,.0f} |
| Best Config | {best.timeframe} {best.signal_mode} {best.time_filter} |
| Best PnL | ${best.total_pnl:,.0f} |

"""

    # Exit reason breakdown
    total_vwap_exits = sum(r.vwap_exits for r in results)
    total_signal_exits = sum(r.signal_exits for r in results)
    total_sl_exits = sum(r.sl_exits for r in results)
    total_exits = total_vwap_exits + total_signal_exits + total_sl_exits

    if total_exits > 0:
        report += f"""## Exit Reason Breakdown

| Exit Reason | Count | % of Total |
|-------------|-------|------------|
| VWAP Curves Against | {total_vwap_exits} | {total_vwap_exits/total_exits*100:.1f}% |
| Opposite WT Signal | {total_signal_exits} | {total_signal_exits/total_exits*100:.1f}% |
| Stop Loss | {total_sl_exits} | {total_sl_exits/total_exits*100:.1f}% |

"""

    report += """---

*Report generated by VMC Trading Bot V6 Fixed Evaluation*
"""

    with open(OUTPUT_DIR / "V6_FIXED_EVALUATION_REPORT.md", "w") as f:
        f.write(report)


if __name__ == "__main__":
    run_evaluation()
