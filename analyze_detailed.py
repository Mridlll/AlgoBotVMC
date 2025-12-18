#!/usr/bin/env python3
"""Detailed coin-wise strategy breakdown."""

import pandas as pd
from pathlib import Path

# Load the results
project_root = Path(__file__).parent
results_path = project_root / "output" / "walkforward_results" / "v31_walkforward_365d_20251218_0458.csv"
df = pd.read_csv(results_path)

# Filter out errors and zero trades
df = df[(df['error'].isna()) & (df['trades'] > 0)]

# Sort by PnL descending
df = df.sort_values('pnl', ascending=False)

for asset in ['BTC', 'ETH', 'SOL']:
    print(f"""
{'='*100}
  {asset} - ALL STRATEGY COMBINATIONS (1-Year Backtest with Slippage)
{'='*100}
""")

    asset_df = df[df['asset'] == asset].copy()

    # Group by strategy config (signal + exit + timeframe + time_filter)
    print(f"{'TF':<4} | {'Signal':<12} | {'Exit':<14} | {'Filter':<12} | {'PnL':>10} | {'Trades':>6} | {'WR%':>5} | {'PF':>5} | {'MaxDD':>6} | {'Long':>5} | {'Short':>5}")
    print("-"*100)

    for _, row in asset_df.iterrows():
        pf = row['profit_factor'] if row['profit_factor'] != float('inf') else 99.99
        print(f"{row['timeframe']:<4} | {row['signal_mode']:<12} | {row['exit_strategy']:<14} | {row['time_filter']:<12} | ${row['pnl']:>8,.0f} | {row['trades']:>6} | {row['win_rate']:>5.1f} | {pf:>5.2f} | {row['max_drawdown']:>5.1f}% | {row['long_trades']:>5} | {row['short_trades']:>5}")

    # Summary stats
    profitable = len(asset_df[asset_df['pnl'] > 0])
    total = len(asset_df)
    total_pnl = asset_df['pnl'].sum()

    print(f"""
  Summary: {profitable}/{total} profitable configs ({100*profitable/total:.1f}%)
  Total PnL across all configs: ${total_pnl:,.2f}
""")

# Also print top 10 overall
print(f"""
{'='*100}
  TOP 20 STRATEGIES OVERALL (All Assets)
{'='*100}

{'Asset':<5} | {'TF':<4} | {'Signal':<12} | {'Exit':<14} | {'Filter':<12} | {'PnL':>10} | {'Trades':>6} | {'WR%':>5} | {'PF':>5} | {'Long':>5} | {'Short':>5}
{'-'*100}""")

top20 = df.nlargest(20, 'pnl')
for _, row in top20.iterrows():
    pf = row['profit_factor'] if row['profit_factor'] != float('inf') else 99.99
    print(f"{row['asset']:<5} | {row['timeframe']:<4} | {row['signal_mode']:<12} | {row['exit_strategy']:<14} | {row['time_filter']:<12} | ${row['pnl']:>8,.0f} | {row['trades']:>6} | {row['win_rate']:>5.1f} | {pf:>5.2f} | {row['long_trades']:>5} | {row['short_trades']:>5}")

# Bottom 10 (worst performers)
print(f"""

{'='*100}
  BOTTOM 10 STRATEGIES (Worst Performers)
{'='*100}

{'Asset':<5} | {'TF':<4} | {'Signal':<12} | {'Exit':<14} | {'Filter':<12} | {'PnL':>10} | {'Trades':>6} | {'WR%':>5} | {'PF':>5} | {'Long':>5} | {'Short':>5}
{'-'*100}""")

bottom10 = df.nsmallest(10, 'pnl')
for _, row in bottom10.iterrows():
    pf = row['profit_factor'] if row['profit_factor'] != float('inf') else 99.99
    print(f"{row['asset']:<5} | {row['timeframe']:<4} | {row['signal_mode']:<12} | {row['exit_strategy']:<14} | {row['time_filter']:<12} | ${row['pnl']:>8,.0f} | {row['trades']:>6} | {row['win_rate']:>5.1f} | {pf:>5.2f} | {row['long_trades']:>5} | {row['short_trades']:>5}")
