#!/usr/bin/env python3
"""Analyze time filter performance by asset from walkforward results."""

import pandas as pd
from pathlib import Path

# Load the results
project_root = Path(__file__).parent
results_path = project_root / "output" / "walkforward_results" / "v31_walkforward_365d_20251218_0458.csv"
df = pd.read_csv(results_path)

# Filter out errors and zero trades
df = df[(df['error'].isna()) & (df['trades'] > 0)]

print("""
================================================================================
  TIME FILTER ANALYSIS BY ASSET (1-Year Backtest)
================================================================================
""")

# Overall time filter performance
print("=" * 80)
print("  OVERALL TIME FILTER PERFORMANCE")
print("=" * 80)
time_filters = ['ALL_HOURS', 'OFF_MARKET', 'WEEKENDS_ONLY', 'NY_HOURS_ONLY']
for tf in time_filters:
    tf_df = df[df['time_filter'] == tf]
    total_pnl = tf_df['pnl'].sum()
    total_trades = tf_df['trades'].sum()
    avg_win_rate = tf_df['win_rate'].mean()
    profitable_configs = len(tf_df[tf_df['pnl'] > 0])
    total_configs = len(tf_df)
    print(f"\n  {tf}:")
    print(f"    Total PnL: ${total_pnl:,.2f}")
    print(f"    Total Trades: {total_trades}")
    print(f"    Avg Win Rate: {avg_win_rate:.1f}%")
    print(f"    Profitable Configs: {profitable_configs}/{total_configs} ({100*profitable_configs/total_configs:.1f}%)")

# Per-asset breakdown
print("\n")
print("=" * 80)
print("  PER-ASSET TIME FILTER COMPARISON")
print("=" * 80)

for asset in ['BTC', 'ETH', 'SOL']:
    print(f"\n  {asset}:")
    print("-" * 60)
    asset_df = df[df['asset'] == asset]

    for tf in time_filters:
        tf_df = asset_df[asset_df['time_filter'] == tf]
        if len(tf_df) == 0:
            continue
        total_pnl = tf_df['pnl'].sum()
        total_trades = tf_df['trades'].sum()
        avg_win_rate = tf_df['win_rate'].mean()
        best_config = tf_df.loc[tf_df['pnl'].idxmax()]
        print(f"    {tf}:")
        print(f"      Total PnL: ${total_pnl:,.2f} | Trades: {total_trades} | Avg WR: {avg_win_rate:.1f}%")
        print(f"      Best: {best_config['timeframe']} {best_config['signal_mode']} {best_config['exit_strategy']} = ${best_config['pnl']:,.2f}")

# Top configs per time filter
print("\n")
print("=" * 80)
print("  TOP 5 CONFIGS PER TIME FILTER")
print("=" * 80)

for tf in time_filters:
    print(f"\n  {tf}:")
    tf_df = df[df['time_filter'] == tf].nlargest(5, 'pnl')
    for i, (_, row) in enumerate(tf_df.iterrows(), 1):
        print(f"    {i}. {row['asset']}/{row['timeframe']} {row['signal_mode']} + {row['exit_strategy']}")
        print(f"       PnL: ${row['pnl']:,.2f} | Trades: {row['trades']} | WR: {row['win_rate']:.1f}% | PF: {row['profit_factor']:.2f}")

# Signal mode comparison
print("\n")
print("=" * 80)
print("  SIGNAL MODE PERFORMANCE BY TIME FILTER")
print("=" * 80)

for sig in ['SIMPLE', 'ENHANCED_60', 'ENHANCED_70']:
    print(f"\n  {sig}:")
    sig_df = df[df['signal_mode'] == sig]
    for tf in time_filters:
        tf_df = sig_df[sig_df['time_filter'] == tf]
        if len(tf_df) == 0:
            continue
        total_pnl = tf_df['pnl'].sum()
        total_trades = tf_df['trades'].sum()
        avg_win_rate = tf_df['win_rate'].mean()
        print(f"    {tf}: ${total_pnl:,.2f} | {total_trades} trades | {avg_win_rate:.1f}% WR")

# Exit strategy comparison
print("\n")
print("=" * 80)
print("  EXIT STRATEGY PERFORMANCE BY TIME FILTER")
print("=" * 80)

for exit_s in ['FIXED_RR', 'FULL_SIGNAL', 'WT_CROSS', 'FIRST_REVERSAL']:
    print(f"\n  {exit_s}:")
    exit_df = df[df['exit_strategy'] == exit_s]
    for tf in time_filters:
        tf_df = exit_df[exit_df['time_filter'] == tf]
        if len(tf_df) == 0:
            continue
        total_pnl = tf_df['pnl'].sum()
        total_trades = tf_df['trades'].sum()
        avg_win_rate = tf_df['win_rate'].mean()
        print(f"    {tf}: ${total_pnl:,.2f} | {total_trades} trades | {avg_win_rate:.1f}% WR")

# Summary recommendations
print("\n")
print("=" * 80)
print("  SUMMARY & RECOMMENDATIONS")
print("=" * 80)

# Best overall per asset with time filter
print("\n  BEST CONFIG PER ASSET (with time filter):")
for asset in ['BTC', 'ETH', 'SOL']:
    asset_df = df[df['asset'] == asset]
    best = asset_df.loc[asset_df['pnl'].idxmax()]
    print(f"\n  {asset}:")
    print(f"    Config: {best['timeframe']} {best['signal_mode']} + {best['exit_strategy']}")
    print(f"    Time Filter: {best['time_filter']}")
    print(f"    PnL: ${best['pnl']:,.2f} ({best['pnl_percent']:.1f}%)")
    print(f"    Trades: {best['trades']} | Win Rate: {best['win_rate']:.1f}% | PF: {best['profit_factor']:.2f}")

print("\n")
print("=" * 80)
