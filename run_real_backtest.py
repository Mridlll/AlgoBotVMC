#!/usr/bin/env python3
"""
REAL DATA BACKTEST - With Proof of Data Source
===============================================
This script shows EXACTLY where the data comes from and proves it's real.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
from datetime import datetime
import requests
import time

from indicators import HeikinAshi, WaveTrend, MoneyFlow
from indicators.heikin_ashi import convert_to_heikin_ashi
from strategy.signals import SignalDetector, SignalType


def fetch_and_show_real_data(symbol: str, interval: str, limit: int = 500):
    """
    Fetch REAL data from Binance and SHOW THE PROOF.
    """
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        'symbol': f"{symbol}USDT",
        'interval': interval,
        'limit': limit
    }

    print(f"\n{'='*70}")
    print(f"  FETCHING REAL DATA FROM BINANCE")
    print(f"{'='*70}")
    print(f"  API URL: {url}")
    print(f"  Symbol:  {symbol}USDT")
    print(f"  Interval: {interval}")
    print(f"  Limit:   {limit} candles")
    print()

    response = requests.get(url, params=params, timeout=15)
    print(f"  HTTP Status: {response.status_code}")
    print(f"  Response Size: {len(response.content)} bytes")

    if response.status_code != 200:
        print(f"  ERROR: {response.text}")
        return None

    data = response.json()
    print(f"  Candles Received: {len(data)}")

    # Show RAW data from first 3 candles
    print(f"\n  RAW API RESPONSE (first 3 candles):")
    print(f"  " + "-"*66)
    for i, candle in enumerate(data[:3]):
        ts = datetime.fromtimestamp(candle[0]/1000)
        print(f"  [{i}] Time: {ts}")
        print(f"      Open: {candle[1]}, High: {candle[2]}, Low: {candle[3]}, Close: {candle[4]}")
        print(f"      Volume: {candle[5]}")

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    df = df[['open', 'high', 'low', 'close', 'volume']]

    # Show data range
    print(f"\n  DATA RANGE:")
    print(f"  " + "-"*66)
    print(f"  Start: {df.index[0]}")
    print(f"  End:   {df.index[-1]}")
    print(f"  Total Candles: {len(df)}")

    # Show recent prices (you can verify these on TradingView/Binance)
    print(f"\n  MOST RECENT CANDLES (verify on Binance/TradingView):")
    print(f"  " + "-"*66)
    print(f"  {'Timestamp':<22} {'Open':>12} {'High':>12} {'Low':>12} {'Close':>12}")
    for idx in df.index[-5:]:
        row = df.loc[idx]
        print(f"  {str(idx):<22} {row['open']:>12,.2f} {row['high']:>12,.2f} {row['low']:>12,.2f} {row['close']:>12,.2f}")

    return df


def run_backtest_on_real_data(df: pd.DataFrame, asset: str, timeframe: str):
    """Run backtest on the real data."""

    print(f"\n{'='*70}")
    print(f"  RUNNING BACKTEST: {asset} {timeframe}")
    print(f"{'='*70}")

    if df is None or len(df) < 150:
        print("  ERROR: Insufficient data")
        return

    # Initialize indicators
    wavetrend = WaveTrend(channel_len=9, average_len=12, ma_len=3, overbought_2=60, oversold_2=-60)
    money_flow = MoneyFlow()
    signal_detector = SignalDetector(anchor_level_long=-60, anchor_level_short=60, trigger_lookback=20)

    # Convert to Heikin Ashi
    ha_df = convert_to_heikin_ashi(df)

    # Calculate indicators
    wt_result = wavetrend.calculate(ha_df)
    mf_result = money_flow.calculate(ha_df)

    print(f"\n  INDICATOR VALUES (last 5 bars):")
    print(f"  " + "-"*66)
    print(f"  {'Time':<22} {'WT1':>10} {'WT2':>10} {'VWAP':>10} {'MFI':>10}")

    for i in range(-5, 0):
        idx = df.index[i]
        print(f"  {str(idx):<22} {wt_result.wt1.iloc[i]:>10.2f} {wt_result.wt2.iloc[i]:>10.2f} {wt_result.vwap.iloc[i]:>10.2f} {mf_result.mfi.iloc[i]:>10.2f}")

    # Track signals and state
    initial_balance = 10000.0
    balance = initial_balance
    risk_percent = 3.0
    risk_reward = 2.0

    trades = []
    active_trade = None

    # State tracking
    anchors_long = 0
    triggers_long = 0
    mfi_long = 0
    signals_long = 0
    anchors_short = 0
    triggers_short = 0
    mfi_short = 0
    signals_short = 0

    prev_long_state = "IDLE"
    prev_short_state = "IDLE"

    print(f"\n  PROCESSING {len(df) - 100} BARS FOR SIGNALS...")
    print(f"  " + "-"*66)

    for i in range(100, len(df)):
        current_bar = df.iloc[i]
        current_time = df.index[i]
        current_price = current_bar['close']
        current_high = current_bar['high']
        current_low = current_bar['low']

        # Check exit conditions
        if active_trade:
            exit_reason = None
            exit_price = None

            if active_trade['direction'] == 'LONG':
                if current_low <= active_trade['stop_loss']:
                    exit_reason = 'Stop Loss'
                    exit_price = active_trade['stop_loss']
                elif current_high >= active_trade['take_profit']:
                    exit_reason = 'Take Profit'
                    exit_price = active_trade['take_profit']
            else:
                if current_high >= active_trade['stop_loss']:
                    exit_reason = 'Stop Loss'
                    exit_price = active_trade['stop_loss']
                elif current_low <= active_trade['take_profit']:
                    exit_reason = 'Take Profit'
                    exit_price = active_trade['take_profit']

            if exit_reason:
                if active_trade['direction'] == 'LONG':
                    pnl = (exit_price - active_trade['entry_price']) * active_trade['size']
                else:
                    pnl = (active_trade['entry_price'] - exit_price) * active_trade['size']

                balance += pnl

                trades.append({
                    'entry_time': active_trade['entry_time'],
                    'exit_time': current_time,
                    'direction': active_trade['direction'],
                    'entry_price': active_trade['entry_price'],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'exit_reason': exit_reason
                })

                print(f"  TRADE CLOSED: {active_trade['direction']} at {current_time}")
                print(f"    Entry: ${active_trade['entry_price']:,.2f} -> Exit: ${exit_price:,.2f}")
                print(f"    P&L: ${pnl:+,.2f} ({exit_reason})")

                active_trade = None

        # Check for new signals
        if not active_trade:
            wt_subset = type(wt_result)(
                wt1=wt_result.wt1.iloc[:i+1],
                wt2=wt_result.wt2.iloc[:i+1],
                vwap=wt_result.vwap.iloc[:i+1],
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

            # Track state changes
            state_before = signal_detector.get_current_state()

            signal = signal_detector.process_bar(
                timestamp=current_time,
                close_price=current_price,
                wt_result=wt_subset,
                mf_result=mf_subset,
                bar_idx=i
            )

            state_after = signal_detector.get_current_state()

            # Log state transitions
            if state_after['long_state'] != prev_long_state:
                if state_after['long_state'] == 'ANCHOR_DETECTED':
                    anchors_long += 1
                    print(f"  [{current_time}] LONG ANCHOR detected at WT2={wt_subset.wt2.iloc[-1]:.2f}")
                elif state_after['long_state'] == 'TRIGGER_DETECTED':
                    triggers_long += 1
                    print(f"  [{current_time}] LONG TRIGGER detected")
                elif state_after['long_state'] == 'AWAITING_VWAP':
                    mfi_long += 1
                    print(f"  [{current_time}] LONG MFI confirmed, awaiting VWAP cross")
                prev_long_state = state_after['long_state']

            if state_after['short_state'] != prev_short_state:
                if state_after['short_state'] == 'ANCHOR_DETECTED':
                    anchors_short += 1
                    print(f"  [{current_time}] SHORT ANCHOR detected at WT2={wt_subset.wt2.iloc[-1]:.2f}")
                elif state_after['short_state'] == 'TRIGGER_DETECTED':
                    triggers_short += 1
                    print(f"  [{current_time}] SHORT TRIGGER detected")
                elif state_after['short_state'] == 'AWAITING_VWAP':
                    mfi_short += 1
                    print(f"  [{current_time}] SHORT MFI confirmed, awaiting VWAP cross")
                prev_short_state = state_after['short_state']

            if signal:
                is_long = signal.signal_type == SignalType.LONG
                direction = 'LONG' if is_long else 'SHORT'

                if is_long:
                    signals_long += 1
                else:
                    signals_short += 1

                # Calculate position parameters
                swing_lookback = 5
                if is_long:
                    recent_lows = df['low'].iloc[max(0, i-swing_lookback):i+1]
                    stop_loss = recent_lows.min() * 0.995
                else:
                    recent_highs = df['high'].iloc[max(0, i-swing_lookback):i+1]
                    stop_loss = recent_highs.max() * 1.005

                risk_distance = abs(current_price - stop_loss)
                reward_distance = risk_distance * risk_reward

                if is_long:
                    take_profit = current_price + reward_distance
                else:
                    take_profit = current_price - reward_distance

                risk_amount = balance * (risk_percent / 100)
                size = risk_amount / risk_distance if risk_distance > 0 else 0

                active_trade = {
                    'entry_time': current_time,
                    'direction': direction,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'size': size
                }

                print(f"\n  *** SIGNAL: {direction} at {current_time} ***")
                print(f"      Price: ${current_price:,.2f}")
                print(f"      SL: ${stop_loss:,.2f}, TP: ${take_profit:,.2f}")
                print(f"      WT2: {wt_subset.wt2.iloc[-1]:.2f}, MFI: {mf_subset.mfi.iloc[-1]:.2f}, VWAP: {wt_subset.vwap.iloc[-1]:.2f}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"  BACKTEST RESULTS: {asset} {timeframe}")
    print(f"{'='*70}")
    print(f"\n  DATA PERIOD: {df.index[0]} to {df.index[-1]}")
    print(f"  TOTAL CANDLES: {len(df)}")

    print(f"\n  STATE MACHINE FLOW:")
    print(f"    LONG:  Anchors={anchors_long} -> Triggers={triggers_long} -> MFI={mfi_long} -> Signals={signals_long}")
    print(f"    SHORT: Anchors={anchors_short} -> Triggers={triggers_short} -> MFI={mfi_short} -> Signals={signals_short}")

    if trades:
        winners = [t for t in trades if t['pnl'] > 0]
        losers = [t for t in trades if t['pnl'] <= 0]
        total_pnl = sum(t['pnl'] for t in trades)

        print(f"\n  TRADE SUMMARY:")
        print(f"    Total Trades: {len(trades)}")
        print(f"    Winners: {len(winners)}")
        print(f"    Losers: {len(losers)}")
        print(f"    Win Rate: {len(winners)/len(trades)*100:.1f}%")
        print(f"    Net P&L: ${total_pnl:+,.2f}")
        print(f"    Final Balance: ${balance:,.2f}")

        print(f"\n  ALL TRADES:")
        for i, t in enumerate(trades, 1):
            result = "WIN" if t['pnl'] > 0 else "LOSS"
            print(f"    {i}. {t['direction']} @ {t['entry_time']}")
            print(f"       Entry: ${t['entry_price']:,.2f} -> Exit: ${t['exit_price']:,.2f}")
            print(f"       P&L: ${t['pnl']:+,.2f} ({t['exit_reason']}) - {result}")
    else:
        print(f"\n  NO TRADES GENERATED")
        print(f"  This could mean:")
        print(f"    - No extreme WT2 levels reached (< -60 or > +60)")
        print(f"    - Trigger waves didn't form after anchors")
        print(f"    - MFI didn't confirm (wrong sign)")
        print(f"    - VWAP didn't cross zero")


def main():
    print("\n" + "="*70)
    print("  VMC TRADING BOT - REAL DATA BACKTEST WITH PROOF")
    print("="*70)
    print()
    print("  This script fetches REAL data from Binance Futures API")
    print("  You can verify prices on: https://www.binance.com/en/futures/BTCUSDT")
    print()

    # Test on multiple assets and timeframes
    test_configs = [
        ('BTC', '15m'),
        ('BTC', '1h'),
        ('BTC', '4h'),
        ('ETH', '15m'),
        ('ETH', '1h'),
        ('SOL', '15m'),
        ('SOL', '1h'),
    ]

    for asset, tf in test_configs:
        df = fetch_and_show_real_data(asset, tf, limit=500)

        if df is not None:
            run_backtest_on_real_data(df, asset, tf)

        time.sleep(0.5)  # Rate limiting
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
