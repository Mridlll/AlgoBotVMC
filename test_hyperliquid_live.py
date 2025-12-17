#!/usr/bin/env python3
"""
Hyperliquid Live Connection Test
================================
Tests connection to Hyperliquid with provided credentials.

Hyperliquid Wallet Architecture:
- API Wallet: Generated at https://app.hyperliquid.xyz/API
  Used for signing transactions (private key required)
- Main Wallet: Your main account that holds the funds
  Used for querying balance/positions
"""

import sys
from pathlib import Path

# Fix Windows encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import asyncio
from datetime import datetime

# =============================================================================
# CREDENTIALS
# =============================================================================
# TESTNET API Wallet (created at https://app.hyperliquid-testnet.xyz/API)
PRIVATE_KEY = "0x03cae4be7e968cffd49f7cd026931fd30a74e05d18264d631a1eb6484d1982aa"
API_WALLET = "0x12c9209091276c8c184be058cdd4589f550e0282"  # Derived from private key

# Main wallet where funds are held (~$999 on testnet)
MAIN_WALLET = "0x8A41f91F21260137Ef17aafde0ac2A13e5036f91"

# Network
TESTNET = True  # Using TESTNET for safer testing


async def test_hyperliquid():
    print("\n" + "="*70)
    print("  HYPERLIQUID CONNECTION TEST")
    print("="*70)
    print()
    print(f"  API Wallet (signing):    {API_WALLET}")
    print(f"  Main Wallet (funds):     {MAIN_WALLET}")
    print(f"  Network:                 {'TESTNET' if TESTNET else 'MAINNET'}")
    print()

    # Import after path setup
    from exchanges.hyperliquid import HyperliquidExchange

    # Initialize client
    print("="*70)
    print("  TEST 1: Connecting to Hyperliquid...")
    print("="*70)

    exchange = HyperliquidExchange(
        api_key="",  # Not used for Hyperliquid
        api_secret=PRIVATE_KEY,
        wallet_address=API_WALLET,
        account_address=MAIN_WALLET,  # Main wallet where funds are held
        testnet=TESTNET
    )

    try:
        connected = await exchange.connect()
        if connected:
            print(f"  [OK] SUCCESS: Connected to Hyperliquid {'testnet' if TESTNET else 'mainnet'}!")
        else:
            print("  [FAIL] FAILED: Could not connect")
            return
    except Exception as e:
        print(f"  [FAIL] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 2: Get account balance
    print()
    print("="*70)
    print("  TEST 2: Fetching Account Balance...")
    print("="*70)

    try:
        balance = await exchange.get_balance()
        print(f"  [OK] SUCCESS: Account balance retrieved!")
        print()
        print(f"      Total Balance:     ${balance.total_balance:,.2f}")
        print(f"      Available Balance: ${balance.available_balance:,.2f}")
        print(f"      Used Margin:       ${balance.used_margin:,.2f}")
        print(f"      Unrealized P&L:    ${balance.unrealized_pnl:+,.2f}")
        print(f"      Currency:          {balance.currency}")
    except Exception as e:
        print(f"  [FAIL] ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Get open positions
    print()
    print("="*70)
    print("  TEST 3: Fetching Open Positions...")
    print("="*70)

    try:
        positions = await exchange.get_positions()
        print(f"  [OK] SUCCESS: Positions retrieved!")
        print()
        if positions:
            print(f"      Active Positions: {len(positions)}")
            for pos in positions:
                print(f"      - {pos.symbol} {pos.side.value}: {pos.size} @ ${pos.entry_price:,.2f}")
                print(f"        Unrealized P&L: ${pos.unrealized_pnl:+,.2f}")
                print(f"        Leverage: {pos.leverage}x")
        else:
            print("      No open positions")
    except Exception as e:
        print(f"  [FAIL] ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Test 4: Get ticker prices
    print()
    print("="*70)
    print("  TEST 4: Fetching Current Prices...")
    print("="*70)

    for symbol in ['BTC', 'ETH', 'SOL']:
        try:
            ticker = await exchange.get_ticker(symbol)
            print(f"  [OK] {symbol}: ${ticker['price']:,.2f}")
        except Exception as e:
            print(f"  [FAIL] {symbol}: Error - {e}")

    # Test 5: Get candles
    print()
    print("="*70)
    print("  TEST 5: Fetching BTC 1h Candles...")
    print("="*70)

    try:
        candles = await exchange.get_candles("BTC", "1h", limit=10)
        print(f"  [OK] SUCCESS: Retrieved {len(candles)} candles")
        print()
        print(f"      {'Timestamp':<20} {'Open':>12} {'High':>12} {'Low':>12} {'Close':>12}")
        print(f"      {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        for c in candles[-5:]:
            print(f"      {c.timestamp.strftime('%Y-%m-%d %H:%M'):<20} ${c.open:>11,.2f} ${c.high:>11,.2f} ${c.low:>11,.2f} ${c.close:>11,.2f}")
    except Exception as e:
        print(f"  [FAIL] ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Test 6: Get symbol info
    print()
    print("="*70)
    print("  TEST 6: Fetching Symbol Info (BTC)...")
    print("="*70)

    try:
        info = await exchange.get_symbol_info("BTC")
        print(f"  [OK] SUCCESS: Symbol info retrieved!")
        print()
        print(f"      Symbol:       {info.symbol}")
        print(f"      Base Asset:   {info.base_asset}")
        print(f"      Quote Asset:  {info.quote_asset}")
        print(f"      Tick Size:    {info.tick_size}")
        print(f"      Lot Size:     {info.lot_size}")
        print(f"      Min Size:     {info.min_size}")
        print(f"      Max Leverage: {info.max_leverage}x")
    except Exception as e:
        print(f"  [FAIL] ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Test 7: Get open orders
    print()
    print("="*70)
    print("  TEST 7: Fetching Open Orders...")
    print("="*70)

    try:
        orders = await exchange.get_open_orders()
        print(f"  [OK] SUCCESS: Open orders retrieved!")
        if orders:
            print(f"      Active Orders: {len(orders)}")
            for order in orders:
                print(f"      - {order.symbol} {order.side.value} {order.size} @ ${order.price:,.2f}")
        else:
            print("      No open orders")
    except Exception as e:
        print(f"  [FAIL] ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Disconnect
    await exchange.disconnect()

    # Summary
    print()
    print("="*70)
    print("  TEST SUMMARY")
    print("="*70)
    print()
    print("  All read-only tests completed!")
    print()
    print("  NOTE: No orders were placed (this was a read-only test)")
    print("  To test order placement, explicit confirmation is needed")
    print()
    print("="*70)


if __name__ == "__main__":
    asyncio.run(test_hyperliquid())
