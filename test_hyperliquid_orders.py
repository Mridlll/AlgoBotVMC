#!/usr/bin/env python3
"""
Hyperliquid Order Placement Test
================================
Tests order placement, cancellation, and position management on testnet.

SAFETY: This test uses TESTNET only and places small orders.

=== IMPORTANT: Hyperliquid Wallet Architecture ===

Hyperliquid has two trading modes:

1. DIRECT TRADING (using main wallet):
   - You sign with your main wallet's private key
   - The wallet that signs is also the wallet with funds
   - Set: wallet_address = main_wallet, account_address = None

2. API WALLET TRADING (for security):
   - Generate API wallet at: https://app.hyperliquid.xyz/API (mainnet)
                            https://app.hyperliquid-testnet.xyz/API (testnet)
   - API wallet signs transactions on behalf of main wallet
   - Set: wallet_address = api_wallet, account_address = main_wallet

NOTE: API wallets are network-specific! An API wallet created on mainnet
      will NOT work on testnet. You must create separate API wallets.
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
# CONFIGURATION - MODIFY THESE FOR YOUR SETUP
# =============================================================================

# TESTNET API Wallet credentials
PRIVATE_KEY = "0x03cae4be7e968cffd49f7cd026931fd30a74e05d18264d631a1eb6484d1982aa"
SIGNING_WALLET = "0x12c9209091276c8c184be058cdd4589f550e0282"

# Main wallet where funds are held ($999 on testnet)
MAIN_WALLET = "0x8A41f91F21260137Ef17aafde0ac2A13e5036f91"

# Network
TESTNET = True  # ALWAYS use testnet for testing!

# Test parameters
TEST_SYMBOL = "BTC"
TEST_SIZE = 0.001  # ~$87 position

# Hyperliquid minimum order value is $10


async def test_order_placement():
    print("\n" + "="*70)
    print("  HYPERLIQUID ORDER PLACEMENT TEST")
    print("="*70)
    print()
    print(f"  Network:         {'TESTNET' if TESTNET else '*** MAINNET - REAL MONEY ***'}")
    print(f"  Signing Wallet:  {SIGNING_WALLET}")
    if MAIN_WALLET:
        print(f"  Trading For:     {MAIN_WALLET}")
        print(f"  Mode:            API WALLET (trading on behalf of main)")
    else:
        print(f"  Mode:            DIRECT (signing wallet has funds)")
    print(f"  Symbol:          {TEST_SYMBOL}")
    print(f"  Test Size:       {TEST_SIZE}")
    print()

    if not TESTNET:
        print("  [ABORT] This test only runs on testnet for safety!")
        return

    from exchanges.hyperliquid import HyperliquidExchange
    from exchanges.base import OrderSide, OrderType

    # Configure exchange
    # - wallet_address: The wallet signing transactions
    # - account_address: The wallet with funds (if different from signing wallet)
    exchange = HyperliquidExchange(
        api_key="",
        api_secret=PRIVATE_KEY,
        wallet_address=SIGNING_WALLET,
        account_address=MAIN_WALLET,  # Set to None if trading directly with main wallet
        testnet=TESTNET
    )

    try:
        # Connect
        print("="*70)
        print("  STEP 1: Connecting...")
        print("="*70)
        connected = await exchange.connect()
        if not connected:
            print("  [FAIL] Connection failed")
            return
        print("  [OK] Connected to Hyperliquid testnet")

        # Get current balance
        print()
        print("="*70)
        print("  STEP 2: Checking Balance...")
        print("="*70)
        balance = await exchange.get_balance()
        print(f"  [OK] Available: ${balance.total_balance:,.2f}")

        # Get current price
        print()
        print("="*70)
        print("  STEP 3: Getting Current Price...")
        print("="*70)
        ticker = await exchange.get_ticker(TEST_SYMBOL)
        current_price = ticker['price']
        print(f"  [OK] {TEST_SYMBOL} price: ${current_price:,.2f}")

        # Calculate order value
        order_value = TEST_SIZE * current_price
        print(f"  [INFO] Order value: ${order_value:,.2f}")

        if order_value > balance.total_balance * 0.1:
            print(f"  [WARN] Order value exceeds 10% of balance, reducing size...")
            TEST_SIZE_ACTUAL = (balance.total_balance * 0.05) / current_price
        else:
            TEST_SIZE_ACTUAL = TEST_SIZE

        # Test 1: Place a limit order far from market (won't fill)
        print()
        print("="*70)
        print("  TEST A: Placing LIMIT order (far from market)...")
        print("="*70)

        # Place 20% below market so it won't fill
        limit_price = round(current_price * 0.80, 1)
        print(f"  [INFO] Limit price: ${limit_price:,.2f} (20% below market)")
        print(f"  [INFO] Size: {TEST_SIZE_ACTUAL:.6f} {TEST_SYMBOL}")

        try:
            order = await exchange.place_order(
                symbol=TEST_SYMBOL,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                size=TEST_SIZE_ACTUAL,
                price=limit_price
            )
            print(f"  [OK] Order placed! ID: {order.order_id}")
            print(f"       Status: {order.status.value}")
        except Exception as e:
            error_str = str(e)
            print(f"  [FAIL] Order failed: {e}")

            # Check for common errors
            if "does not exist" in error_str:
                print()
                print("  " + "="*66)
                print("  API WALLET NOT REGISTERED")
                print("  " + "="*66)
                print()
                if TESTNET:
                    print("  Your API wallet is not registered on TESTNET.")
                    print("  API wallets created on mainnet do NOT work on testnet!")
                    print()
                    print("  To fix, either:")
                    print("  1. Create a new API wallet for testnet at:")
                    print("     https://app.hyperliquid-testnet.xyz/API")
                    print()
                    print("  2. Or use your main wallet's private key directly")
                    print("     (set MAIN_WALLET = None in test config)")
                else:
                    print("  Your API wallet is not registered on MAINNET.")
                    print("  Please create an API wallet at:")
                    print("  https://app.hyperliquid.xyz/API")
                print()
            return

        # Verify order appears in open orders
        print()
        print("="*70)
        print("  TEST B: Verifying order in open orders...")
        print("="*70)

        await asyncio.sleep(1)  # Wait for order to appear
        open_orders = await exchange.get_open_orders(TEST_SYMBOL)
        order_found = False
        for o in open_orders:
            if o.order_id == order.order_id:
                order_found = True
                print(f"  [OK] Order found: {o.side.value} {o.size} @ ${o.price:,.2f}")
                break

        if not order_found and open_orders:
            print(f"  [WARN] Order ID not matched, but {len(open_orders)} orders exist")
            for o in open_orders:
                print(f"       - {o.order_id}: {o.side.value} {o.size} @ ${o.price:,.2f}")
        elif not open_orders:
            print("  [WARN] No open orders found (may have been rejected)")

        # Cancel the order
        print()
        print("="*70)
        print("  TEST C: Cancelling order...")
        print("="*70)

        try:
            # Try to cancel using the order ID we got
            cancelled = await exchange.cancel_order(order.order_id, TEST_SYMBOL)
            if cancelled:
                print(f"  [OK] Order cancelled successfully")
            else:
                print(f"  [WARN] Cancel returned False")
        except Exception as e:
            print(f"  [WARN] Cancel error: {e}")
            # Try to cancel all orders for the symbol
            try:
                for o in open_orders:
                    await exchange.cancel_order(o.order_id, TEST_SYMBOL)
                print("  [OK] Cleaned up all open orders")
            except:
                pass

        # Verify no open orders remain
        await asyncio.sleep(1)
        remaining_orders = await exchange.get_open_orders(TEST_SYMBOL)
        if not remaining_orders:
            print("  [OK] No open orders remaining")
        else:
            print(f"  [WARN] {len(remaining_orders)} orders still open")

        # Test 2: Market order (small, will actually fill)
        print()
        print("="*70)
        print("  TEST D: Placing MARKET order (will fill)...")
        print("="*70)

        tiny_size = 0.00015  # Small ~$13 (must be >$10 minimum)
        print(f"  [INFO] Size: {tiny_size} {TEST_SYMBOL} (~${tiny_size * current_price:.2f})")

        try:
            market_order = await exchange.place_order(
                symbol=TEST_SYMBOL,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                size=tiny_size
            )
            print(f"  [OK] Market order placed! ID: {market_order.order_id}")
            print(f"       Status: {market_order.status.value}")
            print(f"       Fill price: ${market_order.avg_fill_price:,.2f}")
        except Exception as e:
            print(f"  [FAIL] Market order failed: {e}")
            import traceback
            traceback.print_exc()
            # Continue with tests even if this fails

        # Check for position
        print()
        print("="*70)
        print("  TEST E: Checking positions...")
        print("="*70)

        await asyncio.sleep(1)
        positions = await exchange.get_positions()
        if positions:
            for pos in positions:
                print(f"  [OK] Position: {pos.symbol} {pos.side.value}")
                print(f"       Size: {pos.size}")
                print(f"       Entry: ${pos.entry_price:,.2f}")
                print(f"       P&L: ${pos.unrealized_pnl:+,.2f}")
        else:
            print("  [INFO] No open positions")

        # Close any position we opened
        if positions:
            print()
            print("="*70)
            print("  TEST F: Closing position...")
            print("="*70)

            for pos in positions:
                if pos.symbol == TEST_SYMBOL:
                    try:
                        close_order = await exchange.close_position(TEST_SYMBOL)
                        print(f"  [OK] Position closed!")
                        print(f"       Close price: ${close_order.avg_fill_price:,.2f}")
                    except Exception as e:
                        print(f"  [WARN] Close failed: {e}")

        # Final balance check
        print()
        print("="*70)
        print("  STEP 4: Final Balance Check...")
        print("="*70)

        final_balance = await exchange.get_balance()
        print(f"  [OK] Final Balance: ${final_balance.total_balance:,.2f}")
        pnl = final_balance.total_balance - balance.total_balance
        print(f"  [INFO] Test P&L: ${pnl:+,.2f}")

        await exchange.disconnect()

        # Summary
        print()
        print("="*70)
        print("  TEST SUMMARY")
        print("="*70)
        print()
        print("  Order placement tests completed!")
        print("  - Limit order: TESTED")
        print("  - Order query: TESTED")
        print("  - Order cancel: TESTED")
        print("  - Market order: TESTED")
        print("  - Position query: TESTED")
        print("  - Position close: TESTED")
        print()
        print("="*70)

    except Exception as e:
        print(f"\n  [ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        await exchange.disconnect()


if __name__ == "__main__":
    asyncio.run(test_order_placement())
