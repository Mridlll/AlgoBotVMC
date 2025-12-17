"""Tests for Discord notifications."""

import asyncio
import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from notifications.discord import DiscordNotifier


# Test webhook URL (replace with your own for testing)
TEST_WEBHOOK_URL = "https://discord.com/api/webhooks/1450580072938340432/vE53l5FR351DaIdSULYjstgJ0lgYC2fqYaagH7G15OpkznivnoVs-Rz1QSms8s1HVATk"


async def test_bot_started():
    """Test bot started notification."""
    notifier = DiscordNotifier(
        webhook_url=TEST_WEBHOOK_URL,
        notify_on_signal=True,
        notify_on_trade_open=True,
        notify_on_trade_close=True,
        notify_on_error=True
    )

    try:
        result = await notifier.notify("bot_started", {
            "assets": ["BTC", "ETH"]
        })
        print(f"Bot started notification: {'SUCCESS' if result else 'FAILED'}")
        return result
    finally:
        await notifier.close()


async def test_signal_detected():
    """Test signal detected notification."""
    notifier = DiscordNotifier(webhook_url=TEST_WEBHOOK_URL)

    try:
        result = await notifier.notify("signal_detected", {
            "signal_type": "long",
            "symbol": "BTC",
            "entry_price": 98500.00,
            "wt1": -45.2,
            "wt2": -52.8,
            "vwap": 7.6,
            "confidence": 0.85
        })
        print(f"Signal detected notification: {'SUCCESS' if result else 'FAILED'}")
        return result
    finally:
        await notifier.close()


async def test_trade_opened():
    """Test trade opened notification."""
    notifier = DiscordNotifier(webhook_url=TEST_WEBHOOK_URL)

    try:
        result = await notifier.notify("trade_opened", {
            "signal_type": "long",
            "symbol": "BTC",
            "entry_price": 98500.00,
            "size": 0.0305,
            "stop_loss": 97200.00,
            "take_profit": 101100.00,
            "trade_id": "TEST-001"
        })
        print(f"Trade opened notification: {'SUCCESS' if result else 'FAILED'}")
        return result
    finally:
        await notifier.close()


async def test_trade_closed():
    """Test trade closed notification."""
    notifier = DiscordNotifier(webhook_url=TEST_WEBHOOK_URL)

    try:
        result = await notifier.notify("trade_closed", {
            "signal_type": "long",
            "symbol": "BTC",
            "entry_price": 98500.00,
            "exit_price": 101100.00,
            "pnl": 79.30,
            "pnl_percent": 2.64
        })
        print(f"Trade closed notification: {'SUCCESS' if result else 'FAILED'}")
        return result
    finally:
        await notifier.close()


async def test_error_notification():
    """Test error notification."""
    notifier = DiscordNotifier(webhook_url=TEST_WEBHOOK_URL)

    try:
        result = await notifier.notify("error", {
            "message": "Test error notification - VMC Bot testing complete!"
        })
        print(f"Error notification: {'SUCCESS' if result else 'FAILED'}")
        return result
    finally:
        await notifier.close()


async def run_all_tests():
    """Run all Discord notification tests."""
    print("\n" + "="*50)
    print("DISCORD NOTIFICATION TESTS")
    print("="*50 + "\n")

    results = []

    print("1. Testing bot_started notification...")
    results.append(await test_bot_started())
    await asyncio.sleep(1)  # Rate limit buffer

    print("2. Testing signal_detected notification...")
    results.append(await test_signal_detected())
    await asyncio.sleep(1)

    print("3. Testing trade_opened notification...")
    results.append(await test_trade_opened())
    await asyncio.sleep(1)

    print("4. Testing trade_closed notification...")
    results.append(await test_trade_closed())
    await asyncio.sleep(1)

    print("5. Testing error notification...")
    results.append(await test_error_notification())

    print("\n" + "="*50)
    print(f"RESULTS: {sum(results)}/{len(results)} notifications sent successfully")
    print("="*50 + "\n")

    return all(results)


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
