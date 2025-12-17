"""Discord webhook notifications."""

import sys
from pathlib import Path

# Add project root and src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
import aiohttp

from utils.logger import get_logger

logger = get_logger("discord")


class DiscordNotifier:
    """
    Discord webhook notification sender.

    Sends formatted messages for trading events:
    - Signal detected
    - Trade opened
    - Trade closed
    - Errors
    """

    # Colors for embeds
    COLOR_SUCCESS = 0x00FF00  # Green
    COLOR_ERROR = 0xFF0000    # Red
    COLOR_WARNING = 0xFFA500  # Orange
    COLOR_INFO = 0x0099FF     # Blue
    COLOR_LONG = 0x00FF00     # Green
    COLOR_SHORT = 0xFF0000    # Red

    def __init__(
        self,
        webhook_url: str,
        notify_on_signal: bool = True,
        notify_on_trade_open: bool = True,
        notify_on_trade_close: bool = True,
        notify_on_error: bool = True
    ):
        """
        Initialize Discord notifier.

        Args:
            webhook_url: Discord webhook URL
            notify_on_signal: Send notifications for signals
            notify_on_trade_open: Send notifications for opened trades
            notify_on_trade_close: Send notifications for closed trades
            notify_on_error: Send notifications for errors
        """
        self.webhook_url = webhook_url
        self.notify_on_signal = notify_on_signal
        self.notify_on_trade_open = notify_on_trade_open
        self.notify_on_trade_close = notify_on_trade_close
        self.notify_on_error = notify_on_error
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def send_message(
        self,
        content: Optional[str] = None,
        embeds: Optional[List[Dict[str, Any]]] = None,
        username: str = "VMC Trading Bot"
    ) -> bool:
        """
        Send a message to Discord webhook.

        Args:
            content: Text content
            embeds: List of embed objects
            username: Bot username to display

        Returns:
            True if sent successfully
        """
        if not self.webhook_url:
            logger.warning("Discord webhook URL not configured")
            return False

        try:
            session = await self._get_session()

            payload = {
                "username": username,
            }

            if content:
                payload["content"] = content
            if embeds:
                payload["embeds"] = embeds

            async with session.post(self.webhook_url, json=payload) as response:
                if response.status == 204:
                    return True
                else:
                    logger.warning(f"Discord webhook returned status {response.status}")
                    return False

        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False

    async def notify(self, event_type: str, data: Dict[str, Any]) -> bool:
        """
        Send notification for an event.

        Args:
            event_type: Type of event
            data: Event data

        Returns:
            True if notification sent
        """
        if event_type == "signal_detected" and self.notify_on_signal:
            return await self._notify_signal(data)
        elif event_type == "trade_opened" and self.notify_on_trade_open:
            return await self._notify_trade_opened(data)
        elif event_type == "trade_closed" and self.notify_on_trade_close:
            return await self._notify_trade_closed(data)
        elif event_type == "error" and self.notify_on_error:
            return await self._notify_error(data)
        elif event_type == "bot_started":
            return await self._notify_bot_started(data)
        elif event_type == "bot_stopped":
            return await self._notify_bot_stopped(data)

        return False

    async def _notify_signal(self, data: Dict[str, Any]) -> bool:
        """Send signal notification."""
        signal_type = data.get("signal_type", "unknown").upper()
        is_long = signal_type == "LONG"

        embed = {
            "title": f"{'📈' if is_long else '📉'} {signal_type} Signal Detected",
            "color": self.COLOR_LONG if is_long else self.COLOR_SHORT,
            "fields": [
                {"name": "Symbol", "value": data.get("symbol", "N/A"), "inline": True},
                {"name": "Entry Price", "value": f"${data.get('entry_price', 0):,.2f}", "inline": True},
                {"name": "Confidence", "value": f"{data.get('confidence', 1) * 100:.0f}%", "inline": True},
                {"name": "WT1", "value": f"{data.get('wt1', 0):.2f}", "inline": True},
                {"name": "WT2", "value": f"{data.get('wt2', 0):.2f}", "inline": True},
                {"name": "VWAP", "value": f"{data.get('vwap', 0):.2f}", "inline": True},
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "VMC Trading Bot"}
        }

        return await self.send_message(embeds=[embed])

    async def _notify_trade_opened(self, data: Dict[str, Any]) -> bool:
        """Send trade opened notification."""
        signal_type = data.get("signal_type", "unknown").upper()
        is_long = signal_type == "LONG"

        embed = {
            "title": f"{'🟢' if is_long else '🔴'} Trade Opened - {signal_type}",
            "color": self.COLOR_LONG if is_long else self.COLOR_SHORT,
            "fields": [
                {"name": "Symbol", "value": data.get("symbol", "N/A"), "inline": True},
                {"name": "Entry Price", "value": f"${data.get('entry_price', 0):,.2f}", "inline": True},
                {"name": "Size", "value": f"{data.get('size', 0):.4f}", "inline": True},
                {"name": "Stop Loss", "value": f"${data.get('stop_loss', 0):,.2f}", "inline": True},
                {"name": "Take Profit", "value": f"${data.get('take_profit', 0):,.2f}", "inline": True},
                {"name": "Trade ID", "value": data.get("trade_id", "N/A"), "inline": True},
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "VMC Trading Bot"}
        }

        return await self.send_message(embeds=[embed])

    async def _notify_trade_closed(self, data: Dict[str, Any]) -> bool:
        """Send trade closed notification."""
        pnl = data.get("pnl", 0)
        pnl_percent = data.get("pnl_percent", 0)
        is_profit = pnl >= 0

        embed = {
            "title": f"{'💰' if is_profit else '💸'} Trade Closed",
            "color": self.COLOR_SUCCESS if is_profit else self.COLOR_ERROR,
            "fields": [
                {"name": "Symbol", "value": data.get("symbol", "N/A"), "inline": True},
                {"name": "Type", "value": data.get("signal_type", "N/A").upper(), "inline": True},
                {"name": "Exit Price", "value": f"${data.get('exit_price', 0):,.2f}", "inline": True},
                {"name": "Entry Price", "value": f"${data.get('entry_price', 0):,.2f}", "inline": True},
                {"name": "PnL", "value": f"${pnl:,.2f}", "inline": True},
                {"name": "PnL %", "value": f"{'+' if pnl_percent >= 0 else ''}{pnl_percent:.2f}%", "inline": True},
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "VMC Trading Bot"}
        }

        return await self.send_message(embeds=[embed])

    async def _notify_error(self, data: Dict[str, Any]) -> bool:
        """Send error notification."""
        embed = {
            "title": "⚠️ Error Occurred",
            "color": self.COLOR_ERROR,
            "description": data.get("message", "Unknown error"),
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "VMC Trading Bot"}
        }

        return await self.send_message(embeds=[embed])

    async def _notify_bot_started(self, data: Dict[str, Any]) -> bool:
        """Send bot started notification."""
        assets = data.get("assets", [])

        embed = {
            "title": "🚀 Bot Started",
            "color": self.COLOR_SUCCESS,
            "fields": [
                {"name": "Trading Assets", "value": ", ".join(assets) or "None", "inline": False},
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "VMC Trading Bot"}
        }

        return await self.send_message(embeds=[embed])

    async def _notify_bot_stopped(self, data: Dict[str, Any]) -> bool:
        """Send bot stopped notification."""
        embed = {
            "title": "🛑 Bot Stopped",
            "color": self.COLOR_WARNING,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "VMC Trading Bot"}
        }

        return await self.send_message(embeds=[embed])
