"""Daily Summary Logger for VMC Trading Bot.

Generates daily performance summaries at midnight UTC.
Logs to file and sends Discord notification if enabled.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

from utils.logger import get_logger

logger = get_logger("daily_summary")


@dataclass
class TradeSummary:
    """Summary of a single trade for daily report."""
    symbol: str
    direction: str  # LONG or SHORT
    entry_price: float
    exit_price: Optional[float]
    pnl: float
    exit_reason: str
    is_closed: bool


@dataclass
class DailySummary:
    """Daily performance summary."""
    date: str
    trades_opened: int
    trades_closed: int
    long_trades: int
    short_trades: int
    closed_trades: List[TradeSummary]
    daily_pnl: float
    running_balance: float
    wins: int
    losses: int
    active_positions: List[Dict[str, Any]]
    unrealized_pnl: float

    @property
    def win_rate(self) -> float:
        """Calculate win rate for closed trades."""
        total = self.wins + self.losses
        return (self.wins / total * 100) if total > 0 else 0.0

    def format(self) -> str:
        """Format summary as readable string."""
        lines = [
            "=" * 60,
            f"DAILY SUMMARY - {self.date}",
            "=" * 60,
            "",
            f"Trades Opened: {self.trades_opened} ({self.long_trades} LONG, {self.short_trades} SHORT)",
            f"Trades Closed: {self.trades_closed}",
        ]

        if self.closed_trades:
            for trade in self.closed_trades:
                pnl_str = f"+${trade.pnl:.2f}" if trade.pnl >= 0 else f"-${abs(trade.pnl):.2f}"
                lines.append(f"  - {trade.symbol} {trade.direction}: {pnl_str} ({trade.exit_reason})")

        lines.extend([
            "",
            f"Daily PnL: {'+'if self.daily_pnl >= 0 else ''}{self.daily_pnl:,.2f}",
            f"Running Balance: ${self.running_balance:,.2f}",
        ])

        if self.trades_closed > 0:
            lines.append(f"Win Rate Today: {self.win_rate:.0f}% ({self.wins}/{self.trades_closed})")

        if self.active_positions:
            lines.extend([
                "",
                f"Active Positions: {len(self.active_positions)}",
            ])
            for pos in self.active_positions:
                symbol = pos.get('symbol', 'Unknown')
                direction = pos.get('direction', 'Unknown')
                entry = pos.get('entry_price', 0)
                unrealized = pos.get('unrealized_pnl', 0)
                unrealized_str = f"+${unrealized:.2f}" if unrealized >= 0 else f"-${abs(unrealized):.2f}"
                lines.append(f"  - {symbol} {direction} @ ${entry:,.2f} (unrealized: {unrealized_str})")

        lines.extend([
            "",
            "=" * 60,
        ])

        return "\n".join(lines)

    def to_discord_embed(self) -> Dict[str, Any]:
        """Format summary for Discord embed."""
        # Determine color based on PnL
        if self.daily_pnl > 0:
            color = 0x00FF00  # Green
        elif self.daily_pnl < 0:
            color = 0xFF0000  # Red
        else:
            color = 0x808080  # Gray

        fields = [
            {
                "name": "Trades",
                "value": f"Opened: {self.trades_opened}\nClosed: {self.trades_closed}",
                "inline": True
            },
            {
                "name": "Daily PnL",
                "value": f"{'+'if self.daily_pnl >= 0 else ''}${self.daily_pnl:,.2f}",
                "inline": True
            },
            {
                "name": "Balance",
                "value": f"${self.running_balance:,.2f}",
                "inline": True
            },
        ]

        if self.trades_closed > 0:
            fields.append({
                "name": "Win Rate",
                "value": f"{self.win_rate:.0f}% ({self.wins}/{self.trades_closed})",
                "inline": True
            })

        if self.active_positions:
            positions_text = "\n".join([
                f"{p.get('symbol', '?')} {p.get('direction', '?')}"
                for p in self.active_positions[:5]  # Limit to 5
            ])
            if len(self.active_positions) > 5:
                positions_text += f"\n... and {len(self.active_positions) - 5} more"

            fields.append({
                "name": f"Active Positions ({len(self.active_positions)})",
                "value": positions_text,
                "inline": False
            })

        return {
            "title": f"Daily Summary - {self.date}",
            "color": color,
            "fields": fields,
            "footer": {"text": "VMC Trading Bot V6"}
        }


class DailySummaryLogger:
    """Manages daily summary generation and logging.

    Tracks trades throughout the day and generates a summary at midnight UTC.
    """

    def __init__(self, log_dir: str = "logs"):
        """Initialize the daily summary logger.

        Args:
            log_dir: Directory to write daily summary logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Track the last day we generated a summary
        self._last_summary_date: Optional[str] = None

        # Daily tracking
        self._trades_opened_today: List[TradeSummary] = []
        self._trades_closed_today: List[TradeSummary] = []
        self._daily_start_balance: float = 0.0

    def should_generate(self, current_time: Optional[datetime] = None) -> bool:
        """Check if we should generate a daily summary.

        Summary is generated when the day rolls over (after midnight UTC).

        Args:
            current_time: Current time (defaults to now UTC)

        Returns:
            True if summary should be generated
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Ensure timezone aware
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        current_date = current_time.strftime("%Y-%m-%d")

        # Generate if we haven't generated for today yet
        if self._last_summary_date is None:
            # First run, set last date but don't generate
            self._last_summary_date = current_date
            return False

        if current_date != self._last_summary_date:
            return True

        return False

    def record_trade_opened(
        self,
        symbol: str,
        direction: str,
        entry_price: float
    ) -> None:
        """Record a trade that was opened today.

        Args:
            symbol: Trading symbol
            direction: LONG or SHORT
            entry_price: Entry price
        """
        self._trades_opened_today.append(TradeSummary(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=None,
            pnl=0.0,
            exit_reason="",
            is_closed=False
        ))

    def record_trade_closed(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        exit_reason: str
    ) -> None:
        """Record a trade that was closed today.

        Args:
            symbol: Trading symbol
            direction: LONG or SHORT
            entry_price: Entry price
            exit_price: Exit price
            pnl: Profit/loss
            exit_reason: Reason for exit
        """
        self._trades_closed_today.append(TradeSummary(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            exit_reason=exit_reason,
            is_closed=True
        ))

    def set_start_balance(self, balance: float) -> None:
        """Set the starting balance for the day.

        Args:
            balance: Account balance at start of day
        """
        self._daily_start_balance = balance

    def generate(
        self,
        current_balance: float,
        active_positions: Optional[List[Dict[str, Any]]] = None,
        current_time: Optional[datetime] = None
    ) -> DailySummary:
        """Generate the daily summary.

        Args:
            current_balance: Current account balance
            active_positions: List of active positions with details
            current_time: Current time (defaults to now UTC)

        Returns:
            DailySummary object
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # The summary is for the previous day
        yesterday = current_time - timedelta(days=1)
        date_str = yesterday.strftime("%Y-%m-%d")

        # Calculate stats
        long_trades = sum(1 for t in self._trades_opened_today if t.direction == "LONG")
        short_trades = sum(1 for t in self._trades_opened_today if t.direction == "SHORT")
        wins = sum(1 for t in self._trades_closed_today if t.pnl > 0)
        losses = sum(1 for t in self._trades_closed_today if t.pnl <= 0)
        daily_pnl = sum(t.pnl for t in self._trades_closed_today)

        # Calculate unrealized PnL from active positions
        unrealized_pnl = 0.0
        if active_positions:
            unrealized_pnl = sum(p.get('unrealized_pnl', 0) for p in active_positions)

        summary = DailySummary(
            date=date_str,
            trades_opened=len(self._trades_opened_today),
            trades_closed=len(self._trades_closed_today),
            long_trades=long_trades,
            short_trades=short_trades,
            closed_trades=self._trades_closed_today.copy(),
            daily_pnl=daily_pnl,
            running_balance=current_balance,
            wins=wins,
            losses=losses,
            active_positions=active_positions or [],
            unrealized_pnl=unrealized_pnl
        )

        # Log to file
        self._write_to_file(summary)

        # Update tracking
        self._last_summary_date = current_time.strftime("%Y-%m-%d")
        self._reset_daily_tracking()

        logger.info(f"Generated daily summary for {date_str}")

        return summary

    def _write_to_file(self, summary: DailySummary) -> None:
        """Write summary to log file.

        Args:
            summary: DailySummary to write
        """
        log_file = self.log_dir / "daily_summary.log"

        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{summary.format()}\n")
        except Exception as e:
            logger.error(f"Failed to write daily summary to file: {e}")

    def _reset_daily_tracking(self) -> None:
        """Reset tracking for new day."""
        self._trades_opened_today = []
        self._trades_closed_today = []
        self._daily_start_balance = 0.0

    def get_today_stats(self) -> Dict[str, Any]:
        """Get current day statistics (before summary).

        Returns:
            Dict with today's stats
        """
        return {
            "trades_opened": len(self._trades_opened_today),
            "trades_closed": len(self._trades_closed_today),
            "daily_pnl": sum(t.pnl for t in self._trades_closed_today),
            "wins": sum(1 for t in self._trades_closed_today if t.pnl > 0),
            "losses": sum(1 for t in self._trades_closed_today if t.pnl <= 0),
        }
