"""SQLite-based trade persistence for recovery and history tracking."""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from utils.logger import get_logger

logger = get_logger("trade_store")


class TradeStore:
    """
    SQLite-based trade persistence.

    Provides:
    - Save trades on state changes (open, close, partial)
    - Load active trades on startup for recovery
    - Query trade history and statistics
    """

    def __init__(self, db_path: str = "data/trades.db"):
        """
        Initialize trade store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database connection and create tables."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info(f"Trade store initialized at {self.db_path}")

    def _create_tables(self) -> None:
        """Create required database tables."""
        cursor = self.conn.cursor()

        # Main trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                entry_price REAL NOT NULL,
                size REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                entry_order_id TEXT,
                sl_order_id TEXT,
                tp_order_id TEXT,
                opened_at TEXT,
                closed_at TEXT,
                exit_price REAL,
                pnl REAL DEFAULT 0.0,
                pnl_percent REAL DEFAULT 0.0,
                partial_closed_size REAL DEFAULT 0.0,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Index for active trade lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_status
            ON trades (status)
        """)

        # Index for symbol-based queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_symbol
            ON trades (symbol)
        """)

        # Sessions table for tracking bot sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                initial_balance REAL,
                final_balance REAL,
                trades_opened INTEGER DEFAULT 0,
                trades_closed INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0.0
            )
        """)

        self.conn.commit()

    def save_trade(self, trade_dict: Dict[str, Any]) -> bool:
        """
        Save or update a trade.

        Args:
            trade_dict: Trade data as dictionary (from Trade.to_dict())

        Returns:
            True if successful
        """
        try:
            cursor = self.conn.cursor()

            # Serialize metadata to JSON
            metadata = trade_dict.get("metadata")
            if metadata and isinstance(metadata, dict):
                metadata = json.dumps(metadata)

            cursor.execute("""
                INSERT OR REPLACE INTO trades (
                    trade_id, symbol, signal_type, entry_price, size,
                    stop_loss, take_profit, status, entry_order_id,
                    sl_order_id, tp_order_id, opened_at, closed_at,
                    exit_price, pnl, pnl_percent, partial_closed_size,
                    metadata, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_dict["trade_id"],
                trade_dict["symbol"],
                trade_dict["signal_type"],
                trade_dict["entry_price"],
                trade_dict.get("size", 0),
                trade_dict.get("stop_loss", 0),
                trade_dict.get("take_profit", 0),
                trade_dict["status"],
                trade_dict.get("entry_order_id"),
                trade_dict.get("sl_order_id"),
                trade_dict.get("tp_order_id"),
                trade_dict.get("opened_at"),
                trade_dict.get("closed_at"),
                trade_dict.get("exit_price"),
                trade_dict.get("pnl", 0),
                trade_dict.get("pnl_percent", 0),
                trade_dict.get("partial_closed_size", 0),
                metadata,
                datetime.utcnow().isoformat()
            ))

            self.conn.commit()
            logger.debug(f"Saved trade {trade_dict['trade_id']}")
            return True

        except Exception as e:
            logger.error(f"Failed to save trade: {e}")
            return False

    def load_active_trades(self) -> List[Dict[str, Any]]:
        """
        Load all active (non-closed) trades.

        Returns:
            List of trade dictionaries
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM trades
                WHERE status IN ('pending', 'open', 'partial_tp')
                ORDER BY opened_at ASC
            """)

            trades = []
            for row in cursor.fetchall():
                trade = dict(row)
                # Deserialize metadata
                if trade.get("metadata"):
                    try:
                        trade["metadata"] = json.loads(trade["metadata"])
                    except json.JSONDecodeError:
                        trade["metadata"] = {}
                trades.append(trade)

            logger.info(f"Loaded {len(trades)} active trades from database")
            return trades

        except Exception as e:
            logger.error(f"Failed to load active trades: {e}")
            return []

    def get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific trade by ID.

        Args:
            trade_id: Trade ID to look up

        Returns:
            Trade dictionary or None
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM trades WHERE trade_id = ?", (trade_id,))
            row = cursor.fetchone()

            if row:
                trade = dict(row)
                if trade.get("metadata"):
                    try:
                        trade["metadata"] = json.loads(trade["metadata"])
                    except json.JSONDecodeError:
                        trade["metadata"] = {}
                return trade

            return None

        except Exception as e:
            logger.error(f"Failed to get trade {trade_id}: {e}")
            return None

    def mark_trade_closed(
        self,
        trade_id: str,
        exit_price: float,
        pnl: float,
        pnl_percent: float,
        reason: str = "closed"
    ) -> bool:
        """
        Mark a trade as closed.

        Args:
            trade_id: Trade ID to close
            exit_price: Exit price
            pnl: Profit/loss amount
            pnl_percent: PnL percentage
            reason: Closure reason (for logging)

        Returns:
            True if successful
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE trades SET
                    status = 'closed',
                    exit_price = ?,
                    pnl = ?,
                    pnl_percent = ?,
                    closed_at = ?,
                    updated_at = ?
                WHERE trade_id = ?
            """, (
                exit_price,
                pnl,
                pnl_percent,
                datetime.utcnow().isoformat(),
                datetime.utcnow().isoformat(),
                trade_id
            ))

            self.conn.commit()
            logger.info(f"Marked trade {trade_id} as closed ({reason}): PnL=${pnl:.2f}")
            return True

        except Exception as e:
            logger.error(f"Failed to mark trade closed: {e}")
            return False

    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get closed trade history.

        Args:
            symbol: Filter by symbol (optional)
            limit: Maximum trades to return
            offset: Offset for pagination

        Returns:
            List of closed trade dictionaries
        """
        try:
            cursor = self.conn.cursor()

            if symbol:
                cursor.execute("""
                    SELECT * FROM trades
                    WHERE status = 'closed' AND symbol = ?
                    ORDER BY closed_at DESC
                    LIMIT ? OFFSET ?
                """, (symbol, limit, offset))
            else:
                cursor.execute("""
                    SELECT * FROM trades
                    WHERE status = 'closed'
                    ORDER BY closed_at DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))

            trades = []
            for row in cursor.fetchall():
                trade = dict(row)
                if trade.get("metadata"):
                    try:
                        trade["metadata"] = json.loads(trade["metadata"])
                    except json.JSONDecodeError:
                        trade["metadata"] = {}
                trades.append(trade)

            return trades

        except Exception as e:
            logger.error(f"Failed to get trade history: {e}")
            return []

    def get_stats(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate trading statistics from history.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            Statistics dictionary
        """
        try:
            cursor = self.conn.cursor()

            if symbol:
                cursor.execute("""
                    SELECT
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                        SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                        SUM(pnl) as total_pnl,
                        AVG(pnl) as avg_pnl,
                        MAX(pnl) as best_trade,
                        MIN(pnl) as worst_trade,
                        AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                        AVG(CASE WHEN pnl < 0 THEN pnl END) as avg_loss
                    FROM trades
                    WHERE status = 'closed' AND symbol = ?
                """, (symbol,))
            else:
                cursor.execute("""
                    SELECT
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                        SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                        SUM(pnl) as total_pnl,
                        AVG(pnl) as avg_pnl,
                        MAX(pnl) as best_trade,
                        MIN(pnl) as worst_trade,
                        AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                        AVG(CASE WHEN pnl < 0 THEN pnl END) as avg_loss
                    FROM trades
                    WHERE status = 'closed'
                """)

            row = cursor.fetchone()

            if not row or row["total_trades"] == 0:
                return {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate": 0,
                    "total_pnl": 0,
                    "avg_pnl": 0,
                    "best_trade": 0,
                    "worst_trade": 0,
                    "profit_factor": 0
                }

            total = row["total_trades"]
            winners = row["winning_trades"] or 0
            avg_win = row["avg_win"] or 0
            avg_loss = abs(row["avg_loss"] or 0)

            return {
                "total_trades": total,
                "winning_trades": winners,
                "losing_trades": row["losing_trades"] or 0,
                "win_rate": (winners / total * 100) if total > 0 else 0,
                "total_pnl": row["total_pnl"] or 0,
                "avg_pnl": row["avg_pnl"] or 0,
                "best_trade": row["best_trade"] or 0,
                "worst_trade": row["worst_trade"] or 0,
                "profit_factor": (avg_win / avg_loss) if avg_loss > 0 else 0
            }

        except Exception as e:
            logger.error(f"Failed to calculate stats: {e}")
            return {}

    def start_session(self, initial_balance: float = 0) -> int:
        """
        Start a new trading session.

        Args:
            initial_balance: Starting account balance

        Returns:
            Session ID
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (started_at, initial_balance)
                VALUES (?, ?)
            """, (datetime.utcnow().isoformat(), initial_balance))

            self.conn.commit()
            session_id = cursor.lastrowid
            logger.info(f"Started session {session_id} with balance ${initial_balance:.2f}")
            return session_id

        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            return -1

    def end_session(
        self,
        session_id: int,
        final_balance: float,
        trades_opened: int,
        trades_closed: int,
        total_pnl: float
    ) -> bool:
        """
        End a trading session.

        Args:
            session_id: Session ID to end
            final_balance: Ending account balance
            trades_opened: Number of trades opened
            trades_closed: Number of trades closed
            total_pnl: Total PnL for session

        Returns:
            True if successful
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE sessions SET
                    ended_at = ?,
                    final_balance = ?,
                    trades_opened = ?,
                    trades_closed = ?,
                    total_pnl = ?
                WHERE session_id = ?
            """, (
                datetime.utcnow().isoformat(),
                final_balance,
                trades_opened,
                trades_closed,
                total_pnl,
                session_id
            ))

            self.conn.commit()
            logger.info(f"Ended session {session_id}: PnL=${total_pnl:.2f}")
            return True

        except Exception as e:
            logger.error(f"Failed to end session: {e}")
            return False

    def cleanup_stale_trades(self, older_than_days: int = 30) -> int:
        """
        Clean up old closed trades to manage database size.

        Args:
            older_than_days: Delete trades older than this

        Returns:
            Number of deleted records
        """
        try:
            cursor = self.conn.cursor()
            cutoff_date = datetime.utcnow()
            # Note: Simple calculation, could use dateutil for more precision
            cutoff_str = cutoff_date.isoformat()

            cursor.execute("""
                DELETE FROM trades
                WHERE status = 'closed'
                AND closed_at < date(?, '-' || ? || ' days')
            """, (cutoff_str, older_than_days))

            deleted = cursor.rowcount
            self.conn.commit()

            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old trades")

            return deleted

        except Exception as e:
            logger.error(f"Failed to cleanup stale trades: {e}")
            return 0

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.debug("Trade store closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
