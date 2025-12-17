"""State management for the trading bot."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from pathlib import Path

from strategy.signals import SignalState, Signal, SignalType


class BotState(str, Enum):
    """Bot operational state."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class TimeframeState:
    """State for a specific timeframe within an asset (for MTF mode)."""
    timeframe: str
    long_state: SignalState = SignalState.IDLE
    short_state: SignalState = SignalState.IDLE
    wt2: float = 0.0  # Current WT2 value
    bias: str = "neutral"  # "bullish", "bearish", or "neutral"
    last_update: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "long_state": self.long_state.name,
            "short_state": self.short_state.name,
            "wt2": self.wt2,
            "bias": self.bias,
            "last_update": self.last_update.isoformat() if self.last_update else None,
        }


@dataclass
class AssetState:
    """State for a single trading asset."""
    symbol: str
    long_state: SignalState = SignalState.IDLE
    short_state: SignalState = SignalState.IDLE
    last_signal: Optional[Signal] = None
    last_update: Optional[datetime] = None

    # Anchor wave tracking
    long_anchor_wt2: Optional[float] = None
    long_anchor_bar: Optional[int] = None
    short_anchor_wt2: Optional[float] = None
    short_anchor_bar: Optional[int] = None

    # Trigger wave tracking
    long_trigger_wt2: Optional[float] = None
    short_trigger_wt2: Optional[float] = None

    # Multi-timeframe state (for MTF mode)
    timeframe_states: Dict[str, TimeframeState] = field(default_factory=dict)
    current_bias: str = "neutral"  # Overall HTF bias
    bias_confidence: float = 0.0

    def get_timeframe_state(self, timeframe: str) -> TimeframeState:
        """Get or create timeframe state."""
        if timeframe not in self.timeframe_states:
            self.timeframe_states[timeframe] = TimeframeState(timeframe=timeframe)
        return self.timeframe_states[timeframe]

    def reset_long(self) -> None:
        """Reset long signal state."""
        self.long_state = SignalState.IDLE
        self.long_anchor_wt2 = None
        self.long_anchor_bar = None
        self.long_trigger_wt2 = None

    def reset_short(self) -> None:
        """Reset short signal state."""
        self.short_state = SignalState.IDLE
        self.short_anchor_wt2 = None
        self.short_anchor_bar = None
        self.short_trigger_wt2 = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "symbol": self.symbol,
            "long_state": self.long_state.name,
            "short_state": self.short_state.name,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "long_anchor_wt2": self.long_anchor_wt2,
            "short_anchor_wt2": self.short_anchor_wt2,
            "current_bias": self.current_bias,
            "bias_confidence": self.bias_confidence,
        }
        if self.timeframe_states:
            result["timeframe_states"] = {
                k: v.to_dict() for k, v in self.timeframe_states.items()
            }
        return result


@dataclass
class TradingState:
    """Complete trading state."""
    bot_state: BotState = BotState.STOPPED
    assets: Dict[str, AssetState] = field(default_factory=dict)
    last_error: Optional[str] = None
    started_at: Optional[datetime] = None
    total_signals: int = 0
    total_trades: int = 0

    def get_asset_state(self, symbol: str) -> AssetState:
        """Get or create asset state."""
        if symbol not in self.assets:
            self.assets[symbol] = AssetState(symbol=symbol)
        return self.assets[symbol]

    def record_signal(self, signal: Signal) -> None:
        """Record a generated signal."""
        self.total_signals += 1
        symbol = signal.metadata.get('symbol', 'UNKNOWN')
        asset_state = self.get_asset_state(symbol)
        asset_state.last_signal = signal
        asset_state.last_update = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bot_state": self.bot_state.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "total_signals": self.total_signals,
            "total_trades": self.total_trades,
            "last_error": self.last_error,
            "assets": {k: v.to_dict() for k, v in self.assets.items()},
        }

    def save(self, filepath: str) -> None:
        """Save state to file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "TradingState":
        """Load state from file."""
        path = Path(filepath)
        if not path.exists():
            return cls()

        with open(path, 'r') as f:
            data = json.load(f)

        state = cls()
        state.bot_state = BotState(data.get('bot_state', 'stopped'))
        state.total_signals = data.get('total_signals', 0)
        state.total_trades = data.get('total_trades', 0)
        state.last_error = data.get('last_error')

        if data.get('started_at'):
            state.started_at = datetime.fromisoformat(data['started_at'])

        return state
