"""Configuration models and loader for VMC Trading Bot."""

from enum import Enum
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class ExchangeName(str, Enum):
    """Supported exchanges."""
    HYPERLIQUID = "hyperliquid"
    BITUNIX = "bitunix"


class StopLossMethod(str, Enum):
    """Stop loss calculation methods."""
    SWING = "swing"
    ATR = "atr"
    FIXED_PERCENT = "fixed_percent"


class TakeProfitMethod(str, Enum):
    """Take profit methods."""
    FIXED_RR = "fixed_rr"
    OSCILLATOR = "oscillator"
    PARTIAL = "partial"
    TRAILING = "trailing"


class OscillatorExitMode(str, Enum):
    """Oscillator exit sub-modes (used when TakeProfitMethod is OSCILLATOR)."""
    FULL_SIGNAL = "full_signal"      # Wait for complete opposite VMC signal
    WT_CROSS = "wt_cross"            # Exit when WT1 crosses WT2 opposite direction
    FIRST_REVERSAL = "first_reversal" # Exit on any reversal sign (MFI, VWAP, or WT)


class ExchangeConfig(BaseModel):
    """Exchange connection configuration.

    For Hyperliquid, there are two wallet modes:

    1. Direct Trading (simpler, less secure):
       - api_secret: Private key of your main wallet
       - wallet_address: Your main wallet address
       - account_address: Leave empty/None

    2. API Wallet Trading (recommended, more secure):
       - Generate API wallet at:
         * Mainnet: https://app.hyperliquid.xyz/API
         * Testnet: https://app.hyperliquid-testnet.xyz/API
       - api_secret: Private key of your API wallet
       - wallet_address: Your API wallet address
       - account_address: Your main wallet address (where funds are)

    Note: API wallets are network-specific! Create separate ones for testnet/mainnet.
    """
    name: ExchangeName = Field(default=ExchangeName.HYPERLIQUID)
    api_key: str = Field(default="")
    api_secret: str = Field(default="", description="Private key (API wallet or main wallet)")
    wallet_address: Optional[str] = Field(
        default=None,
        description="Wallet address for signing (API wallet or main wallet)"
    )
    account_address: Optional[str] = Field(
        default=None,
        description="Main wallet address with funds (only if using API wallet)"
    )
    testnet: bool = Field(default=True, description="Use testnet for testing")

    @field_validator("api_key", "api_secret")
    @classmethod
    def check_not_placeholder(cls, v: str) -> str:
        if v in ["your_api_key_here", "your_api_secret_here", "..."]:
            raise ValueError("Please set your actual API credentials")
        return v


class TimeframesConfig(BaseModel):
    """Multi-timeframe configuration for scalping with HTF bias."""
    # Entry timeframes - scan these for entry signals (scalping)
    entry: List[str] = Field(
        default=["5m", "15m"],
        description="Timeframes for entry signal detection"
    )
    # Bias timeframes - determine trend direction (HTF)
    bias: List[str] = Field(
        default=["4h", "12h", "1d"],
        description="Higher timeframes for trend bias"
    )
    # Alignment settings
    require_bias_alignment: bool = Field(
        default=True,
        description="Only enter when entry signal matches HTF bias"
    )
    min_bias_timeframes_aligned: int = Field(
        default=2,
        ge=1,
        description="Minimum number of bias TFs that must agree"
    )
    entry_on_any_timeframe: bool = Field(
        default=True,
        description="True=signal on ANY entry TF triggers, False=ALL must agree"
    )
    # Bias hierarchy weights (higher = more important)
    bias_weights: dict = Field(
        default={"1d": 4, "12h": 3, "8h": 2, "4h": 1},
        description="Weights for bias calculation (1D > 12h > 8h > 4h)"
    )

    @field_validator("entry", "bias")
    @classmethod
    def validate_timeframes_list(cls, v: List[str]) -> List[str]:
        # Extended timeframes including custom ones that will be aggregated
        valid_timeframes = [
            "1m", "3m", "5m", "10m", "15m", "30m",
            "1h", "2h", "4h", "6h", "8h", "12h",
            "1d", "1w"
        ]
        result = []
        for tf in v:
            tf_lower = tf.lower()
            if tf_lower not in valid_timeframes:
                raise ValueError(f"Invalid timeframe '{tf}'. Must be one of: {valid_timeframes}")
            result.append(tf_lower)
        return result


class TradingConfig(BaseModel):
    """Trading parameters configuration."""
    assets: List[str] = Field(default=["BTC", "ETH"])
    # Legacy single timeframe (for backward compatibility)
    timeframe: str = Field(default="4h", description="Legacy single timeframe (use timeframes for MTF)")
    # Multi-timeframe configuration
    timeframes: TimeframesConfig = Field(default_factory=TimeframesConfig)
    use_multi_timeframe: bool = Field(default=False, description="Enable multi-timeframe mode")
    risk_percent: float = Field(default=3.0, ge=0.1, le=100)
    max_positions: int = Field(default=2, ge=1, le=20)
    max_positions_per_asset: int = Field(default=1, ge=1, le=5)
    leverage: float = Field(default=1.0, ge=1, le=100)

    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, v: str) -> str:
        valid_timeframes = ["1m", "3m", "5m", "10m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "1w"]
        if v.lower() not in valid_timeframes:
            raise ValueError(f"Timeframe must be one of: {valid_timeframes}")
        return v.lower()


class IndicatorConfig(BaseModel):
    """VMC indicator parameters."""
    # WaveTrend settings
    wt_channel_len: int = Field(default=9, ge=1)
    wt_average_len: int = Field(default=12, ge=1)
    wt_ma_len: int = Field(default=3, ge=1)

    # Overbought/Oversold levels
    wt_overbought_1: int = Field(default=53)
    wt_overbought_2: int = Field(default=60)
    wt_oversold_1: int = Field(default=-53)
    wt_oversold_2: int = Field(default=-60)

    # Money Flow settings
    mfi_period: int = Field(default=60, ge=1)
    mfi_multiplier: float = Field(default=150.0)
    mfi_y_pos: float = Field(default=2.5)


class StopLossConfig(BaseModel):
    """Stop loss configuration."""
    method: StopLossMethod = Field(default=StopLossMethod.SWING)
    swing_lookback: int = Field(default=5, ge=1, le=50)
    buffer_percent: float = Field(default=0.5, ge=0)
    atr_multiplier: float = Field(default=1.5, ge=0.1)
    atr_period: int = Field(default=14, ge=1)
    fixed_percent: float = Field(default=2.0, ge=0.1)


class TakeProfitConfig(BaseModel):
    """Take profit configuration."""
    method: TakeProfitMethod = Field(default=TakeProfitMethod.FIXED_RR)
    risk_reward: float = Field(default=2.0, ge=0.5)

    # Oscillator exit mode (used when method=oscillator)
    oscillator_mode: OscillatorExitMode = Field(
        default=OscillatorExitMode.WT_CROSS,
        description="Exit mode when method=oscillator: full_signal, wt_cross, or first_reversal"
    )

    # Partial TP settings
    partial_tp_percent: float = Field(default=50.0, ge=1, le=99)
    partial_tp_rr: float = Field(default=1.0, ge=0.1)
    move_sl_to_breakeven: bool = Field(default=True)

    # Trailing stop settings
    trailing_activation_rr: float = Field(default=1.5, ge=0.1)
    trailing_distance_percent: float = Field(default=1.0, ge=0.1)


class WebhookConfig(BaseModel):
    """TradingView webhook configuration."""
    enabled: bool = Field(default=False)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080, ge=1, le=65535)
    secret: str = Field(default="")
    path: str = Field(default="/webhook")


class DiscordConfig(BaseModel):
    """Discord notification configuration."""
    enabled: bool = Field(default=False)
    webhook_url: str = Field(default="")
    notify_on_signal: bool = Field(default=True)
    notify_on_trade_open: bool = Field(default=True)
    notify_on_trade_close: bool = Field(default=True)
    notify_on_error: bool = Field(default=True)


class BacktestConfig(BaseModel):
    """Backtesting configuration."""
    start_date: Optional[str] = Field(default=None, description="Format: YYYY-MM-DD")
    end_date: Optional[str] = Field(default=None, description="Format: YYYY-MM-DD")
    initial_balance: float = Field(default=10000.0, ge=100)
    commission_percent: float = Field(default=0.06, ge=0)


class Config(BaseModel):
    """Main configuration model."""
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    indicators: IndicatorConfig = Field(default_factory=IndicatorConfig)
    stop_loss: StopLossConfig = Field(default_factory=StopLossConfig)
    take_profit: TakeProfitConfig = Field(default_factory=TakeProfitConfig)
    webhook: WebhookConfig = Field(default_factory=WebhookConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)


def load_config(config_path: str | Path) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration YAML file

    Returns:
        Config object with validated settings

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config values are invalid
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r") as f:
        raw_config = yaml.safe_load(f)

    return Config(**raw_config)


def save_config(config: Config, config_path: str | Path) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Config object to save
        config_path: Path to save configuration file
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)


def create_default_config(config_path: str | Path) -> Config:
    """
    Create a default configuration file.

    Args:
        config_path: Path to save the default config

    Returns:
        Default Config object
    """
    config = Config()
    save_config(config, config_path)
    return config
