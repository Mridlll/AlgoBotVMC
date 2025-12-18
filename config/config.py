"""Configuration models and loader for VMC Trading Bot."""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

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


class SignalMode(str, Enum):
    """Signal detection modes for VMC strategy.

    Based on analysis of original VMC indicator vs our implementation:
    - SIMPLE: Original VMC (WT cross while oversold/overbought)
    - ENHANCED: Our 4-step state machine (anchor -> trigger -> MFI -> VWAP)
    - MTF: Multi-timeframe (HTF bias + LTF entry) based on Discord feedback
    - MTF_V4: V4 Multi-timeframe (4H bias + 1H entry) optimized for maximum returns
    """
    SIMPLE = "simple"        # Original VMC: WT1 crosses WT2 while oversold/overbought
    ENHANCED = "enhanced"    # Our 4-step: anchor -> trigger -> MFI -> VWAP
    MTF = "mtf"             # Multi-timeframe: HTF bias determines direction, LTF for entry
    MTF_V4 = "mtf_v4"       # V4: 4H bias + 1H entry, optimized config


class DirectionFilter(str, Enum):
    """Trade direction filter for V4 strategy.

    Based on 1-year backtest analysis:
    - 57% of all trades are longs
    - 69% of 4H trades are longs
    - Bull market bias favors long-only mode
    """
    BOTH = "both"            # Take all signals (default)
    LONG_ONLY = "long_only"  # Only take long signals (bull market mode)
    SHORT_ONLY = "short_only"  # Only take short signals (bear market mode)


class AssetConfig(BaseModel):
    """Per-asset configuration overrides.

    Allows different settings per asset. Any setting not specified
    falls back to the global defaults from TradingConfig/StrategyConfig.

    Based on 1-year backtest results:
    - BTC: 4H ENHANCED + FIXED_RR (+$5,104, 11% DD)
    - ETH: 30m ENHANCED + FULL_SIGNAL (+$5,410, 47% DD)
    - SOL: 4H ENHANCED + FIXED_RR (+$2,617, 21% DD)
    """
    enabled: bool = Field(
        default=True,
        description="Whether to trade this asset"
    )
    timeframe: Optional[str] = Field(
        default=None,
        description="Override timeframe for this asset (e.g., '4h', '30m')"
    )
    signal_mode: Optional[SignalMode] = Field(
        default=None,
        description="Override signal mode for this asset"
    )
    exit_strategy: Optional[TakeProfitMethod] = Field(
        default=None,
        description="Override exit strategy for this asset"
    )
    oscillator_mode: Optional[OscillatorExitMode] = Field(
        default=None,
        description="Override oscillator exit mode (when exit_strategy=oscillator)"
    )
    risk_reward: Optional[float] = Field(
        default=None,
        ge=0.5,
        le=5.0,
        description="Override R:R ratio for this asset"
    )
    anchor_level: Optional[float] = Field(
        default=None,
        ge=40,
        le=90,
        description="Override anchor level for ENHANCED mode"
    )
    risk_percent: Optional[float] = Field(
        default=None,
        ge=0.1,
        le=10.0,
        description="Override risk percent for this asset"
    )
    max_positions: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
        description="Override max positions for this asset"
    )

    @field_validator("timeframe")
    @classmethod
    def validate_asset_timeframe(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        valid_timeframes = ["1m", "3m", "5m", "10m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "1w"]
        if v.lower() not in valid_timeframes:
            raise ValueError(f"Timeframe must be one of: {valid_timeframes}")
        return v.lower()


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


class TimeFilterMode(str, Enum):
    """Time filter action modes."""
    SKIP = "skip"           # Skip trading entirely during filtered hours
    REDUCE = "reduce"       # Reduce position size during filtered hours


class TimeFilterConfig(BaseModel):
    """Time-based trading filter configuration.

    Based on backtest data showing Off Hours significantly outperform Market Hours:
    - Off Hours: +$55,431 total PnL
    - Market Hours: -$11,036 total PnL
    - Weekends: +$18,908 total PnL
    """
    enabled: bool = Field(
        default=False,
        description="Enable time-based trading filter"
    )
    mode: TimeFilterMode = Field(
        default=TimeFilterMode.SKIP,
        description="skip=no trades during filtered hours, reduce=50% position size"
    )
    # US Market hours in UTC (9:30 AM - 4:00 PM EST = 14:30 - 21:00 UTC)
    avoid_us_market_hours: bool = Field(
        default=True,
        description="Avoid trading during US market hours"
    )
    market_hours_start_utc: int = Field(
        default=14,
        ge=0,
        le=23,
        description="US market hours start (UTC). 14 = 9:30 AM EST"
    )
    market_hours_end_utc: int = Field(
        default=21,
        ge=0,
        le=23,
        description="US market hours end (UTC). 21 = 4:00 PM EST"
    )
    trade_weekends: bool = Field(
        default=True,
        description="Allow trading on weekends (Saturday/Sunday)"
    )
    reduced_position_multiplier: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="Position size multiplier when mode=reduce (0.5 = 50%)"
    )


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
    # Time-based trading filter (v3)
    time_filter: TimeFilterConfig = Field(default_factory=TimeFilterConfig)
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


class StrategyConfig(BaseModel):
    """Signal detection strategy configuration.

    Three modes available based on V2 rework analysis:
    - SIMPLE: Original VMC indicator behavior (WT cross + zone)
    - ENHANCED: Our 4-step state machine (anchor-trigger-MFI-VWAP)
    - MTF: Multi-timeframe with HTF bias + LTF entry (Discord feedback)
    """
    # Signal mode selection
    signal_mode: SignalMode = Field(
        default=SignalMode.ENHANCED,
        description="Signal detection mode: simple, enhanced, or mtf"
    )

    # === SIMPLE MODE SETTINGS ===
    # Original VMC uses -53/+53 for oversold/overbought
    simple_oversold: float = Field(
        default=-53,
        le=0,
        description="SIMPLE mode: WT2 level for oversold (original VMC uses -53)"
    )
    simple_overbought: float = Field(
        default=53,
        ge=0,
        description="SIMPLE mode: WT2 level for overbought (original VMC uses +53)"
    )

    # === ENHANCED MODE SETTINGS ===
    # Our 4-step state machine settings
    anchor_level: float = Field(
        default=60,
        ge=40,
        le=90,
        description="ENHANCED mode: WT2 level for anchor wave detection"
    )
    trigger_timeout_bars: int = Field(
        default=20,
        ge=5,
        le=50,
        description="ENHANCED mode: Max bars to wait for trigger after anchor"
    )
    require_mfi_confirm: bool = Field(
        default=True,
        description="ENHANCED mode: Require MFI curve confirmation"
    )
    require_vwap_confirm: bool = Field(
        default=True,
        description="ENHANCED mode: Require VWAP zero cross confirmation"
    )

    # === MTF MODE SETTINGS ===
    # Higher timeframes for bias (4H and higher)
    htf_timeframes: List[str] = Field(
        default=["4h", "12h", "1d"],
        description="MTF mode: Higher timeframes for bias determination"
    )
    # Lower timeframes for entry signals
    ltf_timeframes: List[str] = Field(
        default=["15m", "30m"],
        description="MTF mode: Lower timeframes for entry signals"
    )
    # HTF bias minimum confidence (0-1)
    min_bias_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="MTF mode: Minimum HTF bias confidence to take trades"
    )
    # Allow trades against bias if confidence is low
    allow_neutral_bias_trades: bool = Field(
        default=False,
        description="MTF mode: Allow trades when HTF bias is neutral"
    )

    # === DCA (Dollar Cost Averaging) SETTINGS ===
    enable_dca: bool = Field(
        default=False,
        description="MTF mode: Enable DCA (second entry on VWAP curve)"
    )
    dca_on_vwap_curve: bool = Field(
        default=True,
        description="MTF mode: Trigger DCA when VWAP curves in trade direction"
    )
    dca_size_multiplier: float = Field(
        default=0.5,
        ge=0.1,
        le=2.0,
        description="MTF mode: DCA position size as multiplier of original"
    )
    max_dca_entries: int = Field(
        default=1,
        ge=1,
        le=3,
        description="MTF mode: Maximum number of DCA entries per trade"
    )

    # === V4 MTF SETTINGS (4H bias + 1H entry) ===
    v4_htf_timeframe: str = Field(
        default="4h",
        description="V4: Higher timeframe for bias determination"
    )
    v4_ltf_timeframe: str = Field(
        default="1h",
        description="V4: Lower timeframe for entry signals"
    )
    v4_direction_filter: DirectionFilter = Field(
        default=DirectionFilter.BOTH,
        description="V4: Direction filter (both, long_only, short_only)"
    )
    v4_bias_wt2_threshold: float = Field(
        default=30.0,
        ge=20.0,
        le=60.0,
        description="V4: WT2 threshold for HTF bias detection (< -threshold = bullish, > +threshold = bearish)"
    )
    v4_require_mfi_confirm: bool = Field(
        default=True,
        description="V4: Require MFI curving confirmation for HTF bias"
    )
    v4_allow_neutral_trades: bool = Field(
        default=False,
        description="V4: Allow trades when HTF bias is neutral"
    )

    @field_validator("htf_timeframes", "ltf_timeframes")
    @classmethod
    def validate_strategy_timeframes(cls, v: List[str]) -> List[str]:
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
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    indicators: IndicatorConfig = Field(default_factory=IndicatorConfig)
    stop_loss: StopLossConfig = Field(default_factory=StopLossConfig)
    take_profit: TakeProfitConfig = Field(default_factory=TakeProfitConfig)
    webhook: WebhookConfig = Field(default_factory=WebhookConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    # Per-asset configuration overrides
    assets_config: Dict[str, AssetConfig] = Field(
        default_factory=dict,
        description="Per-asset configuration overrides (e.g., {'BTC': {...}, 'ETH': {...}})"
    )

    def get_asset_config(self, asset: str) -> AssetConfig:
        """Get configuration for a specific asset, with defaults filled in."""
        return self.assets_config.get(asset.upper(), AssetConfig())

    def get_asset_timeframe(self, asset: str) -> str:
        """Get effective timeframe for an asset (per-asset override or global default)."""
        asset_cfg = self.get_asset_config(asset)
        return asset_cfg.timeframe or self.trading.timeframe

    def get_asset_signal_mode(self, asset: str) -> SignalMode:
        """Get effective signal mode for an asset."""
        asset_cfg = self.get_asset_config(asset)
        return asset_cfg.signal_mode or self.strategy.signal_mode

    def get_asset_exit_strategy(self, asset: str) -> TakeProfitMethod:
        """Get effective exit strategy for an asset."""
        asset_cfg = self.get_asset_config(asset)
        return asset_cfg.exit_strategy or self.take_profit.method

    def get_asset_oscillator_mode(self, asset: str) -> OscillatorExitMode:
        """Get effective oscillator exit mode for an asset."""
        asset_cfg = self.get_asset_config(asset)
        return asset_cfg.oscillator_mode or self.take_profit.oscillator_mode

    def get_asset_risk_reward(self, asset: str) -> float:
        """Get effective R:R ratio for an asset."""
        asset_cfg = self.get_asset_config(asset)
        return asset_cfg.risk_reward or self.take_profit.risk_reward

    def get_asset_anchor_level(self, asset: str) -> float:
        """Get effective anchor level for an asset."""
        asset_cfg = self.get_asset_config(asset)
        return asset_cfg.anchor_level or self.strategy.anchor_level

    def get_asset_risk_percent(self, asset: str) -> float:
        """Get effective risk percent for an asset."""
        asset_cfg = self.get_asset_config(asset)
        return asset_cfg.risk_percent or self.trading.risk_percent

    def get_enabled_assets(self) -> List[str]:
        """Get list of enabled assets to trade."""
        enabled = []
        for asset in self.trading.assets:
            asset_cfg = self.get_asset_config(asset)
            if asset_cfg.enabled:
                enabled.append(asset)
        return enabled

    def is_asset_enabled(self, asset: str) -> bool:
        """Check if an asset is enabled for trading."""
        if asset.upper() not in [a.upper() for a in self.trading.assets]:
            return False
        return self.get_asset_config(asset).enabled


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
