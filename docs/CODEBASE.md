# VMC Trading Bot - Codebase Documentation

## Project Structure Overview

```
vmc_trading_bot/
├── config/                     # Configuration files
│   ├── __init__.py            # Package exports
│   ├── config.py              # Pydantic models for config validation
│   └── config.example.yaml    # Template config for users
│
├── src/                       # Main source code
│   ├── __init__.py
│   ├── main.py               # CLI entry point (click commands)
│   │
│   ├── core/                 # Bot core functionality
│   │   ├── __init__.py
│   │   ├── bot.py           # VMCBot - main orchestrator
│   │   └── state.py         # TradingState, AssetState management
│   │
│   ├── indicators/          # Technical indicator calculations
│   │   ├── __init__.py      # Exports: HeikinAshi, WaveTrend, MoneyFlow
│   │   ├── heikin_ashi.py   # HA candle conversion
│   │   ├── wavetrend.py     # WaveTrend oscillator (WT1, WT2, VWAP)
│   │   └── money_flow.py    # RSI+MFI indicator
│   │
│   ├── exchanges/           # Exchange API integrations
│   │   ├── __init__.py
│   │   ├── base.py         # Abstract BaseExchange class
│   │   ├── hyperliquid.py  # Hyperliquid implementation
│   │   └── bitunix.py      # Bitunix implementation
│   │
│   ├── strategy/           # Trading strategy logic
│   │   ├── __init__.py
│   │   ├── signals.py      # SignalDetector state machine
│   │   ├── risk.py         # Position sizing, SL/TP calculation
│   │   ├── trade_manager.py # Trade execution & management
│   │   └── multi_timeframe.py # MTF coordination (NEW)
│   │
│   ├── webhook/            # TradingView webhook server
│   │   ├── __init__.py
│   │   └── server.py       # FastAPI webhook endpoint
│   │
│   └── utils/              # Utility functions
│       ├── __init__.py
│       ├── logger.py       # Loguru logging setup
│       └── helpers.py      # Common utilities
│
├── backtest/               # Backtesting system
│   ├── __init__.py
│   ├── engine.py          # BacktestEngine - simulation
│   ├── data_loader.py     # Historical data fetching
│   └── metrics.py         # Performance metrics calculation
│
├── notifications/         # Notification systems
│   ├── __init__.py
│   └── discord.py        # Discord webhook notifications
│
├── tests/                 # Test suite
│   ├── __init__.py
│   ├── test_indicators.py # Indicator unit tests
│   ├── test_signals.py    # Signal detection tests
│   ├── test_backtest.py   # Backtest tests
│   ├── test_discord.py    # Discord notification tests
│   └── integration/       # Live API tests
│
├── docs/                  # Documentation
│   ├── USER_GUIDE.md     # User setup guide
│   ├── CONFIGURATION.md  # Config reference
│   ├── STRATEGY.md       # Strategy explanation
│   ├── CODEBASE.md       # This file
│   └── MULTI_TIMEFRAME.md # MTF feature docs
│
├── requirements.txt      # Python dependencies
├── run.py               # CLI runner script
└── logs/                # Log output directory
```

---

## Key Components

### 1. Entry Point (`run.py` → `src/main.py`)

The bot is launched via `run.py` which delegates to `src/main.py`:

```bash
python run.py run --config config/config.yaml      # Start bot
python run.py backtest --start 2025-01-01          # Run backtest
python run.py init                                  # Create default config
```

**CLI Commands:**
- `run`: Start the trading bot with live exchange connection
- `backtest`: Run historical simulation
- `init`: Generate default configuration file

### 2. Configuration (`config/config.py`)

Uses Pydantic models for type-safe configuration validation:

```python
# Key configuration classes
Config              # Root config object
├── ExchangeConfig  # Exchange credentials (Hyperliquid/Bitunix)
├── TradingConfig   # Assets, timeframes, risk settings
│   └── TimeframesConfig  # MTF entry/bias timeframes (NEW)
├── IndicatorConfig # WaveTrend and MFI parameters
├── StopLossConfig  # SL method and settings
├── TakeProfitConfig # TP method and settings
├── WebhookConfig   # TradingView webhook settings
├── DiscordConfig   # Discord notifications
└── BacktestConfig  # Backtesting parameters
```

### 3. Indicators (`src/indicators/`)

Technical indicator calculations ported from TradingView Pine Script:

**HeikinAshi** (`heikin_ashi.py`):
- Converts standard OHLC candles to smoothed Heikin Ashi candles
- Reduces noise for trend identification

**WaveTrend** (`wavetrend.py`):
- Calculates WT1, WT2, and VWAP (WT1 - WT2)
- Detects overbought/oversold conditions
- Identifies crosses for signal generation

```python
# WaveTrend Calculation
ESA = EMA(HLC3, channel_len)    # 9-period EMA of HLC3
D = EMA(|HLC3 - ESA|, channel_len)
CI = (HLC3 - ESA) / (0.015 * D)
WT1 = EMA(CI, average_len)       # 12-period EMA
WT2 = SMA(WT1, ma_len)           # 3-period SMA
VWAP = WT1 - WT2                 # Momentum
```

**MoneyFlow** (`money_flow.py`):
- RSI+MFI area indicator
- Shows buying/selling pressure
- Used for confirmation of entries

### 4. Signal Detection (`src/strategy/signals.py`)

State machine for detecting VMC trading signals:

```
State Machine Flow:
IDLE → ANCHOR_DETECTED → TRIGGER_DETECTED → AWAITING_MFI → AWAITING_VWAP → SIGNAL_READY

Long Entry:
1. Anchor: WT2 < -60 (extreme oversold)
2. Trigger: Subsequent smaller dip (higher low)
3. MFI: Was negative, now curving up
4. VWAP: Crosses above 0
5. Execute long

Short Entry:
1. Anchor: WT2 > 60 (extreme overbought)
2. Trigger: Subsequent smaller peak + WT cross down
3. MFI: Was positive, now curving down
4. VWAP: Crosses below 0
5. Execute short
```

### 5. Multi-Timeframe Coordination (`src/strategy/multi_timeframe.py`)

NEW: Coordinates signal detection across multiple timeframes:

```python
# Entry timeframes (scalping): 3m, 5m, 10m, 15m
# Bias timeframes (HTF): 4h, 8h, 12h, 1D

# Bias Hierarchy: 1D > 12h > 8h > 4h
# Bias determined by WT2 position:
#   WT2 < 0 = Bullish zone
#   WT2 > 0 = Bearish zone

# Only execute entry signals that align with HTF bias
```

See `docs/MULTI_TIMEFRAME.md` for detailed documentation.

### 6. Exchange Clients (`src/exchanges/`)

Abstract `BaseExchange` interface with concrete implementations:

**BaseExchange** (`base.py`):
- Abstract methods for all exchange operations
- Data classes: `Candle`, `Order`, `Position`, `AccountBalance`
- Multi-timeframe candle fetching with custom TF aggregation

**HyperliquidExchange** (`hyperliquid.py`):
- Uses official `hyperliquid-python-sdk`
- Supports testnet and mainnet
- Perpetual futures trading

**BitunixExchange** (`bitunix.py`):
- REST API with HMAC-SHA256 signing
- Futures trading support

### 7. Bot Orchestrator (`src/core/bot.py`)

`VMCBot` class coordinates all components:

```python
class VMCBot:
    # Main loop flow:
    async def run_loop():
        while running:
            for asset in assets:
                signal = await _check_asset(asset)
                if signal:
                    await _handle_signal(signal)
            await sleep(interval)

    # Single TF mode: _check_asset_single_tf()
    # Multi TF mode:  _check_asset_mtf()
```

### 8. Risk Management (`src/strategy/risk.py`)

Position sizing and stop loss/take profit calculations:

**Stop Loss Methods:**
- `swing`: Based on recent swing high/low
- `atr`: ATR-based dynamic stop
- `fixed_percent`: Fixed percentage from entry

**Take Profit Methods:**
- `fixed_rr`: Fixed risk/reward ratio
- `oscillator`: Exit on opposite signal
- `partial`: Partial TP with breakeven SL
- `trailing`: Trailing stop

### 9. Trade Execution (`src/strategy/trade_manager.py`)

Handles order placement and position management:

- Validates signal before execution
- Calculates position size based on risk
- Places orders with SL/TP
- Tracks active positions
- Calculates P&L

---

## Data Flow

```
[Exchange API]
      ↓
   Candles (OHLCV)
      ↓
[HeikinAshi Converter]
      ↓
   HA Candles
      ↓
[WaveTrend] + [MoneyFlow]
      ↓
   Indicator Values (WT1, WT2, VWAP, MFI)
      ↓
[SignalDetector State Machine]
   or
[MultiTimeframeCoordinator] (MTF mode)
      ↓
   Signal (LONG/SHORT) or None
      ↓
[RiskManager]
      ↓
   Position Size, SL, TP
      ↓
[TradeManager]
      ↓
   Order Execution
      ↓
[DiscordNotifier]
      ↓
   Notifications
```

---

## Key Algorithms

### WaveTrend Calculation

```python
def calculate(df):
    hlc3 = (df['high'] + df['low'] + df['close']) / 3
    esa = ema(hlc3, channel_len)  # 9
    d = ema(abs(hlc3 - esa), channel_len)
    ci = (hlc3 - esa) / (0.015 * d)
    wt1 = ema(ci, average_len)  # 12
    wt2 = sma(wt1, ma_len)  # 3
    vwap = wt1 - wt2
    return WaveTrendResult(wt1, wt2, vwap, cross_up, cross_down)
```

### Bias Determination (MTF)

```python
def calculate_bias(htf_data):
    weights = {"1d": 4, "12h": 3, "8h": 2, "4h": 1}
    bullish_score = 0
    bearish_score = 0

    for tf, indicators in htf_data.items():
        weight = weights.get(tf, 1)
        if indicators.wt2 < 0:
            bullish_score += weight
        else:
            bearish_score += weight

    if bullish_score > bearish_score:
        return BiasDirection.BULLISH
    elif bearish_score > bullish_score:
        return BiasDirection.BEARISH
    else:
        return BiasDirection.NEUTRAL
```

---

## Testing

### Test Coverage

- **26 unit tests** for indicators and signals
- **5 backtest tests** for simulation engine
- **5 Discord tests** for notifications

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_indicators.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Dependencies

Key Python packages:

```
hyperliquid-python-sdk>=0.9.0  # Hyperliquid API
pandas>=2.0.0                   # Data manipulation
numpy>=1.24.0                   # Numerical computing
pydantic>=2.0.0                 # Config validation
pydantic-settings>=2.0.0        # Settings management
fastapi>=0.100.0                # Webhook server
uvicorn>=0.23.0                 # ASGI server
aiohttp>=3.8.0                  # Async HTTP
pyyaml>=6.0                     # YAML parsing
loguru>=0.7.0                   # Logging
click>=8.0.0                    # CLI framework
pytest>=7.4.0                   # Testing
```

---

## Configuration Reference

See `docs/CONFIGURATION.md` for full configuration reference.

See `config/config.example.yaml` for annotated example.
