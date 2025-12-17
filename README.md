# VMC Trading Bot

Automated cryptocurrency trading bot using the **VuManChu Cipher B** (VMC) indicator strategy. Supports live trading on Hyperliquid and comprehensive backtesting with multiple exit strategies.

> **New to the bot?** See [SETUP.md](SETUP.md) for complete step-by-step instructions.

## Features

- **VMC Signal Detection**: Anchor wave → Trigger wave → MFI confirmation → VWAP cross
- **Multiple Exit Strategies**: Fixed R:R, Full Signal, WT Cross, 1st Reversal
- **Hyperliquid Integration**: Live trading on Hyperliquid perpetual futures
- **Comprehensive Backtesting**: Multi-asset, multi-timeframe analysis
- **Risk Management**: Configurable position sizing and stop loss methods
- **Discord Notifications**: Real-time trade alerts

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Mridlll/AlgoBotVMC.git
cd AlgoBotVMC

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy example config
cp config/config.example.yaml config/config.yaml

# Edit with your settings
notepad config/config.yaml  # Windows
nano config/config.yaml     # Linux/Mac
```

### 3. Run Backtest

```bash
# Run comprehensive backtest
python fetch_and_backtest.py
```

### 4. Run Live Bot

```bash
# Start the trading bot
python run.py
```

---

## Configuration Guide

### Exchange Setup (Hyperliquid)

```yaml
exchange:
  name: "hyperliquid"
  api_key: ""                    # Leave empty for Hyperliquid
  api_secret: "your_private_key" # Your wallet private key
  wallet_address: "0x..."        # Your wallet address
  testnet: true                  # Set false for mainnet
```

**Getting Hyperliquid Credentials:**
1. Go to [Hyperliquid](https://app.hyperliquid.xyz/)
2. Connect your wallet
3. Export your private key (keep it secure!)
4. Copy your wallet address

### Trading Parameters

```yaml
trading:
  assets:
    - "BTC"
    - "ETH"
    - "SOL"
  timeframe: "30m"      # Recommended: 30m (best performance)
  risk_percent: 3.0     # Risk 3% of account per trade
  max_positions: 3      # Maximum open positions
  leverage: 1.0         # Leverage multiplier
```

### Exit Strategies

```yaml
take_profit:
  method: "oscillator"           # Options: fixed_rr, oscillator
  risk_reward: 2.0               # For fixed_rr method
  oscillator_mode: "wt_cross"    # For oscillator method
```

| Method | oscillator_mode | Description |
|--------|-----------------|-------------|
| `fixed_rr` | - | Exit at fixed 2:1 risk-reward ratio |
| `oscillator` | `full_signal` | Exit on complete opposite VMC signal |
| `oscillator` | `wt_cross` | Exit when WT1 crosses WT2 opposite direction |
| `oscillator` | `first_reversal` | Exit on first reversal sign (fastest) |

### Stop Loss Configuration

```yaml
stop_loss:
  method: "swing"         # Options: swing, atr, fixed_percent
  swing_lookback: 5       # Candles to look back
  buffer_percent: 0.5     # Buffer below/above swing level
  atr_multiplier: 1.5     # For ATR method
  fixed_percent: 2.0      # For fixed percent method
```

### VMC Indicator Settings

```yaml
indicators:
  wt_channel_len: 9
  wt_average_len: 12
  wt_ma_len: 3
  wt_overbought_2: 60     # Anchor level for shorts
  wt_oversold_2: -60      # Anchor level for longs
  mfi_period: 60
  mfi_multiplier: 150.0
```

---

## Strategy Explanation

### VMC Signal Detection (4-Step Process)

1. **Anchor Wave**: WT2 crosses into extreme zone (< -60 for longs, > +60 for shorts)
2. **Trigger Wave**: WT2 crosses back from extreme zone
3. **MFI Confirmation**: Money Flow confirms direction (green for long, red for short)
4. **VWAP Cross**: VWAP crosses zero line in trade direction

### Exit Strategies Explained

| Strategy | When It Exits | Best For |
|----------|---------------|----------|
| **Fixed R:R** | At 2x risk distance | Ranging markets |
| **Full Signal** | When all 4 conditions flip opposite | Strong trends |
| **WT Cross** | When WT1 crosses WT2 opposite | Most conditions |
| **1st Reversal** | First warning sign (MFI/VWAP/WT) | High volatility |

---

## Backtesting

### Quick Backtest

```bash
# Fetch data from Binance and run all backtests
python fetch_and_backtest.py
```

This will:
- Download 3 months of data for BTC, ETH, SOL
- Test all 6 timeframes (3m, 5m, 15m, 30m, 1h, 4h)
- Compare all 4 exit strategies
- Generate report at `output/full_comparison_binance.txt`

### Custom Backtest

```python
from backtest.engine import BacktestEngine
from config import TakeProfitMethod, OscillatorExitMode

engine = BacktestEngine(
    initial_balance=10000,
    risk_percent=3.0,
    risk_reward=2.0,
    tp_method=TakeProfitMethod.OSCILLATOR,
    oscillator_mode=OscillatorExitMode.WT_CROSS
)

result = engine.run(df)  # df = your OHLCV DataFrame
print(f"Win Rate: {result.win_rate}%")
print(f"Total PnL: ${result.total_pnl}")
```

---

## Recommended Settings

Based on backtesting 3 months of data (72 combinations tested):

### Best Performing Setups

| Rank | Asset | Timeframe | Strategy | Return | Max DD |
|:----:|:-----:|:---------:|:---------|-------:|-------:|
| 1 | ETH | 30m | Full Signal | +52.0% | 28.7% |
| 2 | SOL | 30m | WT Cross | +45.3% | 12.8% |
| 3 | BTC | 30m | Full Signal | +23.1% | 28.8% |

### Optimal Configuration

```yaml
trading:
  assets: ["ETH", "SOL", "BTC"]
  timeframe: "30m"
  risk_percent: 3.0

take_profit:
  method: "oscillator"
  oscillator_mode: "wt_cross"  # Most consistent
```

### What to Avoid

- **3m and 5m timeframes**: 80-90% drawdowns
- **Fixed R:R on lower timeframes**: Underperforms oscillator exits
- **4h timeframe**: Too few signals

---

## Project Structure

```
AlgoBotVMC/
├── config/
│   ├── config.py              # Configuration models
│   ├── config.yaml            # Your settings (create from example)
│   └── config.example.yaml    # Example configuration
├── src/
│   ├── core/
│   │   ├── bot.py             # Main bot logic
│   │   └── state.py           # State management
│   ├── exchanges/
│   │   ├── base.py            # Exchange interface
│   │   └── hyperliquid.py     # Hyperliquid implementation
│   ├── indicators/
│   │   ├── wavetrend.py       # WaveTrend indicator
│   │   ├── money_flow.py      # Money Flow indicator
│   │   └── heikin_ashi.py     # Heikin Ashi candles
│   ├── strategy/
│   │   ├── signals.py         # Signal detection
│   │   ├── risk.py            # Risk management
│   │   └── trade_manager.py   # Trade execution
│   └── main.py                # Entry point
├── backtest/
│   ├── engine.py              # Backtest engine
│   ├── data_loader.py         # Data fetching
│   └── metrics.py             # Performance metrics
├── output/
│   └── *.txt                  # Generated reports
├── run.py                     # Start the bot
├── fetch_and_backtest.py      # Run backtests
└── requirements.txt           # Dependencies
```

---

## Discord Notifications

Enable Discord alerts for trade notifications:

```yaml
discord:
  enabled: true
  webhook_url: "https://discord.com/api/webhooks/..."
  notify_on_signal: true
  notify_on_trade_open: true
  notify_on_trade_close: true
```

---

## Commands

| Command | Description |
|---------|-------------|
| `python run.py` | Start live trading bot |
| `python fetch_and_backtest.py` | Run comprehensive backtest |
| `python run_comparison_backtest.py` | Compare exit strategies |
| `python test_hyperliquid_live.py` | Test Hyperliquid connection |

---

## Troubleshooting

### Hyperliquid Connection Issues

```python
# Test connection
python test_hyperliquid_live.py
```

### Rate Limiting

If you see 429 errors, the bot automatically handles rate limiting with exponential backoff.

### Missing Data

Backtest data is cached in `data/binance_cache/`. Delete this folder to re-fetch.

---

## Disclaimer

This software is for educational purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.

---

## License

MIT License
