# VMC Trading Bot - User Guide

## Overview

VMC Trading Bot is an automated trading system based on the Visual Market Cipher (VMC) oscillator strategy. It trades perpetual futures on Hyperliquid and Bitunix exchanges.

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the Bot](#running-the-bot)
5. [Backtesting](#backtesting)
6. [Discord Notifications](#discord-notifications)
7. [TradingView Webhooks](#tradingview-webhooks)
8. [Strategy Explanation](#strategy-explanation)
9. [Troubleshooting](#troubleshooting)

---

## Requirements

- Python 3.9 or higher
- Exchange account (Hyperliquid or Bitunix)
- API credentials
- Stable internet connection

## Installation

### Step 1: Clone/Download the Bot

Place the `vmc_trading_bot` folder in your desired location.

### Step 2: Install Dependencies

```bash
cd vmc_trading_bot
pip install -r requirements.txt
```

### Step 3: Create Configuration

```bash
# Create config from example
cp config/config.example.yaml config/config.yaml
```

### Step 4: Edit Configuration

Edit `config/config.yaml` with your settings (see [Configuration](#configuration) section).

---

## Configuration

### Exchange Setup

#### Hyperliquid

1. Go to https://app.hyperliquid.xyz/API
2. Generate a new API wallet
3. Copy the private key (this is your `api_secret`)
4. Copy your main wallet address (this is `wallet_address`)

```yaml
exchange:
  name: "hyperliquid"
  api_key: ""  # Not used for Hyperliquid
  api_secret: "your_private_key_here"
  wallet_address: "0xYourWalletAddress"
  testnet: true  # Set to false for live trading
```

#### Bitunix

1. Log into Bitunix
2. Go to Account Settings > API Management
3. Create new API key with trading permissions
4. Copy both API Key and Secret

```yaml
exchange:
  name: "bitunix"
  api_key: "your_api_key"
  api_secret: "your_api_secret"
  testnet: true
```

### Trading Parameters

```yaml
trading:
  assets:
    - "BTC"
    - "ETH"
  timeframe: "4h"        # Recommended: 4h
  risk_percent: 3.0      # Risk per trade (1-5% recommended)
  max_positions: 2       # Maximum open positions
  max_positions_per_asset: 1
  leverage: 1.0          # Leverage (be careful!)
```

### Stop Loss Configuration

```yaml
stop_loss:
  method: "swing"        # swing, atr, or fixed_percent
  swing_lookback: 5      # Candles to look back
  buffer_percent: 0.5    # Buffer below swing level
```

### Take Profit Configuration

```yaml
take_profit:
  method: "fixed_rr"     # fixed_rr, oscillator, partial, trailing
  risk_reward: 2.0       # 2:1 risk reward

  # For partial method:
  partial_tp_percent: 50
  partial_tp_rr: 1.0
  move_sl_to_breakeven: true

  # For trailing method:
  trailing_activation_rr: 1.5
  trailing_distance_percent: 1.0
```

---

## Running the Bot

### Start Live Trading

```bash
python run.py run --config config/config.yaml
```

### Start with Paper Trading (Testnet)

Make sure `testnet: true` in your config, then:

```bash
python run.py run --config config/config.yaml
```

### Initialize Default Config

```bash
python run.py init --output config/config.yaml
```

---

## Backtesting

Run a backtest on historical data:

```bash
python run.py backtest --config config/config.yaml --start 2024-01-01 --end 2024-12-01
```

This will:
1. Download historical data from the exchange
2. Run the strategy simulation
3. Print performance metrics
4. Generate equity curve chart

### Backtest Output

```
==================================================
BACKTEST RESULTS: BTC
==================================================
Initial Balance:   $10,000.00
Final Balance:     $12,500.00
Total PnL:         $2,500.00 (25.00%)
Max Drawdown:      8.50%
Total Trades:      45
Win Rate:          55.6%
Profit Factor:     1.85
Sharpe Ratio:      1.42
```

---

## Discord Notifications

### Setup

1. Create a Discord webhook:
   - Server Settings > Integrations > Webhooks
   - Create new webhook
   - Copy webhook URL

2. Configure in config.yaml:

```yaml
discord:
  enabled: true
  webhook_url: "https://discord.com/api/webhooks/..."
  notify_on_signal: true
  notify_on_trade_open: true
  notify_on_trade_close: true
  notify_on_error: true
```

### Notification Types

- **Signal Detected**: New trading signal found
- **Trade Opened**: Position entered
- **Trade Closed**: Position exited with PnL
- **Error**: Any errors that occur

---

## TradingView Webhooks

You can optionally receive signals from TradingView instead of calculating locally.

### Setup

1. Enable webhook in config:

```yaml
webhook:
  enabled: true
  host: "0.0.0.0"
  port: 8080
  secret: "your_secret_key"
  path: "/webhook"
```

2. Start the bot (webhook server starts automatically)

3. In TradingView alert:
   - Webhook URL: `http://your-server:8080/webhook`
   - Alert message format:
```json
{
  "action": "buy",
  "symbol": "BTC",
  "price": {{close}},
  "wt1": {{plot_0}},
  "wt2": {{plot_1}}
}
```

---

## Strategy Explanation

### Entry Conditions

**Long Entry:**
1. **Anchor Wave**: WT2 drops below -60
2. **Trigger Wave**: WT2 makes a higher low (smaller dip)
3. **Money Flow**: Red wave starts curving up
4. **VWAP**: Crosses above 0
5. **Enter Long**

**Short Entry:**
1. **Anchor Wave**: WT2 rises above 60
2. **Trigger Wave**: WT2 makes a lower high + cross down
3. **Money Flow**: Green wave starts curving down
4. **VWAP**: Crosses below 0
5. **Enter Short**

### Risk Management

- Position size calculated based on risk % and stop loss distance
- Stop loss placed below recent swing low (longs) or above swing high (shorts)
- Take profit based on risk:reward ratio or oscillator reversal

---

## Troubleshooting

### Common Issues

#### "Failed to connect to exchange"
- Check API credentials
- Verify internet connection
- Try testnet first

#### "Not enough candles"
- Exchange may have limited historical data
- Try a longer timeframe

#### "Max positions reached"
- Bot won't open new trades until existing positions close
- Adjust `max_positions` in config

#### No signals generated
- Strategy is selective - signals may be rare
- Check indicator parameters match your TradingView setup
- Try backtesting to verify strategy works on historical data

### Logs

Check logs in `logs/bot.log` for detailed information.

### Support

For issues, check the logs first. Most problems are related to:
1. Invalid API credentials
2. Network connectivity
3. Exchange rate limits

---

## Important Warnings

1. **Test on testnet first** - Always verify the bot works correctly before using real funds

2. **Start with small amounts** - Even after testing, start with minimal capital

3. **Monitor actively** - Don't leave the bot completely unattended initially

4. **Understand the risks** - Trading involves significant risk of loss

5. **No guarantees** - Past performance doesn't guarantee future results
