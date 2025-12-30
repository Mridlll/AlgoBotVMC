# VMC Trading Bot V6 - Client Deployment Guide

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Hyperliquid Wallet Setup](#hyperliquid-wallet-setup)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Running the Bot](#running-the-bot)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

---

## Overview

VMC Trading Bot V6 is an automated cryptocurrency trading system based on the VuManChu Cipher B indicator strategy. It trades BTC, ETH, and SOL on Hyperliquid exchange.

### V6 Performance (1-Year Backtest)

| Metric | Value |
|--------|-------|
| Starting Capital | $10,000 |
| Risk Per Trade | 3% |
| Total PnL (Top 15 configs) | $822,128 |
| Realistic PnL (with overlap) | ~$534,383 |
| Average Sharpe Ratio | 2.79 |
| Win Rate | 31-47% |

### Per-Asset Performance

| Asset | Net PnL | VWAP Filter | Direction |
|-------|---------|-------------|-----------|
| BTC | $99,716 | ENABLED | Both |
| ETH | $237,340 | DISABLED | Both |
| SOL | $485,072 | DISABLED | Both |

---

## Prerequisites

### System Requirements
- Python 3.10 or higher
- 4GB RAM minimum
- Stable internet connection
- Windows 10/11, macOS, or Linux

### Required Accounts
1. **Hyperliquid Account** - https://app.hyperliquid.xyz
2. **Discord Account** (optional) - For trade notifications

---

## Hyperliquid Wallet Setup

Hyperliquid uses a two-wallet architecture for API trading:

### Step 1: Create Main Wallet
1. Go to https://app.hyperliquid.xyz
2. Connect your wallet (MetaMask, Rabby, etc.)
3. Deposit USDC to your account
4. Note your **Main Wallet Address** (e.g., `0x1234...abcd`)

### Step 2: Create API Wallet
1. Go to https://app.hyperliquid.xyz/API
2. Click "Generate API Wallet"
3. **SAVE THE PRIVATE KEY SECURELY** - You will only see it once!
4. Note your **API Wallet Address**

### Step 3: Authorize API Wallet
1. On the API page, authorize the API wallet to trade on behalf of your main wallet
2. Set the authorization amount (e.g., max position size)
3. Confirm the transaction

### Important Security Notes
- **NEVER share your private key**
- Store the private key in a secure password manager
- The API wallet signs transactions; the main wallet holds funds
- You can revoke API wallet access anytime from the API page

---

## Installation

### Option 1: Using Setup Script (Recommended)

**Windows:**
```batch
cd vmc_trading_bot
setup.bat
```

**Linux/macOS:**
```bash
cd vmc_trading_bot
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Installation

```bash
# Clone the repository
git clone https://github.com/YourUsername/vmc_trading_bot.git
cd vmc_trading_bot

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Required Python Packages
```
hyperliquid-python-sdk>=0.5.0
pandas>=2.0.0
numpy>=1.24.0
pyyaml>=6.0
aiohttp>=3.8.0
eth-account>=0.8.0
python-dotenv>=1.0.0
```

---

## Configuration

### Step 1: Copy Example Config
```bash
cp config/config_v6_production.yaml config/config.yaml
```

### Step 2: Edit Configuration

Open `config/config.yaml` and update these critical sections:

#### Exchange Credentials
```yaml
exchange:
  name: "hyperliquid"
  api_key: ""                          # Leave empty for Hyperliquid
  api_secret: "YOUR_API_WALLET_PRIVATE_KEY"  # From Step 2 above
  wallet_address: "YOUR_API_WALLET_ADDRESS"   # API wallet address
  account_address: "YOUR_MAIN_WALLET_ADDRESS" # Main wallet (with funds)
  testnet: true                        # START WITH TRUE!
```

#### Risk Settings
```yaml
trading:
  risk_percent: 2.0          # 2% recommended for production
  max_positions: 3           # 1 per asset
  leverage: 2.0              # Conservative leverage
```

#### Discord Notifications (Optional)
```yaml
discord:
  enabled: true
  webhook_url: "YOUR_DISCORD_WEBHOOK_URL"
```

### Step 3: Test on Testnet First!

1. Get testnet funds from Hyperliquid testnet faucet
2. Set `testnet: true` in config
3. Run the bot for 24-48 hours
4. Verify trades are executing correctly
5. Only then switch to `testnet: false`

---

## Running the Bot

### Start the Bot

**Windows:**
```batch
run_bot.bat
```

**Linux/macOS:**
```bash
./run_bot.sh
```

**Direct Python:**
```bash
python main.py --config config/config.yaml
```

### Run in Background (Linux)
```bash
nohup python main.py --config config/config.yaml > logs/bot.log 2>&1 &
```

### Run with Screen (Linux)
```bash
screen -S vmc_bot
python main.py --config config/config.yaml
# Press Ctrl+A, then D to detach
# screen -r vmc_bot to reattach
```

### Run as Windows Service
Use NSSM (Non-Sucking Service Manager):
```batch
nssm install VMCBot "C:\path\to\python.exe" "C:\path\to\main.py --config config/config.yaml"
nssm start VMCBot
```

---

## Monitoring

### Log Files
- Main log: `logs/vmc_bot.log`
- Trade log: `logs/trades.log`

### Discord Notifications
The bot sends notifications for:
- New signals detected
- Trade opened (with entry price, size, stop loss)
- Trade closed (with PnL)
- Errors

### Check Positions on Hyperliquid
1. Go to https://app.hyperliquid.xyz
2. Connect your wallet
3. View "Positions" tab

### Database
Trade history is stored in `data/trades.db` (SQLite)

---

## Strategy Overview

### How It Works

1. **Data Collection**: Fetches OHLCV candles from Hyperliquid
2. **Heikin Ashi**: Converts to smoothed candles
3. **Indicators**: Calculates WaveTrend, MFI, VWAP
4. **Signal Detection**:
   - LONG: WT1 crosses above WT2 from oversold (-53/-60/-70)
   - SHORT: WT1 crosses below WT2 from overbought (+53/+60/+70)
5. **VWAP Confirmation** (BTC only):
   - LONG: Price must be above VWAP
   - SHORT: Price must be below VWAP
6. **Entry**: Market order with ATR-based stop loss
7. **Exit**: Wait for opposite WaveTrend signal OR stop loss

### Time Filters
- **NY_HOURS_ONLY**: Weekdays 14:00-21:00 UTC (NYSE hours)
- **WEEKENDS_ONLY**: Saturday and Sunday only

### Why These Settings?
- BTC with VWAP filter prevents bad entries during downtrends
- ETH/SOL without VWAP allows more trades
- Low win rate (35%) but wins are 2-7x larger than losses

---

## Troubleshooting

### Common Issues

#### "Not connected to Hyperliquid"
- Check internet connection
- Verify API credentials are correct
- Ensure wallet addresses are correct (with 0x prefix)

#### "Order failed: insufficient margin"
- Reduce position size
- Reduce leverage
- Add more funds to account

#### "Private key invalid"
- Private key should start with `0x`
- Ensure no extra spaces
- Regenerate API wallet if needed

#### Bot not taking trades
- Check time filter settings (NY hours or weekends)
- Verify the strategy is enabled in config
- Check logs for signal detection

### Getting Help
1. Check `logs/vmc_bot.log` for errors
2. Review this guide
3. Contact support with log files

---

## Risk Disclaimer

**IMPORTANT**:
- Past performance does not guarantee future results
- Cryptocurrency trading involves significant risk
- Only trade with funds you can afford to lose
- Start with testnet and small amounts
- The developers are not responsible for any losses

---

## Quick Start Checklist

- [ ] Python 3.10+ installed
- [ ] Hyperliquid account created
- [ ] Main wallet funded with USDC
- [ ] API wallet created and authorized
- [ ] Private key saved securely
- [ ] Config file updated with credentials
- [ ] `testnet: true` set initially
- [ ] Bot tested on testnet
- [ ] Discord webhook configured (optional)
- [ ] Switch to `testnet: false` for live trading

---

## Version History

### V6 (Current)
- Per-asset VWAP optimization
- BTC: VWAP enabled (both directions)
- ETH/SOL: VWAP disabled
- Fixed VWAP calculation bug
- Average Sharpe: 2.79

### V5 (Previous)
- 9 strategies (3 per asset)
- BTC SHORT_ONLY restriction
- Average Sharpe: 0.31

---

*Last Updated: December 2025*
