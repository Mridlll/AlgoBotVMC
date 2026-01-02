# VMC Trading Bot V6

Automated cryptocurrency trading bot based on the VuManChu Cipher B indicator strategy. Trades BTC, ETH, and SOL perpetual futures on Hyperliquid exchange.

## Performance Summary

**Backtest Period:** December 2023 - December 2024 (1 Year)
**Starting Capital:** $10,000
**Risk Per Trade:** 3%

| Metric | Value |
|--------|-------|
| **Total PnL** | $974,457 |
| **Average Sharpe Ratio** | 2.79 |
| **Win Rate** | 31-47% |
| **Profit Factor** | 1.14-6.27 |

### Per-Asset Performance

| Asset | Net PnL | VWAP Filter | Strategies |
|-------|---------|-------------|------------|
| BTC | $99,716 | Enabled | 5 |
| ETH | $237,340 | Disabled | 5 |
| SOL | $485,072 | Disabled | 5 |

### Top Strategies

| Strategy | PnL | Sharpe | Time Filter |
|----------|-----|--------|-------------|
| SOL 5m SIMPLE | $166,586 | 1.26 | NY Hours |
| SOL 5m ENHANCED_60 | $152,788 | 1.39 | NY Hours |
| ETH 5m SIMPLE | $78,946 | 0.93 | NY Hours |
| SOL 1h ENHANCED_70 | $74,115 | 6.88 | Weekends |
| BTC 5m ENHANCED_60 | $54,339 | 2.82 | NY Hours |

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YourUsername/vmc_trading_bot.git
cd vmc_trading_bot

# 2. Run setup (creates virtual environment, installs dependencies)
# Windows:
setup.bat
# Linux/Mac:
./setup.sh

# 3. Run the configuration wizard
python setup_wizard.py

# 4. Start the bot
# Windows:
run_bot.bat
# Linux/Mac:
./run_bot.sh

# Or use production mode with auto-restart:
python run_production.py
```

---

## Prerequisites

- **Python 3.10+** - [Download Python](https://www.python.org/downloads/)
- **Hyperliquid Account** - [Sign Up](https://app.hyperliquid.xyz)
- **USDC Funds** - Deposit USDC to your Hyperliquid wallet
- **Discord** (optional) - For trade notifications

---

## Hyperliquid Wallet Setup

Hyperliquid uses a two-wallet architecture for security:

### 1. Main Wallet (Holds Funds)
- This is your primary wallet connected to Hyperliquid
- Deposit USDC here
- Note your wallet address (0x...)

### 2. API Wallet (Signs Trades)
1. Go to https://app.hyperliquid.xyz/API
2. Click **"Generate API Wallet"**
3. **SAVE THE PRIVATE KEY** - You will only see it once!
4. Note the API wallet address
5. Authorize the API wallet to trade on your behalf

### Security Notes
- The API wallet can only trade - it cannot withdraw funds
- Your main wallet funds are protected even if API key is compromised
- Never share your private key with anyone
- Store the private key in a password manager

---

## Installation

### Automatic Setup (Recommended)

**Windows:**
```batch
setup.bat
```

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create a Python virtual environment
- Install all dependencies
- Create necessary directories
- Copy the example config file

### Manual Setup

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data logs
```

---

## Configuration

### Using the Setup Wizard (Recommended)

```bash
python setup_wizard.py
```

The wizard guides you through:
1. Wallet configuration (addresses, private key)
2. Risk settings (risk per trade, leverage)
3. Strategy selection (15 strategies available)
4. Discord notifications (optional)

### Manual Configuration

Edit `config/config.yaml`:

```yaml
exchange:
  name: "hyperliquid"
  api_secret: "0x..."           # Your API wallet private key
  wallet_address: "0x..."       # Your API wallet address
  account_address: "0x..."      # Your main wallet address
  testnet: true                 # Start with testnet!

trading:
  risk_percent: 2.0             # Risk per trade (1-5%)
  leverage: 2.0                 # 1-5x leverage
  max_positions: 3              # Max simultaneous positions

discord:
  enabled: true
  webhook_url: "https://discord.com/api/webhooks/..."
```

### Strategy Configuration

All 15 strategies are enabled by default. To disable a strategy:

```yaml
strategies:
  btc_5m_enhanced60_ny:
    enabled: false              # Set to false to disable
```

---

## Testnet vs Mainnet

### Testnet Mode (Default - Recommended for Testing)

The bot runs on **testnet by default** for safe practice trading:

```yaml
exchange:
  testnet: true    # Paper trading, no real funds
```

**Testnet features:**
- Uses Hyperliquid testnet (fake money)
- Get testnet USDC at: https://app.hyperliquid.xyz/drip
- All features work identically to mainnet
- Perfect for testing configuration and strategies

### Mainnet Mode (Real Money Trading)

**WARNING: Mainnet uses real funds! Only switch after thorough testnet testing.**

To switch to mainnet:

1. **Use the mainnet template:**
   ```bash
   cp config/config_v6_mainnet.example.yaml config/config.yaml
   ```

2. **Or modify your existing config:**
   ```yaml
   exchange:
     testnet: false    # REAL MONEY TRADING!
   ```

3. **Add your mainnet credentials** (from https://app.hyperliquid.xyz/API)

### Mainnet Safety Checklist

Before going live, ensure you have:

- [ ] Tested on testnet for at least 1 week
- [ ] Verified all settings work correctly
- [ ] Set conservative risk (1-2% per trade)
- [ ] Enabled Discord notifications for monitoring
- [ ] Only deposited funds you can afford to lose
- [ ] Understood the strategy and expected drawdowns

### Risk Settings Comparison

| Setting | Testnet (Testing) | Mainnet (Conservative) | Mainnet (Moderate) |
|---------|-------------------|------------------------|-------------------|
| `risk_percent` | 3.0% | 1.0% | 2.0% |
| `leverage` | 3.0x | 2.0x | 3.0x |
| `max_positions` | 3 | 3 | 3 |

---

## Running the Bot

### Basic Mode

```bash
# Windows
run_bot.bat

# Linux/Mac
./run_bot.sh
```

### Production Mode (Recommended)

The production runner includes:
- Auto-restart on crash with exponential backoff
- Emergency position closing on crash
- Health monitoring and logging
- State persistence across restarts

```bash
python run_production.py --config config/config.yaml
```

Options:
```
--config, -c        Config file path (default: config/config.yaml)
--max-restarts, -m  Max restart attempts (default: 10)
--no-close-positions Don't close positions on crash
--reset-state       Reset restart counter
```

### Running as a Service

See the [Service Setup](#service-setup) section below for running the bot automatically on system boot.

---

## Monitoring

### Log Files

- `logs/production.log` - Main bot activity
- `logs/bot.log` - Detailed trading logs

### Discord Notifications

When enabled, you'll receive notifications for:
- Bot started/stopped
- Trade opened (entry price, size, stop loss)
- Trade closed (exit price, PnL)
- Errors and crashes

### Hyperliquid Dashboard

View your positions at: https://app.hyperliquid.xyz

---

## Strategy Overview

### How It Works

1. **Data Collection**: Fetches OHLCV candles from Hyperliquid
2. **Heikin Ashi**: Converts to smoothed candles
3. **Indicators**: Calculates WaveTrend oscillator and MFI
4. **Signal Detection**:
   - LONG: WT1 crosses above WT2 from oversold (-53/-60/-70)
   - SHORT: WT1 crosses below WT2 from overbought (+53/+60/+70)
5. **VWAP Confirmation** (BTC only):
   - LONG: Price must be above VWAP
   - SHORT: Price must be below VWAP
6. **Entry**: Market order with ATR-based stop loss
7. **Exit**: Wait for opposite WaveTrend signal OR stop loss hit

### Time Filters

- **NY_HOURS_ONLY**: Weekdays 14:00-21:00 UTC (NYSE trading hours)
- **WEEKENDS_ONLY**: Saturday and Sunday only

### Risk Management

| Parameter | Value |
|-----------|-------|
| Stop Loss | 2 x ATR(14) |
| Exit Strategy | Opposite WaveTrend signal |
| Max Positions | 1 per asset |
| Commission | 0.06% (Hyperliquid maker fee) |

### Why Low Win Rate Works

- Win rate is 31-47% but average wins are 2-7x larger than losses
- The strategy lets winners run and cuts losers quickly
- Sharpe ratios above 2.0 indicate excellent risk-adjusted returns

---

## Service Setup

### Linux (systemd)

1. Create the service file:

```bash
sudo nano /etc/systemd/system/vmc-bot.service
```

2. Add this content (adjust paths):

```ini
[Unit]
Description=VMC Trading Bot V6
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/path/to/vmc_trading_bot
ExecStart=/path/to/vmc_trading_bot/venv/bin/python run_production.py
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

3. Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable vmc-bot
sudo systemctl start vmc-bot
```

4. Check status:

```bash
sudo systemctl status vmc-bot
```

### Windows (Task Scheduler)

1. Open Task Scheduler
2. Create Basic Task -> Name: "VMC Trading Bot"
3. Trigger: "When the computer starts"
4. Action: "Start a program"
5. Program: `C:\path\to\vmc_trading_bot\venv\Scripts\python.exe`
6. Arguments: `run_production.py`
7. Start in: `C:\path\to\vmc_trading_bot`
8. Check "Run with highest privileges"
9. Check "Run whether user is logged on or not"

---

## Troubleshooting

### "Config file not found"

Run the setup wizard first:
```bash
python setup_wizard.py
```

### "Failed to connect to exchange"

1. Check your internet connection
2. Verify wallet addresses are correct (start with 0x, 42 characters)
3. Verify private key is correct (64 hex characters)
4. Check if Hyperliquid is operational

### "Insufficient margin"

1. Add more USDC to your account
2. Reduce position size (lower risk_percent)
3. Reduce leverage

### "Order failed"

1. Check if the market is open
2. Verify you have enough margin
3. Check for rate limiting (too many orders)

### Bot Not Taking Trades

1. Check time filter settings (NY hours or weekends)
2. Verify strategy is enabled in config
3. Check logs for signal detection: `logs/bot.log`

### View Logs

```bash
# Windows
type logs\production.log

# Linux/Mac
tail -f logs/production.log
```

---

## Backtesting

Verify strategy performance on historical data before going live.

### Running a Backtest

```bash
# Test all 15 strategies (1 year of data)
python run_backtest.py

# Test specific asset
python run_backtest.py --asset BTC

# Test last 90 days
python run_backtest.py --days 90

# Test SOL strategies for 6 months
python run_backtest.py --asset SOL --days 180
```

### Backtest Output

The backtest generates:
1. **Summary table** - PnL, win rate, Sharpe ratio per asset
2. **Strategy details** - Performance of each strategy
3. **Trade log CSV** - All trades exported to `output/backtest_trades.csv`

Example output:
```
============================================================
VMC Trading Bot V6 - Backtest Results
============================================================
Period: Last 365 days
Starting Capital: $10,000

RESULTS BY ASSET:
Asset       PnL      Win Rate   Trades   Sharpe
BTC      $99,716      42.3%       156     2.82
ETH     $237,340      38.5%       412     2.91
SOL     $485,072      31.2%       687     2.64
============================================================
```

---

## Daily Summary Logs

The bot automatically generates daily performance summaries at midnight UTC.

### Log Location

Daily summaries are written to: `logs/daily_summary.log`

### Summary Contents

Each daily summary includes:
- Trades opened/closed that day
- Daily PnL
- Running balance
- Win rate for the day
- Active positions with unrealized PnL

### Discord Notifications

If Discord is enabled, daily summaries are also sent as notifications with:
- Color-coded embeds (green = profit, red = loss)
- Key metrics at a glance
- Active position overview

---

## File Structure

```
vmc_trading_bot/
├── config/
│   ├── config.yaml                    # Your configuration
│   ├── config_v6_production.yaml      # Testnet template
│   └── config_v6_mainnet.example.yaml # Mainnet template
├── src/
│   ├── core/                    # Bot core logic
│   ├── exchanges/               # Hyperliquid integration
│   ├── indicators/              # Technical indicators
│   ├── strategy/                # Trading strategies
│   └── utils/                   # Utilities (logger, daily summary)
├── backtest/                    # Backtesting engine
├── logs/
│   ├── production.log           # Main bot log
│   └── daily_summary.log        # Daily performance summaries
├── output/                      # Backtest results
├── data/                        # Trade database
├── run_backtest.py              # Client backtest script
├── setup_wizard.py              # Configuration wizard
├── run_production.py            # Production runner
├── setup.bat / setup.sh         # Setup scripts
└── run_bot.bat / run_bot.sh     # Run scripts
```

---

## Updating

```bash
# Pull latest changes
git pull origin main

# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Restart the bot
```

---

## Risk Disclaimer

**IMPORTANT - READ CAREFULLY:**

- **Past performance does not guarantee future results**
- Cryptocurrency trading involves significant risk of loss
- This bot is provided "as-is" without warranties
- Only trade with funds you can afford to lose
- Start with testnet to understand the system
- The developers are not responsible for any trading losses
- Backtest results may not reflect live trading performance

**Recommendations:**
- Start with testnet mode
- Use conservative risk settings (1-2% per trade)
- Monitor the bot regularly
- Keep emergency funds available
- Understand the strategy before going live

---

## Support

- Check the logs for error messages
- Review this README for common issues
- Contact support with log files

---

## Version History

### V6 (Current)
- Per-asset VWAP optimization
- 15 optimized strategies
- Fixed VWAP calculation bug
- Average Sharpe: 2.79

### V5 (Previous)
- 9 strategies
- BTC SHORT_ONLY restriction
- Average Sharpe: 0.31

---

*Last Updated: January 2026*
