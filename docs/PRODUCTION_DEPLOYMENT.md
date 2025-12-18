# VMC Trading Bot - Production Deployment Guide

This guide covers deploying the VMC Trading Bot for live trading with the V3.1 optimized configuration.

---

## Overview

The V3.1 configuration has been optimized through extensive backtesting across 1 year of data (Dec 2024 - Dec 2025).

### Expected Performance (Based on Backtest)

| Asset | Timeframe | Exit Strategy | Annual Return | Max Drawdown | Win Rate | Trades/Year |
|-------|-----------|---------------|---------------|--------------|----------|-------------|
| BTC   | 4H        | Fixed R:R 2.0 | +51%          | 11.3%        | ~50%     | 34          |
| ETH   | 30m       | Full Signal   | +54%          | 47.2%        | ~45%     | 285         |
| SOL   | 4H        | Fixed R:R 2.0 | +26%          | 20.6%        | ~48%     | 32          |
| **Combined** | -  | -             | **+131%**     | -            | -        | ~351        |

**Note:** Past performance does not guarantee future results. Start with testnet and conservative position sizing.

---

## Quick Start

### 1. Install Dependencies

**Windows:**
```batch
setup.bat
```

**macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

### 2. Configure Credentials

Edit `config/config_production.yaml`:

```yaml
exchange:
  api_secret: "your_private_key_here"
  wallet_address: "0xYourWalletAddress"
  account_address: ""  # Leave empty unless using API wallet
  testnet: true        # Keep true for initial testing!
```

### 3. Set Up Discord Notifications (Recommended)

```yaml
discord:
  enabled: true
  webhook_url: "https://discord.com/api/webhooks/..."
```

### 4. Run the Bot

**Windows:**
```batch
run_bot.bat production
```

**macOS/Linux:**
```bash
./run_bot.sh production
```

---

## Detailed Configuration

### Per-Asset Settings

The production config uses different settings per asset based on backtest optimization:

```yaml
assets_config:
  BTC:
    timeframe: "4h"           # 4-hour candles
    signal_mode: "enhanced"   # 4-step state machine
    exit_strategy: "fixed_rr" # Fixed Risk:Reward
    risk_reward: 2.0          # 2:1 R:R ratio

  ETH:
    timeframe: "30m"          # 30-minute candles (more active)
    signal_mode: "enhanced"
    exit_strategy: "oscillator"
    oscillator_mode: "full_signal"  # Wait for opposite VMC signal

  SOL:
    timeframe: "4h"
    signal_mode: "enhanced"
    exit_strategy: "fixed_rr"
    risk_reward: 2.0
```

### Risk Management

Default production settings (conservative):

```yaml
trading:
  risk_percent: 1.5      # Risk 1.5% per trade
  max_positions: 6       # Max 6 total positions
  max_positions_per_asset: 2  # Max 2 per asset
  leverage: 1.5          # 1.5x leverage
```

**Adjusting Risk:**

| Risk Level | risk_percent | leverage | Expected DD |
|------------|--------------|----------|-------------|
| Conservative | 1.0        | 1.0      | ~10-15%     |
| Moderate   | 1.5          | 1.5      | ~15-25%     |
| Aggressive | 2.5          | 2.0      | ~25-40%     |

---

## Production Features

### Trade Persistence

Trades are automatically saved to a SQLite database (`data/trades.db`). This enables:
- **Crash Recovery:** Bot resumes managing open trades after restart
- **Trade History:** Full history of all trades for analysis
- **Session Tracking:** Track performance across bot sessions

### Exit Monitoring

The bot monitors active trades every 30 seconds for:
- Stop Loss / Take Profit order fills
- Oscillator exit conditions (for ETH's FULL_SIGNAL mode)
- Position changes on the exchange

### Position Reconciliation

Every 5 minutes, the bot syncs with the exchange to handle:
- Positions closed externally (manual or liquidation)
- Orphaned positions (opened outside the bot)
- State drift from network issues

---

## Deployment Checklist

### Pre-Deployment

- [ ] Run `setup.bat` or `setup.sh` successfully
- [ ] Configure `config_production.yaml` with credentials
- [ ] Set `testnet: true` for initial testing
- [ ] Set up Discord webhook for notifications
- [ ] Verify connection with testnet

### Testnet Validation (1-2 Weeks)

- [ ] Bot runs continuously without crashes
- [ ] Signals detected match expected frequency
- [ ] Trades open and close correctly
- [ ] SL/TP orders placed on exchange
- [ ] Discord notifications working
- [ ] Restart bot and verify trade recovery works

### Go-Live

- [ ] Change `testnet: false` in config
- [ ] Start with 50% of intended position sizes
- [ ] Monitor closely for first week
- [ ] Scale up gradually after stable operation

---

## Monitoring

### Log Files

Logs are output to console. For file logging, redirect output:

**Windows:**
```batch
run_bot.bat production > logs\bot.log 2>&1
```

**Linux:**
```bash
./run_bot.sh production > logs/bot.log 2>&1 &
```

### Key Log Messages

| Message | Meaning |
|---------|---------|
| `Signal detected: LONG BTC` | New trading signal identified |
| `Opened LONG trade: BTC @ 50000` | Trade executed successfully |
| `Trade closed: TP_HIT @ 52000` | Take profit reached |
| `Trade closed: SL_HIT @ 49000` | Stop loss hit |
| `Reconciling positions` | Syncing with exchange |
| `Recovered trade` | Trade loaded from database |

### Discord Alerts

Configure notifications for real-time alerts:

```yaml
discord:
  notify_on_signal: true      # New signal detected
  notify_on_trade_open: true  # Trade opened
  notify_on_trade_close: true # Trade closed (with P&L)
  notify_on_error: true       # Errors and warnings
```

---

## Running as a Service

### Windows (Task Scheduler)

1. Open Task Scheduler
2. Create Basic Task
3. Trigger: "When the computer starts"
4. Action: Start a program
   - Program: `C:\path\to\vmc_trading_bot\run_bot.bat`
   - Arguments: `production`
   - Start in: `C:\path\to\vmc_trading_bot`

### Linux (systemd)

Create `/etc/systemd/system/vmc-bot.service`:

```ini
[Unit]
Description=VMC Trading Bot
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/vmc_trading_bot
ExecStart=/path/to/vmc_trading_bot/run_bot.sh production
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable vmc-bot
sudo systemctl start vmc-bot
sudo systemctl status vmc-bot
```

---

## Troubleshooting

### Bot Not Trading

**Check:**
1. Is the bot connected? Look for "Connected to hyperliquid" in logs
2. Are there funds available? Check exchange balance
3. Is max_positions reached? Check active trades
4. Is it waiting for signals? VMC strategy can be quiet in ranging markets

### Trades Not Closing

**Check:**
1. Are SL/TP orders on exchange? Check Hyperliquid positions page
2. Is exit monitoring running? Look for "Monitoring active trades" in logs
3. For oscillator exits, bot waits for opposite signal

### Database Errors

If `data/trades.db` is corrupted:
```bash
# Backup and recreate
mv data/trades.db data/trades.db.bak
# Bot will create new database on restart
```

### Exchange Connection Issues

```yaml
# Try increasing timeouts if on slow connection
# Or check if Hyperliquid is operational
```

---

## Updating the Bot

```bash
git pull origin main
pip install -r requirements.txt
# Restart the bot
```

---

## Support

For issues:
1. Check logs for error messages
2. Verify configuration is correct
3. Ensure Hyperliquid is operational
4. Check your wallet has sufficient funds

---

## Disclaimer

Trading cryptocurrencies involves substantial risk. This bot is provided as-is without warranty. Always:
- Start with testnet
- Use only funds you can afford to lose
- Monitor the bot regularly
- Understand the strategy before deploying
