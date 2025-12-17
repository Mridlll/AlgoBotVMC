# VMC Trading Bot - Setup Guide

Complete step-by-step instructions to get the bot running.

---

## Prerequisites

Before starting, make sure you have:

- [ ] Windows 10/11, macOS, or Linux
- [ ] Python 3.10 or higher installed
- [ ] A Hyperliquid account with funds
- [ ] Git installed (optional, can download ZIP instead)

---

## Step 1: Download the Bot

### Option A: Using Git (Recommended)

```bash
git clone https://github.com/Mridlll/AlgoBotVMC.git
cd AlgoBotVMC
```

### Option B: Download ZIP

1. Go to https://github.com/Mridlll/AlgoBotVMC
2. Click the green "Code" button
3. Click "Download ZIP"
4. Extract the ZIP file
5. Open a terminal in the extracted folder

---

## Step 2: Set Up Python Environment

### Windows

```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### macOS / Linux

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

You should see `(venv)` at the start of your terminal prompt when activated.

---

## Step 3: Get Your Hyperliquid Credentials

### 3.1 Get Your Wallet Address

1. Go to https://app.hyperliquid.xyz/
2. Connect your wallet (MetaMask, etc.)
3. Click on your address in the top right
4. Copy the full address (starts with `0x...`)

### 3.2 Get Your Private Key

**For MetaMask:**

1. Open MetaMask
2. Click the three dots (⋮) next to your account
3. Click "Account Details"
4. Click "Show Private Key"
5. Enter your password
6. Copy the private key

**IMPORTANT:**
- Never share your private key with anyone
- Consider using a separate wallet for trading with limited funds
- The bot needs this to sign transactions

---

## Step 4: Configure the Bot

### 4.1 Create Your Config File

```bash
# Windows
copy config\config.example.yaml config\config.yaml

# macOS/Linux
cp config/config.example.yaml config/config.yaml
```

### 4.2 Edit the Config File

Open `config/config.yaml` in a text editor (Notepad, VS Code, etc.)

**Required Changes:**

```yaml
exchange:
  name: "hyperliquid"
  api_key: ""                              # Leave empty
  api_secret: "your_private_key_here"      # Paste your private key
  wallet_address: "0x..."                  # Paste your wallet address
  testnet: true                            # Start with testnet!
```

**Trading Settings (adjust as needed):**

```yaml
trading:
  assets:
    - "BTC"
    - "ETH"
    - "SOL"
  timeframe: "30m"           # Recommended
  risk_percent: 2.0          # 2% risk per trade (conservative)
  max_positions: 3
  leverage: 1.0              # No leverage to start
```

**Exit Strategy (recommended):**

```yaml
take_profit:
  method: "oscillator"
  oscillator_mode: "wt_cross"    # Most consistent performer
  risk_reward: 2.0
```

### 4.3 Save the File

Save and close the config file.

---

## Step 5: Test on Testnet First!

### 5.1 Get Testnet Funds

1. Go to https://app.hyperliquid-testnet.xyz/
2. Connect the same wallet
3. You'll get free testnet funds to practice

### 5.2 Verify Connection

```bash
python test_hyperliquid_live.py
```

You should see:
```
Connected to Hyperliquid testnet
Balance: $10,000.00 (or similar)
```

If you see errors, check:
- Is your private key correct?
- Is your wallet address correct?
- Is `testnet: true` in config?

---

## Step 6: Run a Backtest (Optional but Recommended)

Before risking real money, see how the bot performs on historical data:

```bash
python fetch_and_backtest.py
```

This will:
1. Download 3 months of price data
2. Test all strategies
3. Show you expected performance

Wait for it to complete (takes ~10-15 minutes).

Check the report at: `output/full_comparison_binance.txt`

---

## Step 7: Start the Bot

### Testnet Mode (Practice)

Make sure `testnet: true` in your config, then:

```bash
python run.py
```

### Live Mode (Real Money)

**Only after testing on testnet!**

1. Edit `config/config.yaml`
2. Change `testnet: true` to `testnet: false`
3. Run the bot:

```bash
python run.py
```

---

## Step 8: Monitor the Bot

The bot will print:
- When it detects signals
- When it opens/closes trades
- Current positions and P&L

### Stop the Bot

Press `Ctrl+C` to stop the bot safely.

---

## Recommended Settings by Risk Level

### Conservative (Lower Risk)

```yaml
trading:
  risk_percent: 1.0
  leverage: 1.0

take_profit:
  method: "oscillator"
  oscillator_mode: "first_reversal"
```

### Balanced (Medium Risk)

```yaml
trading:
  risk_percent: 2.0
  leverage: 1.0

take_profit:
  method: "oscillator"
  oscillator_mode: "wt_cross"
```

### Aggressive (Higher Risk)

```yaml
trading:
  risk_percent: 3.0
  leverage: 2.0

take_profit:
  method: "oscillator"
  oscillator_mode: "full_signal"
```

---

## Troubleshooting

### "Module not found" Error

Make sure your virtual environment is activated:
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### "Connection refused" Error

- Check your internet connection
- Hyperliquid might be down - try again later

### "Invalid API key" Error

- Double-check your private key in config.yaml
- Make sure there are no extra spaces

### Bot Runs But No Trades

This is normal! The VMC strategy waits for specific conditions:
1. Extreme WT2 level (anchor)
2. WT2 crosses back (trigger)
3. MFI confirms direction
4. VWAP crosses zero

This might take hours or days depending on market conditions.

### High CPU Usage

The bot checks for signals every candle. On 30m timeframe, it's mostly idle.
Lower timeframes (5m, 15m) will use more resources.

---

## Optional: Discord Notifications

Get alerts on your phone when trades happen:

1. Create a Discord server (or use existing)
2. Go to Server Settings → Integrations → Webhooks
3. Click "New Webhook"
4. Copy the webhook URL
5. Add to config:

```yaml
discord:
  enabled: true
  webhook_url: "https://discord.com/api/webhooks/..."
  notify_on_signal: true
  notify_on_trade_open: true
  notify_on_trade_close: true
```

---

## Quick Reference

| Command | What it does |
|---------|--------------|
| `python run.py` | Start the trading bot |
| `python fetch_and_backtest.py` | Run backtest analysis |
| `python test_hyperliquid_live.py` | Test exchange connection |
| `Ctrl+C` | Stop the bot |

---

## Support

If you have issues:
1. Check the Troubleshooting section above
2. Make sure all steps were followed correctly
3. Check if Hyperliquid is operational

---

## Safety Reminders

1. **Start with testnet** - Practice before using real money
2. **Start small** - Use low risk_percent (1-2%) initially
3. **Monitor regularly** - Don't leave the bot unattended for long periods
4. **Secure your keys** - Never share your private key
5. **Use a trading wallet** - Keep main funds in a separate wallet

Good luck!
