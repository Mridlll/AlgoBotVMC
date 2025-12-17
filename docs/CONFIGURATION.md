# Configuration Reference

Complete reference for all configuration options in `config/config.yaml`.

---

## Exchange Configuration

```yaml
exchange:
  name: "hyperliquid"     # Exchange name: hyperliquid or bitunix
  api_key: ""             # API key (required for Bitunix)
  api_secret: ""          # API secret / private key
  wallet_address: ""      # Wallet address (Hyperliquid only)
  testnet: true           # Use testnet for testing
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Exchange to use: `hyperliquid` or `bitunix` |
| `api_key` | string | Bitunix only | API key from exchange |
| `api_secret` | string | Yes | API secret or private key |
| `wallet_address` | string | Hyperliquid only | Your wallet address |
| `testnet` | boolean | No | Use testnet (default: true) |

---

## Trading Configuration

```yaml
trading:
  assets:
    - "BTC"
    - "ETH"
  timeframe: "4h"
  risk_percent: 3.0
  max_positions: 2
  max_positions_per_asset: 1
  leverage: 1.0
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `assets` | list | ["BTC", "ETH"] | Trading pairs |
| `timeframe` | string | "4h" | Candle timeframe |
| `risk_percent` | float | 3.0 | Risk per trade (%) |
| `max_positions` | int | 2 | Max total positions |
| `max_positions_per_asset` | int | 1 | Max positions per asset |
| `leverage` | float | 1.0 | Trading leverage |

**Supported Timeframes:**
- `1m`, `5m`, `15m`, `30m`
- `1h`, `2h`, `4h`, `6h`, `12h`
- `1d`, `1w`

---

## Indicator Configuration

```yaml
indicators:
  wt_channel_len: 9
  wt_average_len: 12
  wt_ma_len: 3
  wt_overbought_1: 53
  wt_overbought_2: 60
  wt_oversold_1: -53
  wt_oversold_2: -60
  mfi_period: 60
  mfi_multiplier: 150.0
  mfi_y_pos: 2.5
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `wt_channel_len` | int | 9 | WaveTrend channel length |
| `wt_average_len` | int | 12 | WaveTrend average length |
| `wt_ma_len` | int | 3 | WaveTrend MA length |
| `wt_overbought_1` | int | 53 | First overbought level |
| `wt_overbought_2` | int | 60 | Anchor level for shorts |
| `wt_oversold_1` | int | -53 | First oversold level |
| `wt_oversold_2` | int | -60 | Anchor level for longs |
| `mfi_period` | int | 60 | Money Flow period |
| `mfi_multiplier` | float | 150.0 | Money Flow multiplier |
| `mfi_y_pos` | float | 2.5 | Money Flow Y offset |

---

## Stop Loss Configuration

```yaml
stop_loss:
  method: "swing"
  swing_lookback: 5
  buffer_percent: 0.5
  atr_multiplier: 1.5
  atr_period: 14
  fixed_percent: 2.0
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `method` | string | "swing" | Method: swing, atr, fixed_percent |
| `swing_lookback` | int | 5 | Candles for swing detection |
| `buffer_percent` | float | 0.5 | Buffer below/above swing |
| `atr_multiplier` | float | 1.5 | ATR multiplier (if using ATR) |
| `atr_period` | int | 14 | ATR calculation period |
| `fixed_percent` | float | 2.0 | Fixed SL distance (%) |

**Methods:**
- `swing`: Place SL below recent swing low (long) or above swing high (short)
- `atr`: Place SL at ATR distance from entry
- `fixed_percent`: Place SL at fixed percentage from entry

---

## Take Profit Configuration

```yaml
take_profit:
  method: "fixed_rr"
  risk_reward: 2.0
  partial_tp_percent: 50.0
  partial_tp_rr: 1.0
  move_sl_to_breakeven: true
  trailing_activation_rr: 1.5
  trailing_distance_percent: 1.0
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `method` | string | "fixed_rr" | Method: fixed_rr, oscillator, partial, trailing |
| `risk_reward` | float | 2.0 | Target R:R ratio |
| `partial_tp_percent` | float | 50.0 | % to close at first TP |
| `partial_tp_rr` | float | 1.0 | First TP R:R level |
| `move_sl_to_breakeven` | bool | true | Move SL after partial TP |
| `trailing_activation_rr` | float | 1.5 | R:R to activate trailing |
| `trailing_distance_percent` | float | 1.0 | Trailing distance (%) |

**Methods:**
- `fixed_rr`: Close at fixed R:R ratio (e.g., 2R)
- `oscillator`: Close when opposite signal appears
- `partial`: Take partial profit, move SL to breakeven
- `trailing`: Use trailing stop after activation

---

## Webhook Configuration

```yaml
webhook:
  enabled: false
  host: "0.0.0.0"
  port: 8080
  secret: ""
  path: "/webhook"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | false | Enable webhook server |
| `host` | string | "0.0.0.0" | Server bind address |
| `port` | int | 8080 | Server port |
| `secret` | string | "" | Webhook validation secret |
| `path` | string | "/webhook" | Webhook endpoint path |

---

## Discord Configuration

```yaml
discord:
  enabled: false
  webhook_url: ""
  notify_on_signal: true
  notify_on_trade_open: true
  notify_on_trade_close: true
  notify_on_error: true
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | false | Enable Discord notifications |
| `webhook_url` | string | "" | Discord webhook URL |
| `notify_on_signal` | bool | true | Notify on new signals |
| `notify_on_trade_open` | bool | true | Notify when trade opens |
| `notify_on_trade_close` | bool | true | Notify when trade closes |
| `notify_on_error` | bool | true | Notify on errors |

---

## Backtest Configuration

```yaml
backtest:
  start_date: "2024-01-01"
  end_date: "2024-12-01"
  initial_balance: 10000.0
  commission_percent: 0.06
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `start_date` | string | null | Backtest start date |
| `end_date` | string | null | Backtest end date |
| `initial_balance` | float | 10000.0 | Starting balance |
| `commission_percent` | float | 0.06 | Trading fee (%) |
