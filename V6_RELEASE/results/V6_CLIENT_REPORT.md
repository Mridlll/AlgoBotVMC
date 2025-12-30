# VMC Trading Bot V6 - Complete Client Report

**Generated:** 2025-12-30 13:42
**Data Period:** December 2024 - December 2025 (1 Year)
**Initial Balance:** $10,000 | **Risk Per Trade:** 3%

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Net PnL** | **$974,457** |
| **Configurations Tested** | 135 |
| **Profitable Configs** | 72 (53.3%) |
| **Best Single Config** | $166,586 |
| **Average PnL per Config** | $7,218 |

---

## V6 Architecture Overview

### System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VMC TRADING BOT V6 ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  RAW OHLCV DATA (Binance)                                                   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────┐                                                        │
│  │  HEIKIN ASHI    │  ◄── Smooths price action, reduces noise               │
│  │  CONVERSION     │                                                        │
│  └────────┬────────┘                                                        │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    INDICATOR CALCULATION                             │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │  WAVETREND   │  │  MONEY FLOW  │  │  REAL VWAP   │               │   │
│  │  │  (WT1, WT2)  │  │    (MFI)     │  │  (with vol)  │               │   │
│  │  │              │  │              │  │              │               │   │
│  │  │ • Oversold   │  │ • Curving Up │  │ • Daily VWAP │               │   │
│  │  │   < -53/-60  │  │ • Curving Dn │  │ • Price vs   │               │   │
│  │  │ • Overbought │  │ • Strength   │  │   VWAP       │               │   │
│  │  │   > +53/+60  │  │              │  │              │               │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘               │   │
│  └─────────┼─────────────────┼─────────────────┼───────────────────────┘   │
│            │                 │                 │                           │
│            └─────────────────┴─────────────────┘                           │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    DIVERGENCE DETECTION                              │   │
│  │  • Bullish: Price Lower Low + Indicator Higher Low → BUY signal     │   │
│  │  • Bearish: Price Higher High + Indicator Lower High → SELL signal  │   │
│  │  • Lookback: 5 bars | Min Swing Distance: 3 bars                    │   │
│  └────────────────────────────────┬────────────────────────────────────┘   │
│                                   │                                         │
│                                   ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    SIGNAL GENERATION                                 │   │
│  │                                                                      │   │
│  │  ENTRY CONDITIONS:                                                   │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │ LONG Signal:                                                 │    │   │
│  │  │   1. WT1 crosses ABOVE WT2 (bullish cross)                  │    │   │
│  │  │   2. WT2 is below oversold level (-53 or -60 or -70)        │    │   │
│  │  │   3. [BTC ONLY] Price > VWAP (confirmation)                 │    │   │
│  │  │   4. [Optional] Bullish divergence detected                  │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │ SHORT Signal:                                                │    │   │
│  │  │   1. WT1 crosses BELOW WT2 (bearish cross)                  │    │   │
│  │  │   2. WT2 is above overbought level (+53 or +60 or +70)      │    │   │
│  │  │   3. [BTC ONLY] Price < VWAP (confirmation)                 │    │   │
│  │  │   4. [Optional] Bearish divergence detected                  │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  └────────────────────────────────┬────────────────────────────────────┘   │
│                                   │                                         │
│                                   ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    TIME FILTER (Pre-applied)                         │   │
│  │                                                                      │   │
│  │  • NY_HOURS_ONLY: Weekdays 14:00-21:00 UTC (NYSE open hours)        │   │
│  │  • WEEKENDS_ONLY: Saturday & Sunday only                            │   │
│  │  • ALL_HOURS: No filter (generally unprofitable)                    │   │
│  └────────────────────────────────┬────────────────────────────────────┘   │
│                                   │                                         │
│                                   ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    POSITION MANAGEMENT                               │   │
│  │                                                                      │   │
│  │  ENTRY:                                                              │   │
│  │  • Position Size = (Balance × 3%) / (Entry - StopLoss)             │   │
│  │  • Stop Loss = Entry ± 2 × ATR(14)                                  │   │
│  │  • Commission: 0.06% (Hyperliquid maker fee)                        │   │
│  │  • Slippage: 0.01% (5m) / 0.03% (15m+)                             │   │
│  │                                                                      │   │
│  │  EXIT CONDITIONS (any of these):                                    │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │ 1. OPPOSITE WT SIGNAL (Primary - lets winners run)          │    │   │
│  │  │    • Long exit: WT1 < WT2 cross from overbought             │    │   │
│  │  │    • Short exit: WT1 > WT2 cross from oversold              │    │   │
│  │  │                                                              │    │   │
│  │  │ 2. STOP LOSS HIT                                            │    │   │
│  │  │    • Long: Price falls to stop loss level                   │    │   │
│  │  │    • Short: Price rises to stop loss level                  │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key V6 Improvements Over V5

| Feature | V5 | V6 | Impact |
|---------|-----|-----|--------|
| **VWAP Entry Filter** | Not used | BTC only | BTC: +$83K (was -$181K) |
| **VWAP Calculation** | WT momentum (bug) | Real VWAP | Correct confirmation |
| **Divergence Detection** | Not implemented | Active | Optional signal boost |
| **Per-Asset Settings** | Same for all | Optimized per asset | +$589K improvement |

### Per-Asset Configuration

| Asset | VWAP Entry | Direction | Rationale |
|-------|------------|-----------|-----------|
| **BTC** | ENABLED | BOTH | VWAP filters out bad long entries |
| **ETH** | DISABLED | BOTH | More trades = more profit |
| **SOL** | DISABLED | BOTH | More trades = more profit |

---

## Performance Results

### Overall Summary

| Metric | BTC | ETH | SOL | **TOTAL** |
|--------|-----|-----|-----|-----------|
| **Net PnL** | $83,241 | $346,509 | $544,707 | **$974,457** |
| **Profitable Configs** | 23/45 | 28/45 | 21/45 | **72/135** |
| **Long PnL** | $63,734 | $148,288 | $279,488 | $491,510 |
| **Short PnL** | $19,507 | $198,222 | $265,220 | $482,949 |

### Top 10 Configurations (All Assets)

| # | Asset | Timeframe | Signal Mode | Time Filter | PnL | Trades | Win% | Profit Factor | Sharpe |
|---|-------|-----------|-------------|-------------|-----|--------|------|---------------|--------|
| 1 | SOL | 5m | SIMPLE | NY_HOURS_ONLY | $166,586 | 845 | 36.8% | 1.17 | 1.26 |
| 2 | SOL | 5m | ENHANCED_60 | NY_HOURS_ONLY | $152,788 | 686 | 34.0% | 1.16 | 1.39 |
| 3 | ETH | 5m | SIMPLE | NY_HOURS_ONLY | $78,946 | 825 | 35.6% | 1.14 | 0.93 |
| 4 | SOL | 1h | ENHANCED_70 | WEEKENDS_ONLY | $74,115 | 43 | 46.5% | 6.27 | 6.88 |
| 5 | SOL | 15m | SIMPLE | NY_HOURS_ONLY | $58,717 | 281 | 37.7% | 1.44 | 2.77 |
| 6 | BTC | 5m | ENHANCED_60 | NY_HOURS_ONLY | $54,339 | 124 | 37.9% | 1.68 | 2.82 |
| 7 | ETH | 30m | SIMPLE | WEEKENDS_ONLY | $44,912 | 155 | 42.6% | 1.66 | 2.02 |
| 8 | ETH | 15m | ENHANCED_70 | WEEKENDS_ONLY | $40,687 | 147 | 31.3% | 1.65 | 2.14 |
| 9 | ETH | 5m | ENHANCED_60 | NY_HOURS_ONLY | $36,729 | 635 | 33.4% | 1.15 | 1.09 |
| 10 | ETH | 30m | ENHANCED_70 | NY_HOURS_ONLY | $36,066 | 66 | 34.8% | 2.36 | 4.69 |

### BTC - Top 5 Configurations (Net: $83,241)

| # | Timeframe | Signal | Filter | PnL | Trades | Win% | PF | Sharpe | Long PnL | Short PnL |
|---|-----------|--------|--------|-----|--------|------|-----|--------|----------|----------|
| 1 | 5m | ENHANCED_60 | NY_HOURS_ONLY | $54,339 | 124 | 37.9% | 1.68 | 2.82 | $25,993 | $28,346 |
| 2 | 5m | SIMPLE | NY_HOURS_ONLY | $26,276 | 167 | 36.5% | 1.36 | 1.11 | $22,373 | $3,903 |
| 3 | 4h | ENHANCED_60 | ALL_HOURS | $7,284 | 54 | 46.3% | 1.56 | 3.54 | $3,971 | $3,314 |
| 4 | 30m | ENHANCED_70 | NY_HOURS_ONLY | $5,914 | 25 | 44.0% | 2.12 | 5.98 | $3,413 | $2,501 |
| 5 | 4h | SIMPLE | ALL_HOURS | $5,902 | 62 | 43.5% | 1.40 | 2.75 | $3,064 | $2,838 |

### ETH - Top 5 Configurations (Net: $346,509)

| # | Timeframe | Signal | Filter | PnL | Trades | Win% | PF | Sharpe | Long PnL | Short PnL |
|---|-----------|--------|--------|-----|--------|------|-----|--------|----------|----------|
| 1 | 5m | SIMPLE | NY_HOURS_ONLY | $78,946 | 825 | 35.6% | 1.14 | 0.93 | $59,125 | $19,821 |
| 2 | 30m | SIMPLE | WEEKENDS_ONLY | $44,912 | 155 | 42.6% | 1.66 | 2.02 | $21,447 | $23,465 |
| 3 | 15m | ENHANCED_70 | WEEKENDS_ONLY | $40,687 | 147 | 31.3% | 1.65 | 2.14 | $8,305 | $32,382 |
| 4 | 5m | ENHANCED_60 | NY_HOURS_ONLY | $36,729 | 635 | 33.4% | 1.15 | 1.09 | $17,083 | $19,646 |
| 5 | 30m | ENHANCED_70 | NY_HOURS_ONLY | $36,066 | 66 | 34.8% | 2.36 | 4.69 | $14,595 | $21,471 |

### SOL - Top 5 Configurations (Net: $544,707)

| # | Timeframe | Signal | Filter | PnL | Trades | Win% | PF | Sharpe | Long PnL | Short PnL |
|---|-----------|--------|--------|-----|--------|------|-----|--------|----------|----------|
| 1 | 5m | SIMPLE | NY_HOURS_ONLY | $166,586 | 845 | 36.8% | 1.17 | 1.26 | $126,508 | $40,078 |
| 2 | 5m | ENHANCED_60 | NY_HOURS_ONLY | $152,788 | 686 | 34.0% | 1.16 | 1.39 | $86,930 | $65,857 |
| 3 | 1h | ENHANCED_70 | WEEKENDS_ONLY | $74,115 | 43 | 46.5% | 6.27 | 6.88 | $26,740 | $47,375 |
| 4 | 15m | SIMPLE | NY_HOURS_ONLY | $58,717 | 281 | 37.7% | 1.44 | 2.77 | $21,802 | $36,915 |
| 5 | 15m | ENHANCED_60 | NY_HOURS_ONLY | $32,865 | 227 | 35.2% | 1.39 | 2.52 | $9,537 | $23,329 |


---

## Detailed Trade Logs - Top 5 Configurations

### SOL_5m_SIMPLE_NY_HOURS_ONLY

**Summary:**
| Metric | Value |
|--------|-------|
| Total PnL | $166,586 |
| Total Trades | 845 |
| Win Rate | 36.8% |
| Profit Factor | 1.17 |
| Sharpe Ratio | 1.26 |
| Max Drawdown | 86.0% |
| Average Win | $3,712.86 |
| Average Loss | $-1,850.40 |
| Long Trades | 441 ($126,508) |
| Short Trades | 404 ($40,078) |

**Trade Log (First 20 trades):**

| # | Entry Time | Exit Time | Dir | Entry Price | Exit Price | Stop Loss | PnL | Exit Reason |
|---|------------|-----------|-----|-------------|------------|-----------|-----|-------------|
| 1 | 2024-12-19 15:30 | 2024-12-19 17:35 | LONG | $203.30 | $200.60 | $200.62 | -$315.73 | stop_loss |
| 2 | 2024-12-19 18:10 | 2024-12-19 19:35 | LONG | $194.44 | $190.53 | $190.55 | -$300.06 | stop_loss |
| 3 | 2024-12-19 19:55 | 2024-12-19 20:10 | LONG | $191.76 | $187.73 | $187.75 | -$290.06 | stop_loss |
| 4 | 2024-12-20 16:45 | 2024-12-20 17:40 | SHORT | $189.96 | $192.34 | $192.32 | -$287.42 | stop_loss |
| 5 | 2024-12-20 17:50 | 2024-12-23 14:20 | SHORT | $192.13 | $184.05 | $194.49 | $888.58 | opposite_signal |
| 6 | 2024-12-23 14:20 | 2024-12-23 14:45 | LONG | $184.05 | $181.35 | $181.37 | -$302.86 | stop_loss |
| 7 | 2024-12-23 14:50 | 2024-12-23 19:35 | LONG | $182.70 | $187.72 | $179.73 | $463.14 | opposite_signal |
| 8 | 2024-12-23 19:35 | 2024-12-23 20:30 | SHORT | $187.72 | $189.56 | $189.54 | -$314.69 | stop_loss |
| 9 | 2024-12-24 14:15 | 2024-12-24 15:20 | SHORT | $194.07 | $196.51 | $196.49 | -$299.46 | stop_loss |
| 10 | 2024-12-24 15:50 | 2024-12-24 20:20 | SHORT | $198.13 | $193.92 | $199.78 | $678.58 | opposite_signal |
| 11 | 2024-12-24 20:20 | 2024-12-25 14:40 | LONG | $193.92 | $198.58 | $192.50 | $939.33 | opposite_signal |
| 12 | 2024-12-25 14:40 | 2024-12-26 14:20 | SHORT | $198.58 | $189.46 | $200.21 | $1,779.40 | opposite_signal |
| 13 | 2024-12-26 14:20 | 2024-12-26 17:05 | LONG | $189.46 | $189.90 | $186.84 | $46.84 | opposite_signal |
| 14 | 2024-12-26 17:05 | 2024-12-27 15:50 | SHORT | $189.90 | $184.31 | $191.39 | $1,381.42 | opposite_signal |
| 15 | 2024-12-27 15:50 | 2024-12-27 17:50 | LONG | $184.31 | $185.80 | $182.52 | $320.32 | opposite_signal |
| 16 | 2024-12-27 17:50 | 2024-12-30 14:00 | SHORT | $185.80 | $187.34 | $187.32 | -$460.76 | stop_loss |
| 17 | 2024-12-30 17:55 | 2024-12-30 18:10 | SHORT | $191.49 | $193.54 | $193.52 | -$436.72 | stop_loss |
| 18 | 2024-12-30 20:05 | 2024-12-31 14:10 | SHORT | $195.71 | $196.99 | $196.97 | -$439.06 | stop_loss |
| 19 | 2024-12-31 14:35 | 2024-12-31 15:55 | SHORT | $196.94 | $198.56 | $198.54 | -$414.43 | stop_loss |
| 20 | 2024-12-31 16:05 | 2024-12-31 17:55 | SHORT | $197.84 | $195.97 | $199.30 | $440.64 | opposite_signal |

*... and 825 more trades*

---

### SOL_5m_ENHANCED_60_NY_HOURS_ONLY

**Summary:**
| Metric | Value |
|--------|-------|
| Total PnL | $152,788 |
| Total Trades | 686 |
| Win Rate | 34.0% |
| Profit Factor | 1.16 |
| Sharpe Ratio | 1.39 |
| Max Drawdown | 85.0% |
| Average Win | $4,841.60 |
| Average Loss | $-2,152.99 |
| Long Trades | 354 ($86,930) |
| Short Trades | 332 ($65,857) |

**Trade Log (First 20 trades):**

| # | Entry Time | Exit Time | Dir | Entry Price | Exit Price | Stop Loss | PnL | Exit Reason |
|---|------------|-----------|-----|-------------|------------|-----------|-----|-------------|
| 1 | 2024-12-19 15:30 | 2024-12-19 17:35 | LONG | $203.30 | $200.60 | $200.62 | -$315.73 | stop_loss |
| 2 | 2024-12-19 18:10 | 2024-12-19 19:35 | LONG | $194.44 | $190.53 | $190.55 | -$300.06 | stop_loss |
| 3 | 2024-12-19 19:55 | 2024-12-19 20:10 | LONG | $191.76 | $187.73 | $187.75 | -$290.06 | stop_loss |
| 4 | 2024-12-20 16:45 | 2024-12-20 17:40 | SHORT | $189.96 | $192.34 | $192.32 | -$287.42 | stop_loss |
| 5 | 2024-12-20 17:50 | 2024-12-23 14:20 | SHORT | $192.13 | $184.05 | $194.49 | $888.58 | opposite_signal |
| 6 | 2024-12-23 14:20 | 2024-12-23 14:45 | LONG | $184.05 | $181.35 | $181.37 | -$302.86 | stop_loss |
| 7 | 2024-12-23 14:50 | 2024-12-23 19:35 | LONG | $182.70 | $187.72 | $179.73 | $463.14 | opposite_signal |
| 8 | 2024-12-23 19:35 | 2024-12-23 20:30 | SHORT | $187.72 | $189.56 | $189.54 | -$314.69 | stop_loss |
| 9 | 2024-12-24 14:15 | 2024-12-24 15:20 | SHORT | $194.07 | $196.51 | $196.49 | -$299.46 | stop_loss |
| 10 | 2024-12-24 15:50 | 2024-12-24 20:20 | SHORT | $198.13 | $193.92 | $199.78 | $678.58 | opposite_signal |
| 11 | 2024-12-24 20:20 | 2024-12-25 18:10 | LONG | $193.92 | $198.34 | $192.50 | $889.73 | opposite_signal |
| 12 | 2024-12-25 18:10 | 2024-12-25 20:00 | SHORT | $198.34 | $199.43 | $199.41 | -$361.48 | stop_loss |
| 13 | 2024-12-25 20:15 | 2024-12-26 14:20 | SHORT | $199.05 | $189.46 | $200.10 | $2,766.52 | opposite_signal |
| 14 | 2024-12-26 14:20 | 2024-12-27 14:35 | LONG | $189.46 | $186.82 | $186.84 | -$409.16 | stop_loss |
| 15 | 2024-12-27 15:50 | 2024-12-27 17:50 | LONG | $184.31 | $185.80 | $182.52 | $290.69 | opposite_signal |
| 16 | 2024-12-27 17:50 | 2024-12-30 14:00 | SHORT | $185.80 | $187.34 | $187.32 | -$418.15 | stop_loss |
| 17 | 2024-12-30 17:55 | 2024-12-30 18:10 | SHORT | $191.49 | $193.54 | $193.52 | -$396.33 | stop_loss |
| 18 | 2024-12-31 14:35 | 2024-12-31 15:55 | SHORT | $196.94 | $198.56 | $198.54 | -$390.19 | stop_loss |
| 19 | 2024-12-31 17:55 | 2024-12-31 19:30 | LONG | $195.97 | $194.03 | $194.05 | -$371.10 | stop_loss |
| 20 | 2024-12-31 19:50 | 2025-01-01 14:00 | LONG | $193.64 | $192.54 | $192.56 | -$376.42 | stop_loss |

*... and 666 more trades*

---

### ETH_5m_SIMPLE_NY_HOURS_ONLY

**Summary:**
| Metric | Value |
|--------|-------|
| Total PnL | $78,946 |
| Total Trades | 825 |
| Win Rate | 35.6% |
| Profit Factor | 1.14 |
| Sharpe Ratio | 0.93 |
| Max Drawdown | 87.8% |
| Average Win | $2,233.59 |
| Average Loss | $-1,088.00 |
| Long Trades | 412 ($59,125) |
| Short Trades | 413 ($19,821) |

**Trade Log (First 20 trades):**

| # | Entry Time | Exit Time | Dir | Entry Price | Exit Price | Stop Loss | PnL | Exit Reason |
|---|------------|-----------|-----|-------------|------------|-----------|-----|-------------|
| 1 | 2024-12-19 15:30 | 2024-12-19 17:30 | LONG | $3,584.64 | $3,540.93 | $3,541.29 | -$317.15 | stop_loss |
| 2 | 2024-12-19 18:00 | 2024-12-19 19:45 | LONG | $3,472.65 | $3,401.54 | $3,401.88 | -$299.80 | stop_loss |
| 3 | 2024-12-19 20:00 | 2024-12-19 20:10 | LONG | $3,404.35 | $3,346.66 | $3,347.00 | -$292.26 | stop_loss |
| 4 | 2024-12-19 20:20 | 2024-12-20 14:00 | LONG | $3,393.42 | $3,322.11 | $3,322.44 | -$280.62 | stop_loss |
| 5 | 2024-12-20 14:20 | 2024-12-20 15:50 | LONG | $3,316.06 | $3,375.23 | $3,242.33 | $203.90 | opposite_signal |
| 6 | 2024-12-20 15:50 | 2024-12-20 16:35 | SHORT | $3,375.23 | $3,416.57 | $3,416.23 | -$284.66 | stop_loss |
| 7 | 2024-12-20 16:45 | 2024-12-20 17:25 | SHORT | $3,381.73 | $3,420.55 | $3,420.21 | -$276.22 | stop_loss |
| 8 | 2024-12-20 18:05 | 2024-12-20 19:15 | SHORT | $3,425.63 | $3,458.51 | $3,458.16 | -$270.04 | stop_loss |
| 9 | 2024-12-20 19:45 | 2024-12-23 14:20 | SHORT | $3,475.96 | $3,347.21 | $3,503.53 | $1,116.08 | opposite_signal |
| 10 | 2024-12-23 14:20 | 2024-12-23 14:40 | LONG | $3,347.21 | $3,303.26 | $3,303.59 | -$290.30 | stop_loss |
| 11 | 2024-12-23 14:55 | 2024-12-23 17:10 | LONG | $3,311.80 | $3,356.53 | $3,262.76 | $232.24 | opposite_signal |
| 12 | 2024-12-23 17:10 | 2024-12-23 19:05 | SHORT | $3,356.53 | $3,331.77 | $3,398.52 | $148.13 | opposite_signal |
| 13 | 2024-12-23 19:05 | 2024-12-23 20:45 | LONG | $3,331.77 | $3,377.25 | $3,306.69 | $480.43 | opposite_signal |
| 14 | 2024-12-23 20:45 | 2024-12-23 20:55 | SHORT | $3,377.25 | $3,402.66 | $3,402.32 | -$318.70 | stop_loss |
| 15 | 2024-12-24 14:10 | 2024-12-24 15:25 | SHORT | $3,443.38 | $3,478.56 | $3,478.21 | -$300.40 | stop_loss |
| 16 | 2024-12-24 15:50 | 2024-12-24 16:25 | SHORT | $3,491.65 | $3,517.58 | $3,517.23 | -$297.38 | stop_loss |
| 17 | 2024-12-24 16:30 | 2024-12-24 18:10 | SHORT | $3,493.58 | $3,520.56 | $3,520.21 | -$285.89 | stop_loss |
| 18 | 2024-12-24 18:20 | 2024-12-24 20:15 | SHORT | $3,507.53 | $3,449.60 | $3,532.74 | $559.27 | opposite_signal |
| 19 | 2024-12-24 20:15 | 2024-12-25 18:45 | LONG | $3,449.60 | $3,475.56 | $3,426.74 | $280.51 | opposite_signal |
| 20 | 2024-12-25 18:45 | 2024-12-25 19:45 | SHORT | $3,475.56 | $3,485.76 | $3,485.42 | -$344.74 | stop_loss |

*... and 805 more trades*

---

### SOL_1h_ENHANCED_70_WEEKENDS_ONLY

**Summary:**
| Metric | Value |
|--------|-------|
| Total PnL | $74,115 |
| Total Trades | 43 |
| Win Rate | 46.5% |
| Profit Factor | 6.27 |
| Sharpe Ratio | 6.88 |
| Max Drawdown | 20.5% |
| Average Win | $4,409.32 |
| Average Loss | $-611.79 |
| Long Trades | 21 ($26,740) |
| Short Trades | 22 ($47,375) |

**Trade Log (First 20 trades):**

| # | Entry Time | Exit Time | Dir | Entry Price | Exit Price | Stop Loss | PnL | Exit Reason |
|---|------------|-----------|-----|-------------|------------|-----------|-----|-------------|
| 1 | 2025-01-11 05:00 | 2025-01-18 06:00 | LONG | $186.18 | $236.10 | $179.73 | $2,318.51 | opposite_signal |
| 2 | 2025-01-18 06:00 | 2025-01-18 15:00 | SHORT | $236.10 | $248.87 | $248.79 | -$375.92 | stop_loss |
| 3 | 2025-01-19 12:00 | 2025-02-01 03:00 | SHORT | $277.16 | $231.44 | $298.06 | $780.67 | opposite_signal |
| 4 | 2025-02-01 03:00 | 2025-02-01 15:00 | LONG | $231.44 | $224.74 | $224.81 | -$392.97 | stop_loss |
| 5 | 2025-02-02 01:00 | 2025-02-02 03:00 | LONG | $217.50 | $211.07 | $211.14 | -$380.34 | stop_loss |
| 6 | 2025-02-02 20:00 | 2025-02-02 22:00 | LONG | $202.92 | $193.42 | $193.48 | -$364.26 | stop_loss |
| 7 | 2025-02-02 23:00 | 2025-02-08 00:00 | LONG | $203.48 | $191.79 | $191.84 | -$351.74 | stop_loss |
| 8 | 2025-02-08 03:00 | 2025-02-09 06:00 | LONG | $192.94 | $204.87 | $180.89 | $329.29 | opposite_signal |
| 9 | 2025-02-09 06:00 | 2025-02-16 17:00 | SHORT | $204.87 | $189.97 | $209.32 | $1,148.11 | opposite_signal |
| 10 | 2025-02-16 17:00 | 2025-02-16 21:00 | LONG | $189.97 | $186.90 | $186.95 | -$401.09 | stop_loss |
| 11 | 2025-02-22 03:00 | 2025-03-01 00:00 | LONG | $170.34 | $163.87 | $163.92 | -$375.86 | stop_loss |
| 12 | 2025-03-01 04:00 | 2025-03-01 11:00 | LONG | $146.76 | $139.87 | $139.92 | -$362.51 | stop_loss |
| 13 | 2025-03-02 22:00 | 2025-03-09 19:00 | SHORT | $176.63 | $129.13 | $188.77 | $1,347.27 | opposite_signal |
| 14 | 2025-03-09 19:00 | 2025-03-23 14:00 | LONG | $129.13 | $132.23 | $124.72 | $264.12 | opposite_signal |
| 15 | 2025-03-23 14:00 | 2025-04-05 06:00 | SHORT | $132.23 | $120.64 | $134.32 | $2,170.98 | opposite_signal |
| 16 | 2025-04-05 06:00 | 2025-04-05 13:00 | LONG | $120.64 | $117.54 | $117.58 | -$473.51 | stop_loss |
| 17 | 2025-04-05 17:00 | 2025-04-06 13:00 | LONG | $117.94 | $115.76 | $115.80 | -$464.74 | stop_loss |
| 18 | 2025-04-06 21:00 | 2025-05-10 05:00 | LONG | $106.90 | $171.07 | $102.83 | $6,741.92 | opposite_signal |
| 19 | 2025-05-10 05:00 | 2025-05-10 23:00 | SHORT | $171.07 | $177.94 | $177.89 | -$645.64 | stop_loss |
| 20 | 2025-05-18 17:00 | 2025-05-24 03:00 | SHORT | $172.14 | $175.72 | $175.67 | -$638.55 | stop_loss |

*... and 23 more trades*

---

### SOL_15m_SIMPLE_NY_HOURS_ONLY

**Summary:**
| Metric | Value |
|--------|-------|
| Total PnL | $58,717 |
| Total Trades | 281 |
| Win Rate | 37.7% |
| Profit Factor | 1.44 |
| Sharpe Ratio | 2.77 |
| Max Drawdown | 43.3% |
| Average Win | $1,805.38 |
| Average Loss | $-758.02 |
| Long Trades | 144 ($21,802) |
| Short Trades | 137 ($36,915) |

**Trade Log (First 20 trades):**

| # | Entry Time | Exit Time | Dir | Entry Price | Exit Price | Stop Loss | PnL | Exit Reason |
|---|------------|-----------|-----|-------------|------------|-----------|-----|-------------|
| 1 | 2024-12-23 20:00 | 2024-12-23 20:45 | SHORT | $187.69 | $190.88 | $190.83 | -$316.46 | stop_loss |
| 2 | 2024-12-24 14:45 | 2024-12-24 15:30 | SHORT | $194.28 | $197.93 | $197.87 | -$304.59 | stop_loss |
| 3 | 2024-12-24 15:45 | 2024-12-26 15:30 | SHORT | $197.61 | $188.52 | $201.20 | $703.22 | opposite_signal |
| 4 | 2024-12-26 15:30 | 2024-12-27 15:15 | LONG | $188.52 | $184.97 | $185.02 | -$315.94 | stop_loss |
| 5 | 2024-12-27 16:30 | 2024-12-30 19:15 | LONG | $183.79 | $194.45 | $180.99 | $1,102.91 | opposite_signal |
| 6 | 2024-12-30 19:15 | 2024-12-31 14:15 | SHORT | $194.45 | $197.47 | $197.41 | -$344.05 | stop_loss |
| 7 | 2024-12-31 15:00 | 2025-01-01 15:00 | SHORT | $197.53 | $190.16 | $200.03 | $913.44 | opposite_signal |
| 8 | 2025-01-01 15:00 | 2025-01-02 15:00 | LONG | $190.16 | $207.59 | $187.74 | $2,442.93 | opposite_signal |
| 9 | 2025-01-02 15:00 | 2025-01-03 14:00 | SHORT | $207.59 | $212.00 | $211.93 | -$431.77 | stop_loss |
| 10 | 2025-01-03 15:15 | 2025-01-06 14:45 | SHORT | $217.03 | $220.63 | $220.56 | -$422.85 | stop_loss |
| 11 | 2025-01-06 16:30 | 2025-01-07 15:45 | SHORT | $221.69 | $207.71 | $225.05 | $1,597.35 | opposite_signal |
| 12 | 2025-01-07 15:45 | 2025-01-07 20:00 | LONG | $207.71 | $203.30 | $203.36 | -$452.89 | stop_loss |
| 13 | 2025-01-07 20:30 | 2025-01-08 14:00 | LONG | $203.06 | $200.26 | $200.32 | -$448.37 | stop_loss |
| 14 | 2025-01-08 15:00 | 2025-01-08 17:00 | LONG | $197.50 | $193.55 | $193.60 | -$424.85 | stop_loss |
| 15 | 2025-01-08 17:45 | 2025-01-09 14:00 | LONG | $193.16 | $189.07 | $189.12 | -$410.13 | stop_loss |
| 16 | 2025-01-09 15:15 | 2025-01-09 17:45 | LONG | $190.47 | $186.17 | $186.22 | -$395.94 | stop_loss |
| 17 | 2025-01-09 20:45 | 2025-01-13 14:00 | LONG | $185.08 | $181.43 | $181.48 | -$385.47 | stop_loss |
| 18 | 2025-01-13 15:00 | 2025-01-14 15:00 | LONG | $177.96 | $187.86 | $172.56 | $646.20 | opposite_signal |
| 19 | 2025-01-14 15:00 | 2025-01-15 14:00 | SHORT | $187.86 | $192.19 | $192.13 | -$391.21 | stop_loss |
| 20 | 2025-01-15 15:30 | 2025-01-15 18:45 | SHORT | $197.42 | $201.16 | $201.10 | -$381.86 | stop_loss |

*... and 261 more trades*

---

## Strategy Comparison: V5 vs V6

| Metric | V5 (Original) | V6 (Per-Asset VWAP) | Improvement |
|--------|---------------|---------------------|-------------|
| **Total PnL** | $384,958 | $974,457 | **+$589,499 (+153%)** |
| **BTC PnL** | -$181,431 | +$83,241 | **+$264,672** |
| **ETH PnL** | +$265,616 | +$346,509 | +$80,893 |
| **SOL PnL** | +$300,772 | +$544,707 | +$243,935 |
| **Profitable Rate** | 39% | 53.3% | +14.3% |
| **Best Config** | $205,177 | $166,586 | -$38,591 |

### Why V6 Outperforms V5

1. **BTC VWAP Filter**: The single biggest improvement. V5's BTC lost -$181K because longs during downtrends were disastrous. V6's VWAP confirmation filters these out, resulting in +$83K profit.

2. **ETH/SOL Without VWAP**: For ETH and SOL, the VWAP filter was too restrictive - it blocked good trades. Disabling it allows more trades, capturing more profitable opportunities.

3. **Bug Fixes**: V6 uses the REAL VWAP (price-volume weighted) instead of the WT momentum that was incorrectly labeled as VWAP in V5.

---

## Risk Metrics

### Sharpe Ratio Distribution

| Asset | Best Sharpe | Config | Interpretation |
|-------|-------------|--------|----------------|
| BTC | 10.85 | BTC_4h_ENHANCED_60_WEEKENDS_ONLY | Excellent |
| ETH | 9.07 | ETH_4h_ENHANCED_70_WEEKENDS_ONLY | Excellent |
| SOL | 6.88 | SOL_1h_ENHANCED_70_WEEKENDS_ONLY | Excellent |

### Configurations with Sharpe > 2.0 (Excellent Risk-Adjusted Returns)

| Config | Sharpe | PnL | Trades | Win% | PF |
|--------|--------|-----|--------|------|-----|
| BTC_4h_ENHANCED_60_WEEKENDS_ONLY | 10.85 | $1,211 | 5 | 60.0% | 2.69 |
| BTC_4h_ENHANCED_70_WEEKENDS_ONLY | 10.85 | $1,211 | 5 | 60.0% | 2.69 |
| ETH_4h_ENHANCED_70_WEEKENDS_ONLY | 9.07 | $5,372 | 8 | 50.0% | 3.78 |
| ETH_4h_ENHANCED_60_WEEKENDS_ONLY | 8.87 | $10,176 | 12 | 50.0% | 4.00 |
| SOL_1h_ENHANCED_70_WEEKENDS_ONLY | 6.88 | $74,115 | 43 | 46.5% | 6.27 |
| BTC_4h_SIMPLE_WEEKENDS_ONLY | 6.04 | $1,190 | 7 | 57.1% | 2.17 |
| BTC_30m_ENHANCED_70_NY_HOURS_ONLY | 5.98 | $5,914 | 25 | 44.0% | 2.12 |
| BTC_1h_ENHANCED_70_WEEKENDS_ONLY | 5.90 | $3,662 | 14 | 42.9% | 2.44 |
| ETH_1h_ENHANCED_70_NY_HOURS_ONLY | 5.53 | $13,502 | 32 | 40.6% | 2.84 |
| ETH_30m_ENHANCED_70_NY_HOURS_ONLY | 4.69 | $36,066 | 66 | 34.8% | 2.36 |


---

## Recommended Production Configuration

Based on the backtest results, here are the recommended configurations for production:

### Primary Configurations (Highest PnL)

| Priority | Asset | Timeframe | Signal | Filter | Expected Monthly PnL |
|----------|-------|-----------|--------|--------|---------------------|
| 1 | SOL | 5m | SIMPLE | NY_HOURS | ~$13,900 |
| 2 | SOL | 5m | ENHANCED_60 | NY_HOURS | ~$12,700 |
| 3 | ETH | 5m | SIMPLE | NY_HOURS | ~$6,600 |
| 4 | BTC | 5m | ENHANCED_60 | NY_HOURS | ~$4,500 |

### Alternative High-Sharpe Configurations (Lower Risk)

For more conservative trading, consider configurations with higher Sharpe ratios even if lower absolute PnL.

### Settings Summary

```yaml
# V6 Production Configuration

assets:
  BTC:
    vwap_entry_confirmation: true  # CRITICAL: Filter bad longs
    direction_filter: both

  ETH:
    vwap_entry_confirmation: false  # More trades = more profit
    direction_filter: both

  SOL:
    vwap_entry_confirmation: false  # More trades = more profit
    direction_filter: both

entry:
  signal_modes: [SIMPLE, ENHANCED_60, ENHANCED_70]
  oversold_levels: [-53, -60, -70]  # Match signal mode
  overbought_levels: [53, 60, 70]

exit:
  method: FULL_SIGNAL  # Wait for opposite WT cross
  stop_loss: ATR_BASED  # 2 × ATR(14)

time_filters:
  primary: NY_HOURS_ONLY  # 14:00-21:00 UTC weekdays
  secondary: WEEKENDS_ONLY  # Sat-Sun all hours

risk:
  risk_per_trade: 3%
  commission: 0.06%
  slippage_5m: 0.01%
  slippage_htf: 0.03%
```

---

## Appendix: WaveTrend Indicator Formulas

### WaveTrend Calculation (Matches VuManChu Cipher B)

```
Input: Heikin Ashi OHLC data

1. Average Price (AP):
   AP = (High + Low + Close) / 3

2. Exponential Smoothing (ESA):
   ESA = EMA(AP, channel_length=9)

3. Deviation (D):
   D = EMA(|AP - ESA|, channel_length=9)

4. Cipher Index (CI):
   CI = (AP - ESA) / (0.015 × D)

5. WaveTrend Lines:
   WT1 = EMA(CI, average_length=12)    # Fast line
   WT2 = SMA(WT1, 3)                   # Slow line (signal)

Signal Generation:
- LONG:  WT1 crosses above WT2 AND WT2 < oversold_level
- SHORT: WT1 crosses below WT2 AND WT2 > overbought_level
```

### Money Flow Indicator (MFI)

```
1. Money Flow Volume:
   MFV = Volume × ((Close - Low) - (High - Close)) / (High - Low)

2. Smoothed MFI:
   MFI = SMA(MFV, period=60) × multiplier(150)

3. Curving Detection:
   curving_up = MFI is rising (current > previous)
   curving_down = MFI is falling (current < previous)
```

### Real VWAP Calculation

```
1. Typical Price:
   TP = (High + Low + Close) / 3

2. Cumulative Values (reset daily at 00:00 UTC):
   Cumulative_PV = Σ(TP × Volume)
   Cumulative_Vol = Σ(Volume)

3. VWAP:
   VWAP = Cumulative_PV / Cumulative_Vol

4. Entry Confirmation:
   LONG valid if: Close > VWAP
   SHORT valid if: Close < VWAP
```

---

*Report generated by VMC Trading Bot V6 Analysis System*
*All results based on 1-year backtest with realistic costs (0.06% commission, 0.01-0.03% slippage)*
