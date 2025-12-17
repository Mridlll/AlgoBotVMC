# VMC Strategy Implementation Overview

## For Client Review: Algorithm Interpretation & Decision Flow

---

## 1. VMC Signal Detection Logic

Our implementation follows the Visual Market Cipher (VMC) oscillator strategy as specified. Here's our interpretation:

### Long Entry Sequence

```
Step 1: ANCHOR WAVE
├── Condition: WT2 drops below -60 (extreme oversold)
├── Purpose: Identifies potential bottom/reversal zone
└── Action: Record anchor level, start monitoring for trigger

Step 2: TRIGGER WAVE
├── Condition: WT2 makes a HIGHER LOW (less negative than anchor)
├── Purpose: Confirms momentum shift - sellers exhausting
└── Example: Anchor at -65, Trigger at -45 = valid setup

Step 3: MONEY FLOW CONFIRMATION
├── Condition: MFI was negative, now curving upward
├── Purpose: Buying pressure entering the market
└── Detection: Compare last 3 bars - was falling, now rising

Step 4: VWAP CROSS
├── Condition: VWAP (WT1 - WT2) crosses ABOVE zero
├── Purpose: Final confirmation of bullish momentum
└── Action: EXECUTE LONG ENTRY
```

### Short Entry Sequence

```
Step 1: ANCHOR WAVE
├── Condition: WT2 rises above +60 (extreme overbought)
├── Purpose: Identifies potential top/reversal zone
└── Action: Record anchor level, start monitoring for trigger

Step 2: TRIGGER WAVE
├── Condition: WT2 makes a LOWER HIGH (less positive than anchor)
├── Additional: WT cross down detected (green dot in indicator)
└── Example: Anchor at +68, Trigger at +52 with cross = valid setup

Step 3: MONEY FLOW CONFIRMATION
├── Condition: MFI was positive, now curving downward
├── Purpose: Selling pressure entering the market
└── Detection: Compare last 3 bars - was rising, now falling

Step 4: VWAP CROSS
├── Condition: VWAP (WT1 - WT2) crosses BELOW zero
├── Purpose: Final confirmation of bearish momentum
└── Action: EXECUTE SHORT ENTRY
```

---

## 2. Multi-Timeframe Scalping Enhancement

### Bias Determination (HTF)

We use higher timeframes to determine overall market direction:

| Timeframe | Weight | Purpose |
|-----------|--------|---------|
| 1D | 4 | Primary trend direction |
| 12h | 3 | Medium-term bias |
| 8h | 2 | Intermediate confirmation |
| 4h | 1 | Short-term bias |

**Bias Logic:**
- WT2 < 0 on HTF = **Bullish** (price in oversold zone, likely to rise)
- WT2 > 0 on HTF = **Bearish** (price in overbought zone, likely to fall)

**Decision:**
```
Bullish Score = Sum of weights where WT2 < 0
Bearish Score = Sum of weights where WT2 > 0

If Bullish > Bearish → Overall Bias = BULLISH
If Bearish > Bullish → Overall Bias = BEARISH
If Equal → NEUTRAL (no trades)
```

### Entry Alignment Rule

```
LONG signal on entry TF + BULLISH HTF bias = EXECUTE
LONG signal on entry TF + BEARISH HTF bias = SKIP
SHORT signal on entry TF + BEARISH HTF bias = EXECUTE
SHORT signal on entry TF + BULLISH HTF bias = SKIP
Any signal + NEUTRAL bias = SKIP
```

---

## 3. State Machine Implementation

Our bot maintains a state machine for each asset:

```
           ┌──────────────────────────────────────────────────┐
           │                                                  │
           ▼                                                  │
        [IDLE] ──────────────────────────────────────────────►│
           │                                                  │
           │ WT2 crosses anchor level (-60 or +60)            │
           ▼                                                  │
   [ANCHOR_DETECTED] ────────────────────────────────────────►│
           │                                                  │
           │ Higher low (long) or Lower high + cross (short)  │
           ▼                                                  │
  [TRIGGER_DETECTED] ────────────────────────────────────────►│
           │                                                  │
           │ MFI curving in expected direction                │
           ▼                                                  │
   [AWAITING_VWAP] ──────────────────────────────────────────►│
           │                                                  │
           │ VWAP crosses zero                                │
           ▼                                                  │
    [SIGNAL_READY] ───► Execute Trade ───► Reset to IDLE ────►┘
```

**Timeout:** If trigger not found within 20 bars of anchor, reset to IDLE.

---

## 4. Risk Management

### Position Sizing
```
Risk Amount = Account Balance × Risk Percent (default 3%)
Position Size = Risk Amount / (Entry Price - Stop Loss)
```

### Stop Loss Methods
- **Swing**: Below recent swing low (longs) / above swing high (shorts)
- **ATR**: Entry ± (ATR × multiplier)
- **Fixed**: Entry ± fixed percentage

### Take Profit
- **Fixed R:R**: Default 2:1 reward-to-risk
- **Partial TP**: 50% at 1R, move SL to breakeven, remainder at 2R

---

## 5. Key Implementation Decisions

| Decision | Our Interpretation | Rationale |
|----------|-------------------|-----------|
| Anchor Level | -60 / +60 for WT2 | Matches wt_oversold_2 / wt_overbought_2 from indicator |
| Trigger Definition | Higher low / Lower high | Confirms momentum shift, not just any subsequent wave |
| MFI Curve | 3-bar lookback | Balances responsiveness with noise filtering |
| VWAP Cross | Must cross zero, not just be positive/negative | Confirms actual momentum change |
| HTF Bias | WT2 position only | Simpler than full VMC signal on HTF, faster computation |
| Bias Weights | 1D(4) > 12h(3) > 8h(2) > 4h(1) | Higher TFs more reliable for trend |

---

## 6. Validation Checklist

Please confirm our interpretation matches your expectations:

- [ ] **Anchor Wave**: WT2 crossing -60 (long) or +60 (short) initiates setup
- [ ] **Trigger Wave**: Must be LESS extreme than anchor (higher low / lower high)
- [ ] **Short Trigger**: Requires WT cross down (green dot) in addition to lower high
- [ ] **MFI Confirmation**: Curve direction change, not absolute value
- [ ] **VWAP Cross**: Final entry trigger when crossing zero
- [ ] **HTF Bias**: Determined by WT2 position (<0 = bullish, >0 = bearish)
- [ ] **Alignment**: Only trade when entry direction matches HTF bias
- [ ] **Custom TFs**: 3m, 10m, 8h aggregated from smaller intervals

---

## 7. What Happens When Bot Runs

```
Every [interval] seconds:
  For each asset (BTC, ETH):
    1. Fetch candles for all timeframes (parallel)
    2. Convert to Heikin Ashi
    3. Calculate WaveTrend (WT1, WT2, VWAP)
    4. Calculate Money Flow (MFI)
    5. If MTF mode:
       a. Calculate HTF bias from 4h/8h/12h/1D
       b. Check entry TFs (5m/15m) for VMC signals
       c. If signal found AND aligned with bias → trade
    6. If single TF mode:
       a. Check for VMC signal on configured TF
       b. If signal found → trade
    7. Calculate position size, SL, TP
    8. Execute order on exchange
    9. Send Discord notification
```

---

## 8. System Architecture

### Component Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│                           VMCBot (Orchestrator)                      │
│                         src/core/bot.py                              │
└─────────────────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐  ┌─────────────────────┐  ┌──────────────────┐
│   Exchange    │  │  Strategy Layer     │  │  Notifications   │
│   Layer       │  │                     │  │                  │
├───────────────┤  ├─────────────────────┤  ├──────────────────┤
│ BaseExchange  │  │ SignalDetector      │  │ DiscordNotifier  │
│ Hyperliquid   │  │ MultiTFCoordinator  │  │                  │
│ Bitunix       │  │ RiskManager         │  │                  │
│               │  │ TradeManager        │  │                  │
└───────────────┘  └─────────────────────┘  └──────────────────┘
        │                    │
        ▼                    ▼
┌───────────────┐  ┌─────────────────────┐
│  Indicators   │  │  State Management   │
├───────────────┤  ├─────────────────────┤
│ HeikinAshi    │  │ TradingState        │
│ WaveTrend     │  │ AssetState          │
│ MoneyFlow     │  │ TimeframeState      │
└───────────────┘  └─────────────────────┘
```

### Data Flow Architecture

```
[Exchange API] ──► [Candle Fetcher] ──► [DataFrame]
                         │
                         ▼
              ┌──────────────────────┐
              │   Heikin Ashi        │
              │   Conversion         │
              └──────────────────────┘
                         │
           ┌─────────────┴─────────────┐
           ▼                           ▼
    ┌─────────────┐             ┌─────────────┐
    │  WaveTrend  │             │  MoneyFlow  │
    │  Calculator │             │  Calculator │
    └─────────────┘             └─────────────┘
           │                           │
           └─────────────┬─────────────┘
                         ▼
              ┌──────────────────────┐
              │   Signal Detector    │◄──── State Machine
              │   (per timeframe)    │      (IDLE→ANCHOR→TRIGGER→...)
              └──────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  MTF Coordinator     │◄──── Bias Calculation
              │  (if MTF enabled)    │      (weighted voting)
              └──────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   Risk Manager       │◄──── Position Sizing
              │   (SL/TP calc)       │      Stop Loss / Take Profit
              └──────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   Trade Manager      │──────► [Exchange Order]
              │   (execution)        │
              └──────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   Discord Notifier   │──────► [Webhook]
              └──────────────────────┘
```

### Multi-Timeframe Processing Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Parallel Candle Fetching                      │
│                    asyncio.gather()                              │
├─────────┬─────────┬─────────┬─────────┬─────────┬───────────────┤
│   5m    │   15m   │   4h    │   8h    │   12h   │      1D       │
│ (entry) │ (entry) │ (bias)  │ (bias)  │ (bias)  │    (bias)     │
└────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴───────┬───────┘
     │         │         │         │         │            │
     ▼         ▼         ▼         ▼         ▼            ▼
┌─────────────────┐  ┌──────────────────────────────────────────┐
│ Entry TF Signal │  │         Bias TF Indicators               │
│ Detection       │  │         (WT2 position only)              │
└────────┬────────┘  └──────────────────┬───────────────────────┘
         │                              │
         │         ┌────────────────────┘
         ▼         ▼
    ┌─────────────────────────────────┐
    │      Alignment Check            │
    │  Signal Direction == Bias?      │
    │  Min TFs Aligned?               │
    └─────────────────────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
    [EXECUTE]           [SKIP + LOG]
```

---

## 9. Edge Cases & How We Handle Them

### Signal Detection Edge Cases

| Edge Case | Scenario | Our Handling |
|-----------|----------|--------------|
| **Stale Anchor** | Anchor detected but no trigger for 20+ bars | Reset to IDLE, anchor expires |
| **Multiple Anchors** | New anchor while waiting for trigger | Replace old anchor with new one |
| **Rapid Signals** | Signal on multiple entry TFs simultaneously | Take first signal, ignore others |
| **Missing Candles** | API returns incomplete data | Skip asset for this iteration, log warning |
| **Flat Market** | WT2 stays in neutral zone (-53 to +53) | No anchors detected, no signals generated |

### Multi-Timeframe Edge Cases

| Edge Case | Scenario | Our Handling |
|-----------|----------|--------------|
| **Partial HTF Data** | Some bias TFs unavailable | Calculate bias with available TFs only |
| **All HTF Missing** | No bias TF data returned | Skip signal even if entry detected |
| **Exact 50/50 Split** | Equal bullish/bearish scores | Bias = NEUTRAL, no trades |
| **Custom TF Aggregation** | Not enough base candles for 3m/10m/8h | Return empty, skip that timeframe |
| **Rapid Bias Flip** | HTF bias changes between checks | Use latest bias at signal time |

### Order Execution Edge Cases

| Edge Case | Scenario | Our Handling |
|-----------|----------|--------------|
| **Max Positions** | Already at max_positions limit | Skip new signal, log "max positions reached" |
| **Existing Position** | Already have position in this asset | Skip if max_positions_per_asset reached |
| **Order Rejected** | Exchange rejects order | Log error, notify via Discord, continue |
| **Insufficient Balance** | Not enough margin | Skip trade, log warning |
| **Price Slippage** | Market moved during execution | Use market order, accept slippage |

### Indicator Edge Cases

| Edge Case | Scenario | Our Handling |
|-----------|----------|--------------|
| **Division by Zero** | D (deviation) = 0 in WaveTrend | Add small epsilon (0.0001) to prevent |
| **NaN Values** | Insufficient history for EMA | Require minimum 100 candles before trading |
| **Extreme Values** | WT2 beyond ±100 | Still valid, just very extreme signal |

---

## 10. Conflict Resolution

### Conflict: Long and Short Setup Simultaneously

```
Scenario: Both long and short state machines are active
Example:  Long anchor at -65, Short anchor at +62 (volatile market)

Resolution:
├── Both state machines run independently
├── First to reach SIGNAL_READY wins
├── Upon signal execution, the OTHER state machine is NOT reset
└── Allows for quick reversal if market turns
```

### Conflict: Multiple Entry Timeframes Signal

```
Scenario: 5m signals LONG, 15m signals SHORT
Mode: entry_on_any_timeframe = true

Resolution:
├── Process timeframes in configured order
├── First valid signal is used
├── Check alignment with HTF bias
├── If 5m LONG aligns with BULLISH bias → execute LONG
└── 15m SHORT is ignored (already have signal this iteration)
```

### Conflict: Entry Signal vs HTF Bias Disagreement

```
Scenario: Strong LONG signal on 5m, but HTF bias is BEARISH

Resolution (require_bias_alignment = true):
├── Signal is SKIPPED
├── Log: "Signal skipped: LONG on 5m conflicts with BEARISH bias"
├── State machine resets for that asset
└── Wait for aligned opportunity

Resolution (require_bias_alignment = false):
├── Signal is EXECUTED anyway
├── Log includes warning about bias conflict
└── Higher risk trade (documented for user)
```

### Conflict: Bias Timeframes Disagree

```
Scenario: 4h=bullish, 8h=bearish, 12h=bullish, 1D=bearish

Calculation:
├── Bullish: 4h(1) + 12h(3) = 4
├── Bearish: 8h(2) + 1D(4) = 6
├── Total: 10
└── Result: BEARISH with 60% confidence

Resolution:
├── If min_bias_timeframes_aligned = 2 → Check: 2 bearish TFs ✓
├── Bias is BEARISH
└── Only SHORT signals will execute
```

### Conflict: Signal During Active Trade

```
Scenario: Already in LONG BTC position, new LONG signal appears

Resolution:
├── Check max_positions_per_asset (default: 1)
├── If at limit → Skip new signal
├── Log: "Skipping signal: max positions for BTC reached"
└── Existing position continues with original SL/TP

Note: We do NOT add to positions or adjust existing trades
```

### Conflict: Webhook vs Local Signal

```
Scenario: Webhook mode enabled, but local calculation also running

Resolution:
├── If webhook.enabled = true AND signal received via webhook
│   └── Webhook signal takes priority, skip local calculation
├── If webhook.enabled = true AND no webhook received
│   └── Fall back to local calculation
└── Prevents duplicate signals from both sources
```

---

## 11. Failure Modes & Recovery

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Exchange disconnect | API timeout/error | Auto-reconnect with exponential backoff |
| Invalid API credentials | 401/403 response | Log error, stop bot, notify Discord |
| Rate limiting | 429 response | Pause requests, respect retry-after header |
| Network outage | Connection timeout | Retry up to 3 times, then skip iteration |
| State corruption | Invalid state values | Reset to IDLE, log warning |
| Indicator NaN | isnan() check | Skip bar, wait for valid data |

---

*Document Version: 1.1 | Implementation Date: December 2025*
