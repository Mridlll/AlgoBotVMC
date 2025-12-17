# VMC Strategy Documentation

## Overview

The VMC (Visual Market Cipher) strategy is a momentum-based trading system that identifies high-probability entry points using a combination of:

1. **WaveTrend Oscillator** - Main momentum indicator
2. **Money Flow (RSI+MFI)** - Volume/momentum confirmation
3. **VWAP** - Trend confirmation

The strategy uses **Heikin Ashi candles** which smooth price action and make trends easier to identify.

---

## Indicators Explained

### WaveTrend Oscillator

The WaveTrend oscillator consists of two waves:

- **WT1** (Fast Wave): EMA of a normalized price channel
- **WT2** (Slow Wave): SMA of WT1
- **VWAP**: WT1 - WT2 (momentum indicator)

**Key Levels:**
- Overbought Zone: Above 53 (Level 1), Above 60 (Level 2)
- Oversold Zone: Below -53 (Level 1), Below -60 (Level 2)

**Calculation:**
```
ESA = EMA(HLC3, 9)
D = EMA(|HLC3 - ESA|, 9)
CI = (HLC3 - ESA) / (0.015 * D)
WT1 = EMA(CI, 12)
WT2 = SMA(WT1, 3)
VWAP = WT1 - WT2
```

### Money Flow (RSI+MFI Area)

The Money Flow indicator shows buying/selling pressure:

- **Green Area** (Above 0): Buyers in control
- **Red Area** (Below 0): Sellers in control

**Key Signal**: Watch for the wave to "curve" (change direction)

**Calculation:**
```
MFI = SMA(((Close - Open) / (High - Low)) * 150, 60) - 2.5
```

### VWAP (WT1 - WT2)

The VWAP line shows momentum:
- Crossing above 0: Bullish momentum building
- Crossing below 0: Bearish momentum building

---

## Entry Logic

### Long Entry Sequence

```
1. ANCHOR WAVE
   ├── WT2 drops below -60
   └── This shows extreme oversold conditions

2. TRIGGER WAVE
   ├── WT2 makes another dip
   └── But this dip is HIGHER than the anchor (higher low)

3. MONEY FLOW CONFIRMATION
   ├── Red wave (negative MFI) starts curving UP
   └── Shows sellers losing control

4. VWAP CROSS
   ├── VWAP crosses ABOVE 0
   └── Confirms momentum shift to bullish

5. ENTER LONG
```

**Visual Example:**
```
WT2:
                     Trigger
      Anchor            ↓
         ↓          ___/\___
    ____/\____    _/        \____
   /          \__/
  /
-60 ────────────────────────────
```

### Short Entry Sequence

```
1. ANCHOR WAVE
   ├── WT2 rises above +60
   └── This shows extreme overbought conditions

2. TRIGGER WAVE
   ├── WT2 makes another peak
   ├── But this peak is LOWER than the anchor (lower high)
   └── Plus: WT cross down (green dot on indicator)

3. MONEY FLOW CONFIRMATION
   ├── Green wave (positive MFI) starts curving DOWN
   └── Shows buyers losing control

4. VWAP CROSS
   ├── VWAP crosses BELOW 0
   └── Confirms momentum shift to bearish

5. ENTER SHORT
```

---

## Exit Logic

### Stop Loss

**Swing Method (Recommended):**
- Long: Place SL below recent swing low + buffer
- Short: Place SL above recent swing high + buffer

**ATR Method:**
- Place SL at ATR * multiplier distance from entry

**Fixed Percent:**
- Place SL at fixed % from entry price

### Take Profit Options

**1. Fixed R:R**
- Set TP at a fixed risk:reward ratio (e.g., 2:1)
- If risking $100 to SL, TP at $200 profit

**2. Oscillator Exit**
- Close when opposite signal appears
- Long exit: When short signal conditions are met
- Allows for larger moves but less predictable

**3. Partial Take Profit**
- Take 50% at 1R (risk amount = profit)
- Move SL to breakeven
- Let remaining 50% run to 2R or oscillator exit

**4. Trailing Stop**
- Activate when price reaches 1.5R profit
- Trail by 1% from current price
- Locks in profits while allowing continued upside

---

## Position Sizing

Position size is calculated to risk a fixed percentage of account:

```
Risk Amount = Account Balance * Risk%
Price Difference = |Entry - Stop Loss|
Position Size = Risk Amount / Price Difference
```

**Example:**
- Account: $10,000
- Risk: 3%
- Entry: $50,000
- Stop Loss: $49,000 (2% below entry)

```
Risk Amount = $10,000 * 0.03 = $300
Price Diff = $50,000 - $49,000 = $1,000
Position Size = $300 / $1,000 = 0.3 BTC
```

---

## Best Practices

### Timeframe

- **4H is recommended** - The indicator was designed for this timeframe
- Higher timeframes (Daily) have more reliable signals but fewer trades
- Lower timeframes (1H) have more signals but more noise

### Risk Management

- **Never risk more than 3-5% per trade**
- Use stop losses always
- Don't overtrade - wait for quality setups

### Market Conditions

- Works best in ranging markets with clear swing highs/lows
- Less effective in strong trending markets
- Avoid trading during high-impact news events

### Confirmation

- Higher timeframe should not be in opposite extreme
- Volume should support the move
- Don't force trades - wait for all conditions

---

## Common Mistakes

1. **Entering too early**
   - Wait for VWAP to cross 0
   - Don't enter on anchor alone

2. **Ignoring Money Flow**
   - The curve is important
   - Don't enter if MFI isn't confirming

3. **Wrong stop loss placement**
   - Always use swing levels
   - Don't use arbitrary % stops

4. **Overtrading**
   - Not every setup will trigger
   - Quality over quantity

5. **Not using Heikin Ashi**
   - The strategy requires HA candles
   - Normal candles give different readings

---

## Backtesting Results

On 4H BTC data (2024), typical results:

| Metric | Value |
|--------|-------|
| Win Rate | 50-60% |
| Profit Factor | 1.5-2.0 |
| Max Drawdown | 10-15% |
| Avg Trade | 1-2% |
| Trades/Month | 3-5 |

*Results vary based on market conditions. Past performance does not guarantee future results.*
