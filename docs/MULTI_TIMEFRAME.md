# Multi-Timeframe (MTF) Scalping Mode

## Overview

The VMC Trading Bot supports multi-timeframe analysis for scalping strategies. This mode uses:

- **Entry Timeframes** (fast): 3m, 5m, 10m, 15m - for detecting VMC entry signals
- **Bias Timeframes** (HTF): 4h, 8h, 12h, 1D - for determining trend direction

The key principle: **Only execute entry signals that align with the higher timeframe bias.**

---

## How It Works

### 1. Entry Signal Detection

Entry timeframes are scanned for the standard VMC signal sequence:
1. Anchor wave (WT2 < -60 for longs, > 60 for shorts)
2. Trigger wave (subsequent smaller dip/peak)
3. MFI confirmation (curve direction)
4. VWAP cross (above 0 for longs, below 0 for shorts)

When a signal is detected on any entry timeframe, it's evaluated against the HTF bias.

### 2. Bias Determination

Bias is calculated from higher timeframes using WT2 position:
- **WT2 < 0**: Bullish zone (upward bias)
- **WT2 > 0**: Bearish zone (downward bias)

#### Bias Hierarchy

Higher timeframes carry more weight:

| Timeframe | Weight |
|-----------|--------|
| 1D        | 4      |
| 12h       | 3      |
| 8h        | 2      |
| 4h        | 1      |

The overall bias is determined by weighted voting:

```python
# Example: 4h=bullish, 8h=bullish, 12h=bearish, 1D=bullish
bullish_score = 1 + 2 + 4 = 7  # 4h + 8h + 1D
bearish_score = 3              # 12h only
total_weight = 10

# Result: BULLISH with 70% confidence
```

### 3. Alignment Check

For a trade to execute:
1. Entry signal must match bias direction (LONG + BULLISH or SHORT + BEARISH)
2. Minimum number of bias timeframes must agree (default: 2)
3. Bias cannot be NEUTRAL (50/50 split = no trades)

---

## Configuration

### Enabling MTF Mode

In your `config.yaml`:

```yaml
trading:
  use_multi_timeframe: true  # Enable MTF mode

  timeframes:
    # Entry timeframes - scan for VMC signals
    entry:
      - "5m"
      - "15m"
      # - "3m"   # Uncomment for aggressive scalping
      # - "10m"  # Custom timeframe

    # Bias timeframes - determine trend direction
    bias:
      - "4h"
      - "12h"
      - "1d"
      # - "8h"   # Add for additional confirmation

    # Alignment settings
    require_bias_alignment: true   # Only trade when aligned
    min_bias_timeframes_aligned: 2 # Require 2+ HTF to agree
    entry_on_any_timeframe: true   # Signal on ANY entry TF triggers

    # Bias weights
    bias_weights:
      "1d": 4
      "12h": 3
      "8h": 2
      "4h": 1
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `use_multi_timeframe` | `false` | Enable MTF mode |
| `entry` | `["5m", "15m"]` | Entry signal timeframes |
| `bias` | `["4h", "12h", "1d"]` | Bias determination timeframes |
| `require_bias_alignment` | `true` | Only trade when aligned |
| `min_bias_timeframes_aligned` | `2` | Minimum agreeing bias TFs |
| `entry_on_any_timeframe` | `true` | `true`=any TF, `false`=all TFs must agree |
| `bias_weights` | See above | Hierarchical weights for bias calc |

---

## Custom Timeframes

Some timeframes are not natively supported by exchanges but are automatically aggregated:

| Custom TF | Base TF | Aggregation |
|-----------|---------|-------------|
| 3m        | 1m      | 3 candles   |
| 10m       | 5m      | 2 candles   |
| 8h        | 4h      | 2 candles   |

The bot automatically handles this aggregation when fetching candles.

---

## Signal Flow

```
1. Fetch candles for ALL timeframes (entry + bias) in parallel
           ↓
2. Calculate indicators on each timeframe
           ↓
3. Scan entry TFs for VMC signals
           ↓
4. If signal found, calculate HTF bias
           ↓
5. Check alignment:
   - Signal direction matches bias?
   - Minimum TFs aligned?
   - Bias is not NEUTRAL?
           ↓
6. If aligned → Execute trade
   If not aligned → Skip (log reason)
```

---

## Example Scenarios

### Scenario 1: Aligned Long

```
Entry TF: 5m signal = LONG (WT2 crossed -60, trigger found, MFI up, VWAP > 0)

Bias TFs:
  4h:  WT2 = -25 → Bullish (weight 1)
  12h: WT2 = -15 → Bullish (weight 3)
  1D:  WT2 = -8  → Bullish (weight 4)

Bias: BULLISH (8/8 = 100% confidence)
Aligned: YES (LONG matches BULLISH)
Result: EXECUTE TRADE
```

### Scenario 2: Misaligned Long

```
Entry TF: 5m signal = LONG

Bias TFs:
  4h:  WT2 = -10 → Bullish (weight 1)
  12h: WT2 = +20 → Bearish (weight 3)
  1D:  WT2 = +5  → Bearish (weight 4)

Bias: BEARISH (7/8 = 87.5% confidence)
Aligned: NO (LONG vs BEARISH)
Result: SKIP TRADE
```

### Scenario 3: Neutral Bias

```
Entry TF: 15m signal = SHORT

Bias TFs:
  4h:  WT2 = +30 → Bearish (weight 1)
  8h:  WT2 = +15 → Bearish (weight 2)
  12h: WT2 = -10 → Bullish (weight 3)
  1D:  WT2 = -5  → Bullish (weight 4)

Bullish: 3 + 4 = 7
Bearish: 1 + 2 = 3
Total: 10

Bias: BULLISH (7/10 = 70%)
Aligned: NO (SHORT vs BULLISH)
Result: SKIP TRADE
```

---

## Performance Considerations

### API Rate Limits

MTF mode makes more API calls per iteration:
- Single TF: 1 candle request per asset
- MTF with 6 TFs: 6 candle requests per asset (fetched in parallel)

The bot uses `asyncio.gather()` to fetch all timeframes in parallel, minimizing latency.

### Custom Timeframe Overhead

Custom timeframes (3m, 10m, 8h) require fetching more base candles for aggregation:
- 3m needs 3x more 1m candles
- 10m needs 2x more 5m candles
- 8h needs 2x more 4h candles

---

## Logging

MTF signals include additional information in logs:

```
2025-01-15 20:00:00 | INFO | MTF Signal: LONG BTC @ 98500.00 [bias=bullish, aligned=True, conf=0.85]
```

Signal metadata includes:
- `mtf_aligned`: Whether signal matches bias
- `bias_direction`: Current HTF bias (bullish/bearish/neutral)
- `alignment_percent`: Percentage of bias TFs agreeing
- `entry_timeframe`: Which entry TF generated the signal

---

## Discord Notifications

MTF signals include bias information in Discord notifications:

```
Signal Detected: LONG BTC @ $98,500.00
Entry TF: 5m
Bias: Bullish (85% alignment)
Timeframe Biases: 4h=bullish, 12h=bullish, 1D=bullish
```

---

## Best Practices

1. **Start Conservative**: Begin with just 2 entry TFs (5m, 15m) and 3 bias TFs (4h, 12h, 1D)

2. **Require Strong Alignment**: Set `min_bias_timeframes_aligned: 2` or higher

3. **Monitor Skip Rate**: If too many signals are being skipped, bias TFs might be too strict

4. **Adjust for Volatility**: During high volatility, consider using fewer entry TFs

5. **Test on Backtest**: Run backtests with MTF enabled to compare performance

---

## Troubleshooting

### No Signals Generated

- Check if `require_bias_alignment: true` is too strict
- Verify bias TFs are returning data
- Look at logs for "SKIP" messages with reasons

### Too Many Signals

- Enable `require_bias_alignment: true`
- Increase `min_bias_timeframes_aligned`
- Remove aggressive entry TFs (3m)

### Missing Timeframe Data

- Check exchange supports the base timeframe
- Custom TFs (3m, 10m, 8h) require base TF data
- Verify API connectivity

---

## Technical Reference

### Key Classes

```python
# src/strategy/multi_timeframe.py

class MultiTimeframeCoordinator:
    """Coordinates signal detection across timeframes."""

class BiasResult:
    """Result of HTF bias calculation."""
    direction: BiasDirection  # BULLISH, BEARISH, NEUTRAL
    confidence: float         # 0-1
    timeframe_biases: Dict[str, str]

class MultiTimeframeSignal:
    """Combined signal from entry + bias TFs."""
    primary_signal: Signal
    bias_result: BiasResult
    is_aligned: bool
    alignment_percent: float
```

### Key Methods

```python
# Calculate bias from HTF indicators
coordinator.calculate_bias(htf_data) -> BiasResult

# Process all timeframes and generate signal
coordinator.process_candles(entry_candles, bias_candles) -> MultiTimeframeSignal

# Check if signal aligns with bias
coordinator._check_alignment(signal_type, bias_result) -> bool
```
