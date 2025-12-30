# VMC Trading Bot V6 - Architecture Documentation

## Overview

V6 implements 5 major features to align with the client's flowchart-based trading strategy:

1. **Real VWAP Indicator** - True Volume Weighted Average Price with daily reset
2. **Heikin Ashi Enforcement** - All indicator calculations use HA candles
3. **Divergence Detection** - Price vs indicator divergence detection
4. **Multi-TF Divergence Scanner** - Scan 8 standard timeframes with weighting
5. **Partial Exit System** - Scale out 50% at 1R, 50% at 2R

---

## Signal Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VMC V6 SIGNAL FLOW                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐                                                           │
│  │  RAW OHLCV   │                                                           │
│  │  (8 TFs)     │                                                           │
│  └──────┬───────┘                                                           │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────┐     ┌─────────────────────────────────────────────────┐  │
│  │ HEIKIN ASHI  │────▶│              INDICATOR LAYER                    │  │
│  │  CONVERTER   │     │  ┌─────────┐ ┌─────────┐ ┌─────────┐           │  │
│  └──────────────┘     │  │  VWAP   │ │WaveTrend│ │   MFI   │           │  │
│                       │  │ (Real!) │ │ (WT1/2) │ │         │           │  │
│                       │  └────┬────┘ └────┬────┘ └────┬────┘           │  │
│                       └───────┼──────────┼──────────┼──────────────────┘  │
│                               │          │          │                     │
│         ┌─────────────────────┴──────────┴──────────┘                     │
│         ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    DIVERGENCE DETECTOR                                │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │  │
│  │  │ Find Swing   │  │   Compare    │  │   Classify   │                │  │
│  │  │ Highs/Lows   │─▶│  Price vs    │─▶│  Divergence  │                │  │
│  │  │ (Price+Ind)  │  │  Indicator   │  │    Type      │                │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                │  │
│  └───────────────────────────┬──────────────────────────────────────────┘  │
│                              │                                              │
│         ┌────────────────────┴────────────────────┐                        │
│         ▼                                         ▼                        │
│  ┌──────────────────────┐              ┌──────────────────────┐            │
│  │  MTF DIVERGENCE      │              │   HTF CONFIRMATION   │            │
│  │  SCANNER             │◀────────────▶│   (VWAP/MFI Curve)   │            │
│  │                      │              │                      │            │
│  │  Scan: 5m→1D         │              │  Check: 4h,8h,12h,1D │            │
│  │  Weight by TF        │              │  Curving Up/Down?    │            │
│  └──────────┬───────────┘              └──────────────────────┘            │
│             │                                                               │
│             ▼                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      SIGNAL GENERATOR                                 │  │
│  │                                                                       │  │
│  │  IF divergence_found AND htf_confirms:                               │  │
│  │      signal = LONG if bullish_divergence else SHORT                  │  │
│  │      confidence = tf_weight * htf_confirmation_strength              │  │
│  │                                                                       │  │
│  └───────────────────────────┬──────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    PARTIAL EXIT SYSTEM                                │  │
│  │                                                                       │  │
│  │  Entry: 100% position                                                │  │
│  │    ├─▶ Exit 50% at 1R target                                        │  │
│  │    ├─▶ Move SL to breakeven                                         │  │
│  │    └─▶ Exit remaining 50% at 2R or opposite signal                   │  │
│  │                                                                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Real VWAP Indicator (`src/indicators/vwap.py`)

**Purpose:** Calculate true Volume Weighted Average Price with daily reset.

**Key Features:**
- Daily reset at 00:00 UTC (matches TradingView)
- Curving detection (direction change signals)
- Standard deviation bands (±2σ)
- Price vs VWAP calculation

**Algorithm:**
```
typical_price = (high + low + close) / 3
cumulative_pv = cumsum(typical_price * volume)  # Resets daily
cumulative_vol = cumsum(volume)                  # Resets daily
vwap = cumulative_pv / cumulative_vol

curving_up = (vwap_diff > 0) AND (prev_vwap_diff <= 0)
curving_down = (vwap_diff < 0) AND (prev_vwap_diff >= 0)
```

**Data Structure:**
```python
@dataclass
class VWAPResult:
    vwap: pd.Series           # True VWAP values
    price_vs_vwap: pd.Series  # Close - VWAP
    curving_up: pd.Series     # Direction changing up
    curving_down: pd.Series   # Direction changing down
    bands_upper: pd.Series    # VWAP + 2*std
    bands_lower: pd.Series    # VWAP - 2*std
```

### 2. Heikin Ashi Enforcement

**Current Implementation:**
- **Indicators (WT, MFI):** Calculated on Heikin Ashi data ✓
- **VWAP:** Uses raw OHLCV (correct - needs real volume data)
- **Stop Loss:** Uses raw prices (realistic execution)
- **Fills:** Use raw prices (realistic execution)

**Locations in `backtest/engine.py`:**
- Line 396: `ha_df = convert_to_heikin_ashi(df)`
- Line 398-400: Indicators calculated on `ha_df`

### 3. Divergence Detection (`src/indicators/divergence.py`)

**Purpose:** Detect divergences between price and oscillator (WT2/MFI).

**Algorithm:**
```
1. Find swing lows/highs in price using local extrema detection
2. Record indicator values at same bar indices
3. Compare consecutive swing points:

   BULLISH REGULAR DIVERGENCE:
   - Price:     Low1 → Low2 (LOWER)
   - Indicator: Low1 → Low2 (HIGHER)
   - Signal: Potential reversal UP

   BEARISH REGULAR DIVERGENCE:
   - Price:     High1 → High2 (HIGHER)
   - Indicator: High1 → High2 (LOWER)
   - Signal: Potential reversal DOWN
```

**Parameters:**
- `lookback`: Bars on each side to confirm swing point (default: 5)
- `min_swing_distance`: Minimum bars between swings (default: 3)
- `max_swing_distance`: Maximum bars between swings (default: 50)

**Data Structure:**
```python
@dataclass
class DivergenceResult:
    bullish_regular: pd.Series   # Boolean mask
    bearish_regular: pd.Series   # Boolean mask
    divergences: List[Divergence]  # Detailed list
    strength: pd.Series          # 0-1 score
```

### 4. MTF Divergence Scanner (`src/strategy/mtf_divergence_scanner.py`)

**Purpose:** Scan multiple timeframes for divergences and aggregate signals.

**Standard Timeframes:**
| TF   | Weight |
|------|--------|
| 5m   | 0.15   |
| 15m  | 0.25   |
| 30m  | 0.35   |
| 1h   | 0.50   |
| 4h   | 0.70   |
| 8h   | 0.80   |
| 12h  | 0.90   |
| 1D   | 1.00   |

**HTF Confirmation:**
- Checks 4h, 8h, 12h, 1D timeframes
- Confirms if VWAP is curving in signal direction
- Or if MFI is curving in signal direction

**Algorithm:**
```
FOR each timeframe:
    1. Convert to Heikin Ashi
    2. Calculate WaveTrend, MFI, VWAP
    3. Detect divergences (price vs WT2)
    4. Record signals with TF weight

FOR each signal:
    1. Check HTF confirmation
    2. Calculate weighted_strength = strength * tf_weight
    3. Aggregate into overall bias

RETURN:
    - All signals found
    - Strongest signal (highest weighted_strength)
    - Overall bias (BULLISH/BEARISH/NEUTRAL)
    - Confidence score
```

**Data Structure:**
```python
@dataclass
class MTFScanResult:
    signals: List[MTFDivergenceSignal]
    strongest_signal: Optional[MTFDivergenceSignal]
    overall_bias: Bias
    confidence: float
    htf_bias: Bias
    bullish_count: int
    bearish_count: int
```

### 5. Partial Exit System (`src/strategy/partial_exits.py`)

**Purpose:** Scale out of positions to reduce risk and lock in profits.

**Default Configuration:**
- First partial: 50% at 1R (1:1 risk-reward)
- Second partial: 50% at 2R (2:1 risk-reward)
- Move stop loss to breakeven after first partial

**Exit Flow:**
```
ENTRY: 100% position
  │
  ├─▶ Price hits 1R target
  │     └─▶ Exit 50%
  │     └─▶ Move SL to breakeven
  │
  ├─▶ Price hits 2R target
  │     └─▶ Exit remaining 50%
  │
  ├─▶ Opposite signal appears
  │     └─▶ Exit all remaining
  │
  └─▶ Breakeven SL hit
        └─▶ Exit all remaining (no loss)
```

**Exit Reasons:**
| Reason | Description |
|--------|-------------|
| `partial_1r` | First partial at 1R target |
| `partial_2r` | Second partial at 2R target |
| `full_signal` | Opposite signal exit |
| `stop_loss` | Initial stop loss hit |
| `breakeven` | Breakeven stop hit |
| `vwap_curve` | VWAP curved against position |
| `mfi_curve` | MFI curved against position |

**Data Structures:**
```python
@dataclass
class ExitLeg:
    exit_time: datetime
    exit_price: float
    size_closed: float
    size_percent: float
    pnl: float
    pnl_percent: float
    exit_reason: ExitReason
    remaining_size: float

@dataclass
class PartialExitPosition:
    entry_time: datetime
    entry_price: float
    original_size: float
    is_long: bool
    stop_loss: float
    take_profit_1r: float
    take_profit_2r: float
    remaining_size: float
    current_stop_loss: float
    exit_legs: List[ExitLeg]
    first_partial_done: bool
    position_closed: bool
```

---

## File Structure

```
src/
├── indicators/
│   ├── __init__.py          # Updated with new exports
│   ├── heikin_ashi.py       # Existing
│   ├── wavetrend.py         # Existing
│   ├── money_flow.py        # Existing
│   ├── vwap.py              # NEW: Real VWAP
│   └── divergence.py        # NEW: Divergence detection
│
├── strategy/
│   ├── __init__.py          # Updated with new exports
│   ├── signals.py           # Existing
│   ├── risk.py              # Existing
│   ├── mtf_divergence_scanner.py  # NEW: MTF scanner
│   └── partial_exits.py     # NEW: Partial exit system
│
└── backtest/
    └── engine.py            # Updated: VWAPCalculator integrated
```

---

## Usage Examples

### Calculate Real VWAP
```python
from src.indicators import VWAPCalculator

calc = VWAPCalculator()
result = calc.calculate(df)

if result.curving_up.iloc[-1]:
    print("VWAP is curving up - bullish")
```

### Detect Divergences
```python
from src.indicators import DivergenceDetector

detector = DivergenceDetector(lookback=5)
result = detector.detect(price_series, wt2_series)

if result.bullish_regular.iloc[-1]:
    print("Bullish divergence detected!")
```

### MTF Divergence Scan
```python
from src.strategy import MTFDivergenceScanner

scanner = MTFDivergenceScanner()
result = scanner.scan(tf_data_dict)

if result.strongest_signal and result.strongest_signal.htf_confirmed:
    print(f"High confidence {result.overall_bias.value} signal")
```

### Manage Partial Exits
```python
from src.strategy import PartialExitManager

manager = PartialExitManager()
position = manager.create_position(
    entry_time=now, entry_price=100, size=1.0,
    is_long=True, stop_loss=95
)

# On each bar:
exits = manager.check_exits(position, price, high, low, time)
for exit in exits:
    print(f"Closed {exit.size_closed} at {exit.exit_price}")
```

---

## Configuration

### User Preferences (V6)

| Setting | Value |
|---------|-------|
| VWAP Reset | Daily (00:00 UTC) |
| Partial Exits | 50% at 1R, 50% at 2R |
| Divergence Types | Regular only (no hidden) |
| Timeframes | 5m, 15m, 30m, 1h, 4h, 8h, 12h, 1D |
| Heikin Ashi | Required for all indicators |

---

## Integration Notes

### Backtest Engine Changes

The backtest engine (`backtest/engine.py`) has been updated:

1. **Import:** `from indicators import ... VWAPCalculator`
2. **Init:** `self.vwap_calc = VWAPCalculator()`
3. **Calculation:** Added `vwap_result = self.vwap_calc.calculate(df)` after MFI calculation in all `run*` methods

### Testing

All components have been tested:

```bash
# Test VWAP
python -c "from src.indicators import VWAPCalculator; ..."

# Test Divergence
python -c "from src.indicators import DivergenceDetector; ..."

# Test MTF Scanner
python -c "from src.strategy import MTFDivergenceScanner; ..."

# Test Partial Exits
python -c "from src.strategy import PartialExitManager; ..."
```

---

## Future Enhancements

1. **V6.1:** Integrate MTF scanner into live trading bot
2. **V6.2:** Add trailing stop based on VWAP curve
3. **V6.3:** Implement hidden divergence detection (optional)
4. **V6.4:** Add multi-asset correlation filter

---

*Documentation generated for VMC Trading Bot V6*
*Date: December 2025*
