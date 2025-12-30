# VMC Trading Bot V6 - Changelog

## Version 6.0.0 (December 2025)

### Major Changes

#### 1. Per-Asset VWAP Optimization
The biggest change in V6 is the discovery that VWAP entry confirmation works differently for each asset:

| Asset | VWAP Setting | Impact |
|-------|--------------|--------|
| BTC | ENABLED | Fixed -$181K to +$99K (+$280K improvement) |
| ETH | DISABLED | More trades = more profit |
| SOL | DISABLED | More trades = more profit |

#### 2. Critical Bug Fix: VWAP Calculation
**Before (V5 - BROKEN):**
```python
vwap = wt_result.vwap.iloc[i]  # This was WT1-WT2 momentum, NOT real VWAP!
```

**After (V6 - FIXED):**
```python
vwap = vwap_result.vwap.iloc[i]  # Actual price-volume weighted average
```

This bug meant V5 was comparing price against WaveTrend momentum difference instead of actual VWAP. All VWAP-based signals were incorrect.

#### 3. BTC Direction Filter Changed
- **V5**: BTC SHORT_ONLY (avoided longs completely)
- **V6**: BTC BOTH directions with VWAP confirmation

The VWAP filter now properly screens out bad long entries during downtrends, making BTC longs profitable.

### Performance Comparison

| Metric | V5 | V6 | Improvement |
|--------|-----|-----|-------------|
| Total PnL | $384,958 | $974,457 | +$589,499 (+153%) |
| BTC PnL | -$181,431 | +$83,241 | +$264,672 |
| ETH PnL | +$265,616 | +$346,509 | +$80,893 |
| SOL PnL | +$300,772 | +$544,707 | +$243,935 |
| Profitable Rate | 39% | 53.3% | +14.3% |
| Average Sharpe | 0.31 | 2.79 | +2.48 |

### New Files

| File | Description |
|------|-------------|
| `config/config_v6_production.yaml` | V6 production config with 15 strategies |
| `run_v6_fixed_evaluation.py` | V6 backtest evaluation script |
| `V6_RELEASE/` | Complete V6 documentation and results |

### Configuration Changes

#### New Per-Asset Settings
```yaml
asset_settings:
  BTC:
    vwap_entry_confirmation: true   # ENABLED for BTC
    direction_filter: "both"
  ETH:
    vwap_entry_confirmation: false  # DISABLED for ETH
    direction_filter: "both"
  SOL:
    vwap_entry_confirmation: false  # DISABLED for SOL
    direction_filter: "both"
```

#### VWAP Entry Logic
```yaml
vwap:
  enabled: true
  reset_time: "00:00"  # Daily reset at midnight UTC
  # Entry rules (when VWAP enabled):
  # - LONG: Only enter if price > VWAP
  # - SHORT: Only enter if price < VWAP
```

### Top 15 Production Strategies (V6)

| # | Asset | Config | PnL | Sharpe |
|---|-------|--------|-----|--------|
| 1 | SOL | 5m_SIMPLE_NY | $166,586 | 1.26 |
| 2 | SOL | 5m_ENH60_NY | $152,788 | 1.39 |
| 3 | ETH | 5m_SIMPLE_NY | $78,946 | 0.93 |
| 4 | SOL | 1h_ENH70_WKND | $74,115 | 6.88 |
| 5 | SOL | 15m_SIMPLE_NY | $58,717 | 2.77 |
| 6 | BTC | 5m_ENH60_NY | $54,339 | 2.82 |
| 7 | ETH | 30m_SIMPLE_WKND | $44,912 | 2.02 |
| 8 | ETH | 15m_ENH70_WKND | $40,687 | 2.14 |
| 9 | ETH | 5m_ENH60_NY | $36,729 | 1.09 |
| 10 | ETH | 30m_ENH70_NY | $36,066 | 4.69 |
| 11 | SOL | 15m_ENH60_NY | $32,865 | 2.52 |
| 12 | BTC | 5m_SIMPLE_NY | $26,276 | 1.11 |
| 13 | BTC | 4h_ENH60_ALL | $7,284 | 3.54 |
| 14 | BTC | 30m_ENH70_NY | $5,914 | 5.98 |
| 15 | BTC | 4h_SIMPLE_ALL | $5,902 | 2.75 |

### Risk Parameters (Unchanged)

| Parameter | Value |
|-----------|-------|
| Initial Capital | $10,000 |
| Risk Per Trade | 3% (backtest) / 2% (production) |
| Stop Loss | 2 x ATR(14) |
| Exit | Opposite WaveTrend signal |
| Commission | 0.06% |
| Slippage (5m) | 0.01% |
| Slippage (HTF) | 0.03% |

### Breaking Changes

1. **Config Structure**: `use_vwap_confirmation` is now per-asset instead of global
2. **BTC Strategy**: No longer SHORT_ONLY, now uses VWAP filter for both directions

### Migration from V5

1. Update config to `config_v6_production.yaml`
2. Or manually add `asset_settings` section to existing config
3. Re-run backtest to verify performance

### Known Limitations

1. Backtest uses 3% risk; recommend 2% for production
2. Signal overlap not accounted in combined PnL
3. Walk-forward shows ~50% OOS degradation expected

---

*V6 Released: December 2025*
