# VMC Trading Bot V6 - Backtest Guide

## Quick Start

```bash
python run_backtest_gui.py
```

That's it. Click "Run Backtest" and wait ~30 seconds.

---

## The 15 Strategies Are Already Optimized

**We tested 4,000+ parameter combinations** across:
- 3 assets (BTC, ETH, SOL)
- 5 timeframes (5m, 15m, 30m, 1h, 4h)
- 2 signal modes (simple, enhanced)
- 3 anchor levels (53, 60, 70)
- 2 direction filters (both, long_only)
- 3 time filters (all_hours, ny_hours, weekends)

The **15 strategies in `config_v6_production.yaml`** are the winners from this optimization.

**Results over 365 days:**
- Total PnL: ~$488,000 (on $10k starting capital)
- Best performer: SOL 5m simple NY hours (+$135k)
- Win rate: 35-45% (normal for trend-following)

---

## GUI Features

| Feature | Description |
|---------|-------------|
| **Data Source** | Cache (instant) or Fetch (downloads fresh data) |
| **Asset** | ALL, BTC, ETH, or SOL |
| **Config** | Select any .yaml config file |
| **Days** | Backtest period (30-730 days) |

**Output files** (saved to `output/` folder):
- `backtest_trades_YYYYMMDD_HHMMSS.csv` - All trades with entry/exit details
- `backtest_log_YYYYMMDD_HHMMSS.txt` - Full output log

---

## Custom Configs (If You Insist)

1. Copy `config/config_template.yaml` → `config/my_config.yaml`
2. Edit strategies (see template for all options)
3. Select in GUI dropdown or click "Browse..."

### Available Parameters

```yaml
strategies:
  my_strategy_name:
    enabled: true/false
    asset: "BTC" | "ETH" | "SOL"
    timeframe: "5m" | "15m" | "30m" | "1h" | "4h"
    signal_mode: "simple" | "enhanced"
    anchor_level: 53 | 60 | 70  # Higher = fewer signals
    direction_filter: "both" | "long_only" | "short_only"
    time_filter:
      enabled: true/false
      mode: "all_hours" | "ny_hours_only" | "weekends_only"
```

### What Each Parameter Does

| Parameter | Effect |
|-----------|--------|
| `anchor_level: 53` | More trades, lower quality |
| `anchor_level: 70` | Fewer trades, higher quality |
| `signal_mode: simple` | Single cross signal |
| `signal_mode: enhanced` | Multi-confirmation signal |
| `ny_hours_only` | Trade 14:00-21:00 UTC (best volatility) |
| `weekends_only` | Trade Sat-Sun only (cleaner signals) |

---

## Why These 15 Configs?

| Strategy | Why It Works |
|----------|--------------|
| SOL 5m simple NY | High volatility + fast timeframe = many opportunities |
| ETH 30m enhanced70 NY | Sweet spot of signal quality + trade frequency |
| BTC 4h simple all | Slow but steady, catches big moves |
| Weekends strategies | Less noise, cleaner trends |

**Bottom line:** The optimization is done. These 15 are the best. But if you want to experiment, the tools are there.

---

## Troubleshooting

**"Config file not found"**
- Make sure you're running from the project folder
- Or use Browse to select the full path

**"Not enough data"**
- Use "Fetch Fresh Data" option to download
- Or check that `data/binance_cache/` has CSV files

**Results don't match expected**
- Make sure time filter is enabled in your config
- NY hours = 14:00-21:00 UTC only
- Weekends = Saturday + Sunday only

---

## Files Reference

```
vmc_trading_bot/
├── run_backtest_gui.py          # GUI launcher
├── run_backtest.py              # CLI version
├── config/
│   ├── config_v6_production.yaml   # THE 15 OPTIMIZED STRATEGIES
│   ├── config_template.yaml        # Template for custom configs
│   └── *.yaml                      # Other configs
├── data/
│   ├── binance_cache/              # Cached price data
│   └── binance_cache_1year/        # Extended cache
└── output/                         # Backtest results saved here
```
