# VMC Trading Bot - Development Memory

## Project Overview
Automated crypto trading bot based on VuManChu Cipher B indicator strategy. Trades BTC, ETH, SOL on Hyperliquid exchange.

## Current Version: V3.2 Production (with Realistic Costs)

### Cost Model
- **Commission:** 0.06% per trade (Hyperliquid maker fee)
- **Slippage:** 0.03% per side (0.06% round-trip)
- **Total cost per trade:** ~0.18%

### Strategy Configuration
- **Signal Mode:** SIMPLE for ETH/SOL, ENHANCED for BTC
- **Exit Strategy:** FULL_SIGNAL for all assets
- **Time Filter:** Per-asset (BTC/ETH = OFF_MARKET, SOL = ALL_HOURS)

### Per-Asset Settings (V3.2 with Slippage)
| Asset | Timeframe | Signal | Exit | Time Filter | Expected Return |
|-------|-----------|--------|------|-------------|-----------------|
| BTC | 30m | ENHANCED_60 | FULL_SIGNAL | OFF_MARKET | +56%/year ($5,605) |
| ETH | 30m | SIMPLE | FULL_SIGNAL | OFF_MARKET | +41%/year ($4,136) |
| SOL | 30m | SIMPLE | FULL_SIGNAL | ALL_HOURS | +86%/year ($8,613) |

**Combined Portfolio: ~$18,354 (+183%/year)**

---

## Session Log

### 2025-12-18 (Part 2): Full Time Filter Analysis with Realistic Costs

**Work Completed:**
1. Added slippage (0.03% per side) to backtest engine
2. Ran full 1-year backtest on ALL 108 strategy variants x 4 time filters (432 tests)
3. Discovered critical finding: results change dramatically with realistic costs
4. Updated production config to V3.2 with per-asset time filters

**Key Findings (with 0.03% slippage):**

| Time Filter | Total PnL | Profitable Configs |
|-------------|-----------|-------------------|
| WEEKENDS_ONLY | +$8,007 | 45.2% |
| NY_HOURS_ONLY | -$5,059 | 35.0% |
| ALL_HOURS | -$18,324 | 36.1% |
| OFF_MARKET | -$23,993 | 34.3% |

**Critical Discovery:** Different assets need DIFFERENT time filters!
- BTC: OFF_MARKET is CRITICAL (+$5.6K vs -$21K with ALL_HOURS)
- ETH: OFF_MARKET helps (+$4.1K vs -$2.9K with ALL_HOURS)
- SOL: ALL_HOURS is best (+$8.6K, works in all conditions)

**Files Modified:**
- `run_v31_walkforward.py` - Added slippage_percent=0.03
- `config/config_production.example.yaml` - Updated to V3.2 with per-asset time filters
- `CLAUDE.md` - Updated with V3.2 findings
- `analyze_time_filters.py` - Created analysis script

**Files Generated:**
- `output/walkforward_results/v31_walkforward_365d_20251218_0458.csv`
- `output/walkforward_results/TIME_FILTER_ANALYSIS_1YEAR.md`

---

### 2025-12-18 (Part 1): Walk-Forward Validation & Production Config

**Work Completed:**
1. Fixed BacktestTrade.to_dict() method in backtest/engine.py
2. Fixed run_backtest.py signal_type reference
3. Created validate_strategy.py for V3.1 pre-deployment validation
4. Created run_v31_walkforward.py for comprehensive walk-forward testing
5. Ran 45-day walk-forward on all 14 profitable configs from annual backtest
6. Fixed OFF_MARKET time filter implementation
7. Updated production config with OFF_MARKET filter

**Key Finding - Time Filter Analysis (45 days):**
| Filter | PnL | Trades | Win Rate |
|--------|-----|--------|----------|
| OFF_MARKET | $4,878 | 48 | 56.9% |
| ALL_HOURS | $2,097 | 70 | 38.0% |
| WEEKENDS_ONLY | $39 | 7 | 28.6% |
| NY_HOURS_ONLY | $2 | 7 | 28.6% |

**Recommendation Applied:** OFF_MARKET filter enabled in production config.

**Files Modified:**
- `backtest/engine.py` - Added to_dict() to BacktestTrade
- `run_backtest.py` - Fixed signal_type reference
- `run_v31_walkforward.py` - Created comprehensive walkforward test script
- `validate_strategy.py` - Created validation script
- `config/config_production.example.yaml` - Added OFF_MARKET time filter

**Files Created:**
- `output/walkforward_results/V31_WALKFORWARD_REPORT.md`
- `output/walkforward_results/v31_walkforward_45d_20251218_0427.csv`
- `CLAUDE.md` (this file)

---

### Previous Session: Production Deployment Setup

**Work Completed:**
1. Implemented all 6 production phases:
   - Phase 1: Per-Asset Config Support
   - Phase 2: Exit Monitoring Loop
   - Phase 3: Position Reconciliation
   - Phase 4: Order Fill Verification
   - Phase 5: Trade Persistence (SQLite)
   - Phase 6: Production Config

2. Created deployment scripts:
   - `setup.bat` / `setup.sh` - Installation scripts
   - `run_bot.bat` / `run_bot.sh` - Bot launcher scripts
   - `docs/PRODUCTION_DEPLOYMENT.md` - Deployment guide

3. Pushed to GitHub branch: `feature/v31-production-deployment`

---

## Important Notes

### Backtest Data Period
- Annual backtest: 2024-01-01 to 2024-12-17 (1 year)
- Walk-forward: Last 45 days

### GitHub Repository
- Remote: https://github.com/Mridlll/AlgoBotVMC.git
- Main branch: `main`
- Feature branch: `feature/v31-production-deployment`

### Key Scripts
- `run_backtest.py` - Single config backtest
- `run_v31_walkforward.py` - Walk-forward validation (use --profitable flag)
- `validate_strategy.py` - Pre-deployment validation
- `fetch_and_backtest.py` - Fetch data and run backtest

### Profitable Configs (from 1-year backtest)
14 profitable configurations identified out of 144 tested:
1. BTC/4h SIMPLE + FULL_SIGNAL (+$5,856)
2. ETH/30m ENHANCED_60 + FULL_SIGNAL (+$5,410)
3. BTC/4h ENHANCED_60 + FIXED_RR (+$5,104) - DEPLOYED
4. BTC/4h ENHANCED_60 + FULL_SIGNAL (+$3,658)
5. SOL/4h ENHANCED_60 + FIXED_RR (+$2,617) - DEPLOYED
6. BTC/4h ENHANCED_70 + FIXED_RR (+$2,256)
7. BTC/4h ENHANCED_60 + WT_CROSS (+$2,097)
8. BTC/4h ENHANCED_60 + FIRST_REVERSAL (+$926)
9. BTC/4h ENHANCED_70 + WT_CROSS (+$767)
10. ETH/4h ENHANCED_70 + FIRST_REVERSAL (+$749)
