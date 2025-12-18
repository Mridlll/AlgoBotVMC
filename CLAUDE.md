# VMC Trading Bot - Development Memory

## Project Overview
Automated crypto trading bot based on VuManChu Cipher B indicator strategy. Trades BTC, ETH, SOL on Hyperliquid exchange.

## Current Version: V3.1 Production

### Strategy Configuration
- **Signal Mode:** ENHANCED (4-step state machine with anchor levels at ±60)
- **Exit Strategy:** FULL_SIGNAL for ETH, FIXED_RR for BTC/SOL
- **Time Filter:** OFF_MARKET (avoid NY 9:30am-4pm EST) - yields 2.3x better returns

### Per-Asset Settings
| Asset | Timeframe | Signal | Exit | Expected Return |
|-------|-----------|--------|------|-----------------|
| BTC | 4h | ENHANCED_60 | FIXED_RR | +51%/year |
| ETH | 30m | ENHANCED_60 | FULL_SIGNAL | +54%/year |
| SOL | 4h | ENHANCED_60 | FIXED_RR | +26%/year |

---

## Session Log

### 2025-12-18: Walk-Forward Validation & Production Config

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
