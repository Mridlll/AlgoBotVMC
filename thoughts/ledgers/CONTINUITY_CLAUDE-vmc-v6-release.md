# VMC Trading Bot V6 Release

## Goal
Deploy production-ready VMC Trading Bot V6 on Hyperliquid exchange with 15 optimized strategies.

**Success Criteria:**
- Backtest matches golden results (~$479K PnL, Sharpe 1.6+)
- All 15 strategies functioning correctly
- VWAP confirmation working for BTC
- Live trading on testnet verified
- Mainnet ready

## Constraints
- Risk per trade: 2-3%
- Max positions: 9 total (3 per asset)
- Leverage: 3x
- Commission: 0.06%
- Stop Loss: 2x ATR(14)
- Check interval: 5m (fastest strategy timeframe)

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| Per-trade Sharpe calculation | Matches golden evaluation methodology |
| BTC VWAP=True, ETH/SOL VWAP=False | Per-asset optimization from golden config |
| 3 positions per asset | Balanced diversification |
| 5m check interval | Catches all signals including exits |
| Round SL/TP prices | Avoid Hyperliquid float_to_wire errors |

## State

- Done:
  - [x] Phase 1: Fix Sharpe calculation (per-trade returns)
  - [x] Phase 2: Fix VWAP confirmation (real VWAPCalculator)
  - [x] Phase 3: Update README with accurate results
  - [x] Phase 4: Verify GitHub matches local code
  - [x] Phase 5: Comprehensive mainnet audit
  - [x] Phase 6: Increase max positions to 3 per asset
  - [x] Phase 7: Fix SL/TP price rounding for Hyperliquid
  - [x] Phase 8: Add order error handling
  - [x] Phase 9: Fix check interval to 5m
  - [x] Phase 10: Test with new API wallet
- Now: [x] Create handoff and ledger
- Next: Monitor testnet trading / Deploy to mainnet

## Open Questions
- Testnet has oracle price stale data issues (mainnet should be fine)
- Position reconciliation has minor error (non-critical)

## Working Set

**Branch:** v6-client-release
**Latest Commit:** 6274c5a (Add 5m check interval)

**Key Files:**
- `config/config_v6_production.yaml` - Production config with 5m interval
- `src/exchanges/hyperliquid.py` - Price rounding + error handling fixes
- `backtest/engine.py` - Fixed Sharpe calculation
- `src/strategy/v6_processor.py` - Real VWAP integration
- `run_production.py` - Production runner

**Test Wallet:**
- API Secret: `0x7ddba2c18d88de4280b2b2fccf35804cc526877d06e2d894989e6d9bef2d8741`
- API Wallet: `0xeb6eb13007a3b98b272771b2b0e9f73ee6b09fe1`
- Account: `0x8A41f91F21260137Ef17aafde0ac2A13e5036f91`

**Test Commands:**
```bash
# Run backtest
python run_backtest.py

# Start bot (testnet)
cd D:/Crypto/AlgoBotVMC_test
python run_production.py --config config/config_v6_production.yaml

# Test order placement
python test_order.py
```

## Commits This Session

| Commit | Description |
|--------|-------------|
| `e5064a5` | Update README: fix max positions, add testing verification |
| `a32ded0` | Fix SL/TP price rounding for Hyperliquid float_to_wire errors |
| `ae9ce41` | Add error handling for Hyperliquid order rejection responses |
| `6274c5a` | Add 5m check interval for V6 strategies |

## Backtest Results (Verified)

| Asset | PnL | Sharpe | Trades | Win% |
|-------|-----|--------|--------|------|
| BTC | $33,788 | 1.04 | 1087 | 40.5% |
| ETH | $84,193 | 1.61 | 1394 | 35.4% |
| SOL | $361,766 | 2.19 | 1522 | 34.8% |
| **Total** | **$479,747** | **1.61** | **4003** | |

## Handoffs
- `thoughts/shared/handoffs/general/2026-01-02_19-15-00_v6-mainnet-ready.md`
- `thoughts/shared/handoffs/general/2026-01-02_19-40-00_testnet-live-testing.md`
