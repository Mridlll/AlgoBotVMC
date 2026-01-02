---
date: 2026-01-02T19:40:00+05:30
session_name: general
researcher: Claude
git_commit: 6274c5a
branch: v6-client-release
repository: AlgoBotVMC
topic: "VMC Trading Bot V6 Testnet Live Testing"
tags: [trading-bot, testnet, hyperliquid, live-testing]
status: complete
last_updated: 2026-01-02
last_updated_by: Claude
type: implementation_strategy
---

# Handoff: VMC V6 Testnet Live Testing

## Task(s)

| Task | Status |
|------|--------|
| Configure new test wallet | Completed |
| Fix SL/TP price rounding | Completed |
| Add order error handling | Completed |
| Fix check interval to 5m | Completed |
| Verify order placement works | Completed |
| Monitor bot live trading | Partial (testnet latency issues) |

## Critical References

- `D:\Crypto\vmc_trading_bot\thoughts\ledgers\CONTINUITY_CLAUDE-vmc-v6-release.md` - Main ledger
- `D:\Crypto\vmc_trading_bot\src\exchanges\hyperliquid.py` - Exchange integration with fixes
- `D:\Crypto\vmc_trading_bot\config\config_v6_production.yaml` - Production config

## Recent Changes

1. **src/exchanges/hyperliquid.py:438-439** - Added `_round_price()` for stop loss
2. **src/exchanges/hyperliquid.py:467-468** - Added `_round_price()` for take profit
3. **src/exchanges/hyperliquid.py:389-391** - Added error handling for order rejections
4. **config/config_v6_production.yaml:55-56** - Added `timeframe: "5m"` for check interval
5. **README.md** - Updated with testing verification section

## Learnings

### Hyperliquid API Quirks
- `float_to_wire causes rounding` error when prices have too many decimals
- Fix: Use `_round_price()` (5 significant figures) for all prices
- Order response can have `status: 'ok'` but contain `error` in statuses array
- Testnet has stale oracle data causing "Price too far from oracle" errors

### Order Flow
```
place_order() -> check status == 'ok' -> check for 'error' in statuses
                                      -> extract 'resting' or 'filled' oid
```

### Check Interval
- Must use smallest strategy timeframe (5m) for polling
- Exit signals can occur on any candle, not just strategy timeframe
- Config setting: `trading.timeframe: "5m"` controls interval

### Test Order Verified
```python
# Successful test order on testnet:
result = {'status': 'ok', 'response': {'type': 'order', 'data': {
    'statuses': [{'filled': {'totalSz': '0.01', 'avgPx': '3216.99', 'oid': 45956082471}}]
}}}
```

## Post-Mortem

### What Worked
- Test order script (`test_order.py`) for debugging
- `market_open()` with proper size filled correctly
- Price rounding eliminated float_to_wire errors
- Error handling now surfaces order rejections

### What Failed
- Initial orders showed empty order ID (silent rejection)
- Testnet oracle prices stale causing close failures
- Bot hung during candle fetch (testnet latency)

### Key Decisions
- **Decision**: Add `_round_price()` to SL/TP methods
  - Alternatives: Round in caller, use different precision
  - Reason: Centralizes fix, matches main order rounding

- **Decision**: Check for 'error' key in statuses
  - Alternatives: Only check status=='ok'
  - Reason: Hyperliquid returns ok status with error in payload

## Artifacts

- `D:\Crypto\vmc_trading_bot\src\exchanges\hyperliquid.py:389-391,438-439,467-468`
- `D:\Crypto\vmc_trading_bot\config\config_v6_production.yaml:55-56`
- `D:\Crypto\AlgoBotVMC_test\test_order.py` - Test script for order debugging
- `D:\Crypto\vmc_trading_bot\README.md` - Testing verification section

## Action Items & Next Steps

1. **Monitor Testnet**: Let bot run on testnet, check for signals and fills
2. **Fix Position Reconciliation**: Minor error `'Position' object has no attribute 'get'`
3. **Mainnet Deployment**: When testnet verified, switch `testnet: false`
4. **Conservative Start**: Begin mainnet with 1% risk, increase gradually

## Other Notes

### Test Wallet Credentials
```
API Secret: 0x7ddba2c18d88de4280b2b2fccf35804cc526877d06e2d894989e6d9bef2d8741
API Wallet: 0xeb6eb13007a3b98b272771b2b0e9f73ee6b09fe1
Account: 0x8A41f91F21260137Ef17aafde0ac2A13e5036f91
```

### GitHub Commits This Session
```
e5064a5 - Update README: fix max positions, add testing verification
a32ded0 - Fix SL/TP price rounding for Hyperliquid float_to_wire errors
ae9ce41 - Add error handling for Hyperliquid order rejection responses
6274c5a - Add 5m check interval for V6 strategies
```

### Bot Startup Checklist
1. Clone fresh from GitHub: `git clone https://github.com/Mridlll/AlgoBotVMC.git`
2. Checkout branch: `git checkout v6-client-release`
3. Install deps: `pip install -r requirements.txt`
4. Update config with wallet credentials
5. Run: `python run_production.py --config config/config_v6_production.yaml`

### Known Testnet Issues
- Oracle price staleness causes order rejections
- Slow candle data fetch (can take 60+ seconds)
- These should not affect mainnet
