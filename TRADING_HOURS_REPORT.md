# VMC Trading Bot - Trading Hours Analysis Report

**Generated:** 2025-12-17 01:07:15 UTC

**Data Source:** Binance Futures API (Real Historical Data)

---

## 1. Session Definitions

| Session | Description | Time (ET) |
|---------|-------------|-----------|
| **US_HOURS** | US TradFi Market Hours | Mon-Fri 9:30 AM - 4:00 PM ET |
| **OFF_HOURS** | Weekday Off-Market Hours | Mon-Fri outside 9:30-4:00 ET |
| **WEEKEND** | Weekend Trading | Saturday & Sunday |

## 2. Test Configuration

| Parameter | Value |
|-----------|-------|
| Test Period | ~30 Days |
| Initial Balance | $10,000.00 |
| Risk Per Trade | 3% |
| Risk:Reward | 1:2 |
| Assets | BTC, ETH, SOL |
| Timeframes | 5m, 15m, 1h, 4h |

---

## 3. Performance by Session Type

### Overall Session Performance (All Assets/Timeframes)

| Session | Trades | Winners | Win Rate | Net P&L | Avg P&L/Trade |
|---------|--------|---------|----------|---------|---------------|
| US_HOURS | 31 | 10 | 32.3% | $-368.57 | $-11.89 |
| OFF_HOURS | 35 | 12 | 34.3% | $+98.96 | $+2.83 |
| WEEKEND | 33 | 12 | 36.4% | $+901.20 | $+27.31 |

---

## 4. Detailed Results by Asset & Timeframe

### BTC

#### BTC 5m

**Period:** 2025-12-11 to 2025-12-17
**Total Candles:** 1500

| Session | Trades | Win Rate | Net P&L | Profit Factor |
|---------|--------|----------|---------|---------------|
| US_HOURS | 3 | 33.3% | $-1.81 | 1.00 |
| OFF_HOURS | 1 | 0.0% | $-300.00 | N/A |
| WEEKEND | 3 | 66.7% | $+845.79 | 4.00 |

<details>
<summary>View Trade Details (7 trades)</summary>

| # | Date | Time (ET) | Session | Dir | Entry | Exit | P&L | Result |
|---|------|-----------|---------|-----|-------|------|-----|--------|
| 1 | 2025-12-12 | 04:45 | OFF_HOURS | LONG | $92,252.00 | $91,589.75 | $-300.00 | LOSS |
| 2 | 2025-12-12 | 12:40 | US_HOURS | LONG | $90,143.00 | $89,328.12 | $-291.00 | LOSS |
| 3 | 2025-12-14 | 08:30 | WEEKEND | LONG | $89,280.20 | $88,589.92 | $-282.27 | LOSS |
| 4 | 2025-12-14 | 18:40 | WEEKEND | LONG | $88,000.00 | $89,866.44 | $+547.60 | WIN |
| 5 | 2025-12-14 | 22:30 | WEEKEND | SHORT | $89,247.30 | $87,852.53 | $+580.46 | WIN |
| 6 | 2025-12-15 | 11:55 | US_HOURS | LONG | $86,360.60 | $85,570.00 | $-307.64 | LOSS |
| 7 | 2025-12-15 | 13:40 | US_HOURS | LONG | $85,715.70 | $87,851.83 | $+596.83 | WIN |

</details>

#### BTC 15m

**Period:** 2025-12-01 to 2025-12-17
**Total Candles:** 1500

| Session | Trades | Win Rate | Net P&L | Profit Factor |
|---------|--------|----------|---------|---------------|
| US_HOURS | 4 | 0.0% | $-1,145.60 | N/A |
| OFF_HOURS | 3 | 33.3% | $-90.98 | 0.85 |
| WEEKEND | 3 | 33.3% | $+42.72 | 1.08 |

<details>
<summary>View Trade Details (10 trades)</summary>

| # | Date | Time (ET) | Session | Dir | Entry | Exit | P&L | Result |
|---|------|-----------|---------|-----|-------|------|-----|--------|
| 1 | 2025-12-04 | 14:45 | US_HOURS | LONG | $91,941.50 | $90,368.39 | $-300.00 | LOSS |
| 2 | 2025-12-05 | 14:15 | US_HOURS | LONG | $89,388.10 | $88,329.73 | $-291.00 | LOSS |
| 3 | 2025-12-07 | 18:15 | WEEKEND | LONG | $90,009.10 | $93,136.20 | $+564.54 | WIN |
| 4 | 2025-12-10 | 01:30 | OFF_HOURS | LONG | $92,643.90 | $91,888.25 | $-299.21 | LOSS |
| 5 | 2025-12-11 | 04:00 | OFF_HOURS | LONG | $90,236.30 | $89,481.35 | $-290.23 | LOSS |
| 6 | 2025-12-12 | 09:45 | US_HOURS | LONG | $92,393.00 | $91,391.74 | $-281.52 | LOSS |
| 7 | 2025-12-12 | 15:45 | US_HOURS | LONG | $90,160.00 | $89,393.79 | $-273.08 | LOSS |
| 8 | 2025-12-14 | 10:30 | WEEKEND | LONG | $89,024.10 | $88,387.54 | $-264.89 | LOSS |
| 9 | 2025-12-14 | 23:30 | WEEKEND | LONG | $89,632.00 | $88,734.30 | $-256.94 | LOSS |
| 10 | 2025-12-15 | 16:15 | OFF_HOURS | LONG | $86,042.60 | $88,003.30 | $+498.46 | WIN |

</details>

#### BTC 1h

**Period:** 2025-11-17 to 2025-12-17
**Total Candles:** 720

| Session | Trades | Win Rate | Net P&L | Profit Factor |
|---------|--------|----------|---------|---------------|
| US_HOURS | 1 | 100.0% | $+691.34 | 69133.72 |
| OFF_HOURS | 5 | 60.0% | $+1,258.47 | 3.01 |
| WEEKEND | 3 | 66.7% | $+996.63 | 3.96 |

<details>
<summary>View Trade Details (9 trades)</summary>

| # | Date | Time (ET) | Session | Dir | Entry | Exit | P&L | Result |
|---|------|-----------|---------|-----|-------|------|-----|--------|
| 1 | 2025-11-22 | 02:00 | WEEKEND | LONG | $84,537.40 | $86,866.12 | $+600.00 | WIN |
| 2 | 2025-11-24 | 18:00 | OFF_HOURS | SHORT | $88,247.10 | $89,623.19 | $-318.00 | LOSS |
| 3 | 2025-12-01 | 09:00 | OFF_HOURS | LONG | $86,017.30 | $84,276.60 | $-308.46 | LOSS |
| 4 | 2025-12-02 | 05:00 | OFF_HOURS | LONG | $87,241.80 | $89,946.61 | $+598.41 | WIN |
| 5 | 2025-12-03 | 19:00 | OFF_HOURS | SHORT | $93,013.90 | $89,729.85 | $+634.32 | WIN |
| 6 | 2025-12-06 | 05:00 | WEEKEND | LONG | $89,466.00 | $88,720.07 | $-336.19 | LOSS |
| 7 | 2025-12-09 | 02:00 | OFF_HOURS | LONG | $90,449.90 | $93,505.59 | $+652.20 | WIN |
| 8 | 2025-12-11 | 12:00 | US_HOURS | LONG | $89,935.60 | $92,298.80 | $+691.34 | WIN |
| 9 | 2025-12-13 | 23:00 | WEEKEND | SHORT | $90,159.30 | $88,693.50 | $+732.82 | WIN |

</details>

---

### ETH

#### ETH 5m

**Period:** 2025-12-11 to 2025-12-17
**Total Candles:** 1500

| Session | Trades | Win Rate | Net P&L | Profit Factor |
|---------|--------|----------|---------|---------------|
| US_HOURS | 4 | 25.0% | $-312.78 | 0.64 |
| OFF_HOURS | 4 | 50.0% | $+524.79 | 1.88 |
| WEEKEND | 5 | 20.0% | $-614.27 | 0.47 |

<details>
<summary>View Trade Details (13 trades)</summary>

| # | Date | Time (ET) | Session | Dir | Entry | Exit | P&L | Result |
|---|------|-----------|---------|-----|-------|------|-----|--------|
| 1 | 2025-12-12 | 04:45 | OFF_HOURS | LONG | $3,244.55 | $3,214.84 | $-300.00 | LOSS |
| 2 | 2025-12-12 | 10:45 | US_HOURS | LONG | $3,114.80 | $3,068.58 | $-291.00 | LOSS |
| 3 | 2025-12-12 | 12:45 | US_HOURS | LONG | $3,073.21 | $3,129.87 | $+564.54 | WIN |
| 4 | 2025-12-13 | 09:40 | WEEKEND | LONG | $3,106.87 | $3,078.89 | $-299.21 | LOSS |
| 5 | 2025-12-13 | 17:20 | WEEKEND | LONG | $3,106.25 | $3,085.28 | $-290.23 | LOSS |
| 6 | 2025-12-14 | 08:35 | WEEKEND | LONG | $3,088.13 | $3,068.82 | $-281.52 | LOSS |
| 7 | 2025-12-14 | 19:55 | WEEKEND | LONG | $3,073.58 | $3,139.72 | $+546.15 | WIN |
| 8 | 2025-12-14 | 22:25 | WEEKEND | SHORT | $3,115.79 | $3,140.18 | $-289.46 | LOSS |
| 9 | 2025-12-15 | 05:35 | OFF_HOURS | SHORT | $3,156.39 | $3,083.40 | $+561.56 | WIN |
| 10 | 2025-12-15 | 11:55 | US_HOURS | LONG | $2,987.49 | $2,951.49 | $-297.62 | LOSS |
| 11 | 2025-12-15 | 15:30 | US_HOURS | LONG | $2,934.04 | $2,900.60 | $-288.70 | LOSS |
| 12 | 2025-12-16 | 00:35 | OFF_HOURS | LONG | $2,916.72 | $2,956.45 | $+560.07 | WIN |
| 13 | 2025-12-16 | 07:55 | OFF_HOURS | SHORT | $2,950.18 | $2,973.03 | $-296.84 | LOSS |

</details>

#### ETH 15m

**Period:** 2025-12-01 to 2025-12-17
**Total Candles:** 1500

| Session | Trades | Win Rate | Net P&L | Profit Factor |
|---------|--------|----------|---------|---------------|
| US_HOURS | 5 | 20.0% | $-576.38 | 0.51 |
| OFF_HOURS | 6 | 33.3% | $-58.57 | 0.95 |
| WEEKEND | 4 | 50.0% | $+503.35 | 1.76 |

<details>
<summary>View Trade Details (15 trades)</summary>

| # | Date | Time (ET) | Session | Dir | Entry | Exit | P&L | Result |
|---|------|-----------|---------|-----|-------|------|-----|--------|
| 1 | 2025-12-03 | 13:45 | US_HOURS | SHORT | $3,113.76 | $3,156.88 | $-300.00 | LOSS |
| 2 | 2025-12-04 | 10:15 | US_HOURS | LONG | $3,171.94 | $3,122.46 | $-291.00 | LOSS |
| 3 | 2025-12-04 | 17:15 | OFF_HOURS | LONG | $3,139.35 | $3,101.52 | $-282.27 | LOSS |
| 4 | 2025-12-05 | 14:15 | US_HOURS | LONG | $3,024.43 | $2,963.11 | $-273.80 | LOSS |
| 5 | 2025-12-07 | 18:30 | WEEKEND | LONG | $3,038.87 | $3,126.71 | $+531.18 | WIN |
| 6 | 2025-12-09 | 00:45 | OFF_HOURS | LONG | $3,106.95 | $3,163.79 | $+563.05 | WIN |
| 7 | 2025-12-10 | 07:15 | OFF_HOURS | LONG | $3,312.77 | $3,380.15 | $+596.83 | WIN |
| 8 | 2025-12-11 | 04:00 | OFF_HOURS | LONG | $3,199.03 | $3,159.32 | $-316.32 | LOSS |
| 9 | 2025-12-11 | 15:30 | US_HOURS | LONG | $3,214.33 | $3,171.65 | $-306.83 | LOSS |
| 10 | 2025-12-12 | 15:00 | US_HOURS | LONG | $3,067.42 | $3,111.29 | $+595.25 | WIN |
| 11 | 2025-12-13 | 05:45 | WEEKEND | SHORT | $3,126.17 | $3,077.16 | $+630.96 | WIN |
| 12 | 2025-12-14 | 10:45 | WEEKEND | LONG | $3,083.33 | $3,053.98 | $-334.41 | LOSS |
| 13 | 2025-12-14 | 23:45 | WEEKEND | LONG | $3,122.38 | $3,086.57 | $-324.38 | LOSS |
| 14 | 2025-12-15 | 16:15 | OFF_HOURS | LONG | $2,943.34 | $2,900.60 | $-314.65 | LOSS |
| 15 | 2025-12-16 | 03:30 | OFF_HOURS | LONG | $2,934.22 | $2,901.59 | $-305.21 | LOSS |

</details>

#### ETH 1h

**Period:** 2025-11-17 to 2025-12-17
**Total Candles:** 720

| Session | Trades | Win Rate | Net P&L | Profit Factor |
|---------|--------|----------|---------|---------------|
| US_HOURS | 3 | 66.7% | $+756.04 | 3.77 |
| OFF_HOURS | 2 | 0.0% | $-573.80 | N/A |
| WEEKEND | 5 | 0.0% | $-1,376.10 | N/A |

<details>
<summary>View Trade Details (10 trades)</summary>

| # | Date | Time (ET) | Session | Dir | Entry | Exit | P&L | Result |
|---|------|-----------|---------|-----|-------|------|-----|--------|
| 1 | 2025-11-21 | 06:00 | OFF_HOURS | LONG | $2,687.44 | $2,629.44 | $-300.00 | LOSS |
| 2 | 2025-11-22 | 02:00 | WEEKEND | LONG | $2,755.13 | $2,711.61 | $-291.00 | LOSS |
| 3 | 2025-11-23 | 18:00 | WEEKEND | SHORT | $2,800.73 | $2,858.21 | $-282.27 | LOSS |
| 4 | 2025-11-28 | 02:00 | OFF_HOURS | SHORT | $3,000.14 | $3,043.14 | $-273.80 | LOSS |
| 5 | 2025-11-29 | 14:00 | WEEKEND | LONG | $2,987.49 | $2,945.21 | $-265.59 | LOSS |
| 6 | 2025-12-01 | 14:00 | US_HOURS | LONG | $2,739.58 | $2,809.78 | $+515.24 | WIN |
| 7 | 2025-12-03 | 10:00 | US_HOURS | SHORT | $3,084.84 | $3,160.87 | $-273.08 | LOSS |
| 8 | 2025-12-06 | 05:00 | WEEKEND | LONG | $3,029.56 | $2,996.94 | $-264.89 | LOSS |
| 9 | 2025-12-10 | 16:00 | US_HOURS | SHORT | $3,338.54 | $3,088.92 | $+513.88 | WIN |
| 10 | 2025-12-13 | 17:00 | WEEKEND | LONG | $3,112.77 | $3,059.62 | $-272.35 | LOSS |

</details>

---

### SOL

#### SOL 5m

**Period:** 2025-12-11 to 2025-12-17
**Total Candles:** 1500

| Session | Trades | Win Rate | Net P&L | Profit Factor |
|---------|--------|----------|---------|---------------|
| US_HOURS | 4 | 50.0% | $+574.05 | 1.99 |
| OFF_HOURS | 3 | 33.3% | $-33.08 | 0.94 |
| WEEKEND | 6 | 33.3% | $-52.71 | 0.96 |

<details>
<summary>View Trade Details (13 trades)</summary>

| # | Date | Time (ET) | Session | Dir | Entry | Exit | P&L | Result |
|---|------|-----------|---------|-----|-------|------|-----|--------|
| 1 | 2025-12-12 | 05:50 | OFF_HOURS | LONG | $138.31 | $137.22 | $-300.00 | LOSS |
| 2 | 2025-12-12 | 12:50 | US_HOURS | LONG | $131.87 | $130.61 | $-291.00 | LOSS |
| 3 | 2025-12-12 | 17:00 | OFF_HOURS | LONG | $131.99 | $133.97 | $+564.54 | WIN |
| 4 | 2025-12-13 | 08:35 | WEEKEND | LONG | $132.96 | $131.89 | $-299.21 | LOSS |
| 5 | 2025-12-13 | 19:50 | WEEKEND | SHORT | $133.22 | $131.16 | $+580.46 | WIN |
| 6 | 2025-12-14 | 06:10 | WEEKEND | LONG | $131.64 | $130.70 | $-307.64 | LOSS |
| 7 | 2025-12-14 | 08:30 | WEEKEND | LONG | $131.15 | $130.19 | $-298.41 | LOSS |
| 8 | 2025-12-14 | 16:55 | WEEKEND | LONG | $129.81 | $128.91 | $-289.46 | LOSS |
| 9 | 2025-12-14 | 19:55 | WEEKEND | LONG | $130.04 | $132.95 | $+561.56 | WIN |
| 10 | 2025-12-15 | 05:05 | OFF_HOURS | SHORT | $132.39 | $133.62 | $-297.62 | LOSS |
| 11 | 2025-12-15 | 11:50 | US_HOURS | LONG | $126.36 | $124.86 | $-288.70 | LOSS |
| 12 | 2025-12-15 | 15:30 | US_HOURS | LONG | $124.71 | $127.77 | $+560.07 | WIN |
| 13 | 2025-12-16 | 12:50 | US_HOURS | LONG | $127.40 | $129.33 | $+593.67 | WIN |

</details>

#### SOL 15m

**Period:** 2025-12-01 to 2025-12-17
**Total Candles:** 1500

| Session | Trades | Win Rate | Net P&L | Profit Factor |
|---------|--------|----------|---------|---------------|
| US_HOURS | 4 | 25.0% | $-326.21 | 0.64 |
| OFF_HOURS | 8 | 25.0% | $-602.23 | 0.67 |
| WEEKEND | 3 | 33.3% | $-41.04 | 0.93 |

<details>
<summary>View Trade Details (15 trades)</summary>

| # | Date | Time (ET) | Session | Dir | Entry | Exit | P&L | Result |
|---|------|-----------|---------|-----|-------|------|-----|--------|
| 1 | 2025-12-03 | 22:00 | OFF_HOURS | SHORT | $145.13 | $141.81 | $+600.00 | WIN |
| 2 | 2025-12-04 | 10:15 | US_HOURS | LONG | $142.46 | $140.67 | $-318.00 | LOSS |
| 3 | 2025-12-04 | 17:30 | OFF_HOURS | LONG | $139.54 | $137.74 | $-308.46 | LOSS |
| 4 | 2025-12-05 | 14:15 | US_HOURS | LONG | $132.19 | $129.85 | $-299.21 | LOSS |
| 5 | 2025-12-07 | 18:30 | WEEKEND | LONG | $131.13 | $133.85 | $+580.46 | WIN |
| 6 | 2025-12-08 | 22:30 | OFF_HOURS | LONG | $133.40 | $131.74 | $-307.64 | LOSS |
| 7 | 2025-12-09 | 20:15 | OFF_HOURS | LONG | $137.95 | $136.43 | $-298.41 | LOSS |
| 8 | 2025-12-10 | 11:15 | US_HOURS | LONG | $136.81 | $140.59 | $+578.92 | WIN |
| 9 | 2025-12-11 | 04:00 | OFF_HOURS | LONG | $130.83 | $133.67 | $+613.66 | WIN |
| 10 | 2025-12-12 | 19:30 | OFF_HOURS | LONG | $132.52 | $131.45 | $-325.24 | LOSS |
| 11 | 2025-12-14 | 10:45 | WEEKEND | LONG | $130.60 | $129.59 | $-315.48 | LOSS |
| 12 | 2025-12-14 | 14:45 | WEEKEND | LONG | $130.21 | $129.22 | $-306.02 | LOSS |
| 13 | 2025-12-15 | 03:45 | OFF_HOURS | SHORT | $132.32 | $133.32 | $-296.84 | LOSS |
| 14 | 2025-12-15 | 13:15 | US_HOURS | LONG | $124.36 | $123.56 | $-287.93 | LOSS |
| 15 | 2025-12-16 | 06:45 | OFF_HOURS | SHORT | $128.57 | $129.64 | $-279.29 | LOSS |

</details>

#### SOL 1h

**Period:** 2025-11-17 to 2025-12-17
**Total Candles:** 720

| Session | Trades | Win Rate | Net P&L | Profit Factor |
|---------|--------|----------|---------|---------------|
| US_HOURS | 3 | 33.3% | $-27.21 | 0.96 |
| OFF_HOURS | 3 | 33.3% | $-25.64 | 0.96 |
| WEEKEND | 1 | 100.0% | $+596.83 | 59682.90 |

<details>
<summary>View Trade Details (7 trades)</summary>

| # | Date | Time (ET) | Session | Dir | Entry | Exit | P&L | Result |
|---|------|-----------|---------|-----|-------|------|-----|--------|
| 1 | 2025-11-21 | 19:00 | OFF_HOURS | LONG | $128.40 | $136.45 | $+600.00 | WIN |
| 2 | 2025-11-28 | 18:00 | OFF_HOURS | LONG | $137.28 | $135.19 | $-318.00 | LOSS |
| 3 | 2025-12-01 | 13:00 | US_HOURS | LONG | $124.47 | $128.54 | $+616.92 | WIN |
| 4 | 2025-12-03 | 14:00 | US_HOURS | SHORT | $141.23 | $144.19 | $-326.97 | LOSS |
| 5 | 2025-12-05 | 14:00 | US_HOURS | LONG | $132.40 | $129.85 | $-317.16 | LOSS |
| 6 | 2025-12-11 | 21:00 | OFF_HOURS | LONG | $137.15 | $134.63 | $-307.64 | LOSS |
| 7 | 2025-12-13 | 07:00 | WEEKEND | SHORT | $132.89 | $128.93 | $+596.83 | WIN |

</details>

---

## 5. Best Performing Configurations

### Top Profitable Configurations

| Rank | Config | Session | Trades | Win Rate | Net P&L | PF |
|------|--------|---------|--------|----------|---------|-----|
| 1 | BTC 1h | OFF_HOURS | 5 | 60.0% | $+1,258.47 | 3.01 |
| 2 | BTC 1h | WEEKEND | 3 | 66.7% | $+996.63 | 3.96 |
| 3 | BTC 5m | WEEKEND | 3 | 66.7% | $+845.79 | 4.00 |
| 4 | ETH 1h | US_HOURS | 3 | 66.7% | $+756.04 | 3.77 |
| 5 | BTC 1h | US_HOURS | 1 | 100.0% | $+691.34 | 69133.72 |
| 6 | SOL 1h | WEEKEND | 1 | 100.0% | $+596.83 | 59682.90 |
| 7 | SOL 5m | US_HOURS | 4 | 50.0% | $+574.05 | 1.99 |
| 8 | ETH 5m | OFF_HOURS | 4 | 50.0% | $+524.79 | 1.88 |
| 9 | ETH 15m | WEEKEND | 4 | 50.0% | $+503.35 | 1.76 |
| 10 | BTC 15m | WEEKEND | 3 | 33.3% | $+42.72 | 1.08 |

## 6. Key Insights

- **Best Overall Session:** WEEKEND with $+901.20 total P&L
- **Off Hours outperform US Hours:** 34.3% vs 32.3% win rate
- **Weekend Performance:** 33 trades, 36.4% win rate, $+901.20

---

## Disclaimer

*This analysis uses real historical data from Binance Futures API. Trading hours are based on US Eastern Time. Past performance does not guarantee future results.*

---

*Report generated by VMC Trading Bot v1.0*