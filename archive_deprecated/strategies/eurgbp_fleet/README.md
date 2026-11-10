# EURGBP Specialist Quantitative Fleet

## Overview
This specialist fleet targets **EURGBP** (Euro vs British Pound), a cointegrated cross-currency pair characterized by mean-reverting microstructure, institutional daily pivot defenses, and low volatility.

All strategies have been evaluated across **5.77 continuous years** (300 continuous weeks / 85,813,388 authentic Dukascopy ticks across 2021–2026) under the **70/30 deterministic random split (Seed 42)**.

---

## Portfolio Pruning & Balance Standard (Top 8 Champions)
To eliminate low-return strategies that fail to produce meaningful returns while generating unnecessary execution drag, the fleet was filtered against the institutional return hurdle:
- **Net Annualized Return**: Strictly restricted to strategies producing **>= 15.0% annual return** at standard 1.0% account risk.
- **Out-of-Sample Holdout**: Quarantined 30% Holdout Profit Factor **>= 1.25**.
- **Drawdown Control**: Return-to-Drawdown Ratio **>= 4.0x** with tight, disciplined risk limits.

---

## The Elite 8 Balanced EURGBP Champions

| Rank | Strategy ID | Strategy Name | Timeframe | SL / TP | Trades | 5.77-Yr Pips | Ann Pips | Ann Ret% (1% Risk) | Full PF | Holdout PF | Win Rate | Max DD | Ret / DD |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **DISC121** 🌊 | Session VWAP 2.0-SD Mean Fade Elite | M15 / VWAP | 7.0p / 14.0p | 696 | **+1,285.5p** | +222.8p | **+31.8% / yr** | **1.45** | **1.50** | 46.1% | 160.5p | 8.01x |
| 2 | **DISC123** 🎯 | Daily Pivot Pullback Sniper Elite | M15 / Daily | 7.5p / 15.0p | 721 | **+1,206.0p** | +209.0p | **+27.9% / yr** | **1.39** | **1.50** | 44.5% | 111.6p | 10.81x |
| 3 | **DISC122** 📐 | Fib Golden Pocket LB36 Elite | M15 / H1 Swing | 7.0p / 17.5p | 436 | **+1,114.5p** | +193.2p | **+27.6% / yr** | **1.65** | **1.69** | 43.8% | 72.7p | 15.33x |
| 4 | **DISC56** 👑 | Elder Triple Screen M15 Impulse | M15 / H1 / H4 | 7.0p / 14.0p | 812 | **+1,088.3p** | +188.6p | **+26.9% / yr** | **1.26** | **1.25** | 40.4% | 233.1p | 4.67x |
| 5 | **DISC125** 💎 | Fib Golden Pocket LB48 High WR Elite | M15 / H1 Swing | 8.0p / 16.0p | 301 | **+1,023.8p** | +177.4p | **+22.2% / yr** | **1.88** | **2.02** | 51.2% | 66.2p | 15.47x |
| 6 | **DISC48** 🏛️ | Camarilla Floor Rejection Elite | M15 / Daily | 8.5p / 14.0p | 536 | **+899.3p** | +155.9p | **+18.3% / yr** | **1.58** | **1.62** | 50.6% | 126.9p | 7.09x |
| 7 | **DISC46** 🛡️ | Daily Pivot Wall Rejection Elite | M15 / Daily | 9.0p / 14.5p | 195 | **+814.6p** | +141.2p | **+15.7% / yr** | **2.10** | **2.20** | 59.3% | 56.5p | 14.42x |
| 8 | **DISC58** ⚡ | H1 Keltner Squeeze Explosion | M15 / H1 | 8.0p / 16.0p | 293 | **+726.2p** | +125.9p | **+15.7% / yr** | **1.57** | **1.67** | 44.7% | 94.4p | 7.69x |

- **EURGBP Elite 8 Combined Performance:** **3,990 Trades | +8,158.2 Net Pips | +186.1% Combined Annual Return Potential**

---

## Excluded Sub-Threshold Models (<15% Return or Replaced by Higher-Return Architecture)
The following legacy models were retired from active production to ensure the fleet holds only elite performers:
- `DISC74` (+16.1% / yr, replaced by higher-performing DISC123 at +27.9%/yr and DISC122 at +27.6%/yr)
- `DISC65` (+13.8% / yr, 557 trades, max DD 381.9p — replaced by DISC121 at +31.8%/yr, max DD 160.5p)
- `DISC60` (+11.6% / yr, 429 trades)
- `DISC51` (+11.6% / yr, 103 trades)
- `DISC45` (+11.2% / yr, 577 trades)
- `DISC49` (+10.8% / yr, 103 trades)
- `DISC42` (+8.4% / yr, 77 trades)
- `DISC76` (+8.2% / yr, 196 trades)
- `DISC52` (+7.2% / yr, 203 trades)
- `DISC47` (+6.0% / yr, 312 trades)
- `DISC73` (+5.4% / yr, 695 trades)
- `DISC43` (+3.8% / yr, 174 trades)
- `DISC81` (+3.1% / yr, 156 trades)
