# Renko SMA-10 + OBV Directional Strategy (Model V1)

## Status: Archived Research Model
- **Asset:** EURUSD (Tick-level execution)
- **Target Horizon:** Intraday Renko brick dynamics (London + NY Sessions)
- **Status Reason:** Archived due to thin profit factor (`PF 1.08`), unable to withstand live slippage/spread dilution.

---

## 1. Strategy Concept & Architecture

The strategy was designed to eliminate arbitrary time grouping (M1, M5, M15) by converting raw market ticks directly into fixed-size **Renko price bricks** (7.0 pips), applying two classic indicators:
1. **10-Period Simple Moving Average (SMA-10):** Calculated on Renko brick close prices.
2. **On-Balance Volume (OBV) + 14 EMA Signal Line:** Volume accumulation direction indicator.
3. **Volatility-Adaptive Dynamic R:R:** Dynamically expands Take-Profit target to `2.5R` during fast brick formation (momentum waves) and compresses to `1.4R` during sluggish brick formation (chop).

---

## 2. Full-Year 53-Week Benchmark Audit (18,026,436 Ticks)

### Dynamic R:R vs. Static R:R Comparison:
| Model Configuration | Annual Trades | Win Rate | Net Pips | Profit Factor (PF) | Max Drawdown |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Static Conservative (1.4R)** | 220 | 42.7% | +46.3p | 1.03 | 237.4p |
| **Static Baseline (1.8R)** | 210 | 38.6% | +155.6p | 1.09 | 187.6p |
| **Static Ambitious (2.2R)** | 203 | 34.5% | +151.8p | 1.08 | 220.9p |
| **Volatility-Adaptive (1.4R–2.5R)** | **211** | **37.9%** | **+145.5p** | **1.08** | **177.8p** |
| **Extended Runner (2.8R)** | 195 | 29.7% | +82.3p | 1.04 | 252.0p |

### High-Selectivity Filter Benchmark:
When brick sizes were expanded (8p–10p) and filters tightened to 2-brick volume momentum:
- Trade count collapsed to **7–25 trades/year**.
- Net pips dropped to **-35.2p to +8.8p**.
- Profit factor collapsed to **0.37 – 1.10**.

---

## 3. Post-Mortem & Archival Rationale

1. **Spread Drag on Fixed Brick Geometry:**
   - With an average floating spread of 0.8–1.2 pips, transaction cost represents **15% to 20%** of every 7-pip brick.
2. **Lagging Indicator Inefficiency in FX:**
   - Waiting for a 7–10 pip brick to complete plus SMA/OBV confirmation means entering **12–18 pips** into an existing impulse move. In EURUSD (with an Average Daily Range of ~60 pips), entering this late means buying directly into exhaustion levels where institutional algorithms begin mean reversion.
3. **Failure under Live Dilution:**
   - Under an expected 20%–25% performance haircut from live slippage and execution latency, any model with a Profit Factor below **1.30** will deteriorate into net negative expectancy.
