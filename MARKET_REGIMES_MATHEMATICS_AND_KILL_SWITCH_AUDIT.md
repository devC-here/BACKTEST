# Market Regime Mathematics, Empirical Trade Inference & Archetype Kill-Switch Audit

> [!IMPORTANT]
> **Production Fleet Regime Audit & Optimization Blueprint**: This document establishes the deep mathematical framework for **Intraday FX Market Regime Classification** (developed in quantitative consultation with **NEON**) and synthesizes it with an exhaustive empirical audit of all **7,928 authentic historical trades** across the **71 dual-certified strategies** in the production fleet.
> 
> It provides the exact closed-form indicators, the 4 canonical market regimes, and the **Archetype Kill-Switch Matrix** defining precisely where each specific strategy family experiences negative expectancy and **MUST NOT TRADE**.

---

## 1. Executive Summary & Core Empirical Discoveries

Analyzing 7,928 historical backtest executions across 10 currency pairs reveals that strategy drawdowns are not random; they occur when strategies trade outside their intended mathematical regime:

```
=========================================================================================================
  EMPIRICAL FLEET AUDIT: 7,928 TRADES ACROSS 71 CERTIFIED STRATEGIES (W1..300)
=========================================================================================================
  Total Executed Trades Analyzed:         7,928 Trades (100% Causal Subsequent-Tick Execution)
  Total Realized Net PnL:                 +47,214.0 Net Pips (Under ECN Broker Commission & Raw Floating Spread)
  Dominant Alpha Driver:                  Trend & Momentum Pullbacks (+44,807.8 Net Pips across 6,928 trades)
  Macro Retest Edge:                      Highest Win Rate & Trade Expectancy (55.7% WR, +5.58 pips/trade)
=========================================================================================================
```

### Critical Negative-Expectancy Regime Signatures:
1. **The "Micro-Scalp" Immediate Stop-Out Trap ($<30$ Minutes Duration)**:
   - When **Trend & Momentum Pullbacks** enter during high-entropy chop, they are stopped out in $<30$ minutes: **Win Rate 18.1%**, **Profit Factor 0.42**, net loss of **-7,820.3 pips** (-7.22 pips/trade).
   - In contrast, when entered in genuine directional regimes where trades develop past 2 hours, Win Rate surges to **51.2% – 63.6%**, Profit Factor jumps to **2.07 – 3.56**, generating **+42,383.7 net pips**.
2. **The New York Afternoon Breakout Trap (16:00 – 21:00 UTC)**:
   - **Volatility Breakouts & Expansions** entering in the NY afternoon suffered a **6.25% Win Rate** and **Profit Factor 0.10** (-10.84 pips/trade loss), because volume collapses after the London close, causing immediate false-breakout snapbacks.
3. **The Wednesday Mid-Week Macro Shock**:
   - **Mean-Reversion & Pivot Bounces** suffered a **24.1% Win Rate** and **PF 0.58** (-218.8 pips loss) on Wednesdays due to mid-week macro trend continuation and high-impact central bank releases blowing through static support/resistance levels.
4. **London Open Structural Retest Turbulance (06:00 – 12:00 UTC)**:
   - **Macro Structural Retests** had negative expectancy (**PF 0.84**, -26.74 pips) during the chaotic London morning open, whereas during Asian ($PF=3.73$) and London/NY Overlap ($PF=1.74$) they produced consistent profits.

---

## 2. Mathematical Indicator Formulations for Regime Classification

To dynamically detect and filter these hostile market environments in real time, six closed-form quantitative indicators were formulated and implemented in [`tick_engine/quality/market_regime_classifier.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/tick_engine/quality/market_regime_classifier.py):

### A. Kaufman Efficiency Ratio ($\text{KER}$)
Quantifies the signal-to-noise ratio of price trajectory by measuring directional net displacement relative to cumulative price path:
$$\text{KER}_t(n) = \frac{|C_t - C_{t-n}|}{\sum_{i=0}^{n-1} |C_{t-i} - C_{t-i-1}|}, \quad \text{KER} \in [0, 1]$$
- $\text{KER} \ge 0.40$: Strong directional drift (ideal for Trend Pullbacks).
- $\text{KER} < 0.22$: Stationary Gaussian noise or choppy consolidation (ideal for Mean Reversion; fatal for Trend Pullbacks).

### B. Choppiness Index ($\text{CHOP}$)
Measures market fractal dimension and whether price is moving within a continuous trend or oscillating inside a bounded range:
$$\text{CHOP}_t(n) = 100 \cdot \frac{\log_{10}\left( \frac{\sum_{i=0}^{n-1} \text{TR}_{t-i}}{\max_{0 \le i < n}(H_{t-i}) - \min_{0 \le i < n}(L_{t-i})} \right)}{\log_{10}(n)}$$
where $\text{TR}_t = \max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)$.
- $\text{CHOP} < 38.2$: Highly directional trending market.
- $\text{CHOP} > 61.8$: High-entropy consolidation and whip-saw chop (Kill-switch triggered for Trend and Breakout models).

### C. Local Hurst Exponent ($H$) via Variance Ratio Test
Measures memory persistence versus anti-persistence (mean-reversion):
$$\text{VR}(q) = \frac{\text{Var}(r_{t,q})}{q \cdot \text{Var}(r_{t,1})}$$
$$\hat{H}_t = 0.5 + \frac{\ln(\text{VR}(q))}{2 \ln(q)}$$
- $H > 0.55$: Persistent trending memory (past increments predict future direction).
- $H < 0.45$: Anti-persistent mean-reverting memory (stationary oscillations).
- $H \approx 0.50$: Pure Geometric Brownian Motion (untradable random walk).

### D. Relative Volatility Ratio ($\text{RVR}$)
Tracks current short-term volatility relative to medium-term baseline:
$$\text{RVR}_t(m, k) = \frac{\text{ATR}_m(t)}{\text{ATR}_k(t)}$$
- $\text{RVR} < 0.70$: Volatility compression / squeeze (coiled for breakouts).
- $0.75 \le \text{RVR} \le 1.35$: Balanced steady-state volatility.
- $\text{RVR} > 1.40$: High-volatility shock / expansion.

### E. Bollinger Bandwidth Percentile ($\text{BBWP}$)
The percentile ranking of current Bollinger Bandwidth over a 252-bar rolling lookback:
$$\text{BBW}_t = \frac{\text{UpperBand}_t - \text{LowerBand}_t}{\text{SMA}_t(20)}$$
$$\text{BBWP}_t = \frac{1}{W} \sum_{i=0}^{W-1} \mathbb{I}(\text{BBW}_{t-i} \le \text{BBW}_t) \times 100$$
- $\text{BBWP} < 15.0$: Severe volatility compression (Squeeze regime).
- $\text{BBWP} > 80.0$: Volatility exhaustion climax (dangerous for late breakout entries).

### F. Spread-to-Volatility Friction Penalty ($\text{SVP}$)
Quantifies broker friction as a percentage of expected trade room:
$$\text{SVP}_t = \frac{\text{Spread}_t}{\text{ATR}_m(t)}$$
- $\text{SVP} > 0.12$: Spread consumes $>12\%$ of expected move (Kill-switch for scalps and tight fades).

---

## 3. The 4 Canonical FX Market Regimes

Using priority-ordered non-overlapping logical boundaries, the market at bar $t$ is mapped into one of four states:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 CANONICAL FX REGIMES                                  │
├──────────────────────────┬──────────────────────────┬──────────────────────────────────┤
│ Regime Identifier        │ Quantitative Boundaries  │ Favorable Strategy Archetypes    │
├──────────────────────────┼──────────────────────────┼──────────────────────────────────┤
│ 1. Persistent Trend      │ KER ≥ 0.40, CHOP < 38.2, │ Trend & Momentum Pullbacks       │
│    (Strong Direction)    │ H > 0.56, RVR ∈ [1.05,1.8│ Macro Directional Retests        │
├──────────────────────────┼──────────────────────────┼──────────────────────────────────┤
│ 2. Stationary Range      │ KER < 0.22, CHOP > 55.0, │ Mean-Reversion & Pivot Bounces   │
│    (Gaussian Boundary)   │ H < 0.44, BBWP ∈ [20,60] │ Floor Trader Camarilla Fades     │
├──────────────────────────┼──────────────────────────┼──────────────────────────────────┤
│ 3. Volatile Chop         │ KER < 0.20, CHOP > 61.8, │ CASH / NO TRADING                │
│    (High-Entropy Noise)  │ H ∈ [0.46,0.54], RVR>1.35│ (All intraday entries gated)     │
├──────────────────────────┼──────────────────────────┼──────────────────────────────────┤
│ 4. Liquidity Squeeze     │ RVR < 0.70, BBWP < 15.0, │ Volatility Breakout & Expansion  │
│    (Pre-Explosive Coil)  │ CHOP > 50.0, KER ∈ [0.15,│ Asian Box Expansions (disc173)   │
└──────────────────────────┴──────────────────────────┴──────────────────────────────────┘
```

---

## 4. Archetype Kill-Switch Matrix (Where Strategies MUST NOT Trade)

Every strategy in the 71-model fleet evaluates its specific gating predicate $G_k(x_t) \in \{0, 1\}$ before executing an order. If $G_k = 0$, entry is **blocked**:

| Strategy Archetype | Hostile Market Regimes | Mathematical Kill-Switch Condition ($G_k = 0$) | Microstructural Failure Rationale |
| :--- | :--- | :--- | :--- |
| **1. Trend & Momentum Pullbacks**<br>*(EMA Pullback, PSAR, MACD)* | • Regime 2 (Range)<br>• Regime 3 (Chop)<br>• Regime 4 (Squeeze) | $\text{KER}_{20} < 0.32 \;\lor\; \text{CHOP}_{14} > 52.0 \;\lor\; H < 0.52 \;\lor\; \text{SVP} > 0.10$ | Pullbacks fail to resume trend; price enters micro-scalp whipsaw ($PF=0.42$, 18.1% WR). |
| **2. Mean-Reversion & Pivot Bounces**<br>*(Daily/Weekly Pivots, Camarilla)* | • Regime 1 (Trend)<br>• Regime 3 (High-Vol Chop)<br>• Wednesday Macro Trend | $H > 0.54 \;\lor\; \text{KER}_{14} > 0.38 \;\lor\; \text{RVR}_{5,50} > 1.40 \;\lor\; \text{SVP} > 0.12 \;\lor\; \text{Day} = \text{Wed}$ | Order flow blows straight through static pivot walls without absorption ("catching falling knives"). |
| **3. Volatility Breakouts & Expansion**<br>*(Keltner Squeeze, Chaikin Surge)* | • Regime 3 (Chop)<br>• NY Afternoon (16:00-21:00 UTC)<br>• Post-Exhaustion Climax | $\text{CHOP}_{14} > 61.8 \;\lor\; \text{BBWP} > 80.0 \;\lor\; \text{Hour} \ge 16 \;\lor\; \text{SVP} > 0.15$ | Late afternoon entries suffer 6.25% WR and $PF=0.10$ due to liquidity drying up post-London close. |
| **4. Macro Structural Retests**<br>*(H1/H4 Donchian Retests)* | • Regime 3 (Chop)<br>• London Open Noise (06:00-12:00)<br>• Dead Squeeze | $\text{CHOP}_{H4} > 58.0 \;\lor\; \text{RVR}_{H4} < 0.65 \;\lor\; H_{H4} \in [0.47, 0.53] \;\lor\; \text{London Morning}$ | London open volatility spike invalidates old structural levels before establishing clean direction ($PF=0.84$). |
| **5. Session Liquidity Sweeps**<br>*(Asian & London Sweeps)* | • Regime 1 (HTF Strong Trend)<br>• Over-Extended Asian Box | $\text{KER}_{H4} > 0.48 \;\lor\; \frac{\text{Range}_{Asia}}{\text{ATR}_{D1}} > 0.80 \;\lor\; \frac{\text{Range}_{Asia}}{\text{ATR}_{D1}} < 0.20 \;\lor\; \text{SVP} > 0.14$ | In strong HTF trends, the sweep is genuine trend continuation, resulting in massive blowout losses when attempting to fade it. |

---

## 5. Production Implementation & Verification

The complete engine has been implemented and validated in:
- **Source Code**: [`tick_engine/quality/market_regime_classifier.py`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/tick_engine/quality/market_regime_classifier.py)
- **Numba / Vectorized Kernels**: High-speed calculation of True Range, Kaufman Efficiency Ratio, Choppiness Index, Variance Ratio Hurst Exponent, and Bollinger Bandwidth Percentile.
- **Graceful Fallback**: Automatically compiles with Numba JIT on production runners, and seamlessly falls back to pure NumPy if running in environments without Numba.
- **Empirical Dataset**: [`results/regime_performance_empirical_audit.json`](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/results/regime_performance_empirical_audit.json)
