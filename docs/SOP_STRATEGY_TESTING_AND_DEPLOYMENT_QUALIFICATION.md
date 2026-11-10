# Standard Operating Procedure (SOP): Quantitative Strategy Testing & Live Deployment Qualification

**Document ID:** `SOP-TRD-VAL-2026-003`  
**Supercedes:** All informal backtest scoring criteria  
**Effective Date:** October 2026  
**Applies To:** All Algorithmic Trading Strategies, Model Generations, Machine Learning Engines, and Multi-Asset Fleets  
**Enforcement:** Mandatory pre-requisite prior to committing real capital, live broker execution, or prop firm challenge deployment.

---

## 1. Executive Mandate: Why "Simple Backtest Profit" is an Illusion

In retail and institutional algorithmic trading alike, the single most common cause of catastrophic live failure is **naive optimization on raw net profit and win rate**. 

A strategy that generates `+3,000 pips` over a 5-year backtest may appear exceptional at first glance, but in real market execution, it can easily blow an account or fail a prop firm evaluation within 72 hours due to:
1. **Regime Over-Concentration (One-Hit Wonder):** Generating 85% of its total profit during a single 3-month macro trend (e.g., the 2022 EUR parity crash), while bleeding capital steadily through 2023, 2024, and 2025.
2. **Right-Tail Outlier Dependency (Luck Bias):** Total net profitability relying on 3 to 5 massive news-spike wins. When those 5 trades are omitted, the strategy is deeply negative.
3. **Sequence Risk & Drawdown Clustering:** Suffering 8 to 11 consecutive losses in normal distribution reshuffling, instantly breaching prop firm daily drawdown limits (e.g., 4% to 5%) or trailing drawdown ceilings (e.g., 8% to 10%).
4. **Execution Friction Asphyxiation:** Operating with an average expectancy of only `+0.7 pips/trade`. Real-world spread markups (0.3 to 0.8 pips), broker commissions (0.2 pips round-trip), and execution latency/slippage (0.3 to 0.6 pips) turn a "profitable" backtest into a steady cash drain.
5. **Prolonged Capital Stagnation (Underwater Drag):** Sitting underwater for 9 to 15 months without making a new equity high, destroying capital efficiency.

**The Purpose of this SOP:**  
To establish an uncompromised, 7-Gate Quantitative Qualification Pipeline that evaluates strategy robustness across multi-year annual consistency, empirical Risk of Ruin, Monte Carlo sequence risk, execution friction tolerance, outlier trimming, and quarantined unseen holdout verification.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                   INSTITUTIONAL 7-GATE STRATEGY QUALIFICATION PIPELINE                           │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│  [GATE 1] Statistical Sample Size & Significance (N >= 250 trades, t-stat > 2.50)                │
│       │                                                                                          │
│  [GATE 2] Multi-Year Annual Consistency (2021 to 2026 breakdown, min 4/5 positive years)        │
│       │                                                                                          │
│  [GATE 3] Downside Risk & Capital Preservation (Max DD <= 250p, Ret/DD >= 3.5x, Consec Loss <= 8)│
│       │                                                                                          │
│  [GATE 4] Risk of Ruin (RoR) & Monte Carlo Sequence Stress (5,000 paths, Breach Prob < 0.5%)     │
│       │                                                                                          │
│  [GATE 5] Execution Friction & Slippage Resilience (Expectancy >= 1.5p, Breakeven Slip >= 1.5p)  │
│       │                                                                                          │
│  [GATE 6] Right-Tail Outlier Trimming (Top 25% winners removed -> remains >= breakeven)          │
│       │                                                                                          │
│  [GATE 7] Quarantined 30% Unseen Holdout Audit (Holdout PF >= 1.15, Decay Ratio >= 0.80)        │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                  FINAL QUALIFICATION VERDICT                                     │
│     ★ TIER 1: Approved for Live / Prop Fleet  |  ▲ TIER 2: Conditional  |  ✕ TIER 3: Rejected    │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The 7 Quantitative Qualification Gates

Every strategy candidate must be tested against all 7 gates using the High-Speed Tick Engine (`tick_engine`) across the authentic 5-year tick database (419M+ ticks).

### GATE 1: Statistical Sample Size & Significance
A strategy cannot be evaluated on small sample sizes. A backtest with 40 or 70 trades across 5 years has zero statistical significance ($p > 0.10$).
* **Total Trade Sample ($N$):** Must have $N \ge 250$ completed trades across the 5-year testing period (minimum $\ge 1.0$ trade per active trading week).
* **Statistical Significance ($t$-statistic):** 
  $$t = \frac{\bar{x} - 0}{s / \sqrt{N}} \ge 2.50 \quad (p < 0.01)$$
  Where $\bar{x}$ is mean pips per trade, $s$ is the sample standard deviation of pips, and $N$ is sample size.
* **Anti-Lottery Ticket Rule (Single Trade Impact):** No single winning trade may account for more than **$8.0\%$** of total net lifetime pips. If one trade made +300 pips out of +1,000 total pips, the strategy fails Gate 1.

---

### GATE 2: Multi-Year Annual Consistency (Returns per Year)
Trading edge must persist across shifting macro regimes, central bank cycles, and volatility environments. The strategy must be sliced and evaluated across each calendar year individually:
* **2021:** Post-COVID liquidity recovery; low rates; range-trend transitions.
* **2022:** Historic inflation shock, aggressive Fed/ECB rate hikes (+75 bps), EUR parity breakdown, extreme trend volatility.
* **2023:** Regional US banking crisis (SVB), plateauing rates, choppy mean-reverting ranges.
* **2024:** Global disinflation, central bank pivot transition toward rate cuts.
* **2025–2026:** Modern algorithmic liquidity regime, tight spreads, multi-timeframe liquidity sweeps.

#### Mandatory Gate 2 Rules:
1. **Profitable Years:** At least **4 out of 5 calendar years** must be strictly net positive ($\text{PnL} > 0$).
2. **Worst Year Tolerance:** At most 1 slightly negative year is tolerated, with an absolute drawdown ceiling of **$< 50.0\text{ pips}$** (or $< 2.0\%$ account equity at standard risk). Zero catastrophic losing years permitted.
3. **Annual Profit Distribution:** No single calendar year may account for more than **$45\%$** of total lifetime profit. Edge must be distributed across the timeline.
4. **Minimum Annualized Expectancy:** Strategy must generate a minimum annualized yield of $\ge 100$ net pips/year on majors (EURUSD, USDJPY) or $\ge 70$ net pips/year on crosses (EURGBP).

---

### GATE 3: Downside Risk, Drawdown & Capital Preservation
Capital preservation dictates survivability. Deep drawdowns or prolonged underwater periods lead to psychological abandonment, prop firm breaches, or liquidation.
* **Maximum Historical Drawdown ($DD_{Hist}$):**
  * EURUSD / USDJPY: $\le 250.0\text{ pips}$.
  * EURGBP / Crosses: $\le 180.0\text{ pips}$.
* **Return-to-Drawdown Ratio (Calmar / MAR Ratio):**
  $$\text{Ret/DD} = \frac{\text{Total Net Pips}}{\text{Max Historical Drawdown (pips)}} \ge 3.50\times \quad (\text{Target: } \ge 5.0\times)$$
* **Maximum Consecutive Losses ($L_{consec}$):**
  $$L_{consec} \le 8\text{ consecutive losing trades}$$
  *(Crucial for prop firms: 8 consecutive losses at 0.5% risk = 4.0% drawdown, remaining safely below the 5.0% daily limit).*
* **Maximum Underwater Duration ($T_{underwater}$):**
  $$T_{underwater} \le 90\text{ calendar days}$$
  The strategy must not remain below its previous equity peak for more than 3 consecutive months.

---

### GATE 4: Risk of Ruin (RoR) & Monte Carlo Sequence Stress
Historical equity curves represent only **one single chronological path**. Real markets can easily reshuffle the order of wins and losses. To evaluate path dependency and true account ruin probability, a **5,000-iteration Monte Carlo permutation stress test** is mandatory.

#### A. Monte Carlo Sequence Reshuffling (5,000 Paths without Replacement)
The trade sequence is randomly permuted 5,000 times to simulate alternate market realities:
* **Median Drawdown ($DD_{P50}$):** Typical drawdown expected across random paths.
* **95th Percentile Drawdown ($DD_{P95}$):** Must satisfy $DD_{P95} \le 1.35 \times DD_{Hist}$.
* **99th Percentile Drawdown ($DD_{P99}$):** Must satisfy $DD_{P99} \le 1.60 \times DD_{Hist}$ and $\le 350\text{ pips}$.

#### B. Prop Firm Risk of Ruin (RoR) Calculation
Evaluated under standard prop firm rules ($100,000 account, 0.5% fixed fractional risk per trade, 8.0% max total drawdown ceiling = 16 units of risk; 4.0% max daily drawdown ceiling = 8 units of risk):
* **Analytical Risk of Ruin (Perry / Ralph Vince Formula):**
  $$RoR = \left(\frac{1 - A}{1 + A}\right)^U$$
  Where $A = \frac{\mu}{\sigma}$ is the standardized trade advantage, and $U$ is capital drawdown units ($U = \frac{\text{Max Allowed DD}}{\text{Risk Per Trade}}$).
* **Empirical Monte Carlo Breach Probability:**
  $$\text{Prob}(\text{Peak-to-Trough Drawdown} \ge \text{Prop Limit}) < \mathbf{0.5\%}$$
  A viable deployment model must have a **$< 0.5\%$ (near zero)** probability of ever hitting the maximum drawdown limit over 5,000 simulations.

---

### GATE 5: Execution Friction & Real-World Slippage Resilience
Live trading is subject to real-world friction: bid-ask spread expansion, broker commission fees, and execution slippage during high-impact liquidity events.
* **Base Mathematical Expectancy ($\text{Exp}$):**
  $$\text{Exp} = \frac{\sum \text{Pips}}{N} \ge \begin{cases} \mathbf{+1.50\text{ pips/trade}} & \text{EURUSD, USDJPY} \\ \mathbf{+1.20\text{ pips/trade}} & \text{EURGBP} \end{cases}$$
* **Breakeven Slippage Threshold:** The amount of adverse slippage per trade required to reduce total PnL to 0:
  $$\text{Slippage}_{BE} = \text{Exp} \ge \mathbf{1.50\text{ pips}}$$
* **Adverse Slippage Stress Tests:**
  1. **$+0.5$ Pip Added Friction Test:** Strategy must remain profitable with an adjusted Profit Factor $PF_{0.5p} \ge 1.15$.
  2. **$+1.0$ Pip Added Friction Test:** Strategy must remain net positive ($\text{PnL}_{1.0p} > 0$).

---

### GATE 6: Right-Tail Outlier Trimming (Fragility Stress Test)
A truly robust quantitative edge generates consistent baseline profits across ordinary trading sessions. Fragile or curve-fitted strategies rely almost entirely on extreme right-tail outlier wins (the top 5-10% of trades).

#### The Top 25% Winners Elimination Audit:
1. Sort all winning trades in descending order of pips.
2. **Remove the top 25% largest winning trades entirely**.
3. Retain **100% of all losing trades** and the remaining 75% of winning trades.
4. Recalculate Trimmed PnL, Trimmed Profit Factor ($PF_{Trimmed}$), and Profit Retention Ratio ($PRR$):
   $$PRR = \frac{\text{Trimmed Net Pips}}{\text{Baseline Net Pips}}$$

#### Mandatory Gate 6 Rules:
* **Trimmed Net Pips:** Must remain **$\ge 0$** (or within $-5\%$ of breakeven).
* **Trimmed Profit Factor:** $PF_{Trimmed} \ge \mathbf{1.00}$.
* **Profit Retention Ratio ($PRR$):** Must retain at least **$\ge 25.0\%$** of baseline profit after losing its top quarter of winners.
* **Verdict:** Any strategy that collapses into severe losses when top winners are removed is disqualified as **Fragile / Luck-Dependent**.

---

### GATE 7: Quarantined 30% Unseen Holdout Audit
Under the [Institutional 70/30 Split Standard](file:///d:/PROJECT/TRD/MODELS%20GENERATIONS/BACKTEST/DATASET_70_30_SPLIT_STANDARD.md), all candidate discovery is restricted to the 70% partition. The remaining **30% quarantined holdout weeks (Seed 42 / 275 total weeks)** represent pristine, unseen market data.

#### Mandatory Gate 7 Rules:
1. **Holdout Profitability:** Quarantined Holdout Net Pips must be **$> 0$**.
2. **Holdout Profit Factor:** $PF_{Holdout} \ge \mathbf{1.15}$.
3. **Edge Degradation Ratio ($EDR$):**
   $$EDR = \frac{PF_{Holdout}}{PF_{Discovery}} \ge \mathbf{0.80}$$
   The strategy's Profit Factor on unseen holdout data must not degrade by more than 20% compared to its discovery performance.
4. **Holdout Win Rate Stability:** Holdout Win Rate must remain within $\pm 4.0\%$ of Discovery Win Rate.

---

## 3. Deployment Tier Classification Matrix

Upon completion of the 7-Gate audit, the strategy is assigned an official deployment tier:

| Qualification Tier | Status | Criteria / Requirements | Deployment Action |
|:---|:---:|:---|:---|
| **TIER 1: ELITE** | **APPROVED** | **Passes ALL 7 GATES with zero flags.**<br>• Annual returns: 5/5 or 4/5 positive.<br>• Risk of Ruin: $< 0.1\%$.<br>• Ret/DD: $\ge 4.0\times$.<br>• Outlier Trimming: Profitable.<br>• Holdout EDR: $\ge 0.85$. | **Immediate Live / Prop Allocation.**<br>Assigned full weight in multi-asset prop challenge portfolios and live trading fleets. |
| **TIER 2: RESERVE** | **CONDITIONAL** | **Passes Gates 1, 2, 4, 6, 7.**<br>Minor non-critical flag in Gate 3 or 5 (e.g. Ret/DD between $3.0\times$ and $3.5\times$, or underwater duration 90–110 days). | **Portfolio Reserve & Synergist.**<br>Deployed with 50% risk weighting or utilized strictly as a non-correlated hedging sleeve in multi-strategy baskets. |
| **TIER 3: REJECTED** | **DISQUALIFIED** | **Fails ANY critical gate:**<br>• Multiple losing years.<br>• Risk of Ruin $\ge 0.5\%$.<br>• Consec Losses $> 8$.<br>• Collapses under Outlier Trimming.<br>• Negative Unseen Holdout. | **Archive / Decommission.**<br>Prohibited from receiving live capital. Parameter set flagged as curve-fitted. |

---

## 4. Standard Metric Summary Reference Card

When reviewing or presenting strategy audit scorecards, the following metrics must be reported:

| Metric Category | Symbol / Code | Institutional Benchmark Target | Disqualification Floor |
|:---|:---:|:---:|:---:|
| **Sample Size** | $N$ | $\ge 400$ trades | $< 250$ trades |
| **Statistical Power** | $t$-stat | $> 3.0$ ($p < 0.001$) | $< 2.50$ |
| **Annual Consistency** | $\text{Pos\_Years}$ | 5 out of 5 positive | $\le 3$ positive years |
| **Annual Yield** | $\text{Ann\_Pips}$ | $\ge 250$ pips/year | $< 100$ pips/year |
| **Max Historical DD** | $DD_{Hist}$ | $\le 180$ pips | $> 250$ pips |
| **Return / DD Ratio** | $\text{Ret/DD}$ | $\ge 5.0\times$ | $< 3.5\times$ |
| **Max Consec Losses** | $L_{consec}$ | $\le 6$ trades | $> 8$ trades |
| **Max Underwater Days** | $T_{underwater}$ | $\le 60$ days | $> 90$ days |
| **Monte Carlo 95% DD** | $DD_{P95}$ | $\le 1.25 \times DD_{Hist}$ | $> 1.35 \times DD_{Hist}$ |
| **Prop Risk of Ruin** | $RoR$ | **$0.00\%$** | $\ge 0.50\%$ |
| **Net Expectancy** | $\text{Exp}$ | $\ge 2.20$ pips/trade | $< 1.50$ pips/trade |
| **Breakeven Slippage** | $\text{Slip}_{BE}$ | $\ge 2.20$ pips | $< 1.50$ pips |
| **PF at +0.5p Slip** | $PF_{0.5p}$ | $\ge 1.30$ | $< 1.15$ |
| **Top 25% Trimmed PF** | $PF_{Trim}$ | $\ge 1.15$ | $< 1.00$ (Negative) |
| **Holdout Degradation** | $EDR$ | $\ge 0.90$ | $< 0.80$ |

---

## 5. Automated Execution & Verification Tooling

To ensure that no manual bias, cherry-picking, or spreadsheet errors occur, all strategies must be audited using the automated SOP evaluation harness:

```bash
# Audit a specific strategy from historical master trade logs:
.venv\Scripts\python.exe scratch/run_strategy_sop_evaluation.py --strategy DISC42_H4_M30 --asset EURUSD

# Audit an entire production fleet (EURUSD, USDJPY, or EURGBP):
.venv\Scripts\python.exe scratch/run_strategy_sop_evaluation.py --fleet eurusd_production_fleet

# Audit all active production models across all 3 assets:
.venv\Scripts\python.exe scratch/run_strategy_sop_evaluation.py --all-fleets
```

The script automatically executes:
1. Trade log extraction and 5-year chronological sequencing.
2. Annual PnL, trade count, and win rate calculation (2021–2026).
3. Drawdown, consecutive loss, and underwater duration tracking.
4. 5,000-iteration Monte Carlo permutation stress test.
5. Analytical and empirical Risk of Ruin calculation under prop firm daily/trailing drawdown rules.
6. Multi-tier friction and slippage degradation modeling (+0.3p, +0.6p, +1.0p, +1.5p).
7. Top 25% right-tail winner trimming stress test.
8. Quarantined 30% unseen holdout partition audit (`Seed 42`).
9. Output generation: Console summary, JSON audit logs, and `SOP_QUALIFICATION_SCORECARD.csv`.
