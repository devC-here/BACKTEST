# Market Regime & Archetype Kill-Switch System: Master Agent Guide (v5.0)

> **Strictly Causal 5-State Non-USDJPY Regime System**  
> Calibrated strictly on **Discovery Weeks 1..210** causal M15 bars across all **9 non-USDJPY pairs** (`AUDCAD, AUDNZD, EURGBP, EURNZD, EURUSD, GBPCHF, GBPJPY, NZDCAD, USDCAD`; USDJPY strictly excluded). Across ~883k Discovery M15 bars, average regime shares are: **Regime 1 (Trend) 16.9%**, **Regime 2 (Calm MR) 16.5%**, **Regime 3 (Volatile Chop) 7.8%**, **Regime 4 (Dead/Wide Spread) 9.6%**, **Regime 5 (Normal Balanced) 49.1%** (Adverse R3+R4 = **13.3%** during active London/NY hours, **17.5%** all hours).

---

## 1. 5-State Canonical Market Regime Taxonomy

| Regime ID | Canonical Name | Causal Mathematical Signature (Closed M15 Bars) | Empirical Share (9-Pair Mean) | Default Policy |
| :---: | :--- | :--- | :---: | :--- |
| **0** | **Warmup / Uninitialized** | History $< 272$ closed M15 bars | — | Fail-safe Warmup Pass |
| **1** | **Strong Clean Trend** | $\text{ER}_{20} \ge 0.26 \land \text{ADX}_{14} \ge 23.0 \land \text{CHOP}_{14} < 51.0$ | ~16.9% | Permitted for Trend, Breakout, Macro, Balanced |
| **2** | **Calm Mean-Reversion** | $\text{ADX}_{14} < 21.0 \land \text{ER}_{20} < 0.20 \land \text{ATR}_{\text{rel}} \le 1.15 \land (H < 0.46 \lor \text{CHOP} \ge 53.0)$ | ~16.5% | Permitted for Mean Reversion, Balanced, Pullback/Macro (unless Discovery proves toxic) |
| **3** | **Volatile / High-ATR Chop** | $(\text{ATR}_{\text{rel}} > 1.15 \lor \text{RVR} > 1.20) \land \text{ER}_{20} < 0.20 \land \text{CHOP}_{14} > 54.0$ | ~7.8% | **BLOCKED** (Extreme high-vol whipsaw tail) |
| **4** | **Dead / Low Liquidity** | $\text{SpreadRatio} > 3.0 \lor \text{SVP}_{14} > 1.20 \lor \text{NYRollover}(17\text{h ET}) \lor (\text{ATR}_{\text{rel}} < 0.68 \land \text{BBWP} < 20.0)$ | ~9.6% | **BLOCKED** (Illiquid rollover / wide spread / dead ATR tail) |
| **5** | **Normal Balanced Market** | All valid closed M15 bars not classified in Regimes 1–4 | ~49.1% | **PERMITTED** (Healthy baseline flow; replaces broken Regime 3 catch-all) |


### Out-of-Sample Holdout Validation (Weeks 211..300)

| Metric | Unfiltered v5.0 Fleet | Regime-Gated v5.0 Fleet | Delta |
| :--- | :---: | :---: | :---: |
| **Active Strategies (Non-USDJPY)** | 4 | 4 | — |
| **Holdout Trades** | 151 | 111 | -40 (26.5% pruned) |
| **Win Rate** | 51.0% | 51.4% | +0.4% |
| **Profit Factor** | 1.68 | 1.59 | -0.09 |
| **Net Pips** | +1,136.2p | +831.7p | -304.5p |

---

## 2. Empirical Non-USDJPY Discovery Baselines (Weeks 1..210)

| Symbol | Discovery Median Spread | Discovery Median ATR(14) | 3.0x Wide Spread Guard |
| :---: | :---: | :---: | :---: |
| **AUDCAD** | 1.9831p | 5.860p | 5.949p |
| **AUDNZD** | 2.0607p | 5.182p | 6.182p |
| **EURGBP** | 0.8739p | 3.994p | 2.622p |
| **EURNZD** | 3.2468p | 11.983p | 9.740p |
| **EURUSD** | 0.2858p | 5.944p | 0.857p |
| **GBPCHF** | 1.9743p | 7.131p | 5.923p |
| **GBPJPY** | 2.0262p | 12.943p | 6.079p |
| **NZDCAD** | 2.3906p | 5.792p | 7.172p |
| **USDCAD** | 1.2142p | 6.792p | 3.643p |

---

## 3. Strictly Causal Closed-Form Formulas (Closed M15 Bars)

All indicators operate strictly causally on closed M15 OHLC bars (`closed='left', label='right'`, using only bars $\le t$, zero `.bfill()`, zero future quantiles):

1. **Wilder ATR & Relative Volatility (`ATR`, `ATR_rel`, `RVR`)**:
   - $\text{TR}_t = \max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)$
   - Wilder RMA: $\text{ATR}_p(t) = \text{ATR}_p(t-1) + \frac{1}{p}(\text{TR}_t - \text{ATR}_p(t-1))$ for $p \in \{5, 14, 50\}$
   - $\text{RVR}_t = \text{ATR}_5(t) / \text{ATR}_{50}(t)$
   - $\text{ATR}_{\text{rel}}(t) = \text{ATR}_{14}(t) / \text{median}(\text{ATR}_{14}[t-287..t])$
2. **Wilder ADX (`ADX_14`)**:
   - Causal $+\text{DM}_t, -\text{DM}_t$ smoothed via 14-bar Wilder RMA $\to +\text{DI}_{14}, -\text{DI}_{14}, \text{DX}_t$, then 14-bar Wilder RMA for $\text{ADX}_{14}(t)$.
3. **Kaufman Efficiency Ratio (`ER` / `KER`)**:
   - $\text{ER}_t(20) = \frac{|C_t - C_{t-20}|}{\sum_{i=0}^{19} |C_{t-i} - C_{t-i-1}|} \in [0, 1]$
4. **Choppiness Index (`CHOP_14`)**:
   - $\text{CHOP}_t(14) = 100 \cdot \frac{\log_{10}\left(\sum_{i=0}^{13} \text{TR}_{t-i} / (\max_{0\le i<14} H_{t-i} - \min_{0\le i<14} L_{t-i})\right)}{\log_{10}(14)}$
5. **Variance Ratio Local Hurst (`H`)**:
   - $\text{VR}_t(5) = \frac{\text{Var}_{100}(r_t(5))}{5 \cdot \text{Var}_{100}(r_t(1))}, \quad H_t = 0.5 + \frac{\ln(\text{VR}_t(5))}{2\ln(5)}$
6. **Bollinger Bandwidth & Causal Percentile (`BBWidth`, `BBWP`)**:
   - $\text{BBW}_t = \frac{4 \sigma_{20}(C_t)}{\text{SMA}_{20}(C_t)}, \quad \text{BBWP}_t = \frac{100}{252} \sum_{j=0}^{251} \mathbb{I}(\text{BBW}_{t-j} \le \text{BBW}_t)$
7. **Causal Spread Ratio & Spread Friction (`SpreadRatio`, `SVP14`)**:
   - $\text{SpreadRatio}_t = \frac{\text{Spread}_t (\text{pips})}{\text{median}_{288}(\text{Spread}_{\le t})}$ (falls back to Discovery median spread per pair when bar spread history is absent; **> 3.0x guard** triggers Regime 4).
   - $\text{SVP}_{14}(t) = \frac{\text{Spread}_t (\text{pips})}{\text{ATR}_{14}(t) (\text{pips})}$ (**> 1.20 guard** triggers Regime 4).

---

## 4. Exact Strategy Key & SessionClock Harmonization

- **Exact Normalized Keys**: Strategy lookups normalize to exact `"{SYMBOL}__{strategy_id}"` (lowercase ID, no unsafe substring matching!). Unknown unregistered strategies pass through a universal extreme-tail safety gate (blocking only Regimes 3 and 4, or respecting explicit `archetype=`).
- **SessionClock / UTC Harmonization**:
  - `NY_ROLLOVER`: `ny_hour == 17` (`America/New_York` 17:00–17:59 ET, automatically adapting to EDT 21:00 UTC vs EST 22:00 UTC).
  - `ASIAN`: `21:00-06:00 UTC`, `LONDON`: `06:00-12:00 UTC`, `LONDON_NY_OVERLAP`: `12:00-16:00 UTC`, `NY_AFTERNOON`: `16:00-21:00 UTC`.

---

## 5. Non-USDJPY Strategy Regime Registry (4 Strategies)

| # | Registry Key | Symbol | Archetype | Permitted Regimes | Prohibited Regimes | Toxic Sessions | Toxic Days | Tailored Rule |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 01 | `AUDCAD__strategy_01_audcad_h4_bb_ema_mr` | **AUDCAD** | trend_pullback | 1, 2, 5 | 3, 4 | None | Friday | Block days: Friday |
| 02 | `EURUSD__strategy_02_eurusd_daily_pivot_ny` | **EURUSD** | trend_pullback | 1, 2, 5 | 3, 4 | None | None | 5-state causal archetype rules |
| 03 | `GBPJPY__strategy_03_gbpjpy_h4_sqz_brk` | **GBPJPY** | trend_pullback | 1, 2, 5 | 3, 4 | None | None | 5-state causal archetype rules |
| 04 | `USDCAD__strategy_04_usdcad_pdhl_sweep_ldn` | **USDCAD** | session_sweep | 1, 2, 5 | 3, 4 | None | Friday | Block days: Friday |

---

## 6. Wiring Into Portfolio Simulation & Live Execution

```python
from live_regime_filter import LiveMarketRegimeFilter

regime_engine = LiveMarketRegimeFilter()

# Option A: Single-bar causal query (minimum 272 closed M15 bars, recommended >= 350)
permitted, reason, metrics = regime_engine.evaluate_live_permission(
    strategy_name=strategy_id,
    symbol=symbol,
    high=m15_closed["high"].to_numpy(),
    low=m15_closed["low"].to_numpy(),
    close=m15_closed["close"].to_numpy(),
    spread_pips=live_spread_pips,
    current_utc_dt=decision_utc_ts,
    archetype=strategy_archetype,  # optional explicit fallback
)

# Option B: Fast O(N) vectorized precompute + O(log N) causal lookup at T_close <= e_time
precomputed = regime_engine.precompute_m15_regimes(m15_df, symbol=symbol)
permitted, reason, metrics = regime_engine.evaluate_at_timestamp(
    strategy_name=strategy_id,
    symbol=symbol,
    precomputed=precomputed,
    decision_ts_ms=entry_time_ms,
    archetype=strategy_archetype,
)
```
