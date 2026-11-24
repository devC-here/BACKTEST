# v5.0 Strictly Causal Production Fleet

**Status:** Certified & Strictly Causal (`USDJPY` Permanently Excluded)  
**Discovery Screening Partition:** Weeks `1..210` (parameter selection & regime calibration locked strictly here)  
**Holdout Verification Partition:** Weeks `211..300` (unseen out-of-sample verification across 89–90 weeks)  
**Execution Contract:** Strict subsequent-tick execution (`T_fill > T_signal`), `2`-week rolling causal warmup, floating raw ECN Bid/Ask spreads, `$4.00/lot` round-turn commission, `0.20` pips entry slippage, and **0 Gate-0 truncation-causality violations**.

---

## Certified Fleet Roster (4 Orthogonal Strategies, Aggregate Holdout: 151 trades, +1136.1p | Discovery: 366 trades, +1774.8p)

| Slot | Symbol | Strategy File | Archetype | Base SL/TP (pips) | Discovery (`Weeks 1..210`) | Holdout (`Weeks 211..300`) | Holdout Ex-Top3 | Broker Stress |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `AUDCAD` | `strategy_01_audcad_h4_bb_ema_mr.py` | `H4_BB_MR_V6` | 48 / 96 | 94 (55.3%, PF 1.47, +627.6p) | 43 (48.8%, PF 1.97, +330.4p) | +95.6p | PASS (5/5) |
| 2 | `EURUSD` | `strategy_02_eurusd_daily_pivot_ny.py` | `DAILY_PIVOT_PB` | 14 / 28 | 89 (40.4%, PF 1.20, +153.6p) | 42 (54.8%, PF 2.06, +292.1p) | +208.2p | PASS (5/5) |
| 3 | `GBPJPY` | `strategy_03_gbpjpy_h4_sqz_brk.py` | `H4_COMP_SQZ_V5` | 40 / 60 | 88 (51.1%, PF 1.38, +658.4p) | 40 (50.0%, PF 1.40, +326.3p) | +146.8p | PASS (Base) |
| 4 | `USDCAD` | `strategy_04_usdcad_pdhl_sweep_ldn.py` | `PDHL_SWEEP` | 16 / 32 | 95 (42.1%, PF 1.36, +335.2p) | 26 (50.0%, PF 1.84, +187.3p) | +92.2p | PASS (5/5) |

*(Note: `strategy_01_audcad_h4_bb_ema_mr.py` uses dynamic causal H4 ATR(14)-scaled SL/TP and forced EMA20/10-bar exits around a `48/96` baseline.)*

## Included Deployment & Simulation Files

- `FLEET_MANIFEST.json` — Machine-readable slot manifest, SHA-256 hashes, Discovery/Holdout metrics, and manifest/CSV paths
- `live_regime_filter.py` — 5-state Discovery-calibrated strictly causal regime classifier (`LiveMarketRegimeFilter`)
- `trade_controller.py` — Risk, spread guard, slot cap, and commission/pip-value controller
- `run_portfolio_simulation.py` — Strictly causal Mark-to-Market 20-slot portfolio simulator (`scripts/run_v5_causal_portfolio_simulation.py`)

## Quick Verification & Simulation Commands

```powershell
$env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'
.venv\Scripts\python.exe v5.0\run_portfolio_simulation.py --manifest v5.0\FLEET_MANIFEST.json --trades-dir results\v5_certified_manifests --output results\v5_holdout_portfolio_simulation.json
```
