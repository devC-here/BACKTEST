# DEPRECATION WARNING — `v4.6` Production Fleet

> [!CAUTION]
> **DO NOT DEPLOY OR TRADE THE `v4.6` STRATEGY FLEET.**
> This directory is retained **for forensic audit history only**. All production deployment should use **`v5.0/`**.

## Why `v4.6` Was Deprecated

1. **HTF Resampling Lookahead Leak (`label='left'` / future bar close knowledge):**
   - Forensic Gate-0 truncation audit (`results/forensic_causal_vs_lookahead_holdout_audit.json`, `FORENSIC_SYSTEM_AND_STRATEGY_AUDIT_REPORT.md`) proved that the `v4.6` fleet's reported Holdout performance (`+43,006.8 net pips`) was an illusion caused by `pd.DataFrame.resample(...)` defaulting to `closed='left', label='left'` followed by `reindex(..., method='ffill')` and unshifted Sunday open pivots.
   - Because bars were labeled by open time `T_open` instead of close time `T_close`, strategies received H4/H1/M30 closing direction up to `3h 45m` before those HTF bars closed.
   - Re-running all 71 `v4.6` strategies under strictly causal subsequent-tick execution collapsed aggregate Holdout PnL from **`+43,006.8p` to `-19,785.5p`**.

2. **Portfolio Concurrency Lookahead & Legacy Slippage Model:**
   - `v4.6/run_portfolio_simulation.py` inspected future realized trade outcomes (`s["pips"]`) at slot-contention decision time instead of causal mark-to-market Bid/Ask prices, and earlier simulations lacked `$4.00/lot` commission + `0.20p` entry slippage consistently enforced across forced cuts.

3. **Retirement of `USDJPY`:**
   - All `USDJPY` strategies were permanently retired after live trading failure (~9% win rate) and causal Holdout collapse.

4. **Legacy Regime Filter Catch-All Bug:**
   - The legacy `v4.6` `MARKET_REGIME_SYSTEM.py` / `live_regime_filter.py` classified **97.6%** (`617/632`) of real EURUSD M15 bars as blocked `Regime 3: High Volatility Chop`.

## Production Replacement (`v5.0/`)

Use **[`v5.0/`](../v5.0/)**, which contains only strategies rebuilt with `tick_engine/core/causal_bars.py`, screened strictly on Discovery (`Weeks 1..210`), and verified to pass all 6 independent gates including **Gate 0 (`0` truncation-causality violations)** in `scripts/verify_strategy_candidate.py`.
