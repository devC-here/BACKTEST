# Quantitative Backtesting & Strategy Engine Rules

## 1. Protected Core Architecture
- **`tick_engine/` Core**: The tick engine (`tick_engine/core/`, `tick_engine/run_backtest.py`) is protected and isolated. Never make accidental or ad-hoc edits. Any modifications must pass all unit tests:
  ```bash
  python tick_engine/run_backtest.py --test
  ```
- **High-Performance Execution**: Numba JIT and NumPy vectorized execution are mandatory for tick processing (~800k ticks/sec). Avoid looping in pure Python over raw tick series.

## 2. Strategy Directory Strictness (2-File Rule)
- Each strategy directory inside `strategies/<strategy_name>/` must contain **strictly two files**:
  1. `strategy.py`: Self-contained class inheriting from `BaseStrategy`, implementing `@jit` signals, `on_tick()`, and standard state.
  2. `ABOUT.md`: Comprehensive strategy specification detailing edge premise, indicator math, entry/exit criteria, parameters, and risk management.
- Never write temporary test outputs, CSV logs, or cache files directly inside strategy directories. Store outputs in `results/`.

## 3. 5-Year SOP Validation Protocol (SOP-TRD-VAL-5Y-002)
- **10-Block Architecture**: The 5-year tick dataset (2021-10-01 to 2026-10-01, 262 weeks) is split into **10 discrete 6-Month Units (B01–B10)**.
- **6/4 Randomized Partitioning**:
  - **In-Sample (Train/Tune)**: 6 random blocks (36 months / 3 years).
  - **Out-of-Sample (Test)**: 4 remaining random blocks (24 months / 2 years) evaluated under strict parameter freeze.
- **Target Performance Benchmarks**:
  - **Profit Factor (PF)**: $\ge 2.0$ across both train and test splits.
  - **Realized Risk-to-Reward (R:R)**: $\ge 2.0 : 1$.
  - **Maximum Drawdown**: Must remain constrained without multi-month equity stagnation.

## 4. Analytics & Optimization Stack
- **Optuna**: Use Optuna for Bayesian hyperparameter optimization (TPE sampler) when tuning strategy parameters on the 6 training blocks. Never overfit on the 4 test blocks.
- **DuckDB**: Use DuckDB for sub-second SQL analytics across multi-million trade logs, tick matrices, and weekly performance tables.
