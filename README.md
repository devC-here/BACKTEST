# High-Speed Tick-Level Backtesting & Institutional Strategy Verification Platform

An institutional-grade, ultra-lightweight, high-speed tick-level Forex backtesting and strategy verification engine in Python optimized with Numba, NumPy, and PyArrow.

---

## 🏆 Production Certified Fleet (72 Strategies)

This repository hosts a portfolio of **72 certified quantitative trading strategies** that have undergone strict out-of-sample holdout certification and adversarial stress testing.

- **Verified Holdout PnL (Weeks 211..300)**: **+11,723.3 pips** (9,726 trades, 1.26 Profit Factor, +1.21 pips/trade expectancy)
- **5-Year Full Cycle PnL (Weeks 1..300)**: **+42,760.7 pips** (28,178 trades, 1.28 Profit Factor, +1.52 pips/trade expectancy)
- **Multi-Pair Diversification**: 6 currency pairs (**EURUSD** 29, **USDJPY** 24, **USDCAD** 8, **AUDNZD** 4, **EURGBP** 4, **GBPCHF** 3)
- **Fleet Manifest**: [`certified_production_fleet/CERTIFIED_FLEET_MANIFEST.json`](certified_production_fleet/CERTIFIED_FLEET_MANIFEST.json)
- **Detailed Fleet Audit**: [`certified_production_fleet/README.md`](certified_production_fleet/README.md)
- **Full Institutional Verification Report**: [`docs/INSTITUTIONAL_STRATEGY_VERIFICATION_PLATFORM_REPORT.md`](docs/INSTITUTIONAL_STRATEGY_VERIFICATION_PLATFORM_REPORT.md)

---

## 🛡️ 10-Point Institutional Verification Standard

Every certified strategy in this fleet is evaluated under non-negotiable verification gates:

1. **Continuous Chronological Partitioning**: Discovery calibrated strictly on Weeks 1..210 (2021 to mid-2025). Certified only on untouched Quarantined Holdout Weeks 211..300 (mid-2025 to 2026). Zero randomized week shuffling.
2. **Mandatory Real-World Broker Friction**: Strict \$4.00/lot commission + 0.20p adverse entry slippage + raw Dukascopy floating spread.
3. **Causal Rolling Warmup**: 100,000-tick rolling prior-week warmup ensuring indicator state precisely matches live execution without resetting at weekly boundaries or leaking test trades.
4. **Enforced Order-Timing Contract**: Engine requires causal timestamp ordering ($T_{\text{fill}} > T_{\text{signal}}$). Zero same-tick fills permitted.
5. **Real-Time Data Quality Gate**: Automatic rejection/flagging of partial weeks (<80h span, <15,000 ticks), inverted spreads (`Ask < Bid`), quote gaps (>4h during active trading), or non-positive prices.
6. **Canonical Run Manifests**: Every backtest produces a tamper-evident JSON manifest recording strategy parameters, Git commit hash, engine code hash, dataset hashes, and trade logs.
7. **Pessimistic Stress Testing**: Verified against 2x spread widening, 2x slippage, +50% commission, and execution latency.
8. **Automated Integration Test Suite**: 10 rigorous institutional platform tests passing in `tick_engine/tests/test_institutional_platform.py`.

---

## ⚡ Performance Highlights

- **Blazing Throughput:** **~35 Million ticks / second** (1,000,000 ticks evaluated in **0.028s**).
- **Annual Batch Simulation:** Resimulates **52 weeks (17.8 Million ticks)** in **~7.8 seconds**.
- **Memory Efficiency:** Uses zero-copy memory-mapped loading (`mmap_mode='r'`) so multi-gigabyte tick datasets consume near-zero RAM.
- **Strictly Pip-Denominated Accounting:** Eliminates currency conversion, margin, leverage, and lot-sizing complexities.
- **High-Fidelity Execution:** Spread-aware matching (Longs buy at Ask, Shorts sell at Bid; Stop Loss and Take Profit trigger on authentic Bid/Ask sides).
- **Batch 1-by-1 Execution:** Runs week-by-week with real-time per-week progress rows and an aggregated annual scorecard.

---

## 🔬 Automated New Symbol Alpha Discovery SOP & Suite

Whenever tick data for a new symbol arrives, we run the automated **Master Discovery Battery** before authoring any bespoke scripts:

- **Official SOP:** [`SOP_NEW_SYMBOL_ALPHA_DISCOVERY.md`](SOP_NEW_SYMBOL_ALPHA_DISCOVERY.md)
- **Master Engine:** [`data_engineering_suite/master_symbol_alpha_discovery_suite.py`](data_engineering_suite/master_symbol_alpha_discovery_suite.py)
- **Directory Manifest:** [`strategies/PROVEN_PROFITABLE_STRATEGIES_DIRECTORY.md`](strategies/PROVEN_PROFITABLE_STRATEGIES_DIRECTORY.md) (102 Production Champions across EURUSD, USDJPY, EURGBP, AUDNZD)

### Run Discovery Pipeline on Any Symbol:
```bash
# Automated 5-stage discovery & 30% holdout certification:
.venv\Scripts\python.exe run_symbol_alpha_discovery.py --symbol <SYMBOL>

# Custom hurdle rate and multi-worker tuning:
.venv\Scripts\python.exe run_symbol_alpha_discovery.py --symbol <SYMBOL> --hurdle-return 15.0 --hurdle-pf 1.25 --max-workers 12
```


```text
BACKTEST/
│
├── tick_engine/                        # Protected, Standalone High-Speed Backtesting Engine
│   ├── core/
│   │   ├── engine.py                   # Numba @njit(fastmath=True) matching loop
│   │   ├── dataloader.py               # Memory-mapped binary .npy loader & CSV converter
│   │   ├── metrics.py                  # Pip analytics (Net Pips, Win Rate, Profit Factor, Max DD)
│   │   └── block_sampler.py            # 10-Block (6-month units) 6/4 randomized split sampler
│   ├── strategies/
│   │   ├── base.py                     # BaseStrategy interface & signal definitions
│   │   ├── example_strategy.py         # EMA Crossover Strategy
│   │   ├── momentum.py                 # Tick Momentum Strategy
│   │   └── top15.py                    # Top 15 Fleet adapter
│   ├── tests/
│   │   └── test_engine.py              # Unit tests verifying fill prices & SL/TP fidelity
│   ├── data/processed/                 # Cached 262 binary .npy weekly tick datasets
│   ├── run_backtest.py                 # Standalone engine CLI runner
│   └── README.md                       # Dedicated engine user guide
│
├── strategies/                         # 7 Production Champion Strategies (strictly 2 files each)
│   ├── high_frequency_m15_scalper_elite/
│   ├── institutional_v12_elite/
│   ├── s01_m15_displacement_retest_elite/
│   ├── s10_dual_ema_pullback_elite/
│   ├── s14_inside_bar_m30_elite/
│   ├── s34_heikin_ashi_v2_elite/
│   └── s66_bhanushali_44ma_elite/
│
├── production_deploy/                  # Self-contained package ready for VM / H1-ONLINE deployment
├── eurusd_weekly_data/                 # 262 weekly CSV datasets (123,223,663 authentic ticks)
├── results/                            # Canonical 5-Year audit outputs & legacy_archive/
├── legacy_labs/                        # Consolidated archive of historical research & scratch labs
├── SOP_3Y_TRAIN_2Y_TEST.md             # Official SOP (10-Block 6/4 Protocol SOP-TRD-VAL-5Y-002)
├── run_backtest.py                     # Root CLI entry point
├── run_5y_fleet_audit.py               # High-speed multi-core 5-year parallel audit runner
├── slice_5y_weekly.py                  # Streaming raw CSV slicer
└── requirements.txt                    # numpy, numba, pyarrow, pandas
```



---

## ⚖️ Execution Rules (Pip-Only Matching)

| Action | Direction | Trigger Condition | Fill Price | Pip Calculation |
| :--- | :---: | :--- | :--- | :--- |
| **Open Long** | Long (`+1`) | `Signal == 1` | `Current Ask` | — |
| **Close Long (SL)** | Flat (`0`) | `Current Bid <= SL Price` | `SL Price` (or `Bid`) | $(P_{\text{exit}} - P_{\text{entry}}) / \text{pip\_size}$ |
| **Close Long (TP)** | Flat (`0`) | `Current Bid >= TP Price` | `TP Price` (or `Bid`) | $(P_{\text{exit}} - P_{\text{entry}}) / \text{pip\_size}$ |
| **Open Short** | Short (`-1`) | `Signal == -1` | `Current Bid` | — |
| **Close Short (SL)** | Flat (`0`) | `Current Ask >= SL Price` | `SL Price` (or `Ask`) | $(P_{\text{entry}} - P_{\text{exit}}) / \text{pip\_size}$ |
| **Close Short (TP)** | Flat (`0`) | `Current Ask <= TP Price` | `TP Price` (or `Ask`) | $(P_{\text{entry}} - P_{\text{exit}}) / \text{pip\_size}$ |
| **Signal Exit** | Flat (`0`) | `Signal == 2` | Long at `Bid`, Short at `Ask` | Pip differential |

---

## 🚀 Quickstart Commands

### 1. Run Full 1-Year Backtest (52 Weeks in 1-by-1 Batches)
```bash
python run_backtest.py --weeks 1-52 --strategy tick_momentum --sl 15 --tp 30
```
Runs each week sequentially, prints live week results row-by-row, prints the Grand Annual Scorecard, and exports `results/tick_backtest_weekly_summary.csv`.

### 2. Run a Custom Weekly Range (e.g. Weeks 1 to 4)
```bash
python run_backtest.py --weeks 1-4 --strategy ema_crossover --fast 150 --slow 600 --sl 15 --tp 30
```

### 3. Run a Single Week
```bash
python run_backtest.py --week 1 --strategy ema_crossover --sl 15 --tp 30
```

### 4. Run the 1-Million Tick Speed Benchmark
```bash
python run_backtest.py --benchmark
```

### 5. Run the Verification Unit Test Suite
```bash
python run_backtest.py --test
```

### 6. Run the Top 15 Strategy Fleet Side-by-Side
```bash
# Run all 15 models across Weeks 1 to 4:
python run_backtest.py --top15 --weeks 1-4

# Run all 15 models across the entire year (all 52 weeks in 1-by-1 batches):
python run_backtest.py --top15 --all

# Run with 24/5 continuous execution (disable empirical hours filter):
python run_backtest.py --top15 --weeks 1-4 --no-hours-filter
```
Exports comprehensive scorecards to `results/top_15_tick_backtest_summary.csv` and `results/top_15_tick_weekly_matrix.csv`.

### 7. Standard Operating Procedure: 10-Block (6-Month Unit) 6/4 Protocol
Refer to the complete specification: [`SOP_3Y_TRAIN_2Y_TEST.md`](SOP_3Y_TRAIN_2Y_TEST.md) (`SOP-TRD-VAL-5Y-002`).

The 5-year dataset is partitioned into **10 discrete 6-Month Units (B01 to B10)**:
- **6 Random Units** (36 Months / 3 Years) for **In-Sample Training**
- **4 Random Units** (24 Months / 2 Years) for **Unseen Out-of-Sample Testing**

```bash
# 1. Run 6 Random Training Blocks (36 Months / ~67.8M ticks, Seed 42):
python run_backtest.py --strategy <STRATEGY> --random-blocks --split train --seed 42

# 2. Run 4 Unseen Testing Blocks (24 Months / ~55.4M ticks, Seed 42):
python run_backtest.py --strategy <STRATEGY> --random-blocks --split test --seed 42

# 3. Run Specific 6-Month Blocks (e.g. Blocks 1, 3, 5, 7, 9):
python run_backtest.py --strategy <STRATEGY> --blocks 1,3,5,7,9

# 4. Run Full 5-Year Consolidated Audit (All 10 Blocks / 123.22M ticks):
python run_backtest.py --strategy <STRATEGY> --split all --export-csv results/<STRATEGY>_5y_audit.csv

# 5. Run Top 15 Fleet across 4 Unseen Test Blocks:
python run_backtest.py --top15 --random-blocks --split test --seed 42
```


### 8. Run Edge Lab Hypothesis Optimization Suites
```bash
# Run Gen-1 hypothesis comparison (V0 to V6):
python edge_lab/run_experiment.py --weeks all

# Run Gen-2 advanced optimization suite (V6 to V11):
python edge_lab/run_advanced_suite.py --weeks all
```
Exports comparative metrics matrix to `edge_lab/advanced_results.csv` and detailed research notes to `edge_lab/experiment_log.md`.

### 9. Dedicated Production Strategies Directory (`strategies/`)


All validated, production-grade algorithmic strategies are maintained in the minimalist [`strategies/`](strategies/) repository. Each strategy folder contains **strictly 2 files and nothing else** (`strategy.py` + `ABOUT.md`):

```text
strategies/
├── institutional_v12_elite/      # Model V12: Mean-Reversion Asian Liquidity Sweep (PF 2.27, RR 2.84 : 1)
│   ├── ABOUT.md
│   └── strategy.py
├── s10_dual_ema_pullback_elite/  # Model S10: Dual EMA 20/200 Dynamic Pullback (PF 2.19, RR 2.92 : 1)
│   ├── ABOUT.md
│   └── strategy.py
├── s14_inside_bar_m30_elite/     # Model S14: Price-Action Inside Bar Momentum Breakout (PF 2.25, RR 2.85 : 1)
│   ├── ABOUT.md
│   └── strategy.py
├── s34_heikin_ashi_v2_elite/     # Model S34: Trend-Following Smoothed Heikin Ashi Momentum (PF 2.24, RR 2.77 : 1)
│   ├── ABOUT.md
│   └── strategy.py
└── s66_bhanushali_44ma_elite/    # Model S66: Siddharth Bhanushali 44 MA Trend Pullback (PF 2.51, RR 2.87 : 1)
    ├── ABOUT.md
    └── strategy.py
```

#### Production Champion Models (Audited on 18,026,436 Dukascopy Ticks):

| Strategy | Methodology | Timeframe | 1-Yr Trades | Win Rate | Net Pips | Profit Factor (PF) | Realized RR | Max Drawdown | Capital Exposure |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **[`Institutional V12 Elite`](strategies/institutional_v12_elite/ABOUT.md)** | Asian Liquidity Sweep Fade | M15 (07-09 UTC) | 18 | 44.4% | **+125.0 p** | **2.27** | **2.84 : 1** | **37.8 p** | 163.6 hrs |
| **[`S34 Heikin Ashi V2 Elite`](strategies/s34_heikin_ashi_v2_elite/ABOUT.md)** | Smoothed Heikin Ashi Momentum | H1 (09 & 16 UTC) | 47 | 44.7% | **+488.5 p** | **2.24** | **2.77 : 1** | **78.8 p** | 688.1 hrs |
| **[`S14 Inside Bar M30 Elite`](strategies/s14_inside_bar_m30_elite/ABOUT.md)** | Inside Bar Compression Breakout | M30 (07-16 UTC) | 43 | 44.2% | **+384.9 p** | **2.25** | **2.85 : 1** | **122.1 p** | 662.0 hrs |
| **[`S10 Dual EMA Pullback Elite`](strategies/s10_dual_ema_pullback_elite/ABOUT.md)** | 20/200 Dynamic Value Pullback | H1 (07-17 UTC) | 42 | 42.9% | **+501.6 p** | **2.19** | **2.92 : 1** | **157.1 p** | 1,203.0 hrs |
| **[`S66 Bhanushali 44 MA Elite`](strategies/s66_bhanushali_44ma_elite/ABOUT.md)** | 44 MA Structural Pullback | H1 (06-18 UTC) | 30 | 46.7% | **+422.9 p** | **2.51** | **2.87 : 1** | **69.7 p** | 752.7 hrs |
| **COMBINED 5-CHAMPION FLEET** | **5 Orthogonal Alpha Sources** | **Multi-TF** | **180** | **44.4%** | **+1,922.9 p** | **2.28** | **2.85 : 1** | **274.1 p** | **1,612.8 hrs (25.36%)** |

```bash
# Run Model V12 Elite 53-week audit:
python strategies/institutional_v12_elite/strategy.py

# Run S34 Heikin Ashi V2 Elite 53-week audit:
python strategies/s34_heikin_ashi_v2_elite/strategy.py

# Run S14 Inside Bar M30 Elite 53-week audit:
python strategies/s14_inside_bar_m30_elite/strategy.py

# Run S10 Dual EMA Pullback H1 Elite 53-week audit:
python strategies/s10_dual_ema_pullback_elite/strategy.py

# Run S66 Bhanushali 44 MA H1 Elite 53-week audit:
python strategies/s66_bhanushali_44ma_elite/strategy.py
```




