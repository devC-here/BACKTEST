# Fast Minimal Tick-Level Strategy Tester (`tick_engine`)

An ultra-lightweight, high-speed, tick-level Forex backtesting engine in Python optimized with Numba and NumPy.

---

## ⚡ Performance Highlights

- **Simulation Throughput:** **~34.6 Million ticks / second** (1M ticks evaluated in **0.0289 seconds**).
- **Benchmark Target:** `< 1.5 seconds per million ticks` (**51.9x faster** than target).
- **Zero-Copy Memory-Mapped Loading:** Reads 210,000+ ticks in **~2 ms** without loading full datasets into RAM.
- **Strictly Pip-Denominated Accounting:** Eliminates currency conversion, margin, leverage, and lot-sizing complexities.
- **High-Fidelity Execution:** Spread-aware matching (Longs buy at Ask, Shorts sell at Bid; Stop Loss and Take Profit trigger on accurate Bid/Ask sides).

---

## 📂 Directory Layout

```text
tick_engine/
│
├── data/
│   ├── raw/                 # Incoming CSV tick files
│   └── processed/           # Converted .npy binary files (memory-mapped)
│
├── core/
│   ├── __init__.py          # Core package exports
│   ├── engine.py            # Numba JIT loop and order-matching logic
│   ├── dataloader.py        # Fast chunking and memory-mapping loader
│   └── metrics.py           # Pip statistics (Win Rate, DD, Profit Factor)
│
├── strategies/
│   ├── __init__.py          # Strategy registry
│   ├── base.py              # BaseStrategy abstract interface & signal codes
│   └── example_strategy.py  # Sample EMA crossover strategy (Numba-accelerated)
│
├── tests/
│   └── test_engine.py       # Unit test suite verifying execution fidelity
│
├── run_backtest.py          # Entry point CLI script & benchmark harness
├── requirements.txt         # numpy, numba, pyarrow
└── README.md                # System documentation
```

---

## 📊 Data Schema

All data structures use contiguous C-arrays for optimal JIT compilation:

### A. Tick Structure (`TICK_DTYPE`)
```python
import numpy as np

TICK_DTYPE = np.dtype([
    ('timestamp', np.int64),    # Unix epoch (milliseconds or nanoseconds)
    ('bid', np.float64),        # Current bid price
    ('ask', np.float64)         # Current ask price
])
```

### B. Trade Log Structure (`TRADE_DTYPE`)
```python
TRADE_DTYPE = np.dtype([
    ('entry_time', np.int64),
    ('exit_time', np.int64),
    ('direction', np.int8),     # 1 = Long, -1 = Short
    ('entry_price', np.float64),
    ('exit_price', np.float64),
    ('pips', np.float64),
    ('exit_reason', np.int8)    # 1 = SL, 2 = TP, 3 = Signal Exit
])
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
python tick_engine/run_backtest.py --all --strategy tick_momentum --sl 15 --tp 30
```
Runs each week sequentially, prints live week results row-by-row, prints the Grand Annual Scorecard, and exports `results/tick_backtest_weekly_summary.csv`.

### 2. Run a Custom Range of Weeks (e.g. Weeks 1 to 12)
```bash
python tick_engine/run_backtest.py --weeks 1-12 --strategy ema_crossover --sl 15 --tp 30
```

### 3. Run a Single Week
```bash
python tick_engine/run_backtest.py --week 1 --strategy ema_crossover --sl 15 --tp 30
```

### 4. Run the 1-Million Tick Speed Benchmark
```bash
python tick_engine/run_backtest.py --benchmark
```

### 5. Run the Verification Unit Test Suite
```bash
python tick_engine/run_backtest.py --test
```

### 6. Batch Convert Raw CSVs to Memory-Mapped `.npy`
```bash
python tick_engine/run_backtest.py --convert --raw-dir eurusd_weekly_data
```
