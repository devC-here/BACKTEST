"""
Pytest Verification Suite for Tick-Level Backtesting Engine.
Verifies:
a) Spread penalty (no zero-cost round trips).
b) Slippage through gaps on SL.
c) Time causality (exit_time >= entry_time).
d) Data parsing of Dukascopy tick CSV (Ask before Bid, Unix epoch ms, mmap_mode='r').
"""

import os
import tempfile
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from tick_engine.core.engine import (
    TICK_DTYPE,
    TRADE_DTYPE,
    run_matching_loop,
    EXIT_REASON_SL,
    EXIT_REASON_TP,
    EXIT_REASON_SIGNAL,
    SIGNAL_HOLD,
    SIGNAL_LONG,
    SIGNAL_SHORT,
    SIGNAL_FORCE_EXIT
)
from tick_engine.core.dataloader import (
    parse_dukascopy_csv,
    load_ticks,
    TICK_DTYPE as DL_TICK_DTYPE
)

PIP_SIZE = 0.0001


# ==============================================================================
# TEST 1: Spread Penalty (No Zero-Cost Round Trips)
# ==============================================================================

def test_spread_penalty_long():
    """
    Verifies that Long entries MUST enter at Ask and exit at Bid.
    A round trip with zero price movement MUST incur a loss equal to the full spread.
    """
    ticks = np.zeros(2, dtype=TICK_DTYPE)
    ticks['timestamp'] = [1000, 2000]
    # Spread is 1.5 pips: Bid = 1.10000, Ask = 1.10015
    ticks['bid'] = [1.10000, 1.10000]
    ticks['ask'] = [1.10015, 1.10015]

    signals = np.array([SIGNAL_LONG, SIGNAL_FORCE_EXIT], dtype=np.int8)

    trades = run_matching_loop(
        ticks=ticks,
        signals=signals,
        pip_size=PIP_SIZE,
        sl_pips=50.0,
        tp_pips=50.0,
        close_at_end=False
    )

    assert len(trades) == 1
    t = trades[0]
    assert t['direction'] == 1
    # Long MUST enter at Ask
    assert t['entry_price'] == pytest.approx(1.10015, abs=1e-6)
    # Long MUST exit at Bid
    assert t['exit_price'] == pytest.approx(1.10000, abs=1e-6)
    # Loss must be exactly -1.5 pips (the full spread penalty, never 0.0)
    assert t['pips'] == pytest.approx(-1.5, abs=1e-4)


def test_spread_penalty_short():
    """
    Verifies that Short entries MUST enter at Bid and exit at Ask.
    A round trip with zero price movement MUST incur a loss equal to the full spread.
    """
    ticks = np.zeros(2, dtype=TICK_DTYPE)
    ticks['timestamp'] = [1000, 2000]
    # Spread is 1.2 pips: Bid = 1.20000, Ask = 1.20012
    ticks['bid'] = [1.20000, 1.20000]
    ticks['ask'] = [1.20012, 1.20012]

    signals = np.array([SIGNAL_SHORT, SIGNAL_FORCE_EXIT], dtype=np.int8)

    trades = run_matching_loop(
        ticks=ticks,
        signals=signals,
        pip_size=PIP_SIZE,
        sl_pips=50.0,
        tp_pips=50.0,
        close_at_end=False
    )

    assert len(trades) == 1
    t = trades[0]
    assert t['direction'] == -1
    # Short MUST enter at Bid
    assert t['entry_price'] == pytest.approx(1.20000, abs=1e-6)
    # Short MUST exit at Ask
    assert t['exit_price'] == pytest.approx(1.20012, abs=1e-6)
    # Loss must be exactly -1.2 pips (the full spread penalty, never 0.0)
    assert t['pips'] == pytest.approx(-1.2, abs=1e-4)


# ==============================================================================
# TEST 2: Slippage Through Gaps on Stop Loss
# ==============================================================================

def test_slippage_through_gaps_on_long_sl():
    """
    Verifies that when price gaps down past a Long SL,
    the exit price is the actual tick Bid price, NOT the nominal SL price.
    """
    ticks = np.zeros(3, dtype=TICK_DTYPE)
    ticks['timestamp'] = [1000, 2000, 3000]
    # Entry at Ask = 1.10000. SL = 10 pips -> nominal SL price = 1.09900
    # Tick 1: Bid is 1.09950 (inside stop)
    # Tick 2: Sudden gap down: Bid drops directly to 1.09800 (10 pips below nominal SL)
    ticks['bid'] = [1.09990, 1.09950, 1.09800]
    ticks['ask'] = [1.10000, 1.09960, 1.09810]

    signals = np.array([SIGNAL_LONG, SIGNAL_HOLD, SIGNAL_HOLD], dtype=np.int8)

    trades = run_matching_loop(
        ticks=ticks,
        signals=signals,
        pip_size=PIP_SIZE,
        sl_pips=10.0,
        tp_pips=30.0,
        use_slippage=True,
        close_at_end=False
    )

    assert len(trades) == 1
    t = trades[0]
    assert t['direction'] == 1
    assert t['exit_reason'] == EXIT_REASON_SL
    assert t['entry_price'] == pytest.approx(1.10000, abs=1e-6)
    # MUST exit at actual slipped price (1.09800), NOT nominal SL (1.09900)
    assert t['exit_price'] == pytest.approx(1.09800, abs=1e-6)
    # Realized loss is -20.0 pips (including -10.0 pips gap slippage)
    assert t['pips'] == pytest.approx(-20.0, abs=1e-4)


def test_slippage_through_gaps_on_short_sl():
    """
    Verifies that when price gaps up past a Short SL,
    the exit price is the actual tick Ask price, NOT the nominal SL price.
    """
    ticks = np.zeros(3, dtype=TICK_DTYPE)
    ticks['timestamp'] = [1000, 2000, 3000]
    # Entry at Bid = 1.20000. SL = 10 pips -> nominal SL price = 1.20100
    # Tick 2: Sudden gap up: Ask jumps directly to 1.20250 (15 pips above nominal SL)
    ticks['bid'] = [1.20000, 1.20040, 1.20240]
    ticks['ask'] = [1.20010, 1.20050, 1.20250]

    signals = np.array([SIGNAL_SHORT, SIGNAL_HOLD, SIGNAL_HOLD], dtype=np.int8)

    trades = run_matching_loop(
        ticks=ticks,
        signals=signals,
        pip_size=PIP_SIZE,
        sl_pips=10.0,
        tp_pips=30.0,
        use_slippage=True,
        close_at_end=False
    )

    assert len(trades) == 1
    t = trades[0]
    assert t['direction'] == -1
    assert t['exit_reason'] == EXIT_REASON_SL
    assert t['entry_price'] == pytest.approx(1.20000, abs=1e-6)
    # MUST exit at actual slipped price (1.20250), NOT nominal SL (1.20100)
    assert t['exit_price'] == pytest.approx(1.20250, abs=1e-6)
    # Realized loss is -25.0 pips (including -15.0 pips gap slippage)
    assert t['pips'] == pytest.approx(-25.0, abs=1e-4)


# ==============================================================================
# TEST 3: Time Causality (exit_time >= entry_time)
# ==============================================================================

def test_time_causality():
    """
    Verifies that for all trades, exit_time is strictly >= entry_time.
    Tests across multiple trades including immediate exits, SLs, TPs, and reversals.
    """
    n = 20
    ticks = np.zeros(n, dtype=TICK_DTYPE)
    base_ts = 1672531200000  # Unix epoch ms
    ticks['timestamp'] = [base_ts + (i * 1000) for i in range(n)]
    # Drifting prices
    ticks['bid'] = [1.1000 + (0.0001 * np.sin(i)) for i in range(n)]
    ticks['ask'] = ticks['bid'] + 0.0001

    signals = np.zeros(n, dtype=np.int8)
    signals[0] = SIGNAL_LONG
    signals[4] = SIGNAL_SHORT  # Reversal
    signals[10] = SIGNAL_FORCE_EXIT
    signals[12] = SIGNAL_SHORT
    signals[18] = SIGNAL_LONG  # Reversal

    trades = run_matching_loop(
        ticks=ticks,
        signals=signals,
        pip_size=PIP_SIZE,
        sl_pips=20.0,
        tp_pips=30.0,
        close_at_end=True
    )

    assert len(trades) > 0
    for i, t in enumerate(trades):
        # Strict time causality
        assert t['exit_time'] >= t['entry_time'], (
            f"Trade {i} violated causality: entry={t['entry_time']}, exit={t['exit_time']}"
        )
        assert t['entry_time'] >= base_ts
        assert t['exit_time'] <= ticks['timestamp'][-1]


# ==============================================================================
# TEST 4: Dukascopy CSV Data Parsing & Memory Mapping
# ==============================================================================

def test_dukascopy_csv_parsing_and_mmap():
    """
    Verifies parsing of Dukascopy CSV with columns [Time, Ask, Bid, AskVolume, BidVolume].
    Ensures:
    1. Ask comes BEFORE Bid.
    2. Timestamps are parsed to int64 Unix milliseconds (UTC).
    3. Output .npy has dtype [('timestamp', 'i8'), ('bid', 'f8'), ('ask', 'f8')].
    4. Memory mapping (mmap_mode='r') is fully functional.
    """
    csv_content = (
        "Time,Ask,Bid,AskVolume,BidVolume\n"
        "2023.01.02 00:00:00.123,1.07050,1.07038,1.5,2.0\n"
        "2023.01.02 00:00:01.456,1.07052,1.07040,1.2,1.8\n"
        "2023.01.02 00:00:02.789,1.07048,1.07036,2.0,2.5\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "EURUSD_test.csv"
        csv_path.write_text(csv_content, encoding="utf-8")

        npy_path = Path(tmpdir) / "EURUSD_test.npy"
        res_path = parse_dukascopy_csv(csv_path, npy_path)

        assert res_path.exists()

        # Load with memory mapping
        loaded_ticks = load_ticks(res_path, mmap=True)

        assert isinstance(loaded_ticks, np.memmap)
        assert len(loaded_ticks) == 3

        # Verify dtype
        assert loaded_ticks.dtype == np.dtype([('timestamp', 'i8'), ('bid', 'f8'), ('ask', 'f8')])

        # Verify Ask came before Bid in parsing
        # Line 1: Ask = 1.07050, Bid = 1.07038
        assert loaded_ticks[0]['ask'] == pytest.approx(1.07050, abs=1e-6)
        assert loaded_ticks[0]['bid'] == pytest.approx(1.07038, abs=1e-6)

        # Verify timestamp parsed to int64 milliseconds
        assert loaded_ticks[0]['timestamp'] == 1672617600123
        assert loaded_ticks[1]['timestamp'] == 1672617601456
        assert loaded_ticks[2]['timestamp'] == 1672617602789

        # Explicitly release Windows file lock on memmap file
        del loaded_ticks
        import gc
        gc.collect()


# ==============================================================================
# TEST 5: MT5 Reference Benchmark Cross-Validation (Aug 31 - Sep 04, 2026)
# ==============================================================================

def test_mt5_benchmark_reconciliation():
    """
    Verifies the alternating BUY/SELL benchmark strategy against the MT5 reference baseline
    on the exact Dukascopy tick slice: Aug 31, 2026 00:00:00 UTC to Sep 04, 2026 23:59:59 UTC.
    All 6 reference trades must match direction, exit reason (TP/SL), and exit price (< 0.0001).
    """
    from scratch.test_mt5_benchmark import (
        run_benchmark_simulation,
        MT5_TARGET_TRADES,
        PIP_EURUSD,
    )

    data_path = Path("tick_engine/data/processed/week_49.npy")
    assert data_path.exists(), "week_49.npy required for MT5 benchmark test"

    ticks_all = np.load(data_path, mmap_mode="r")
    start_ts = int(pd.Timestamp("2026-08-31 00:00:00", tz="UTC").timestamp() * 1000)
    end_ts = int(pd.Timestamp("2026-09-04 23:59:59", tz="UTC").timestamp() * 1000)

    mask = (ticks_all["timestamp"] >= start_ts) & (ticks_all["timestamp"] <= end_ts)
    ticks_slice = np.array(ticks_all[mask])

    trades = run_benchmark_simulation(ticks_slice, sl_pips=15.0, tp_pips=30.0, pip_size=PIP_EURUSD)

    assert len(trades) >= len(MT5_TARGET_TRADES)

    dir_map = {1: "BUY", -1: "SELL"}
    reason_map = {1: "SL", 2: "TP", 3: "END"}

    for i, ref in enumerate(MT5_TARGET_TRADES):
        n_trade = trades[i]
        n_dir = dir_map[n_trade["direction"]]
        n_reason = reason_map[n_trade["exit_reason"]]

        assert n_dir == ref["action"], f"Trade {i+1} direction mismatch: {n_dir} vs {ref['action']}"
        assert n_reason == ref["reason"], f"Trade {i+1} exit reason mismatch: {n_reason} vs {ref['reason']}"
        price_diff = abs(n_trade["exit_price"] - ref["exit_price"])
        assert price_diff < 0.0001, f"Trade {i+1} exit price diff {price_diff:.5f} exceeds 1 pip tolerance"

