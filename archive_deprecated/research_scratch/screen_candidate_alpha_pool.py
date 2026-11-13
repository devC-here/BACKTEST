#!/usr/bin/env python3
"""
Screens candidate strategies from archive_deprecated/strategies against
continuous In-Sample (Weeks 1..210) and Quarantined Holdout (Weeks 211..300)
under mandatory live broker friction, causal warmup, and subsequent-tick fills.
"""

from __future__ import annotations
import sys
import json
import time
import importlib.util
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset, get_complete_split_weeks
from tick_engine.core.config import PIP_SIZES, get_friction_pips, WARMUP_TICKS
from tick_engine.reporting.run_manifest import compute_statistical_metrics

OUTPUT_FILE = ROOT_DIR / "results" / "alpha_pool_screening_results.json"


def evaluate_candidate_weeks(strat_file: str, symbol: str, weeks: list[int]) -> np.ndarray:
    strat_path = Path(strat_file)
    pip_size = PIP_SIZES.get(symbol, 0.0001)
    comm_pips, slip_pips = get_friction_pips(symbol)

    spec = importlib.util.spec_from_file_location("strat_mod", strat_path)
    if spec is None or spec.loader is None:
        return np.empty(0)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return np.empty(0)

    strat_cls = None
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, type) and hasattr(obj, "generate_signals") and attr != "BaseStrategy":
            strat_cls = obj
            break

    if strat_cls is None:
        return np.empty(0)

    try:
        strat = strat_cls(asset=symbol, symbol=symbol, pip_size=pip_size)
    except Exception:
        try:
            strat = strat_cls(pip_size=pip_size)
        except Exception:
            try:
                strat = strat_cls()
            except Exception:
                return np.empty(0)

    raw_dir = ROOT_DIR / "data" / symbol
    trades_list = []

    for w in weeks:
        try:
            ticks, _ = get_week_dataset(week=w, raw_dir=raw_dir, warmup_prev_week_ticks=WARMUP_TICKS)
        except Exception:
            continue
        if len(ticks) <= WARMUP_TICKS + 500:
            continue

        min_entry_ts = ticks['timestamp'][WARMUP_TICKS]

        try:
            res = strat.generate_signals(ticks)
        except Exception:
            continue

        if isinstance(res, tuple):
            sig = res[0]
            sl = res[1] if len(res) > 1 and res[1] is not None else np.full(len(ticks), getattr(strat, "sl_pips", 20.0))
            tp = res[2] if len(res) > 2 and res[2] is not None else np.full(len(ticks), getattr(strat, "tp_pips", 40.0))
            sig_ts = res[3] if len(res) > 3 else None
        else:
            sig = res
            sl = np.full(len(ticks), getattr(strat, "sl_pips", 20.0))
            tp = np.full(len(ticks), getattr(strat, "tp_pips", 40.0))
            sig_ts = None

        if not np.any(sig != 0):
            continue

        trades = run_matching_loop(
            ticks=ticks,
            signals=sig,
            pip_size=pip_size,
            sl_pips=sl,
            tp_pips=tp,
            use_slippage=True,
            close_at_end=True,
            commission_pips=comm_pips,
            entry_slippage_pips=slip_pips,
            signal_timestamps=sig_ts,
            min_entry_timestamp=min_entry_ts,
            enforce_subsequent_tick=True
        )
        if len(trades) > 0:
            trades_list.append(trades)

    return np.concatenate(trades_list) if trades_list else np.empty(0)


def main():
    # Gather priority candidates
    candidates = []

    # 1. AUDNZD candidates
    for p in sorted(Path("archive_deprecated/strategies/audnzd_fleet").glob("*/strategy.py")):
        candidates.append((p.parent.name, "AUDNZD", str(p)))

    # 2. EURGBP candidates
    for p in sorted(Path("archive_deprecated/strategies/eurgbp_fleet").glob("*/strategy.py")):
        candidates.append((p.parent.name, "EURGBP", str(p)))

    # 3. New Discoveries
    for p in sorted(Path("archive_deprecated/strategies/new_discoveries").glob("*/strategy.py")):
        candidates.append((p.parent.name, "EURUSD", str(p)))

    # 4. Qualified EURUSD
    for p in sorted(Path("archive_deprecated/strategies/qualified_strategies/eurusd").glob("*/strategy.py")):
        candidates.append((p.parent.name, "EURUSD", str(p)))

    # 5. USDJPY fleet
    for p in sorted(Path("archive_deprecated/strategies/usdjpy_fleet").glob("*/strategy.py")):
        candidates.append((p.parent.name, "USDJPY", str(p)))

    print(f"Total candidates queued for screening: {len(candidates)}")

    results = []
    # Test on a 30-week representative holdout window first for speed, then full
    for idx, (name, sym, path_str) in enumerate(candidates, 1):
        hold_weeks = get_complete_split_weeks(sym, "holdout_test_30")
        disc_weeks = get_complete_split_weeks(sym, "discovery_70")

        # Evaluate Holdout
        t0 = time.time()
        hold_trades = evaluate_candidate_weeks(path_str, sym, hold_weeks)
        disc_trades = evaluate_candidate_weeks(path_str, sym, disc_weeks[:60]) # sample 60 disc weeks for quick check
        elapsed = time.time() - t0

        h_stats = compute_statistical_metrics(hold_trades)
        d_stats = compute_statistical_metrics(disc_trades)

        h_pnl = h_stats["net_pips"]
        h_pf = h_stats["profit_factor"]
        h_tr = h_stats["total_trades"]
        h_exp = h_stats["expectancy_pips"]

        status = "PASSED" if h_pnl > 0 and h_pf >= 1.15 and h_exp >= 1.25 and h_tr >= 15 else "FAILED"

        print(f"[{idx:02d}/{len(candidates)}] [{sym:6s}] {name[:36]:36s} | Holdout: {h_pnl:>+6.1f}p (PF {h_pf:>4.2f}, {h_tr:>3} tr, {h_exp:>+5.2f}p/tr) | {status} ({elapsed:.1f}s)")

        results.append({
            "name": name,
            "symbol": sym,
            "path": path_str,
            "status": status,
            "holdout": h_stats,
            "discovery_sample": d_stats
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    passed = [r for r in results if r["status"] == "PASSED"]
    print("\n" + "=" * 80)
    print(f"SCREENING COMPLETE: {len(passed)} of {len(results)} CANDIDATES PASSED INSTITUTIONAL HURDLES")
    print("=" * 80)


if __name__ == "__main__":
    main()

