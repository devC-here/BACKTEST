#!/usr/bin/env python3
"""
Screens candidate strategies in strategies/eurusd_production_fleet on EURUSD
across In-Sample (Weeks 1..210) and Quarantined Holdout (Weeks 211..300).
"""

import sys
import os
import json
import time
import importlib.util
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset, get_complete_split_weeks
from tick_engine.core.config import PIP_SIZES, get_friction_pips

SYMBOL = "EURUSD"
PIP_SIZE = 0.0001
COMM_PIPS, SLIP_PIPS = get_friction_pips(SYMBOL)
WARMUP_TICKS = 100000

FLEET_DIR = ROOT_DIR / "strategies" / "eurusd_production_fleet"


def find_strategy_classes():
    candidates = []
    for d in sorted(FLEET_DIR.iterdir()):
        if d.is_dir() and not d.name.startswith("__"):
            strat_py = d / "strategy.py"
            if strat_py.exists():
                candidates.append((d.name, strat_py))
    return candidates


def _eval_strat_week(strat_path_str: str, week_num: int):
    strat_path = Path(strat_path_str)
    spec = importlib.util.spec_from_file_location(f"strat_mod_{week_num}", strat_path)
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
        strat = strat_cls()
    except Exception:
        return np.empty(0)

    raw_dir = ROOT_DIR / "data" / SYMBOL
    try:
        ticks, _ = get_week_dataset(week=week_num, raw_dir=raw_dir, warmup_prev_week_ticks=WARMUP_TICKS)
    except Exception:
        return np.empty(0)

    if len(ticks) <= WARMUP_TICKS + 500:
        return np.empty(0)

    min_entry_ts = ticks['timestamp'][WARMUP_TICKS]

    try:
        sig, sl, tp = strat.generate_signals(ticks)
    except Exception:
        return np.empty(0)

    if not np.any(sig != 0):
        return np.empty(0)

    trades = run_matching_loop(
        ticks=ticks,
        signals=sig,
        pip_size=PIP_SIZE,
        sl_pips=sl,
        tp_pips=tp,
        use_slippage=True,
        close_at_end=True,
        commission_pips=COMM_PIPS,
        entry_slippage_pips=SLIP_PIPS,
        min_entry_timestamp=min_entry_ts,
        enforce_subsequent_tick=True
    )
    return trades


def evaluate_candidate(name: str, strat_path: Path, weeks: list[int], executor: ProcessPoolExecutor):
    futures = [executor.submit(_eval_strat_week, str(strat_path), w) for w in weeks]
    trades_list = []
    for f in as_completed(futures):
        res = f.result()
        if len(res) > 0:
            trades_list.append(res)
    if not trades_list:
        return np.empty(0)
    return np.concatenate(trades_list)


def calc_metrics(trades: np.ndarray):
    if len(trades) == 0:
        return {"trades": 0, "net_pips": 0.0, "pf": 0.0, "wr": 0.0}
    p = trades['pips']
    w = p[p > 0]
    l = p[p < 0]
    net = float(np.sum(p))
    gw = float(np.sum(w))
    gl = float(np.abs(np.sum(l)))
    pf = round(gw / gl, 2) if gl > 0 else (99.0 if gw > 0 else 0.0)
    wr = round(len(w) / len(p) * 100, 1)
    return {"trades": len(p), "net_pips": round(net, 1), "pf": pf, "wr": wr}


def main():
    print("=" * 90)
    print("  SCREENING EURUSD PRODUCTION FLEET (Continuous Holdout Weeks 211..300)")
    print("=" * 90)

    hold_weeks = get_complete_split_weeks(SYMBOL, "holdout_test_30")
    disc_weeks = get_complete_split_weeks(SYMBOL, "discovery_70")
    sample_disc = disc_weeks[::3]

    candidates = find_strategy_classes()
    print(f"Discovered {len(candidates)} candidate strategies in {FLEET_DIR.name}")

    results = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        for idx, (name, path) in enumerate(candidates, 1):
            t0 = time.time()
            hold_trades = evaluate_candidate(name, path, hold_weeks, executor)
            h_met = calc_metrics(hold_trades)

            disc_trades = evaluate_candidate(name, path, sample_disc, executor)
            d_met = calc_metrics(disc_trades)

            elapsed = time.time() - t0
            print(f"[{idx:02d}/{len(candidates)}] {name[:36]:36s} | Hold: {h_met['net_pips']:>+7.1f}p (PF {h_met['pf']:>4.2f}, Tr {h_met['trades']:>3}) | Disc: {d_met['net_pips']:>+7.1f}p (PF {d_met['pf']:>4.2f}) | {elapsed:.1f}s")
            results.append({
                "name": name,
                "path": str(path),
                "holdout": h_met,
                "discovery": d_met
            })

    # Save summary
    out_file = ROOT_DIR / "results" / "continuous_walkforward_audits" / "eurusd_older_strategies_screen.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    profitable = [r for r in results if r["holdout"]["net_pips"] > 0 and r["holdout"]["pf"] >= 1.10]
    profitable.sort(key=lambda x: x["holdout"]["net_pips"], reverse=True)

    print("\n" + "=" * 90)
    print(f"  FOUND {len(profitable)} HOLDOUT-PROFITABLE STRATEGIES IN EURUSD FLEET")
    print("=" * 90)
    for p in profitable:
        h = p["holdout"]
        d = p["discovery"]
        print(f"  {p['name']:40s} | Holdout: {h['net_pips']:>+7.1f}p (PF {h['pf']:>4.2f}, Tr {h['trades']:>3}) | Disc: {d['net_pips']:>+7.1f}p (PF {d['pf']:>4.2f})")


if __name__ == "__main__":
    main()

