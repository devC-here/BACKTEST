#!/usr/bin/env python3
"""
Universal institutional fleet screener for any symbol and directory of strategy folders.
Evaluates candidates across Quarantined Holdout (Weeks 211..300) and In-Sample (Weeks 1..210)
under strict subsequent-tick causal execution, 100k prior-week warmup, and live broker friction.
"""

import sys
import json
import time
import argparse
import importlib.util
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset, get_complete_split_weeks
from tick_engine.core.config import PIP_SIZES, get_friction_pips, WARMUP_TICKS


def find_strategy_classes(target_dir: Path):
    candidates = []
    for d in sorted(target_dir.iterdir()):
        if d.is_dir() and not d.name.startswith("__"):
            strat_py = d / "strategy.py"
            if strat_py.exists():
                candidates.append((d.name, strat_py))
    return candidates


def _eval_strat_week(strat_path_str: str, symbol: str, week_num: int):
    strat_path = Path(strat_path_str)
    pip_size = PIP_SIZES.get(symbol, 0.0001)
    comm_pips, slip_pips = get_friction_pips(symbol)

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

    raw_dir = ROOT_DIR / "data" / symbol
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
        pip_size=pip_size,
        sl_pips=sl,
        tp_pips=tp,
        use_slippage=True,
        close_at_end=True,
        commission_pips=comm_pips,
        entry_slippage_pips=slip_pips,
        min_entry_timestamp=min_entry_ts,
        enforce_subsequent_tick=True
    )
    return trades


def evaluate_candidate(name: str, strat_path: Path, symbol: str, weeks: list[int], executor: ProcessPoolExecutor):
    futures = [executor.submit(_eval_strat_week, str(strat_path), symbol, w) for w in weeks]
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


def screen_fleet(symbol: str, target_dir_path: str):
    target_dir = Path(target_dir_path)
    comm_pips, slip_pips = get_friction_pips(symbol)

    print("=" * 95)
    print(f"  INSTITUTIONAL FLEET SCREENER: {symbol} ({target_dir.name})")
    print(f"  Friction: {comm_pips:.2f}p Commission ($4/lot) + {slip_pips:.2f}p Adverse Slippage + Floating Spread")
    print("=" * 95)

    hold_weeks = get_complete_split_weeks(symbol, "holdout_test_30")
    disc_weeks = get_complete_split_weeks(symbol, "discovery_70")
    sample_disc = disc_weeks[::3]

    candidates = find_strategy_classes(target_dir)
    print(f"Found {len(candidates)} candidate strategies in {target_dir}")

    results = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        for idx, (name, path) in enumerate(candidates, 1):
            t0 = time.time()
            hold_trades = evaluate_candidate(name, path, symbol, hold_weeks, executor)
            h_met = calc_metrics(hold_trades)

            disc_trades = evaluate_candidate(name, path, symbol, sample_disc, executor)
            d_met = calc_metrics(disc_trades)

            elapsed = time.time() - t0
            print(f"[{idx:02d}/{len(candidates)}] {name[:36]:36s} | Hold: {h_met['net_pips']:>+7.1f}p (PF {h_met['pf']:>4.2f}, Tr {h_met['trades']:>3}) | Disc: {d_met['net_pips']:>+7.1f}p (PF {d_met['pf']:>4.2f}) | {elapsed:.1f}s")
            results.append({
                "name": name,
                "symbol": symbol,
                "path": str(path),
                "holdout": h_met,
                "discovery": d_met
            })

    out_file = ROOT_DIR / "results" / "continuous_walkforward_audits" / f"{symbol.lower()}_fleet_screen.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    # Filter for dual-partition profitable candidates
    dual_prof = [r for r in results if r["holdout"]["net_pips"] > 0 and r["discovery"]["net_pips"] > 0 and r["holdout"]["pf"] >= 1.05 and r["discovery"]["pf"] >= 1.05]
    dual_prof.sort(key=lambda x: (x["holdout"]["net_pips"] + x["discovery"]["net_pips"]), reverse=True)

    print("\n" + "=" * 95)
    print(f"  FOUND {len(dual_prof)} DUAL-PARTITION PROFITABLE STRATEGIES FOR {symbol}")
    print("=" * 95)
    for idx, p in enumerate(dual_prof, 1):
        h = p["holdout"]
        d = p["discovery"]
        tot = h["net_pips"] + d["net_pips"]
        print(f"  [{idx:02d}] {p['name']:38s} | Hold: {h['net_pips']:>+7.1f}p (PF {h['pf']:>4.2f}) | Disc: {d['net_pips']:>+7.1f}p (PF {d['pf']:>4.2f}) | Tot: {tot:>+8.1f}p")

    return dual_prof


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="USDJPY")
    parser.add_argument("--dir", type=str, default="archive_deprecated/strategies/usdjpy_fleet")
    args = parser.parse_args()

    screen_fleet(args.symbol, args.dir)

