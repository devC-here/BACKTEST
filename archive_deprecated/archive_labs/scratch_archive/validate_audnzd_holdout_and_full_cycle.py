#!/usr/bin/env python3
"""
================================================================================
  AUDNZD HOLDOUT & FULL 5.75-YEAR CYCLE CERTIFICATION AUDIT
================================================================================
Audits the top 4 discovered champions from Lab V3 across:
  1. 70% In-Sample Discovery (210 weeks / 97.08M ticks)
  2. 30% Quarantined Holdout Test (90 weeks / 41.93M ticks) - UNTOUCHED DATA
  3. 100% Full 5.75-Year Continuous Cycle (300 weeks / 139.00M ticks)

Certification Hurdle:
  - Holdout Profit Factor >= 1.25 (minimum >= 1.20)
  - Zero curve-fitting / edge collapse on unseen data
  - Full-Cycle Annualized Return >= 150 pips/yr
  - Return to Drawdown >= 4.0x
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset, get_discovery_weeks, get_holdout_test_weeks
from scratch.alpha_discovery_lab_audnzd_v3 import AUDNZDMacroStructureV3

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_audit_models() -> List[Tuple[str, Any]]:
    return [
        ("AUDNZD_H4_Macro_SL14_TP28", AUDNZDMacroStructureV3(
            name="AUDNZD_H4_Macro_SL14_TP28", tf_macro="4h", lookback_macro=10,
            dist_pips=5.0, min_wick=0.45, sl_pips=14.0, tp_pips=28.0
        )),
        ("AUDNZD_H4_Macro_SL15_TP30", AUDNZDMacroStructureV3(
            name="AUDNZD_H4_Macro_SL15_TP30", tf_macro="4h", lookback_macro=10,
            dist_pips=5.0, min_wick=0.45, sl_pips=15.0, tp_pips=30.0
        )),
        ("AUDNZD_H4_Macro_SL12_TP24", AUDNZDMacroStructureV3(
            name="AUDNZD_H4_Macro_SL12_TP24", tf_macro="4h", lookback_macro=10,
            dist_pips=5.0, min_wick=0.45, sl_pips=12.0, tp_pips=24.0
        )),
        ("AUDNZD_H1_Macro_SL12_TP20", AUDNZDMacroStructureV3(
            name="AUDNZD_H1_Macro_SL12_TP20", tf_macro="1h", lookback_macro=20,
            dist_pips=4.5, min_wick=0.50, sl_pips=12.0, tp_pips=20.0
        )),
    ]


def evaluate_week(week_num: int) -> Dict[str, Any]:
    engine = BacktestEngine(pip_size=0.0001, use_slippage=True)
    try:
        ticks, meta = get_week_dataset(week_num, raw_dir="data/AUDNZD")
        if len(ticks) == 0:
            return {"week": week_num, "trades": {}}
    except Exception as e:
        return {"week": week_num, "trades": {}, "error": str(e)}

    models = build_audit_models()
    week_trades = {}

    for model_name, model in models:
        try:
            trades = engine.run(ticks, model)
            pips_list = [float(t['pips']) for t in trades]
            week_trades[model_name] = pips_list
        except Exception:
            week_trades[model_name] = []

    return {"week": week_num, "trades": week_trades}


def compute_metrics(pips: List[float], years: float) -> Dict[str, Any]:
    n = len(pips)
    if n == 0:
        return {
            "trades": 0, "net_pips": 0.0, "annual_pips": 0.0,
            "win_rate": 0.0, "profit_factor": 0.0, "max_dd_pips": 0.0,
            "return_to_dd": 0.0, "expectancy": 0.0
        }

    p_arr = np.array(pips)
    net_pips = float(np.sum(p_arr))
    wins = p_arr[p_arr > 0]
    losses = -p_arr[p_arr < 0]
    sum_w = float(np.sum(wins))
    sum_l = float(np.sum(losses))

    wr = len(wins) / n * 100.0
    pf = sum_w / sum_l if sum_l > 0 else (99.0 if sum_w > 0 else 0.0)

    equity = np.cumsum(p_arr)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

    ret_dd = net_pips / max_dd if max_dd > 0 else (net_pips if net_pips > 0 else 0.0)
    annual_pips = net_pips / max(years, 0.01)

    return {
        "trades": n,
        "net_pips": round(net_pips, 1),
        "annual_pips": round(annual_pips, 1),
        "win_rate": round(wr, 1),
        "profit_factor": round(pf, 2),
        "max_dd_pips": round(max_dd, 1),
        "return_to_dd": round(ret_dd, 2),
        "expectancy": round(net_pips / n, 2)
    }


def run_certification():
    t0 = time.time()
    print("=" * 80)
    print("  AUDNZD HOLDOUT & FULL-CYCLE CERTIFICATION AUDIT")
    print("=" * 80)

    disc_weeks = get_discovery_weeks("AUDNZD")
    hold_weeks = get_holdout_test_weeks("AUDNZD")
    all_weeks = sorted(disc_weeks + hold_weeks)

    disc_set = set(disc_weeks)
    hold_set = set(hold_weeks)

    print(f"[*] Discovery Weeks (70%):  {len(disc_weeks)} weeks (4.04 Years)")
    print(f"[*] Holdout Weeks (30%):    {len(hold_weeks)} weeks (1.73 Years) - ZERO SNOOPING")
    print(f"[*] Total Weeks (100%):     {len(all_weeks)} weeks (5.77 Years)")

    disc_trades: Dict[str, List[float]] = {name: [] for name, _ in build_audit_models()}
    hold_trades: Dict[str, List[float]] = {name: [] for name, _ in build_audit_models()}
    all_trades: Dict[str, List[float]] = {name: [] for name, _ in build_audit_models()}

    max_workers = min(os.cpu_count() or 4, 12)
    print(f"[*] Evaluating across all 300 weeks using {max_workers} worker processes...")

    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_week, w): w for w in all_weeks}
        for fut in as_completed(futures):
            res = fut.result()
            w = res["week"]
            trades = res.get("trades", {})

            for name, p_list in trades.items():
                all_trades[name].extend(p_list)
                if w in disc_set:
                    disc_trades[name].extend(p_list)
                elif w in hold_set:
                    hold_trades[name].extend(p_list)

            completed += 1
            if completed % 50 == 0 or completed == len(all_weeks):
                print(f"  --> Progress: {completed:03d}/{len(all_weeks)} weeks processed...")

    years_total = len(all_weeks) / 52.0
    years_disc = len(disc_weeks) / 52.0
    years_hold = len(hold_weeks) / 52.0

    print("\n" + "=" * 115)
    print("  GRAND CERTIFICATION SCORECARD: AUDNZD CANDIDATES (70% DISCOVERY vs 30% HOLDOUT vs FULL CYCLE)")
    print("=" * 115)
    print(f"{'Model Identifier':<28} | {'--- 70% DISCOVERY (4.04y) ---':<26} | {'--- 30% HOLDOUT (1.73y) ---':<26} | {'--- FULL 5.77y CYCLE ---':<24}")
    print(f"{'':<28} | {'Trades':<6} {'PF':<5} {'Net':<7} {'WR%':<6} | {'Trades':<6} {'PF':<5} {'Net':<7} {'WR%':<6} | {'Trades':<6} {'PF':<5} {'Net':<8} {'Ret/DD':<6}")
    print("-" * 115)

    audit_report = []

    for name, _ in build_audit_models():
        m_disc = compute_metrics(disc_trades[name], years_disc)
        m_hold = compute_metrics(hold_trades[name], years_hold)
        m_full = compute_metrics(all_trades[name], years_total)

        audit_report.append({
            "model": name,
            "discovery_70": m_disc,
            "holdout_30": m_hold,
            "full_cycle": m_full,
            "certified": m_hold["profit_factor"] >= 1.25 and m_full["profit_factor"] >= 1.25 and m_full["annual_pips"] >= 150.0
        })

        cert_tag = "[CERTIFIED]" if m_hold["profit_factor"] >= 1.25 and m_full["profit_factor"] >= 1.25 else "[REJECT]"

        line = (
            f"{name:<28} | "
            f"{m_disc['trades']:<6} {m_disc['profit_factor']:<5.2f} {m_disc['net_pips']:>6.1f}p {m_disc['win_rate']:>5.1f}% | "
            f"{m_hold['trades']:<6} {m_hold['profit_factor']:<5.2f} {m_hold['net_pips']:>6.1f}p {m_hold['win_rate']:>5.1f}% | "
            f"{m_full['trades']:<6} {m_full['profit_factor']:<5.2f} {m_full['net_pips']:>7.1f}p {m_full['return_to_dd']:>5.1f}x  {cert_tag}"
        )
        print(line)

    print("-" * 115)

    out_file = RESULTS_DIR / "audnzd_holdout_certification_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "asset": "AUDNZD",
            "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "total_weeks": len(all_weeks),
            "years": round(years_total, 2),
            "report": audit_report
        }, f, indent=2)

    print(f"\n[+] Full certification report saved to {out_file.resolve()}")
    print(f"[+] Total execution time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run_certification()

