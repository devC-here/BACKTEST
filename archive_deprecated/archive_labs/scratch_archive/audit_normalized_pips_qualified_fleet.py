#!/usr/bin/env python3
"""
================================================================================
  AUDIT QUALIFIED FLEET: EURUSD-EQUIVALENT NORMALIZED PIPS CONVERSION
================================================================================
Converts all raw pips into EURUSD-Equivalent Normalized Pips:
  - 1 Pip in EURUSD = $10.00 / lot -> Multiplier = 1.0000x
  - 1 Pip in USDJPY = $7.19 / lot  -> Multiplier = 0.7193x (100 / USDJPY_price)
  - 1 Pip in EURGBP = $12.92 / lot -> Multiplier = 1.2921x (EURUSD / EURGBP)

Calculates the exact normalized profit, expectancy, and ranking of all qualified
strategies across the 5-year cycle (2021 - 2026).
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
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
from tick_engine.core.dataloader import get_week_dataset
import strategies.qualified_strategies as qs

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 5-Year Benchmark Multipliers for EURUSD-Equivalence
NORM_FACTORS = {
    "EURUSD": 1.0000,
    "USDJPY": 0.7193,
    "EURGBP": 1.2921,
}


def run_strategy_asset_week(args) -> List[Dict[str, Any]]:
    asset, strat_class_name, week_num = args
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    try:
        ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
        if ticks is None or len(ticks) < 500:
            return []

        # Find strategy class in qs subpackages
        strat_cls = None
        for subpkg in [qs.all_pairs, qs.eurusd, qs.usdjpy, qs.eurgbp]:
            if hasattr(subpkg, strat_class_name):
                strat_cls = getattr(subpkg, strat_class_name)
                break

        if strat_cls is None:
            return []

        strat = strat_cls()
        raw_trades = engine.run(ticks, strat)
        t_list = []
        for i in range(len(raw_trades)):
            row = raw_trades[i]
            t_list.append({
                "asset": asset,
                "strategy": strat_class_name,
                "week": week_num,
                "raw_pips": float(row["pips"]),
                "entry_time": int(row["entry_time"]),
            })
        return t_list
    except Exception:
        return []


def main():
    print("=" * 90)
    print("  AUDITING ALL QUALIFIED STRATEGIES UNDER EURUSD-EQUIVALENT NORMALIZATION")
    print("=" * 90)

    # Compile list of (asset, strategy_class) to audit
    targets = []
    # All pairs champions across all 3 assets
    for sname in qs.all_pairs.__all__:
        for a in ["EURUSD", "USDJPY", "EURGBP"]:
            targets.append((a, sname))

    # EURUSD specialists
    for sname in qs.eurusd.__all__:
        targets.append(("EURUSD", sname))

    # USDJPY specialists
    for sname in qs.usdjpy.__all__:
        targets.append(("USDJPY", sname))

    # EURGBP specialists
    for sname in qs.eurgbp.__all__:
        targets.append(("EURGBP", sname))

    # Deduplicate targets
    targets = list(set(targets))
    print(f"Total Unique Strategy-Asset Combinations: {len(targets)}")

    # Build backtest tasks across all 300 weeks
    tasks = []
    for a, sname in targets:
        for w in range(1, 301):
            tasks.append((a, sname, w))

    print(f"Total Tasks to Execute: {len(tasks):,} across 300 weeks...")
    t0 = time.time()

    all_trades = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_strategy_asset_week, t) for t in tasks]
        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            res = fut.result()
            all_trades.extend(res)
            done += 1
            if done % 1000 == 0 or done == total:
                el = time.time() - t0
                print(f"  Progress: [{done}/{total}] tasks ({done/total*100:.1f}%) in {el:.1f}s")

    el = time.time() - t0
    print(f"\nExecution finished in {el:.1f}s! Total trades captured: {len(all_trades):,}")

    if not all_trades:
        print("No trades found.")
        return

    df_trades = pd.DataFrame(all_trades)
    df_trades["norm_factor"] = df_trades["asset"].map(NORM_FACTORS)
    df_trades["norm_pips"] = df_trades["raw_pips"] * df_trades["norm_factor"]
    df_trades["usd_cash_1lot"] = df_trades["norm_pips"] * 10.0  # $10/pip on EURUSD-equivalent standard lot

    summary_rows = []
    for (asset, sname), grp in df_trades.groupby(["asset", "strategy"]):
        raw_p = grp["raw_pips"].values
        norm_p = grp["norm_pips"].values
        n = len(raw_p)

        wins_raw = raw_p[raw_p > 0]
        losses_raw = raw_p[raw_p < 0]
        wr = len(wins_raw) / n * 100.0 if n > 0 else 0.0
        gw_raw = float(np.sum(wins_raw)) if len(wins_raw) > 0 else 0.0
        gl_raw = float(abs(np.sum(losses_raw))) if len(losses_raw) > 0 else 0.0
        pf = gw_raw / gl_raw if gl_raw > 0 else 999.0

        net_raw = float(np.sum(raw_p))
        net_norm = float(np.sum(norm_p))
        usd_cash = net_norm * 10.0

        # Drawdowns in normalized pips
        eq_norm = np.cumsum(norm_p)
        pk_norm = np.maximum.accumulate(eq_norm)
        dd_norm = pk_norm - eq_norm
        max_dd_norm = float(np.max(dd_norm)) if len(dd_norm) > 0 else 0.0

        # Drawdown in raw pips
        eq_raw = np.cumsum(raw_p)
        pk_raw = np.maximum.accumulate(eq_raw)
        dd_raw = pk_raw - eq_raw
        max_dd_raw = float(np.max(dd_raw)) if len(dd_raw) > 0 else 0.0

        ret_dd_norm = net_norm / max_dd_norm if max_dd_norm > 0 else 999.0
        exp_norm = net_norm / n if n > 0 else 0.0

        summary_rows.append({
            "asset": asset,
            "strategy": sname,
            "trades": n,
            "win_rate": round(wr, 1),
            "pf": round(pf, 2),
            "raw_pips": round(net_raw, 1),
            "norm_pips": round(net_norm, 1),
            "exp_norm": round(exp_norm, 2),
            "raw_max_dd": round(max_dd_raw, 1),
            "norm_max_dd": round(max_dd_norm, 1),
            "ret_dd_norm": round(ret_dd_norm, 2),
            "usd_profit_1lot": round(usd_cash, 2),
            "annual_norm_pips": round(net_norm / 5.0, 1),
            "annual_usd_cash": round(usd_cash / 5.0, 2),
        })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.sort_values(by="norm_pips", ascending=False, inplace=True)

    print("\n" + "=" * 115)
    print(f"  {'Asset':<8} {'Strategy Class':<38} {'Trades':<7} {'PF':<6} {'Raw Pips':<12} {'Norm Pips':<12} {'Norm DD':<10} {'USD Profit (1 Lot)':<18}")
    print("=" * 115)
    for _, r in df_sum.iterrows():
        print(f"  {r['asset']:<8} {r['strategy']:<38} {r['trades']:<7d} {r['pf']:<6.2f} +{r['raw_pips']:<10.1f}p +{r['norm_pips']:<10.1f}p {r['norm_max_dd']:<8.1f}p ${r['usd_profit_1lot']:<16,.2f}")

    # Totals by Asset
    print("\n" + "=" * 90)
    print("  PORTFOLIO TOTALS UNDER EURUSD-EQUIVALENT NORMALIZATION")
    print("=" * 90)
    for a, agrp in df_sum.groupby("asset"):
        tot_raw = agrp["raw_pips"].sum()
        tot_norm = agrp["norm_pips"].sum()
        tot_usd = agrp["usd_profit_1lot"].sum()
        ann_norm = tot_norm / 5.0
        ann_usd = tot_usd / 5.0
        print(f"  {a:<8} Sub-Fleet: Raw: +{tot_raw:<9.1f}p | Norm: +{tot_norm:<9.1f}p | 5Y Cash: ${tot_usd:<12,.2f} | Annual Cash: ${ann_usd:<10,.2f}/yr")

    grand_raw = df_sum["raw_pips"].sum()
    grand_norm = df_sum["norm_pips"].sum()
    grand_usd = df_sum["usd_profit_1lot"].sum()
    print("-" * 90)
    print(f"  GRAND FLEET TOTAL: Raw: +{grand_raw:<9.1f}p | Norm: +{grand_norm:<9.1f}p | 5Y Cash: ${grand_usd:<12,.2f} | Annual Cash: ${grand_usd/5.0:<10,.2f}/yr")

    csv_path = RESULTS_DIR / "qualified_fleet_eurusd_normalized_audit.csv"
    df_sum.to_csv(csv_path, index=False)
    print(f"\nSaved Full Normalized Scorecard to: {csv_path}")


if __name__ == "__main__":
    main()
