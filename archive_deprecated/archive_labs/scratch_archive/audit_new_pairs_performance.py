#!/usr/bin/env python3
"""
================================================================================
  CROSS-ASSET AUDIT: TESTING MODELS ON NEW PAIRS (USDJPY & 5-YEAR EURGBP)
================================================================================
Audits all production models across:
  1. USDJPY: 300 continuous weeks (2021–2026, ~192.32M ticks)
  2. EURGBP: 300 continuous weeks (2021–2026, ~85.81M ticks - upgraded dataset)

Evaluates:
  - Total Trades, Win Rate, Profit Factor, Net Pips, Max Drawdown, Ret/DD
  - 70% Discovery vs 30% Quarantined Holdout performance
  - Friction and expectancy threshold
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
from tick_engine.core.dataloader import get_week_dataset, load_manifest

# EURUSD Fleet Models
from strategies.eurusd_production_fleet import (
    MacroStructureH4M30Elite,
    ContinuousM5RejectionElite,
    RSIRegimeRangeShiftElite,
    IntradayPrecisionH1M5Elite,
    S66Bhanushali44MAElite,
    HighFrequencyM15ScalperElite,
    S10DualEMAPullbackElite,
    S14InsideBarM30Elite,
    NewYorkOpenOverlapBreakoutElite,
    ThursdayMacroTrendIgnitionElite,
    LondonMiddayMACDAccelerationElite,
    StochasticDeepPullbackElite,
    BollingerBandwidthExpansionElite,
    BollingerSupertrendHybridElite,
)

# EURGBP Fleet Models
from strategies.eurgbp_fleet import (
    EURGBPDailyPivotWallElite,
    EURGBPCamarillaFloorElite,
    EURGBPWeeklyPivotWallElite,
)

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_models_for_asset(asset: str) -> List[Tuple[str, Any]]:
    """Builds instantiated strategies adapted for the asset's pip size and typical spread."""
    if asset == "USDJPY":
        pip_size = 0.01
        max_spread = 1.0  # Dukascopy USDJPY spread is ~0.25 to 0.50 pips
    else:  # EURGBP
        pip_size = 0.0001
        max_spread = 2.0  # Dukascopy EURGBP spread is ~1.0 to 1.5 pips

    models = [
        # EURGBP Champions
        ("DISC46_Daily_Pivot", EURGBPDailyPivotWallElite(pip_size=pip_size, max_spread=max_spread)),
        ("DISC48_Camarilla", EURGBPCamarillaFloorElite(pip_size=pip_size, max_spread=max_spread)),
        ("DISC47_Weekly_Pivot", EURGBPWeeklyPivotWallElite(pip_size=pip_size, max_spread=max_spread)),
        # Top Universal Champions
        ("DISC42_H4_M30", MacroStructureH4M30Elite(pip_size=pip_size, max_spread=max_spread)),
        ("DISC45_M5", ContinuousM5RejectionElite(pip_size=pip_size, max_spread=max_spread)),
        ("DISC40_RSI", RSIRegimeRangeShiftElite(pip_size=pip_size, max_spread_pips=max_spread)),
        ("DISC43_H1_M5", IntradayPrecisionH1M5Elite(pip_size=pip_size, max_spread=max_spread)),
        ("DISC03_NY", NewYorkOpenOverlapBreakoutElite()),
        ("DISC41_THURSDAY", ThursdayMacroTrendIgnitionElite(pip_size=pip_size, max_spread_pips=max_spread)),
        ("DISC38_MACD", LondonMiddayMACDAccelerationElite(pip_size=pip_size, max_spread_pips=max_spread)),
        ("DISC39_STOCH", StochasticDeepPullbackElite(pip_size=pip_size, max_spread_pips=max_spread)),
        ("COMB12_BB", BollingerBandwidthExpansionElite()),
        ("COMB27_SUPERTREND", BollingerSupertrendHybridElite()),
        ("HF_M15", HighFrequencyM15ScalperElite()),
        ("S10_EMA", S10DualEMAPullbackElite()),
        ("S66_44MA", S66Bhanushali44MAElite()),
        ("S14_IB", S14InsideBarM30Elite()),
    ]
    return models


def evaluate_week_task(args: Tuple[str, int]) -> Dict[str, Any]:
    asset, week_num = args
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)
    
    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if len(ticks) == 0:
        return {"asset": asset, "week": week_num, "trades": {}}

    models = build_models_for_asset(asset)
    week_trades = {}

    for model_name, model in models:
        try:
            raw_trades = engine.run(ticks, model)
            t_list = []
            for i in range(len(raw_trades)):
                row = raw_trades[i]
                t_list.append({
                    "strategy": model_name,
                    "asset": asset,
                    "week": week_num,
                    "entry_time": int(row["entry_time"]),
                    "exit_time": int(row["exit_time"]),
                    "direction": int(row["direction"]),
                    "pips": float(row["pips"]),
                    "exit_reason": int(row["exit_reason"]),
                })
            week_trades[model_name] = t_list
        except Exception as e:
            week_trades[model_name] = []

    return {
        "asset": asset,
        "week": week_num,
        "ticks": len(ticks),
        "trades": week_trades
    }


def compute_metrics(trades: List[Dict[str, Any]], discovery_weeks_set: set) -> Dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "net_pips": 0.0,
            "max_dd_pips": 0.0,
            "ret_dd_ratio": 0.0,
            "expectancy_pips": 0.0,
            "disc_pips": 0.0,
            "hold_pips": 0.0,
        }

    pips_arr = np.array([t["pips"] for t in trades])
    net_pips = float(np.sum(pips_arr))
    wins = pips_arr[pips_arr > 0]
    losses = -pips_arr[pips_arr < 0]
    
    wr = float(len(wins) / n * 100.0)
    sum_w = float(np.sum(wins))
    sum_l = float(np.sum(losses))
    pf = round(sum_w / sum_l, 2) if sum_l > 0 else (99.0 if sum_w > 0 else 0.0)
    
    # Drawdown
    equity = np.cumsum(pips_arr)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
    ret_dd = round(net_pips / max_dd, 2) if max_dd > 0 else 0.0
    expectancy = round(net_pips / n, 2)

    # 70/30 split breakdown
    disc_pips = float(sum(t["pips"] for t in trades if t["week"] in discovery_weeks_set))
    hold_pips = float(sum(t["pips"] for t in trades if t["week"] not in discovery_weeks_set))

    return {
        "total_trades": n,
        "win_rate_pct": round(wr, 1),
        "profit_factor": pf,
        "net_pips": round(net_pips, 1),
        "max_dd_pips": round(max_dd, 1),
        "ret_dd_ratio": ret_dd,
        "expectancy_pips": expectancy,
        "disc_pips": round(disc_pips, 1),
        "hold_pips": round(hold_pips, 1),
    }


def run_asset_audit(asset: str, max_workers: int = 4):
    t_start = time.time()
    manifest = load_manifest(f"data/{asset}")
    weeks = sorted(list(manifest.keys()))
    
    split_file = ROOT_DIR / "data" / asset / "split_70_30.json"
    disc_set = set()
    if split_file.exists():
        with open(split_file, "r") as f:
            s_data = json.load(f)
            disc_set = set(s_data["discovery_70"]["weeks"])

    print("\n" + "=" * 80)
    print(f"  STARTING MULTI-THREADED AUDIT FOR {asset} ({len(weeks)} WEEKS)")
    print("=" * 80)

    all_trades_by_model: Dict[str, List[Dict[str, Any]]] = {}
    total_weeks_processed = 0

    tasks = [(asset, w) for w in weeks]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_week_task, task): task for task in tasks}
        for future in as_completed(futures):
            res = future.result()
            total_weeks_processed += 1
            w_trades = res["trades"]
            for m_name, t_list in w_trades.items():
                if m_name not in all_trades_by_model:
                    all_trades_by_model[m_name] = []
                all_trades_by_model[m_name].extend(t_list)

            if total_weeks_processed % 50 == 0 or total_weeks_processed == len(weeks):
                print(f"  [+] Completed {total_weeks_processed:3d}/{len(weeks)} weeks in {time.time() - t_start:.1f}s...")

    # Sort trades chronologically
    for m in all_trades_by_model:
        all_trades_by_model[m].sort(key=lambda x: x["entry_time"])

    # Compute scorecards
    rows = []
    for m_name, t_list in all_trades_by_model.items():
        m_stats = compute_metrics(t_list, disc_set)
        row = {
            "Asset": asset,
            "Strategy": m_name,
            "Trades": m_stats["total_trades"],
            "Win_Rate_Pct": m_stats["win_rate_pct"],
            "Profit_Factor": m_stats["profit_factor"],
            "Net_Pips": m_stats["net_pips"],
            "Max_DD_Pips": m_stats["max_dd_pips"],
            "Ret_DD_Ratio": m_stats["ret_dd_ratio"],
            "Expectancy_pips": m_stats["expectancy_pips"],
            "Disc_70_Pips": m_stats["disc_pips"],
            "Hold_30_Pips": m_stats["hold_pips"],
            "Status": "PROFITABLE EDGE" if m_stats["net_pips"] > 0 and m_stats["profit_factor"] >= 1.15 else ("SLIGHT PROFIT" if m_stats["net_pips"] > 0 else "NEGATIVE")
        }
        rows.append(row)

    df_summary = pd.DataFrame(rows).sort_values(by="Net_Pips", ascending=False)
    csv_out = RESULTS_DIR / f"{asset.lower()}_5y_audit_summary.csv"
    df_summary.to_csv(csv_out, index=False)
    print(f"\n[+] Saved {asset} summary scorecard to: {csv_out.resolve()}")

    # Save detailed master trades JSON
    json_out = RESULTS_DIR / f"master_{asset.lower()}_trades.json"
    with open(json_out, "w") as f:
        json.dump(all_trades_by_model, f, indent=2)
    print(f"[+] Saved {asset} master trades to: {json_out.resolve()}")

    return df_summary


def main():
    t_global = time.time()
    # 1. Audit USDJPY (300 weeks)
    df_usdjpy = run_asset_audit("USDJPY", max_workers=6)
    
    # 2. Audit EURGBP (300 weeks)
    df_eurgbp = run_asset_audit("EURGBP", max_workers=6)

    print("\n" + "=" * 80)
    print(f"  ALL ASSETS AUDITED IN {time.time() - t_global:.1f}s TOTAL!")
    print("=" * 80)


if __name__ == "__main__":
    main()
