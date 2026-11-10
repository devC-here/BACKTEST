#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB AUDNZD V1: SYSTEMATIC CROSS-PAIR CANDIDATE SCREENING
================================================================================
Evaluates our proven champion architectures from EURUSD, EURGBP, and USDJPY
across the 70% Quarantined Discovery Partition of AUDNZD:
  - 210 Discovery Weeks (~97.08M authentic ticks)
  - Pip Size: 0.0001
  - Max Spread Accommodation: 2.8 pips (accommodating Dukascopy ~2.1-2.5p spread)
  - Realistic SL/TP calibration for AUDNZD's ~110p weekly range

Candidate Archetypes Tested:
  1. Mean-Reversion / Support-Resistance Bounces:
     - Camarilla Floor/Ceiling Fades (L3/H3)
     - Daily Pivot Wall Rejection Fades
     - Session VWAP 1-SD Band Rebound
  2. Multi-Timeframe Structural Trend Runners:
     - Macro Asymmetric Trend Runner (H1 Donchian + EMA + M15 Rejection, 2.5R)
     - Intraday Macro Retest (H1 Donchian + EMA + M15 Rejection, 1.7R)
     - Double Swing Neckline Retest (M15 Double Swings)
     - Double Structure Neckline Breakout (M15 W/M Patterns)
  3. Wick Absorption & Microstructure:
     - Continuous M5 Wick Rejection Absorption
  4. Volatility Squeeze & Momentum Expansion:
     - H1 Keltner Squeeze Explosion (TTM Squeeze Release)
     - Chaikin Volatility Surge (M15 Institutional Volatility)
     - Asian Compression Range Breakout (Sydney/Tokyo Range Break)
     - Opening Range Impulse Expansion (Session Ignition)
     - Cardwell RSI-50 Bull/Bear Momentum Bounce
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
from tick_engine.core.dataloader import get_week_dataset, get_discovery_weeks

# Import from certified 29 champions
from strategies.certified_29_champions.eurgbp import (
    EURGBPCamarillaFloorElite,
    EURGBPDailyPivotWallElite,
    EURGBPH1KeltnerSqueezeExplosionElite,
    EURGBPMacroAsymmetricRunnerElite,
    EURGBPIntradayMacroRetestElite,
    EURGBPAsianCompressionBreakoutElite,
    EURGBPVWAP1SDReboundElite,
)
from strategies.certified_29_champions.eurusd import (
    ContinuousM5RejectionElite,
    EURUSDDoubleSwingNecklineRetestElite,
    EURUSDChaikinVolatilitySurgeElite,
    EURUSDCardwellRSI50MomentumBounceElite,
    EURUSDEMACrossATRExpansionElite,
    EURUSDLondonVWAPPullbackElite,
)
from strategies.certified_29_champions.usdjpy import (
    USDJPYOpeningRangeImpulseExpansionElite,
    USDJPYDoubleStructureNecklineBreakoutElite,
    USDJPYCardwellRSI50MomentumBounceElite,
)

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_audnzd_candidates() -> List[Tuple[str, Any]]:
    """Builds full fleet of candidate models with AUDNZD-specific adaptations."""
    pip_size = 0.0001
    max_spread = 2.8  # AUDNZD Dukascopy tick spread averages 2.1-2.5 pips

    models = [
        # --- 1. Camarilla Floor / Ceiling Fades ---
        ("Camarilla_SL10_TP16", EURGBPCamarillaFloorElite(
            sl_pips=10.0, tp_pips=16.0, min_wick=0.45, max_dist_pips=4.0,
            max_spread=max_spread, pip_size=pip_size
        )),
        ("Camarilla_SL12_TP20", EURGBPCamarillaFloorElite(
            sl_pips=12.0, tp_pips=20.0, min_wick=0.45, max_dist_pips=4.5,
            max_spread=max_spread, pip_size=pip_size
        )),
        ("Camarilla_SL14_TP22", EURGBPCamarillaFloorElite(
            sl_pips=14.0, tp_pips=22.0, min_wick=0.40, max_dist_pips=5.0,
            max_spread=max_spread, pip_size=pip_size
        )),

        # --- 2. Daily Pivot Wall Fades ---
        ("DailyPivot_SL10_TP16", EURGBPDailyPivotWallElite(
            sl_pips=10.0, tp_pips=16.0, min_wick=0.45, max_dist_pips=4.5,
            max_spread=max_spread, pip_size=pip_size
        )),
        ("DailyPivot_SL12_TP20", EURGBPDailyPivotWallElite(
            sl_pips=12.0, tp_pips=20.0, min_wick=0.45, max_dist_pips=5.0,
            max_spread=max_spread, pip_size=pip_size
        )),
        ("DailyPivot_SL14_TP24", EURGBPDailyPivotWallElite(
            sl_pips=14.0, tp_pips=24.0, min_wick=0.40, max_dist_pips=5.5,
            max_spread=max_spread, pip_size=pip_size
        )),

        # --- 3. Session VWAP 1-SD Rebound ---
        ("VWAP_1SD_SL10_TP16", EURGBPVWAP1SDReboundElite(
            asset="AUDNZD", sl_pips=10.0, tp_pips=16.0,
            max_spread=max_spread, pip_size=pip_size
        )),
        ("VWAP_1SD_SL12_TP20", EURGBPVWAP1SDReboundElite(
            asset="AUDNZD", sl_pips=12.0, tp_pips=20.0,
            max_spread=max_spread, pip_size=pip_size
        )),

        # --- 4. Macro Asymmetric Trend Runner (2.5R) ---
        ("MacroRunner_SL10_TP25", EURGBPMacroAsymmetricRunnerElite(
            asset="AUDNZD", sl_pips=10.0, tp_pips=25.0,
            max_spread=max_spread, pip_size=pip_size
        )),
        ("MacroRunner_SL12_TP30", EURGBPMacroAsymmetricRunnerElite(
            asset="AUDNZD", sl_pips=12.0, tp_pips=30.0,
            max_spread=max_spread, pip_size=pip_size
        )),
        ("MacroRunner_SL14_TP35", EURGBPMacroAsymmetricRunnerElite(
            asset="AUDNZD", sl_pips=14.0, tp_pips=35.0,
            max_spread=max_spread, pip_size=pip_size
        )),

        # --- 5. Intraday Macro Retest (1.7R) ---
        ("MacroRetest_SL10_TP18", EURGBPIntradayMacroRetestElite(
            asset="AUDNZD", sl_pips=10.0, tp_pips=18.0,
            max_spread=max_spread, pip_size=pip_size
        )),
        ("MacroRetest_SL12_TP20", EURGBPIntradayMacroRetestElite(
            asset="AUDNZD", sl_pips=12.0, tp_pips=20.0,
            max_spread=max_spread, pip_size=pip_size
        )),
        ("MacroRetest_SL14_TP24", EURGBPIntradayMacroRetestElite(
            asset="AUDNZD", sl_pips=14.0, tp_pips=24.0,
            max_spread=max_spread, pip_size=pip_size
        )),

        # --- 6. Continuous M5 Wick Rejection Absorption ---
        ("ContinuousM5_SL09_TP14", ContinuousM5RejectionElite(
            sl_pips=9.0, tp_pips=14.0, min_wick=0.48, max_dist=5.5,
            max_spread=max_spread, pip_size=pip_size
        )),
        ("ContinuousM5_SL11_TP18", ContinuousM5RejectionElite(
            sl_pips=11.0, tp_pips=18.0, min_wick=0.45, max_dist=6.0,
            max_spread=max_spread, pip_size=pip_size
        )),

        # --- 7. Double Swing Neckline Retest (M15) ---
        ("DoubleSwing_SL10_TP18", EURUSDDoubleSwingNecklineRetestElite(
            asset="AUDNZD", sl_pips=10.0, tp_pips=18.0,
            max_spread=max_spread, pip_size=pip_size
        )),
        ("DoubleSwing_SL12_TP22", EURUSDDoubleSwingNecklineRetestElite(
            asset="AUDNZD", sl_pips=12.0, tp_pips=22.0,
            max_spread=max_spread, pip_size=pip_size
        )),

        # --- 8. Double Structure Breakout (M15 W/M Bottoms/Tops) ---
        ("DoubleStructure_SL11_TP22", USDJPYDoubleStructureNecklineBreakoutElite(
            sl_pips=11.0, tp_pips=22.0, max_spread=max_spread, pip_size=pip_size
        )),
        ("DoubleStructure_SL13_TP26", USDJPYDoubleStructureNecklineBreakoutElite(
            sl_pips=13.0, tp_pips=26.0, max_spread=max_spread, pip_size=pip_size
        )),

        # --- 9. H1 Keltner Squeeze Explosion (TTM Squeeze Release) ---
        ("H1Keltner_SL10_TP20", EURGBPH1KeltnerSqueezeExplosionElite(
            asset="AUDNZD", sl_pips=10.0, tp_pips=20.0,
            max_spread=max_spread, pip_size=pip_size
        )),
        ("H1Keltner_SL12_TP24", EURGBPH1KeltnerSqueezeExplosionElite(
            asset="AUDNZD", sl_pips=12.0, tp_pips=24.0,
            max_spread=max_spread, pip_size=pip_size
        )),

        # --- 10. Chaikin Volatility Surge (M15) ---
        ("Chaikin_SL10_TP20", EURUSDChaikinVolatilitySurgeElite(
            asset="AUDNZD", sl_pips=10.0, tp_pips=20.0,
            max_spread=max_spread, pip_size=pip_size
        )),
        ("Chaikin_SL12_TP24", EURUSDChaikinVolatilitySurgeElite(
            asset="AUDNZD", sl_pips=12.0, tp_pips=24.0,
            max_spread=max_spread, pip_size=pip_size
        )),

        # --- 11. Asian Compression Breakout ---
        ("AsianCompress_SL10_TP18", EURGBPAsianCompressionBreakoutElite(
            sl_pips=10.0, tp_pips=18.0, max_asia_pips=16.0,
            max_spread=max_spread, pip_size=pip_size
        )),
        ("AsianCompress_SL12_TP22", EURGBPAsianCompressionBreakoutElite(
            sl_pips=12.0, tp_pips=22.0, max_asia_pips=20.0,
            max_spread=max_spread, pip_size=pip_size
        )),

        # --- 12. Cardwell RSI-50 Momentum Bounce ---
        ("CardwellRSI50_SL10_TP20", EURUSDCardwellRSI50MomentumBounceElite(
            sl_pips=10.0, tp_pips=20.0, max_spread=max_spread, pip_size=pip_size
        )),
        ("CardwellRSI50_SL12_TP24", EURUSDCardwellRSI50MomentumBounceElite(
            sl_pips=12.0, tp_pips=24.0, max_spread=max_spread, pip_size=pip_size
        )),

        # --- 13. Opening Range Impulse Expansion ---
        ("OpenImpulse_SL11_TP22", USDJPYOpeningRangeImpulseExpansionElite(
            asset="AUDNZD", sl_pips=11.0, tp_pips=22.0,
            max_spread=max_spread, pip_size=pip_size
        )),
        ("OpenImpulse_SL13_TP26", USDJPYOpeningRangeImpulseExpansionElite(
            asset="AUDNZD", sl_pips=13.0, tp_pips=26.0,
            max_spread=max_spread, pip_size=pip_size
        )),

        # --- 14. EMA Cross ATR Expansion ---
        ("EMACrossATR_SL10_TP20", EURUSDEMACrossATRExpansionElite(
            sl_pips=10.0, tp_pips=20.0, max_spread=max_spread, pip_size=pip_size
        )),

        # --- 15. London / Morning VWAP Pullback ---
        ("VWAPPullback_SL10_TP20", EURUSDLondonVWAPPullbackElite(
            sl_pips=10.0, tp_pips=20.0, max_spread=max_spread, pip_size=pip_size
        )),
    ]
    return models


def evaluate_week_task(week_num: int) -> Dict[str, Any]:
    """Evaluates all candidate models against a single week of tick data."""
    engine = BacktestEngine(pip_size=0.0001, use_slippage=True)
    try:
        ticks, meta = get_week_dataset(week_num, raw_dir="data/AUDNZD")
        if len(ticks) == 0:
            return {"week": week_num, "trades": {}}
    except Exception as e:
        return {"week": week_num, "trades": {}, "error": str(e)}

    models = build_audnzd_candidates()
    week_trades = {}

    for model_name, model in models:
        try:
            trades = engine.run(ticks, model)
            pips_list = [float(t['pips']) for t in trades]
            week_trades[model_name] = pips_list
        except Exception:
            week_trades[model_name] = []

    return {"week": week_num, "trades": week_trades}


def run_discovery_screening():
    t_start = time.time()
    print("=" * 80)
    print("  ALPHA DISCOVERY LAB AUDNZD V1: 70% DISCOVERY SCREENING")
    print("=" * 80)

    disc_weeks = get_discovery_weeks("AUDNZD")
    total_disc_weeks = len(disc_weeks)
    print(f"[*] Total Quarantined Discovery Weeks: {total_disc_weeks}")
    print(f"[*] Total Candidate Architectures:      {len(build_audnzd_candidates())}")
    print(f"[*] Quarantined Holdout (30%):          Strictly Isolated & Preserved")

    # Aggregate trades per model
    models_trades: Dict[str, List[float]] = {name: [] for name, _ in build_audnzd_candidates()}
    weekly_pnl: Dict[str, Dict[int, float]] = {name: {} for name, _ in build_audnzd_candidates()}

    max_workers = min(os.cpu_count() or 4, 12)
    print(f"[*] Launching parallel evaluation across {max_workers} worker processes...")

    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_week_task, w): w for w in disc_weeks}
        for future in as_completed(futures):
            res = future.result()
            w = res["week"]
            trades_dict = res.get("trades", {})

            for model_name, pips_list in trades_dict.items():
                models_trades[model_name].extend(pips_list)
                weekly_pnl[model_name][w] = sum(pips_list)

            completed += 1
            if completed % 25 == 0 or completed == total_disc_weeks:
                elapsed = time.time() - t_start
                rate = completed / max(elapsed, 0.001)
                print(f"  --> Progress: {completed:03d}/{total_disc_weeks} discovery weeks processed ({rate:.1f} w/s)...")

    # Score and rank all models
    print("\n" + "=" * 80)
    print("  AUDNZD DISCOVERY RESULTS SCORECARD (70% PARTITION / 210 WEEKS)")
    print("=" * 80)

    scorecard = []
    # 210 weeks out of 300 total weeks = 210 / 52 = 4.038 years of discovery data
    years_equiv = total_disc_weeks / 52.0

    for model_name, pips in models_trades.items():
        n_trades = len(pips)
        if n_trades == 0:
            continue

        p_arr = np.array(pips)
        net_pips = float(np.sum(p_arr))
        wins = p_arr[p_arr > 0]
        losses = -p_arr[p_arr < 0]
        sum_wins = float(np.sum(wins))
        sum_losses = float(np.sum(losses))

        wr = len(wins) / n_trades * 100.0 if n_trades > 0 else 0.0
        pf = sum_wins / sum_losses if sum_losses > 0 else (99.0 if sum_wins > 0 else 0.0)

        # Max drawdown in pips
        equity = np.cumsum(p_arr)
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

        ret_dd = net_pips / max_dd if max_dd > 0 else (net_pips if net_pips > 0 else 0.0)
        annual_pips = net_pips / years_equiv

        scorecard.append({
            "model": model_name,
            "trades": n_trades,
            "win_rate": round(wr, 1),
            "profit_factor": round(pf, 2),
            "net_pips": round(net_pips, 1),
            "annual_pips": round(annual_pips, 1),
            "max_dd_pips": round(max_dd, 1),
            "return_to_dd": round(ret_dd, 2),
            "expectancy_pips": round(net_pips / n_trades, 2)
        })

    # Sort by Net Pips descending
    scorecard.sort(key=lambda x: x["net_pips"], reverse=True)

    header = f"{'Rank':<4} {'Model Identifier':<28} {'Trades':<7} {'WinRate':<8} {'PF':<6} {'Net Pips':<10} {'Pips/Yr':<10} {'Max DD':<9} {'Ret/DD':<8} {'Exp(p)':<7}"
    print(header)
    print("-" * len(header))

    profitable_count = 0
    elite_count = 0

    for i, row in enumerate(scorecard, 1):
        is_elite = row["profit_factor"] >= 1.25 and row["annual_pips"] >= 150.0 and row["return_to_dd"] >= 4.0
        is_prof = row["profit_factor"] >= 1.15 and row["net_pips"] > 0
        tag = "[ELITE]" if is_elite else ("[PASS]" if is_prof else "      ")

        if is_elite: elite_count += 1
        if is_prof: profitable_count += 1

        print(f"{i:<4} {row['model']:<28} {row['trades']:<7} {row['win_rate']:>5.1f}%  {row['profit_factor']:>5.2f} {row['net_pips']:>9.1f}p {row['annual_pips']:>9.1f}p {row['max_dd_pips']:>7.1f}p {row['return_to_dd']:>7.2f}x {row['expectancy_pips']:>6.2f}p  {tag}")

    print("-" * len(header))
    print(f"Summary: Evaluated {len(scorecard)} models across {total_disc_weeks} discovery weeks ({years_equiv:.2f} yrs).")
    print(f"Profitable Models (PF >= 1.15): {profitable_count} | Institutional Elite (PF >= 1.25, >=150p/yr, Ret/DD >= 4x): {elite_count}")

    # Save discovery results to JSON
    out_file = RESULTS_DIR / "audnzd_discovery_screening_v1.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "asset": "AUDNZD",
            "partition": "discovery_70",
            "total_weeks": total_disc_weeks,
            "years_equivalent": round(years_equiv, 2),
            "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "scorecard": scorecard
        }, f, indent=2)

    print(f"[+] Detailed discovery results saved to {out_file.resolve()}")
    print(f"[+] Total execution time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    run_discovery_screening()

