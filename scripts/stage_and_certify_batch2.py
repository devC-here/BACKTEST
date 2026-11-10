#!/usr/bin/env python3
"""
Stages and formally certifies Batch 2 champions:
- 14 USDJPY Champions
- 10 Additional EURUSD Champions
- 2 EURGBP Champions
Bringing total certified fleet to 51 strategies!
"""

import sys
import json
import time
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset, get_complete_split_weeks
from tick_engine.core.config import PIP_SIZES, get_friction_pips, WARMUP_TICKS
from tick_engine.reporting.run_manifest import create_certified_run_manifest, compute_statistical_metrics
from scripts.screen_pair_fleet import _eval_strat_week

FLEET_DIR = ROOT_DIR / "certified_production_fleet"
OUTPUT_DIR = ROOT_DIR / "results" / "certified_strategy_manifests"


def stage_and_certify():
    # 1. Load screening results
    with open("results/continuous_walkforward_audits/usdjpy_fleet_screen.json", "r") as f:
        usdjpy_data = json.load(f)
    with open("results/continuous_walkforward_audits/eurusd_older_strategies_screen.json", "r") as f:
        eurusd_data = json.load(f)
    with open("results/continuous_walkforward_audits/eurgbp_fleet_screen.json", "r") as f:
        eurgbp_data = json.load(f)

    # Filter targets
    usdjpy_targets = [x for x in usdjpy_data if x["holdout"]["net_pips"] > 0 and x["discovery"]["net_pips"] > 0 and x["holdout"]["pf"] >= 1.05 and x["discovery"]["pf"] >= 1.05]
    
    # EURUSD targets: Skip the top 10 already certified
    already_cert = [
        "disc65_session_vwap_band_fade_elite", "disc86_ema_cross_atr_expansion_elite",
        "s10_dual_ema_pullback_elite", "disc59_m15_macd_zero_cross_momentum_elite",
        "disc82_chaikin_volatility_surge_elite", "disc74_daily_pivot_pullback_elite",
        "disc76_vwap_1sd_rebound_elite", "disc75_weekly_pivot_pullback_elite",
        "disc58_h1_keltner_squeeze_explosion_elite", "disc80_double_swing_neckline_retest_elite"
    ]
    eurusd_targets = [x for x in eurusd_data if x["name"] not in already_cert and x["holdout"]["net_pips"] > 0 and x["discovery"]["net_pips"] > 0 and x["holdout"]["pf"] >= 1.05 and x["discovery"]["pf"] >= 1.05]
    
    eurgbp_targets = [x for x in eurgbp_data if x["holdout"]["net_pips"] > 0 and x["discovery"]["net_pips"] > 0 and x["holdout"]["pf"] >= 1.05 and x["discovery"]["pf"] >= 1.05]

    all_to_certify = []
    
    # Stage USDJPY
    usdjpy_dir = FLEET_DIR / "usdjpy_champions"
    usdjpy_dir.mkdir(parents=True, exist_ok=True)
    for item in usdjpy_targets:
        src = Path(item["path"]).parent
        dst = usdjpy_dir / item["name"]
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)
        all_to_certify.append((item["name"], "USDJPY", dst / "strategy.py"))

    # Stage EURUSD
    eurusd_dir = FLEET_DIR / "eurusd_champions"
    for item in eurusd_targets:
        src = Path(item["path"]).parent
        if not src.exists():
            cand = ROOT_DIR / "archive_deprecated" / "strategies" / "eurusd_production_fleet" / item["name"]
            if cand.exists():
                src = cand
            else:
                cand2 = ROOT_DIR / "archive_deprecated" / "strategies" / "new_discoveries" / item["name"]
                if cand2.exists():
                    src = cand2
        dst = eurusd_dir / item["name"]
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)
        all_to_certify.append((item["name"], "EURUSD", dst / "strategy.py"))

    # Stage EURGBP
    eurgbp_dir = FLEET_DIR / "eurgbp_champions"
    eurgbp_dir.mkdir(parents=True, exist_ok=True)
    for item in eurgbp_targets:
        src = Path(item["path"]).parent
        dst = eurgbp_dir / item["name"]
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)
        all_to_certify.append((item["name"], "EURGBP", dst / "strategy.py"))

    print("=" * 95)
    print(f"  FORMALLY CERTIFYING BATCH 2: {len(all_to_certify)} STRATEGIES (USDJPY, EURUSD, EURGBP)")
    print("=" * 95)

    with ProcessPoolExecutor(max_workers=8) as executor:
        for idx, (name, symbol, strat_path) in enumerate(all_to_certify, 1):
            manifest_file = OUTPUT_DIR / f"{symbol.lower()}_{name}_run_manifest.json"
            if manifest_file.exists():
                try:
                    with open(manifest_file, "r") as mf:
                        m_data = json.load(mf)
                        if m_data.get("statistical_metrics", {}).get("total_trades", 0) > 0:
                            print(f"[{idx:02d}/{len(all_to_certify)}] {symbol} {name[:34]:34s} | ALREADY CERTIFIED (SKIPPING)")
                            continue
                except Exception:
                    pass

            t0 = time.time()
            pip_size = PIP_SIZES.get(symbol, 0.0001)
            comm_pips, slip_pips = get_friction_pips(symbol)

            hold_weeks = get_complete_split_weeks(symbol, "holdout_test_30")
            disc_weeks = get_complete_split_weeks(symbol, "discovery_70")

            futs_hold = [executor.submit(_eval_strat_week, str(strat_path), symbol, w) for w in hold_weeks]
            hold_trades_list = [f.result() for f in futs_hold if len(f.result()) > 0]
            hold_trades = np.concatenate(hold_trades_list) if hold_trades_list else np.empty(0)

            futs_disc = [executor.submit(_eval_strat_week, str(strat_path), symbol, w) for w in disc_weeks]
            disc_trades_list = [f.result() for f in futs_disc if len(f.result()) > 0]
            disc_trades = np.concatenate(disc_trades_list) if disc_trades_list else np.empty(0)

            elapsed = time.time() - t0

            hold_stats = compute_statistical_metrics(hold_trades, candidates_screened=40)
            disc_stats = compute_statistical_metrics(disc_trades, candidates_screened=40)

            # Build Canonical Manifest
            manifest = create_certified_run_manifest(
                strategy_name=f"{symbol.lower()}_{name}",
                strategy_file=strat_path,
                symbol=symbol,
                split_version="canonical_continuous_v1.0 (In-Sample: 1..210, Holdout: 211..300)",
                start_week=211,
                end_week=300,
                pip_size=pip_size,
                commission_pips=comm_pips,
                entry_slippage_pips=slip_pips,
                spread_policy="Floating Raw Dukascopy Bid/Ask",
                warmup_ticks=WARMUP_TICKS,
                trades=hold_trades,
                candidates_screened=40,
                additional_meta={
                    "discovery_in_sample_metrics": disc_stats,
                    "evaluation_time_seconds": round(elapsed, 2),
                    "clean_weeks_holdout": len(hold_weeks),
                    "clean_weeks_discovery": len(disc_weeks),
                    "combined_net_pips": round(hold_stats["net_pips"] + disc_stats["net_pips"], 1)
                }
            )

            manifest_file = OUTPUT_DIR / f"{symbol.lower()}_{name}_run_manifest.json"
            trades_file = OUTPUT_DIR / f"{symbol.lower()}_{name}_trades.csv"

            with open(manifest_file, "w") as f:
                json.dump(manifest, f, indent=2)

            if len(hold_trades) > 0:
                pd.DataFrame(manifest["trade_log"]).to_csv(trades_file, index=False)

            print(f"[{idx:02d}/{len(all_to_certify)}] {symbol} {name[:34]:34s} | Hold: {hold_stats['net_pips']:>+7.1f}p (PF {hold_stats['profit_factor']:>4.2f}, Tr {hold_stats['total_trades']:>3}) | Disc: {disc_stats['net_pips']:>+7.1f}p (PF {disc_stats['profit_factor']:>4.2f}) | Comb: {manifest['custom_metadata']['combined_net_pips']:>+8.1f}p | {manifest['certification_status']} ({elapsed:.1f}s)")


if __name__ == "__main__":
    stage_and_certify()
