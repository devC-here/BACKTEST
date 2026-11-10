#!/usr/bin/env python3
"""
Stages and formally certifies Batch 4 champions:
- USDJPY: disc106, disc91, disc90, disc95, disc94, disc99, disc102, disc89, disc100
- EURGBP: disc107
- EURUSD: disc03, disc78
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

from tick_engine.core.dataloader import get_complete_split_weeks
from tick_engine.core.config import PIP_SIZES, get_friction_pips, WARMUP_TICKS
from tick_engine.reporting.run_manifest import create_certified_run_manifest, compute_statistical_metrics
from scripts.screen_pair_fleet import _eval_strat_week

FLEET_DIR = ROOT_DIR / "certified_production_fleet"
OUTPUT_DIR = ROOT_DIR / "results" / "certified_strategy_manifests"


def main():
    targets = [
        # USDJPY champions
        ("USDJPY", "disc106_chaikin_volatility_surge_elite", ROOT_DIR / "archive_deprecated" / "strategies" / "certified_29_champions" / "usdjpy" / "disc106_chaikin_volatility_surge_elite"),
        ("USDJPY", "disc91_session_donchian_midline_bounce_elite", ROOT_DIR / "archive_deprecated" / "strategies" / "certified_29_champions" / "usdjpy" / "disc91_session_donchian_midline_bounce_elite"),
        ("USDJPY", "disc90_opening_range_impulse_expansion_elite", ROOT_DIR / "archive_deprecated" / "strategies" / "certified_29_champions" / "usdjpy" / "disc90_opening_range_impulse_expansion_elite"),
        ("USDJPY", "disc95_camarilla_expansion_elite", ROOT_DIR / "archive_deprecated" / "strategies" / "certified_29_champions" / "usdjpy" / "disc95_camarilla_expansion_elite"),
        ("USDJPY", "disc94_london_vwap_pullback_elite", ROOT_DIR / "archive_deprecated" / "strategies" / "certified_29_champions" / "usdjpy" / "disc94_london_vwap_pullback_elite"),
        ("USDJPY", "disc99_m15_macd_zero_cross_momentum_elite", ROOT_DIR / "archive_deprecated" / "strategies" / "certified_29_champions" / "usdjpy" / "disc99_m15_macd_zero_cross_momentum_elite"),
        ("USDJPY", "disc102_keltner_midline_expansion_elite", ROOT_DIR / "archive_deprecated" / "strategies" / "certified_29_champions" / "usdjpy" / "disc102_keltner_midline_expansion_elite"),
        ("USDJPY", "disc89_ny_london_vwap_ignition_elite", ROOT_DIR / "archive_deprecated" / "strategies" / "certified_29_champions" / "usdjpy" / "disc89_ny_london_vwap_ignition_elite"),
        ("USDJPY", "disc100_double_structure_neckline_breakout_elite", ROOT_DIR / "archive_deprecated" / "strategies" / "certified_29_champions" / "usdjpy" / "disc100_double_structure_neckline_breakout_elite"),
        
        # EURGBP champion
        ("EURGBP", "disc107_h1_keltner_squeeze_explosion_elite", ROOT_DIR / "archive_deprecated" / "strategies" / "certified_29_champions" / "eurgbp" / "disc107_h1_keltner_squeeze_explosion_elite"),
        
        # EURUSD champions
        ("EURUSD", "disc03_ny_open_overlap_elite", ROOT_DIR / "archive_deprecated" / "strategies" / "eurusd_production_fleet" / "disc03_ny_open_overlap_elite"),
        ("EURUSD", "disc78_three_bar_inside_confirmation_elite", ROOT_DIR / "archive_deprecated" / "strategies" / "eurusd_production_fleet" / "disc78_three_bar_inside_confirmation_elite")
    ]
    
    all_to_certify = []
    for symbol, name, src in targets:
        subfolder = f"{symbol.lower()}_champions"
        target_dir = FLEET_DIR / subfolder
        target_dir.mkdir(parents=True, exist_ok=True)
        dst = target_dir / name
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)
        all_to_certify.append((name, symbol, dst / "strategy.py"))
        
    print("=" * 95)
    print(f"  FORMALLY CERTIFYING BATCH 4: {len(all_to_certify)} STRATEGIES (USDJPY, EURGBP, EURUSD)")
    print("=" * 95)
    
    with ProcessPoolExecutor(max_workers=4) as executor:
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

            hold_stats = compute_statistical_metrics(hold_trades, candidates_screened=25)
            disc_stats = compute_statistical_metrics(disc_trades, candidates_screened=25)

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
                candidates_screened=25,
                additional_meta={
                    "discovery_in_sample_metrics": disc_stats,
                    "evaluation_time_seconds": round(elapsed, 2),
                    "clean_weeks_holdout": len(hold_weeks),
                    "clean_weeks_discovery": len(disc_weeks),
                    "combined_net_pips": round(hold_stats["net_pips"] + disc_stats["net_pips"], 1)
                }
            )

            trades_file = OUTPUT_DIR / f"{symbol.lower()}_{name}_trades.csv"
            with open(manifest_file, "w") as f:
                json.dump(manifest, f, indent=2)

            if len(hold_trades) > 0:
                pd.DataFrame(manifest["trade_log"]).to_csv(trades_file, index=False)

            print(f"[{idx:02d}/{len(all_to_certify)}] {symbol} {name[:34]:34s} | Hold: {hold_stats['net_pips']:>+7.1f}p (PF {hold_stats['profit_factor']:>4.2f}, Tr {hold_stats['total_trades']:>3}) | Disc: {disc_stats['net_pips']:>+7.1f}p (PF {disc_stats['profit_factor']:>4.2f}) | Comb: {manifest['custom_metadata']['combined_net_pips']:>+8.1f}p | {manifest['certification_status']} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()

