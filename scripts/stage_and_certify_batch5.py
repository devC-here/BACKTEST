#!/usr/bin/env python3
"""
Stages and formally certifies Batch 5 champions to reach the 72-strategy milestone:
1. EURUSD: high_frequency_m15_scalper_elite
2. EURGBP: disc108_macro_asymmetric_runner_elite
3. GBPCHF: strategy_26_gbpchf_asian_sweep
4. GBPCHF: strategy_27_gbpchf_asian_sweep
5. GBPCHF: strategy_29_gbpchf_asian_sweep
6. USDJPY: disc96_cardwell_rsi50_momentum_bounce_elite
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
        ("EURUSD", "high_frequency_m15_scalper_elite", ROOT_DIR / "archive_deprecated" / "strategies" / "eurusd_production_fleet" / "high_frequency_m15_scalper_elite"),
        ("EURGBP", "disc108_macro_asymmetric_runner_elite", ROOT_DIR / "archive_deprecated" / "strategies" / "certified_29_champions" / "eurgbp" / "disc108_macro_asymmetric_runner_elite"),
        ("GBPCHF", "strategy_26_gbpchf_asian_sweep", ROOT_DIR / "archive_deprecated" / "v 4.5" / "strategy_26_gbpchf_asian_sweep.py"),
        ("GBPCHF", "strategy_27_gbpchf_asian_sweep", ROOT_DIR / "archive_deprecated" / "v 4.5" / "strategy_27_gbpchf_asian_sweep.py"),
        ("GBPCHF", "strategy_29_gbpchf_asian_sweep", ROOT_DIR / "archive_deprecated" / "v 4.5" / "strategy_29_gbpchf_asian_sweep.py"),
        ("USDJPY", "disc96_cardwell_rsi50_momentum_bounce_elite", ROOT_DIR / "archive_deprecated" / "strategies" / "certified_29_champions" / "usdjpy" / "disc96_cardwell_rsi50_momentum_bounce_elite"),
    ]
    
    all_to_certify = []
    for symbol, name, src in targets:
        subfolder = f"{symbol.lower()}_champions"
        target_dir = FLEET_DIR / subfolder
        target_dir.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            dst = target_dir / name
            dst.mkdir(parents=True, exist_ok=True)
            strat_dst = dst / "strategy.py"
            if not strat_dst.exists():
                shutil.copyfile(src, strat_dst)
            all_to_certify.append((name, symbol, strat_dst))
        else:
            dst = target_dir / name
            if src.exists() and not dst.exists():
                shutil.copytree(src, dst)
            all_to_certify.append((name, symbol, dst / "strategy.py"))
            
    print("=" * 95)
    print(f"  FORMALLY CERTIFYING BATCH 5: {len(all_to_certify)} STRATEGIES (TARGET: 72 TOTAL)")
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

            hold_stats = compute_statistical_metrics(hold_trades, candidates_screened=30)
            disc_stats = compute_statistical_metrics(disc_trades, candidates_screened=30)

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
                candidates_screened=30,
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

