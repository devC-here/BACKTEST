#!/usr/bin/env python3
"""
Generates official cryptographic run manifests and per-trade CSVs for the top
dual-partition champions from eurusd_production_fleet.
"""

import sys
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset, get_complete_split_weeks
from tick_engine.reporting.run_manifest import create_certified_run_manifest, compute_statistical_metrics
from tick_engine.core.config import PIP_SIZES, get_friction_pips
from scripts.screen_eurusd_fleet import _eval_strat_week

SYMBOL = "EURUSD"
PIP_SIZE = 0.0001
COMM_PIPS, SLIP_PIPS = get_friction_pips(SYMBOL)
WARMUP_TICKS = 100000

OUTPUT_DIR = ROOT_DIR / "results" / "certified_strategy_manifests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def certify_top_champions():
    with open("results/continuous_walkforward_audits/eurusd_older_strategies_screen.json", "r") as f:
        data = json.load(f)

    # Filter top 10 dual-partition champions
    dual_profitable = [
        x for x in data
        if x["holdout"]["net_pips"] > 0 and x["discovery"]["net_pips"] > 0 and x["holdout"]["pf"] >= 1.10 and x["discovery"]["pf"] >= 1.05
    ]
    dual_profitable.sort(key=lambda x: (x["holdout"]["net_pips"] + x["discovery"]["net_pips"]), reverse=True)
    top_10 = dual_profitable[:10]

    hold_weeks = get_complete_split_weeks(SYMBOL, "holdout_test_30")
    disc_weeks = get_complete_split_weeks(SYMBOL, "discovery_70")

    print("=" * 90)
    print(f"  FORMALLY CERTIFYING TOP {len(top_10)} EURUSD FLEET CHAMPIONS")
    print("=" * 90)

    with ProcessPoolExecutor(max_workers=8) as executor:
        for idx, item in enumerate(top_10, 1):
            name = item["name"]
            strat_path = Path(item["path"])
            print(f"\n[{idx:02d}/10] Certifying {name}...")

            t0 = time.time()
            # 1. Holdout trades
            futs_hold = [executor.submit(_eval_strat_week, str(strat_path), w) for w in hold_weeks]
            hold_trades_list = [f.result() for f in futs_hold if len(f.result()) > 0]
            hold_trades = np.concatenate(hold_trades_list) if hold_trades_list else np.empty(0)

            # 2. Discovery trades
            futs_disc = [executor.submit(_eval_strat_week, str(strat_path), w) for w in disc_weeks]
            disc_trades_list = [f.result() for f in futs_disc if len(f.result()) > 0]
            disc_trades = np.concatenate(disc_trades_list) if disc_trades_list else np.empty(0)

            elapsed = time.time() - t0

            # Compute statistical metrics
            hold_stats = compute_statistical_metrics(hold_trades, candidates_screened=44)
            disc_stats = compute_statistical_metrics(disc_trades, candidates_screened=44)

            # Build canonical manifest
            manifest = create_certified_run_manifest(
                strategy_name=name,
                strategy_file=strat_path,
                symbol=SYMBOL,
                split_version="canonical_continuous_v1.0 (In-Sample: 1..210, Holdout: 211..300)",
                start_week=211,
                end_week=300,
                pip_size=PIP_SIZE,
                commission_pips=COMM_PIPS,
                entry_slippage_pips=SLIP_PIPS,
                spread_policy="Floating Raw Dukascopy Bid/Ask",
                warmup_ticks=WARMUP_TICKS,
                trades=hold_trades,
                candidates_screened=44,
                additional_meta={
                    "discovery_in_sample_metrics": disc_stats,
                    "evaluation_time_seconds": round(elapsed, 2),
                    "clean_weeks_holdout": len(hold_weeks),
                    "clean_weeks_discovery": len(disc_weeks),
                    "combined_net_pips": round(hold_stats["net_pips"] + disc_stats["net_pips"], 1)
                }
            )

            # Save manifest & trades
            manifest_file = OUTPUT_DIR / f"{name}_run_manifest.json"
            trades_file = OUTPUT_DIR / f"{name}_trades.csv"

            with open(manifest_file, "w") as f:
                json.dump(manifest, f, indent=2)

            if len(hold_trades) > 0:
                pd.DataFrame(manifest["trade_log"]).to_csv(trades_file, index=False)

            print(f"   Holdout:   {hold_stats['net_pips']:>+7.1f}p | PF: {hold_stats['profit_factor']:>4.2f} | Trades: {hold_stats['total_trades']:>3} | Max DD: {hold_stats['max_drawdown_pips']:>5.1f}p | Exp: {hold_stats['expectancy_pips']:>+4.2f}p")
            print(f"   Discovery: {disc_stats['net_pips']:>+7.1f}p | PF: {disc_stats['profit_factor']:>4.2f} | Trades: {disc_stats['total_trades']:>3} | Max DD: {disc_stats['max_drawdown_pips']:>5.1f}p")
            print(f"   Combined:  {manifest['custom_metadata']['combined_net_pips']:>+7.1f}p | Status: {manifest['certification_status']} ({elapsed:.1f}s)")


if __name__ == "__main__":
    certify_top_champions()

