#!/usr/bin/env python3
"""
Screen certified_29_champions candidates for EURGBP and USDJPY across:
  - In-Sample Discovery: Weeks 1..210
  - Quarantined Continuous Holdout: Weeks 211..300
"""

import sys
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.dataloader import get_complete_split_weeks
from tick_engine.core.config import PIP_SIZES, get_friction_pips
from tick_engine.reporting.run_manifest import compute_statistical_metrics
from scripts.screen_pair_fleet import _eval_strat_week

def get_c29_candidates():
    candidates = []
    base_dir = ROOT_DIR / "archive_deprecated" / "strategies" / "certified_29_champions"
    
    # EURGBP
    eurgbp_dir = base_dir / "eurgbp"
    if eurgbp_dir.exists():
        for d in sorted(eurgbp_dir.iterdir()):
            if d.is_dir() and (d / "strategy.py").exists():
                candidates.append(("EURGBP", d.name, str(d / "strategy.py")))
                
    # USDJPY
    usdjpy_dir = base_dir / "usdjpy"
    if usdjpy_dir.exists():
        for d in sorted(usdjpy_dir.iterdir()):
            if d.is_dir() and (d / "strategy.py").exists():
                candidates.append(("USDJPY", d.name, str(d / "strategy.py")))
                
    return candidates

def main():
    candidates = get_c29_candidates()
    print("=" * 95)
    print(f"  SCREENING CERTIFIED_29 FLEET: {len(candidates)} CANDIDATE STRATEGIES")
    print("=" * 95)
    
    out_dir = ROOT_DIR / "results" / "continuous_walkforward_audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "certified_29_screen.json"
    
    results = []
    winners = []
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        for idx, (symbol, name, strat_path) in enumerate(candidates, 1):
            t0 = time.time()
            hold_weeks = get_complete_split_weeks(symbol, "holdout_test_30")
            disc_weeks = get_complete_split_weeks(symbol, "discovery_70")
            
            futs_hold = [executor.submit(_eval_strat_week, strat_path, symbol, w) for w in hold_weeks]
            hold_trades_list = [f.result() for f in futs_hold if len(f.result()) > 0]
            hold_trades = np.concatenate(hold_trades_list) if hold_trades_list else np.empty(0)
            
            futs_disc = [executor.submit(_eval_strat_week, strat_path, symbol, w) for w in disc_weeks]
            disc_trades_list = [f.result() for f in futs_disc if len(f.result()) > 0]
            disc_trades = np.concatenate(disc_trades_list) if disc_trades_list else np.empty(0)
            
            elapsed = time.time() - t0
            
            hold_stats = compute_statistical_metrics(hold_trades, candidates_screened=len(candidates))
            disc_stats = compute_statistical_metrics(disc_trades, candidates_screened=len(candidates))
            
            h_pips = hold_stats["net_pips"]
            h_pf = hold_stats["profit_factor"]
            h_tr = hold_stats["total_trades"]
            d_pips = disc_stats["net_pips"]
            d_pf = disc_stats["profit_factor"]
            d_tr = disc_stats["total_trades"]
            
            is_winner = (h_pips > 0 and d_pips > 0 and h_pf >= 1.05 and d_pf >= 1.05 and h_tr >= 5)
            status = "WINNER" if is_winner else "REJECT"
            
            item = {
                "symbol": symbol,
                "name": name,
                "path": strat_path,
                "is_winner": is_winner,
                "holdout": hold_stats,
                "discovery": disc_stats,
                "elapsed_sec": round(elapsed, 2)
            }
            results.append(item)
            if is_winner:
                winners.append(item)
                
            print(f"[{idx:02d}/{len(candidates)}] {symbol:7s} {name[:34]:34s} | Hold: {h_pips:>+7.1f}p (PF {h_pf:>4.2f}, Tr {h_tr:>3}) | Disc: {d_pips:>+7.1f}p (PF {d_pf:>4.2f}) | {status} ({elapsed:.1f}s)")
            
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 95)
    print(f"  SCREENING COMPLETE: {len(winners)} / {len(candidates)} DUAL-PARTITION SURVIVORS FOUND")
    print("=" * 95)
    for w in sorted(winners, key=lambda x: x["holdout"]["net_pips"], reverse=True):
        print(f"  {w['symbol']:7s} {w['name']:36s} | Hold: {w['holdout']['net_pips']:>+7.1f}p (PF {w['holdout']['profit_factor']:.2f}) | Disc: {w['discovery']['net_pips']:>+7.1f}p (PF {w['discovery']['profit_factor']:.2f})")

if __name__ == "__main__":
    main()

