import json
import pandas as pd
from pathlib import Path

ROOT = Path("D:/PROJECT/TRD/MODELS GENERATIONS/BACKTEST")
RES_DIR = ROOT / "results"
SPLITS_FILE = ROOT / "data_splits" / "master_70_30_splits.json"

all_10 = ['EURUSD', 'USDJPY', 'EURGBP', 'AUDNZD', 'AUDCAD', 'USDCAD', 'GBPJPY', 'EURNZD', 'GBPCHF', 'NZDCAD']

with open(SPLITS_FILE, 'r', encoding='utf-8') as f:
    splits_data = json.load(f)

print("=" * 110)
print("  COMPLETE 10-SYMBOL TICK DATASET AUDIT")
print("=" * 110)
print(f"{'Symbol':<10} | {'Weeks':<6} | {'Disc Wks':<9} | {'Hold Wks':<9} | {'Total Ticks':<15} | {'Discovery Scorecard':<20} | {'Fleet Recommendation'}")
print("-" * 110)

tot_weeks = 0
tot_disc = 0
tot_hold = 0
tot_ticks = 0

for s in all_10:
    asset_meta = splits_data["assets"].get(s, {})
    w = asset_meta.get("total_weeks", 0)
    dw = asset_meta.get("discovery_70", {}).get("count", 0)
    hw = asset_meta.get("holdout_test_30", {}).get("count", 0)
    t = asset_meta.get("discovery_70", {}).get("total_ticks", 0) + asset_meta.get("holdout_test_30", {}).get("total_ticks", 0)
    
    tot_weeks += w
    tot_disc += dw
    tot_hold += hw
    tot_ticks += t

    sc_path = RES_DIR / f"{s.lower()}_master_discovery_scorecard.csv"
    rec_path = RES_DIR / f"{s.lower()}_master_fleet_recommendation.json"

    sc_status = "Available" if sc_path.exists() else "None"
    rec_status = "Available (8 Champs)" if rec_path.exists() else "Legacy Portfolio"

    print(f"{s:<10} | {w:<6} | {dw:<9} | {hw:<9} | {t:>15,} | {sc_status:<20} | {rec_status}")

print("-" * 110)
print(f"{'TOTAL':<10} | {tot_weeks:<6} | {tot_disc:<9} | {tot_hold:<9} | {tot_ticks:>15,}")
print("=" * 110)

