#!/usr/bin/env python3
"""
Generate strictly chronological train/test partitions for all assets.
Eliminates regime leakage and temporal interpolation inherent in randomized week splitting.
Partition:
  - In-Sample (Calibration): First 70% of continuous calendar weeks (e.g. Weeks 1 to 210)
  - Out-of-Sample (Forward Holdout): Last 30% of continuous calendar weeks (e.g. Weeks 211 to 300)
"""

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RANDOM_SPLIT_FILE = ROOT_DIR / "data_splits" / "master_70_30_splits.json"
CHRONO_SPLIT_FILE = ROOT_DIR / "data_splits" / "master_chronological_splits.json"

def generate_chronological_splits():
    with open(RANDOM_SPLIT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    chrono_assets = {}
    for sym, prof in data.get("assets", {}).items():
        total_w = prof.get("total_weeks", 300)
        n_in_sample = int(total_w * 0.70)
        
        in_sample_weeks = list(range(1, n_in_sample + 1))
        out_of_sample_weeks = list(range(n_in_sample + 1, total_w + 1))

        chrono_assets[sym] = {
            "symbol": sym,
            "total_weeks": total_w,
            "in_sample_calibration_70": {
                "count": len(in_sample_weeks),
                "percentage": round(len(in_sample_weeks) / total_w * 100, 1),
                "weeks": in_sample_weeks,
                "role": "Historical model calibration & parameter tuning (Zero forward peeking)"
            },
            "out_of_sample_holdout_30": {
                "count": len(out_of_sample_weeks),
                "percentage": round(len(out_of_sample_weeks) / total_w * 100, 1),
                "weeks": out_of_sample_weeks,
                "role": "Strictly chronological unseen forward market evaluation"
            }
        }

    chrono_data = {
        "standard": "Strict Chronological Partitioning (Causal Walk-Forward Standard)",
        "policy": "First 70% of continuous weeks for training/calibration; final 30% continuous forward weeks for out-of-sample testing",
        "total_assets": len(chrono_assets),
        "assets": chrono_assets
    }

    with open(CHRONO_SPLIT_FILE, "w", encoding="utf-8") as f:
        json.dump(chrono_data, f, indent=2)

    print(f"Generated chronological split registry: {CHRONO_SPLIT_FILE}")

if __name__ == "__main__":
    generate_chronological_splits()

