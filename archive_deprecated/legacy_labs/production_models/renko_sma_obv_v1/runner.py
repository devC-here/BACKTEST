"""
Execution and Verification Runner for RenkoSmaObvStrategy (Model V1).
"""

import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from production_models.renko_sma_obv_v1.strategy import RenkoSmaObvStrategy
from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.core.metrics import compute_metrics, print_scorecard

def run_renko_v1(weeks_count: int = 10):
    engine = BacktestEngine(pip_size=0.0001)
    strat = RenkoSmaObvStrategy.from_config()

    print(f"Loading {weeks_count} weeks for {strat.name} verification...")
    weeks = [get_week_dataset(w)[0] for w in range(1, weeks_count + 1)]

    trades_list = []
    for ticks in weeks:
        tr = engine.run(ticks, strat)
        if len(tr) > 0:
            trades_list.append(tr)

    if len(trades_list) > 0:
        all_tr = np.concatenate(trades_list)
        m = compute_metrics(all_tr)
        print_scorecard(m, title=f"{strat.name} Verification ({weeks_count} Weeks)")
    else:
        print("No trades generated.")

if __name__ == "__main__":
    run_renko_v1(weeks_count=10)
