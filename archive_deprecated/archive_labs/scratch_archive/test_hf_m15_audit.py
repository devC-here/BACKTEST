#!/usr/bin/env python3
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset
from strategies.high_frequency_m15_scalper_elite.strategy import HighFrequencyM15ScalperElite

def run_week(w: int):
    ticks, meta = get_week_dataset(w)
    engine = BacktestEngine(pip_size=0.0001, use_slippage=True)
    strat = HighFrequencyM15ScalperElite()
    trades = engine.run(ticks, strat)
    return [float(t["pips"]) for t in trades]

def main():
    print("Testing original HF_M15...")
    with ProcessPoolExecutor() as executor:
        week_res = list(executor.map(run_week, range(1, 53)))
    all_pips = []
    for r in week_res:
        all_pips.extend(r)
    pips = np.array(all_pips)
    wins = pips[pips > 0]
    losses = pips[pips < 0]
    wr = len(wins) / len(pips) * 100 if len(pips) > 0 else 0
    pf = np.sum(wins) / abs(np.sum(losses)) if len(losses) > 0 else 0
    rr = np.mean(wins) / abs(np.mean(losses)) if len(losses) > 0 else 0
    print(f"Original HF_M15: Trades={len(pips)}, WR={wr:.1f}%, Realized RR={rr:.2f}, PF={pf:.2f}, Net Pips={np.sum(pips):+.1f}")

if __name__ == "__main__":
    main()
