#!/usr/bin/env python3
"""Test AUDNZD on Intraday Macro Retest."""
import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset, load_manifest
from scratch.validate_macro_retest_cross_holdouts import IntradayMacroRetestUniversal

manifest = load_manifest("data/AUDNZD")
engine = BacktestEngine(pip_size=0.0001, use_slippage=True)
model = IntradayMacroRetestUniversal(asset="AUDNZD", sl_pips=10.0, tp_pips=18.0, pip_size=0.0001, max_spread=2.5)

all_trades = []
for w in sorted(manifest.keys()):
    ticks, meta = get_week_dataset(w, raw_dir="data/AUDNZD")
    if len(ticks) == 0:
        continue
    trades = engine.run(ticks, model)
    for t in trades:
        all_trades.append(float(t['pips']))

n = len(all_trades)
if n > 0:
    p_arr = np.array(all_trades)
    net_pips = np.sum(p_arr)
    wins = p_arr[p_arr > 0]
    losses = -p_arr[p_arr < 0]
    wr = len(wins) / n * 100.0
    pf = np.sum(wins) / np.sum(losses) if np.sum(losses) > 0 else 99.0
    equity = np.cumsum(p_arr)
    dd = np.maximum.accumulate(equity) - equity
    max_dd = np.max(dd) if len(dd) > 0 else 0.0
    print(f"AUDNZD (53 Weeks): Trades={n}, WR={wr:.1f}%, PF={pf:.2f}, Net={net_pips:.1f}p, MaxDD={max_dd:.1f}p")
else:
    print("AUDNZD: No trades generated.")
