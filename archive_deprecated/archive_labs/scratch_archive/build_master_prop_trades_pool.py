#!/usr/bin/env python3
"""
================================================================================
  BUILD MASTER 7-MODEL PROP TRADES POOL ACROSS 5 YEARS (262 WEEKS)
================================================================================
Simulates all 7 high-performing models:
  1. DISC42: MacroStructureH4M30Elite (72.9% WR, PF 3.97, Max DD 20.4p)
  2. DISC43: IntradayPrecisionH1M5Elite (48.3% WR, PF 1.47, Max DD 68.7p)
  3. DISC40: RSIRegimeRangeShiftElite (44.0% WR, PF 1.31, Net +1112.8p)
  4. DISC41: ThursdayMacroTrendIgnitionElite (43.6% WR, PF 1.30, Net +336.1p)
  5. DISC38: LondonMiddayMACDAccelerationElite (43.0% WR, PF 1.28, Net +372.7p)
  6. DISC39: StochasticDeepPullbackElite (42.5% WR, PF 1.22, Net +483.4p)
  7. HF_M15: HighFrequencyM15ScalperElite (35.7% WR, PF 1.13, Net +1755.9p)

Saves all trades to results/master_7_prop_trades_5y.json for instant combo testing.
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset

from strategies.new_discoveries import (
    LondonMiddayMACDAccelerationElite,
    StochasticDeepPullbackElite,
    RSIRegimeRangeShiftElite,
    ThursdayMacroTrendIgnitionElite,
    MacroStructureH4M30Elite,
    IntradayPrecisionH1M5Elite
)
from strategies.high_frequency_m15_scalper_elite.strategy import HighFrequencyM15ScalperElite

PIP_SIZE = 0.0001
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "master_7_prop_trades_5y.json"


def _simulate_week(w: int):
    ticks, _ = get_week_dataset(w)
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)

    strats = [
        ("DISC42_H4_M30", MacroStructureH4M30Elite()),
        ("DISC43_H1_M5", IntradayPrecisionH1M5Elite()),
        ("DISC40_RSI", RSIRegimeRangeShiftElite()),
        ("DISC41_THURSDAY", ThursdayMacroTrendIgnitionElite()),
        ("DISC38_MACD", LondonMiddayMACDAccelerationElite()),
        ("DISC39_STOCH", StochasticDeepPullbackElite()),
        ("HF_M15", HighFrequencyM15ScalperElite())
    ]

    week_trades = []
    for s_id, s in strats:
        tr = engine.run(ticks, s)
        for t in tr:
            week_trades.append({
                'week': w,
                'entry_time': int(t['entry_time']),
                'exit_time': int(t['exit_time']),
                'direction': int(t['direction']),
                'entry_price': float(t['entry_price']),
                'exit_price': float(t['exit_price']),
                'pips': float(t['pips']),
                'exit_reason': int(t['exit_reason']),
                'strategy': s_id
            })

    return week_trades


def main():
    print("=" * 90)
    print("   SIMULATING MASTER 7-MODEL PROP POOL ACROSS 262 WEEKS (123.22M TICKS)   ".center(90))
    print("=" * 90)

    t0 = time.time()
    weeks = list(range(1, 263))
    num_workers = min(4, os.cpu_count() or 4)
    print(f"[*] Running with {num_workers} worker processes...")

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        week_results = list(ex.map(_simulate_week, weeks))

    dt = time.time() - t0
    all_trades = [t for wr in week_results for t in wr]
    all_trades.sort(key=lambda x: x['entry_time'])

    print(f"[+] All 262 weeks simulated in {dt:.1f}s!")
    print(f"[+] Extracted total: {len(all_trades)} trades across 7 elite models.")

    from collections import Counter
    counts = Counter(t['strategy'] for t in all_trades)
    print("\nTrades per Strategy:")
    for s_id, count in sorted(counts.items(), key=lambda x: -x[1]):
        strat_trades = [t for t in all_trades if t['strategy'] == s_id]
        net_p = sum(t['pips'] for t in strat_trades)
        wins = [t for t in strat_trades if t['pips'] > 0]
        wr = len(wins) / count * 100 if count else 0
        print(f"  - {s_id:<16}: {count:>5} trades | Win Rate: {wr:>5.1f}% | Net Pips: {net_p:>+8.1f}p")

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_trades, f)
    print(f"\n[+] Saved master trade dataset to {OUTPUT_FILE} ({OUTPUT_FILE.stat().st_size / 1024:.1f} KB)")


if __name__ == '__main__':
    main()
