#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.path.insert(0, r'D:\PROJECT\TRD\MODELS GENERATIONS\BACKTEST')
from tick_engine.core.engine import BacktestEngine, TRADE_DTYPE
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.core.metrics import compute_metrics

from strategies.disc03_ny_open_overlap_elite.strategy import NewYorkOpenOverlapBreakoutElite
from strategies.comb12_bollinger_squeeze_elite.strategy import BollingerBandwidthExpansionElite
from strategies.comb27_squeeze_supertrend_hybrid_elite.strategy import BollingerSupertrendHybridElite
from strategies.inst05_market_structure_shift_elite.strategy import MarketStructureShiftRetestElite
from strategies.disc31_london_lunch_trap_fade_elite.strategy import LondonLunchTrapFadeElite

core_strats = [
    NewYorkOpenOverlapBreakoutElite,
    BollingerBandwidthExpansionElite,
    BollingerSupertrendHybridElite,
    MarketStructureShiftRetestElite,
    LondonLunchTrapFadeElite
]

def eval_week(w):
    ticks, _ = get_week_dataset(w)
    engine = BacktestEngine(pip_size=0.0001, use_slippage=True)
    all_tr = []
    for cls in core_strats:
        st = cls()
        tr = engine.run(ticks, st)
        if len(tr) > 0:
            all_tr.append(tr)
    return np.concatenate(all_tr) if all_tr else np.empty(0, dtype=TRADE_DTYPE)

if __name__ == "__main__":
    weeks = list(range(1, 263))
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        week_trades = list(ex.map(eval_week, weeks))

    all_t = [t for t in week_trades if len(t) > 0]
    combined = np.concatenate(all_t) if all_t else np.empty(0, dtype=TRADE_DTYPE)
    combined_sorted = np.sort(combined, order='entry_time')

    m_5y = compute_metrics(combined_sorted)
    cum = np.cumsum(combined_sorted['pips'])
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = float(np.max(dd))

    # Train vs Test
    tr_train = [t for i, t in enumerate(week_trades) if i < 156 and len(t) > 0]
    tr_test = [t for i, t in enumerate(week_trades) if i >= 156 and len(t) > 0]
    m_train = compute_metrics(np.concatenate(tr_train))
    m_test = compute_metrics(np.concatenate(tr_test))

    print("=" * 80)
    print("  TOP 5 DUAL-REGIME POSITIVE DISCOVERED FLEET (5-YEAR / 262 WEEKS)")
    print("=" * 80)
    print(f"Total Trades:  {m_5y['total_trades']} ({m_5y['total_trades']/262:.1f}/wk, {m_5y['total_trades']/1250:.2f}/day)")
    print(f"Win Rate:      {m_5y['win_rate_pct']:.1f}%")
    print(f"Profit Factor: {m_5y['profit_factor']:.2f}")
    print(f"3Y Train:      {m_train['net_pips']:+.1f} pips (PF: {m_train['profit_factor']:.2f}, WR: {m_train['win_rate_pct']:.1f}%)")
    print(f"2Y Test:       {m_test['net_pips']:+.1f} pips (PF: {m_test['profit_factor']:.2f}, WR: {m_test['win_rate_pct']:.1f}%)")
    print(f"5Y Net Return: {m_5y['net_pips']:+.1f} pips")
    print(f"5Y Max DD:     {max_dd:.1f} pips")
    print("=" * 80)
