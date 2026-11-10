#!/usr/bin/env python3
"""
================================================================================
  AUDIT: 7-STAR ALL-WEATHER DISCOVERED FLEET (5 YEARS / 262 WEEKS / 123.2M TICKS)
  Combines the 7 top all-weather dual-regime discovered champions:
    1. DISC03: NY Open Overlap Breakout Elite
    2. INST05: Market Structure Shift Retest Elite
    3. DISC36: Frankfurt Ultra-Coil London ORB Elite
    4. COMB12: Bollinger Squeeze Elite
    5. COMB27: Squeeze Supertrend Hybrid Elite
    6. DISC37: European Midday Ultra-Coil NY Breakout Elite
    7. DISC31: London Lunch Trap Fade Elite
================================================================================
"""

from __future__ import annotations
import sys
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine, TRADE_DTYPE
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.core.metrics import compute_metrics

from strategies.disc03_ny_open_overlap_elite.strategy import NewYorkOpenOverlapBreakoutElite
from strategies.inst05_market_structure_shift_elite.strategy import MarketStructureShiftRetestElite
from strategies.disc36_frankfurt_ultra_coil_orb_elite.strategy import FrankfurtUltraCoilLondonORBElite
from strategies.comb12_bollinger_squeeze_elite.strategy import BollingerBandwidthExpansionElite
from strategies.comb27_squeeze_supertrend_hybrid_elite.strategy import BollingerSupertrendHybridElite
from strategies.disc37_european_midday_ultracoil_breakout_elite.strategy import EuropeanMiddayUltraCoilBreakoutElite
from strategies.disc31_london_lunch_trap_fade_elite.strategy import LondonLunchTrapFadeElite

ALL_7_STRATS = [
    NewYorkOpenOverlapBreakoutElite,
    MarketStructureShiftRetestElite,
    FrankfurtUltraCoilLondonORBElite,
    BollingerBandwidthExpansionElite,
    BollingerSupertrendHybridElite,
    EuropeanMiddayUltraCoilBreakoutElite,
    LondonLunchTrapFadeElite
]

def eval_week(w):
    ticks, _ = get_week_dataset(w)
    engine = BacktestEngine(pip_size=0.0001, use_slippage=True)
    trades_list = []
    for cls in ALL_7_STRATS:
        strat = cls()
        tr = engine.run(ticks, strat)
        if len(tr) > 0:
            trades_list.append(tr)
    return np.concatenate(trades_list) if trades_list else np.empty(0, dtype=TRADE_DTYPE)

if __name__ == "__main__":
    weeks = list(range(1, 263))
    num_workers = min(8, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
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

    print("=" * 85)
    print("      CONSOLIDATED 7-STAR ALL-WEATHER DISCOVERED FLEET (5-YEAR / 262 WEEKS)      ")
    print("=" * 85)
    print(f"Total Fleet Trades:       {m_5y['total_trades']} trades ({m_5y['total_trades']/262:.1f}/wk, {m_5y['total_trades']/1250:.2f}/day)")
    print(f"Portfolio Win Rate:       {m_5y['win_rate_pct']:.1f}%")
    print(f"Realized Risk-to-Reward:  {(m_5y['gross_profit_pips']/max(m_5y['win_count'],1))/(m_5y['gross_loss_pips']/max(m_5y['loss_count'],1)):.2f} : 1")
    print(f"Portfolio Profit Factor:  {m_5y['profit_factor']:.2f}")
    print("-" * 85)
    print(f"3-Year In-Sample Return:  {m_train['net_pips']:+10.1f} pips | PF: {m_train['profit_factor']:.2f} | WR: {m_train['win_rate_pct']:.1f}% ({m_train['total_trades']} trds)")
    print(f"2-Year Out-of-Sample:     {m_test['net_pips']:+10.1f} pips | PF: {m_test['profit_factor']:.2f} | WR: {m_test['win_rate_pct']:.1f}% ({m_test['total_trades']} trds)")
    print("-" * 85)
    print(f"5-Year Net Total Return:  {m_5y['net_pips']:+10.1f} pips")
    print(f"5-Year Maximum Drawdown:  {max_dd:10.1f} pips (~{max_dd/100:.2f}% account DD)")
    print(f"Return-to-Drawdown Ratio: {m_5y['net_pips']/max(max_dd, 1e-9):10.2f} : 1")
    print("=" * 85)
