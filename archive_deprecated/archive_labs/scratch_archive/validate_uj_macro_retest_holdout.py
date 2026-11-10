#!/usr/bin/env python3
"""
================================================================================
  VALIDATION OF UJ_MACRO_RETEST ACROSS DISCOVERY & QUARANTINED HOLDOUT PARTITIONS
================================================================================
Performs rigorous In-Sample (210 weeks) vs Out-of-Sample (90 weeks) holdout validation
for the newly discovered USDJPY Intraday Macro Retest strategy.
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset, get_discovery_weeks, get_holdout_test_weeks
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class USDJPYIntradayMacroRetest(BaseStrategy):
    """
    DISC42 variant scaled to H1/M15 for higher trade velocity.
    H1 20-bar rolling swing highs and lows + M15 rejection wick (>= 45%).
    """
    def __init__(self, sl_pips: float = 12.0, tp_pips: float = 20.0, pip_size: float = 0.01, max_spread: float = 0.8, **kwargs):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = "USDJPY_Intraday_Macro_Retest"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 30:
            return signals, sl_arr, tp_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid}, index=ts)
        h1 = base.resample('1h').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(h1) < 25:
            return signals, sl_arr, tp_arr

        h1['ema20'] = h1['close'].ewm(span=20, adjust=False).mean()
        h1['bias'] = np.where(h1['close'] > h1['ema20'], 1, -1)
        h1['roll_h'] = h1['high'].rolling(20).max().shift(1).bfill()
        h1['roll_l'] = h1['low'].rolling(20).min().shift(1).bfill()

        bias_m15 = h1['bias'].reindex(m15.index, method='ffill').fillna(0).values
        supp_m15 = h1['roll_l'].reindex(m15.index, method='ffill').values
        res_m15 = h1['roll_h'].reindex(m15.index, method='ffill').values

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_idx = -999

        for i in range(15, len(m15)):
            if hours[i] not in {1, 2, 3, 7, 8, 9, 12, 13, 14, 15}:
                continue
            if spread[i] > self.max_spread:
                continue
            if (i - last_trade_idx) < 4:
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < 4.0:
                continue

            b = bias_m15[i]
            sig = 0

            # Bullish Retest of H1 Support
            if b > 0:
                dist_supp = abs(l[i] - supp_m15[i]) / self.pip_size
                lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
                if dist_supp <= 4.0 and (lower_wick / bar_rng) >= 0.45 and c[i] > o[i]:
                    sig = SIGNAL_LONG

            # Bearish Retest of H1 Resistance
            elif b < 0:
                dist_res = abs(res_m15[i] - h[i]) / self.pip_size
                upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size
                if dist_res <= 4.0 and (upper_wick / bar_rng) >= 0.45 and c[i] < o[i]:
                    sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_idx = i

        return signals, sl_arr, tp_arr


def evaluate_week(week_num: int) -> Tuple[int, List[Dict[str, Any]]]:
    engine = BacktestEngine(pip_size=0.01, use_slippage=True)
    ticks, meta = get_week_dataset(week_num, raw_dir="data/USDJPY")
    if len(ticks) == 0:
        return week_num, []

    model = USDJPYIntradayMacroRetest(pip_size=0.01, max_spread=0.8, sl_pips=12.0, tp_pips=20.0)
    trades = []
    try:
        raw_trades = engine.run(ticks, model)
        for i in range(len(raw_trades)):
            row = raw_trades[i]
            trades.append({
                "week": week_num,
                "pips": float(row["pips"]),
                "direction": int(row["direction"])
            })
    except Exception as e:
        pass
    return week_num, trades


def run_uj_macro_retest_validation():
    t_start = time.time()
    disc_weeks = set(get_discovery_weeks("data/USDJPY"))
    hold_weeks = set(get_holdout_test_weeks("data/USDJPY"))
    all_weeks = sorted(list(disc_weeks.union(hold_weeks)))

    print("=" * 80)
    print(f"AUDITING USDJPY_INTRADAY_MACRO_RETEST OVER FULL 300 WEEKS (5 YEARS)")
    print(f"Discovery Partition: {len(disc_weeks)} weeks | Quarantined Holdout: {len(hold_weeks)} weeks")
    print("=" * 80)

    trades_by_week = {}
    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(evaluate_week, w): w for w in all_weeks}
        for f in as_completed(futures):
            w, t_list = f.result()
            trades_by_week[w] = t_list

    disc_trades = []
    hold_trades = []
    all_trades = []

    for w in sorted(all_weeks):
        t_list = trades_by_week.get(w, [])
        all_trades.extend(t_list)
        if w in disc_weeks:
            disc_trades.extend(t_list)
        elif w in hold_weeks:
            hold_trades.extend(t_list)

    def compute_stats(t_list: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
        n = len(t_list)
        if n == 0:
            return {"partition": name, "trades": 0, "net_pips": 0.0, "pf": 0.0, "wr": 0.0, "max_dd": 0.0, "expectancy": 0.0}
        p_arr = np.array([t["pips"] for t in t_list])
        net_pips = float(np.sum(p_arr))
        wins = p_arr[p_arr > 0]
        losses = -p_arr[p_arr < 0]
        wr = float(len(wins) / n * 100.0)
        sum_w = float(np.sum(wins))
        sum_l = float(np.sum(losses))
        pf = round(sum_w / sum_l, 2) if sum_l > 0 else (99.0 if sum_w > 0 else 0.0)

        equity = np.cumsum(p_arr)
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
        expectancy = round(net_pips / n, 2)

        return {
            "Partition": name,
            "Weeks": len(disc_weeks) if "Discovery" in name else (len(hold_weeks) if "Holdout" in name else len(all_weeks)),
            "Trades": n,
            "Win_Rate": f"{wr:.1f}%",
            "Profit_Factor": pf,
            "Net_Pips": round(net_pips, 1),
            "Max_DD_Pips": round(max_dd, 1),
            "Expectancy_pips": expectancy
        }

    s_disc = compute_stats(disc_trades, "70% Discovery Partition (In-Sample)")
    s_hold = compute_stats(hold_trades, "30% Holdout Partition (Out-of-Sample)")
    s_all = compute_stats(all_trades, "100% Full 5-Year Cycle (300 Weeks)")

    df = pd.DataFrame([s_disc, s_hold, s_all])
    print("\n" + "=" * 80)
    print("USDJPY_INTRADAY_MACRO_RETEST: FULL PARTITION AUDIT SCORECARD")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    out_csv = RESULTS_DIR / "uj_macro_retest_validation_scorecard.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[+] Saved validation results to: {out_csv.resolve()}")
    print(f"[+] Completed in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    run_uj_macro_retest_validation()
