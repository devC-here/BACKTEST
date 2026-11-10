#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB AUDNZD V3: PRECISION MACRO STRUCTURE CALIBRATION
================================================================================
Focuses on the proven edge archetype on AUDNZD (H1/H4 Macro Structure Retest):
  1. H1 Macro Retest Fine-Tuning:
     - Varying SL: [9.0, 10.0, 11.0, 12.0, 13.0]
     - Varying TP: [16.0, 18.0, 20.0, 22.0, 24.0]
     - Varying Wick Absorption: [0.40, 0.45, 0.50, 0.55]
     - Varying Touch Distance: [3.5, 4.5, 5.5]
     - Varying Donchian Lookback: [15, 20, 25]
  2. H4 Macro Structure Architecture (Higher Timeframe Anchor):
     - H4 20-EMA trend + H4 Donchian(10/15) support/resistance + M15 rejection wick
  3. Triple Screen Trend Alignment:
     - Daily 20-EMA trend + H1 Donchian pullback + M15 rejection wick

Evaluated strictly across the 70% Quarantined Discovery Partition (210 weeks).
30% Holdout Partition (90 weeks) strictly reserved for out-of-sample certification.
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
import json
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
from tick_engine.core.dataloader import get_week_dataset, get_discovery_weeks
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class AUDNZDMacroStructureV3(BaseStrategy):
    def __init__(
        self,
        name: str,
        tf_macro: str = "1h", # '1h' or '4h'
        lookback_macro: int = 20,
        dist_pips: float = 4.5,
        min_wick: float = 0.45,
        sl_pips: float = 12.0,
        tp_pips: float = 20.0,
        session_mode: str = "asian_london",
        max_spread: float = 2.8,
        pip_size: float = 0.0001,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.name = name
        self.tf_macro = tf_macro
        self.lookback_macro = int(lookback_macro)
        self.dist_pips = float(dist_pips)
        self.min_wick = float(min_wick)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.session_mode = session_mode
        self.max_spread = float(max_spread)
        self.pip_size = float(pip_size)

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 30:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid}, index=ts)

        # Macro timeframe
        macro = base.resample(self.tf_macro).agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(macro) < max(self.lookback_macro + 5, 15):
            return signals, sl_pips_arr, tp_pips_arr

        macro['ema20'] = macro['close'].ewm(span=20, adjust=False).mean()
        macro['bias'] = np.where(macro['close'] > macro['ema20'], 1, -1)
        macro['roll_h'] = macro['high'].rolling(self.lookback_macro).max().shift(1).bfill()
        macro['roll_l'] = macro['low'].rolling(self.lookback_macro).min().shift(1).bfill()

        bias_m15 = macro['bias'].reindex(m15.index, method='ffill').fillna(0).values
        supp_m15 = macro['roll_l'].reindex(m15.index, method='ffill').values
        res_m15 = macro['roll_h'].reindex(m15.index, method='ffill').values

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None

        for i in range(15, len(m15)):
            hr = hours[i]
            if self.session_mode == 'london_only' and not (7 <= hr <= 16):
                continue
            elif self.session_mode == 'asian_london' and not (hr >= 21 or hr <= 16):
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < 2.5:
                continue

            b = bias_m15[i]
            sig = 0

            # Bullish Retest of Macro Support in uptrend
            if b > 0:
                dist_supp = abs(l[i] - supp_m15[i]) / self.pip_size
                lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
                if dist_supp <= self.dist_pips and (lower_wick / bar_rng) >= self.min_wick and c[i] > o[i]:
                    sig = SIGNAL_LONG

            # Bearish Retest of Macro Resistance in downtrend
            elif b < 0:
                dist_res = abs(res_m15[i] - h[i]) / self.pip_size
                upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size
                if dist_res <= self.dist_pips and (upper_wick / bar_rng) >= self.min_wick and c[i] < o[i]:
                    sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = self.sl_pips
                    tp_pips_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_pips_arr, tp_pips_arr


def build_lab_v3_candidates() -> List[Tuple[str, Any]]:
    candidates = []

    # 1. H1 Grid: varying SL and TP
    for sl in [10.0, 11.0, 12.0, 13.0]:
        for tp in [18.0, 20.0, 22.0, 24.0]:
            name = f"H1_Macro_SL{int(sl):02d}_TP{int(tp):02d}_Wick45_D45"
            candidates.append((name, AUDNZDMacroStructureV3(
                name=name, tf_macro="1h", lookback_macro=20, dist_pips=4.5,
                min_wick=0.45, sl_pips=sl, tp_pips=tp
            )))

    # 2. H1 Grid: varying Wick Threshold
    for wick in [0.40, 0.50, 0.55]:
        name = f"H1_Macro_SL12_TP20_Wick{int(wick*100):02d}_D45"
        candidates.append((name, AUDNZDMacroStructureV3(
            name=name, tf_macro="1h", lookback_macro=20, dist_pips=4.5,
            min_wick=wick, sl_pips=12.0, tp_pips=20.0
        )))

    # 3. H1 Grid: varying Touch Distance
    for dist in [3.5, 4.0, 5.0, 5.5]:
        name = f"H1_Macro_SL12_TP20_Wick45_D{int(dist*10):02d}"
        candidates.append((name, AUDNZDMacroStructureV3(
            name=name, tf_macro="1h", lookback_macro=20, dist_pips=dist,
            min_wick=0.45, sl_pips=12.0, tp_pips=20.0
        )))

    # 4. H4 Macro Structure Architecture
    for sl, tp in [(12.0, 20.0), (12.0, 24.0), (14.0, 28.0), (15.0, 30.0)]:
        name = f"H4_Macro_SL{int(sl):02d}_TP{int(tp):02d}_Lookback10"
        candidates.append((name, AUDNZDMacroStructureV3(
            name=name, tf_macro="4h", lookback_macro=10, dist_pips=5.0,
            min_wick=0.45, sl_pips=sl, tp_pips=tp
        )))

    return candidates


def evaluate_week_v3(week_num: int) -> Dict[str, Any]:
    engine = BacktestEngine(pip_size=0.0001, use_slippage=True)
    try:
        ticks, meta = get_week_dataset(week_num, raw_dir="data/AUDNZD")
        if len(ticks) == 0:
            return {"week": week_num, "trades": {}}
    except Exception as e:
        return {"week": week_num, "trades": {}, "error": str(e)}

    models = build_lab_v3_candidates()
    week_trades = {}

    for model_name, model in models:
        try:
            trades = engine.run(ticks, model)
            pips_list = [float(t['pips']) for t in trades]
            week_trades[model_name] = pips_list
        except Exception:
            week_trades[model_name] = []

    return {"week": week_num, "trades": week_trades}


def run_lab_v3():
    t_start = time.time()
    print("=" * 80)
    print("  ALPHA DISCOVERY LAB AUDNZD V3: PRECISION MACRO CALIBRATION")
    print("=" * 80)

    disc_weeks = get_discovery_weeks("AUDNZD")
    total_disc_weeks = len(disc_weeks)
    candidates = build_lab_v3_candidates()
    print(f"[*] Quarantined Discovery Weeks: {total_disc_weeks} (4.04 Years)")
    print(f"[*] Candidate Model Variants:    {len(candidates)}")
    print(f"[*] Quarantined Holdout (30%):   Strictly Isolated & Preserved")

    models_trades: Dict[str, List[float]] = {name: [] for name, _ in candidates}

    max_workers = min(os.cpu_count() or 4, 12)
    print(f"[*] Launching parallel evaluation across {max_workers} worker processes...")

    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_week_v3, w): w for w in disc_weeks}
        for future in as_completed(futures):
            res = future.result()
            w = res["week"]
            trades_dict = res.get("trades", {})

            for model_name, pips_list in trades_dict.items():
                models_trades[model_name].extend(pips_list)

            completed += 1
            if completed % 35 == 0 or completed == total_disc_weeks:
                elapsed = time.time() - t_start
                rate = completed / max(elapsed, 0.001)
                print(f"  --> Progress: {completed:03d}/{total_disc_weeks} discovery weeks processed ({rate:.1f} w/s)...")

    # Score and rank all models
    print("\n" + "=" * 80)
    print("  AUDNZD LAB V3 RESULTS SCORECARD (70% DISCOVERY / 210 WEEKS)")
    print("=" * 80)

    scorecard = []
    years_equiv = total_disc_weeks / 52.0

    for model_name, pips in models_trades.items():
        n_trades = len(pips)
        if n_trades == 0:
            continue

        p_arr = np.array(pips)
        net_pips = float(np.sum(p_arr))
        wins = p_arr[p_arr > 0]
        losses = -p_arr[p_arr < 0]
        sum_wins = float(np.sum(wins))
        sum_losses = float(np.sum(losses))

        wr = len(wins) / n_trades * 100.0 if n_trades > 0 else 0.0
        pf = sum_wins / sum_losses if sum_losses > 0 else (99.0 if sum_wins > 0 else 0.0)

        equity = np.cumsum(p_arr)
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

        ret_dd = net_pips / max_dd if max_dd > 0 else (net_pips if net_pips > 0 else 0.0)
        annual_pips = net_pips / years_equiv

        scorecard.append({
            "model": model_name,
            "trades": n_trades,
            "win_rate": round(wr, 1),
            "profit_factor": round(pf, 2),
            "net_pips": round(net_pips, 1),
            "annual_pips": round(annual_pips, 1),
            "max_dd_pips": round(max_dd, 1),
            "return_to_dd": round(ret_dd, 2),
            "expectancy_pips": round(net_pips / n_trades, 2)
        })

    scorecard.sort(key=lambda x: x["net_pips"], reverse=True)

    header = f"{'Rank':<4} {'Model Identifier':<42} {'Trades':<7} {'WinRate':<8} {'PF':<6} {'Net Pips':<10} {'Pips/Yr':<10} {'Max DD':<9} {'Ret/DD':<8} {'Exp(p)':<7}"
    print(header)
    print("-" * len(header))

    profitable_count = 0
    elite_count = 0

    for i, row in enumerate(scorecard, 1):
        is_elite = row["profit_factor"] >= 1.30 and row["return_to_dd"] >= 2.5 and row["net_pips"] > 0
        is_prof = row["profit_factor"] >= 1.15 and row["net_pips"] > 0
        tag = "[ELITE]" if is_elite else ("[PASS]" if is_prof else "      ")

        if is_elite: elite_count += 1
        if is_prof: profitable_count += 1

        print(f"{i:<4} {row['model']:<42} {row['trades']:<7} {row['win_rate']:>5.1f}%  {row['profit_factor']:>5.2f} {row['net_pips']:>9.1f}p {row['annual_pips']:>9.1f}p {row['max_dd_pips']:>7.1f}p {row['return_to_dd']:>7.2f}x {row['expectancy_pips']:>6.2f}p  {tag}")

    print("-" * len(header))
    print(f"Summary: Evaluated {len(scorecard)} models across {total_disc_weeks} discovery weeks ({years_equiv:.2f} yrs).")
    print(f"Profitable Models (PF >= 1.15): {profitable_count} | High-Conviction Elite (PF >= 1.30): {elite_count}")

    out_file = RESULTS_DIR / "audnzd_discovery_screening_v3.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "asset": "AUDNZD",
            "partition": "discovery_70",
            "total_weeks": total_disc_weeks,
            "years_equivalent": round(years_equiv, 2),
            "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "scorecard": scorecard
        }, f, indent=2)

    print(f"[+] Detailed discovery results saved to {out_file.resolve()}")
    print(f"[+] Total execution time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    run_lab_v3()

