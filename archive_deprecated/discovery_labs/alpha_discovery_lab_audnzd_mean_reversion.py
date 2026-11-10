#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB: AUDNZD MEAN-REVERSION & LIQUIDITY SWEEP (OPTIMIZED SPEED)
================================================================================
Evaluates 30+ specialized cointegration-friendly mean-reversion architectures:
  1. Sydney Range False Breakout Liquidity Sweep Fade ("Turtle Soup")
  2. Multi-Day Fibonacci Golden Pocket (LB36, LB48, LB60, LB72) with Pin and Engulfing
  3. Macro H4 Donchian Retest Multipliers (2.0R to 3.0R)
  4. Tokyo Fix Overextension Reversion Fades

Optimized: M15/H1/H4 bars precomputed ONCE per week for ultra-high throughput.
Evaluated on the 70% Quarantined Discovery Partition (210 weeks / 97.08M ticks).
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

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset, get_discovery_weeks
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars

PIP_SIZE = 0.0001
MAX_SPREAD = 2.8
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Base Strategy
# ==============================================================================
class FastAUDNZDStrategy:
    def __init__(self, name: str, sl_pips: float, tp_pips: float):
        self.name = name
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)

    def generate_signals_from_bars(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        raise NotImplementedError


# ==============================================================================
# 1. Sydney Range False Breakout Liquidity Sweep Fade ("Turtle Soup")
# ==============================================================================
class AUDNZDSydneySweepFade(FastAUDNZDStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, sweep_pips: float = 2.5):
        super().__init__(name=name, sl_pips=sl_pips, tp_pips=tp_pips)
        self.sweep_min = float(sweep_pips) * PIP_SIZE

    def generate_signals_from_bars(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)

        if len(m15) < 30:
            return signals

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

        # Precompute Sydney 21:00-22:45 range per day
        syd_high = {}
        syd_low = {}
        unique_dates = np.unique(dates)
        for d in unique_dates:
            mask = (dates == d) & ((hours == 21) | (hours == 22))
            if np.any(mask):
                syd_high[d] = np.max(h[mask])
                syd_low[d] = np.min(l[mask])

        last_trade_date = None

        for i in range(15, len(m15)):
            hr = hours[i]
            # Fade window: 23:00 to 05:00 UTC (Tokyo active window)
            if not (hr >= 23 or hr <= 5):
                continue
            if spread[i] > MAX_SPREAD:
                continue
            dt = dates[i]
            if dt == last_trade_date or dt not in syd_high:
                continue

            sh = syd_high[dt]
            sl = syd_low[dt]
            bar_rng = h[i] - l[i]
            if bar_rng <= 0:
                continue

            # Sweep of Sydney High -> Bearish False Breakout Fade
            if h[i] >= sh + self.sweep_min and c[i] < sh:
                upper_wick = h[i] - max(o[i], c[i])
                if (upper_wick / bar_rng) >= 0.35 and c[i] < o[i]:
                    idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                    if idx < n:
                        signals[idx] = SIGNAL_SHORT
                        last_trade_date = dt

            # Sweep of Sydney Low -> Bullish False Breakout Fade
            elif l[i] <= sl - self.sweep_min and c[i] > sl:
                lower_wick = min(o[i], c[i]) - l[i]
                if (lower_wick / bar_rng) >= 0.35 and c[i] > o[i]:
                    idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                    if idx < n:
                        signals[idx] = SIGNAL_LONG
                        last_trade_date = dt

        return signals


# ==============================================================================
# 2. Multi-Day Fibonacci Golden Pocket (Enhanced with Engulfing and Lookbacks)
# ==============================================================================
class AUDNZDFibGoldenPocketEnhanced(FastAUDNZDStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, lookback_h1: int = 48, use_engulfing: bool = False):
        super().__init__(name=name, sl_pips=sl_pips, tp_pips=tp_pips)
        self.lookback_h1 = int(lookback_h1)
        self.use_engulfing = bool(use_engulfing)

    def generate_signals_from_bars(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)

        if len(m15) < 30 or len(h1) < self.lookback_h1 or len(h4) < 15:
            return signals

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_trend = np.where(h4['close'] > h4_ema20, 1, -1)
        h4_bias = pd.Series(h4_trend, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        h1_high = h1['high'].rolling(self.lookback_h1).max().shift(1)
        h1_low = h1['low'].rolling(self.lookback_h1).min().shift(1)
        m15_swing_h = h1_high.reindex(m15.index, method='ffill').values
        m15_swing_l = h1_low.reindex(m15.index, method='ffill').values

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_entry_bar = -999

        for i in range(self.lookback_h1, len(m15)):
            if i - last_entry_bar < 6:
                continue
            hr = hours[i]
            if not (hr >= 21 or hr <= 16):
                continue
            if spread[i] > MAX_SPREAD:
                continue

            bias = h4_bias[i]
            sw_h = m15_swing_h[i]
            sw_l = m15_swing_l[i]
            if np.isnan(sw_h) or np.isnan(sw_l):
                continue
            rng = sw_h - sw_l
            if rng < 25.0 * PIP_SIZE:
                continue

            bar_rng = h[i] - l[i]
            if bar_rng <= 0:
                continue

            if bias > 0:
                fib_618 = sw_h - 0.618 * rng
                fib_786 = sw_h - 0.786 * rng
                if l[i] <= fib_618 and c[i] >= fib_786:
                    trigger = False
                    if not self.use_engulfing:
                        lower_wick = min(o[i], c[i]) - l[i]
                        trigger = (lower_wick / bar_rng) >= 0.35 and c[i] > o[i]
                    else:
                        # Bullish engulfing of previous candle
                        trigger = c[i] > o[i] and c[i] > h[i-1] and o[i] < c[i-1]
                    
                    if trigger:
                        idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                        if idx < n:
                            signals[idx] = SIGNAL_LONG
                            last_entry_bar = i

            elif bias < 0:
                fib_618 = sw_l + 0.618 * rng
                fib_786 = sw_l + 0.786 * rng
                if h[i] >= fib_618 and c[i] <= fib_786:
                    trigger = False
                    if not self.use_engulfing:
                        upper_wick = h[i] - max(o[i], c[i])
                        trigger = (upper_wick / bar_rng) >= 0.35 and c[i] < o[i]
                    else:
                        # Bearish engulfing of previous candle
                        trigger = c[i] < o[i] and c[i] < l[i-1] and o[i] > c[i-1]

                    if trigger:
                        idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                        if idx < n:
                            signals[idx] = SIGNAL_SHORT
                            last_entry_bar = i

        return signals


# ==============================================================================
# 3. Macro H4 Donchian Retest Multipliers (2.0R to 3.0R)
# ==============================================================================
class AUDNZDH4MacroDonchianRetest(FastAUDNZDStrategy):
    def __init__(self, name: str, sl_pips: float, tp_pips: float, lookback_bars: int = 10, min_wick: float = 0.40):
        super().__init__(name=name, sl_pips=sl_pips, tp_pips=tp_pips)
        self.lookback_bars = int(lookback_bars)
        self.min_wick = float(min_wick)

    def generate_signals_from_bars(self, ticks: np.ndarray, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)

        if len(m15) < 30 or len(h4) < self.lookback_bars + 5:
            return signals

        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_bias = np.where(h4['close'] > h4_ema20, 1, -1)
        h4_high = h4['high'].rolling(self.lookback_bars).max().shift(1).bfill()
        h4_low = h4['low'].rolling(self.lookback_bars).min().shift(1).bfill()

        bias_m15 = pd.Series(h4_bias, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values
        res_m15 = h4_high.reindex(m15.index, method='ffill').values
        supp_m15 = h4_low.reindex(m15.index, method='ffill').values

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
            if not (hr >= 21 or hr <= 16):
                continue
            if spread[i] > MAX_SPREAD:
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue

            bias = bias_m15[i]
            supp = supp_m15[i]
            res = res_m15[i]
            if np.isnan(supp) or np.isnan(res):
                continue

            bar_rng = h[i] - l[i]
            if bar_rng <= 0:
                continue

            # Bullish retest of prior H4 support
            if bias > 0 and abs(l[i] - supp) <= 3.0 * PIP_SIZE and c[i] > supp:
                lower_wick = min(o[i], c[i]) - l[i]
                if (lower_wick / bar_rng) >= self.min_wick and c[i] > o[i]:
                    idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                    if idx < n:
                        signals[idx] = SIGNAL_LONG
                        last_trade_date = dt

            # Bearish retest of prior H4 resistance
            elif bias < 0 and abs(h[i] - res) <= 3.0 * PIP_SIZE and c[i] < res:
                upper_wick = h[i] - max(o[i], c[i])
                if (upper_wick / bar_rng) >= self.min_wick and c[i] < o[i]:
                    idx = np.searchsorted(tick_ts, time_ms[i] + dur_ms)
                    if idx < n:
                        signals[idx] = SIGNAL_SHORT
                        last_trade_date = dt

        return signals


# ==============================================================================
# Build Screening Fleet
# ==============================================================================
def build_audnzd_mean_reversion_fleet() -> List[FastAUDNZDStrategy]:
    fleet = []

    # 1. Sydney Sweep Fade ("Turtle Soup")
    for sweep in [2.0, 2.5, 3.0]:
        for sl, tp in [(10.0, 20.0), (12.0, 24.0), (14.0, 28.0), (12.0, 20.0)]:
            name = f"AUDNZD_SydneySweep_Sw{sweep}_SL{sl}_TP{tp}"
            fleet.append(AUDNZDSydneySweepFade(name=name, sl_pips=sl, tp_pips=tp, sweep_pips=sweep))

    # 2. Enhanced Fibonacci Golden Pocket (Lookbacks 36, 48, 60, 72 with Pin and Engulfing)
    for lb in [36, 48, 60, 72]:
        for use_eng in [False, True]:
            eng_str = "Engulf" if use_eng else "Pin"
            for sl, tp in [(12.0, 24.0), (14.0, 28.0), (12.0, 30.0)]:
                name = f"AUDNZD_Fib_{eng_str}_LB{lb}_SL{sl}_TP{tp}"
                fleet.append(AUDNZDFibGoldenPocketEnhanced(name=name, sl_pips=sl, tp_pips=tp, lookback_h1=lb, use_engulfing=use_eng))

    # 3. H4 Donchian Retest Multipliers
    for lb in [10, 15, 20]:
        for sl, tp in [(12.0, 24.0), (12.0, 30.0), (14.0, 28.0), (14.0, 35.0)]:
            name = f"AUDNZD_H4Donchian_LB{lb}_SL{sl}_TP{tp}"
            fleet.append(AUDNZDH4MacroDonchianRetest(name=name, sl_pips=sl, tp_pips=tp, lookback_bars=lb))

    return fleet


# ==============================================================================
# Worker Function (Bars built ONCE per week)
# ==============================================================================
def evaluate_week_batch(week_num: int) -> Dict[str, List[Dict[str, Any]]]:
    t_res = get_week_dataset(week_num, raw_dir="data/AUDNZD")
    if t_res is None:
        return {}
    ticks, meta = t_res if isinstance(t_res, tuple) else (t_res, None)
    if ticks is None or len(ticks) < 1000:
        return {}

    m15 = build_time_bars(ticks, timeframe_min=15, pip_size=PIP_SIZE)
    if len(m15) < 30:
        return {}
    h1 = m15.resample('1h').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()
    h4 = m15.resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'}).dropna()

    fleet = build_audnzd_mean_reversion_fleet()
    week_results = {}

    for strat in fleet:
        signals = strat.generate_signals_from_bars(ticks, m15, h1, h4)
        trade_log = run_matching_loop(ticks, signals, pip_size=PIP_SIZE, sl_pips=strat.sl_pips, tp_pips=strat.tp_pips)

        parsed_trades = []
        for t in trade_log:
            parsed_trades.append({
                "pnl_pips": float(t['pips']),
                "is_win": bool(t['pips'] > 0),
                "sl_pips": strat.sl_pips,
            })
        week_results[strat.name] = parsed_trades

    return week_results


# ==============================================================================
# Main Discovery Runner
# ==============================================================================
def main():
    start_time = time.time()
    print("=" * 80)
    print("  AUDNZD MEAN-REVERSION & LIQUIDITY SWEEP DISCOVERY (210 WEEKS)")
    print("=" * 80)

    discovery_weeks = get_discovery_weeks(raw_dir="data/AUDNZD")
    print(f"Loaded {len(discovery_weeks)} discovery weeks for AUDNZD.")

    sample_fleet = build_audnzd_mean_reversion_fleet()
    print(f"Screening {len(sample_fleet)} architectures (bars precomputed once per week)...")

    fleet_trades: Dict[str, List[Dict[str, Any]]] = {strat.name: [] for strat in sample_fleet}

    completed = 0
    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(evaluate_week_batch, w): w for w in discovery_weeks}
        for future in as_completed(futures):
            res = future.result()
            for name, trades in res.items():
                fleet_trades[name].extend(trades)
            completed += 1
            if completed % 35 == 0 or completed == len(discovery_weeks):
                pct = completed / len(discovery_weeks) * 100
                print(f"  Progress: {completed}/{len(discovery_weeks)} weeks ({pct:.1f}%) in {time.time() - start_time:.1f}s")

    print("\nProcessing metrics and ranking candidates by Annualized Net Return...")

    DISCOVERY_YEARS = 4.04  # 210 weeks / 52.0
    scorecard = []

    for strat in sample_fleet:
        trades = fleet_trades[strat.name]
        n_trades = len(trades)
        if n_trades < 30:
            continue

        pnls = np.array([t["pnl_pips"] for t in trades])
        sl_val = trades[0]["sl_pips"]
        net_pips = float(np.sum(pnls))
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]

        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 1e-9
        pf = gross_profit / gross_loss
        wr = len(wins) / n_trades * 100.0

        cum = np.cumsum(pnls)
        peaks = np.maximum.accumulate(cum)
        dd_series = peaks - cum
        max_dd = float(np.max(dd_series)) if len(dd_series) > 0 else 1.0
        ret_to_dd = net_pips / max_dd if max_dd > 0 else 0.0

        ann_pips = net_pips / DISCOVERY_YEARS
        ann_ret_pct = (ann_pips / sl_val) * 1.0 * 100.0  # 1% risk per trade

        scorecard.append({
            "name": strat.name,
            "sl": sl_val,
            "trades": n_trades,
            "net_pips": round(net_pips, 1),
            "ann_pips": round(ann_pips, 1),
            "ann_ret_pct": round(ann_ret_pct, 1),
            "profit_factor": round(pf, 2),
            "win_rate": round(wr, 1),
            "max_dd": round(max_dd, 1),
            "ret_to_dd": round(ret_to_dd, 2),
            "exp_pips": round(net_pips / n_trades, 2)
        })

    if len(scorecard) == 0:
        print("No models met the trade count threshold.")
        return

    df = pd.DataFrame(scorecard)
    df = df.sort_values(by="ann_ret_pct", ascending=False)

    out_csv = RESULTS_DIR / "audnzd_mean_reversion_discovery_scorecard.csv"
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 120)
    print("  TOP AUDNZD MEAN-REVERSION CANDIDATES RANKED BY ANNUALIZED RETURN %")
    print("=" * 120)
    print(f"{'Rank':<4} | {'Strategy Name':<42} | {'Trades':<6} | {'Net Pips':<9} | {'Ann Ret%':<8} | {'PF':<5} | {'WR%':<5} | {'Max DD':<7} | {'Ret/DD':<7} | {'Exp'}")
    print("-" * 120)

    top_candidates = df.head(30)
    for idx, (_, row) in enumerate(top_candidates.iterrows(), 1):
        flag = "⭐" if row['ann_ret_pct'] >= 15.0 and row['profit_factor'] >= 1.25 else "  "
        print(f"{idx:<4} | {row['name']:<42} | {int(row['trades']):<6} | {row['net_pips']:<+9.1f} | {row['ann_ret_pct']:<+7.1f}% | {row['profit_factor']:<5.2f} | {row['win_rate']:<5.1f} | {row['max_dd']:<7.1f} | {row['ret_to_dd']:<6.2f}x | {row['exp_pips']:<+5.2f} {flag}")

    print("-" * 120)
    qualified = df[(df['ann_ret_pct'] >= 15.0) & (df['profit_factor'] >= 1.25) & (df['ret_to_dd'] >= 4.0)]
    print(f"Total architectures exceeding >= 15.0% annual return and PF >= 1.25: {len(qualified)}")
    print(f"Detailed scorecard saved to: {out_csv}")
    print(f"Total Elapsed Time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()

