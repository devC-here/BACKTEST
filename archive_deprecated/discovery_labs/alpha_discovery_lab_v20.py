#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V20: VOLUME DELTA ABSORPTION, FIX CONTINUATION & RE-ACCUMULATION
================================================================================
Explores 4 novel algorithmic architectures across the 70% In-Sample Discovery
partition (Seed 42) across EURUSD, USDJPY, and EURGBP (603 discovery weeks):

  1. DELTA_DIVERGENCE_REVERSAL:
     - Donchian extreme high/low with tick volume delta divergence (passive absorption).
     - Counter-trend fade towards 20-EMA.
     - Target: 1.5R.

  2. MONDAY_OPEN_GAP_FILL:
     - Monday (00:00 - 08:00 UTC) gap fade when price is displaced >= 5.0 pips from Friday close.
     - Target: 1.5R.

  3. LONDON_CLOSE_TREND_EXPANSION:
     - 15:45 - 17:30 UTC breakout of pre-fix 14:00 - 15:45 consolidation range in H4 trend direction.
     - Target: 2.0R.

  4. DONCHIAN_EMA_PULLBACK:
     - Established Donchian trend with 2-4 bar pullback into M15 20-EMA and rejection wick (>= 45%).
     - Target: 2.0R.
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


# ==============================================================================
# Model 1: Volume Delta Divergence Reversal (DISC75)
# ==============================================================================
class VolumeDeltaAbsorptionFade(BaseStrategy):
    """
    On M15, price reaches 20-bar Donchian extreme.
    If bar is a new high but net uptick-downtick volume delta is negative (passive absorption),
    or new low with positive volume delta, triggers fade.
    Target: 1.5R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Delta_Absorption_Fade"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 35:
            return signals, sl_arr, tp_arr

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

        # Approximate volume delta using bar body ratio * tick count
        bar_vols = m15['tick_count'].values if 'tick_count' in m15.columns else np.ones(len(m15))
        rng = np.maximum(h - l, 1e-6)
        delta_approx = ((c - o) / rng) * bar_vols
        cum_delta_5 = pd.Series(delta_approx).rolling(5).sum().values

        # Donchian 20
        h_s = pd.Series(h)
        l_s = pd.Series(l)
        donch_hi = h_s.rolling(20).max().shift(1).values
        donch_lo = l_s.rolling(20).min().shift(1).values

        last_trade_date = None

        for i in range(25, len(m15)):
            dt = dates[i]
            hr = hours[i]

            if not (7 <= hr <= 16):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue
            if np.isnan(donch_hi[i]) or np.isnan(donch_lo[i]):
                continue

            bar_rng = h[i] - l[i]
            if bar_rng < 2.5 * self.pip_size:
                continue

            sig = 0
            # New high above donchian, but 5-bar cumulative delta is strictly negative (heavy hidden selling)
            if h[i] > donch_hi[i] and cum_delta_5[i] < -0.15 * np.sum(bar_vols[i-4:i+1]):
                upper_wick = (h[i] - max(o[i], c[i])) / bar_rng
                if upper_wick >= 0.35:
                    sig = SIGNAL_SHORT

            # New low below donchian, but 5-bar cumulative delta is strictly positive (heavy hidden buying)
            elif l[i] < donch_lo[i] and cum_delta_5[i] > 0.15 * np.sum(bar_vols[i-4:i+1]):
                lower_wick = (min(o[i], c[i]) - l[i]) / bar_rng
                if lower_wick >= 0.35:
                    sig = SIGNAL_LONG

            if sig != 0:
                bar_close_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, bar_close_ms)
                if idx < n:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 2: Monday Opening Gap Fill (DISC76)
# ==============================================================================
class MondayOpeningGapFade(BaseStrategy):
    """
    Detects if Monday open (00:00 - 08:00 UTC) is displaced >= 4.5 pips from the first bar's open.
    Enters fade back towards the baseline.
    Target: 1.5R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Monday_Gap_Fade"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 20:
            return signals, sl_arr, tp_arr

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        day_of_week = m15.index.dayofweek.values
        hours = m15.index.hour.values
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None
        week_open_price = o[0]

        for i in range(5, len(m15)):
            dt = dates[i]
            dow = day_of_week[i]
            hr = hours[i]

            # Monday only (dow == 0) between 01:00 - 08:00 UTC
            if dow != 0 or not (1 <= hr <= 8):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            gap = c[i] - week_open_price
            gap_pips = abs(gap) / self.pip_size

            if gap_pips < 4.5:
                continue

            sig = 0
            # Gapped up away from week open -> Short fade
            if gap > 0 and c[i] < o[i]:
                sig = SIGNAL_SHORT
            # Gapped down away from week open -> Long fade
            elif gap < 0 and c[i] > o[i]:
                sig = SIGNAL_LONG

            if sig != 0:
                bar_close_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, bar_close_ms)
                if idx < n:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 3: London Close / 16:00 Fix Breakout Surge (DISC77)
# ==============================================================================
class LondonFixExpansionSurge(BaseStrategy):
    """
    Between 15:45 - 17:30 UTC.
    Measures 14:00 - 15:45 UTC pre-fix range.
    Triggers breakout in H4 20-EMA trend direction.
    Target: 2.0R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_London_Fix_Surge"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m15) < 35 or len(h4) < 10:
            return signals, sl_arr, tp_arr

        h4_c = pd.Series(h4['close'].values)
        h4_ema20 = h4_c.ewm(span=20, adjust=False).mean().values
        h4_bias = np.where(h4_c.values > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'h4_bias': h4_bias}, index=h4.index)

        m15_df = pd.merge_asof(
            m15.sort_index(),
            h4_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )

        c = m15_df['close'].values
        h = m15_df['high'].values
        l = m15_df['low'].values
        spread = m15_df['spread_pips'].values
        h4_trend = m15_df['h4_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        minutes = m15_df.index.minute.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None

        # Precompute 14:00 - 15:45 range per date
        fix_ranges: Dict[Any, Tuple[float, float]] = {}
        for d in np.unique(dates):
            mask = (dates == d) & (hours >= 14) & ((hours < 15) | ((hours == 15) & (minutes <= 45)))
            if np.sum(mask) >= 6:
                fix_ranges[d] = (float(np.max(h[mask])), float(np.min(l[mask])))

        for i in range(25, len(m15_df)):
            dt = dates[i]
            hr = hours[i]
            mn = minutes[i]

            in_window = (hr == 15 and mn >= 45) or (hr == 16) or (hr == 17 and mn <= 15)
            if not in_window:
                continue
            if dt == last_trade_date:
                continue
            if dt not in fix_ranges:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            fix_hi, fix_lo = fix_ranges[dt]
            fix_rng = fix_hi - fix_lo
            if fix_rng < 5.0 * self.pip_size:
                continue

            sig = 0
            if macro == 1 and c[i] > fix_hi and c[i-1] <= fix_hi:
                sig = SIGNAL_LONG
            elif macro == -1 and c[i] < fix_lo and c[i-1] >= fix_lo:
                sig = SIGNAL_SHORT

            if sig != 0:
                bar_close_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, bar_close_ms)
                if idx < n:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 4: Donchian Trend EMA Pullback Re-Accumulation (DISC78)
# ==============================================================================
class DonchianEMAPullback(BaseStrategy):
    """
    On M15, 20-period Donchian high is established.
    Price pulls back 2 to 4 bars to touch the M15 20-EMA.
    Prints rejection wick >= 45% in H4 20-EMA trend direction.
    Target: 2.0R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Donchian_EMA_Pullback"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m15) < 35 or len(h4) < 10:
            return signals, sl_arr, tp_arr

        h4_c = pd.Series(h4['close'].values)
        h4_ema20 = h4_c.ewm(span=20, adjust=False).mean().values
        h4_bias = np.where(h4_c.values > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'h4_bias': h4_bias}, index=h4.index)

        m15_df = pd.merge_asof(
            m15.sort_index(),
            h4_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )

        c = m15_df['close'].values
        h = m15_df['high'].values
        l = m15_df['low'].values
        o = m15_df['open'].values
        spread = m15_df['spread_pips'].values
        h4_trend = m15_df['h4_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        c_s = pd.Series(c)
        m15_ema20 = c_s.ewm(span=20, adjust=False).mean().values

        last_trade_date = None

        for i in range(25, len(m15_df)):
            dt = dates[i]
            hr = hours[i]

            if not (7 <= hr <= 16):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            bar_rng = h[i] - l[i]
            if bar_rng < 2.5 * self.pip_size:
                continue

            sig = 0
            tol = 1.5 * self.pip_size

            # Bullish pullback: touched 20-EMA, closed above, strong lower wick
            if macro == 1 and l[i] <= m15_ema20[i] + tol and c[i] > m15_ema20[i]:
                lower_wick = (min(o[i], c[i]) - l[i]) / bar_rng
                if lower_wick >= 0.45:
                    sig = SIGNAL_LONG

            # Bearish pullback: touched 20-EMA, closed below, strong upper wick
            elif macro == -1 and h[i] >= m15_ema20[i] - tol and c[i] < m15_ema20[i]:
                upper_wick = (h[i] - max(o[i], c[i])) / bar_rng
                if upper_wick >= 0.45:
                    sig = SIGNAL_SHORT

            if sig != 0:
                bar_close_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, bar_close_ms)
                if idx < n:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model Factory & Multiprocessing Orchestrator
# ==============================================================================
def build_model_v20(model_name: str, asset: str):
    if model_name == "DELTA_ABSORPTION_FADE":
        if asset == "EURUSD":
            return VolumeDeltaAbsorptionFade(asset=asset, sl_pips=9.0, tp_pips=13.5, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return VolumeDeltaAbsorptionFade(asset=asset, sl_pips=12.0, tp_pips=18.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return VolumeDeltaAbsorptionFade(asset=asset, sl_pips=7.5, tp_pips=11.25, pip_size=0.0001, max_spread=0.8)
    elif model_name == "MONDAY_GAP_FADE":
        if asset == "EURUSD":
            return MondayOpeningGapFade(asset=asset, sl_pips=9.0, tp_pips=13.5, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return MondayOpeningGapFade(asset=asset, sl_pips=12.0, tp_pips=18.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return MondayOpeningGapFade(asset=asset, sl_pips=7.5, tp_pips=11.25, pip_size=0.0001, max_spread=0.8)
    elif model_name == "LONDON_FIX_SURGE":
        if asset == "EURUSD":
            return LondonFixExpansionSurge(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return LondonFixExpansionSurge(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return LondonFixExpansionSurge(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)
    elif model_name == "DONCHIAN_EMA_PULLBACK":
        if asset == "EURUSD":
            return DonchianEMAPullback(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return DonchianEMAPullback(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return DonchianEMAPullback(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)
    return None


def run_week_task_v20(task: Tuple[str, str, int]) -> Dict[str, Any]:
    asset, model_name, week_num = task
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if ticks is None or len(ticks) < 500:
        return {"asset": asset, "model": model_name, "week": week_num, "trades": []}

    strat = build_model_v20(model_name, asset)
    if strat is None:
        return {"asset": asset, "model": model_name, "week": week_num, "trades": []}

    try:
        raw_trades = engine.run(ticks, strat)
        t_list = []
        for i in range(len(raw_trades)):
            row = raw_trades[i]
            t_list.append({
                "pips": float(row["pips"]),
                "direction": int(row["direction"])
            })
        return {"asset": asset, "model": model_name, "week": week_num, "trades": t_list}
    except Exception:
        return {"asset": asset, "model": model_name, "week": week_num, "trades": []}


def calculate_metrics(pips: List[float]) -> Dict[str, Any]:
    if not pips:
        return {"trades": 0, "win_rate": 0.0, "pf": 0.0, "net_pips": 0.0, "max_dd": 0.0, "ret_dd": 0.0, "expectancy": 0.0}
    arr = np.array(pips, dtype=np.float64)
    trades = len(arr)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    wr = len(wins) / trades * 100.0 if trades > 0 else 0.0
    gross_win = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else 999.0
    net_pips = float(np.sum(arr))
    expectancy = net_pips / trades if trades > 0 else 0.0

    cumsum = np.cumsum(arr)
    running_max = np.maximum.accumulate(cumsum)
    drawdowns = running_max - cumsum
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    ret_dd = net_pips / max_dd if max_dd > 0 else 0.0

    return {
        "trades": trades,
        "win_rate": round(wr, 1),
        "pf": round(pf, 2),
        "net_pips": round(net_pips, 1),
        "max_dd": round(max_dd, 1),
        "ret_dd": round(ret_dd, 2),
        "expectancy": round(expectancy, 2)
    }


def main():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("  LAUNCHING ALPHA DISCOVERY LAB V20: ABSORPTION, FIX & DONCHIAN PULLBACK", flush=True)
    print("=" * 80, flush=True)

    assets = [os.environ["DISCOVERY_TARGET_SYMBOL"]] if "DISCOVERY_TARGET_SYMBOL" in os.environ else ["EURUSD", "USDJPY", "EURGBP"]
    models = [
        "DELTA_ABSORPTION_FADE",
        "MONDAY_GAP_FADE",
        "LONDON_FIX_SURGE",
        "DONCHIAN_EMA_PULLBACK"
    ]

    all_tasks = []
    for asset in assets:
        disc_weeks = get_discovery_weeks(f"data/{asset}")
        for m_name in models:
            for w in disc_weeks:
                all_tasks.append((asset, m_name, w))

    print(f"[*] Total week-tasks across 3 assets & 4 models: {len(all_tasks)}", flush=True)

    trades_by_key: Dict[Tuple[str, str], List[float]] = {}
    completed = 0

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(run_week_task_v20, t): t for t in all_tasks}
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            if completed % 300 == 0 or completed == len(all_tasks):
                elapsed = time.time() - t_start
                print(f"  [+] Audited {completed}/{len(all_tasks)} week-tasks in {elapsed:.1f}s...", flush=True)

            k = (res["asset"], res["model"])
            for tr in res["trades"]:
                trades_by_key.setdefault(k, []).append(tr["pips"])

    print("\n" + "=" * 90, flush=True)
    print(f"{'ASSET':<8} | {'MODEL':<24} | {'TRADES':<6} | {'WIN RATE':<8} | {'PF':<5} | {'NET PIPS':<10} | {'MAX DD':<8} | {'RET/DD':<7}", flush=True)
    print("=" * 90, flush=True)

    summary_records = []
    for asset in assets:
        for model in models:
            pips = trades_by_key.get((asset, model), [])
            m = calculate_metrics(pips)
            rec = {
                "asset": asset,
                "model": model,
                **m
            }
            summary_records.append(rec)
            print(f"{asset:<8} | {model:<24} | {m['trades']:<6} | {m['win_rate']:<7}% | {m['pf']:<5.2f} | {m['net_pips']:<10.1f} | {m['max_dd']:<8.1f} | {m['ret_dd']:<6.2f}x", flush=True)
        print("-" * 90, flush=True)

    out_file = RESULTS_DIR / "lab_v20_discovery_results.json"
    with open(out_file, "w") as f:
        json.dump(summary_records, f, indent=2)
    print(f"\n[+] Discovery results saved to: {out_file}", flush=True)
    print(f"[+] Total execution time: {time.time() - t_start:.2f}s", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
