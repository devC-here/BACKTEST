#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V15: PRE-FIX EXPANSION, STOCH-RSI, TOKYO FIX & TRIPLE EMA
================================================================================
Evaluates 4 Quantitative Microstructure Architectures across 70% Discovery:
  1. DISC74_PRE_FIX_BREAKOUT:
     Midday European consolidation (12:00 - 13:45 UTC). Decisive M15 breakout
     during London afternoon rebalancing (14:00 - 15:30 UTC) in H4 trend direction.
     2.0R target.
  2. DISC75_STOCH_RSI_PULLBACK:
     H4 20-EMA trend agreement. M15 StochRSI(14, 14, 3, 3) reaches extreme (< 20
     in uptrend, > 80 in downtrend) and hooks back during 07:00 - 17:00 UTC.
     2.0R target.
  3. DISC76_TOKYO_FIX_CONTINUATION:
     Post-Tokyo Fix (00:55 UTC) momentum. M15 pullback to 20-EMA between 01:30
     and 05:00 UTC in direction of Asian Fix drift. 1.8R target.
  4. DISC77_TRIPLE_EMA_M15_IMPULSE:
     M15 9-EMA crosses 21-EMA in direction of 55-EMA during 07:00 - 16:00 UTC
     with candle range >= 0.8x ATR. 2.0R target.

Tested across EURUSD, EURGBP, and USDJPY on 70% Discovery Partition ONLY.
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
# Model 1: London Pre-Fix Breakout (DISC74)
# ==============================================================================
class LondonPreFixBreakout(BaseStrategy):
    """
    Operates during London afternoon (14:00 - 15:30 UTC).
    Computes Midday consolidation range (12:00 - 13:45 UTC).
    First M15 candle closing outside the midday range in direction of H4 20-EMA.
    Target: 2.0R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Pre_Fix_Breakout"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m15) < 30 or len(h4) < 10:
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
        minutes = m15_df.index.minute.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Precompute midday ranges per date (12:00 to 13:45 UTC)
        unique_dates = np.unique(dates)
        midday_hi = {}
        midday_lo = {}
        for d in unique_dates:
            mask = (dates == d) & (((hours == 12)) | ((hours == 13) & (minutes <= 45)))
            if np.any(mask):
                midday_hi[d] = np.max(h[mask])
                midday_lo[d] = np.min(l[mask])

        last_trade_date = None
        for i in range(1, len(m15_df)):
            dt = dates[i]
            hr = hours[i]
            mn = minutes[i]

            # 14:00 to 15:30 UTC
            in_window = (hr == 14) or (hr == 15 and mn <= 30)
            if not in_window:
                continue
            if dt == last_trade_date:
                continue
            if dt not in midday_hi or dt not in midday_lo:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            hi_lvl = midday_hi[dt]
            lo_lvl = midday_lo[dt]
            rng_pips = (hi_lvl - lo_lvl) / self.pip_size
            if rng_pips < 6.0:
                continue

            sig = 0
            if macro == 1 and c[i] > hi_lvl and c[i-1] <= hi_lvl:
                sig = SIGNAL_LONG
            elif macro == -1 and c[i] < lo_lvl and c[i-1] >= lo_lvl:
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
# Model 2: StochRSI Pullback Hook (DISC75)
# ==============================================================================
class StochRSIPullbackHook(BaseStrategy):
    """
    Operates during active hours (07:00 - 17:00 UTC).
    H4 20-EMA trend agreement.
    Computes 14-period RSI on M15, then 14-period Stochastic of RSI.
    Bullish Hook: Uptrend, StochRSI was < 20, hooks above 20.
    Bearish Hook: Downtrend, StochRSI was > 80, hooks below 80.
    Target: 2.0R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Stoch_RSI_Hook"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m15) < 40 or len(h4) < 10:
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
        spread = m15_df['spread_pips'].values
        h4_trend = m15_df['h4_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Calculate RSI(14)
        delta = pd.Series(c).diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # StochRSI(14)
        rsi_min = rsi.rolling(14).min()
        rsi_max = rsi.rolling(14).max()
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan) * 100.0
        stoch_k = stoch_rsi.rolling(3).mean().values

        last_trade_date = None
        for i in range(30, len(m15_df)):
            if not (7 <= hours[i] <= 17):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0 or np.isnan(stoch_k[i]) or np.isnan(stoch_k[i-1]):
                continue

            sig = 0
            if macro == 1 and stoch_k[i-1] <= 20.0 and stoch_k[i] > 20.0:
                sig = SIGNAL_LONG
            elif macro == -1 and stoch_k[i-1] >= 80.0 and stoch_k[i] < 80.0:
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
# Model 3: Tokyo Fix Imbalance Continuation (DISC76)
# ==============================================================================
class TokyoFixImbalanceContinuation(BaseStrategy):
    """
    Operates during 01:30 - 05:00 UTC (post-Tokyo Fix Asian flow).
    Determines Fix direction: Close at 01:15 UTC vs Open at 00:00 UTC.
    M15 pulls back to touch 20-EMA and closes in Fix direction.
    Target: 1.8R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Tokyo_Fix_Continuation"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 25:
            return signals, sl_arr, tp_arr

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        minutes = m15.index.minute.values
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        c_s = pd.Series(c)
        ema20 = c_s.ewm(span=20, adjust=False).mean().values

        # Determine Tokyo Fix direction per day
        unique_dates = np.unique(dates)
        fix_dir = {}
        for d in unique_dates:
            mask_open = (dates == d) & (hours == 0) & (minutes == 0)
            mask_fix = (dates == d) & (hours == 1) & (minutes == 15)
            if np.any(mask_open) and np.any(mask_fix):
                open_val = o[mask_open][0]
                fix_val = c[mask_fix][0]
                diff = (fix_val - open_val) / self.pip_size
                if diff >= 5.0:
                    fix_dir[d] = 1
                elif diff <= -5.0:
                    fix_dir[d] = -1

        last_trade_date = None
        for i in range(10, len(m15)):
            dt = dates[i]
            hr = hours[i]
            if not (1 <= hr <= 5):
                continue
            if dt == last_trade_date:
                continue
            if dt not in fix_dir:
                continue
            if spread[i] > self.max_spread:
                continue

            direction = fix_dir[dt]
            sig = 0
            if direction == 1 and l[i] <= ema20[i] and c[i] > ema20[i] and c[i] > o[i]:
                sig = SIGNAL_LONG
            elif direction == -1 and h[i] >= ema20[i] and c[i] < ema20[i] and c[i] < o[i]:
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
# Model 4: Triple EMA Impulse Cross (DISC77)
# ==============================================================================
class TripleEMAImpulseCross(BaseStrategy):
    """
    Operates during European/US hours (07:00 - 16:00 UTC).
    M15 EMA 9 crosses 21 in direction of 55.
    Candle range >= 0.8x ATR(14) to confirm directional impulse.
    Target: 2.0R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Triple_EMA_Cross"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 60:
            return signals, sl_arr, tp_arr

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        c_s = pd.Series(c)
        ema9 = c_s.ewm(span=9, adjust=False).mean().values
        ema21 = c_s.ewm(span=21, adjust=False).mean().values
        ema55 = c_s.ewm(span=55, adjust=False).mean().values

        # ATR 14
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        tr = np.insert(tr, 0, h[0] - l[0])
        atr = pd.Series(tr).rolling(14).mean().values

        last_trade_date = None
        for i in range(56, len(m15)):
            if not (7 <= hours[i] <= 16):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            bar_range = h[i] - l[i]
            if bar_range < 0.8 * atr[i]:
                continue

            sig = 0
            # Bullish cross: EMA 9 crosses above EMA 21, both above EMA 55
            if ema9[i-1] <= ema21[i-1] and ema9[i] > ema21[i] and ema21[i] > ema55[i]:
                sig = SIGNAL_LONG
            # Bearish cross: EMA 9 crosses below EMA 21, both below EMA 55
            elif ema9[i-1] >= ema21[i-1] and ema9[i] < ema21[i] and ema21[i] < ema55[i]:
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
# Discovery Worker for Alpha Discovery Lab V15
# ==============================================================================
def evaluate_v15_discovery_week(task: Tuple[str, int]) -> Dict[str, Any]:
    asset, week_num = task
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if ticks is None or len(ticks) < 500:
        return {"asset": asset, "week": week_num, "trades": {}}

    models = []
    if asset == "EURUSD":
        models.append(("PRE_FIX_BO", LondonPreFixBreakout(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("STOCH_RSI_HOOK", StochRSIPullbackHook(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("TOKYO_FIX_CONT", TokyoFixImbalanceContinuation(asset=asset, sl_pips=9.0, tp_pips=16.2, pip_size=0.0001, max_spread=0.6)))
        models.append(("TRIPLE_EMA_CROSS", TripleEMAImpulseCross(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))
    elif asset == "USDJPY":
        models.append(("PRE_FIX_BO", LondonPreFixBreakout(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
        models.append(("STOCH_RSI_HOOK", StochRSIPullbackHook(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
        models.append(("TOKYO_FIX_CONT", TokyoFixImbalanceContinuation(asset=asset, sl_pips=12.0, tp_pips=21.6, pip_size=0.01, max_spread=0.8)))
        models.append(("TRIPLE_EMA_CROSS", TripleEMAImpulseCross(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
    elif asset == "EURGBP":
        models.append(("PRE_FIX_BO", LondonPreFixBreakout(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)))
        models.append(("STOCH_RSI_HOOK", StochRSIPullbackHook(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)))
        models.append(("TOKYO_FIX_CONT", TokyoFixImbalanceContinuation(asset=asset, sl_pips=8.0, tp_pips=14.4, pip_size=0.0001, max_spread=0.8)))
        models.append(("TRIPLE_EMA_CROSS", TripleEMAImpulseCross(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)))

    week_trades = {}
    for m_name, model in models:
        try:
            raw_trades = engine.run(ticks, model)
            t_list = []
            for i in range(len(raw_trades)):
                row = raw_trades[i]
                t_list.append({
                    "strategy": m_name,
                    "asset": asset,
                    "week": week_num,
                    "pips": float(row["pips"]),
                    "direction": int(row["direction"])
                })
            week_trades[m_name] = t_list
        except Exception:
            week_trades[m_name] = []

    return {"asset": asset, "week": week_num, "trades": week_trades}


def run_discovery_v15():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("  LAUNCHING ALPHA DISCOVERY V15 EXPEDITION (70% DISCOVERY PARTITIONS)", flush=True)
    print("=" * 80, flush=True)

    assets = [os.environ["DISCOVERY_TARGET_SYMBOL"]] if "DISCOVERY_TARGET_SYMBOL" in os.environ else ["USDJPY", "EURGBP", "EURUSD"]
    all_tasks = []
    for asset in assets:
        disc_weeks = get_discovery_weeks(f"data/{asset}")
        print(f"[*] Queued {len(disc_weeks)} discovery weeks for {asset}...", flush=True)
        for w in disc_weeks:
            all_tasks.append((asset, w))

    print(f"\n[*] Total discovery tasks across all assets: {len(all_tasks)} weeks", flush=True)

    results_by_key: Dict[Tuple[str, str], List[float]] = {}
    completed = 0

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(evaluate_v15_discovery_week, t): t for t in all_tasks}
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            asset = res["asset"]
            for m_name, t_list in res["trades"].items():
                key = (asset, m_name)
                if key not in results_by_key:
                    results_by_key[key] = []
                results_by_key[key].extend([t["pips"] for t in t_list])

            if completed % 100 == 0 or completed == len(all_tasks):
                print(f"  [+] Processed {completed:3d}/{len(all_tasks)} discovery weeks in {time.time() - t_start:.1f}s...", flush=True)

    # Scorecard
    print("\n" + "=" * 90, flush=True)
    print(f"{'ASSET':<8} {'MODEL NAME':<26} {'TRADES':<8} {'WIN RATE':<10} {'PROFIT FACTOR':<15} {'NET PIPS':<12} {'STATUS'}", flush=True)
    print("=" * 90, flush=True)

    summary_rows = []
    for (asset, m_name), pips in sorted(results_by_key.items()):
        n_t = len(pips)
        if n_t == 0:
            print(f"{asset:<8} {m_name:<26} {'0':<8} {'0.0%':<10} {'0.00':<15} {'+0.0p':<12} {'NO TRADES'}", flush=True)
            continue

        wins = [p for p in pips if p > 0]
        losses = [p for p in pips if p < 0]
        wr = len(wins) / n_t * 100.0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        pf = gross_profit / gross_loss if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
        net_pips = sum(pips)

        status = "REJECTED"
        if net_pips > 250.0 and pf >= 1.25 and n_t >= 40:
            status = "PASS_HOLDOUT_CANDIDATE"

        p_str = f"{'+' if net_pips >= 0 else ''}{net_pips:.1f}p"
        print(f"{asset:<8} {m_name:<26} {n_t:<8} {wr:5.1f}%     {pf:5.2f}           {p_str:<12} {status}", flush=True)
        summary_rows.append({
            "asset": asset,
            "model": m_name,
            "trades": n_t,
            "win_rate": round(wr, 1),
            "profit_factor": round(pf, 2),
            "net_pips": round(net_pips, 1),
            "status": status
        })

    print("=" * 90, flush=True)
    print(f"[*] Discovery Lab V15 complete in {time.time() - t_start:.2f} seconds.", flush=True)

    out_file = RESULTS_DIR / "alpha_discovery_lab_v15_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"[*] Results saved to {out_file}", flush=True)


if __name__ == "__main__":
    run_discovery_v15()
