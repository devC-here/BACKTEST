#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V27: LONDON FIX SURGE, SUPERTREND SQUEEZE,
  ASIAN LIQUIDITY SPRING & VWAP 2.5SD EXHAUSTION
================================================================================
Systematic exploration of 4 novel algorithmic market microstructures across
the 70% In-Sample Discovery Partition (Seed 42) across EURUSD, USDJPY, EURGBP:

  1. LONDON_FIX_REBALANCE_SURGE:
     - Capitalizes on the 16:00 London WMR Fix daily rebalance volume surge.
     - Between 15:15 - 16:30 UTC, M15 expansion candle (Range >= 1.20x ATR10, Body >= 50%)
       breaks beyond prior 16-bar (4-hour) high/low in H4 20-EMA trend direction. Target: 2.0R.

  2. SUPERTREND_KELTNER_SQUEEZE:
     - M15 Keltner/Bollinger volatility squeeze release in direction of H4 20-EMA trend.
     - Target: 2.0R asymmetric trend expansion.

  3. ASIAN_LIQUIDITY_SWEEP_SPRING:
     - Asian range (00:00 - 06:30 UTC) extreme sweep (1.0 to 6.0 pips past high/low).
     - Fades the false breakout with M15 rejection wick >= 40% closing back inside range.
     - Active Hours: 06:45 - 09:30 UTC. Target: 2.0R.

  4. VWAP_2P5SD_EXTREME_EXHAUSTION:
     - Intraday session VWAP 2.5-Standard Deviation band penetration.
     - Prints rejection wick >= 35% and closes back inside 2.5-SD band.
     - Counter-trend mean reversion. Target: 1.67R.
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
# Model 1: London Fix Rebalance Surge
# ==============================================================================
class LondonFixRebalanceSurge(BaseStrategy):
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
        o = m15_df['open'].values
        spread = m15_df['spread_pips'].values
        h4_trend = m15_df['h4_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        minutes = m15_df.index.minute.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr10 = pd.Series(tr).rolling(10).mean().bfill().values

        last_trade_date = None

        for i in range(16, len(m15_df)):
            dt = dates[i]
            hr = hours[i]
            mn = minutes[i]

            # Fixing surge window: 15:15 to 16:30 UTC
            if not ((hr == 15 and mn >= 15) or (hr == 16 and mn <= 30)):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            rng = h[i] - l[i]
            body = abs(c[i] - o[i])
            curr_atr = atr10[i]

            if curr_atr <= 0 or rng < 1.20 * curr_atr or body < 0.50 * rng:
                continue

            prior_16_h = np.max(h[i-16:i])
            prior_16_l = np.min(l[i-16:i])

            sig = 0
            if macro == 1 and c[i] > o[i] and c[i] > prior_16_h:
                sig = SIGNAL_LONG
            elif macro == -1 and c[i] < o[i] and c[i] < prior_16_l:
                sig = SIGNAL_SHORT

            if sig != 0:
                end_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, end_ms)
                if idx < n:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 2: SuperTrend Keltner Squeeze
# ==============================================================================
class SuperTrendKeltnerSqueeze(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_SuperTrend_Keltner_Squeeze"

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
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        c_ser = pd.Series(c)
        bb_mid = c_ser.rolling(20).mean().values
        bb_std = c_ser.rolling(20).std().bfill().values
        bb_upper = bb_mid + 2.0 * bb_std
        bb_lower = bb_mid - 2.0 * bb_std

        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr20 = pd.Series(tr).rolling(20).mean().bfill().values
        kc_mid = c_ser.ewm(span=20, adjust=False).mean().values
        kc_upper = kc_mid + 1.5 * atr20
        kc_lower = kc_mid - 1.5 * atr20

        last_trade_date = None

        for i in range(21, len(m15_df)):
            dt = dates[i]
            hr = hours[i]

            if not (7 <= hr <= 17):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            was_squeezed = (bb_upper[i-1] <= kc_upper[i-1]) and (bb_lower[i-1] >= kc_lower[i-1])
            is_firing = (bb_upper[i] > kc_upper[i]) or (bb_lower[i] < kc_lower[i])

            if not (was_squeezed and is_firing):
                continue

            sig = 0
            if macro == 1 and c[i] > bb_upper[i]:
                sig = SIGNAL_LONG
            elif macro == -1 and c[i] < bb_lower[i]:
                sig = SIGNAL_SHORT

            if sig != 0:
                end_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, end_ms)
                if idx < n:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 3: Asian Liquidity Sweep Spring
# ==============================================================================
class AsianLiquiditySweepSpring(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Asian_Liquidity_Sweep_Spring"

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

        asia_high = None
        asia_low = None
        curr_day = None
        last_trade_date = None

        for i in range(len(m15)):
            dt = dates[i]
            hr = hours[i]
            mn = minutes[i]

            if dt != curr_day:
                curr_day = dt
                asia_high = None
                asia_low = None

            # Asian accumulation: 00:00 - 06:30 UTC
            if hr < 6 or (hr == 6 and mn <= 30):
                if asia_high is None or h[i] > asia_high:
                    asia_high = h[i]
                if asia_low is None or l[i] < asia_low:
                    asia_low = l[i]
                continue

            # Sweep window: 06:45 - 09:30 UTC
            if (hr > 6 or (hr == 6 and mn >= 45)) and (hr < 9 or (hr == 9 and mn <= 30)):
                if asia_high is None or asia_low is None:
                    continue
                if dt == last_trade_date:
                    continue
                if spread[i] > self.max_spread:
                    continue

                asia_rng = (asia_high - asia_low) / self.pip_size
                if asia_rng > 35.0 or asia_rng < 8.0:
                    continue

                rng = h[i] - l[i]
                if rng <= 0:
                    continue

                sig = 0
                # Bearish Upthrust Sweep of Asian High
                if h[i] > asia_high and c[i] < asia_high:
                    sweep_pips = (h[i] - asia_high) / self.pip_size
                    upper_wick = (h[i] - max(o[i], c[i])) / rng
                    if 1.0 <= sweep_pips <= 6.0 and upper_wick >= 0.40:
                        sig = SIGNAL_SHORT

                # Bullish Spring Sweep of Asian Low
                elif l[i] < asia_low and c[i] > asia_low:
                    sweep_pips = (asia_low - l[i]) / self.pip_size
                    lower_wick = (min(o[i], c[i]) - l[i]) / rng
                    if 1.0 <= sweep_pips <= 6.0 and lower_wick >= 0.40:
                        sig = SIGNAL_LONG

                if sig != 0:
                    end_ms = time_ms[i] + dur_ms
                    idx = np.searchsorted(tick_ts, end_ms)
                    if idx < n:
                        signals[idx] = sig
                        sl_arr[idx] = self.sl_pips
                        tp_arr[idx] = self.tp_pips
                        last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 4: VWAP 2.5-SD Extreme Exhaustion
# ==============================================================================
class VWAP2P5SDExtremeExhaustion(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_VWAP_2p5SD_Exhaustion"

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

        curr_day = None
        last_trade_date = None

        cum_vol = 0.0
        cum_vp = 0.0
        cum_sq = 0.0

        for i in range(len(m15)):
            dt = dates[i]
            hr = hours[i]

            if dt != curr_day:
                curr_day = dt
                cum_vol = 0.0
                cum_vp = 0.0
                cum_sq = 0.0

            typ = (h[i] + l[i] + c[i]) / 3.0
            cum_vol += 1.0
            cum_vp += typ
            cum_sq += (typ ** 2)

            vwap = cum_vp / cum_vol
            variance = max(0.0, (cum_sq / cum_vol) - (vwap ** 2))
            sd = np.sqrt(variance)

            # Active hours: 08:00 - 16:30 UTC
            if not (8 <= hr <= 16):
                continue
            if dt == last_trade_date or sd <= 0:
                continue
            if spread[i] > self.max_spread:
                continue

            upper_band = vwap + 2.5 * sd
            lower_band = vwap - 2.5 * sd

            rng = h[i] - l[i]
            if rng <= 0:
                continue

            sig = 0
            if h[i] >= upper_band and c[i] < upper_band:
                upper_wick = (h[i] - max(o[i], c[i])) / rng
                if upper_wick >= 0.35:
                    sig = SIGNAL_SHORT
            elif l[i] <= lower_band and c[i] > lower_band:
                lower_wick = (min(o[i], c[i]) - l[i]) / rng
                if lower_wick >= 0.35:
                    sig = SIGNAL_LONG

            if sig != 0:
                end_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, end_ms)
                if idx < n:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model Factory & Test Harness
# ==============================================================================
def build_model_v27(model_key: str, asset: str) -> BaseStrategy:
    if model_key == "LONDON_FIX_SURGE":
        if asset == "EURUSD":
            return LondonFixRebalanceSurge(asset=asset, sl_pips=8.5, tp_pips=17.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return LondonFixRebalanceSurge(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return LondonFixRebalanceSurge(asset=asset, sl_pips=7.5, tp_pips=15.0, pip_size=0.0001, max_spread=0.6)

    elif model_key == "SUPERTREND_SQUEEZE":
        if asset == "EURUSD":
            return SuperTrendKeltnerSqueeze(asset=asset, sl_pips=8.5, tp_pips=17.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return SuperTrendKeltnerSqueeze(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return SuperTrendKeltnerSqueeze(asset=asset, sl_pips=7.5, tp_pips=15.0, pip_size=0.0001, max_spread=0.6)

    elif model_key == "ASIAN_SWEEP_SPRING":
        if asset == "EURUSD":
            return AsianLiquiditySweepSpring(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return AsianLiquiditySweepSpring(asset=asset, sl_pips=11.0, tp_pips=22.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return AsianLiquiditySweepSpring(asset=asset, sl_pips=7.0, tp_pips=14.0, pip_size=0.0001, max_spread=0.6)

    elif model_key == "VWAP_2P5SD_EXHAUSTION":
        if asset == "EURUSD":
            return VWAP2P5SDExtremeExhaustion(asset=asset, sl_pips=8.5, tp_pips=15.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return VWAP2P5SDExtremeExhaustion(asset=asset, sl_pips=12.0, tp_pips=20.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return VWAP2P5SDExtremeExhaustion(asset=asset, sl_pips=7.5, tp_pips=13.0, pip_size=0.0001, max_spread=0.6)


def run_week_task(task: Tuple[str, str, int]) -> Dict[str, Any]:
    asset, model_key, week_num = task
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if ticks is None or len(ticks) < 500:
        return {"asset": asset, "model": model_key, "week": week_num, "trades": []}

    strat = build_model_v27(model_key, asset)
    try:
        raw_trades = engine.run(ticks, strat)
        t_list = []
        for i in range(len(raw_trades)):
            row = raw_trades[i]
            t_list.append({
                "pips": float(row["pips"]),
                "direction": int(row["direction"])
            })
        return {"asset": asset, "model": model_key, "week": week_num, "trades": t_list}
    except Exception:
        return {"asset": asset, "model": model_key, "week": week_num, "trades": []}


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

    equity = np.cumsum(arr)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
    ret_dd = net_pips / max_dd if max_dd > 0 else 999.0

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
    print("=" * 95)
    print("  ALPHA DISCOVERY LAB V27: DISCOVERING NEXT-GENERATION ALGORITHMIC EDGES")
    print("  Partition: 70% In-Sample Discovery Weeks (Seed 42) across EURUSD, USDJPY, EURGBP")
    print("=" * 95)

    models = [
        "LONDON_FIX_SURGE",
        "SUPERTREND_SQUEEZE",
        "ASIAN_SWEEP_SPRING",
        "VWAP_2P5SD_EXHAUSTION"
    ]
    assets = [os.environ["DISCOVERY_TARGET_SYMBOL"]] if "DISCOVERY_TARGET_SYMBOL" in os.environ else ["EURUSD", "USDJPY", "EURGBP"]

    tasks = []
    for asset in assets:
        disc_weeks = get_discovery_weeks(raw_dir=f"data/{asset}")
        for m in models:
            for w in disc_weeks:
                tasks.append((asset, m, w))

    print(f"Total discovery tasks queued: {len(tasks):,} across {len(assets)} assets and {len(models)} architectures")

    results_by_key = {}
    t0 = time.time()

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_week_task, t) for t in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            key = f"{res['asset']}_{res['model']}"
            if key not in results_by_key:
                results_by_key[key] = []
            results_by_key[key].extend(res["trades"])

    elapsed = time.time() - t0
    print(f"Discovery simulation completed in {elapsed:.1f}s!\n")

    summary_rows = []
    print("=" * 95)
    print(f"  {'Asset':<8} {'Model':<25} {'Trades':<8} {'Win Rate':<10} {'PF':<8} {'Expectancy':<12} {'Net Pips':<10} {'Max DD':<10} {'Verdict'}")
    print("  " + "-" * 90)

    for key, trades in sorted(results_by_key.items()):
        asset, model = key.split("_", 1)
        pips = [t["pips"] for t in trades]
        m = calculate_metrics(pips)

        verdict = "[DISCOVERY CHAMPION]" if (m["net_pips"] >= 200 and m["pf"] >= 1.25 and m["expectancy"] >= 1.5) else "[PROMISING]" if m["net_pips"] > 0 else "[DISCARDED]"

        print(f"  {asset:<8} {model:<25} {m['trades']:<8d} {m['win_rate']:<9.1f}% {m['pf']:<7.2f} +{m['expectancy']:<10.2f}p +{m['net_pips']:<9.1f}p {m['max_dd']:<9.1f}p {verdict}")

        summary_rows.append({
            "asset": asset,
            "model": model,
            "trades": m["trades"],
            "win_rate": m["win_rate"],
            "profit_factor": m["pf"],
            "expectancy": m["expectancy"],
            "net_pips": m["net_pips"],
            "max_dd": m["max_dd"],
            "ret_dd": m["ret_dd"],
            "verdict": verdict
        })

    df_out = pd.DataFrame(summary_rows)
    df_out.to_csv(RESULTS_DIR / "alpha_discovery_lab_v27_scorecard.csv", index=False)
    print("\n" + "=" * 95)
    print("Saved Alpha Discovery Lab V27 discovery scorecard to results/alpha_discovery_lab_v27_scorecard.csv")
    print("=" * 95)


if __name__ == "__main__":
    main()
