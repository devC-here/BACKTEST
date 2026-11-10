#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V25: FIX REBALANCE, DOUBLE INSIDE BAR, KELTNER & OPEN IMPULSE
================================================================================
Explores 4 novel institutional alpha architectures across the 70% In-Sample Discovery
partition (Seed 42) across EURUSD, USDJPY, and EURGBP (603 discovery weeks):

  1. LONDON_FIX_1530_THRUST:
     - London 16:00 Fix rebalancing window: 15:15 - 16:00 UTC.
     - Directional expansion impulse candle (Range >= 1.25x ATR10, Body >= 55%).
     - Closes at a 4-bar extreme in alignment with higher timeframe H4 20-EMA trend.
     - Asymmetric Target: 2.0R.

  2. DOUBLE_INSIDE_BAR_EXPLOSION:
     - Compound 3-bar volatility compression: Bar i-1 inside Bar i-2, and Bar i-2 inside Bar i-3.
     - Expansion breakout closing beyond the Mother Bar (Bar i-3) extreme.
     - Filtered by H4 20-EMA macro trend alignment.
     - Active Hours: 07:00 - 17:00 UTC. Target: 2.0R.

  3. KELTNER_2SD_EXTREME_ABSORPTION_FADE:
     - M15 Keltner Channel: 20-EMA +/- 2.25x ATR10.
     - Counter-trend exhaustion pin bar: Price pierces outside Keltner band but prints
       a >= 40% rejection wick closing back inside the channel.
     - Active Hours: 07:00 - 17:00 UTC. Target: 1.67R to 1.75R mean-reversion fade.

  4. SESSION_OPEN_MIDPOINT_IMPULSE:
     - London Cash Open reference price anchor (07:00 UTC Open).
     - Expansion impulse candle (Range >= 1.20x ATR10, Body >= 50%) holding strictly
       above/below 07:00 open price between 07:15 and 11:30 UTC in H4 trend direction.
     - Active Hours: 07:15 - 11:30 UTC. Target: 2.0R.
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
# Model 1: London Fix 15:30 Thrust
# ==============================================================================
class LondonFix1530Thrust(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_London_Fix_Thrust"

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

        # ATR 10
        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr10 = pd.Series(tr).rolling(10).mean().values

        last_trade_date = None

        for i in range(10, len(m15_df)):
            dt = dates[i]
            hr = hours[i]
            mn = minutes[i]

            # Rebalancing window: 15:15 to 16:00 UTC (bar completion at 15:15, 15:30, 15:45, 16:00)
            if not ((hr == 15 and mn in (15, 30, 45)) or (hr == 16 and mn == 0)):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            rng = h[i] - l[i]
            if np.isnan(atr10[i]) or atr10[i] <= 0:
                continue
            if rng < 1.25 * atr10[i]:
                continue

            body = abs(c[i] - o[i])
            if body < 0.55 * rng:
                continue

            # Prior 4-bar extreme
            p_high = np.max(h[i-4:i])
            p_low = np.min(l[i-4:i])

            sig = 0
            # Bullish thrust
            if c[i] > o[i] and c[i] > p_high and macro == 1:
                sig = SIGNAL_LONG
            # Bearish thrust
            elif c[i] < o[i] and c[i] < p_low and macro == -1:
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
# Model 2: Double Inside Bar Explosion
# ==============================================================================
class DoubleInsideBarExplosion(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Double_Inside_Bar_Explosion"

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
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None

        for i in range(5, len(m15_df)):
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

            # Double Inside Bar:
            # Bar i-1 is inside Bar i-2: h[i-1] <= h[i-2] and l[i-1] >= l[i-2]
            # Bar i-2 is inside Bar i-3: h[i-2] <= h[i-3] and l[i-2] >= l[i-3]
            is_ib1 = (h[i-1] <= h[i-2]) and (l[i-1] >= l[i-2])
            is_ib2 = (h[i-2] <= h[i-3]) and (l[i-2] >= l[i-3])

            if not (is_ib1 and is_ib2):
                continue

            mother_h = h[i-3]
            mother_l = l[i-3]

            body = abs(c[i] - o[i])
            rng = h[i] - l[i]
            if rng <= 0 or body < 0.50 * rng:
                continue

            sig = 0
            # Bullish explosion breakout
            if c[i] > mother_h and c[i] > o[i] and macro == 1:
                sig = SIGNAL_LONG
            # Bearish explosion breakout
            elif c[i] < mother_l and c[i] < o[i] and macro == -1:
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
# Model 3: Keltner 2SD Extreme Absorption Fade
# ==============================================================================
class Keltner2SDExtremeAbsorptionFade(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Keltner_Absorption_Fade"

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

        # Keltner Channel: EMA20 +/- 2.25 * ATR10
        c_series = pd.Series(c)
        ema20 = c_series.ewm(span=20, adjust=False).mean().values

        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr10 = pd.Series(tr).rolling(10).mean().values

        upper_band = ema20 + 2.25 * atr10
        lower_band = ema20 - 2.25 * atr10

        last_trade_date = None

        for i in range(20, len(m15)):
            dt = dates[i]
            hr = hours[i]

            if not (7 <= hr <= 17):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            rng = h[i] - l[i]
            if rng <= 0 or np.isnan(atr10[i]):
                continue

            sig = 0
            # Bearish Fade: High pierced above upper band, but rejected back inside
            if h[i] >= upper_band[i] and c[i] < upper_band[i]:
                upper_wick = h[i] - max(c[i], o[i])
                if upper_wick >= 0.40 * rng and c[i] <= l[i] + 0.50 * rng:
                    sig = SIGNAL_SHORT

            # Bullish Fade: Low pierced below lower band, but rejected back inside
            elif l[i] <= lower_band[i] and c[i] > lower_band[i]:
                lower_wick = min(c[i], o[i]) - l[i]
                if lower_wick >= 0.40 * rng and c[i] >= l[i] + 0.50 * rng:
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
# Model 4: Session Open Midpoint Impulse
# ==============================================================================
class SessionOpenMidpointImpulse(BaseStrategy):
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Session_Open_Midpoint_Impulse"

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

        # ATR 10
        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr10 = pd.Series(tr).rolling(10).mean().values

        # Track 07:00 UTC open price per day
        session_open_price: Dict[Any, float] = {}
        for idx_bar in range(len(m15_df)):
            dt = dates[idx_bar]
            hr = hours[idx_bar]
            mn = minutes[idx_bar]
            if hr == 7 and mn == 0 and dt not in session_open_price:
                session_open_price[dt] = o[idx_bar]

        last_trade_date = None

        for i in range(10, len(m15_df)):
            dt = dates[i]
            hr = hours[i]

            # Active hours: 07:15 to 11:30 UTC
            if not ((hr == 7 and minutes[i] > 0) or (8 <= hr <= 11)):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            open_p = session_open_price.get(dt)
            if open_p is None:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            rng = h[i] - l[i]
            if np.isnan(atr10[i]) or atr10[i] <= 0:
                continue
            if rng < 1.20 * atr10[i]:
                continue

            body = abs(c[i] - o[i])
            if body < 0.50 * rng:
                continue

            sig = 0
            # Bullish impulse above session open
            if c[i] > o[i] and l[i] >= open_p and c[i] > np.max(h[i-3:i]) and macro == 1:
                sig = SIGNAL_LONG
            # Bearish impulse below session open
            elif c[i] < o[i] and h[i] <= open_p and c[i] < np.min(l[i-3:i]) and macro == -1:
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
# Model Factory
# ==============================================================================
def build_model_v25(model_name: str, asset: str) -> BaseStrategy | None:
    if model_name == "LONDON_FIX_1530_THRUST":
        if asset == "EURUSD":
            return LondonFix1530Thrust(asset=asset, sl_pips=8.5, tp_pips=17.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return LondonFix1530Thrust(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return LondonFix1530Thrust(asset=asset, sl_pips=7.0, tp_pips=14.0, pip_size=0.0001, max_spread=0.8)

    elif model_name == "DOUBLE_INSIDE_BAR_EXPLOSION":
        if asset == "EURUSD":
            return DoubleInsideBarExplosion(asset=asset, sl_pips=8.5, tp_pips=17.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return DoubleInsideBarExplosion(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return DoubleInsideBarExplosion(asset=asset, sl_pips=7.0, tp_pips=14.0, pip_size=0.0001, max_spread=0.8)

    elif model_name == "KELTNER_2SD_EXTREME_ABSORPTION_FADE":
        if asset == "EURUSD":
            return Keltner2SDExtremeAbsorptionFade(asset=asset, sl_pips=8.0, tp_pips=14.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return Keltner2SDExtremeAbsorptionFade(asset=asset, sl_pips=12.0, tp_pips=20.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return Keltner2SDExtremeAbsorptionFade(asset=asset, sl_pips=6.0, tp_pips=10.0, pip_size=0.0001, max_spread=0.8)

    elif model_name == "SESSION_OPEN_MIDPOINT_IMPULSE":
        if asset == "EURUSD":
            return SessionOpenMidpointImpulse(asset=asset, sl_pips=8.5, tp_pips=17.0, pip_size=0.0001, max_spread=0.6)
        elif asset == "USDJPY":
            return SessionOpenMidpointImpulse(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)
        elif asset == "EURGBP":
            return SessionOpenMidpointImpulse(asset=asset, sl_pips=7.0, tp_pips=14.0, pip_size=0.0001, max_spread=0.8)

    return None


def run_week_task_v25(task: Tuple[str, str, int]) -> Dict[str, Any]:
    asset, model_name, week_num = task
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if ticks is None or len(ticks) < 500:
        return {"asset": asset, "model": model_name, "week": week_num, "trades": []}

    strat = build_model_v25(model_name, asset)
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
    print("=" * 80, flush=True)
    print("  LAUNCHING ALPHA DISCOVERY LAB V25: FIX REBALANCE, DOUBLE IB, KELTNER & OPEN IMPULSE", flush=True)
    print("=" * 80, flush=True)
    t0 = time.time()

    assets = [os.environ["DISCOVERY_TARGET_SYMBOL"]] if "DISCOVERY_TARGET_SYMBOL" in os.environ else ["EURUSD", "USDJPY", "EURGBP"]
    models = [
        "LONDON_FIX_1530_THRUST",
        "DOUBLE_INSIDE_BAR_EXPLOSION",
        "KELTNER_2SD_EXTREME_ABSORPTION_FADE",
        "SESSION_OPEN_MIDPOINT_IMPULSE"
    ]

    tasks: List[Tuple[str, str, int]] = []
    for asset in assets:
        disc_weeks = get_discovery_weeks(f"data/{asset}")
        for m in models:
            for w in disc_weeks:
                tasks.append((asset, m, w))

    print(f"[*] Total week-tasks across 3 assets & 4 models: {len(tasks)}", flush=True)

    trades_by_key: Dict[str, List[float]] = {f"{a}_{m}": [] for a in assets for m in models}

    max_workers = max(1, (os.cpu_count() or 4) - 1)
    completed_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_week_task_v25, t): t for t in tasks}

        for fut in as_completed(futures):
            res = fut.result()
            key = f"{res['asset']}_{res['model']}"
            for tr in res["trades"]:
                trades_by_key[key].append(tr["pips"])

            completed_count += 1
            if completed_count % 300 == 0 or completed_count == len(tasks):
                elapsed = time.time() - t0
                print(f"  [+] Audited {completed_count}/{len(tasks)} week-tasks in {elapsed:.1f}s...", flush=True)

    print("\n" + "=" * 90, flush=True)
    print(f"{'ASSET':<8} | {'MODEL':<35} | {'TRADES':<6} | {'WIN RATE':<8} | {'PF':<5} | {'NET PIPS':<10} | {'MAX DD':<8} | {'RET/DD':<6}", flush=True)
    print("=" * 90, flush=True)

    summary_results: Dict[str, Any] = {}

    for a in assets:
        for m in models:
            k = f"{a}_{m}"
            pips = trades_by_key[k]
            met = calculate_metrics(pips)
            summary_results[k] = met
            print(f"{a:<8} | {m:<35} | {met['trades']:<6} | {met['win_rate']:<6.1f} % | {met['pf']:<5.2f} | {met['net_pips']:<10.1f} | {met['max_dd']:<8.1f} | {met['ret_dd']:<5.2f} x", flush=True)
        print("-" * 90, flush=True)

    out_file = RESULTS_DIR / "lab_v25_discovery_results.json"
    with open(out_file, "w") as f:
        json.dump(summary_results, f, indent=2)

    total_time = time.time() - t0
    print(f"\n[+] Discovery results saved to: {out_file}", flush=True)
    print(f"[+] Total execution time: {total_time:.2f}s", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
