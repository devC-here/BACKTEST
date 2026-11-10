#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY & REALISTIC RETURN LAB V42:
  Evaluating Architectures for Institutional Return Hurdle (>= 15% Annual Return)
================================================================================
Target Architectures:
  1. M15_MACD_ZERO_CROSS_ACCELERATED (EURUSD & USDJPY)
  2. TOKYO_EXPANSION_WITH_ATR_VOL (USDJPY & EURUSD)
  3. CARDWELL_RSI_50_MOMENTUM_BOUNCE (EURUSD & USDJPY)
  4. EURGBP_CAMARILLA_EXPANSION_CALIBRATED (EURGBP)

STRICT CONSTRAINT:
  Evaluated EXCLUSIVELY on the 70% In-Sample Discovery Partition (Seed 42).
  ZERO touch of the 30% quarantined holdout partition during optimization!

Institutional Return Hurdle:
  - Account Risk Model: 1.0% risk per trade (Standard Prop/Hedge Fund Risk)
  - Annual Return % = (Net R-Multiple / 3.5 years discovery) * 1.0%
  - Passing Bar: Annual Return >= 15.0% AND Profit Factor >= 1.25
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
from tick_engine.core.dataloader import get_week_dataset, get_discovery_weeks
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EURUSD_MULT = 1.0000
USDJPY_MULT = 0.7193
EURGBP_MULT = 1.2921


# ==============================================================================
# Model 1: M15 MACD Zero Cross Accelerated
# ==============================================================================
class M15MACDZeroCrossAccelerated(BaseStrategy):
    def __init__(
        self,
        asset: str,
        min_body_pct: float,
        use_vol_filter: bool,
        sl_pips: float,
        tp_pips: float,
        pip_size: float,
        max_spread: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.asset = asset
        self.min_body_pct = float(min_body_pct)
        self.use_vol_filter = use_vol_filter
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_MACD_Zero_B{int(min_body_pct*100)}_V{int(use_vol_filter)}_R{tp_pips/sl_pips:.1f}"

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
        h4_trend = np.where(h4_c.values > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'h4_bias': h4_trend}, index=h4.index)

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
        v = m15_df['vol'].values
        spread = m15_df['spread_pips'].values
        h4_bias = m15_df['h4_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # M15 MACD (12, 26, 9)
        s_c = pd.Series(c)
        ema12 = s_c.ewm(span=12, adjust=False).mean()
        ema26 = s_c.ewm(span=26, adjust=False).mean()
        macd_line = (ema12 - ema26).values
        sig_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values

        last_trade_date = None
        for i in range(30, len(m15_df)):
            # Active London & US session (07:30 - 15:30 UTC)
            if not (7 <= hours[i] <= 15):
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            if spread[i] > self.max_spread:
                continue

            macro = h4_bias[i]
            if macro == 0:
                continue

            bar_rng = h[i] - l[i]
            if bar_rng <= 0:
                continue

            body = abs(c[i] - o[i])
            if (body / bar_rng) < self.min_body_pct:
                continue

            if self.use_vol_filter and v[i] <= v[i-1]:
                continue

            # Zero cross in macro trend direction with Signal Line confirmation
            cross_up = (macd_line[i-1] <= 0) and (macd_line[i] > 0) and (macd_line[i] > sig_line[i])
            cross_down = (macd_line[i-1] >= 0) and (macd_line[i] < 0) and (macd_line[i] < sig_line[i])

            sig = 0
            if macro == 1 and cross_up and (c[i] > o[i]):
                sig = SIGNAL_LONG
            elif macro == -1 and cross_down and (c[i] < o[i]):
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
# Model 2: Tokyo Expansion with ATR & Volume Filter
# ==============================================================================
class TokyoExpansionATRVol(BaseStrategy):
    def __init__(
        self,
        asset: str,
        min_atr_mult: float,
        sl_pips: float,
        tp_pips: float,
        pip_size: float,
        max_spread: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.asset = asset
        self.min_atr_mult = float(min_atr_mult)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_TokyoExp_ATR{min_atr_mult}_R{tp_pips/sl_pips:.1f}"

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

        m15_c = pd.Series(m15['close'].values)
        m15_ema20 = m15_c.ewm(span=20, adjust=False).mean().values
        m15['m15_ema20'] = m15_ema20

        tr1 = m15['high'] - m15['low']
        tr2 = (m15['high'] - m15['close'].shift(1)).abs()
        tr3 = (m15['low'] - m15['close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = tr.rolling(14, min_periods=5).mean().values

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
        v = m15_df['vol'].values
        spread = m15_df['spread_pips'].values
        ema20 = m15_df['m15_ema20'].values
        h4_trend = m15_df['h4_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None
        tolerance = 2.5 * self.pip_size

        for i in range(20, len(m15_df)):
            dt = dates[i]
            hr = hours[i]

            # Tokyo session prime trend window: 01:30 - 06:00 UTC
            if not (1 <= hr <= 6):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            val = ema20[i]
            if macro == 0 or np.isnan(val):
                continue

            rng = h[i] - l[i]
            base_atr = atr14[i]
            if np.isnan(base_atr) or rng < self.min_atr_mult * base_atr:
                continue

            if v[i] <= v[i-1]:
                continue

            sig = 0
            if abs(l[i] - val) <= tolerance and c[i] > val and c[i] > o[i] and macro == 1:
                lower_wick = min(c[i], o[i]) - l[i]
                if lower_wick >= 0.35 * rng:
                    sig = SIGNAL_LONG

            elif abs(h[i] - val) <= tolerance and c[i] < val and c[i] < o[i] and macro == -1:
                upper_wick = h[i] - max(c[i], o[i])
                if upper_wick >= 0.35 * rng:
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
# Model 3: Cardwell RSI 50 Momentum Bounce
# ==============================================================================
class CardwellRSI50MomentumBounce(BaseStrategy):
    def __init__(
        self,
        asset: str,
        sl_pips: float,
        tp_pips: float,
        pip_size: float,
        max_spread: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_RSI50_Bounce_R{tp_pips/sl_pips:.1f}"

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
        h4_trend = np.where(h4_c.values > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'h4_bias': h4_trend}, index=h4.index)

        # M15 RSI 14
        m15_c = pd.Series(m15['close'].values)
        delta = m15_c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-9)
        rsi = (100.0 - (100.0 / (1.0 + rs))).values

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

        for i in range(25, len(m15_df)):
            dt = dates[i]
            hr = hours[i]

            # European & US session (07:30 - 15:30 UTC)
            if not (7 <= hr <= 15):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            r = rsi[i]
            r_prev = rsi[i-1]
            if np.isnan(r) or np.isnan(r_prev):
                continue

            rng = h[i] - l[i]
            if rng <= 0:
                continue

            sig = 0
            # Bullish Cardwell RSI 50 bounce: In H4 uptrend, RSI pulled back to 45-55 zone and turns up
            if macro == 1 and 45.0 <= r_prev <= 55.0 and r > r_prev and c[i] > o[i]:
                lower_wick = min(c[i], o[i]) - l[i]
                if lower_wick >= 0.35 * rng:
                    sig = SIGNAL_LONG

            # Bearish Cardwell RSI 50 bounce: In H4 downtrend, RSI pulled up to 45-55 zone and turns down
            elif macro == -1 and 45.0 <= r_prev <= 55.0 and r < r_prev and c[i] < o[i]:
                upper_wick = h[i] - max(c[i], o[i])
                if upper_wick >= 0.35 * rng:
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
# Model 4: EURGBP Camarilla Expansion Calibrated
# ==============================================================================
class EURGBPCamarillaExpansionCalibrated(BaseStrategy):
    def __init__(
        self,
        asset: str,
        sl_pips: float,
        tp_pips: float,
        pip_size: float,
        max_spread: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Camarilla_Exp_R{tp_pips/sl_pips:.1f}"

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

        daily_bars = {}
        for i in range(len(m15)):
            d = dates[i]
            if d not in daily_bars:
                daily_bars[d] = {"high": h[i], "low": l[i], "close": c[i]}
            else:
                daily_bars[d]["high"] = max(daily_bars[d]["high"], h[i])
                daily_bars[d]["low"] = min(daily_bars[d]["low"], l[i])
                daily_bars[d]["close"] = c[i]

        unique_dates = sorted(list(daily_bars.keys()))
        date_to_camarilla = {}
        for idx_d in range(1, len(unique_dates)):
            curr_d = unique_dates[idx_d]
            prev_d = unique_dates[idx_d - 1]
            p_h = daily_bars[prev_d]["high"]
            p_l = daily_bars[prev_d]["low"]
            p_c = daily_bars[prev_d]["close"]
            rng = p_h - p_l
            h4_lvl = p_c + (rng * 1.1 / 2.0)
            l4_lvl = p_c - (rng * 1.1 / 2.0)
            date_to_camarilla[curr_d] = (h4_lvl, l4_lvl)

        last_trade_date = None

        for i in range(len(m15_df)):
            dt = dates[i]
            hr = hours[i]

            if dt not in date_to_camarilla:
                continue

            # Active expansion window: 07:00 - 13:00 UTC
            if not (7 <= hr <= 13):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            h4_lvl, l4_lvl = date_to_camarilla[dt]
            bar_rng = h[i] - l[i]
            if bar_rng <= 0:
                continue

            sig = 0
            if macro == 1 and c[i] > h4_lvl and o[i] < h4_lvl and (c[i] - o[i]) >= 0.40 * bar_rng:
                sig = SIGNAL_LONG
            elif macro == -1 and c[i] < l4_lvl and o[i] > l4_lvl and (o[i] - c[i]) >= 0.40 * bar_rng:
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
# Model Configurations to Audit on 70% In-Sample Discovery Weeks
# ==============================================================================
def get_lab_v42_models() -> List[Tuple[str, str, BaseStrategy, float]]:
    mods = []

    # 1. M15 MACD Zero Cross Accelerated (EURUSD & USDJPY)
    mods.append(("EURUSD", "MACD_ZERO_B55_20R", M15MACDZeroCrossAccelerated("EURUSD", 0.55, True, 8.5, 17.0, 0.0001, 0.6), 8.5))
    mods.append(("EURUSD", "MACD_ZERO_B60_22R", M15MACDZeroCrossAccelerated("EURUSD", 0.60, True, 8.0, 17.6, 0.0001, 0.6), 8.0))
    mods.append(("USDJPY", "MACD_ZERO_B60_22R", M15MACDZeroCrossAccelerated("USDJPY", 0.60, True, 11.5, 25.3, 0.01, 0.8), 11.5))

    # 2. Tokyo Expansion with ATR & Volume Filter (USDJPY & EURUSD)
    mods.append(("USDJPY", "TOKYO_EXP_ATR80_22R", TokyoExpansionATRVol("USDJPY", 0.80, 11.5, 25.3, 0.01, 0.8), 11.5))
    mods.append(("USDJPY", "TOKYO_EXP_ATR90_24R", TokyoExpansionATRVol("USDJPY", 0.90, 11.0, 26.4, 0.01, 0.8), 11.0))
    mods.append(("EURUSD", "TOKYO_EXP_ATR80_22R", TokyoExpansionATRVol("EURUSD", 0.80, 8.0, 17.6, 0.0001, 0.6), 8.0))

    # 3. Cardwell RSI 50 Momentum Bounce (USDJPY & EURUSD)
    mods.append(("USDJPY", "RSI50_BOUNCE_22R", CardwellRSI50MomentumBounce("USDJPY", 11.5, 25.3, 0.01, 0.8), 11.5))
    mods.append(("EURUSD", "RSI50_BOUNCE_22R", CardwellRSI50MomentumBounce("EURUSD", 8.0, 17.6, 0.0001, 0.6), 8.0))

    # 4. EURGBP Camarilla Expansion Calibrated
    mods.append(("EURGBP", "CAM_EXP_CALIB_22R", EURGBPCamarillaExpansionCalibrated("EURGBP", 6.5, 14.3, 0.0001, 0.6), 6.5))
    mods.append(("EURGBP", "CAM_EXP_CALIB_20R", EURGBPCamarillaExpansionCalibrated("EURGBP", 7.0, 14.0, 0.0001, 0.6), 7.0))

    return mods


def run_week_task(task: Tuple[str, str, int, BaseStrategy]) -> Dict[str, Any]:
    asset, model_name, week_num, strat = task
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    try:
        ticks, _ = get_week_dataset(week_num, raw_dir=f"data/{asset}")
        if ticks is None or len(ticks) < 500:
            return {"asset": asset, "model": model_name, "week": week_num, "trades": []}

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


def calculate_metrics(pips: List[float], mult: float, sl_pips: float) -> Dict[str, Any]:
    if not pips:
        return {"trades": 0, "win_rate": 0.0, "pf": 0.0, "net_pips": 0.0, "norm_pips": 0.0, "net_r": 0.0, "annual_return_pct": 0.0, "max_dd": 0.0, "expectancy": 0.0}
    arr = np.array(pips, dtype=np.float64)
    trades = len(arr)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    wr = len(wins) / trades * 100.0 if trades > 0 else 0.0
    gross_win = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else 999.0
    net_pips = float(np.sum(arr))
    norm_pips = net_pips * mult
    expectancy = net_pips / trades if trades > 0 else 0.0

    # R-multiple and Annualized Return % (at 1% risk per trade over 3.5 discovery years)
    net_r = net_pips / sl_pips if sl_pips > 0 else 0.0
    years_discovery = 3.5  # 183-210 weeks / 52 = ~3.5 to 4.0 years
    annual_r = net_r / years_discovery
    annual_return_pct = annual_r * 1.0  # 1% risk per trade -> 1R = 1%

    equity = np.cumsum(arr * mult)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

    return {
        "trades": trades,
        "win_rate": round(wr, 1),
        "pf": round(pf, 2),
        "net_pips": round(net_pips, 1),
        "norm_pips": round(norm_pips, 1),
        "net_r": round(net_r, 1),
        "annual_return_pct": round(annual_return_pct, 1),
        "max_dd": round(max_dd, 1),
        "expectancy": round(expectancy, 2)
    }


def main():
    print("=" * 115)
    print("  ALPHA DISCOVERY & REALISTIC RETURN LAB V42: TESTING >= 15% ANNUAL RETURN HURDLE")
    print("  STRICT PARTITION: 70% In-Sample Discovery Weeks ONLY (Quarantined 30% Holdout Untouched)")
    print("=" * 115)

    models_config = get_lab_v42_models()
    tasks = []
    for asset, model_name, strat, _ in models_config:
        disc_weeks = get_discovery_weeks(raw_dir=f"data/{asset}")
        for w in disc_weeks:
            tasks.append((asset, model_name, w, strat))

    print(f"Total discovery tasks: {len(tasks):,} across {len(models_config)} model configurations")

    results_by_key = {}
    t0 = time.time()

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_week_task, t) for t in tasks]
        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            res = fut.result()
            key = f"{res['asset']}_{res['model']}"
            if key not in results_by_key:
                results_by_key[key] = []
            results_by_key[key].extend(res["trades"])
            done += 1
            if done % 350 == 0 or done == total:
                elapsed = time.time() - t0
                print(f"  Progress: [{done}/{total}] tasks ({done/total*100:.1f}%) in {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\nExecution phase completed in {elapsed:.1f}s!\n")

    summary_rows = []
    print("=" * 125)
    print(f"  {'Asset':<8} {'Model Variant':<23} {'Trades':<8} {'Win Rate':<10} {'PF':<7} {'Net Pips':<10} {'Norm Pips':<10} {'Net R':<9} {'Annual Ret%':<13} {'Verdict'}")
    print("  " + "-" * 120)

    for asset, model_name, _, sl_pips in models_config:
        key = f"{asset}_{model_name}"
        trades = results_by_key.get(key, [])
        pips = [t["pips"] for t in trades]
        mult = EURUSD_MULT if asset == "EURUSD" else USDJPY_MULT if asset == "USDJPY" else EURGBP_MULT
        m = calculate_metrics(pips, mult, sl_pips)

        verdict = "[>= 15% VIABLE CHAMPION]" if (m["pf"] >= 1.25 and m["annual_return_pct"] >= 15.0) else "[PROMISING]" if m["norm_pips"] > 0 else "[SUB-OPTIMAL]"

        print(f"  {asset:<8} {model_name:<23} {m['trades']:<8d} {m['win_rate']:<9.1f}% {m['pf']:<6.2f} +{m['net_pips']:<9.1f}p +{m['norm_pips']:<9.1f}p +{m['net_r']:<8.1f}R +{m['annual_return_pct']:<11.1f}% {verdict}")

        summary_rows.append({
            "asset": asset,
            "model_variant": model_name,
            "trades": m["trades"],
            "win_rate": m["win_rate"],
            "profit_factor": m["pf"],
            "expectancy": m["expectancy"],
            "raw_pips": m["net_pips"],
            "normalized_pips": m["norm_pips"],
            "net_r": m["net_r"],
            "annual_return_pct": m["annual_return_pct"],
            "max_dd_norm": m["max_dd"],
            "verdict": verdict
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(RESULTS_DIR / "alpha_discovery_lab_v42_scorecard.csv", index=False)
    print(f"\nSaved Discovery Scorecard to: {RESULTS_DIR / 'alpha_discovery_lab_v42_scorecard.csv'}")


if __name__ == "__main__":
    main()

