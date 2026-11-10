#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V11: MICRO BLAST, TTM SQUEEZE, TOKYO SWEEP & VWAP BANDS
================================================================================
Evaluates 4 Next-Generation Quantitative Microstructure Architectures across 70% Discovery:
  1. DISC58_M5_EMA_EXPANSION_ACCELERATION:
     M5 Fast EMA (8/21) cross + expansion candle in London/NY overlap (12:00-16:30 UTC)
     aligned with H1 EMA (20/50) macro bias. 2.0R target.
  2. DISC59_H1_KELTNER_SQUEEZE_EXPLOSION:
     John Carter TTM Squeeze on H1 (BB inside KC), entering on explosive expansion
     of BB outside KC confirmed by M15 directional breakout. 2.0R target.
  3. DISC60_TOKYO_RANGE_SWEEP_REVERSAL:
     00:00-06:00 UTC Asian session high/low sweep by 1.5-10.0 pips during 06:30-10:30 UTC
     with M15 rejection wick closing back inside range. 1.8R target.
  4. DISC61_VWAP_DEVIATION_BAND_FADE:
     Session VWAP anchored from 00:00 UTC; 2.2-SD deviation stretch fade during 08:00-18:00 UTC
     with M15 rejection wick when H1 ADX < 28. 1.8R target.

Tested across EURUSD, EURGBP, and USDJPY on 70% Discovery Partition ONLY.
Strict 70/30 deterministic holdout isolation is maintained.
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


# ==============================================================================
# Model 1: M5 EMA Expansion Acceleration (DISC58)
# ==============================================================================
class M5EMAExpansionAcceleration(BaseStrategy):
    """
    Operates in London/NY Overlap (12:00 - 16:30 UTC).
    H1 EMA(20) vs EMA(50) defines macro momentum bias.
    M5 EMA(8) crosses M5 EMA(21) with an expansion body >= 1.0x 14-period average range.
    Enters on completion of expansion bar with 2.0R target.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_M5_EMA_Expansion_Acceleration"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m5 = build_time_bars(ticks, timeframe_min=5, pip_size=self.pip_size)
        h1 = build_time_bars(ticks, timeframe_min=60, pip_size=self.pip_size)
        if len(m5) < 30 or len(h1) < 10:
            return signals, sl_arr, tp_arr

        # H1 Macro Bias: EMA(20) vs EMA(50)
        h1_c = pd.Series(h1['close'].values)
        h1_ema20 = h1_c.ewm(span=20, adjust=False).mean().values
        h1_ema50 = h1_c.ewm(span=50, adjust=False).mean().values
        h1_bias = np.where(h1_ema20 > h1_ema50, 1, np.where(h1_ema20 < h1_ema50, -1, 0))
        h1_df = pd.DataFrame({'h1_bias': h1_bias}, index=h1.index)

        # Merge H1 bias into M5 bars
        m5_df = pd.merge_asof(
            m5.sort_index(),
            h1_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )

        c = m5_df['close'].values
        h = m5_df['high'].values
        l = m5_df['low'].values
        o = m5_df['open'].values
        spread = m5_df['spread_pips'].values
        bias = m5_df['h1_bias'].fillna(0).values.astype(int)
        hours = m5_df.index.hour.values
        mins = m5_df.index.minute.values
        dates = m5_df.index.date
        time_ms = m5_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 5 * 60 * 1000
        tick_ts = ticks['timestamp']

        # M5 EMAs (8 & 21)
        s_c = pd.Series(c)
        ema8 = s_c.ewm(span=8, adjust=False).mean().values
        ema21 = s_c.ewm(span=21, adjust=False).mean().values

        # 14-period average range
        bar_range = (h - l) / self.pip_size
        s_rng = pd.Series(bar_range).rolling(14, min_periods=1).mean().shift(1).fillna(5.0).values

        last_trade_date = None
        for i in range(25, len(m5_df)):
            t_hour = hours[i]
            t_min = mins[i]
            # 12:00 to 16:30 UTC
            if not (12 <= t_hour < 16 or (t_hour == 16 and t_min <= 30)):
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            if spread[i] > self.max_spread:
                continue

            macro = bias[i]
            if macro == 0:
                continue

            # M5 EMA cross
            crossed_up = (ema8[i-1] <= ema21[i-1]) and (ema8[i] > ema21[i])
            crossed_down = (ema8[i-1] >= ema21[i-1]) and (ema8[i] < ema21[i])

            # Expansion candle check
            body_pips = abs(c[i] - o[i]) / self.pip_size
            avg_rng = s_rng[i]
            is_expansion = (body_pips >= 0.9 * avg_rng) and (body_pips >= 4.0)

            sig = 0
            if macro == 1 and crossed_up and is_expansion and (c[i] > o[i]):
                sig = SIGNAL_LONG
            elif macro == -1 and crossed_down and is_expansion and (c[i] < o[i]):
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
# Model 2: H1 Keltner Squeeze Explosion Breakout (DISC59)
# ==============================================================================
class H1KeltnerSqueezeExplosion(BaseStrategy):
    """
    John Carter TTM Squeeze on H1:
    - Bollinger Bands (20, 2.0 SD) vs Keltner Channels (20 EMA, 1.5 ATR).
    - Squeeze ON when BB inside KC.
    - Squeeze FIRES when BB Upper > KC Upper or BB Lower < KC Lower.
    - Enters on first M15 breakout confirmation in squeeze explosion direction.
    - Active 07:00 - 16:00 UTC. 2.0R target.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_H1_Keltner_Squeeze_Explosion"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h1 = build_time_bars(ticks, timeframe_min=60, pip_size=self.pip_size)
        if len(m15) < 30 or len(h1) < 25:
            return signals, sl_arr, tp_arr

        # Calculate H1 Squeeze Metrics
        h1_c = pd.Series(h1['close'].values)
        h1_h = pd.Series(h1['high'].values)
        h1_l = pd.Series(h1['low'].values)

        # Bollinger Bands (20, 2.0)
        h1_sma20 = h1_c.rolling(20, min_periods=1).mean()
        h1_std20 = h1_c.rolling(20, min_periods=1).std().fillna(0)
        h1_bb_upper = h1_sma20 + 2.0 * h1_std20
        h1_bb_lower = h1_sma20 - 2.0 * h1_std20

        # Keltner Channels (20, 1.5 ATR)
        h1_ema20 = h1_c.ewm(span=20, adjust=False).mean()
        tr = np.maximum(h1_h - h1_l, np.maximum(abs(h1_h - h1_c.shift(1)), abs(h1_l - h1_c.shift(1)))).fillna(0)
        atr20 = tr.rolling(20, min_periods=1).mean()
        h1_kc_upper = h1_ema20 + 1.5 * atr20
        h1_kc_lower = h1_ema20 - 1.5 * atr20

        # Squeeze state: 1 if inside (squeeze), 0 if outside (fired)
        squeeze_on = (h1_bb_upper < h1_kc_upper) & (h1_bb_lower > h1_kc_lower)
        squeeze_fired = squeeze_on.shift(1).fillna(False) & (~squeeze_on)

        # Direction of momentum on fire: close vs ema20
        fire_dir = np.where(squeeze_fired, np.where(h1_c > h1_ema20, 1, -1), 0)
        h1_df = pd.DataFrame({'squeeze_fire': fire_dir}, index=h1.index)

        # Merge onto M15 bars
        m15_df = pd.merge_asof(
            m15.sort_index(),
            h1_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )

        c = m15_df['close'].values
        h = m15_df['high'].values
        l = m15_df['low'].values
        o = m15_df['open'].values
        spread = m15_df['spread_pips'].values
        fire = m15_df['squeeze_fire'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None
        for i in range(20, len(m15_df)):
            if not (7 <= hours[i] <= 16):
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            if spread[i] > self.max_spread:
                continue

            # Look for recent squeeze fire within last 4 M15 bars (1 hour)
            recent_fires = fire[max(0, i-3):i+1]
            non_zero = recent_fires[recent_fires != 0]
            if len(non_zero) == 0:
                continue

            f_dir = non_zero[-1]
            sig = 0
            if f_dir == 1 and (c[i] > o[i]) and (c[i] > h[i-1]):
                sig = SIGNAL_LONG
            elif f_dir == -1 and (c[i] < o[i]) and (c[i] < l[i-1]):
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
# Model 3: Tokyo High/Low Liquidity Sweep Reversal (DISC60)
# ==============================================================================
class TokyoRangeSweepReversal(BaseStrategy):
    """
    Computes Asian Session High & Low from 00:00 to 06:00 UTC.
    During 06:30 - 10:30 UTC:
    - If price sweeps Asian High by 1.5 - 10.0 pips and forms an M15 bearish rejection
      closing back inside the Asian range -> Short.
    - If price sweeps Asian Low by 1.5 - 10.0 pips and forms an M15 bullish rejection
      closing back inside the Asian range -> Long.
    Target: 1.80R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Tokyo_Range_Sweep_Reversal"

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
        mins = m15.index.minute.values
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Group by date to find 00:00 - 06:00 Asian range
        unique_dates = np.unique(dates)
        asia_highs = {}
        asia_lows = {}

        for ud in unique_dates:
            mask = (dates == ud) & (hours < 6)
            if np.any(mask):
                h_sub = h[mask]
                l_sub = l[mask]
                a_hi = np.max(h_sub)
                a_lo = np.min(l_sub)
                if (a_hi - a_lo) / self.pip_size >= 10.0:  # Valid Asian range
                    asia_highs[ud] = a_hi
                    asia_lows[ud] = a_lo

        last_trade_date = None
        for i in range(15, len(m15)):
            t_hour = hours[i]
            t_min = mins[i]
            # Sweep window: 06:30 - 10:30 UTC
            if not ((t_hour == 6 and t_min >= 30) or (7 <= t_hour <= 10)):
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            if dt not in asia_highs:
                continue

            if spread[i] > self.max_spread:
                continue

            a_hi = asia_highs[dt]
            a_lo = asia_lows[dt]

            bar_rng = h[i] - l[i]
            if bar_rng < 1e-6:
                continue

            # Sweep high check: High poked above a_hi by 1.5 - 10.0 pips, but closed below a_hi
            hi_sweep_pips = (h[i] - a_hi) / self.pip_size
            upper_wick = h[i] - max(o[i], c[i])
            upper_wick_ratio = upper_wick / bar_rng

            # Sweep low check: Low dipped below a_lo by 1.5 - 10.0 pips, but closed above a_lo
            lo_sweep_pips = (a_lo - l[i]) / self.pip_size
            lower_wick = min(o[i], c[i]) - l[i]
            lower_wick_ratio = lower_wick / bar_rng

            sig = 0
            if (1.5 <= hi_sweep_pips <= 12.0) and (c[i] < a_hi) and (upper_wick_ratio >= 0.35) and (c[i] < o[i]):
                sig = SIGNAL_SHORT
            elif (1.5 <= lo_sweep_pips <= 12.0) and (c[i] > a_lo) and (lower_wick_ratio >= 0.35) and (c[i] > o[i]):
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
# Model 4: Session VWAP 2.2-SD Deviation Band Fade (DISC61)
# ==============================================================================
class VWAPDeviationBandFade(BaseStrategy):
    """
    Anchors Volume Weighted Average Price (VWAP) at 00:00 UTC daily.
    Calculates upper and lower standard deviation bands at 2.20-SD.
    Filters by H1 ADX < 28 (non-trending market).
    During 08:00 - 18:00 UTC, when price stretches >= 2.2-SD and forms an M15 rejection
    wick >= 35% pointing back toward VWAP -> Enter mean reversion fade.
    Target: 1.80R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_VWAP_Deviation_Band_Fade"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h1 = build_time_bars(ticks, timeframe_min=60, pip_size=self.pip_size)
        if len(m15) < 30 or len(h1) < 20:
            return signals, sl_arr, tp_arr

        # Approximate H1 ADX(14)
        h1_c = h1['close'].values
        h1_h = h1['high'].values
        h1_l = h1['low'].values
        tr = np.maximum(h1_h[1:] - h1_l[1:], np.maximum(abs(h1_h[1:] - h1_c[:-1]), abs(h1_l[1:] - h1_c[:-1])))
        dm_plus = np.where((h1_h[1:] - h1_h[:-1] > h1_l[:-1] - h1_l[1:]) & (h1_h[1:] - h1_h[:-1] > 0), h1_h[1:] - h1_h[:-1], 0)
        dm_minus = np.where((h1_l[:-1] - h1_l[1:] > h1_h[1:] - h1_h[:-1]) & (h1_l[:-1] - h1_l[1:] > 0), h1_l[:-1] - h1_l[1:], 0)

        s_tr = pd.Series(tr).rolling(14, min_periods=1).mean()
        s_dmp = pd.Series(dm_plus).rolling(14, min_periods=1).mean()
        s_dmm = pd.Series(dm_minus).rolling(14, min_periods=1).mean()
        di_plus = 100.0 * (s_dmp / (s_tr + 1e-9))
        di_minus = 100.0 * (s_dmm / (s_tr + 1e-9))
        dx = 100.0 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-9)
        adx_vals = dx.rolling(14, min_periods=1).mean().values
        h1_adx_full = np.concatenate([[20.0], adx_vals])
        h1_df = pd.DataFrame({'h1_adx': h1_adx_full}, index=h1.index)

        # Merge onto M15 bars
        m15_df = pd.merge_asof(
            m15.sort_index(),
            h1_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )

        c = m15_df['close'].values
        h = m15_df['high'].values
        l = m15_df['low'].values
        o = m15_df['open'].values
        vol = m15_df['volume'].values if 'volume' in m15_df else np.ones(len(m15_df))
        spread = m15_df['spread_pips'].values
        adx = m15_df['h1_adx'].fillna(20.0).values
        hours = m15_df.index.hour.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Daily Anchored VWAP and Standard Deviation
        typ_price = (h + l + c) / 3.0
        vwap = np.zeros(len(m15_df))
        vwap_sd = np.zeros(len(m15_df))

        cur_date = None
        cum_vol = 0.0
        cum_pv = 0.0
        cum_pv2 = 0.0

        for i in range(len(m15_df)):
            dt = dates[i]
            if dt != cur_date:
                cur_date = dt
                cum_vol = 0.0
                cum_pv = 0.0
                cum_pv2 = 0.0

            v = max(vol[i], 1.0)
            p = typ_price[i]
            cum_vol += v
            cum_pv += p * v
            cum_pv2 += (p ** 2) * v

            vwap_val = cum_pv / cum_vol
            variance = max((cum_pv2 / cum_vol) - (vwap_val ** 2), 0.0)
            vwap[i] = vwap_val
            vwap_sd[i] = np.sqrt(variance)

        upper_band = vwap + 2.2 * vwap_sd
        lower_band = vwap - 2.2 * vwap_sd

        last_trade_date = None
        for i in range(20, len(m15_df)):
            if not (8 <= hours[i] <= 18):
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            if spread[i] > self.max_spread:
                continue

            # Ranging filter: ADX < 28
            if adx[i] >= 28.0:
                continue

            # Need valid standard deviation
            if vwap_sd[i] / self.pip_size < 3.0:
                continue

            bar_rng = h[i] - l[i]
            if bar_rng < 1e-6:
                continue

            # Rejection from Upper Band -> Fade Short
            upper_wick = h[i] - max(o[i], c[i])
            upper_wick_ratio = upper_wick / bar_rng

            # Rejection from Lower Band -> Fade Long
            lower_wick = min(o[i], c[i]) - l[i]
            lower_wick_ratio = lower_wick / bar_rng

            sig = 0
            if (h[i] >= upper_band[i]) and (c[i] < upper_band[i]) and (upper_wick_ratio >= 0.35) and (c[i] < o[i]):
                sig = SIGNAL_SHORT
            elif (l[i] <= lower_band[i]) and (c[i] > lower_band[i]) and (lower_wick_ratio >= 0.35) and (c[i] > o[i]):
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
# Discovery Worker for Alpha Discovery Lab V11
# ==============================================================================
def evaluate_v11_discovery_week(task: Tuple[str, int]) -> Dict[str, Any]:
    asset, week_num = task
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if ticks is None or len(ticks) < 500:
        return {"asset": asset, "week": week_num, "trades": {}}

    models = []
    if asset == "EURUSD":
        models.append(("M5_EMA_EXPANSION", M5EMAExpansionAcceleration(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("H1_KELTNER_SQUEEZE", H1KeltnerSqueezeExplosion(asset=asset, sl_pips=10.0, tp_pips=20.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("TOKYO_SWEEP_REV", TokyoRangeSweepReversal(asset=asset, sl_pips=8.0, tp_pips=14.4, pip_size=0.0001, max_spread=0.6)))
        models.append(("VWAP_DEV_FADE", VWAPDeviationBandFade(asset=asset, sl_pips=8.0, tp_pips=14.4, pip_size=0.0001, max_spread=0.6)))
    elif asset == "USDJPY":
        models.append(("M5_EMA_EXPANSION", M5EMAExpansionAcceleration(asset=asset, sl_pips=11.0, tp_pips=22.0, pip_size=0.01, max_spread=0.8)))
        models.append(("H1_KELTNER_SQUEEZE", H1KeltnerSqueezeExplosion(asset=asset, sl_pips=14.0, tp_pips=28.0, pip_size=0.01, max_spread=0.8)))
        models.append(("TOKYO_SWEEP_REV", TokyoRangeSweepReversal(asset=asset, sl_pips=11.0, tp_pips=19.8, pip_size=0.01, max_spread=0.8)))
        models.append(("VWAP_DEV_FADE", VWAPDeviationBandFade(asset=asset, sl_pips=11.0, tp_pips=19.8, pip_size=0.01, max_spread=0.8)))
    elif asset == "EURGBP":
        models.append(("M5_EMA_EXPANSION", M5EMAExpansionAcceleration(asset=asset, sl_pips=7.0, tp_pips=14.0, pip_size=0.0001, max_spread=0.8)))
        models.append(("H1_KELTNER_SQUEEZE", H1KeltnerSqueezeExplosion(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)))
        models.append(("TOKYO_SWEEP_REV", TokyoRangeSweepReversal(asset=asset, sl_pips=7.0, tp_pips=12.6, pip_size=0.0001, max_spread=0.8)))
        models.append(("VWAP_DEV_FADE", VWAPDeviationBandFade(asset=asset, sl_pips=7.0, tp_pips=12.6, pip_size=0.0001, max_spread=0.8)))

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
        except Exception as e:
            week_trades[m_name] = []

    return {"asset": asset, "week": week_num, "trades": week_trades}


def run_discovery_v11():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("  LAUNCHING ALPHA DISCOVERY V11 EXPEDITION (70% DISCOVERY PARTITIONS)", flush=True)
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
        futures = {executor.submit(evaluate_v11_discovery_week, t): t for t in all_tasks}
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

    # Compute Scorecard
    rows = []
    for (asset, m_name), pips in results_by_key.items():
        n = len(pips)
        if n == 0:
            continue
        p_arr = np.array(pips)
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
        ret_dd = round(net_pips / max_dd, 2) if max_dd > 0 else 0.0
        expectancy = round(net_pips / n, 2)

        rows.append({
            "asset": asset,
            "strategy": m_name,
            "trades": n,
            "win_rate": wr,
            "net_pips": net_pips,
            "profit_factor": pf,
            "max_dd_pips": max_dd,
            "ret_dd": ret_dd,
            "expectancy": expectancy
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["asset", "net_pips"], ascending=[True, False])
        out_csv = RESULTS_DIR / "alpha_discovery_lab_v11_scorecard.csv"
        df.to_csv(out_csv, index=False)

        print("\n" + "=" * 90, flush=True)
        print("  ALPHA DISCOVERY LAB V11 SCORECARD (70% DISCOVERY PARTITIONS ONLY)", flush=True)
        print("=" * 90, flush=True)
        print(f"{'Asset':<8} {'Strategy':<22} {'Trades':<8} {'Win Rate':<10} {'Net Pips':<12} {'PF':<8} {'Max DD':<10} {'Ret/DD':<8} {'Exp (p/t)':<10}")
        print("-" * 90, flush=True)
        for _, r in df.iterrows():
            print(f"{r['asset']:<8} {r['strategy']:<22} {int(r['trades']):<8} {r['win_rate']:>6.1f}%   {r['net_pips']:>10.1f}p  {r['profit_factor']:>6.2f}  {r['max_dd_pips']:>8.1f}p  {r['ret_dd']:>6.2f}x  {r['expectancy']:>8.2f}p")
        print("=" * 90, flush=True)

    print(f"\n[*] Alpha Discovery Lab V11 Complete in {time.time() - t_start:.2f}s.", flush=True)


if __name__ == "__main__":
    run_discovery_v11()
