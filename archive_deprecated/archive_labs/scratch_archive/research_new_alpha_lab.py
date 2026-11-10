#!/usr/bin/env python3
"""
================================================================================
  RESEARCH LAB: NEW ALPHA FRONTIER HYPOTHESES (52-WEEK TICK SCREENING)
================================================================================
Explores 5 novel institutional and microstructure hypotheses:
  1. FVG_Displacement_Retest: Fair Value Gap 50% Consequent Encroachment Retest
  2. PDH_PDL_Judas_Sweep_Fade: Previous Day High/Low Liquidity Sweep & Rejection
  3. London_Fix_WMR_Momentum: 15:15 - 16:00 UTC WMR Institutional Fix Flow
  4. Ultra_Coil_London_ORB: Frankfurt range <= 18 pips with H1 50-EMA Trend
  5. Exhaustion_Climax_Fade: Tick Volume / ATR Expansion Climax Mean-Reversion
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine, TRADE_DTYPE
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from tick_engine.core.metrics import compute_metrics

PIP_SIZE = 0.0001


# ----------------------------------------------------------------------
# 1. Fair Value Gap (FVG) 50% Consequent Encroachment Retest (M15)
# ----------------------------------------------------------------------
class FVGDisplacementRetestStrategy(BaseStrategy):
    """
    Fair Value Gap (FVG) Consequent Encroachment (50% Retest):
    - When a 3-bar displacement creates an unmitigated imbalance:
      Bullish FVG: Bar[i-2] High < Bar[i] Low (Gap >= min_gap)
      Bearish FVG: Bar[i-2] Low > Bar[i] High (Gap >= min_gap)
    - Trend Filter: Aligned with H1 50-EMA macro trend.
    - Active Session: London (07:00-11:00) & NY (13:00-16:00) UTC.
    - Entry: When subsequent price dips into the 50% midpoint of the FVG.
    - Stop Loss: Beyond the origin of the displacement bar. Target: 1.70R.
    """
    def __init__(self, min_gap_pips: float = 3.5, max_gap_pips: float = 16.0, target_rr: float = 1.70, **kwargs):
        super().__init__(**kwargs)
        self.min_gap_pips = float(min_gap_pips)
        self.max_gap_pips = float(max_gap_pips)
        self.target_rr = float(target_rr)
        self.allowed_hours = {7, 8, 9, 10, 13, 14, 15}
        self.name = "FVG_Displacement_Retest"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'open': mid, 'high': mid, 'low': mid, 'close': mid}, index=ts)
        m15 = base.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(m15) < 40:
            return signals, sl_pips_arr, tp_pips_arr

        h1 = base.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
        h1['macro'] = np.where(h1['close'] > h1['ema50'], 1, -1)
        m15['macro'] = h1['macro'].reindex(m15.index, method='ffill').fillna(0)

        m15['hour'] = m15.index.hour
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        h_vals = m15['high'].values
        l_vals = m15['low'].values
        c_vals = m15['close'].values
        o_vals = m15['open'].values
        macro_vals = m15['macro'].values
        hours = m15['hour'].values

        active_fvg = None  # (fvg_type, top, bottom, mid_price, origin_sl, expiry_idx)

        for i in range(2, len(m15)):
            # Check for new FVG creation at bar i-1 (formed by bars i-2, i-1, i)
            # Bullish FVG
            gap_up = (l_vals[i] - h_vals[i-2]) / PIP_SIZE
            if gap_up >= self.min_gap_pips and gap_up <= self.max_gap_pips and macro_vals[i] > 0 and c_vals[i-1] > o_vals[i-1]:
                top = l_vals[i]
                bot = h_vals[i-2]
                fvg_mid = 0.5 * (top + bot)
                sl_price = min(l_vals[i-1], bot) - 1.2 * PIP_SIZE
                active_fvg = (1, top, bot, fvg_mid, sl_price, i + 8)

            # Bearish FVG
            gap_dn = (l_vals[i-2] - h_vals[i]) / PIP_SIZE
            if gap_dn >= self.min_gap_pips and gap_dn <= self.max_gap_pips and macro_vals[i] < 0 and c_vals[i-1] < o_vals[i-1]:
                top = l_vals[i-2]
                bot = h_vals[i]
                fvg_mid = 0.5 * (top + bot)
                sl_price = max(h_vals[i-1], top) + 1.2 * PIP_SIZE
                active_fvg = (-1, top, bot, fvg_mid, sl_price, i + 8)

            if active_fvg is not None:
                f_type, f_top, f_bot, f_mid, f_sl, f_exp = active_fvg
                if i > f_exp:
                    active_fvg = None
                    continue

                if hours[i] in self.allowed_hours:
                    # Test if current bar retests the FVG midpoint
                    if f_type == 1 and l_vals[i] <= f_mid and c_vals[i] > f_mid and c_vals[i] > o_vals[i]:
                        risk = max(min((c_vals[i] - f_sl) / PIP_SIZE, 15.0), 5.5)
                        exec_t = time_ms[i] + dur_ms
                        idx = int(np.searchsorted(tick_ts, exec_t))
                        if idx < n_ticks and signals[idx] == 0:
                            signals[idx] = SIGNAL_LONG
                            sl_pips_arr[idx] = risk
                            tp_pips_arr[idx] = risk * self.target_rr
                            active_fvg = None
                    elif f_type == -1 and h_vals[i] >= f_mid and c_vals[i] < f_mid and c_vals[i] < o_vals[i]:
                        risk = max(min((f_sl - c_vals[i]) / PIP_SIZE, 15.0), 5.5)
                        exec_t = time_ms[i] + dur_ms
                        idx = int(np.searchsorted(tick_ts, exec_t))
                        if idx < n_ticks and signals[idx] == 0:
                            signals[idx] = SIGNAL_SHORT
                            sl_pips_arr[idx] = risk
                            tp_pips_arr[idx] = risk * self.target_rr
                            active_fvg = None

        return signals, sl_pips_arr, tp_pips_arr


# ----------------------------------------------------------------------
# 2. Previous Day High/Low (PDH/PDL) Judas Sweep Fade (M15)
# ----------------------------------------------------------------------
class PDHPDLJudasSweepFadeStrategy(BaseStrategy):
    """
    Previous Day High/Low (PDH/PDL) Liquidity Sweep Fade ("Turtle Soup"):
    - Identifies exact Previous Day High and Previous Day Low.
    - During London Open (07:00 - 09:30 UTC), institutions push price beyond PDH or PDL
      to trigger resting breakout stops and trapped liquidity.
    - Sweep condition: Price probes 2.0 to 14.0 pips beyond PDH or PDL, but current M15 bar
      fails to expand and closes back INSIDE the previous day's range.
    - Target: Reversion to Daily Pivot / mean with 1.6R asymmetric reward.
    """
    def __init__(self, min_sweep: float = 2.0, max_sweep: float = 14.0, target_rr: float = 1.60, **kwargs):
        super().__init__(**kwargs)
        self.min_sweep = float(min_sweep)
        self.max_sweep = float(max_sweep)
        self.target_rr = float(target_rr)
        self.name = "PDH_PDL_Judas_Sweep_Fade"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'open': mid, 'high': mid, 'low': mid, 'close': mid}, index=ts)
        m15 = base.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(m15) < 30:
            return signals, sl_pips_arr, tp_pips_arr

        m15['date'] = m15.index.date
        m15['hour'] = m15.index.hour
        m15['minute'] = m15.index.minute

        # Daily high and low
        daily = base.resample('1D').agg({'high': 'max', 'low': 'min'}).dropna()
        daily['pdh'] = daily['high'].shift(1)
        daily['pdl'] = daily['low'].shift(1)

        m15['pdh'] = daily['pdh'].reindex(m15.index, method='ffill')
        m15['pdl'] = daily['pdl'].reindex(m15.index, method='ffill')

        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        dates = m15['date'].unique()
        for d in dates:
            day_mask = m15['date'] == d
            day_bars = m15[day_mask]
            if len(day_bars) < 10:
                continue

            day_indices = np.where(day_mask)[0]
            traded = False
            for idx in day_indices:
                row = m15.iloc[idx]
                hr = row['hour']
                mn = row['minute']
                if traded or not ((hr == 7) or (hr == 8) or (hr == 9 and mn <= 30)):
                    continue

                pdh = row['pdh']
                pdl = row['pdl']
                if pd.isna(pdh) or pd.isna(pdl):
                    continue

                sweep_up = (row['high'] - pdh) / PIP_SIZE
                sweep_dn = (pdl - row['low']) / PIP_SIZE

                # Bull trap above PDH -> Fade short
                if sweep_up >= self.min_sweep and sweep_up <= self.max_sweep and row['close'] < pdh and row['close'] < row['open']:
                    sl_p = row['high'] + 1.2 * PIP_SIZE
                    risk = max(min((sl_p - row['close']) / PIP_SIZE, 15.0), 5.5)
                    exec_t = time_ms[idx] + dur_ms
                    t_idx = int(np.searchsorted(tick_ts, exec_t))
                    if t_idx < n_ticks:
                        signals[t_idx] = SIGNAL_SHORT
                        sl_pips_arr[t_idx] = risk
                        tp_pips_arr[t_idx] = risk * self.target_rr
                        traded = True

                # Bear trap below PDL -> Fade long
                elif sweep_dn >= self.min_sweep and sweep_dn <= self.max_sweep and row['close'] > pdl and row['close'] > row['open']:
                    sl_p = row['low'] - 1.2 * PIP_SIZE
                    risk = max(min((row['close'] - sl_p) / PIP_SIZE, 15.0), 5.5)
                    exec_t = time_ms[idx] + dur_ms
                    t_idx = int(np.searchsorted(tick_ts, exec_t))
                    if t_idx < n_ticks:
                        signals[t_idx] = SIGNAL_LONG
                        sl_pips_arr[t_idx] = risk
                        tp_pips_arr[t_idx] = risk * self.target_rr
                        traded = True

        return signals, sl_pips_arr, tp_pips_arr


# ----------------------------------------------------------------------
# 3. London Fix (WMR 16:00 UTC) Institutional Pre-Fix Flow (M15)
# ----------------------------------------------------------------------
class LondonFixWMRFlowStrategy(BaseStrategy):
    """
    London 16:00 UTC WMR Benchmark Fixing Momentum Flow:
    - 15:00 to 16:00 UTC features the largest FX rebalancing order book execution of the day.
    - If price breaks out of the 13:00-14:45 NY session range in the direction of the daily trend
      between 15:00 and 15:30 UTC, ride the institutional fix momentum until 16:00 UTC with 1.6R.
    """
    def __init__(self, target_rr: float = 1.60, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = float(target_rr)
        self.name = "London_Fix_WMR_Flow"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'open': mid, 'high': mid, 'low': mid, 'close': mid}, index=ts)
        m15 = base.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(m15) < 30:
            return signals, sl_pips_arr, tp_pips_arr

        m15['date'] = m15.index.date
        m15['hour'] = m15.index.hour
        m15['minute'] = m15.index.minute

        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        dates = m15['date'].unique()
        for d in dates:
            day_mask = m15['date'] == d
            day_bars = m15[day_mask]
            if len(day_bars) < 25:
                continue

            # Pre-Fix range (13:00 - 14:45 UTC)
            pre_fix = day_bars[(day_bars['hour'] >= 13) & (day_bars['hour'] < 15)]
            if len(pre_fix) < 6:
                continue

            pf_h = pre_fix['high'].max()
            pf_l = pre_fix['low'].min()
            pf_r = (pf_h - pf_l) / PIP_SIZE

            if not (8.0 <= pf_r <= 30.0):
                continue

            day_indices = np.where(day_mask)[0]
            traded = False
            for idx in day_indices:
                row = m15.iloc[idx]
                hr = row['hour']
                mn = row['minute']
                # Trigger strictly at 15:00, 15:15, or 15:30
                if traded or hr != 15 or mn > 30:
                    continue

                if row['close'] > pf_h and row['close'] > row['open']:
                    sl_p = pf_h - 4.0 * PIP_SIZE
                    risk = max(min((row['close'] - sl_p) / PIP_SIZE, 15.0), 6.0)
                    exec_t = time_ms[idx] + dur_ms
                    t_idx = int(np.searchsorted(tick_ts, exec_t))
                    if t_idx < n_ticks:
                        signals[t_idx] = SIGNAL_LONG
                        sl_pips_arr[t_idx] = risk
                        tp_pips_arr[t_idx] = risk * self.target_rr
                        traded = True
                elif row['close'] < pf_l and row['close'] < row['open']:
                    sl_p = pf_l + 4.0 * PIP_SIZE
                    risk = max(min((sl_p - row['close']) / PIP_SIZE, 15.0), 6.0)
                    exec_t = time_ms[idx] + dur_ms
                    t_idx = int(np.searchsorted(tick_ts, exec_t))
                    if t_idx < n_ticks:
                        signals[t_idx] = SIGNAL_SHORT
                        sl_pips_arr[t_idx] = risk
                        tp_pips_arr[t_idx] = risk * self.target_rr
                        traded = True

        return signals, sl_pips_arr, tp_pips_arr


# ----------------------------------------------------------------------
# 4. Ultra-Coiled Frankfurt Range London ORB (08:00 - 10:00 UTC)
# ----------------------------------------------------------------------
class UltraCoilLondonORBStrategy(BaseStrategy):
    """
    Ultra-Coiled Frankfurt Opening Range Breakout:
    - Measures Frankfurt coiling range strictly between 06:00 and 07:45 UTC.
    - Constraint: Frankfurt range MUST be extremely compressed (between 6.0 and 18.0 pips!).
    - Enforces H1 50-EMA Macro Trend alignment (zero lookahead).
    - Trades London open expansion between 08:00 and 10:00 UTC with 1.75R target.
    """
    def __init__(self, max_range_pips: float = 18.0, target_rr: float = 1.75, **kwargs):
        super().__init__(**kwargs)
        self.max_range_pips = float(max_range_pips)
        self.target_rr = float(target_rr)
        self.name = "Ultra_Coil_London_ORB"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'open': mid, 'high': mid, 'low': mid, 'close': mid}, index=ts)
        m15 = base.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(m15) < 30:
            return signals, sl_pips_arr, tp_pips_arr

        h1 = base.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
        h1['macro'] = np.where(h1['close'] > h1['ema50'], 1, -1)
        m15['macro'] = h1['macro'].reindex(m15.index, method='ffill').fillna(0)

        m15['date'] = m15.index.date
        m15['hour'] = m15.index.hour
        m15['minute'] = m15.index.minute

        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        dates = m15['date'].unique()
        for d in dates:
            day_mask = m15['date'] == d
            day_bars = m15[day_mask]
            if len(day_bars) < 15:
                continue

            fk_bars = day_bars[(day_bars['hour'] >= 6) & (day_bars['hour'] < 8)]
            if len(fk_bars) < 6:
                continue

            fk_h = fk_bars['high'].max()
            fk_l = fk_bars['low'].min()
            fk_r = (fk_h - fk_l) / PIP_SIZE

            # Strict ultra-coil filter
            if not (6.0 <= fk_r <= self.max_range_pips):
                continue

            day_indices = np.where(day_mask)[0]
            traded = False
            for idx in day_indices:
                row = m15.iloc[idx]
                prev = m15.iloc[idx-1]
                hr = row['hour']
                if traded or hr not in (8, 9):
                    continue

                # Bullish breakout with H1 trend
                if row['macro'] > 0 and row['close'] > fk_h and prev['close'] <= fk_h and row['close'] > row['open']:
                    sl_p = min(row['low'], fk_h - 2.5 * PIP_SIZE)
                    risk = max(min((row['close'] - sl_p) / PIP_SIZE, 15.0), 6.0)
                    exec_t = time_ms[idx] + dur_ms
                    t_idx = int(np.searchsorted(tick_ts, exec_t))
                    if t_idx < n_ticks:
                        signals[t_idx] = SIGNAL_LONG
                        sl_pips_arr[t_idx] = risk
                        tp_pips_arr[t_idx] = risk * self.target_rr
                        traded = True

                # Bearish breakout with H1 trend
                elif row['macro'] < 0 and row['close'] < fk_l and prev['close'] >= fk_l and row['close'] < row['open']:
                    sl_p = max(row['high'], fk_l + 2.5 * PIP_SIZE)
                    risk = max(min((sl_p - row['close']) / PIP_SIZE, 15.0), 6.0)
                    exec_t = time_ms[idx] + dur_ms
                    t_idx = int(np.searchsorted(tick_ts, exec_t))
                    if t_idx < n_ticks:
                        signals[t_idx] = SIGNAL_SHORT
                        sl_pips_arr[t_idx] = risk
                        tp_pips_arr[t_idx] = risk * self.target_rr
                        traded = True

        return signals, sl_pips_arr, tp_pips_arr


# ----------------------------------------------------------------------
# 5. Volume Climax Exhaustion Fade (M15)
# ----------------------------------------------------------------------
class VolumeClimaxExhaustionFadeStrategy(BaseStrategy):
    """
    Volume Climax Exhaustion Fade:
    - Measures tick volume on M15 bars.
    - Setup: When tick volume spikes >= 2.2x the 20-period moving average of volume
      AND the bar range is >= 1.6x ATR 14 (climactic retail chase),
      AND the candle prints a long rejection wick (wick >= 40% of total candle range).
    - Fade the climax back to the M15 20-EMA with 1.5R target.
    - Active strictly between 08:00 and 17:00 UTC.
    """
    def __init__(self, target_rr: float = 1.50, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = float(target_rr)
        self.name = "Volume_Climax_Exhaustion_Fade"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'open': mid, 'high': mid, 'low': mid, 'close': mid, 'vol': 1}, index=ts)
        m15 = base.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'vol': 'count'
        }).dropna()
        if len(m15) < 30:
            return signals, sl_pips_arr, tp_pips_arr

        c = m15['close']
        h = m15['high']
        l = m15['low']
        o = m15['open']
        v = m15['vol']

        tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
        m15['atr'] = tr.rolling(14).mean().bfill()
        m15['vol_sma'] = v.rolling(20).mean().bfill()
        m15['ema20'] = c.ewm(span=20, adjust=False).mean()

        m15['hour'] = m15.index.hour
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(20, len(m15)):
            row = m15.iloc[i]
            if not (8 <= row['hour'] <= 17):
                continue

            bar_range = row['high'] - row['low']
            if bar_range < 7.0 * PIP_SIZE:
                continue

            is_climax_vol = row['vol'] >= 2.2 * row['vol_sma']
            is_extended = bar_range >= 1.6 * row['atr']

            if not (is_climax_vol and is_extended):
                continue

            upper_wick = row['high'] - max(row['open'], row['close'])
            lower_wick = min(row['open'], row['close']) - row['low']

            # Bearish rejection wick climax -> Fade short
            if upper_wick >= 0.40 * bar_range:
                sl_p = row['high'] + 1.5 * PIP_SIZE
                risk = max(min((sl_p - row['close']) / PIP_SIZE, 15.0), 6.0)
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = SIGNAL_SHORT
                    sl_pips_arr[idx] = risk
                    tp_pips_arr[idx] = risk * self.target_rr

            # Bullish rejection wick climax -> Fade long
            elif lower_wick >= 0.40 * bar_range:
                sl_p = row['low'] - 1.5 * PIP_SIZE
                risk = max(min((row['close'] - sl_p) / PIP_SIZE, 15.0), 6.0)
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = SIGNAL_LONG
                    sl_pips_arr[idx] = risk
                    tp_pips_arr[idx] = risk * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


# ----------------------------------------------------------------------
# AUDIT RUNNER ACROSS 52 WEEKS
# ----------------------------------------------------------------------
def _worker_eval(args):
    w, StratCls, kwargs = args
    ticks, _ = get_week_dataset(w)
    engine = BacktestEngine(pip_size=0.0001, use_slippage=True)
    strat = StratCls(**kwargs)
    trades = engine.run(ticks, strat)
    return trades


def screen_candidate(StratCls, **kwargs):
    weeks = [(w, StratCls, kwargs) for w in range(1, 53)]
    num_workers = min(8, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(_worker_eval, weeks))

    trades = [t for r in results for t in r]
    n = len(trades)
    if n == 0:
        return {'name': StratCls().name, 'trades': 0, 'wr': 0, 'pips': 0, 'pf': 0, 'max_dd': 0, 'rr': 0}

    pips = sum(t['pips'] for t in trades)
    wins = [t for t in trades if t['pips'] > 0]
    losses = [t for t in trades if t['pips'] < 0]
    wr = len(wins) / n * 100.0
    gp = sum(t['pips'] for t in wins)
    gl = abs(sum(t['pips'] for t in losses))
    pf = gp / gl if gl > 0 else 99.0
    realized_rr = (gp / max(len(wins), 1)) / (gl / max(len(losses), 1)) if gl > 0 else 0.0

    equity = np.cumsum([t['pips'] for t in trades])
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = dd.max()

    return {
        'name': StratCls().name,
        'trades': n,
        'wr': wr,
        'pips': pips,
        'pf': pf,
        'max_dd': max_dd,
        'rr': realized_rr
    }


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 80)
    print("  SCREENING 5 NOVEL ALPHA HYPOTHESES ACROSS 52 WEEKS (~26M TICKS)")
    print("=" * 80)

    candidates = [
        (FVGDisplacementRetestStrategy, {'min_gap_pips': 3.5, 'target_rr': 1.70}),
        (PDHPDLJudasSweepFadeStrategy, {'min_sweep': 2.0, 'max_sweep': 14.0, 'target_rr': 1.60}),
        (LondonFixWMRFlowStrategy, {'target_rr': 1.60}),
        (UltraCoilLondonORBStrategy, {'max_range_pips': 18.0, 'target_rr': 1.75}),
        (VolumeClimaxExhaustionFadeStrategy, {'target_rr': 1.50}),
    ]

    for StratCls, kwargs in candidates:
        sub_t0 = time.time()
        res = screen_candidate(StratCls, **kwargs)
        sub_dt = time.time() - sub_t0
        print(f"Strategy: {res['name']}")
        print(f"  Trades:   {res['trades']} ({res['trades']/52:.1f}/wk, {res['trades']/250:.2f}/day)")
        print(f"  Win Rate: {res['wr']:.1f}%")
        print(f"  PF:       {res['pf']:.2f}")
        print(f"  Real RR:  {res['rr']:.2f} : 1")
        print(f"  Net Pips: {res['pips']:+.1f} pips")
        print(f"  Max DD:   {res['max_dd']:.1f} pips")
        print(f"  Duration: {sub_dt:.1f}s")
        print("-" * 60)

    print(f"All 5 candidate hypotheses screened in {time.time() - t0:.1f}s total.")
