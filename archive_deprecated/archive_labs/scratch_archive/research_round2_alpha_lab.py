#!/usr/bin/env python3
"""
================================================================================
  RESEARCH LAB ROUND 2: 5 NOVEL QUANTITATIVE ALPHA HYPOTHESES
================================================================================
  1. European_Midday_UltraCoil_Breakout: 11:00-12:45 UTC range <= 15p, 13:00 NY breakout
  2. Asian_Sweep_Frankfurt_Reversal: Asian High/Low swept by 2-8p at 05:00-06:30 UTC
  3. M30_Historical_Volatility_Squeeze_Donchian: M30 Donchian Breakout when ATR <= 25th percentile
  4. Daily_Pivot_S1_R1_Rejection: Institutional bounce off Daily S1/R1 in liquid hours
  5. London_50pct_Fib_Pullback: Retracement to 50% equilibrium of London morning range
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

PIP_SIZE = 0.0001


# ----------------------------------------------------------------------
# 1. European Midday Ultra-Coil NY Cash Open Breakout (M15)
# ----------------------------------------------------------------------
class EuropeanMiddayUltraCoilBreakoutStrategy(BaseStrategy):
    """
    Measures the European Midday Range strictly between 11:00 and 12:45 UTC.
    Constraint: Range MUST be an ultra-tight coil (5.0 to 15.0 pips!).
    Trigger: NY Cash Open (13:00 - 15:00 UTC) directional breakout with H1 50-EMA trend.
    Target: 1.75R.
    """
    def __init__(self, max_range_pips: float = 15.0, target_rr: float = 1.75, **kwargs):
        super().__init__(**kwargs)
        self.max_range_pips = float(max_range_pips)
        self.target_rr = float(target_rr)
        self.name = "European_Midday_UltraCoil_Breakout"

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
            if len(day_bars) < 20:
                continue

            midday = day_bars[(day_bars['hour'] >= 11) & (day_bars['hour'] < 13)]
            if len(midday) < 6:
                continue

            box_h = midday['high'].max()
            box_l = midday['low'].min()
            box_r = (box_h - box_l) / PIP_SIZE

            if not (5.0 <= box_r <= self.max_range_pips):
                continue

            day_indices = np.where(day_mask)[0]
            traded = False
            for idx in day_indices:
                row = m15.iloc[idx]
                prev = m15.iloc[idx-1]
                hr = row['hour']
                if traded or hr not in (13, 14):
                    continue

                if row['macro'] > 0 and row['close'] > box_h and prev['close'] <= box_h and row['close'] > row['open']:
                    sl_p = min(row['low'], box_h - 2.5 * PIP_SIZE)
                    risk = max(min((row['close'] - sl_p) / PIP_SIZE, 15.0), 6.0)
                    exec_t = time_ms[idx] + dur_ms
                    t_idx = int(np.searchsorted(tick_ts, exec_t))
                    if t_idx < n_ticks:
                        signals[t_idx] = SIGNAL_LONG
                        sl_pips_arr[t_idx] = risk
                        tp_pips_arr[t_idx] = risk * self.target_rr
                        traded = True
                elif row['macro'] < 0 and row['close'] < box_l and prev['close'] >= box_l and row['close'] < row['open']:
                    sl_p = max(row['high'], box_l + 2.5 * PIP_SIZE)
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
# 2. Asian Range Sweep into Frankfurt Liquidity Reversal (M15)
# ----------------------------------------------------------------------
class AsianSweepFrankfurtReversalStrategy(BaseStrategy):
    """
    Between 00:00 and 04:30 UTC, Asian session establishes high and low extremes.
    Between 05:00 and 07:00 UTC (Frankfurt open), early orders sweep beyond Asian High/Low by 2.0 to 10.0 pips.
    When price fails to expand and closes back INSIDE the Asian range, fade the false sweep.
    Target: 1.6R back towards Asian median.
    """
    def __init__(self, min_sweep: float = 2.0, max_sweep: float = 10.0, target_rr: float = 1.60, **kwargs):
        super().__init__(**kwargs)
        self.min_sweep = float(min_sweep)
        self.max_sweep = float(max_sweep)
        self.target_rr = float(target_rr)
        self.name = "Asian_Sweep_Frankfurt_Reversal"

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
            if len(day_bars) < 15:
                continue

            asia = day_bars[(day_bars['hour'] >= 0) & (day_bars['hour'] < 5)]
            if len(asia) < 6:
                continue

            asia_h = asia['high'].max()
            asia_l = asia['low'].min()
            asia_r = (asia_h - asia_l) / PIP_SIZE

            if not (8.0 <= asia_r <= 35.0):
                continue

            day_indices = np.where(day_mask)[0]
            traded = False
            for idx in day_indices:
                row = m15.iloc[idx]
                hr = row['hour']
                if traded or hr not in (5, 6):
                    continue

                sweep_up = (row['high'] - asia_h) / PIP_SIZE
                sweep_dn = (asia_l - row['low']) / PIP_SIZE

                if sweep_up >= self.min_sweep and sweep_up <= self.max_sweep and row['close'] < asia_h and row['close'] < row['open']:
                    sl_p = row['high'] + 1.2 * PIP_SIZE
                    risk = max(min((sl_p - row['close']) / PIP_SIZE, 14.0), 5.5)
                    exec_t = time_ms[idx] + dur_ms
                    t_idx = int(np.searchsorted(tick_ts, exec_t))
                    if t_idx < n_ticks:
                        signals[t_idx] = SIGNAL_SHORT
                        sl_pips_arr[t_idx] = risk
                        tp_pips_arr[t_idx] = risk * self.target_rr
                        traded = True
                elif sweep_dn >= self.min_sweep and sweep_dn <= self.max_sweep and row['close'] > asia_l and row['close'] > row['open']:
                    sl_p = row['low'] - 1.2 * PIP_SIZE
                    risk = max(min((row['close'] - sl_p) / PIP_SIZE, 14.0), 5.5)
                    exec_t = time_ms[idx] + dur_ms
                    t_idx = int(np.searchsorted(tick_ts, exec_t))
                    if t_idx < n_ticks:
                        signals[t_idx] = SIGNAL_LONG
                        sl_pips_arr[t_idx] = risk
                        tp_pips_arr[t_idx] = risk * self.target_rr
                        traded = True

        return signals, sl_pips_arr, tp_pips_arr


# ----------------------------------------------------------------------
# 3. M30 Historical Volatility Squeeze Donchian Breakout (M30)
# ----------------------------------------------------------------------
class M30HistoricalVolatilitySqueezeDonchianStrategy(BaseStrategy):
    """
    On M30 timeframe, monitors 20-period Donchian Channels.
    Filter: 14-period ATR must be in the bottom 30% of its 80-bar rolling range (extreme volatility contraction).
    Trigger: When price closes beyond 20-period Donchian Channel during London/NY liquid hours (07:00 - 15:30 UTC).
    Target: 1.80R.
    """
    def __init__(self, lookback: int = 20, target_rr: float = 1.80, **kwargs):
        super().__init__(**kwargs)
        self.lookback = int(lookback)
        self.target_rr = float(target_rr)
        self.allowed_hours = set(range(7, 16))
        self.name = "M30_HV_Squeeze_Donchian"

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
        m30 = base.resample('30min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(m30) < 60:
            return signals, sl_pips_arr, tp_pips_arr

        c = m30['close']
        h = m30['high']
        l = m30['low']

        # Donchian 20
        m30['dc_h'] = h.rolling(self.lookback).max().shift(1)
        m30['dc_l'] = l.rolling(self.lookback).min().shift(1)

        # ATR 14 and rolling percentile
        tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
        m30['atr'] = tr.rolling(14).mean().bfill()
        m30['atr_min80'] = m30['atr'].rolling(80).min().bfill()
        m30['atr_max80'] = m30['atr'].rolling(80).max().bfill()
        m30['atr_rank'] = (m30['atr'] - m30['atr_min80']) / np.maximum(m30['atr_max80'] - m30['atr_min80'], 1e-9)

        m30['hour'] = m30.index.hour
        time_ms = m30.index.astype('datetime64[ms]').view('int64')
        dur_ms = 30 * 60 * 1000
        tick_ts = ticks['timestamp']

        c_vals = c.values
        o_vals = m30['open'].values
        l_vals = l.values
        h_vals = h.values
        dc_h = m30['dc_h'].values
        dc_l = m30['dc_l'].values
        rank_vals = m30['atr_rank'].values
        hours = m30['hour'].values

        for i in range(50, len(m30)):
            if hours[i] not in self.allowed_hours:
                continue

            # Volatility must be compressed (bottom 30% of 80-bar range)
            if rank_vals[i-1] > 0.30:
                continue

            sig = 0
            risk_p = 0.0

            # Bullish expansion
            if c_vals[i] > dc_h[i] and c_vals[i-1] <= dc_h[i-1] and c_vals[i] > o_vals[i]:
                sl_p = min(l_vals[i], dc_l[i])
                risk = max(min((c_vals[i] - sl_p) / PIP_SIZE, 18.0), 6.5)
                sig = SIGNAL_LONG
                risk_p = risk
            # Bearish expansion
            elif c_vals[i] < dc_l[i] and c_vals[i-1] >= dc_l[i-1] and c_vals[i] < o_vals[i]:
                sl_p = max(h_vals[i], dc_h[i])
                risk = max(min((sl_p - c_vals[i]) / PIP_SIZE, 18.0), 6.5)
                sig = SIGNAL_SHORT
                risk_p = risk

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = risk_p
                    tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


# ----------------------------------------------------------------------
# 4. Daily Pivot S1/R1 Institutional Bounce (M15)
# ----------------------------------------------------------------------
class DailyPivotRejectionStrategy(BaseStrategy):
    """
    Computes Standard Floor Trader Daily Pivot points:
    P = (H + L + C) / 3
    S1 = 2P - H, R1 = 2P - L
    When price tests S1 or R1 during liquid hours (08:00 - 16:00 UTC) with an RSI reversal wick,
    trade the mean-reversion back to Central Pivot P with 1.6R.
    """
    def __init__(self, target_rr: float = 1.60, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = float(target_rr)
        self.name = "Daily_Pivot_S1_R1_Rejection"

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

        daily = base.resample('1D').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        daily['p'] = (daily['high'] + daily['low'] + daily['close']) / 3.0
        daily['r1'] = (2.0 * daily['p']) - daily['low']
        daily['s1'] = (2.0 * daily['p']) - daily['high']

        m15['p'] = daily['p'].shift(1).reindex(m15.index, method='ffill')
        m15['r1'] = daily['r1'].shift(1).reindex(m15.index, method='ffill')
        m15['s1'] = daily['s1'].shift(1).reindex(m15.index, method='ffill')

        # RSI 14
        delta = m15['close'].diff()
        gain = delta.clip(lower=0.0).ewm(alpha=1.0/14, adjust=False).mean()
        loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0/14, adjust=False).mean()
        rs = gain / np.maximum(loss, 1e-9)
        m15['rsi'] = 100.0 - (100.0 / (1.0 + rs))

        m15['hour'] = m15.index.hour
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        dates = m15.index.date
        unique_dates = np.unique(dates)

        for d in unique_dates:
            day_mask = dates == d
            day_indices = np.where(day_mask)[0]
            traded = False
            for idx in day_indices:
                row = m15.iloc[idx]
                hr = row['hour']
                if traded or hr not in range(8, 17):
                    continue

                r1 = row['r1']
                s1 = row['s1']
                p = row['p']
                if pd.isna(r1) or pd.isna(s1):
                    continue

                # R1 Resistance rejection -> Short
                if row['high'] >= r1 - 1.5 * PIP_SIZE and row['close'] < r1 and row['close'] < row['open'] and row['rsi'] >= 55.0:
                    sl_p = row['high'] + 1.2 * PIP_SIZE
                    risk = max(min((sl_p - row['close']) / PIP_SIZE, 15.0), 6.0)
                    exec_t = time_ms[idx] + dur_ms
                    t_idx = int(np.searchsorted(tick_ts, exec_t))
                    if t_idx < n_ticks:
                        signals[t_idx] = SIGNAL_SHORT
                        sl_pips_arr[t_idx] = risk
                        tp_pips_arr[t_idx] = risk * self.target_rr
                        traded = True

                # S1 Support rejection -> Long
                elif row['low'] <= s1 + 1.5 * PIP_SIZE and row['close'] > s1 and row['close'] > row['open'] and row['rsi'] <= 45.0:
                    sl_p = row['low'] - 1.2 * PIP_SIZE
                    risk = max(min((row['close'] - sl_p) / PIP_SIZE, 15.0), 6.0)
                    exec_t = time_ms[idx] + dur_ms
                    t_idx = int(np.searchsorted(tick_ts, exec_t))
                    if t_idx < n_ticks:
                        signals[t_idx] = SIGNAL_LONG
                        sl_pips_arr[t_idx] = risk
                        tp_pips_arr[t_idx] = risk * self.target_rr
                        traded = True

        return signals, sl_pips_arr, tp_pips_arr


# ----------------------------------------------------------------------
# 5. London Morning 50% Equilibrium Retracement (M15)
# ----------------------------------------------------------------------
class LondonMorningEquilibriumRetracementStrategy(BaseStrategy):
    """
    Between 07:00 and 10:30 UTC, London session establishes the morning expansion.
    Between 10:30 and 12:30 UTC, if price pulls back to the 50% equilibrium midpoint
    of that morning expansion in the direction of the dominant morning impulse:
    Enter with a tight stop beyond the 61.8% fib level, targeting the morning extreme (1.65R).
    """
    def __init__(self, target_rr: float = 1.65, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = float(target_rr)
        self.name = "London_Morning_Equilibrium_Retracement"

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
            if len(day_bars) < 20:
                continue

            morn = day_bars[(day_bars['hour'] >= 7) & (day_bars['hour'] < 10)]
            if len(morn) < 8:
                continue

            morn_h = morn['high'].max()
            morn_l = morn['low'].min()
            morn_r = (morn_h - morn_l) / PIP_SIZE

            if not (15.0 <= morn_r <= 50.0):
                continue

            eq_mid = 0.5 * (morn_h + morn_l)
            fib618_up = morn_h - 0.618 * (morn_h - morn_l)
            fib618_dn = morn_l + 0.618 * (morn_h - morn_l)

            # Determine dominant morning impulse direction
            first_morn_open = morn.iloc[0]['open']
            last_morn_close = morn.iloc[-1]['close']
            is_bullish = last_morn_close > first_morn_open

            day_indices = np.where(day_mask)[0]
            traded = False
            for idx in day_indices:
                row = m15.iloc[idx]
                hr = row['hour']
                mn = row['minute']
                if traded or not ((hr == 10 and mn >= 30) or hr == 11 or (hr == 12 and mn <= 30)):
                    continue

                if is_bullish:
                    # Pullback to 50% equilibrium -> Buy
                    if row['low'] <= eq_mid and row['close'] > eq_mid and row['close'] > row['open']:
                        sl_p = fib618_up - 1.2 * PIP_SIZE
                        risk = max(min((row['close'] - sl_p) / PIP_SIZE, 15.0), 5.5)
                        exec_t = time_ms[idx] + dur_ms
                        t_idx = int(np.searchsorted(tick_ts, exec_t))
                        if t_idx < n_ticks:
                            signals[t_idx] = SIGNAL_LONG
                            sl_pips_arr[t_idx] = risk
                            tp_pips_arr[t_idx] = risk * self.target_rr
                            traded = True
                else:
                    # Pullback to 50% equilibrium -> Sell
                    if row['high'] >= eq_mid and row['close'] < eq_mid and row['close'] < row['open']:
                        sl_p = fib618_dn + 1.2 * PIP_SIZE
                        risk = max(min((sl_p - row['close']) / PIP_SIZE, 15.0), 5.5)
                        exec_t = time_ms[idx] + dur_ms
                        t_idx = int(np.searchsorted(tick_ts, exec_t))
                        if t_idx < n_ticks:
                            signals[t_idx] = SIGNAL_SHORT
                            sl_pips_arr[t_idx] = risk
                            tp_pips_arr[t_idx] = risk * self.target_rr
                            traded = True

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
    print("  SCREENING ROUND 2: 5 NOVEL ALPHA HYPOTHESES ACROSS 52 WEEKS (~26M TICKS)")
    print("=" * 80)

    candidates = [
        (EuropeanMiddayUltraCoilBreakoutStrategy, {'max_range_pips': 15.0, 'target_rr': 1.75}),
        (AsianSweepFrankfurtReversalStrategy, {'min_sweep': 2.0, 'max_sweep': 10.0, 'target_rr': 1.60}),
        (M30HistoricalVolatilitySqueezeDonchianStrategy, {'lookback': 20, 'target_rr': 1.80}),
        (DailyPivotRejectionStrategy, {'target_rr': 1.60}),
        (LondonMorningEquilibriumRetracementStrategy, {'target_rr': 1.65}),
    ]

    for StratCls, kwargs in candidates:
        sub_t0 = time.time()
        res = screen_candidate(StratCls, **kwargs)
        sub_dt = time.time() - sub_t0
        print(f"Strategy: {res['name']}")
        print(f"  Trades:   {res['trades']} ({res['trades']/52:.1f}/wk)")
        print(f"  Win Rate: {res['wr']:.1f}%")
        print(f"  PF:       {res['pf']:.2f}")
        print(f"  Real RR:  {res['rr']:.2f} : 1")
        print(f"  Net Pips: {res['pips']:+.1f} pips")
        print(f"  Max DD:   {res['max_dd']:.1f} pips")
        print(f"  Duration: {sub_dt:.1f}s")
        print("-" * 60)

    print(f"All 5 candidate hypotheses screened in {time.time() - t0:.1f}s total.")
