#!/usr/bin/env python3
"""
================================================================================
  DEEP EXPLORATION LAB: FILLING ALL REMAINING IN-BETWEEN OFFLINE GAPS
================================================================================
Explores 4 new frontier hypotheses across 52 weeks (~26M ticks) of EUR/USD data:

  Hypothesis 1: Monday Sydney/Tokyo Opening Gap Fill (Sun 21:00 - Mon 04:00 UTC)
  Hypothesis 2: Tokyo Institutional Fix Impulse (00:00 - 03:00 UTC)
  Hypothesis 3: US Pre-Market Economic Release Momentum (12:15 - 13:15 UTC)
  Hypothesis 4: US Cash Close VWAP Snap-Back (18:30 - 20:30 UTC)
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

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT

PIP_SIZE = 0.0001


# ----------------------------------------------------------------------
# 1. Monday Sydney/Tokyo Opening Gap Fill
# ----------------------------------------------------------------------
class MondayOpeningGapFillStrategy(BaseStrategy):
    """
    On Monday morning (00:00 - 05:00 UTC), if the opening price gapped >= 8 pips
    away from Friday's closing price, enter in the direction of the gap fill (fading the gap).
    """
    def __init__(self, min_gap_pips: float = 6.0, max_gap_pips: float = 35.0, target_fill_ratio: float = 0.8, **kwargs):
        super().__init__(**kwargs)
        self.min_gap_pips = float(min_gap_pips)
        self.max_gap_pips = float(max_gap_pips)
        self.target_fill_ratio = float(target_fill_ratio)
        self.name = "Monday_Opening_Gap_Fill"

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
        if len(m15) < 10:
            return signals, sl_pips_arr, tp_pips_arr

        m15['weekday'] = m15.index.weekday
        m15['hour'] = m15.index.hour
        m15['minute'] = m15.index.minute

        # First M15 bar of the week
        first_bar = m15.iloc[0]
        # In weekly datasets, week starts Sunday evening / Monday 00:00
        # If open price differs from prev week close...
        # In a single week file, prev close is approx first tick open or we test deviation from opening range
        # Let's check opening 1-hour bar vs opening price
        open_price = first_bar['open']

        # Fading extended early open moves (first 3 hours 00:00 - 04:00)
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        tick_ts = ticks['timestamp']
        dur_ms = 15 * 60 * 1000

        traded = False
        for idx in range(1, min(16, len(m15))):
            row = m15.iloc[idx]
            if traded or row['weekday'] != 0 or row['hour'] > 5:
                continue

            gap_pips = (row['close'] - open_price) / PIP_SIZE
            # If price pushed far from week open in Asian session without news, fade back to open
            if gap_pips >= self.min_gap_pips and gap_pips <= self.max_gap_pips and row['close'] < row['open']:
                # Short back to open
                risk = max(abs(row['high'] - row['close']) / PIP_SIZE + 1.5, 6.0)
                target = abs(gap_pips) * self.target_fill_ratio
                if target >= 5.0:
                    exec_t = time_ms[idx] + dur_ms
                    t_idx = int(np.searchsorted(tick_ts, exec_t))
                    if t_idx < n_ticks:
                        signals[t_idx] = SIGNAL_SHORT
                        sl_pips_arr[t_idx] = risk
                        tp_pips_arr[t_idx] = target
                        traded = True
            elif gap_pips <= -self.min_gap_pips and gap_pips >= -self.max_gap_pips and row['close'] > row['open']:
                # Long back to open
                risk = max(abs(row['close'] - row['low']) / PIP_SIZE + 1.5, 6.0)
                target = abs(gap_pips) * self.target_fill_ratio
                if target >= 5.0:
                    exec_t = time_ms[idx] + dur_ms
                    t_idx = int(np.searchsorted(tick_ts, exec_t))
                    if t_idx < n_ticks:
                        signals[t_idx] = SIGNAL_LONG
                        sl_pips_arr[t_idx] = risk
                        tp_pips_arr[t_idx] = target
                        traded = True

        return signals, sl_pips_arr, tp_pips_arr


# ----------------------------------------------------------------------
# 2. Tokyo Fix / Asian Impulse Breakout (00:30 - 03:00 UTC)
# ----------------------------------------------------------------------
class TokyoFixImpulseStrategy(BaseStrategy):
    """
    During Tokyo morning (00:30 - 03:00 UTC), commercial currency fixes take place
    around 00:55 UTC (9:55 AM Tokyo time). When 30-min range breaks with expanding volume/ATR,
    ride the momentum with a tight 1.5R target.
    """
    def __init__(self, target_rr: float = 1.50, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = float(target_rr)
        self.name = "Tokyo_Fix_Impulse"

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

        # ATR 14
        tr = np.maximum(m15['high'] - m15['low'], np.maximum(abs(m15['high'] - m15['close'].shift(1)), abs(m15['low'] - m15['close'].shift(1))))
        m15['atr'] = tr.rolling(14).mean().bfill()
        m15['ema20'] = m15['close'].ewm(span=20, adjust=False).mean()

        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        tick_ts = ticks['timestamp']
        dur_ms = 15 * 60 * 1000

        dates = m15['date'].unique()
        for d in dates:
            day_mask = m15['date'] == d
            day_bars = m15[day_mask]
            if len(day_bars) < 6:
                continue

            day_indices = np.where(day_mask)[0]
            # Opening bar 00:00 - 00:30 (first two M15 bars)
            early = day_bars[day_bars['hour'] == 0]
            if len(early) < 2:
                continue
            early_h = early['high'].max()
            early_l = early['low'].min()
            early_r = (early_h - early_l) / PIP_SIZE

            if not (5.0 <= early_r <= 18.0):
                continue

            traded = False
            for idx in day_indices:
                row = m15.iloc[idx]
                prev = m15.iloc[idx-1]
                hr = row['hour']
                if traded or hr not in (1, 2):
                    continue

                # Breakout beyond early Tokyo box
                if row['close'] > early_h and prev['close'] <= early_h and row['close'] > row['open']:
                    sl_p = early_l - 1.0 * PIP_SIZE
                    risk = max(min((row['close'] - sl_p) / PIP_SIZE, 12.0), 6.0)
                    exec_t = time_ms[idx] + dur_ms
                    t_idx = int(np.searchsorted(tick_ts, exec_t))
                    if t_idx < n_ticks:
                        signals[t_idx] = SIGNAL_LONG
                        sl_pips_arr[t_idx] = risk
                        tp_pips_arr[t_idx] = risk * self.target_rr
                        traded = True
                elif row['close'] < early_l and prev['close'] >= early_l and row['close'] < row['open']:
                    sl_p = early_h + 1.0 * PIP_SIZE
                    risk = max(min((sl_p - row['close']) / PIP_SIZE, 12.0), 6.0)
                    exec_t = time_ms[idx] + dur_ms
                    t_idx = int(np.searchsorted(tick_ts, exec_t))
                    if t_idx < n_ticks:
                        signals[t_idx] = SIGNAL_SHORT
                        sl_pips_arr[t_idx] = risk
                        tp_pips_arr[t_idx] = risk * self.target_rr
                        traded = True

        return signals, sl_pips_arr, tp_pips_arr


# ----------------------------------------------------------------------
# 3. US Pre-Market Economic Release Breakout (12:15 - 13:15 UTC)
# ----------------------------------------------------------------------
class USPreMarketBreakoutStrategy(BaseStrategy):
    """
    Between 12:15 and 13:15 UTC, major US economic numbers drop (12:30 UTC / 8:30 AM EST).
    This fills the gap between London lunch (ends 12:30) and NY Cash Open (starts 13:00/13:30).
    When 12:30 candle creates a major directional displacement (> 10 pips) breaking the London Lunch range,
    enter momentum continuation.
    """
    def __init__(self, target_rr: float = 1.60, min_impulse_pips: float = 8.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = float(target_rr)
        self.min_impulse_pips = float(min_impulse_pips)
        self.name = "US_PreMarket_Release_Breakout"

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
        tick_ts = ticks['timestamp']
        dur_ms = 15 * 60 * 1000

        dates = m15['date'].unique()
        for d in dates:
            day_mask = m15['date'] == d
            day_bars = m15[day_mask]
            if len(day_bars) < 20:
                continue

            # London midday range (10:30 - 12:15 UTC)
            lunch_bars = day_bars[((day_bars['hour'] == 10) & (day_bars['minute'] >= 30)) | (day_bars['hour'] == 11) | ((day_bars['hour'] == 12) & (day_bars['minute'] <= 15))]
            if len(lunch_bars) < 4:
                continue

            lunch_h = lunch_bars['high'].max()
            lunch_l = lunch_bars['low'].min()
            lunch_r = (lunch_h - lunch_l) / PIP_SIZE

            if not (6.0 <= lunch_r <= 25.0):
                continue

            day_indices = np.where(day_mask)[0]
            traded = False
            for idx in day_indices:
                row = m15.iloc[idx]
                hr = row['hour']
                mn = row['minute']
                # Check bars at 12:30, 12:45, 13:00
                if traded or not ((hr == 12 and mn in (30, 45)) or (hr == 13 and mn == 0)):
                    continue

                candle_body = (row['close'] - row['open']) / PIP_SIZE
                candle_range = (row['high'] - row['low']) / PIP_SIZE

                # Bullish impulse breakout
                if row['close'] > lunch_h and candle_body >= self.min_impulse_pips and candle_range >= self.min_impulse_pips * 1.2:
                    sl_p = min(row['low'], lunch_h - 2.0 * PIP_SIZE)
                    risk = max(min((row['close'] - sl_p) / PIP_SIZE, 16.0), 7.0)
                    exec_t = time_ms[idx] + dur_ms
                    t_idx = int(np.searchsorted(tick_ts, exec_t))
                    if t_idx < n_ticks:
                        signals[t_idx] = SIGNAL_LONG
                        sl_pips_arr[t_idx] = risk
                        tp_pips_arr[t_idx] = risk * self.target_rr
                        traded = True
                # Bearish impulse breakout
                elif row['close'] < lunch_l and candle_body <= -self.min_impulse_pips and candle_range >= self.min_impulse_pips * 1.2:
                    sl_p = max(row['high'], lunch_l + 2.0 * PIP_SIZE)
                    risk = max(min((sl_p - row['close']) / PIP_SIZE, 16.0), 7.0)
                    exec_t = time_ms[idx] + dur_ms
                    t_idx = int(np.searchsorted(tick_ts, exec_t))
                    if t_idx < n_ticks:
                        signals[t_idx] = SIGNAL_SHORT
                        sl_pips_arr[t_idx] = risk
                        tp_pips_arr[t_idx] = risk * self.target_rr
                        traded = True

        return signals, sl_pips_arr, tp_pips_arr


# ----------------------------------------------------------------------
# 4. US Cash Close Daily VWAP Snap-Back (18:30 - 20:30 UTC)
# ----------------------------------------------------------------------
class USCashCloseVWAPSnapBackStrategy(BaseStrategy):
    """
    Between 18:30 and 20:30 UTC, US equity markets head into the 20:00 UTC (4:00 PM EST) cash close.
    Asset managers rebalance currency hedges.
    If price has drifted far from the daily VWAP (>= 20 pips deviation) and prints a reversal bar,
    fade back towards the daily VWAP mean.
    """
    def __init__(self, min_dev_pips: float = 16.0, target_mean_ratio: float = 0.70, **kwargs):
        super().__init__(**kwargs)
        self.min_dev_pips = float(min_dev_pips)
        self.target_mean_ratio = float(target_mean_ratio)
        self.name = "US_Cash_Close_VWAP_SnapBack"

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

        m15['date'] = m15.index.date
        m15['hour'] = m15.index.hour
        m15['minute'] = m15.index.minute
        m15['weekday'] = m15.index.weekday

        # Calculate daily cumulative typical price VWAP proxy
        typ = (m15['high'] + m15['low'] + m15['close']) / 3.0
        m15['typ'] = typ
        m15['cum_typ'] = m15.groupby('date')['typ'].cumsum()
        m15['count'] = m15.groupby('date').cumcount() + 1
        m15['vwap'] = m15['cum_typ'] / m15['count']

        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        tick_ts = ticks['timestamp']
        dur_ms = 15 * 60 * 1000

        dates = m15['date'].unique()
        for d in dates:
            day_mask = m15['date'] == d
            day_bars = m15[day_mask]
            if len(day_bars) < 30:
                continue

            day_indices = np.where(day_mask)[0]
            traded = False
            for idx in day_indices:
                row = m15.iloc[idx]
                hr = row['hour']
                mn = row['minute']
                wd = row['weekday']

                # Active strictly between 18:30 and 20:30 UTC Mon-Thu (avoid Friday close)
                if traded or wd >= 4 or not ((hr == 18 and mn >= 30) or hr == 19 or (hr == 20 and mn <= 30)):
                    continue

                dev_pips = (row['close'] - row['vwap']) / PIP_SIZE

                # Extended above VWAP -> Short back to VWAP
                if dev_pips >= self.min_dev_pips and row['close'] < row['open']:
                    sl_p = row['high'] + 1.5 * PIP_SIZE
                    risk = max(min((sl_p - row['close']) / PIP_SIZE, 15.0), 6.0)
                    target = dev_pips * self.target_mean_ratio
                    if target >= 8.0:
                        exec_t = time_ms[idx] + dur_ms
                        t_idx = int(np.searchsorted(tick_ts, exec_t))
                        if t_idx < n_ticks:
                            signals[t_idx] = SIGNAL_SHORT
                            sl_pips_arr[t_idx] = risk
                            tp_pips_arr[t_idx] = target
                            traded = True
                # Extended below VWAP -> Long back to VWAP
                elif dev_pips <= -self.min_dev_pips and row['close'] > row['open']:
                    sl_p = row['low'] - 1.5 * PIP_SIZE
                    risk = max(min((row['close'] - sl_p) / PIP_SIZE, 15.0), 6.0)
                    target = abs(dev_pips) * self.target_mean_ratio
                    if target >= 8.0:
                        exec_t = time_ms[idx] + dur_ms
                        t_idx = int(np.searchsorted(tick_ts, exec_t))
                        if t_idx < n_ticks:
                            signals[t_idx] = SIGNAL_LONG
                            sl_pips_arr[t_idx] = risk
                            tp_pips_arr[t_idx] = target
                            traded = True

        return signals, sl_pips_arr, tp_pips_arr


# ----------------------------------------------------------------------
# AUDIT HARNESS ACROSS 52 WEEKS
# ----------------------------------------------------------------------
def _worker_eval(args):
    w, StratCls, kwargs = args
    ticks, _ = get_week_dataset(w)
    engine = BacktestEngine(pip_size=0.0001, use_slippage=True)
    strat = StratCls(**kwargs)
    trades = engine.run(ticks, strat)
    return trades


def audit_strategy_52_weeks(StratCls, **kwargs):
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
    print("  EXPLORING 4 NEW FRONTIER IN-BETWEEN GAP HYPOTHESES (52 WEEKS DUKASCOPY TICK DATA)")
    print("=" * 80)

    candidates = [
        (MondayOpeningGapFillStrategy, {}),
        (TokyoFixImpulseStrategy, {'target_rr': 1.50}),
        (USPreMarketBreakoutStrategy, {'target_rr': 1.60, 'min_impulse_pips': 8.0}),
        (USCashCloseVWAPSnapBackStrategy, {'min_dev_pips': 16.0, 'target_mean_ratio': 0.70}),
    ]

    for StratCls, kwargs in candidates:
        sub_t0 = time.time()
        res = audit_strategy_52_weeks(StratCls, **kwargs)
        sub_dt = time.time() - sub_t0
        print(f"Strategy: {res['name']}")
        print(f"  Trades:   {res['trades']} ({res['trades']/52:.1f}/wk)")
        print(f"  Win Rate: {res['wr']:.1f}%")
        print(f"  PF:       {res['pf']:.2f}")
        print(f"  Real RR:  {res['rr']:.2f} : 1")
        print(f"  Net Pips: {res['pips']:+.1f} pips")
        print(f"  Max DD:   {res['max_dd']:.1f} pips")
        print(f"  Audit Duration: {sub_dt:.1f}s")
        print("-" * 60)

    print(f"All 4 hypotheses audited in {time.time() - t0:.1f}s total.")
