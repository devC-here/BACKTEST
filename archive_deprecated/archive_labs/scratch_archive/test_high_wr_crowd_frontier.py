#!/usr/bin/env python3
"""
================================================================================
  HIGH WIN-RATE FRONTIER LAB: HARNESSING "WISDOM OF THE CROWD" (WR > 50%)
================================================================================
Evaluating 5 candidate models specifically optimized for:
  1. Win Rate strictly > 50% (52% - 75%)
  2. Profit Factor > 1.35
  3. Low Maximum Drawdown (< 50 pips)
  4. Robust statistical sample across 52 weeks (~26M Dukascopy ticks)
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
from data_engineering_suite.bars import build_time_bars

PIP_SIZE = 0.0001


# ==============================================================================
# 1. DISC43_Calibrated_HighWR (H1 Structural Level + M5 Rejection Wick)
# ==============================================================================
class DISC43CalibratedHighWR(BaseStrategy):
    """
    Tests DISC43 with calibrated TP levels (8.0p, 8.5p) to cross the 50% WR mark.
    """
    def __init__(
        self,
        sl_pips: float = 6.0,
        tp_pips: float = 8.5,
        min_wick: float = 0.55,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.min_wick = float(min_wick)
        self.max_spread = 0.55
        self.name = f"DISC43_Calibrated_{self.sl_pips}SL_{self.tp_pips}TP_Wick{int(self.min_wick*100)}"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        m5 = build_time_bars(ticks, timeframe_min=5)
        if len(m5) < 50:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid, 'open': mid}, index=ts)
        h1 = base.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(h1) < 15:
            return signals, sl_pips_arr, tp_pips_arr

        h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
        h1['macro'] = np.where(h1['close'] > h1['ema50'], 1, -1)
        h1['roll_h'] = h1['high'].rolling(12).max().shift(1).bfill()
        h1['roll_l'] = h1['low'].rolling(12).min().shift(1).bfill()

        h1_macro = h1['macro'].reindex(m5.index, method='ffill').fillna(0).values
        h1_supp = h1['roll_l'].reindex(m5.index, method='ffill').values
        h1_res = h1['roll_h'].reindex(m5.index, method='ffill').values

        c = m5['close'].values
        h = m5['high'].values
        l = m5['low'].values
        o = m5['open'].values
        spread = m5['spread_pips'].values
        hours = m5.index.hour.values
        dates = m5.index.date

        time_ms = m5.index.astype('datetime64[ms]').view('int64')
        dur_ms = 5 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_idx = -999

        for i in range(15, len(m5)):
            hr = hours[i]
            if not (7 <= hr <= 16):
                continue
            if spread[i] > self.max_spread:
                continue

            if (i - last_trade_idx) < 12:
                continue

            m = h1_macro[i]
            bar_rng = (h[i] - l[i]) / PIP_SIZE
            if bar_rng < 2.5:
                continue

            sig = 0
            if m > 0:
                dist_supp = abs(l[i] - h1_supp[i]) / PIP_SIZE
                lower_wick = (min(o[i], c[i]) - l[i]) / PIP_SIZE
                if dist_supp <= 4.0 and (lower_wick / bar_rng) >= self.min_wick and c[i] > o[i]:
                    sig = SIGNAL_LONG

            elif m < 0:
                dist_res = abs(h1_res[i] - h[i]) / PIP_SIZE
                upper_wick = (h[i] - max(o[i], c[i])) / PIP_SIZE
                if dist_res <= 4.0 and (upper_wick / bar_rng) >= self.min_wick and c[i] < o[i]:
                    sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = self.sl_pips
                    tp_pips_arr[idx] = self.tp_pips
                    last_trade_idx = i

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# 2. WOTC_Triple_Timeframe_Supermajority_M15 (H4 + H1 + M15 All Aligned)
# ==============================================================================
class WOTCTripleTimeframeSupermajority(BaseStrategy):
    """
    Wisdom of 3 Timeframes:
    - H4 Trend: Close > H4 50-EMA
    - H1 Trend: Close > H1 20-EMA + RSI > 50
    - M15 Trigger: Pullback to M15 20-EMA with rejection wick >= 45% closing in trend direction.
    - Sized at: 8.5p SL, 11.5p TP (1.35R).
    """
    def __init__(
        self,
        sl_pips: float = 8.5,
        tp_pips: float = 11.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.allowed_hours = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
        self.name = f"WOTC_Triple_TF_{self.sl_pips}SL_{self.tp_pips}TP"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = build_time_bars(ticks, timeframe_min=15)
        if len(m15) < 60:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid, 'open': mid}, index=ts)

        h1 = base.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        h4 = base.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()

        if len(h1) < 25 or len(h4) < 15:
            return signals, sl_pips_arr, tp_pips_arr

        # H4 Macro
        h4_ema50 = h4['close'].ewm(span=50, adjust=False).mean()
        h4_macro = np.where(h4['close'] > h4_ema50, 1, -1)
        h4_macro_m15 = pd.Series(h4_macro, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        # H1 Intraday
        h1_ema20 = h1['close'].ewm(span=20, adjust=False).mean()
        h1_delta = h1['close'].diff()
        h1_gain = h1_delta.where(h1_delta > 0, 0.0).rolling(14).mean()
        h1_loss = (-h1_delta.where(h1_delta < 0, 0.0)).rolling(14).mean()
        h1_rs = h1_gain / (h1_loss + 1e-9)
        h1_rsi = 100 - (100 / (1 + h1_rs))
        h1_bull = (h1['close'] > h1_ema20) & (h1_rsi > 52)
        h1_bear = (h1['close'] < h1_ema20) & (h1_rsi < 48)
        h1_state = np.where(h1_bull, 1, np.where(h1_bear, -1, 0))
        h1_state_m15 = pd.Series(h1_state, index=h1.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        # M15 Micro
        m15_ema20 = m15['close'].ewm(span=20, adjust=False).mean().values
        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        n_bars = len(m15)

        tick_ts = ticks['timestamp']
        time_ms = m15.index.astype(np.int64) // 10**6
        dur_ms = 15 * 60 * 1000
        hours = m15.index.hour

        traded_today = False
        last_day = -1

        for i in range(30, n_bars - 1):
            cur_day = m15.index[i].dayofyear
            if cur_day != last_day:
                last_day = cur_day
                traded_today = False

            if traded_today:
                continue

            if hours[i] not in self.allowed_hours:
                continue

            exec_time = time_ms[i] + dur_ms
            t_idx = int(np.searchsorted(tick_ts, exec_time))
            if t_idx >= n_ticks:
                break

            spread = (ticks['ask'][t_idx] - ticks['bid'][t_idx]) / PIP_SIZE
            if spread > 0.60:
                continue

            bar_range = max(h[i] - l[i], 1e-6)

            # Triple Bullish Alignment
            if h4_macro_m15[i] == 1 and h1_state_m15[i] == 1:
                if l[i] <= m15_ema20[i] + 1.5 * PIP_SIZE and c[i] > m15_ema20[i]:
                    lower_wick = min(o[i], c[i]) - l[i]
                    if (lower_wick / bar_range) >= 0.45 and c[i] > o[i]:
                        signals[t_idx] = SIGNAL_LONG
                        sl_pips_arr[t_idx] = self.sl_pips
                        tp_pips_arr[t_idx] = self.tp_pips
                        traded_today = True

            # Triple Bearish Alignment
            elif h4_macro_m15[i] == -1 and h1_state_m15[i] == -1:
                if h[i] >= m15_ema20[i] - 1.5 * PIP_SIZE and c[i] < m15_ema20[i]:
                    upper_wick = h[i] - max(o[i], c[i])
                    if (upper_wick / bar_range) >= 0.45 and c[i] < o[i]:
                        signals[t_idx] = SIGNAL_SHORT
                        sl_pips_arr[t_idx] = self.sl_pips
                        tp_pips_arr[t_idx] = self.tp_pips
                        traded_today = True

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# 3. WOTC_Asian_Sweep_Trap_HighWR (Contrarian Trap Fade)
# ==============================================================================
class WOTCAsianSweepTrapHighWR(BaseStrategy):
    """
    Contrarian Asian Trap Fade:
    - Sweep 1.5 - 5.5 pips of Asian High/Low during London Open (07:00 - 09:30 UTC).
    - Immediate rejection wick >= 50%.
    - Fixed tight target: 7.5p SL, 9.5p TP (1.27R).
    """
    def __init__(
        self,
        sl_pips: float = 7.5,
        tp_pips: float = 9.5,
        min_sweep: float = 1.5,
        max_sweep: float = 5.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.min_sweep = float(min_sweep)
        self.max_sweep = float(max_sweep)
        self.name = f"WOTC_Asian_Trap_{self.sl_pips}SL_{self.tp_pips}TP"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = build_time_bars(ticks, timeframe_min=15)
        if len(m15) < 40:
            return signals, sl_pips_arr, tp_pips_arr

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values

        tick_ts = ticks['timestamp']
        time_ms = m15.index.astype(np.int64) // 10**6
        dur_ms = 15 * 60 * 1000

        m15['day'] = m15.index.date
        m15['hour'] = m15.index.hour

        unique_days = m15['day'].unique()
        for day in unique_days:
            day_mask = m15['day'] == day
            asian_mask = day_mask & (m15['hour'] >= 0) & (m15['hour'] < 6)
            asian_bars = m15[asian_mask]

            if len(asian_bars) < 16:
                continue

            asian_high = asian_bars['high'].max()
            asian_low = asian_bars['low'].min()
            asian_range = (asian_high - asian_low) / PIP_SIZE

            if asian_range < 12.0 or asian_range > 35.0:
                continue

            london_mask = day_mask & ((m15['hour'] >= 7) & (m15['hour'] <= 9))
            london_indices = np.where(london_mask.values)[0]
            traded = False

            for idx in london_indices:
                if traded:
                    break

                row_h = h[idx]
                row_l = l[idx]
                row_c = c[idx]
                row_o = o[idx]

                sweep_up = (row_h - asian_high) / PIP_SIZE
                if self.min_sweep <= sweep_up <= self.max_sweep and row_c < asian_high:
                    upper_wick = row_h - max(row_o, row_c)
                    bar_len = max(row_h - row_l, 1e-6)
                    if (upper_wick / bar_len) >= 0.48 and row_c < row_o:
                        exec_t = time_ms[idx] + dur_ms
                        t_idx = int(np.searchsorted(tick_ts, exec_t))
                        if t_idx < n_ticks:
                            spread = (ticks['ask'][t_idx] - ticks['bid'][t_idx]) / PIP_SIZE
                            if spread <= 0.60:
                                signals[t_idx] = SIGNAL_SHORT
                                sl_pips_arr[t_idx] = self.sl_pips
                                tp_pips_arr[t_idx] = self.tp_pips
                                traded = True

                sweep_dn = (asian_low - row_l) / PIP_SIZE
                if self.min_sweep <= sweep_dn <= self.max_sweep and row_c > asian_low:
                    lower_wick = min(row_o, row_c) - row_l
                    bar_len = max(row_h - row_l, 1e-6)
                    if (lower_wick / bar_len) >= 0.48 and row_c > row_o:
                        exec_t = time_ms[idx] + dur_ms
                        t_idx = int(np.searchsorted(tick_ts, exec_t))
                        if t_idx < n_ticks:
                            spread = (ticks['ask'][t_idx] - ticks['bid'][t_idx]) / PIP_SIZE
                            if spread <= 0.60:
                                signals[t_idx] = SIGNAL_LONG
                                sl_pips_arr[t_idx] = self.sl_pips
                                tp_pips_arr[t_idx] = self.tp_pips
                                traded = True

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# AUDIT RUNNER ACROSS 52 WEEKS
# ==============================================================================
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
    strat_inst = StratCls(**kwargs)
    if n == 0:
        return {'name': strat_inst.name, 'trades': 0, 'wr': 0, 'pips': 0, 'pf': 0, 'max_dd': 0, 'rr': 0}

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
        'name': strat_inst.name,
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
    print("  HIGH WIN-RATE FRONTIER LAB: 52-WEEK TICK SCREENING (~26M TICKS)")
    print("  CRITERIA: WIN RATE > 50%, PROFIT FACTOR > 1.35, LOW DRAWDOWN")
    print("=" * 80)

    candidates = [
        # DISC43 with 8.0p TP vs 8.5p TP vs 9.5p original
        (DISC43CalibratedHighWR, {'sl_pips': 6.0, 'tp_pips': 8.0, 'min_wick': 0.55}),
        (DISC43CalibratedHighWR, {'sl_pips': 6.0, 'tp_pips': 8.5, 'min_wick': 0.55}),
        (DISC43CalibratedHighWR, {'sl_pips': 6.0, 'tp_pips': 8.0, 'min_wick': 0.60}),
        # Triple Timeframe Supermajority
        (WOTCTripleTimeframeSupermajority, {'sl_pips': 8.5, 'tp_pips': 11.5}),
        # Asian Sweep Trap Fade
        (WOTCAsianSweepTrapHighWR, {'sl_pips': 7.5, 'tp_pips': 9.5}),
    ]

    for StratCls, kwargs in candidates:
        sub_t0 = time.time()
        res = screen_candidate(StratCls, **kwargs)
        sub_dt = time.time() - sub_t0
        print(f"Strategy: {res['name']}")
        print(f"  Trades:   {res['trades']} ({res['trades']/52:.1f}/wk)")
        print(f"  Win Rate: {res['wr']:.1f}%  <--- [WR TARGET: > 50%]")
        print(f"  PF:       {res['pf']:.2f}")
        print(f"  Real RR:  {res['rr']:.2f} : 1")
        print(f"  Net Pips: {res['pips']:+.1f} pips")
        print(f"  Max DD:   {res['max_dd']:.1f} pips")
        print(f"  Duration: {sub_dt:.1f}s")
        print("-" * 60)

    print(f"Screening complete in {time.time() - t0:.1f}s total.")
