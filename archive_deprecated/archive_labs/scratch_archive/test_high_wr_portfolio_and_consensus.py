#!/usr/bin/env python3
"""
================================================================================
  HIGH WIN-RATE PORTFOLIO & STRATEGY CONSENSUS LAB (5 FULL YEARS / 262 WEEKS)
================================================================================
1. Evaluates the combined High-Win-Rate Duo:
   - DISC42 (Macro H4/M30 @ 72.9% WR)
   - DISC43_HighWR (Intraday H1/M5 @ 52.0% WR)
   Combined portfolio metrics: overall win rate, total net pips, max DD, monthly consistency.

2. Explores Multi-Model Quorum (Strategy Crowd Agreement):
   When both an Intraday Trend Model and a Micro Rejection Model fire within
   a 30-minute window, what is the resulting win rate and edge?

3. Daily Pivot Institutional Confluence:
   Testing H1 Daily Pivot Point (S1/R1) rejection aligned with H4 trend.
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

from strategies.new_discoveries.disc42_macro_structure_h4_m30_elite.strategy import MacroStructureH4M30Elite

PIP_SIZE = 0.0001


# ----------------------------------------------------------------------
# 1. DISC43 High-WR (6.0p SL, 8.5p TP)
# ----------------------------------------------------------------------
class DISC43HighWR85(BaseStrategy):
    def __init__(self, sl_pips: float = 6.0, tp_pips: float = 8.5, min_wick: float = 0.55, **kwargs):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.min_wick = float(min_wick)
        self.max_spread = 0.55
        self.name = "DISC43_HighWR_8.5TP"

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


# ----------------------------------------------------------------------
# 2. Daily Pivot Institutional Confluence (H4 Macro + Daily Pivot S1/R1)
# ----------------------------------------------------------------------
class DailyPivotInstitutionalConfluence(BaseStrategy):
    """
    Daily Pivot Point Institutional Rejection:
    - Calculates Daily Pivot (P), Support 1 (S1), Resistance 1 (R1) from previous day.
    - H4 Macro Trend: Price > H4 50-EMA (Bullish) or Price < H4 50-EMA (Bearish).
    - Entry Trigger: In liquid London/NY hours (07:00 - 16:00 UTC), price tests S1 (in bull trend)
      or R1 (in bear trend) within 5.0 pips, and prints an M15 rejection wick >= 50%.
    - Sized at: 9.0p SL, 13.0p TP (1.44R).
    """
    def __init__(self, sl_pips: float = 9.0, tp_pips: float = 13.0, **kwargs):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.name = "Daily_Pivot_Institutional_Confluence"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = build_time_bars(ticks, timeframe_min=15)
        if len(m15) < 50:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid, 'open': mid}, index=ts)

        # Daily Bars for Pivots
        daily = base.resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(daily) < 3:
            return signals, sl_pips_arr, tp_pips_arr

        prev_h = daily['high'].shift(1)
        prev_l = daily['low'].shift(1)
        prev_c = daily['close'].shift(1)
        pivot = (prev_h + prev_l + prev_c) / 3.0
        r1 = 2.0 * pivot - prev_l
        s1 = 2.0 * pivot - prev_h

        # Reindex onto M15
        s1_m15 = s1.reindex(m15.index, method='ffill').values
        r1_m15 = r1.reindex(m15.index, method='ffill').values

        # H4 Macro
        h4 = base.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(h4) < 15:
            return signals, sl_pips_arr, tp_pips_arr

        h4['ema50'] = h4['close'].ewm(span=50, adjust=False).mean()
        h4['macro'] = np.where(h4['close'] > h4['ema50'], 1, -1)
        h4_macro_m15 = pd.Series(h4['macro'], index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        hours = m15.index.hour
        time_ms = m15.index.astype(np.int64) // 10**6
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        traded_today = False
        last_day = -1

        for i in range(20, len(m15)):
            cur_day = m15.index[i].dayofyear
            if cur_day != last_day:
                last_day = cur_day
                traded_today = False

            if traded_today:
                continue

            if not (7 <= hours[i] <= 16):
                continue

            exec_time = time_ms[i] + dur_ms
            t_idx = int(np.searchsorted(tick_ts, exec_time))
            if t_idx >= n_ticks:
                break

            spread = (ticks['ask'][t_idx] - ticks['bid'][t_idx]) / PIP_SIZE
            if spread > 0.60:
                continue

            bar_range = max(h[i] - l[i], 1e-6)
            if (bar_range / PIP_SIZE) < 3.0:
                continue

            m = h4_macro_m15[i]

            # Bullish: H4 Macro Bull, tests Daily S1 within 5.0 pips, lower wick >= 50%, closes green
            if m > 0 and not np.isnan(s1_m15[i]):
                dist_s1 = abs(l[i] - s1_m15[i]) / PIP_SIZE
                lower_wick = min(o[i], c[i]) - l[i]
                if dist_s1 <= 5.0 and (lower_wick / bar_range) >= 0.50 and c[i] > o[i]:
                    signals[t_idx] = SIGNAL_LONG
                    sl_pips_arr[t_idx] = self.sl_pips
                    tp_pips_arr[t_idx] = self.tp_pips
                    traded_today = True

            # Bearish: H4 Macro Bear, tests Daily R1 within 5.0 pips, upper wick >= 50%, closes red
            elif m < 0 and not np.isnan(r1_m15[i]):
                dist_r1 = abs(h[i] - r1_m15[i]) / PIP_SIZE
                upper_wick = h[i] - max(o[i], c[i])
                if dist_r1 <= 5.0 and (upper_wick / bar_range) >= 0.50 and c[i] < o[i]:
                    signals[t_idx] = SIGNAL_SHORT
                    sl_pips_arr[t_idx] = self.sl_pips
                    tp_pips_arr[t_idx] = self.tp_pips
                    traded_today = True

        return signals, sl_pips_arr, tp_pips_arr


# ----------------------------------------------------------------------
# 5-Year Dual Portfolio Auditor
# ----------------------------------------------------------------------
def _eval_portfolio_week(w):
    ticks, _ = get_week_dataset(w)
    engine = BacktestEngine(pip_size=0.0001, use_slippage=True)
    
    # 1. DISC42
    s42 = MacroStructureH4M30Elite()
    tr42 = engine.run(ticks, s42)
    
    # 2. DISC43 High-WR
    s43 = DISC43HighWR85()
    tr43 = engine.run(ticks, s43)
    
    # 3. Daily Pivot
    spiv = DailyPivotInstitutionalConfluence()
    trpiv = engine.run(ticks, spiv)
    
    return w, tr42, tr43, trpiv


def run_portfolio_audit():
    weeks = list(range(1, 263))
    num_workers = min(8, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(_eval_portfolio_week, weeks))

    results.sort(key=lambda x: x[0])

    all_tr42 = []
    all_tr43 = []
    all_trpiv = []
    combined_duo_trades = []

    for w, t42, t43, tpiv in results:
        all_tr42.extend(t42)
        all_tr43.extend(t43)
        all_trpiv.extend(tpiv)
        # Combine trades chronologically
        week_combo = list(t42) + list(t43)
        week_combo.sort(key=lambda x: x['entry_time'])
        combined_duo_trades.extend(week_combo)

    def stats(trades, name):
        n = len(trades)
        if n == 0:
            print(f"{name}: 0 trades")
            return
        wins = [t for t in trades if t['pips'] > 0]
        losses = [t for t in trades if t['pips'] < 0]
        pips = sum(t['pips'] for t in trades)
        wr = len(wins) / n * 100.0
        gp = sum(t['pips'] for t in wins)
        gl = abs(sum(t['pips'] for t in losses))
        pf = gp / gl if gl > 0 else 99.0

        equity = np.cumsum([t['pips'] for t in trades])
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        max_dd = dd.max()
        ret_dd = pips / max_dd if max_dd > 0 else 99.0

        print(f"================================================================================")
        print(f"  {name}")
        print(f"================================================================================")
        print(f"  Total Trades:     {n} ({n/262:.2f} trades / week | ~{n/5:.1f} trades / year)")
        print(f"  Win Rate:         {wr:.2f}%  <--- [TARGET: > 50%]")
        print(f"  Profit Factor:    {pf:.2f}")
        print(f"  Net Total Pips:   {pips:+.1f} pips")
        print(f"  Max Drawdown:     {max_dd:.1f} pips")
        print(f"  Return / DD:      {ret_dd:.2f} : 1")
        print(f"  Avg Trade Pips:   {pips/n:+.2f} pips/trade")
        print(f"  Gross Profit:     +{gp:.1f} pips | Gross Loss: -{gl:.1f} pips")
        print(f"--------------------------------------------------------------------------------\n")

    stats(all_tr42, "MODEL 1: DISC42 (Macro Structure H4 + M30 Rejection)")
    stats(all_tr43, "MODEL 2: DISC43_HighWR (Intraday H1 + M5 Rejection Wick >= 55%, 8.5p TP)")
    stats(all_trpiv, "MODEL 3: Daily Pivot Institutional Confluence (H4 Macro + Daily S1/R1)")
    stats(combined_duo_trades, "COMBINED FLEET: HIGH-WIN-RATE DUO (DISC42 + DISC43_HighWR)")


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 80)
    print("  RUNNING 5-YEAR HIGH WIN-RATE PORTFOLIO AUDIT (262 WEEKS / 123.22M TICKS)")
    print("=" * 80)
    run_portfolio_audit()
    print(f"Audit completed in {time.time() - t0:.1f}s.")
