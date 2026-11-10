#!/usr/bin/env python3
"""
================================================================================
  PORTFOLIO AUDIT: 4 IN-BETWEEN GAP CHAMPIONS COMBINED (52 WEEKS TICK AUDIT)
================================================================================
Evaluates the consolidated performance of the 4 new gap strategies running
in parallel across 52 weeks (~26M ticks) of EUR/USD data:
  1. London Lunch Trap Fade (10:30 - 12:30 UTC)
  2. European Early Bird ORB (05:00 - 07:00 UTC)
  3. Late US Afternoon Trend Continuation (16:00 - 20:00 UTC)
  4. Friday Weekly Extreme Mean-Reversion (13:00 - 18:00 UTC Fridays)
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


class UnifiedGapPortfolioStrategy(BaseStrategy):
    """
    Unified strategy encapsulating all 4 gap strategies into a single execution pass.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Unified_Gap_Portfolio"

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

        c = m15['close']
        h = m15['high']
        l = m15['low']

        # ATR 14
        tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
        m15['atr'] = tr.rolling(14).mean().bfill()
        m15['ema20'] = c.ewm(span=20, adjust=False).mean()

        # RSI 14
        delta = c.diff()
        gain = delta.clip(lower=0.0).ewm(alpha=1.0/14, adjust=False).mean()
        loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0/14, adjust=False).mean()
        rs = gain / np.maximum(loss, 1e-9)
        m15['rsi'] = 100.0 - (100.0 / (1.0 + rs))

        # ADX 14
        up_m = h.diff()
        dn_m = -l.diff()
        p_dm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
        m_dm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
        smooth_tr = tr.ewm(alpha=1.0/14, adjust=False).mean()
        p_di = (pd.Series(p_dm, index=m15.index).ewm(alpha=1.0/14, adjust=False).mean() / np.maximum(smooth_tr, 1e-9)) * 100
        m_di = (pd.Series(m_dm, index=m15.index).ewm(alpha=1.0/14, adjust=False).mean() / np.maximum(smooth_tr, 1e-9)) * 100
        dx = (abs(p_di - m_di) / np.maximum(p_di + m_di, 1e-9)) * 100
        m15['adx'] = dx.ewm(alpha=1.0/14, adjust=False).mean()

        # H1 Macro Trend
        h1 = base.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        h1['ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
        h1['macro'] = np.where(h1['close'] > h1['ema50'], 1, -1)
        m15['h1_macro'] = h1['macro'].reindex(m15.index, method='ffill').fillna(0)

        m15['date'] = m15.index.date
        m15['hour'] = m15.index.hour
        m15['minute'] = m15.index.minute
        m15['weekday'] = m15.index.weekday
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        dates = m15['date'].unique()

        # Precompute weekly extremes for Friday reversal
        week_high = m15['high'].max()
        week_low = m15['low'].min()
        week_range = (week_high - week_low) / PIP_SIZE

        for d in dates:
            day_mask = m15['date'] == d
            day_bars = m15[day_mask]
            if len(day_bars) < 10:
                continue

            day_indices = np.where(day_mask)[0]

            # 1. European Early Bird Range (00:00 - 04:30)
            asia_bars = day_bars[(day_bars['hour'] >= 0) & (day_bars['hour'] < 5)]
            if len(asia_bars) >= 6:
                asia_h = asia_bars['high'].max()
                asia_l = asia_bars['low'].min()
                asia_r = (asia_h - asia_l) / PIP_SIZE
            else:
                asia_h, asia_l, asia_r = 999.0, 0.0, 0.0

            # 2. London Morning Range (08:00 - 10:00)
            morn_bars = day_bars[(day_bars['hour'] >= 8) & (day_bars['hour'] < 10)]
            if len(morn_bars) >= 6:
                morn_h = morn_bars['high'].max()
                morn_l = morn_bars['low'].min()
                morn_r = (morn_h - morn_l) / PIP_SIZE
            else:
                morn_h, morn_l, morn_r = 999.0, 0.0, 0.0

            lunch_traded = False
            early_traded = False
            friday_traded = False

            for idx in day_indices:
                row = m15.iloc[idx]
                prev = m15.iloc[idx-1]
                hr = row['hour']
                mn = row['minute']
                wd = row['weekday']

                sig = 0
                risk_p = 0.0
                target_rr = 1.6

                # GAP 1: European Early Bird (05:00 - 06:45 UTC)
                if hr in (5, 6) and not early_traded and 8.0 <= asia_r <= 35.0:
                    if row['h1_macro'] > 0 and row['close'] > asia_h and prev['close'] <= asia_h and row['close'] > row['open']:
                        sl_p = min(row['low'], asia_h - 2.0 * PIP_SIZE)
                        risk = max(min((row['close'] - sl_p) / PIP_SIZE, 15.0), 6.0)
                        sig = SIGNAL_LONG
                        risk_p = risk
                        target_rr = 1.60
                        early_traded = True
                    elif row['h1_macro'] < 0 and row['close'] < asia_l and prev['close'] >= asia_l and row['close'] < row['open']:
                        sl_p = max(row['high'], asia_l + 2.0 * PIP_SIZE)
                        risk = max(min((sl_p - row['close']) / PIP_SIZE, 15.0), 6.0)
                        sig = SIGNAL_SHORT
                        risk_p = risk
                        target_rr = 1.60
                        early_traded = True

                # GAP 2: London Lunch Trap Fade (10:30 - 12:30 UTC)
                elif ((hr == 10 and mn >= 30) or hr == 11 or (hr == 12 and mn <= 30)) and not lunch_traded and 10.0 <= morn_r <= 45.0:
                    probe_up = (row['high'] - morn_h) / PIP_SIZE
                    probe_dn = (morn_l - row['low']) / PIP_SIZE
                    if 2.0 <= probe_up <= 10.0 and row['close'] < morn_h and row['close'] < row['open'] and row['rsi'] >= 52.0:
                        sl_p = row['high'] + 1.2 * PIP_SIZE
                        risk = max(min((sl_p - row['close']) / PIP_SIZE, 14.0), 5.0)
                        sig = SIGNAL_SHORT
                        risk_p = risk
                        target_rr = 1.60
                        lunch_traded = True
                    elif 2.0 <= probe_dn <= 10.0 and row['close'] > morn_l and row['close'] > row['open'] and row['rsi'] <= 48.0:
                        sl_p = row['low'] - 1.2 * PIP_SIZE
                        risk = max(min((row['close'] - sl_p) / PIP_SIZE, 14.0), 5.0)
                        sig = SIGNAL_LONG
                        risk_p = risk
                        target_rr = 1.60
                        lunch_traded = True

                # GAP 3: Late US Afternoon Trend Continuation (16:00 - 19:45 UTC, Mon-Thu)
                elif hr in (16, 17, 18, 19) and wd < 4 and row['adx'] >= 22.0:
                    if row['h1_macro'] > 0 and row['close'] > row['ema20'] and row['low'] <= row['ema20'] and row['close'] > row['open']:
                        sl_p = min(row['low'], prev['low']) - 1.2 * PIP_SIZE
                        risk = max(min((row['close'] - sl_p) / PIP_SIZE, 15.0), 6.0)
                        sig = SIGNAL_LONG
                        risk_p = risk
                        target_rr = 1.80
                    elif row['h1_macro'] < 0 and row['close'] < row['ema20'] and row['high'] >= row['ema20'] and row['close'] < row['open']:
                        sl_p = max(row['high'], prev['high']) + 1.2 * PIP_SIZE
                        risk = max(min((sl_p - row['close']) / PIP_SIZE, 15.0), 6.0)
                        sig = SIGNAL_SHORT
                        risk_p = risk
                        target_rr = 1.80

                # GAP 4: Friday Afternoon Weekly Extreme Reversal (13:00 - 18:00 UTC, Fridays only)
                elif wd == 4 and 13 <= hr <= 18 and not friday_traded and week_range >= 40.0:
                    if row['high'] >= week_high - 3.0 * PIP_SIZE and row['close'] < row['open']:
                        sl_p = max(row['high'], week_high) + 1.2 * PIP_SIZE
                        risk = max(min((sl_p - row['close']) / PIP_SIZE, 16.0), 6.0)
                        sig = SIGNAL_SHORT
                        risk_p = risk
                        target_rr = 1.60
                        friday_traded = True
                    elif row['low'] <= week_low + 3.0 * PIP_SIZE and row['close'] > row['open']:
                        sl_p = min(row['low'], week_low) - 1.2 * PIP_SIZE
                        risk = max(min((row['close'] - sl_p) / PIP_SIZE, 16.0), 6.0)
                        sig = SIGNAL_LONG
                        risk_p = risk
                        target_rr = 1.60
                        friday_traded = True

                if sig != 0:
                    exec_t = time_ms[idx] + dur_ms
                    t_idx = int(np.searchsorted(tick_ts, exec_t))
                    if t_idx < n_ticks and signals[t_idx] == 0:
                        signals[t_idx] = sig
                        sl_pips_arr[t_idx] = risk_p
                        tp_pips_arr[t_idx] = risk_p * target_rr

        return signals, sl_pips_arr, tp_pips_arr


def eval_week(w):
    ticks, _ = get_week_dataset(w)
    engine = BacktestEngine(pip_size=0.0001, use_slippage=True)
    strat = UnifiedGapPortfolioStrategy()
    trades = engine.run(ticks, strat)
    return trades


if __name__ == "__main__":
    t0 = time.time()
    weeks = list(range(1, 53))
    num_workers = min(8, os.cpu_count() or 4)

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(eval_week, weeks))

    all_trades = [t for r in results for t in r]
    dt = time.time() - t0

    n = len(all_trades)
    pips = sum(t['pips'] for t in all_trades)
    wins = [t for t in all_trades if t['pips'] > 0]
    losses = [t for t in all_trades if t['pips'] < 0]
    wr = len(wins) / n * 100.0 if n > 0 else 0
    gp = sum(t['pips'] for t in wins)
    gl = abs(sum(t['pips'] for t in losses))
    pf = gp / gl if gl > 0 else 99.0
    realized_rr = (gp / max(len(wins), 1)) / (gl / max(len(losses), 1)) if gl > 0 else 0.0

    equity = np.cumsum([t['pips'] for t in all_trades])
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = dd.max()

    print("\n" + "=" * 80)
    print(f"  UNIFIED 4-CHAMPION IN-BETWEEN GAP PORTFOLIO (52 WEEKS TICK AUDIT in {dt:.1f}s)")
    print("=" * 80)
    print(f"Total Trades Across Gaps: {n} trades (~{n/52:.1f}/week, ~{n/250:.2f}/day)")
    print(f"Portfolio Win Rate:       {wr:.1f}%")
    print(f"Realized Risk-to-Reward:  {realized_rr:.2f} : 1")
    print(f"Portfolio Profit Factor:  {pf:.2f}")
    print(f"Net Cumulative Return:    {pips:+.1f} pips")
    print(f"Maximum Total Drawdown:   {max_dd:.1f} pips")
    print("=" * 80 + "\n")
