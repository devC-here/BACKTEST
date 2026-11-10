#!/usr/bin/env python3
"""
================================================================================
  DUAL 5-YEAR TICK AUDIT: MACRO SWING ENGINE vs INTRADAY PRECISION ENGINE
================================================================================
Audits both models across all 262 weeks (~123.22M real Dukascopy ticks):

  1. MODEL A: DISC42 - Macro Structure H4 Level + M30 Trigger Elite
     - HTF Level: H4 Support/Resistance & H4 20-EMA trend
     - LTF Trigger: M30 Rejection Wick (Wick >= 50%) at H4 Zone
     - Risk: SL = 10.0 pips | Target: TP = 15.0 pips (1.50R)
     - Hours: 07:00 - 16:30 UTC

  2. MODEL B: DISC43 - Intraday Precision H1 Level + M5 Rejection Elite
     - HTF Level: H1 Support/Resistance & H1 50-EMA trend
     - LTF Trigger: M5 Rejection Wick (Wick >= 55%) at H1 Zone
     - Risk: SL = 6.0 pips | Target: TP = 9.5 pips (1.58R)
     - Hours: 07:15 - 16:00 UTC
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
from tick_engine.core.metrics import compute_metrics
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars

PIP_SIZE = 0.0001


# ==============================================================================
# MODEL A: DISC42 - MACRO STRUCTURE H4 LEVEL + M30 TRIGGER ELITE
# ==============================================================================
class MacroStructureH4M30Elite(BaseStrategy):
    def __init__(
        self,
        sl_pips: float = 10.0,
        tp_pips: float = 15.0,
        max_spread: float = 0.60,
        pip_size: float = PIP_SIZE,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.max_spread = float(max_spread)
        self.pip_size = float(pip_size)
        self.name = "DISC42_Macro_Structure_H4_M30_Elite"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        m30 = build_time_bars(ticks, timeframe_min=30)
        if len(m30) < 30:
            return signals, sl_pips_arr, tp_pips_arr

        # Build H4 levels from tick data
        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid, 'open': mid}, index=ts)
        h4 = base.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(h4) < 10:
            return signals, sl_pips_arr, tp_pips_arr

        h4['ema20'] = h4['close'].ewm(span=20, adjust=False).mean()
        h4['macro'] = np.where(h4['close'] > h4['ema20'], 1, -1)
        h4['roll_h'] = h4['high'].rolling(10).max().shift(1).bfill()
        h4['roll_l'] = h4['low'].rolling(10).min().shift(1).bfill()

        # Reindex onto M30
        h4_macro = h4['macro'].reindex(m30.index, method='ffill').fillna(0).values
        h4_supp = h4['roll_l'].reindex(m30.index, method='ffill').values
        h4_res = h4['roll_h'].reindex(m30.index, method='ffill').values

        c = m30['close'].values
        h = m30['high'].values
        l = m30['low'].values
        o = m30['open'].values
        spread = m30['spread_pips'].values
        hours = m30.index.hour.values
        dates = m30.index.date

        time_ms = m30.index.astype('datetime64[ms]').view('int64')
        dur_ms = 30 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None

        for i in range(15, len(m30)):
            hr = hours[i]
            if not (7 <= hr <= 16):
                continue
            if spread[i] > self.max_spread:
                continue

            dt = dates[i]
            if dt == last_trade_date:
                continue

            m = h4_macro[i]
            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < 4.0:
                continue

            sig = 0
            # Bullish: Macro Up, M30 tests H4 Support within 5 pips, prints lower rejection wick >= 50%
            if m > 0:
                dist_supp = (l[i] - h4_supp[i]) / self.pip_size
                lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
                if abs(dist_supp) <= 6.0 and (lower_wick / bar_rng) >= 0.50 and c[i] > o[i]:
                    sig = SIGNAL_LONG

            # Bearish: Macro Down, M30 tests H4 Resistance within 5 pips, prints upper rejection wick >= 50%
            elif m < 0:
                dist_res = (h4_res[i] - h[i]) / self.pip_size
                upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size
                if abs(dist_res) <= 6.0 and (upper_wick / bar_rng) >= 0.50 and c[i] < o[i]:
                    sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = self.sl_pips
                    tp_pips_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_pips_arr, tp_pips_arr


# ==============================================================================
# MODEL B: DISC43 - INTRADAY PRECISION H1 LEVEL + M5 REJECTION ELITE
# ==============================================================================
class IntradayPrecisionH1M5Elite(BaseStrategy):
    def __init__(
        self,
        sl_pips: float = 6.0,
        tp_pips: float = 9.5,
        max_spread: float = 0.55,
        pip_size: float = PIP_SIZE,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.max_spread = float(max_spread)
        self.pip_size = float(pip_size)
        self.name = "DISC43_Intraday_Precision_H1_M5_Elite"

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

        # Build H1 levels
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
        minutes = m5.index.minute.values
        dates = m5.index.date

        time_ms = m5.index.astype('datetime64[ms]').view('int64')
        dur_ms = 5 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_bar = -999

        for i in range(25, len(m5)):
            hr = hours[i]
            if not ((7 <= hr <= 10) or (12 <= hr <= 15)):
                continue
            if spread[i] > self.max_spread:
                continue

            # Limit to 1 trade every 12 bars (1 hour cooldown)
            if i - last_trade_bar < 12:
                continue

            m = h1_macro[i]
            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < 2.5:
                continue

            sig = 0
            # Bullish M5 rejection at H1 support
            if m > 0:
                dist_supp = (l[i] - h1_supp[i]) / self.pip_size
                lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
                if abs(dist_supp) <= 4.0 and (lower_wick / bar_rng) >= 0.55 and c[i] > o[i]:
                    sig = SIGNAL_LONG

            # Bearish M5 rejection at H1 resistance
            elif m < 0:
                dist_res = (h1_res[i] - h[i]) / self.pip_size
                upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size
                if abs(dist_res) <= 4.0 and (upper_wick / bar_rng) >= 0.55 and c[i] < o[i]:
                    sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = self.sl_pips
                    tp_pips_arr[idx] = self.tp_pips
                    last_trade_bar = i

        return signals, sl_pips_arr, tp_pips_arr


def _test_week_model(args: tuple):
    w, model_type = args
    ticks, _ = get_week_dataset(w)
    if model_type == 'A':
        strat = MacroStructureH4M30Elite()
    else:
        strat = IntradayPrecisionH1M5Elite()

    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)
    trades = engine.run(ticks, strat)
    return w, trades


def run_audit_for_model(model_type: str, model_name: str):
    print("=" * 105)
    print(f"  5-YEAR TICK AUDIT: {model_name} (262 WEEKS / ~123.22M TICKS)  ".center(105))
    print("=" * 105)

    weeks = list(range(1, 263))
    t0 = time.time()
    task_args = [(w, model_type) for w in weeks]
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        results = list(ex.map(_test_week_model, task_args))

    dt = time.time() - t0
    print(f"[+] All 262 weeks simulated in {dt:.1f}s!\n")

    results.sort(key=lambda x: x[0])

    train_trades = [r[1] for r in results if r[0] <= 156 and len(r[1]) > 0]
    test_trades = [r[1] for r in results if r[0] > 156 and len(r[1]) > 0]
    all_trades = [r[1] for r in results if len(r[1]) > 0]

    tr_train = np.concatenate(train_trades) if train_trades else np.empty(0, dtype=TRADE_DTYPE)
    tr_test = np.concatenate(test_trades) if test_trades else np.empty(0, dtype=TRADE_DTYPE)
    tr_all = np.concatenate(all_trades) if all_trades else np.empty(0, dtype=TRADE_DTYPE)

    m_train = compute_metrics(tr_train)
    m_test = compute_metrics(tr_test)
    m_all = compute_metrics(tr_all)

    print("-" * 105)
    print(f"{'Metric':<25} | {'3Y Train (W1-156)':<22} | {'2Y Test (W157-262)':<22} | {'5Y Total (W1-262)':<22}")
    print("-" * 105)
    print(f"{'Total Trades':<25} | {m_train['total_trades']:<22} | {m_test['total_trades']:<22} | {m_all['total_trades']:<22}")
    print(f"{'Trades per Week':<25} | {m_train['total_trades']/156:<22.2f} | {m_test['total_trades']/106:<22.2f} | {m_all['total_trades']/262:<22.2f}")
    print(f"{'Win Rate (%)':<25} | {m_train['win_rate_pct']:<21.1f}% | {m_test['win_rate_pct']:<21.1f}% | {m_all['win_rate_pct']:<21.1f}%")
    print(f"{'Profit Factor':<25} | {m_train['profit_factor']:<22.2f} | {m_test['profit_factor']:<22.2f} | {m_all['profit_factor']:<22.2f}")
    print(f"{'Net Cumulative Pips':<25} | {m_train['net_pips']:<+21.1f}p | {m_test['net_pips']:<+21.1f}p | {m_all['net_pips']:<+21.1f}p")
    print(f"{'Max Drawdown (p)':<25} | {m_train['max_drawdown_pips']:<21.1f}p | {m_test['max_drawdown_pips']:<21.1f}p | {m_all['max_drawdown_pips']:<21.1f}p")
    print(f"{'Return / DD Ratio':<25} | {m_train['net_pips']/max(m_train['max_drawdown_pips'], 1e-6):<22.2f} | {m_test['net_pips']/max(m_test['max_drawdown_pips'], 1e-6):<22.2f} | {m_all['net_pips']/max(m_all['max_drawdown_pips'], 1e-6):<22.2f}")
    print("-" * 105 + "\n")


def run():
    run_audit_for_model('A', "MODEL A: DISC42 - MACRO STRUCTURE H4 LEVEL + M30 TRIGGER ELITE")
    run_audit_for_model('B', "MODEL B: DISC43 - INTRADAY PRECISION H1 LEVEL + M5 REJECTION ELITE")


if __name__ == '__main__':
    run()
