#!/usr/bin/env python3
"""
Prototype & Verification Test for AUDNZD Peer Architectures on Week 1.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars

PIP_SIZE = 0.0001
MAX_SPREAD = 2.8


# 1. Triple Screen M15 Impulse
class AUDNZDTripleScreenM15Impulse(BaseStrategy):
    def __init__(self, sl_pips=12.0, tp_pips=24.0, **kwargs):
        super().__init__(**kwargs)
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.name = f"TripleScreen_M15_SL{int(sl_pips)}_TP{int(tp_pips)}"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=PIP_SIZE)
        if len(m15) < 30:
            return signals, sl_arr, tp_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid}, index=ts)

        h1 = base.resample('1h').agg({'close': 'last'}).dropna()
        h4 = base.resample('4h').agg({'close': 'last'}).dropna()
        if len(h1) < 35 or len(h4) < 15:
            return signals, sl_arr, tp_arr

        h4['ema20'] = h4['close'].ewm(span=20, adjust=False).mean()
        h4_trend = np.where(h4['close'] > h4['ema20'], 1, -1)
        h4_bias = pd.Series(h4_trend, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        h1['ema12'] = h1['close'].ewm(span=12, adjust=False).mean()
        h1['ema26'] = h1['close'].ewm(span=26, adjust=False).mean()
        h1['macd'] = h1['ema12'] - h1['ema26']
        h1['signal'] = h1['macd'].ewm(span=9, adjust=False).mean()
        h1['hist'] = h1['macd'] - h1['signal']
        h1['hist_slope'] = h1['hist'].diff()
        h1_slope_trend = np.where(h1['hist_slope'] > 0, 1, -1)
        h1_bias = pd.Series(h1_slope_trend, index=h1.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        m15['ema13'] = m15['close'].ewm(span=13, adjust=False).mean()
        ema13 = m15['ema13'].values

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

        last_trade_date = None
        tol = 3.0 * PIP_SIZE

        for i in range(25, len(m15)):
            hr = hours[i]
            if not (hr >= 21 or hr <= 16):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > MAX_SPREAD:
                continue

            bias_h4 = h4_bias[i]
            bias_h1 = h1_bias[i]
            if bias_h4 != bias_h1 or bias_h4 == 0:
                continue

            bar_rng = (h[i] - l[i]) / PIP_SIZE
            body = abs(c[i] - o[i]) / PIP_SIZE
            if bar_rng < 3.0 or (body / bar_rng) < 0.45:
                continue

            e13 = ema13[i]
            sig = 0
            if bias_h4 > 0 and l[i] <= e13 + tol and c[i] > e13 and c[i] > o[i]:
                sig = SIGNAL_LONG
            elif bias_h4 < 0 and h[i] >= e13 - tol and c[i] < e13 and c[i] < o[i]:
                sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


if __name__ == "__main__":
    ticks, meta = get_week_dataset(1, raw_dir="data/AUDNZD")
    print(f"Loaded week 1: {len(ticks)} ticks")
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)
    m = AUDNZDTripleScreenM15Impulse(sl_pips=12.0, tp_pips=24.0)
    trades = engine.run(ticks, m)
    print(f"Trades executed: {len(trades)}")
    for t in trades:
        print(f"  {t['side']} Entry: {t['entry_price']:.5f} Exit: {t['exit_price']:.5f} Pips: {t['pips']:.2f} Reason: {t['reason']}")

