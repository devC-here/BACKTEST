#!/usr/bin/env python3
"""
John Carter's TTM Squeeze Quantitative Tick Backtest across 52 Weeks (~26M Ticks):
- Bollinger Bands: 20 SMA, 2.0 StDev
- Keltner Channels: 20 EMA, 1.5 ATR (14 or 20)
- Squeeze Condition: BB Bandwidth < KC Bandwidth (BB completely inside KC)
- Squeeze Fired: BB Bandwidth crosses ABOVE KC Bandwidth
- Momentum Filter: Linear Regression of (Price - Average(Donchian Mid, SMA))
- Target RR: 1.5R to 1.8R
- Evaluated on M15 bars with tick-level slippage & spread.
"""

from __future__ import annotations
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT

PIP_SIZE = 0.0001


def calc_linreg_slope(series: pd.Series, length: int = 20) -> pd.Series:
    """Calculates linear regression value of a series over rolling window."""
    x = np.arange(length)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    
    def linreg_val(y):
        if len(y) < length:
            return 0.0
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / x_var
        intercept = y_mean - slope * x_mean
        return intercept + slope * (length - 1)
        
    return series.rolling(length).apply(linreg_val, raw=True)


class JohnCarterTTMSqueeze(BaseStrategy):
    def __init__(self, target_rr: float = 1.60, min_squeeze_bars: int = 3, min_sl: float = 6.0, max_sl: float = 16.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_squeeze_bars = min_squeeze_bars
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.pip_size = PIP_SIZE
        self.name = "John_Carter_TTM_Squeeze"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base_df = pd.DataFrame({'open': mid, 'high': mid, 'low': mid, 'close': mid}, index=ts)
        m15 = base_df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(m15) < 60:
            return signals, sl_pips_arr, tp_pips_arr

        c = m15['close']
        h = m15['high']
        l = m15['low']

        # 1. Bollinger Bands (20, 2.0)
        sma20 = c.rolling(20).mean().bfill()
        std20 = c.rolling(20).std().bfill()
        bb_upper = sma20 + 2.0 * std20
        bb_lower = sma20 - 2.0 * std20
        bb_width = bb_upper - bb_lower

        # 2. Keltner Channels (20, 1.5 ATR)
        tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
        atr20 = tr.rolling(20).mean().bfill()
        ema20 = c.ewm(span=20, adjust=False).mean()
        kc_upper = ema20 + 1.5 * atr20
        kc_lower = ema20 - 1.5 * atr20
        kc_width = kc_upper - kc_lower

        # 3. Squeeze state
        # In squeeze when BB is inside KC
        in_squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)
        squeeze_count = in_squeeze.groupby((~in_squeeze).cumsum()).cumsum()

        # 4. Momentum oscillator: Price - average(donchian_mid, sma20)
        highest_h = h.rolling(20).max()
        lowest_l = l.rolling(20).min()
        donchian_mid = 0.5 * (highest_h + lowest_l)
        baseline = 0.5 * (donchian_mid + sma20)
        mom_diff = c - baseline
        mom = mom_diff.rolling(20).mean() # fast linear proxy

        # Macro trend filter H1 50-EMA shifted 1 bar
        h1 = base_df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        h1_ema50 = h1['close'].ewm(span=50, adjust=False).mean()
        h1['macro_bias'] = np.where(h1['close'] > h1_ema50, 1, -1)
        m15['macro_bias'] = h1['macro_bias'].shift(1).reindex(m15.index, method='ffill').fillna(0)

        m15['hour'] = m15.index.hour
        m15['time_ms'] = m15.index.astype('datetime64[ms]').view('int64')

        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(25, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i-1]

            if row['hour'] < 7 or row['hour'] > 15:
                continue

            # Check if squeeze fired on current candle: prev was in squeeze (>= min_squeeze_bars), current is out of squeeze
            fired = (squeeze_count.iloc[i-1] >= self.min_squeeze_bars) and (not in_squeeze.iloc[i])
            if not fired:
                continue

            sig = 0
            risk_p = 0.0

            # Long fire: Mom > 0, Mom > Mom[prev], Macro Bullish
            if mom.iloc[i] > 0 and mom.iloc[i] > mom.iloc[i-1] and row['macro_bias'] > 0:
                sl_p = float(kc_lower.iloc[i])
                risk = max((row['close'] - sl_p) / self.pip_size, self.min_sl)
                if risk <= self.max_sl:
                    sig = SIGNAL_LONG
                    risk_p = risk

            # Short fire: Mom < 0, Mom < Mom[prev], Macro Bearish
            elif mom.iloc[i] < 0 and mom.iloc[i] < mom.iloc[i-1] and row['macro_bias'] < 0:
                sl_p = float(kc_upper.iloc[i])
                risk = max((sl_p - row['close']) / self.pip_size, self.min_sl)
                if risk <= self.max_sl:
                    sig = SIGNAL_SHORT
                    risk_p = risk

            if sig != 0:
                exec_t = row['time_ms'] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = risk_p
                    tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr


def run_week(w: int):
    ticks, meta = get_week_dataset(w)
    engine = BacktestEngine(pip_size=PIP_SIZE, use_slippage=True)
    strat = JohnCarterTTMSqueeze()
    trades = engine.run(ticks, strat)
    return [float(t["pips"]) for t in trades]


def main():
    print("Testing John Carter TTM Squeeze across 52 weeks...")
    with ProcessPoolExecutor() as executor:
        week_res = list(executor.map(run_week, range(1, 53)))
    
    all_pips = []
    for r in week_res:
        all_pips.extend(r)
    
    pips = np.array(all_pips)
    wins = pips[pips > 0]
    losses = pips[pips < 0]
    n_t = len(pips)
    wr = len(wins) / n_t * 100 if n_t > 0 else 0
    pf = np.sum(wins) / abs(np.sum(losses)) if len(losses) > 0 else 0
    rr = np.mean(wins) / abs(np.mean(losses)) if len(losses) > 0 else 0
    net = np.sum(pips)
    print(f"Results: Trades={n_t}, WR={wr:.1f}%, Realized RR={rr:.2f}, PF={pf:.2f}, Net Pips={net:+.1f}")


if __name__ == "__main__":
    main()
