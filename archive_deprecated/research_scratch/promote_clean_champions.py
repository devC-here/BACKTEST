import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset, get_complete_split_weeks
from tick_engine.core.config import PIP_SIZES, get_friction_pips, WARMUP_TICKS
from tick_engine.reporting.run_manifest import create_certified_run_manifest, compute_statistical_metrics
from data_engineering_suite.bars import build_time_bars

STRATEGY_TEMPLATE = '''#!/usr/bin/env python3
"""
================================================================================
  {STRAT_ID}: {SYMBOL} Macro Structure Retest Elite ({RR:.1f}R Target)
================================================================================
Discovered and certified across 5.75 continuous years (300 continuous weeks)
on authentic {SYMBOL} Dukascopy tick data.

Quantitative Architecture:
  - Higher Timeframe Anchor (H4): Defines structural multi-day support and resistance
    via {LOOKBACK}-period Donchian channel boundaries aligned with H4 20-EMA trend bias.
  - Intermediate Timeframe (M15): Detects institutional absorption via a rejection pin
    wick (>= {WICK_PCT}% of candle range) testing within {DIST} pips of the H4 structural level.
  - Multi-Session Liquidity: Exploits session mode {SESSION_MODE}.
  - Risk Setup: Asymmetric Risk-to-Reward (SL {SL} pips | TP {TP} pips | Max Spread {MAX_SPREAD} pips).
  - Frequency: Max 1 trade per calendar day.
================================================================================
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars


class {CLASS_NAME}(BaseStrategy):
    """
    {STRAT_ID}: {SYMBOL} Macro Structure Retest Elite.
    """
    def __init__(
        self,
        asset: str = "{SYMBOL}",
        sl_pips: float = {SL},
        tp_pips: float = {TP},
        lookback_h4: int = {LOOKBACK},
        dist_pips: float = {DIST},
        min_wick: float = {WICK},
        max_spread: float = {MAX_SPREAD},
        pip_size: float = {PIP_SIZE},
        **kwargs
    ):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.lookback_h4 = int(lookback_h4)
        self.dist_pips = float(dist_pips)
        self.min_wick = float(min_wick)
        self.max_spread = float(max_spread)
        self.pip_size = float(pip_size)
        self.session_mode = {SESSION_MODE}
        self.name = "{CLASS_NAME}"

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 30:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid}, index=ts)

        h4 = base.resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(h4) < max(self.lookback_h4 + 5, 15):
            return signals, sl_pips_arr, tp_pips_arr

        h4['ema20'] = h4['close'].ewm(span=20, adjust=False).mean()
        h4['bias'] = np.where(h4['close'] > h4['ema20'], 1, -1)
        h4['roll_h'] = h4['high'].rolling(self.lookback_h4).max().shift(1).bfill()
        h4['roll_l'] = h4['low'].rolling(self.lookback_h4).min().shift(1).bfill()

        bias_m15 = h4['bias'].reindex(m15.index, method='ffill').fillna(0).values
        supp_m15 = h4['roll_l'].reindex(m15.index, method='ffill').values
        res_m15 = h4['roll_h'].reindex(m15.index, method='ffill').values

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

        for i in range(15, len(m15)):
            hr = hours[i]
            if self.session_mode == 1:
                if not (hr >= 21 or hr <= 16):
                    continue
            elif self.session_mode == 2:
                if not (6 <= hr <= 18):
                    continue

            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < 2.5:
                continue

            b = bias_m15[i]
            sig = 0

            # Bullish Retest of H4 Support
            if b > 0 and supp_m15[i] > 0:
                dist_supp = abs(l[i] - supp_m15[i]) / self.pip_size
                lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
                if dist_supp <= self.dist_pips and (lower_wick / bar_rng) >= self.min_wick and c[i] > o[i]:
                    sig = SIGNAL_LONG

            # Bearish Retest of H4 Resistance
            elif b < 0 and res_m15[i] > 0:
                dist_res = abs(res_m15[i] - h[i]) / self.pip_size
                upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size
                if dist_res <= self.dist_pips and (upper_wick / bar_rng) >= self.min_wick and c[i] < o[i]:
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
'''

def calc_overlap(trades_a: np.ndarray, trades_b: np.ndarray, window_ms: int = 15 * 60 * 1000) -> float:
    if len(trades_a) == 0 or len(trades_b) == 0:
        return 0.0
    matches = 0
    ts_b_list = list(zip(trades_b['entry_time'], trades_b['direction']))
    b_used = set()
    for tr_a in trades_a:
        ts_a = tr_a['entry_time']
        dir_a = tr_a['direction']
        for idx_b, (ts_b, dir_b) in enumerate(ts_b_list):
            if idx_b in b_used:
                continue
            if dir_a == dir_b and abs(ts_a - ts_b) <= window_ms:
                matches += 1
                b_used.add(idx_b)
                break
    return matches / min(len(trades_a), len(trades_b))
