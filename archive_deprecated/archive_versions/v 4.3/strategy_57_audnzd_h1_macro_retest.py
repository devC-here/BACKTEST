"""
Strategy Slot 57: AUDNZD H1 Macro Intraday Retest Elite
=============================================================================
Symbol: AUDNZD | Pip Size: 0.0001
Stop Loss: 10.0 pips | Take Profit: 20.0 pips | Lookback H1: 24 bars
Distance: 6.5 pips | Min Wick: 40%
Active Window: Sydney / Asian + London Sessions (21:00 - 16:00 UTC)
Spread Limit: 2.8 pips
Performance: In-Sample +12.0% (+120.1p, 15 trades, 60.0% WR)
Zero-Bias Architecture Standards:
- Zero .bfill(): Strict historical warmup.
- Strict Causal Indexing: All H1 indicators shifted by 1 (.shift(1)).
- Bar-Close Execution: Filled strictly on next tick at or after candle close.
"""

import numpy as np
import pandas as pd
from typing import Tuple

def _build_bars_internal(ticks: np.ndarray, timeframe_min: int = 15, eff_pip_size: float = 0.0001) -> pd.DataFrame:
    ts = pd.to_datetime(ticks['timestamp'], unit='ms')
    bid = ticks['bid']
    ask = ticks['ask']
    spread = (ask - bid) / eff_pip_size
    mid = (bid + ask) * 0.5
    df = pd.DataFrame({'mid': mid, 'spread': spread, 'vol': np.ones(len(ticks), dtype=np.int32)}, index=ts)
    rule = f'{timeframe_min}min'
    bars = df.resample(rule).agg({
        'mid': ['first', 'max', 'min', 'last'],
        'spread': 'mean',
        'vol': 'count'
    }).dropna()
    bars.columns = ['open', 'high', 'low', 'close', 'spread_pips', 'vol']
    bars['volume'] = bars['vol']
    bars['range_pips'] = (bars['high'] - bars['low']) / eff_pip_size
    bars['body_pips'] = abs(bars['close'] - bars['open']) / eff_pip_size
    return bars

build_time_bars = _build_bars_internal

SIGNAL_NONE: int = 0
SIGNAL_LONG: int = 1
SIGNAL_SHORT: int = -1

class BaseStrategy:
    def __init__(self, name: str = "", symbol: str = "AUDNZD", sl_pips: float = 10.0, tp_pips: float = 20.0, pip_size: float = 0.0001, spread_limit: float = 2.8, **kwargs):
        self.name = name
        self.symbol = symbol
        self.asset = symbol
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.spread_limit = float(spread_limit)

class Strategy_57_audnzd_h1_macro_retest(BaseStrategy):
    def __init__(
        self,
        asset: str = "AUDNZD",
        sl_pips: float = 10.0,
        tp_pips: float = 20.0,
        lookback_h1: int = 24,
        dist_pips: float = 6.5,
        min_wick: float = 0.40,
        max_spread: float = 2.8,
        pip_size: float = 0.0001
    ):
        super().__init__(
            name="Strategy_57_audnzd_h1_macro_retest",
            symbol=asset,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            pip_size=pip_size,
            spread_limit=max_spread
        )
        self.lookback_h1 = lookback_h1
        self.dist_pips = dist_pips
        self.min_wick = min_wick
        self.max_spread = max_spread

    def generate_signals(self, ticks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if n < 500: return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, eff_pip_size=self.pip_size)
        if len(m15) < 30: return signals, sl_arr, tp_arr

        h1 = m15.resample('1h').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(h1) < self.lookback_h1 + 5: return signals, sl_arr, tp_arr

        h1_ema20 = h1['close'].ewm(span=20, adjust=False).mean()
        h1_bias = pd.Series(np.where(h1['close'] > h1_ema20, 1, -1), index=h1.index)
        bias_m15 = h1_bias.shift(1).reindex(m15.index, method='ffill').fillna(0).values

        roll_h = h1['high'].rolling(self.lookback_h1).max().shift(1).reindex(m15.index, method='ffill').values
        roll_l = h1['low'].rolling(self.lookback_h1).min().shift(1).reindex(m15.index, method='ffill').values

        hours = m15.index.hour.values
        dates = m15.index.date
        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_dt = None
        for i in range(15, len(m15)):
            hr = hours[i]
            if not (hr >= 21 or hr <= 16): continue
            dt = dates[i]
            if dt == last_dt or spread[i] > self.max_spread: continue
            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < 2.0: continue

            b = bias_m15[i]
            supp = roll_l[i]
            res = roll_h[i]
            if np.isnan(supp) or np.isnan(res): continue

            sig = 0
            if b > 0 and supp > 0:
                dist_supp = abs(l[i] - supp) / self.pip_size
                lw = min(o[i], c[i]) - l[i]
                rng_abs = h[i] - l[i]
                if dist_supp <= self.dist_pips and (lw / rng_abs) >= self.min_wick and c[i] > o[i]:
                    sig = SIGNAL_LONG

            if b < 0 and res > 0:
                dist_res = abs(res - h[i]) / self.pip_size
                uw = h[i] - max(o[i], c[i])
                rng_abs = h[i] - l[i]
                if dist_res <= self.dist_pips and (uw / rng_abs) >= self.min_wick and c[i] < o[i]:
                    sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_dt = dt

        return signals, sl_arr, tp_arr

Strategy = Strategy_57_audnzd_h1_macro_retest
