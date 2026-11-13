"""
Strategy Slot 26: GBPCHF Asian Sweep Fade Elite
=============================================================================
Asset: GBPCHF
Timeframe: M15 execution, Asian Range (00:00 - 06:45 UTC) baseline
Architecture: Liquidity Sweep Reversal (Turtle Soup Breakout Fade)
Strict Zero-Lookahead:
  - Asian high/low strictly accumulated during 00:00 to 06:45 UTC
  - London entry window active 07:00 to 13:00 UTC only
  - Next-tick candle close execution on or after M15 bar close
Friction: 0.5p commission + 0.2p slippage + dynamic floating spread
Hurdles: Standalone In-Sample >= 12.0% profit, Zero Bfill
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
    def __init__(self, name: str = "", symbol: str = "GBPCHF", sl_pips: float = 20.0, tp_pips: float = 40.0, pip_size: float = 0.0001, spread_limit: float = 2.2, **kwargs):
        self.name = name
        self.symbol = symbol
        self.asset = symbol
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.spread_limit = float(spread_limit)

class Strategy_26_gbpchf_asian_sweep(BaseStrategy):
    """
    GBPCHF Asian Sweep Fade Elite (Slot 26).
    """
    def __init__(
        self,
        asset: str = "GBPCHF",
        delta_pips: float = 3.5,
        sl_pips: float = 20.0,
        tp_pips: float = 40.0,
        min_range_pips: float = 18.0,
        max_range_pips: float = 75.0,
        max_spread: float = 2.2,
        pip_size: float = 0.0001
    ):
        super().__init__(
            name="Strategy_26_gbpchf_asian_sweep",
            symbol=asset,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            pip_size=pip_size,
            spread_limit=max_spread
        )
        self.delta_pips = delta_pips
        self.min_range_pips = min_range_pips
        self.max_range_pips = max_range_pips
        self.max_spread = max_spread

    def generate_signals(self, ticks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)

        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, eff_pip_size=self.pip_size)
        if len(m15) < 30:
            return signals, sl_arr, tp_arr

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

        curr_date = None
        asian_high = -1e9
        asian_low = 1e9
        traded_today = False

        for i in range(len(m15)):
            dt = dates[i]
            hr = hours[i]

            if dt != curr_date:
                curr_date = dt
                asian_high = -1e9
                asian_low = 1e9
                traded_today = False

            if 0 <= hr < 7:
                if h[i] > asian_high: asian_high = h[i]
                if l[i] < asian_low: asian_low = l[i]
                continue

            if 7 <= hr <= 13 and not traded_today and asian_high > asian_low:
                asian_rng = (asian_high - asian_low) / self.pip_size
                if not (self.min_range_pips <= asian_rng <= self.max_range_pips):
                    continue
                if spread[i] > self.max_spread:
                    continue

                sig = 0
                if h[i] >= asian_high + self.delta_pips * self.pip_size and c[i] < asian_high and c[i] < o[i]:
                    sig = SIGNAL_SHORT
                elif l[i] <= asian_low - self.delta_pips * self.pip_size and c[i] > asian_low and c[i] > o[i]:
                    sig = SIGNAL_LONG

                if sig != 0:
                    exec_t = time_ms[i] + dur_ms
                    idx = int(np.searchsorted(tick_ts, exec_t))
                    if idx < n and signals[idx] == 0:
                        signals[idx] = sig
                        sl_arr[idx] = self.sl_pips
                        tp_arr[idx] = self.tp_pips
                        traded_today = True

        return signals, sl_arr, tp_arr

Strategy = Strategy_26_gbpchf_asian_sweep
