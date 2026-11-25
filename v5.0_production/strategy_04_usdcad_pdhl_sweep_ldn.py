#!/usr/bin/env python3
"""USDCAD v5.0 Candidate: Causal Prior-Day High/Low Liquidity Sweep Reversal (London 07-12 UTC)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

_cur = Path(__file__).resolve()
ROOT_DIR = next((p for p in _cur.parents if (p / "tick_engine").exists()), _cur.parent.parent)
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.causal_bars import (
    build_causal_time_bars,
    get_session_dates,
    get_session_hours,
    compute_prior_day_pivots,
)


class USDCADPDHLSweepLdnV5:
    """Strictly causal Prior-Day High/Low liquidity sweep reversal during London session."""
    symbol: str = "USDCAD"
    archetype: str = "PDHL_SWEEP"
    sl_pips: float = 16.0
    tp_pips: float = 32.0
    pip_size: float = 0.0001
    spread_limit: float = 1.8

    def __init__(self, symbol="USDCAD", sl_pips=16.0, tp_pips=32.0, pip_size=0.0001, spread_limit=1.8, **kwargs):
        self.symbol = symbol
        self.asset = symbol
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.spread_limit = float(spread_limit)
        self.max_spread = float(spread_limit)
        self.name = "v5_usdcad_pdhl_sweep_ldn"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_causal_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 40:
            return signals, sl_arr, tp_arr

        pivots = compute_prior_day_pivots(m15, min_bars_per_day=15, timeframe_min=15)
        dates = get_session_dates(m15, timeframe_min=15)
        hours = get_session_hours(m15, timeframe_min=15)

        o = m15["open"].to_numpy(dtype=np.float64)
        h = m15["high"].to_numpy(dtype=np.float64)
        l = m15["low"].to_numpy(dtype=np.float64)
        c = m15["close"].to_numpy(dtype=np.float64)
        sp = m15["spread_pips"].to_numpy(dtype=np.float64)
        close_ms = m15["close_time_ms"].to_numpy(dtype=np.int64)
        tick_ts = ticks["timestamp"]

        sw_buf = 2.5 * self.pip_size
        last_trade_date = None

        for i in range(25, len(m15)):
            hr = int(hours[i])
            if not (7 <= hr <= 12):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            piv = pivots.get(dt)
            if piv is None or sp[i] > self.spread_limit:
                continue
            rng = h[i] - l[i]
            if rng < 1.5 * self.pip_size:
                continue

            pd_h, pd_l = piv["prev_H"], piv["prev_L"]
            sig = 0
            if l[i] <= pd_l - sw_buf and c[i] > pd_l and c[i] > o[i]:
                if (o[i] - l[i]) / rng >= 0.45:
                    sig = 1
            elif h[i] >= pd_h + sw_buf and c[i] < pd_h and c[i] < o[i]:
                if (h[i] - o[i]) / rng >= 0.45:
                    sig = -1

            if sig != 0:
                idx = int(np.searchsorted(tick_ts, close_ms[i]))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


Strategy = USDCADPDHLSweepLdnV5
