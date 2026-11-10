"""
Archived Strategy: Renko SMA-10 + OBV Pullback Strategy (Model V1).
Explored and audited across 53 weeks (18,026,436 authentic ticks).

Core Methodology:
1. Price Partitioning: Continuous tick-to-Renko brick generator (7.0 - 10.0 pips).
2. Trend Confirmation: 10-period Simple Moving Average on brick close prices.
3. Volume Confirmation: On-Balance Volume (OBV) > 14-period EMA signal line.
4. Entry Trigger: Counter-trend pullback brick followed by in-trend bounce brick.
5. Exit Architecture: Volatility-Adaptive Dynamic R:R (1.4R to 2.5R).
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

from tick_engine.strategies.base import (
    BaseStrategy,
    SIGNAL_HOLD,
    SIGNAL_LONG,
    SIGNAL_SHORT,
    SIGNAL_FORCE_EXIT
)
from renko_lab.renko_builder import build_renko_bricks


class RenkoSmaObvStrategy(BaseStrategy):
    """
    Renko SMA-10 + OBV Directional Strategy.
    """
    def __init__(
        self,
        brick_size_pips: float = 7.0,
        sl_bricks: float = 1.5,
        static_rr: float = 1.8,
        rr_mode: str = "volatility_adaptive", # "static", "volatility_adaptive"
        obv_ema_period: int = 14,
        session_filter: bool = True,
        session_start_hour: int = 7,
        session_end_hour: int = 18,
        pip_size: float = 0.0001,
        **kwargs
    ):
        self.brick_size_pips = float(brick_size_pips)
        self.sl_bricks = float(sl_bricks)
        self.static_rr = float(static_rr)
        self.rr_mode = str(rr_mode)
        self.obv_ema_period = int(obv_ema_period)
        self.session_filter = bool(session_filter)
        self.session_start_hour = int(session_start_hour)
        self.session_end_hour = int(session_end_hour)
        self.pip_size = float(pip_size)

        super().__init__(**kwargs)
        self.name = "RenkoSmaObv_V1"

    @classmethod
    def from_config(cls, config_path: Optional[Path] = None) -> "RenkoSmaObvStrategy":
        if config_path is None:
            config_path = Path(__file__).resolve().parent / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cls(**cfg.get("parameters", {}))

    def generate_signals(
        self,
        ticks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_ticks = len(ticks)
        ts_arr = ticks['timestamp']

        bricks = build_renko_bricks(ticks, brick_size_pips=self.brick_size_pips, pip_size=self.pip_size)
        n_b = len(bricks)

        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        min_req = max(30, self.obv_ema_period * 2)
        if n_b < min_req:
            return signals, sl_pips_arr, tp_pips_arr

        closes = np.array([b.close for b in bricks], dtype=np.float64)
        volumes = np.array([b.volume for b in bricks], dtype=np.float64)
        directions = np.array([b.direction for b in bricks], dtype=np.int8)
        brick_ticks = np.array([b.tick_idx for b in bricks], dtype=np.int64)
        brick_ts = np.array([b.timestamp for b in bricks], dtype=np.int64)

        sma_10 = pd.Series(closes).rolling(10).mean().to_numpy()

        # Cumulative OBV and EMA signal
        raw_obv = np.zeros(n_b, dtype=np.float64)
        c_obv = 0.0
        for i in range(n_b):
            c_obv += (volumes[i] if directions[i] == 1 else -volumes[i])
            raw_obv[i] = c_obv

        obv_ema = pd.Series(raw_obv).ewm(span=self.obv_ema_period, adjust=False).mean().to_numpy()
        vols = pd.Series(volumes).rolling(10).mean().to_numpy()

        sl_pips = self.sl_bricks * self.brick_size_pips
        hours = pd.to_datetime(brick_ts, unit='ms').hour

        for i in range(min_req, n_b):
            b = bricks[i]
            tick_idx = b.tick_idx
            if tick_idx >= n_ticks:
                continue

            if self.session_filter:
                h = hours[i]
                if h < self.session_start_hour or h >= self.session_end_hour:
                    continue

            c = closes[i]
            cur_sma = sma_10[i]
            prev_sma = sma_10[i - 1]
            cur_obv = raw_obv[i]
            obv_sig = obv_ema[i]
            d = directions[i]
            d_prev = directions[i - 1]

            bull_entry = (d == 1) and (d_prev == -1) and (c > cur_sma) and (cur_sma > prev_sma) and (cur_obv > obv_sig)
            bear_entry = (d == -1) and (d_prev == 1) and (c < cur_sma) and (cur_sma < prev_sma) and (cur_obv < obv_sig)

            if self.rr_mode == "volatility_adaptive":
                recent_vol = volumes[i]
                avg_vol = vols[i] if not np.isnan(vols[i]) else recent_vol
                if recent_vol < avg_vol * 0.8:
                    effective_rr = 2.5
                elif recent_vol > avg_vol * 1.3:
                    effective_rr = 1.4
                else:
                    effective_rr = 1.8
            else:
                effective_rr = self.static_rr

            tp_pips = sl_pips * effective_rr

            if bull_entry:
                signals[tick_idx] = SIGNAL_LONG
                sl_pips_arr[tick_idx] = sl_pips
                tp_pips_arr[tick_idx] = tp_pips
            elif bear_entry:
                signals[tick_idx] = SIGNAL_SHORT
                sl_pips_arr[tick_idx] = sl_pips
                tp_pips_arr[tick_idx] = tp_pips

        return signals, sl_pips_arr, tp_pips_arr
