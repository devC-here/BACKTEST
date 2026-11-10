#!/usr/bin/env python3
"""
================================================================================
  STRATEGY 60 / 80: EURUSD_M15_MACD_Zero_Cross_Elite (EURUSD)
================================================================================
Symbol: EURUSD | Pip Size: 0.0001
Stop Loss: 9.0 pips | Take Profit: 18.0 pips | Risk:Reward: 1:2.0
Spread Limit: 2.5 pips
================================================================================
"""

from __future__ import annotations
import sys
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import pandas as pd
try:
    import ta
except ImportError:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
PIP_SIZE = 0.0001


def _build_bars_internal(ticks: np.ndarray, timeframe_min: int, pip_size: Optional[float] = None) -> pd.DataFrame:
    if len(ticks) == 0:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'spread_pips', 'volume', 'range_pips', 'body_pips'])
    eff_pip_size = pip_size if pip_size is not None else (0.01 if ticks['bid'][0] > 20.0 else 0.0001)
    ts = pd.to_datetime(ticks['timestamp'], unit='ms')
    mid = 0.5 * (ticks['bid'] + ticks['ask'])
    spread_pips = (ticks['ask'] - ticks['bid']) / eff_pip_size
    rule = f"{timeframe_min}min" if timeframe_min < 60 else f"{timeframe_min//60}h"
    df = pd.DataFrame({
        'open': mid, 'high': mid, 'low': mid, 'close': mid,
        'spread_pips': spread_pips, 'vol': 1
    }, index=ts)
    bars = df.resample(rule).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'spread_pips': 'mean', 'vol': 'count'
    }).dropna()
    bars['volume'] = bars['vol']
    bars['range_pips'] = (bars['high'] - bars['low']) / eff_pip_size
    bars['body_pips'] = abs(bars['close'] - bars['open']) / eff_pip_size
    return bars

build_time_bars = _build_bars_internal



SIGNAL_NONE: int = 0
SIGNAL_LONG: int = 1
SIGNAL_SHORT: int = -1

class BaseStrategy:
    def __init__(self, name: str = "", symbol: str = "", sl_pips: float = 10.0, tp_pips: float = 20.0, pip_size: float = 0.0001, spread_limit: float = 2.5, **kwargs):
        self.name = name
        self.symbol = symbol
        self.asset = symbol
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.spread_limit = float(spread_limit)


#!/usr/bin/env python3
"""
EURUSD Production Deployment of DISC59: M15 MACD Zero-Cross Momentum Elite.
"""
from strategies.new_discoveries.disc59_m15_macd_zero_cross_momentum_elite.strategy import (
    EURUSDM15MACDZeroCrossElite,
    M15MACDZeroCrossMomentumBase
)

__all__ = ["EURUSDM15MACDZeroCrossElite", "M15MACDZeroCrossMomentumBase"]


class Strategy_60_eurusd_m15_macd_zero_cross_elite(EURUSDM15MACDZeroCrossElite):
    """Production Wrapper ensuring standard interface & attributes."""
    def __init__(self, **kwargs):
        try:
            super().__init__(asset="EURUSD", sl_pips=float(9.0), tp_pips=float(18.0), pip_size=float(0.0001), **kwargs)
        except TypeError:
            try:
                super().__init__(sl_pips=float(9.0), tp_pips=float(18.0), pip_size=float(0.0001), **kwargs)
            except TypeError:
                super().__init__(**kwargs)
        self.symbol = "EURUSD"
        self.asset = "EURUSD"
        self.name = "EURUSD_M15_MACD_Zero_Cross_Elite"
        self.pip_size = float(0.0001)
        self.sl_pips = float(9.0)
        self.tp_pips = float(18.0)
        self.spread_limit = 2.5
        self.max_spread = 2.5
