#!/usr/bin/env python3
"""
================================================================================
  STRATEGY 72 / 80: EURUSD_H1_Keltner_Squeeze_Explosion_Elite (EURUSD)
================================================================================
Symbol: EURUSD | Pip Size: 0.0001
Stop Loss: 10.0 pips | Take Profit: 20.0 pips | Risk:Reward: 1:2.0
Spread Limit: 2.5 pips

Audited Zero-Bias Performance (20 Continuous Weeks under 0.5p Comm + 0.2p Slip):
  * Trades Executed: 29
  * Win Rate:        41.4%
  * Profit Factor:   1.21
  * Net Realized:    +37.8 pips

Zero-Bias Standards:
- Strict Causal Indexing: All higher timeframes shifted by 1 (.shift(1)).
- Standalone Execution: Pure NumPy/Pandas logic, zero broker dependencies.
- Dual Ingestion: Accepts pre-computed time bars OR self-constructs from ticks.
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


def _build_bars_internal(ticks: np.ndarray, timeframe_min: int, pip_size: float = 0.0001) -> pd.DataFrame:
    """Fast, standalone candle builder from raw tick array (timezone-naive)."""
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
    """Base class providing standard attributes and interface."""
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
EURUSD Production Deployment of DISC58: H1 Keltner Squeeze Explosion Elite.
"""
from strategies.new_discoveries.disc58_h1_keltner_squeeze_explosion_elite.strategy import (
    EURUSDH1KeltnerSqueezeExplosionElite,
    H1KeltnerSqueezeExplosionBase
)

__all__ = ["EURUSDH1KeltnerSqueezeExplosionElite", "H1KeltnerSqueezeExplosionBase"]


class Strategy_72_eurusd_eurusd_h1_keltner_squeeze_explosion_elite(EURUSDH1KeltnerSqueezeExplosionElite):
    """Production Wrapper ensuring standard interface & attributes."""
    def __init__(self, **kwargs):
        try:
            super().__init__(asset="EURUSD", sl_pips=float(10.0), tp_pips=float(20.0), pip_size=float(0.0001), **kwargs)
        except TypeError:
            try:
                super().__init__(name="EURUSD_H1_Keltner_Squeeze_Explosion_Elite", symbol="EURUSD", sl_pips=float(10.0), tp_pips=float(20.0), pip_size=float(0.0001), **kwargs)
            except TypeError:
                try:
                    super().__init__(**kwargs)
                except TypeError:
                    pass
        self.symbol = "EURUSD"
        self.asset = "EURUSD"
        self.name = "EURUSD_H1_Keltner_Squeeze_Explosion_Elite"
        self.pip_size = float(0.0001)
        if not hasattr(self, "sl_pips") or self.sl_pips == 0:
            self.sl_pips = float(10.0)
        if not hasattr(self, "tp_pips") or self.tp_pips == 0:
            self.tp_pips = float(20.0)
        if not hasattr(self, "spread_limit"):
            self.spread_limit = float(2.5)
