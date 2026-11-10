"""
Multi-Timeframe Renko Engine and Strategy.
Evaluates Renko SMA-10 + OBV across multiple timeframe configurations:
1. Brick Construction Basis: Pure Ticks vs M1, M5, M15, H1 candle close resampling
2. Higher-Timeframe Trend Filtering: Aligning Renko entries with M15 / H1 / H4 macro EMAs
3. Timeframe-Calibrated Volatility Bricks: Scalp (M5 ATR), Intraday (M15 ATR), Swing (H1 ATR)
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from renko_lab.renko_builder import RenkoBrick
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT, SIGNAL_FORCE_EXIT
from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.core.metrics import compute_metrics


def build_timeframe_candles(ticks: np.ndarray, timeframe: str = "5min") -> pd.DataFrame:
    """Resamples ticks into OHLCV candles of specified timeframe."""
    ts = pd.to_datetime(ticks['timestamp'], unit='ms')
    mid = 0.5 * (ticks['bid'] + ticks['ask'])
    base_df = pd.DataFrame({'price': mid}, index=ts)

    df = base_df.resample(timeframe).agg({
        'price': ['first', 'max', 'min', 'last', 'count']
    }).dropna()
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    return df


def build_renko_from_candles(
    candles_df: pd.DataFrame,
    brick_size_pips: float = 7.0,
    pip_size: float = 0.0001
) -> List[RenkoBrick]:
    """Constructs Renko bricks from timeframe candle closes (standard TradingView/MT5 model)."""
    brick_size = brick_size_pips * pip_size
    closes = candles_df['close'].to_numpy()
    volumes = candles_df['volume'].to_numpy()
    timestamps = (candles_df.index.astype(np.int64) // 10**6).to_numpy()

    n = len(closes)
    if n == 0:
        return []

    base_price = np.floor(closes[0] / brick_size) * brick_size
    bricks: List[RenkoBrick] = []
    current_direction = 0
    last_close = base_price

    for i in range(n):
        price = closes[i]
        vol = int(volumes[i])
        t = int(timestamps[i])

        if current_direction == 0:
            if price >= base_price + brick_size:
                b = RenkoBrick(len(bricks), i, t, base_price, base_price + brick_size, base_price + brick_size, base_price, 1, vol)
                bricks.append(b)
                current_direction = 1
                last_close = b.close
            elif price <= base_price - brick_size:
                b = RenkoBrick(len(bricks), i, t, base_price, base_price - brick_size, base_price, base_price - brick_size, -1, vol)
                bricks.append(b)
                current_direction = -1
                last_close = b.close

        elif current_direction == 1:
            while price >= last_close + brick_size:
                b = RenkoBrick(len(bricks), i, t, last_close, last_close + brick_size, last_close + brick_size, last_close, 1, vol)
                bricks.append(b)
                last_close = b.close

            if price <= last_close - 2.0 * brick_size:
                new_open = last_close - brick_size
                new_close = last_close - 2.0 * brick_size
                b = RenkoBrick(len(bricks), i, t, new_open, new_close, new_open, new_close, -1, vol)
                bricks.append(b)
                current_direction = -1
                last_close = b.close

                while price <= last_close - brick_size:
                    b_cont = RenkoBrick(len(bricks), i, t, last_close, last_close - brick_size, last_close, last_close - brick_size, -1, 1)
                    bricks.append(b_cont)
                    last_close = b_cont.close

        elif current_direction == -1:
            while price <= last_close - brick_size:
                b = RenkoBrick(len(bricks), i, t, last_close, last_close - brick_size, last_close, last_close - brick_size, -1, vol)
                bricks.append(b)
                last_close = b.close

            if price >= last_close + 2.0 * brick_size:
                new_open = last_close + brick_size
                new_close = last_close + 2.0 * brick_size
                b = RenkoBrick(len(bricks), i, t, new_open, new_close, new_close, new_open, 1, vol)
                bricks.append(b)
                current_direction = 1
                last_close = b.close

                while price >= last_close + brick_size:
                    b_cont = RenkoBrick(len(bricks), i, t, last_close, last_close + brick_size, last_close + brick_size, last_close, 1, 1)
                    bricks.append(b_cont)
                    last_close = b_cont.close

    return bricks


class MultiTimeframeRenkoStrategy(BaseStrategy):
    """
    Multi-Timeframe Renko Strategy:
    - Base Timeframe: 'tick', '1min', '5min', '15min', '1h'
    - Macro Trend Filter: 'none', 'm15', 'h1', 'h4'
    """
    def __init__(
        self,
        base_timeframe: str = "tick",       # 'tick', '1min', '5min', '15min', '1h'
        macro_timeframe: str = "15min",     # 'none', '15min', '1h', '4h'
        brick_size_pips: float = 7.0,
        sma_period: int = 10,
        target_rr: float = 1.8,
        sl_bricks: float = 2.0,
        pip_size: float = 0.0001,
        **kwargs
    ):
        self.base_timeframe = base_timeframe.lower()
        self.macro_timeframe = macro_timeframe.lower()
        self.brick_size_pips = float(brick_size_pips)
        self.sma_period = int(sma_period)
        self.target_rr = float(target_rr)
        self.sl_bricks = float(sl_bricks)
        self.pip_size = float(pip_size)

        super().__init__(**kwargs)
        self.name = f"MTF_Renko_{self.base_timeframe}_{self.macro_timeframe}_{int(brick_size_pips)}p"

    def generate_signals(self, ticks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 100:
            return signals, sl_pips_arr, tp_pips_arr

        tick_ts = ticks['timestamp']

        # 1. Build Bricks according to Base Timeframe
        if self.base_timeframe == "tick":
            from renko_lab.renko_builder import build_renko_bricks
            bricks = build_renko_bricks(ticks, brick_size_pips=self.brick_size_pips, pip_size=self.pip_size)
            brick_timestamps = [b.timestamp for b in bricks]
        else:
            candles = build_timeframe_candles(ticks, timeframe=self.base_timeframe)
            bricks = build_renko_from_candles(candles, brick_size_pips=self.brick_size_pips, pip_size=self.pip_size)
            brick_timestamps = [b.timestamp for b in bricks]

        n_b = len(bricks)
        if n_b < self.sma_period + 5:
            return signals, sl_pips_arr, tp_pips_arr

        # 2. Indicator Calculation on Bricks
        closes = np.array([b.close for b in bricks], dtype=np.float64)
        volumes = np.array([b.volume for b in bricks], dtype=np.float64)
        directions = np.array([b.direction for b in bricks], dtype=np.int8)

        sma_10 = pd.Series(closes).rolling(self.sma_period).mean().to_numpy()

        raw_obv = np.zeros(n_b, dtype=np.float64)
        c_obv = 0.0
        for i in range(n_b):
            c_obv += (volumes[i] if directions[i] == 1 else -volumes[i])
            raw_obv[i] = c_obv
        obv_ema = pd.Series(raw_obv).ewm(span=14, adjust=False).mean().to_numpy()

        # 3. Macro Trend Context (if enabled)
        macro_uptrend = None
        macro_downtrend = None

        if self.macro_timeframe in ("15min", "1h", "4h"):
            macro_candles = build_timeframe_candles(ticks, timeframe=self.macro_timeframe)
            if len(macro_candles) >= 30:
                m_closes = macro_candles['close']
                m_ema50 = m_closes.ewm(span=50, adjust=False).mean()
                m_ts = (macro_candles.index.astype(np.int64) // 10**6).to_numpy()
                m_up = (m_closes > m_ema50).to_numpy()
                m_down = (m_closes < m_ema50).to_numpy()

                # Align with brick timestamps using searchsorted
                macro_indices = np.searchsorted(m_ts, brick_timestamps) - 1
                macro_indices = np.clip(macro_indices, 0, len(m_up) - 1)
                macro_uptrend = m_up[macro_indices]
                macro_downtrend = m_down[macro_indices]

        sl_pips = self.sl_bricks * self.brick_size_pips
        tp_pips = sl_pips * self.target_rr

        # 4. Generate Signal on Brick Completion
        for i in range(self.sma_period, n_b):
            b_ts = brick_timestamps[i]
            tick_idx = int(np.searchsorted(tick_ts, b_ts))
            if tick_idx >= n_ticks:
                continue

            c = closes[i]
            sma_cur = sma_10[i]
            sma_prev = sma_10[i - 1]
            obv_cur = raw_obv[i]
            obv_sig = obv_ema[i]
            d = directions[i]
            d_prev = directions[i - 1]

            # Bounce / Momentum condition
            bull_signal = (d == 1) and (d_prev == -1) and (c > sma_cur) and (sma_cur > sma_prev) and (obv_cur > obv_sig)
            bear_signal = (d == -1) and (d_prev == 1) and (c < sma_cur) and (sma_cur < sma_prev) and (obv_cur < obv_sig)

            # Macro Trend Gate
            if macro_uptrend is not None:
                if bull_signal and not macro_uptrend[i]:
                    bull_signal = False
                if bear_signal and not macro_downtrend[i]:
                    bear_signal = False

            if bull_signal:
                signals[tick_idx] = SIGNAL_LONG
                sl_pips_arr[tick_idx] = sl_pips
                tp_pips_arr[tick_idx] = tp_pips
            elif bear_signal:
                signals[tick_idx] = SIGNAL_SHORT
                sl_pips_arr[tick_idx] = sl_pips
                tp_pips_arr[tick_idx] = tp_pips

        return signals, sl_pips_arr, tp_pips_arr
