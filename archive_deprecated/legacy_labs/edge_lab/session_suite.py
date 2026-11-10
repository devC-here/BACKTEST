"""
Edge Lab: Multi-Session Specialist Engine.
Modular institutional liquidity models tailored specifically for:
- London Morning Session (07:00 - 11:30 UTC): Sweeps Asian Range
- New York US Session (12:30 - 16:30 UTC): Sweeps London Morning Range
- Dual-Session Portfolio: Combined execution maximizing throughput and annual pips
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.strategies.base import (
    BaseStrategy,
    SIGNAL_HOLD,
    SIGNAL_LONG,
    SIGNAL_SHORT,
    SIGNAL_FORCE_EXIT
)


class MultiSessionStrategy(BaseStrategy):
    """
    Session-Specific Liquidity Architecture:
    - 'london_only': Focuses purely on high-precision Asian range sweeps (V9 model)
    - 'new_york_only': Focuses on NY sweeps of the London morning liquidity pool
    - 'dual_session': Executes both session engines concurrently (up to 2 trades/day)
    """
    def __init__(
        self,
        session_mode: str = "dual_session",  # 'london_only', 'new_york_only', 'dual_session'
        target_rr_london: float = 2.0,
        target_rr_ny: float = 2.0,
        **kwargs
    ):
        self.session_mode = session_mode.lower()
        self.target_rr_london = float(target_rr_london)
        self.target_rr_ny = float(target_rr_ny)
        super().__init__(**kwargs)
        self.name = f"MultiSession_{self.session_mode.upper()}"

    def generate_signals(self, ticks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 100:
            return signals, sl_pips_arr, tp_pips_arr

        # Resample to 15-minute candles
        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base_df = pd.DataFrame({'open': mid, 'high': mid, 'low': mid, 'close': mid}, index=ts)

        m15 = base_df.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()

        if len(m15) < 20:
            return signals, sl_pips_arr, tp_pips_arr

        m15['date'] = m15.index.date
        m15['hour'] = m15.index.hour
        m15['minute'] = m15.index.minute
        m15['ema_50'] = m15['close'].ewm(span=50, adjust=False).mean()
        m15['ema_200'] = m15['close'].ewm(span=200, adjust=False).mean()

        tick_ts = ticks['timestamp']
        pip_size = 0.0001
        dur_15m_ms = 15 * 60 * 1000

        enable_london = self.session_mode in ("london_only", "dual_session")
        enable_ny = self.session_mode in ("new_york_only", "dual_session")

        unique_dates = np.unique(m15['date'])

        for d in unique_dates:
            day_m15 = m15[m15['date'] == d]
            daily_open = float(day_m15['open'].iloc[0])

            # =========================================================
            # ENGINE 1: LONDON SPECIALIST (07:00 - 11:30 UTC)
            # Reference: Asian Range (00:00 - 07:00 UTC)
            # =========================================================
            if enable_london:
                asian_bars = day_m15[(day_m15['hour'] >= 0) & (day_m15['hour'] < 7)]
                if len(asian_bars) >= 4:
                    asian_high = float(asian_bars['high'].max())
                    asian_low = float(asian_bars['low'].min())
                    asian_range_pips = (asian_high - asian_low) / pip_size

                    # Asian range compression filter (7 to 32 pips)
                    if 7.0 <= asian_range_pips <= 32.0:
                        ldn_trade_taken = False
                        ldn_bars = day_m15[(day_m15['hour'] >= 7) & (day_m15['hour'] < 12)]

                        for bar_idx, bar in ldn_bars.iterrows():
                            if ldn_trade_taken:
                                break

                            b_open = float(bar['open'])
                            b_high = float(bar['high'])
                            b_low = float(bar['low'])
                            b_close = float(bar['close'])
                            b_time_ms = int(bar.name.value // 10**6) if hasattr(bar.name, 'value') else int(bar_idx)
                            b_ema50 = float(bar['ema_50'])
                            b_ema200 = float(bar['ema_200'])

                            swept_high = (b_high >= asian_high + 0.5 * pip_size) and (b_close < asian_high)
                            swept_low = (b_low <= asian_low - 0.5 * pip_size) and (b_close > asian_low)

                            # Asymmetric Trend Filter:
                            # Shorts favor downtrend (below 50 EMA or daily open)
                            if swept_high and not (b_close < b_ema50 or b_close < daily_open):
                                swept_high = False
                            # Longs require strong uptrend (above 50 AND 200 EMA)
                            if swept_low and not (b_close > b_ema50 and b_close > b_ema200):
                                swept_low = False

                            if swept_high:
                                exec_time_ms = b_time_ms + dur_15m_ms
                                tick_pos = int(np.searchsorted(tick_ts, exec_time_ms))
                                if tick_pos < n_ticks:
                                    signals[tick_pos] = SIGNAL_SHORT
                                    risk = max(5.0 * pip_size, min(16.0 * pip_size, (b_high - b_close) + 2.0 * pip_size))
                                    sl_pips = risk / pip_size
                                    sl_pips_arr[tick_pos] = sl_pips
                                    tp_pips_arr[tick_pos] = max(8.0, sl_pips * self.target_rr_london)
                                    ldn_trade_taken = True

                            elif swept_low:
                                exec_time_ms = b_time_ms + dur_15m_ms
                                tick_pos = int(np.searchsorted(tick_ts, exec_time_ms))
                                if tick_pos < n_ticks:
                                    signals[tick_pos] = SIGNAL_LONG
                                    risk = max(5.0 * pip_size, min(16.0 * pip_size, (b_close - b_low) + 2.0 * pip_size))
                                    sl_pips = risk / pip_size
                                    sl_pips_arr[tick_pos] = sl_pips
                                    tp_pips_arr[tick_pos] = max(8.0, sl_pips * self.target_rr_london)
                                    ldn_trade_taken = True

            # =========================================================
            # ENGINE 2: NEW YORK SPECIALIST (12:30 - 16:30 UTC)
            # Reference: London Morning Range (07:00 - 12:30 UTC)
            # =========================================================
            if enable_ny:
                ldn_ref_bars = day_m15[(day_m15['hour'] >= 7) & ((day_m15['hour'] < 12) | ((day_m15['hour'] == 12) & (day_m15['minute'] < 30)))]
                if len(ldn_ref_bars) >= 10:
                    ldn_ref_high = float(ldn_ref_bars['high'].max())
                    ldn_ref_low = float(ldn_ref_bars['low'].min())
                    ldn_ref_range_pips = (ldn_ref_high - ldn_ref_low) / pip_size

                    # Valid London range filter (12 to 45 pips)
                    if 12.0 <= ldn_ref_range_pips <= 45.0:
                        ny_trade_taken = False
                        ny_bars = day_m15[((day_m15['hour'] == 12) & (day_m15['minute'] >= 30)) | ((day_m15['hour'] >= 13) & (day_m15['hour'] < 16.5))]

                        for bar_idx, bar in ny_bars.iterrows():
                            if ny_trade_taken:
                                break

                            b_open = float(bar['open'])
                            b_high = float(bar['high'])
                            b_low = float(bar['low'])
                            b_close = float(bar['close'])
                            b_time_ms = int(bar.name.value // 10**6) if hasattr(bar.name, 'value') else int(bar_idx)

                            swept_high = (b_high >= ldn_ref_high + 0.5 * pip_size) and (b_close < ldn_ref_high)
                            swept_low = (b_low <= ldn_ref_low - 0.5 * pip_size) and (b_close > ldn_ref_low)

                            if swept_high:
                                exec_time_ms = b_time_ms + dur_15m_ms
                                tick_pos = int(np.searchsorted(tick_ts, exec_time_ms))
                                if tick_pos < n_ticks:
                                    signals[tick_pos] = SIGNAL_SHORT
                                    risk = max(6.0 * pip_size, min(18.0 * pip_size, (b_high - b_close) + 2.5 * pip_size))
                                    sl_pips = risk / pip_size
                                    sl_pips_arr[tick_pos] = sl_pips
                                    tp_pips_arr[tick_pos] = max(10.0, sl_pips * self.target_rr_ny)
                                    ny_trade_taken = True

                            elif swept_low:
                                exec_time_ms = b_time_ms + dur_15m_ms
                                tick_pos = int(np.searchsorted(tick_ts, exec_time_ms))
                                if tick_pos < n_ticks:
                                    signals[tick_pos] = SIGNAL_LONG
                                    risk = max(6.0 * pip_size, min(18.0 * pip_size, (b_close - b_low) + 2.5 * pip_size))
                                    sl_pips = risk / pip_size
                                    sl_pips_arr[tick_pos] = sl_pips
                                    tp_pips_arr[tick_pos] = max(10.0, sl_pips * self.target_rr_ny)
                                    ny_trade_taken = True

            # Session close forced liquidation at 19:30 UTC
            eod_bars = day_m15[day_m15['hour'] == 19]
            if len(eod_bars) > 0:
                eod_time_ms = int(eod_bars.index[0].value // 10**6)
                eod_pos = int(np.searchsorted(tick_ts, eod_time_ms))
                if eod_pos < n_ticks:
                    signals[eod_pos] = SIGNAL_FORCE_EXIT

        return signals, sl_pips_arr, tp_pips_arr
