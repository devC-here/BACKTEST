"""
Edge Lab: Advanced Generation Strategy Models (V7 to V11).
Extends the London & NY session liquidity models with pin bar rejection,
dual session windows, asymmetric regime bias, and ATR volatility scaling.
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


class AdvancedLiquidityStrategy(BaseStrategy):
    """
    Second-generation quantitative session liquidity model testing:
    - V7: Pin-Bar Rejection Sweep
    - V8: Dual Session (London + NY Open Overlap)
    - V9: Asymmetric Trend Regime Filter
    - V10: Volatility-Adjusted ATR Bracketing
    - V11: The Hybrid Alpha Model
    """
    def __init__(
        self,
        mode: str = "v7_pinbar_sweep",
        asian_start_hour: int = 0,
        asian_end_hour: int = 7,
        london_start_hour: int = 7,
        london_end_hour: int = 12,
        ny_start_hour: int = 12,
        ny_end_hour: int = 16,
        exit_hour: int = 18,
        min_asian_range_pips: float = 7.0,
        max_asian_range_pips: float = 32.0,
        min_rejection_wick_ratio: float = 0.35,
        target_rr: float = 2.0,
        **kwargs
    ):
        self.mode = mode.lower()
        self.asian_start_hour = int(asian_start_hour)
        self.asian_end_hour = int(asian_end_hour)
        self.london_start_hour = int(london_start_hour)
        self.london_end_hour = int(london_end_hour)
        self.ny_start_hour = int(ny_start_hour)
        self.ny_end_hour = int(ny_end_hour)
        self.exit_hour = int(exit_hour)

        self.min_asian_range_pips = float(min_asian_range_pips)
        self.max_asian_range_pips = float(max_asian_range_pips)
        self.min_rejection_wick_ratio = float(min_rejection_wick_ratio)
        self.target_rr = float(target_rr)

        super().__init__(**kwargs)
        self.name = f"LiquidityModel_{self.mode.upper()}"

    def generate_signals(self, ticks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 100:
            return signals, sl_pips_arr, tp_pips_arr

        # Resample to M15 candles
        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base_df = pd.DataFrame({'open': mid, 'high': mid, 'low': mid, 'close': mid}, index=ts)

        m15 = base_df.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()

        if len(m15) < 15:
            return signals, sl_pips_arr, tp_pips_arr

        m15['date'] = m15.index.date
        m15['hour'] = m15.index.hour
        m15['atr_14'] = (m15['high'] - m15['low']).rolling(14).mean().bfill()
        m15['ema_20'] = m15['close'].ewm(span=20, adjust=False).mean()
        m15['ema_50'] = m15['close'].ewm(span=50, adjust=False).mean()
        m15['ema_200'] = m15['close'].ewm(span=200, adjust=False).mean()

        tick_ts = ticks['timestamp']
        pip_size = 0.0001
        dur_15m_ms = 15 * 60 * 1000

        # Mode configuration flags
        include_ny = self.mode in ("v8_dual_session", "v11_hybrid_alpha")
        strict_pinbar = self.mode in ("v7_pinbar_sweep", "v11_hybrid_alpha")
        asymmetric_trend = self.mode in ("v9_short_bias", "v11_hybrid_alpha")
        atr_scaling = self.mode in ("v10_atr_bracket", "v11_hybrid_alpha")

        unique_dates = np.unique(m15['date'])

        for d in unique_dates:
            day_mask = (m15['date'] == d)
            day_m15 = m15[day_mask]

            # 1. Asian Session (00:00 - 07:00 UTC)
            asian_bars = day_m15[(day_m15['hour'] >= self.asian_start_hour) & (day_m15['hour'] < self.asian_end_hour)]
            if len(asian_bars) < 4:
                continue

            asian_high = float(asian_bars['high'].max())
            asian_low = float(asian_bars['low'].min())
            asian_mid = 0.5 * (asian_high + asian_low)
            asian_range_pips = (asian_high - asian_low) / pip_size
            daily_open = float(asian_bars['open'].iloc[0])

            # Range filter
            if asian_range_pips < self.min_asian_range_pips or asian_range_pips > self.max_asian_range_pips:
                continue

            # 2. Define trading sessions: London Morning (07-12) and optionally NY Open (12-16)
            sessions_to_evaluate = [(self.london_start_hour, self.london_end_hour, asian_high, asian_low, asian_mid)]

            if include_ny:
                # Pre-NY Range (Asian + London Morning range up to 12:00)
                morn_bars = day_m15[day_m15['hour'] < self.ny_start_hour]
                if len(morn_bars) >= 10:
                    ny_ref_high = float(morn_bars['high'].max())
                    ny_ref_low = float(morn_bars['low'].min())
                    ny_ref_mid = 0.5 * (ny_ref_high + ny_ref_low)
                    sessions_to_evaluate.append((self.ny_start_hour, self.ny_end_hour, ny_ref_high, ny_ref_low, ny_ref_mid))

            trade_taken = False

            for s_start, s_end, ref_high, ref_low, ref_mid in sessions_to_evaluate:
                if trade_taken:
                    break

                sess_bars = day_m15[(day_m15['hour'] >= s_start) & (day_m15['hour'] < s_end)]
                if len(sess_bars) == 0:
                    continue

                for bar_idx, bar in sess_bars.iterrows():
                    if trade_taken:
                        break

                    b_open = float(bar['open'])
                    b_high = float(bar['high'])
                    b_low = float(bar['low'])
                    b_close = float(bar['close'])
                    b_time_ms = int(bar.name.value // 10**6) if hasattr(bar.name, 'value') else int(bar_idx)
                    b_atr = float(bar.get('atr_14', 0.0012))
                    b_ema50 = float(bar.get('ema_50', b_close))
                    b_ema200 = float(bar.get('ema_200', b_close))

                    rng = max(1e-5, b_high - b_low)
                    upper_wick = b_high - max(b_open, b_close)
                    lower_wick = min(b_open, b_close) - b_low

                    upper_wick_ratio = upper_wick / rng
                    lower_wick_ratio = lower_wick / rng

                    # Sweep Detection:
                    # Short Sweep: Pierced above reference high, closed back inside
                    swept_high = (b_high >= ref_high + 0.5 * pip_size) and (b_close < ref_high)
                    # Long Sweep: Pierced below reference low, closed back inside
                    swept_low = (b_low <= ref_low - 0.5 * pip_size) and (b_close > ref_low)

                    # Pin Bar Filter (V7 & V11)
                    if strict_pinbar:
                        if swept_high and upper_wick_ratio < self.min_rejection_wick_ratio:
                            swept_high = False
                        if swept_low and lower_wick_ratio < self.min_rejection_wick_ratio:
                            swept_low = False

                    # Macro Trend & Asymmetric Filter (V9 & V11)
                    if asymmetric_trend:
                        # Shorts: Favour downtrends (below 50 EMA or below daily open)
                        if swept_high and not (b_close < b_ema50 or b_close < daily_open):
                            swept_high = False
                        # Longs: Require strict macro uptrend (above 50 EMA AND above 200 EMA)
                        if swept_low and not (b_close > b_ema50 and b_close > b_ema200):
                            swept_low = False
                    else:
                        # Default V6 trend filter
                        if swept_high and not (b_close < b_ema50):
                            swept_high = False
                        if swept_low and not (b_close > b_ema50):
                            swept_low = False

                    # Execute Trade Signal
                    if swept_high:
                        exec_time_ms = b_time_ms + dur_15m_ms
                        tick_pos = int(np.searchsorted(tick_ts, exec_time_ms))
                        if tick_pos < n_ticks:
                            signals[tick_pos] = SIGNAL_SHORT

                            if atr_scaling:
                                # ATR-scaled risk (1.2x ATR)
                                risk_dist = max(5.0 * pip_size, min(16.0 * pip_size, 1.2 * b_atr))
                                sl_pips = risk_dist / pip_size
                                tp_pips = sl_pips * self.target_rr
                            else:
                                risk_dist = max(5.0 * pip_size, min(16.0 * pip_size, (b_high - b_close) + 2.0 * pip_size))
                                sl_pips = risk_dist / pip_size
                                target_dist = max(b_close - ref_mid, sl_pips * self.target_rr * pip_size)
                                tp_pips = max(8.0, target_dist / pip_size)

                            sl_pips_arr[tick_pos] = sl_pips
                            tp_pips_arr[tick_pos] = tp_pips
                            trade_taken = True

                    elif swept_low:
                        exec_time_ms = b_time_ms + dur_15m_ms
                        tick_pos = int(np.searchsorted(tick_ts, exec_time_ms))
                        if tick_pos < n_ticks:
                            signals[tick_pos] = SIGNAL_LONG

                            if atr_scaling:
                                risk_dist = max(5.0 * pip_size, min(16.0 * pip_size, 1.2 * b_atr))
                                sl_pips = risk_dist / pip_size
                                tp_pips = sl_pips * self.target_rr
                            else:
                                risk_dist = max(5.0 * pip_size, min(16.0 * pip_size, (b_close - b_low) + 2.0 * pip_size))
                                sl_pips = risk_dist / pip_size
                                target_dist = max(ref_mid - b_close, sl_pips * self.target_rr * pip_size)
                                tp_pips = max(8.0, target_dist / pip_size)

                            sl_pips_arr[tick_pos] = sl_pips
                            tp_pips_arr[tick_pos] = tp_pips
                            trade_taken = True

            # Session End Force Exit at 18:00 UTC (London / Europe close)
            if trade_taken:
                exit_bars = day_m15[day_m15['hour'] >= self.exit_hour]
                if len(exit_bars) > 0:
                    exit_bar_t = int(exit_bars.index[0].value // 10**6)
                    exit_pos = int(np.searchsorted(tick_ts, exit_bar_t))
                    if exit_pos < n_ticks and signals[exit_pos] == 0:
                        signals[exit_pos] = SIGNAL_FORCE_EXIT

        return signals, sl_pips_arr, tp_pips_arr
