"""
London Session Breakout & Liquidity Range Strategy.
Edge Lab incremental hypothesis-testing model.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

# Add parent path if needed
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


class LondonBreakoutStrategy(BaseStrategy):
    """
    Quantitative London Breakout strategy with modular feature toggles
    for hypothesis testing and edge discovery.
    """
    def __init__(
        self,
        mode: str = "v0_naive",
        asian_start_hour: int = 0,
        asian_end_hour: int = 7,
        trade_start_hour: int = 7,
        trade_end_hour: int = 12,
        exit_hour: int = 17,
        min_asian_range_pips: float = 8.0,
        max_asian_range_pips: float = 25.0,
        fixed_sl_pips: float = 15.0,
        fixed_tp_pips: float = 30.0,
        target_rr: float = 1.8,
        **kwargs
    ):
        self.mode = mode.lower()
        self.asian_start_hour = int(asian_start_hour)
        self.asian_end_hour = int(asian_end_hour)
        self.trade_start_hour = int(trade_start_hour)
        self.trade_end_hour = int(trade_end_hour)
        self.exit_hour = int(exit_hour)

        self.min_asian_range_pips = float(min_asian_range_pips)
        self.max_asian_range_pips = float(max_asian_range_pips)
        self.fixed_sl_pips = float(fixed_sl_pips)
        self.fixed_tp_pips = float(fixed_tp_pips)
        self.target_rr = float(target_rr)

        # Mode configurations:
        # v0_naive: raw breakout on wick, fixed SL/TP, no range filter, no trend filter
        # v1_range_filter: adds compression filter (min/max range)
        # v2_candle_close: requires M15 candle close outside range
        # v3_trend_regime: adds Daily Open / H1 EMA trend alignment
        # v4_optimized: dynamic structural SL (Asian mid), 1.8R TP, session time close
        # v5_sweep_fade: Fades liquidity sweeps of Asian High/Low (Mean Reversion / Trap)
        # v6_sweep_trend: Fades sweeps only aligned with broader daily trend
        super().__init__(**kwargs)
        self.name = f"LondonBreakout_{self.mode.upper()}"

    def generate_signals(self, ticks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)

        if n_ticks < 100:
            return signals, sl_pips_arr, tp_pips_arr

        # Resample to M15 candles for session mapping
        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base_df = pd.DataFrame({'open': mid, 'high': mid, 'low': mid, 'close': mid}, index=ts)

        m15 = base_df.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()

        if len(m15) < 10:
            return signals, sl_pips_arr, tp_pips_arr

        m15['date'] = m15.index.date
        m15['hour'] = m15.index.hour
        m15['atr_14'] = (m15['high'] - m15['low']).rolling(14).mean().bfill()
        m15['ema_50'] = m15['close'].ewm(span=50, adjust=False).mean()

        tick_ts = ticks['timestamp']
        pip_size = 0.0001
        dur_15m_ms = 15 * 60 * 1000

        # Feature flags based on mode
        is_sweep_mode = self.mode in ("v5_sweep_fade", "v6_sweep_trend")
        apply_range_filter = self.mode in ("v1_range_filter", "v2_candle_close", "v3_trend_regime", "v4_optimized", "v5_sweep_fade", "v6_sweep_trend")
        require_candle_close = self.mode in ("v2_candle_close", "v3_trend_regime", "v4_optimized")
        apply_trend_filter = self.mode in ("v3_trend_regime", "v4_optimized", "v6_sweep_trend")
        dynamic_sl_tp = self.mode in ("v4_optimized", "v5_sweep_fade", "v6_sweep_trend")
        apply_session_exit = self.mode in ("v4_optimized", "v5_sweep_fade", "v6_sweep_trend")

        unique_dates = np.unique(m15['date'])

        for d in unique_dates:
            day_mask = (m15['date'] == d)
            day_m15 = m15[day_mask]

            # 1. Calculate Asian Session Range (00:00 - 07:00 UTC)
            asian_bars = day_m15[(day_m15['hour'] >= self.asian_start_hour) & (day_m15['hour'] < self.asian_end_hour)]
            if len(asian_bars) < 4:
                continue

            asian_high = float(asian_bars['high'].max())
            asian_low = float(asian_bars['low'].min())
            asian_mid = 0.5 * (asian_high + asian_low)
            asian_range_pips = (asian_high - asian_low) / pip_size
            daily_open = float(asian_bars['open'].iloc[0])

            # Range Filter
            if apply_range_filter:
                if asian_range_pips < self.min_asian_range_pips or asian_range_pips > self.max_asian_range_pips:
                    continue

            # 2. Evaluate London Trading Window (07:00 - 12:00 UTC)
            trade_bars = day_m15[(day_m15['hour'] >= self.trade_start_hour) & (day_m15['hour'] < self.trade_end_hour)]
            if len(trade_bars) == 0:
                continue

            trade_taken = False

            for bar_idx, bar in trade_bars.iterrows():
                if trade_taken:
                    break

                b_open = float(bar['open'])
                b_high = float(bar['high'])
                b_low = float(bar['low'])
                b_close = float(bar['close'])
                b_time_ms = int(bar.name.value // 10**6) if hasattr(bar.name, 'value') else int(bar_idx)
                b_atr = float(bar.get('atr_14', 0.0012))
                b_ema50 = float(bar.get('ema_50', b_close))

                # Mode Logic
                if is_sweep_mode:
                    # Liquidity Sweep & Fade Mode (Mean Reversion)
                    # Swept high: High pierced above Asian high, but bar closed back inside
                    is_sweep_short = (b_high >= asian_high + 0.5 * pip_size) and (b_close < asian_high)
                    # Swept low: Low pierced below Asian low, but bar closed back inside
                    is_sweep_long = (b_low <= asian_low - 0.5 * pip_size) and (b_close > asian_low)

                    if self.mode == "v6_sweep_trend":
                        # Only short sweeps in broader downtrends, long sweeps in uptrends
                        if is_sweep_short and not (b_close < b_ema50):
                            is_sweep_short = False
                        if is_sweep_long and not (b_close > b_ema50):
                            is_sweep_long = False

                    if is_sweep_short:
                        exec_time_ms = b_time_ms + dur_15m_ms
                        tick_pos = int(np.searchsorted(tick_ts, exec_time_ms))
                        if tick_pos < n_ticks:
                            signals[tick_pos] = SIGNAL_SHORT
                            # Structural SL: 2 pips above sweep wick
                            risk_dist = max(5.0 * pip_size, min(16.0 * pip_size, (b_high - b_close) + 2.0 * pip_size))
                            sl_pips = risk_dist / pip_size
                            # Target Asian Midpoint or 1.8R
                            target_dist = max(b_close - asian_mid, sl_pips * self.target_rr * pip_size)
                            tp_pips = max(8.0, target_dist / pip_size)
                            sl_pips_arr[tick_pos] = sl_pips
                            tp_pips_arr[tick_pos] = tp_pips
                            trade_taken = True

                    elif is_sweep_long:
                        exec_time_ms = b_time_ms + dur_15m_ms
                        tick_pos = int(np.searchsorted(tick_ts, exec_time_ms))
                        if tick_pos < n_ticks:
                            signals[tick_pos] = SIGNAL_LONG
                            # Structural SL: 2 pips below sweep wick
                            risk_dist = max(5.0 * pip_size, min(16.0 * pip_size, (b_close - b_low) + 2.0 * pip_size))
                            sl_pips = risk_dist / pip_size
                            # Target Asian Midpoint or 1.8R
                            target_dist = max(asian_mid - b_close, sl_pips * self.target_rr * pip_size)
                            tp_pips = max(8.0, target_dist / pip_size)
                            sl_pips_arr[tick_pos] = sl_pips
                            tp_pips_arr[tick_pos] = tp_pips
                            trade_taken = True

                else:
                    # Breakout Modes (V0 to V4)
                    is_long_breakout = False
                    is_short_breakout = False

                    if require_candle_close:
                        if b_close > asian_high:
                            is_long_breakout = True
                        elif b_close < asian_low:
                            is_short_breakout = True
                    else:
                        if b_high > asian_high:
                            is_long_breakout = True
                        elif b_low < asian_low:
                            is_short_breakout = True

                    if apply_trend_filter:
                        if is_long_breakout and not (b_close > daily_open and b_close > b_ema50):
                            is_long_breakout = False
                        if is_short_breakout and not (b_close < daily_open and b_close < b_ema50):
                            is_short_breakout = False

                    if is_long_breakout or is_short_breakout:
                        exec_time_ms = b_time_ms + dur_15m_ms
                        tick_pos = int(np.searchsorted(tick_ts, exec_time_ms))

                        if tick_pos < n_ticks:
                            if is_long_breakout:
                                signals[tick_pos] = SIGNAL_LONG
                                if dynamic_sl_tp:
                                    risk_dist = max(7.0 * pip_size, min(20.0 * pip_size, b_close - asian_mid))
                                    sl_pips = risk_dist / pip_size
                                    tp_pips = sl_pips * self.target_rr
                                else:
                                    sl_pips = self.fixed_sl_pips
                                    tp_pips = self.fixed_tp_pips

                                sl_pips_arr[tick_pos] = sl_pips
                                tp_pips_arr[tick_pos] = tp_pips
                                trade_taken = True

                            elif is_short_breakout:
                                signals[tick_pos] = SIGNAL_SHORT
                                if dynamic_sl_tp:
                                    risk_dist = max(7.0 * pip_size, min(20.0 * pip_size, asian_mid - b_close))
                                    sl_pips = risk_dist / pip_size
                                    tp_pips = sl_pips * self.target_rr
                                else:
                                    sl_pips = self.fixed_sl_pips
                                    tp_pips = self.fixed_tp_pips

                                sl_pips_arr[tick_pos] = sl_pips
                                tp_pips_arr[tick_pos] = tp_pips
                                trade_taken = True

            # Session End Force Exit (V4+)
            if apply_session_exit and trade_taken:
                exit_bars = day_m15[day_m15['hour'] >= self.exit_hour]
                if len(exit_bars) > 0:
                    exit_bar_t = int(exit_bars.index[0].value // 10**6)
                    exit_pos = int(np.searchsorted(tick_ts, exit_bar_t))
                    if exit_pos < n_ticks and signals[exit_pos] == 0:
                        signals[exit_pos] = SIGNAL_FORCE_EXIT

        return signals, sl_pips_arr, tp_pips_arr
