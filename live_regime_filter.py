#!/usr/bin/env python3
"""
================================================================================
  LIVE PRODUCTION MARKET REGIME FILTER & ARCHETYPE KILL-SWITCH ENGINE
================================================================================
Zero-dependency, high-speed NumPy implementation for real-time tick/M15 evaluation.
Evaluates incoming strategy signals against instantaneous market microstructure.
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

class LiveMarketRegimeFilter:
    """
    Evaluates real-time market regimes and enforces archetype-specific kill-switches.
    """
    def __init__(
        self,
        ker_period: int = 20,
        chop_period: int = 14,
        hurst_lag: int = 5,
        hurst_window: int = 100,
        atr_fast: int = 5,
        atr_slow: int = 50,
        bb_period: int = 20,
        bbwp_window: int = 252,
    ):
        self.ker_period = ker_period
        self.chop_period = chop_period
        self.hurst_lag = hurst_lag
        self.hurst_window = hurst_window
        self.atr_fast = atr_fast
        self.atr_slow = atr_slow
        self.bb_period = bb_period
        self.bbwp_window = bbwp_window

    def evaluate_live_permission(
        self,
        strategy_name: str,
        symbol: str,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        spread_pips: float,
        current_utc_dt: pd.Timestamp
    ) -> Tuple[bool, str, Dict[str, float]]:
        """
        Returns:
            permitted: bool (True = Allow trade execution; False = Kill-switch quashes trade)
            reason: str (Diagnostic reason if quashed)
            metrics: dict (Instantaneous mathematical values)
        """
        n = len(close)
        if n < max(self.hurst_window + self.hurst_lag, self.bbwp_window + self.bb_period):
            # Not enough warmup history, permit with diagnostic warning
            return True, "WARMUP_PASS", {}

        # 1. Kaufman Efficiency Ratio (KER)
        abs_diff = np.abs(np.diff(close))
        net_change = abs(close[-1] - close[-self.ker_period - 1])
        path_sum = np.sum(abs_diff[-self.ker_period:])
        ker = float(net_change / path_sum) if path_sum > 1e-12 else 0.0

        # 2. True Range & ATRs
        prev_c = np.roll(close, 1)
        prev_c[0] = close[0]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))
        atr_fast_val = float(np.mean(tr[-self.atr_fast:]))
        atr_slow_val = float(np.mean(tr[-self.atr_slow:]))
        rvr = float(atr_fast_val / atr_slow_val) if atr_slow_val > 1e-12 else 1.0

        # 3. Choppiness Index (CHOP)
        tr_sum = np.sum(tr[-self.chop_period:])
        span = np.max(high[-self.chop_period:]) - np.min(low[-self.chop_period:])
        log10_n = np.log10(float(self.chop_period))
        chop = float(100.0 * np.log10(tr_sum / span) / log10_n) if span > 1e-12 and tr_sum > 1e-12 else 50.0

        # 4. Variance Ratio Hurst Exponent
        log_p = np.log(close)
        ret1 = np.diff(log_p)
        retq = log_p[self.hurst_lag:] - log_p[:-self.hurst_lag]
        
        w1 = ret1[-self.hurst_window:]
        wq = retq[-self.hurst_window:]
        v1 = np.var(w1, ddof=1)
        vq = np.var(wq, ddof=1)
        
        if v1 > 1e-16 and vq > 1e-16:
            vr = vq / (self.hurst_lag * v1)
            hurst = float(0.5 + np.log(vr) / (2.0 * np.log(float(self.hurst_lag))))
        else:
            hurst = 0.50

        # 5. Bollinger Bandwidth Percentile (BBWP)
        bbw_history = []
        for i in range(n - self.bbwp_window, n):
            window_slice = close[i - self.bb_period + 1:i + 1]
            mean_val = np.mean(window_slice)
            std_val = np.std(window_slice)
            bbw = (4.0 * std_val) / mean_val if mean_val > 1e-12 else 0.0
            bbw_history.append(bbw)
        curr_bbw = bbw_history[-1]
        bbwp = float(np.sum(np.array(bbw_history) <= curr_bbw) / float(self.bbwp_window) * 100.0)

        # 6. Spread Friction Ratio (SVP)
        pip_unit = 0.01 if "JPY" in symbol else 0.0001
        atr_pips = atr_fast_val / pip_unit
        svp = float(spread_pips / atr_pips) if atr_pips > 1e-12 else 0.0

        metrics = {
            "ker": round(ker, 4),
            "chop": round(chop, 2),
            "hurst": round(hurst, 4),
            "rvr": round(rvr, 3),
            "bbwp": round(bbwp, 1),
            "svp": round(svp, 4),
        }

        # --- LEVEL 1: UNIVERSAL MICROSTRUCTURE KILL-SWITCH ---
        if svp > 0.12:
            return False, f"KILL_EXCESSIVE_SPREAD_FRICTION (SVP={svp:.3f} > 0.12)", metrics

        # --- LEVEL 2: STRATEGY-SPECIFIC REGIME & CALENDAR GATING ---
        day_name = current_utc_dt.day_name()
        hour = current_utc_dt.hour

        # Strategy Slot 60: disc165 (GBPJPY London Breakout)
        if "disc165" in strategy_name:
            if day_name in ["Wednesday", "Thursday"]:
                return False, f"KILL_DISC165_MIDWEEK_CHOP (Day={day_name})", metrics
            if chop > 61.8 or bbwp > 80.0:
                return False, f"KILL_DISC165_UNFAVORABLE_BREAKOUT_REGIME", metrics

        # Strategy Slot 63: disc168 (EURNZD Trend Pullback)
        elif "disc168" in strategy_name:
            if hour >= 14:
                return False, f"KILL_DISC168_NY_SESSION_TOXIC_HOURS (Hour={hour} UTC)", metrics
            if day_name in ["Tuesday", "Wednesday"] and chop > 50.0:
                return False, f"KILL_DISC168_MIDWEEK_CHOP", metrics

        # Strategy Slot 50: disc106 (USDJPY Chaikin Volatility)
        elif "disc106" in strategy_name:
            if hour >= 16 or (day_name == "Monday" and hour < 6):
                return False, f"KILL_DISC106_LOW_MOMENTUM_DRIFT", metrics
            if ker < 0.25:
                return False, f"KILL_DISC106_LOW_KER (KER={ker:.3f} < 0.25)", metrics

        # Strategy Slot 62: disc167 (EURNZD Breakout)
        elif "disc167" in strategy_name:
            if (day_name == "Friday" and hour >= 16) or (day_name == "Monday" and hour < 6):
                return False, f"KILL_DISC167_OFF_HOURS", metrics

        # Strategy Slot 61: disc166 (GBPCHF Keltner Squeeze)
        elif "disc166" in strategy_name:
            if day_name in ["Monday", "Thursday"] and ker < 0.30:
                return False, f"KILL_DISC166_FAKEOUT_CHOP", metrics

        # Strategy Slot 55: disc94 (USDJPY London VWAP Pullback)
        elif "disc94" in strategy_name:
            if day_name in ["Monday", "Friday"] and chop > 52.0:
                return False, f"KILL_DISC94_OVERLAP_CHOP (CHOP={chop:.1f})", metrics

        # Strategy Slot 24/51: disc65 (Session VWAP Fade / Mean Reversion)
        elif "disc65" in strategy_name:
            if symbol == "EURUSD" and day_name == "Thursday":
                return False, f"KILL_DISC65_EURUSD_THURSDAY_EXPANSION", metrics
            if hurst > 0.55 or ker > 0.38:
                return False, f"KILL_DISC65_UNFAVORABLE_TRENDING_REGIME", metrics

        # General Trend Pullback Archetype Gate
        elif "pullback" in strategy_name.lower() or "retest" in strategy_name.lower():
            if ker < 0.30 or chop > 55.0 or hurst < 0.50:
                return False, f"KILL_TREND_ARCHETYPE_CHOP_GATE", metrics

        return True, "PERMITTED_TO_TRADE", metrics
