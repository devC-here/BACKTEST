#!/usr/bin/env python3
"""
================================================================================
  LIVE PRODUCTION MARKET REGIME FILTER & ARCHETYPE KILL-SWITCH ENGINE (v5.0)
================================================================================
Strictly causal 5-state regime filter wrapping MarketRegimeClassifier.
Supports both:
  1. Single-bar causal evaluation: `evaluate_live_permission(...)`
  2. Vectorized causal precompute + O(log N) lookup: `precompute_m15_regimes(...)`
     and `evaluate_at_timestamp(...)`

Strictly excludes USDJPY; calibrated on Discovery Weeks 1..210 across the
9 non-USDJPY currency pairs.
================================================================================
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd

_ROOT_DIR = Path(__file__).resolve().parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from tick_engine.quality.market_regime_classifier import (
    MarketRegimeClassifier,
    NON_USDJPY_SYMBOLS,
    REGIME_LABELS,
    SYMBOL_BASELINE_SPREAD_PIPS,
    SYMBOL_BASELINE_ATR14_PIPS,
    DEFAULT_REGIME_THRESHOLDS,
    ARCHETYPE_RULES,
    normalize_symbol,
    normalize_strategy_id,
    normalize_archetype,
    normalize_session_name,
    to_utc_timestamp,
    to_utc_ms,
    evaluate_permission_from_metrics,
)

# ==============================================================================
# NON-USDJPY PRODUCTION REGISTRY & EXACT ALIASES
# ==============================================================================
STRATEGY_REGIME_REGISTRY: Dict[str, Dict[str, Any]] = {
  "AUDCAD__strategy_01_audcad_h4_bb_ema_mr": {
    "name": "strategy_01_audcad_h4_bb_ema_mr",
    "symbol": "AUDCAD",
    "category": "audcad_v5",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1,
      2,
      5
    ],
    "prohibited_regimes": [
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Friday"
    ],
    "base_win_rate": 55.3,
    "base_profit_factor": 1.47,
    "base_net_pips": 627.6,
    "base_trades": 94
  },
  "EURUSD__strategy_02_eurusd_daily_pivot_ny": {
    "name": "strategy_02_eurusd_daily_pivot_ny",
    "symbol": "EURUSD",
    "category": "eurusd_v5",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1,
      2,
      5
    ],
    "prohibited_regimes": [
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 40.4,
    "base_profit_factor": 1.2,
    "base_net_pips": 153.6,
    "base_trades": 89
  },
  "GBPJPY__strategy_03_gbpjpy_h4_sqz_brk": {
    "name": "strategy_03_gbpjpy_h4_sqz_brk",
    "symbol": "GBPJPY",
    "category": "gbpjpy_v5",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1,
      2,
      5
    ],
    "prohibited_regimes": [
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 51.1,
    "base_profit_factor": 1.38,
    "base_net_pips": 658.4,
    "base_trades": 88
  },
  "USDCAD__strategy_04_usdcad_pdhl_sweep_ldn": {
    "name": "strategy_04_usdcad_pdhl_sweep_ldn",
    "symbol": "USDCAD",
    "category": "usdcad_v5",
    "archetype": "session_sweep",
    "permitted_regimes": [
      1,
      2,
      5
    ],
    "prohibited_regimes": [
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Friday"
    ],
    "base_win_rate": 42.1,
    "base_profit_factor": 1.36,
    "base_net_pips": 335.2,
    "base_trades": 95
  }
}

_UNIQUE_NAME_REGISTRY: Dict[str, Dict[str, Any]] = {
  "strategy_01_audcad_h4_bb_ema_mr": {
    "name": "strategy_01_audcad_h4_bb_ema_mr",
    "symbol": "AUDCAD",
    "category": "audcad_v5",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1,
      2,
      5
    ],
    "prohibited_regimes": [
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Friday"
    ],
    "base_win_rate": 55.3,
    "base_profit_factor": 1.47,
    "base_net_pips": 627.6,
    "base_trades": 94
  },
  "strategy_02_eurusd_daily_pivot_ny": {
    "name": "strategy_02_eurusd_daily_pivot_ny",
    "symbol": "EURUSD",
    "category": "eurusd_v5",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1,
      2,
      5
    ],
    "prohibited_regimes": [
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 40.4,
    "base_profit_factor": 1.2,
    "base_net_pips": 153.6,
    "base_trades": 89
  },
  "strategy_03_gbpjpy_h4_sqz_brk": {
    "name": "strategy_03_gbpjpy_h4_sqz_brk",
    "symbol": "GBPJPY",
    "category": "gbpjpy_v5",
    "archetype": "trend_pullback",
    "permitted_regimes": [
      1,
      2,
      5
    ],
    "prohibited_regimes": [
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [],
    "base_win_rate": 51.1,
    "base_profit_factor": 1.38,
    "base_net_pips": 658.4,
    "base_trades": 88
  },
  "strategy_04_usdcad_pdhl_sweep_ldn": {
    "name": "strategy_04_usdcad_pdhl_sweep_ldn",
    "symbol": "USDCAD",
    "category": "usdcad_v5",
    "archetype": "session_sweep",
    "permitted_regimes": [
      1,
      2,
      5
    ],
    "prohibited_regimes": [
      3,
      4
    ],
    "toxic_sessions": [],
    "toxic_days": [
      "Friday"
    ],
    "base_win_rate": 42.1,
    "base_profit_factor": 1.36,
    "base_net_pips": 335.2,
    "base_trades": 95
  }
}

STRATEGY_ALIASES: Dict[str, str] = {
  "v5audcadh4bbemamr": "AUDCAD__strategy_01_audcad_h4_bb_ema_mr",
  "eurusddailypivotnyv5": "EURUSD__strategy_02_eurusd_daily_pivot_ny",
  "v5gbpjpyh4sqzbrk": "GBPJPY__strategy_03_gbpjpy_h4_sqz_brk",
  "usdcadpdhlsweepldnv5": "USDCAD__strategy_04_usdcad_pdhl_sweep_ldn"
}


class LiveMarketRegimeFilter:
    """
    Causal 5-State Production Market Regime Engine & Archetype Kill-Switch.
    """

    def __init__(
        self,
        ker_period: int = 20,
        chop_period: int = 14,
        adx_period: int = 14,
        hurst_lag: int = 5,
        hurst_window: int = 100,
        atr_fast: int = 5,
        atr_mid: int = 14,
        atr_slow: int = 50,
        bb_period: int = 20,
        bbwp_window: int = 252,
        baseline_window: int = 288,
        baseline_min_periods: int = 96,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        self.classifier = MarketRegimeClassifier(
            ker_period=ker_period,
            chop_period=chop_period,
            adx_period=adx_period,
            hurst_lag=hurst_lag,
            hurst_window=hurst_window,
            atr_fast=atr_fast,
            atr_mid=atr_mid,
            atr_slow=atr_slow,
            bb_period=bb_period,
            bbwp_window=bbwp_window,
            baseline_window=baseline_window,
            baseline_min_periods=baseline_min_periods,
            thresholds=thresholds,
        )
        self.min_bars = self.classifier.min_bars

    def compute_metrics(
        self,
        symbol: str,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        spread_pips: float,
        current_utc_dt: Optional[Union[pd.Timestamp, int, float, str]] = None,
        spread_history: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Calculates all strictly causal M15 microstructure indicators."""
        return self.classifier.compute_single_bar_metrics(
            symbol=symbol,
            high=high,
            low=low,
            close=close,
            spread_pips=spread_pips,
            current_utc_dt=current_utc_dt,
            spread_history=spread_history,
        )

    def classify_regime(self, metrics: Dict[str, Any]) -> int:
        """Classifies metrics into 5-state taxonomy (1..5, or 0 warmup)."""
        return self.classifier.classify_regime(metrics)

    def precompute_m15_regimes(
        self,
        m15_bars: pd.DataFrame,
        symbol: str,
    ) -> Dict[str, Any]:
        """
        Vectorized O(N) strictly causal precomputation across closed M15 bars.
        `m15_bars` must be indexed by bar-close timestamp (`label='right'`).
        """
        if m15_bars is None or m15_bars.empty:
            return {"close_time_ms": np.empty(0, dtype=np.int64), "regime": np.empty(0, dtype=np.int8)}
        idx_ms = pd.DatetimeIndex(m15_bars.index).as_unit("ms").asi8
        spread_arr = m15_bars["spread_pips"].to_numpy(dtype=np.float64) if "spread_pips" in m15_bars.columns else None
        out = self.classifier.compute(
            high=m15_bars["high"].to_numpy(dtype=np.float64),
            low=m15_bars["low"].to_numpy(dtype=np.float64),
            close=m15_bars["close"].to_numpy(dtype=np.float64),
            spread=spread_arr,
            close_timestamps_ms=idx_ms,
            symbol=symbol,
        )
        out["close_time_ms"] = idx_ms
        return out

    def evaluate_at_timestamp(
        self,
        strategy_name: str,
        symbol: str,
        precomputed: Dict[str, Any],
        decision_ts_ms: Union[pd.Timestamp, int, float, str],
        spread_pips: Optional[float] = None,
        archetype: Optional[str] = None,
        strategy_id: Optional[str] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """O(log N) causal lookup on precomputed M15 bars at T_close <= decision_ts_ms."""
        close_ms = precomputed.get("close_time_ms")
        if close_ms is None or len(close_ms) == 0:
            return True, "WARMUP_PASS", {"regime": 0, "regime_label": REGIME_LABELS[0]}

        ts_utc = to_utc_timestamp(decision_ts_ms)
        ts_ms = to_utc_ms(ts_utc)
        idx = int(np.searchsorted(close_ms, ts_ms, side="right")) - 1
        if idx < 0 or int(precomputed["regime"][idx]) == 0:
            return True, "WARMUP_PASS", {"regime": 0, "regime_label": REGIME_LABELS[0]}

        sp = float(spread_pips) if spread_pips is not None else float(precomputed["spread_pips"][idx])
        atr14_pips = float(precomputed["atr14_pips"][idx])
        base_sp = SYMBOL_BASELINE_SPREAD_PIPS.get(normalize_symbol(symbol), 1.5)
        sp_ref = float(precomputed["spread_pips"][idx]) / max(float(precomputed["spread_ratio"][idx]), 1e-6)
        if np.isnan(sp_ref) or sp_ref <= 1e-6:
            sp_ref = base_sp

        metrics = {
            "ker": round(float(precomputed["ker"][idx]), 4),
            "er": round(float(precomputed["ker"][idx]), 4),
            "chop": round(float(precomputed["chop"][idx]), 2),
            "adx": round(float(precomputed["adx"][idx]), 2),
            "hurst": round(float(precomputed["hurst"][idx]), 4),
            "vr": round(float(precomputed["vr"][idx]), 4),
            "rvr": round(float(precomputed["rvr"][idx]), 3),
            "atr14_pips": round(atr14_pips, 3),
            "atr_rel": round(float(precomputed["atr_rel"][idx]), 3),
            "bb_width": round(float(precomputed["bb_width"][idx]), 6),
            "bbwp": round(float(precomputed["bbwp"][idx]), 1),
            "spread_pips": round(sp, 3),
            "spread_ratio": round(sp / sp_ref if sp_ref > 1e-12 else 1.0, 3),
            "svp": round(float(precomputed["svp"][idx]), 4),
            "svp14": round(sp / atr14_pips if atr14_pips > 1e-12 else 0.0, 4),
        }
        return evaluate_permission_from_metrics(
            classifier=self.classifier,
            strategy_name=strategy_name,
            symbol=symbol,
            metrics=metrics,
            current_utc_dt=ts_utc,
            registry=STRATEGY_REGIME_REGISTRY,
            unique_name_registry=_UNIQUE_NAME_REGISTRY,
            strategy_aliases=STRATEGY_ALIASES,
            archetype=archetype,
            strategy_id=strategy_id,
        )

    def evaluate_live_permission(
        self,
        strategy_name: str,
        symbol: str,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        spread_pips: float,
        current_utc_dt: Union[pd.Timestamp, int, float, str],
        *,
        archetype: Optional[str] = None,
        strategy_id: Optional[str] = None,
        spread_history: Optional[np.ndarray] = None,
        precomputed_metrics: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Evaluates incoming strategy signal against causal M15 bar microstructure.
        Returns: (permitted: bool, reason: str, metrics: dict)
        """
        if precomputed_metrics is not None:
            metrics = precomputed_metrics
        else:
            metrics = self.compute_metrics(
                symbol=symbol,
                high=high,
                low=low,
                close=close,
                spread_pips=spread_pips,
                current_utc_dt=current_utc_dt,
                spread_history=spread_history,
            )
        return evaluate_permission_from_metrics(
            classifier=self.classifier,
            strategy_name=strategy_name,
            symbol=symbol,
            metrics=metrics,
            current_utc_dt=current_utc_dt,
            registry=STRATEGY_REGIME_REGISTRY,
            unique_name_registry=_UNIQUE_NAME_REGISTRY,
            strategy_aliases=STRATEGY_ALIASES,
            archetype=archetype,
            strategy_id=strategy_id,
        )


# Export Alias for Drop-In Backward Compatibility
MarketRegimeSystem = LiveMarketRegimeFilter

def _make_balanced_synthetic_bars(n_bars: int = 360, base_price: float = 0.9050, pip_size: float = 0.0001, seed: int = 42):
    rng = np.random.default_rng(seed)
    # Realistic alternating gentle motion so final bars land cleanly in Regime 5 (Normal Balanced)
    step = rng.normal(0.0, 2.2 * pip_size, n_bars)
    closes = base_price + np.cumsum(step)
    # Construct last 25 bars as steady balanced normal candles (~5p range)
    for k in range(n_bars - 25, n_bars):
        drift = 1.6 * pip_size if (k % 2 == 0) else -1.1 * pip_size
        closes[k] = closes[k - 1] + drift
    half_rng = rng.uniform(1.8 * pip_size, 3.2 * pip_size, n_bars)
    highs = np.maximum(closes, np.roll(closes, 1)) + half_rng * 0.5
    lows = np.minimum(closes, np.roll(closes, 1)) - half_rng * 0.5
    highs[0] = closes[0] + 2.5 * pip_size
    lows[0] = closes[0] - 2.5 * pip_size
    return highs, lows, closes


if __name__ == "__main__":
    print("=" * 80)
    print("  MARKET REGIME & ARCHETYPE KILL-SWITCH SYSTEM (v5.0) - SELF-TESTS 1..4")
    print("=" * 80)
    print(f"Total Registered Non-USDJPY Strategies: {len(STRATEGY_REGIME_REGISTRY)}")
    assert all(p.get("symbol") != "USDJPY" for p in STRATEGY_REGIME_REGISTRY.values()), "USDJPY must be excluded!"

    filter_engine = LiveMarketRegimeFilter()

    # Realistic JPY-scale bars for GBPJPY / NZD tests
    h_jpy, l_jpy, c_jpy = _make_balanced_synthetic_bars(360, base_price=190.0, pip_size=0.01, seed=7)
    h_cross, l_cross, c_cross = _make_balanced_synthetic_bars(360, base_price=1.80, pip_size=0.0001, seed=11)

    print("\n[TEST 1] Tailored / Calendar Regime Check (AUDCAD strategy_01_audcad_h4_bb_ema_mr at 2026-11-27 08:15:00 UTC):")
    perm1, reason1, m1 = filter_engine.evaluate_live_permission(
        strategy_name="strategy_01_audcad_h4_bb_ema_mr",
        symbol="AUDCAD",
        high=h_jpy if "AUDCAD".endswith("JPY") else h_cross,
        low=l_jpy if "AUDCAD".endswith("JPY") else l_cross,
        close=c_jpy if "AUDCAD".endswith("JPY") else c_cross,
        spread_pips=1.8,
        current_utc_dt=pd.Timestamp("2026-11-27 08:15:00", tz="UTC"),
    )
    print(f"  Result: Permitted={perm1} | Reason={reason1} | Regime={m1.get('regime')}")
    assert perm1 is False, f"Validation failed: expected blocked, got {reason1}"

    print("\n[TEST 2] Harmonized Session Check (AUDCAD strategy_01_audcad_h4_bb_ema_mr at 2026-11-24 22:15:00 UTC):")
    perm2, reason2, m2 = filter_engine.evaluate_live_permission(
        strategy_name="strategy_01_audcad_h4_bb_ema_mr",
        symbol="AUDCAD",
        high=h_jpy if "AUDCAD".endswith("JPY") else h_cross,
        low=l_jpy if "AUDCAD".endswith("JPY") else l_cross,
        close=c_jpy if "AUDCAD".endswith("JPY") else c_cross,
        spread_pips=2.4,
        current_utc_dt=pd.Timestamp("2026-11-24 22:15:00", tz="UTC"),
    )
    print(f"  Result: Permitted={perm2} | Reason={reason2} | Regime={m2.get('regime')}")
    assert perm2 is False, f"Validation failed: expected session/rollover block, got {reason2}"

    print("\n[TEST 3] Realistic AUDCAD (~0.89-0.92) Normal Spread <=2.5p PASS vs >3x Median Spike BLOCK:")
    h_aud, l_aud, c_aud = _make_balanced_synthetic_bars(360, base_price=0.9050, pip_size=0.0001, seed=42)
    assert 0.88 <= float(np.median(c_aud)) <= 0.93, "AUDCAD prices must stay in realistic ~0.89-0.92 band!"
    perm3_ok, reason3_ok, m3_ok = filter_engine.evaluate_live_permission(
        strategy_name="strategy_01_audcad_h4_bb_ema_mr",
        symbol="AUDCAD",
        high=h_aud,
        low=l_aud,
        close=c_aud,
        spread_pips=2.0,
        current_utc_dt=pd.Timestamp("2026-11-24 14:00:00", tz="UTC"),
    )
    print(f"  3A (Normal 2.0p spread): Permitted={perm3_ok} | Reason={reason3_ok} | Regime={m3_ok.get('regime')} | SpreadRatio={m3_ok.get('spread_ratio')}")
    assert perm3_ok is True, f"Realistic AUDCAD 2.0p spread should be permitted, got {reason3_ok}"

    perm3_blk, reason3_blk, m3_blk = filter_engine.evaluate_live_permission(
        strategy_name="strategy_01_audcad_h4_bb_ema_mr",
        symbol="AUDCAD",
        high=h_aud,
        low=l_aud,
        close=c_aud,
        spread_pips=8.5,
        current_utc_dt=pd.Timestamp("2026-11-24 14:00:00", tz="UTC"),
    )
    print(f"  3B (Wide 8.5p spike >3x median): Permitted={perm3_blk} | Reason={reason3_blk} | Regime={m3_blk.get('regime')} | SpreadRatio={m3_blk.get('spread_ratio')}")
    assert perm3_blk is False and m3_blk.get("regime") == 4, f"Wide spread spike must trigger Regime 4 / block, got {reason3_blk}"

    print("\n[TEST 4] Real EURUSD Discovery M15 Causal Bars Regime Distribution Check:")
    from tick_engine.core.dataloader import get_week_dataset
    from tick_engine.core.causal_bars import build_causal_time_bars
    sample_weeks = [10, 30, 50, 75, 100, 125, 150, 175, 200]
    parts = []
    for w in sample_weeks:
        ticks, _ = get_week_dataset(w, "EURUSD", warmup_prev_weeks=0)
        bars = build_causal_time_bars(ticks, timeframe_min=15, pip_size=0.0001)
        if not bars.empty:
            parts.append(bars)
    eur_df = pd.concat(parts)
    eur_df = eur_df[~eur_df.index.duplicated(keep="last")].sort_index()
    if eur_df is not None and not eur_df.empty:
        pre = filter_engine.precompute_m15_regimes(eur_df, symbol="EURUSD")
        reg = pre["regime"]
        utc_hr = pre["utc_hours"]
        valid = reg > 0
        shares = {r: round(float(np.mean(reg[valid] == r) * 100.0), 2) for r in (1, 2, 3, 4, 5)}
        active_mask = valid & (utc_hr >= 6) & (utc_hr < 21)
        r34_all = float(np.mean((reg[valid] == 3) | (reg[valid] == 4)) * 100.0)
        r34_act = float(np.mean((reg[active_mask] == 3) | (reg[active_mask] == 4)) * 100.0)
        print(f"  EURUSD Discovery Bars (n={int(np.sum(valid))}): Shares={shares}")
        print(f"  Regime 3+4 All Hours: {r34_all:.2f}% | Active London/NY (06-21 UTC): {r34_act:.2f}%")
        assert shares[5] >= 25.0, f"Regime 5 must be active (>=25%), got {shares[5]}%"
        assert all(shares[r] >= 1.0 for r in (1, 2, 3, 4, 5)), f"All 5 regimes must be non-degenerate: {shares}"
        assert r34_act < 40.0, f"Active London/NY Regime 3+4 must be < 40%, got {r34_act:.2f}%"
        assert r34_all < 40.0, f"Overall Regime 3+4 must be < 40%, got {r34_all:.2f}%"

    print("\n" + "=" * 80)
    print("  ALL SELF-TESTS (1..4) PASSED! 5-STATE CAUSAL REGIME SYSTEM READY.")
    print("=" * 80)

