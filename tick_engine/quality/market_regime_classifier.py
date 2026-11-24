#!/usr/bin/env python3
"""
================================================================================
  PRODUCTION CAUSAL MARKET REGIME & KILL-SWITCH CLASSIFIER (v5.0)
================================================================================
Strictly causal 5-state FX market microstructure classifier calibrated empirically
on Discovery Weeks 1..210 causal M15 bars across all 9 non-USDJPY currency pairs
(AUDCAD, AUDNZD, EURGBP, EURNZD, EURUSD, GBPCHF, GBPJPY, NZDCAD, USDCAD).

5-State Canonical Taxonomy (+ 0 Warmup):
  0 = Warmup / Uninitialized (insufficient causal closed M15 bars)
  1 = Strong Clean Trend (high ER + strong ADX + low CHOP)
  2 = Calm Mean-Reversion (low ADX + low ER + bounded volatility)
  3 = Volatile / High-ATR Chop (high-vol expansion + low directional efficiency)
  4 = Dead / Low Liquidity (wide spread >3x median, NY rollover, or dead ATR tail)
  5 = Normal Balanced Market (healthy baseline / transitional market flow)

All indicators (ATR, ADX, ER/KER, CHOP, Hurst, BBWidth/BBWP, SpreadRatio, SVP)
are 100% causal on closed M15 bars up to index t (zero lookahead, zero bfill).
================================================================================
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, Union, Iterable
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]

        def decorator(func):
            return func

        return decorator


# ==============================================================================
# CANONICAL CONSTANTS & DISCOVERY-CALIBRATED PARAMETERS (WEEKS 1..210)
# ==============================================================================
NON_USDJPY_SYMBOLS = (
    "AUDCAD", "AUDNZD", "EURGBP", "EURNZD", "EURUSD",
    "GBPCHF", "GBPJPY", "NZDCAD", "USDCAD",
)

REGIME_LABELS: Dict[int, str] = {
    0: "Warmup / Uninitialized",
    1: "Strong Clean Trend",
    2: "Calm Mean-Reversion",
    3: "Volatile / High-ATR Chop",
    4: "Dead / Low Liquidity",
    5: "Normal Balanced Market",
}

# Empirically calibrated Discovery Weeks 1..210 median M15 bar spreads (pips)
SYMBOL_BASELINE_SPREAD_PIPS: Dict[str, float] = {
    "AUDCAD": 1.9831,
    "AUDNZD": 2.0607,
    "EURGBP": 0.8739,
    "EURNZD": 3.2468,
    "EURUSD": 0.2858,
    "GBPCHF": 1.9743,
    "GBPJPY": 2.0262,
    "NZDCAD": 2.3906,
    "USDCAD": 1.2142,
}

# Empirically calibrated Discovery Weeks 1..210 median M15 ATR(14) (pips)
SYMBOL_BASELINE_ATR14_PIPS: Dict[str, float] = {
    "AUDCAD": 5.860,
    "AUDNZD": 5.182,
    "EURGBP": 3.994,
    "EURNZD": 11.983,
    "EURUSD": 5.944,
    "GBPCHF": 7.131,
    "GBPJPY": 12.943,
    "NZDCAD": 5.792,
    "USDCAD": 6.792,
}

# Calibrated on 9 non-USDJPY pairs across ~883k Discovery M15 causal bars:
# Mean Regime shares: R1=16.9%, R2=16.5%, R3=7.8%, R4=9.6%, R5=49.1%
# Mean adverse (R3+R4): 17.5% all hours, 13.3% active London/NY hours
DEFAULT_REGIME_THRESHOLDS: Dict[str, float] = {
    "r4_max_spread_ratio": 3.00,   # Wide spread > 3.0x causal median guard
    "r4_max_svp14": 1.20,          # Spread > 120% of 14-bar ATR
    "r4_dead_atr_rel": 0.68,       # Dead ATR tail (< 0.68x 3-day causal median)
    "r4_dead_bbwp": 20.0,          # Coupled with low BBWP (< 20th pct)
    "r3_min_atr_rel": 1.15,        # High-ATR expansion (> 1.15x median ATR14)
    "r3_min_rvr": 1.20,            # Or fast/slow ATR surge (> 1.20)
    "r3_max_er": 0.20,             # Low directional efficiency
    "r3_min_chop": 54.0,           # High chop whipsaw
    "r1_min_er": 0.26,             # Clean directional ER
    "r1_min_adx": 23.0,            # Strong trend ADX(14)
    "r1_max_chop": 51.0,           # Low fractal chop
    "r2_max_adx": 21.0,            # Non-trending ADX
    "r2_max_er": 0.20,             # Low net drift
    "r2_max_atr_rel": 1.15,        # Calm volatility
    "r2_max_hurst": 0.46,          # Anti-persistent Hurst
    "r2_min_chop": 53.0,           # Or bounded chop oscillation
}

ARCHETYPE_RULES: Dict[str, Dict[str, Any]] = {
    "trend_pullback": {
        "permitted_regimes": [1, 2, 5],
        "prohibited_regimes": [3, 4],
    },
    "mean_reversion": {
        "permitted_regimes": [2, 5],
        "prohibited_regimes": [1, 3, 4],
    },
    "volatility_breakout": {
        "permitted_regimes": [1, 2, 5],
        "prohibited_regimes": [3, 4],
    },
    "macro_structure_retest": {
        "permitted_regimes": [1, 2, 5],
        "prohibited_regimes": [3, 4],
    },
    "session_sweep": {
        "permitted_regimes": [1, 2, 5],
        "prohibited_regimes": [3, 4],
    },
    "balanced": {
        "permitted_regimes": [1, 2, 5],
        "prohibited_regimes": [3, 4],
    },
}

ARCHETYPE_ALIASES: Dict[str, str] = {
    "trend_pullback": "trend_pullback",
    "trend": "trend_pullback",
    "pullback": "trend_pullback",
    "momentum_trend": "trend_pullback",
    "vwap_retest": "trend_pullback",
    "mean_reversion": "mean_reversion",
    "fade": "mean_reversion",
    "vwap_fade": "mean_reversion",
    "pivot_bounce": "mean_reversion",
    "volatility_breakout": "volatility_breakout",
    "breakout": "volatility_breakout",
    "squeeze": "volatility_breakout",
    "expansion": "volatility_breakout",
    "orb": "volatility_breakout",
    "macro_structure_retest": "macro_structure_retest",
    "macro_retest": "macro_structure_retest",
    "macro_sr_retest": "macro_structure_retest",
    "structure_retest": "macro_structure_retest",
    "session_sweep": "session_sweep",
    "sweep": "session_sweep",
    "liquidity_sweep": "session_sweep",
    "balanced": "balanced",
}

CANONICAL_SESSION_WINDOWS: Dict[str, str] = {
    "ASIAN": "Asian (21:00-06:00 UTC)",
    "LONDON": "London Open (06:00-12:00 UTC)",
    "LONDON_NY_OVERLAP": "London/NY Overlap (12:00-16:00 UTC)",
    "NY_AFTERNOON": "NY Afternoon (16:00-21:00 UTC)",
    "NY_SESSION": "NY Session (13:00-21:00 UTC)",
    "NY_ROLLOVER": "NY Close Rollover (17:00-17:59 America/New_York)",
    "LONDON_DST": "London Regular (08:00-16:00 Europe/London)",
    "NEW_YORK_DST": "New York Regular (08:00-17:00 America/New_York)",
    "TOKYO_DST": "Tokyo Regular (09:00-15:00 Asia/Tokyo)",
}

SESSION_NAME_ALIASES: Dict[str, str] = {
    "asian": "ASIAN",
    "asian (21:00-06:00)": "ASIAN",
    "asian (21:00-06:00 utc)": "ASIAN",
    "asia": "ASIAN",
    "london": "LONDON",
    "london open (06:00-12:00)": "LONDON",
    "london open (06:00-12:00 utc)": "LONDON",
    "london morning (06:00-12:00)": "LONDON",
    "london_open": "LONDON",
    "london/ny overlap (12:00-16:00)": "LONDON_NY_OVERLAP",
    "london/ny overlap (12:00-16:00 utc)": "LONDON_NY_OVERLAP",
    "london_ny_overlap": "LONDON_NY_OVERLAP",
    "overlap": "LONDON_NY_OVERLAP",
    "ny afternoon (16:00-21:00)": "NY_AFTERNOON",
    "ny afternoon (16:00-21:00 utc)": "NY_AFTERNOON",
    "ny_afternoon": "NY_AFTERNOON",
    "ny session (13:00-21:00)": "NY_SESSION",
    "ny_session": "NY_SESSION",
    "ny_rollover": "NY_ROLLOVER",
    "rollover": "NY_ROLLOVER",
    "london_dst": "LONDON_DST",
    "new_york_dst": "NEW_YORK_DST",
    "tokyo_dst": "TOKYO_DST",
}

_TZ_NY = ZoneInfo("America/New_York")
_TZ_LDN = ZoneInfo("Europe/London")
_TZ_TKY = ZoneInfo("Asia/Tokyo")
_TZ_UTC = ZoneInfo("UTC")


# ==============================================================================
# EXACT NORMALIZATION & SESSION CLOCK HELPERS (ZERO SUBSTRING MATCHING)
# ==============================================================================
def normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def normalize_strategy_id(strategy_name: str, symbol: Optional[str] = None) -> str:
    """Normalizes a strategy ID or manifest filename stem without substring matching."""
    sid = str(strategy_name or "").strip()
    for suffix in (
        "_holdout_trades.csv",
        "_discovery_trades.csv",
        "_trades.csv",
        "_run_manifest.json",
        "_holdout_manifest.json",
        "_discovery_manifest.json",
        ".py",
    ):
        if sid.lower().endswith(suffix):
            sid = sid[: -len(suffix)]
            break
    sid_lower = sid.lower()
    if symbol:
        sym_l = normalize_symbol(symbol).lower()
        if sid_lower.startswith(f"{sym_l}__"):
            sid_lower = sid_lower[len(sym_l) + 2:]
        elif sid_lower.startswith(f"{sym_l}_"):
            sid_lower = sid_lower[len(sym_l) + 1:]
    return sid_lower


def make_registry_key(symbol: str, strategy_id: str) -> str:
    sym = normalize_symbol(symbol)
    sid = normalize_strategy_id(strategy_id, symbol=sym)
    return f"{sym}__{sid}"


def normalize_archetype(archetype: Optional[str]) -> Optional[str]:
    if not archetype:
        return None
    key = str(archetype).strip().lower().replace(" ", "_").replace("-", "_")
    return ARCHETYPE_ALIASES.get(key)


def normalize_session_name(session_name: str) -> str:
    s = str(session_name or "").strip().lower()
    if s in SESSION_NAME_ALIASES:
        return SESSION_NAME_ALIASES[s]
    s_upper = str(session_name or "").strip().upper()
    if s_upper in CANONICAL_SESSION_WINDOWS:
        return s_upper
    raise ValueError(f"Unknown session identifier (refusing unsafe substring match): {session_name!r}")


def to_utc_timestamp(ts_input: Union[pd.Timestamp, int, float, str]) -> pd.Timestamp:
    """Converts tz-naive UTC, tz-aware, ISO string, or epoch-ms to UTC pd.Timestamp."""
    if isinstance(ts_input, (int, np.integer, float, np.floating)):
        return pd.to_datetime(int(ts_input), unit="ms", utc=True)
    ts = pd.Timestamp(ts_input)
    if ts.tzinfo is None:
        return ts.tz_localize(_TZ_UTC)
    return ts.tz_convert(_TZ_UTC)


def to_utc_ms(ts_input: Union[pd.Timestamp, int, float, str]) -> int:
    """Converts tz-naive UTC, tz-aware, ISO string, or epoch-ms to exact UTC epoch milliseconds."""
    if isinstance(ts_input, (int, np.integer)):
        return int(ts_input)
    if isinstance(ts_input, (float, np.floating)):
        return int(ts_input)
    ts = to_utc_timestamp(ts_input)
    return int(round(ts.timestamp() * 1000.0))


def classify_utc_session(utc_hour: int) -> str:
    if utc_hour >= 21 or utc_hour < 6:
        return "ASIAN"
    if 6 <= utc_hour < 12:
        return "LONDON"
    if 12 <= utc_hour < 16:
        return "LONDON_NY_OVERLAP"
    return "NY_AFTERNOON"


def get_session_clock_hours(ts_utc: pd.Timestamp) -> Dict[str, int]:
    """Returns UTC, New York (EDT/EST), London (BST/GMT), and Tokyo (JST) hours."""
    return {
        "utc_hour": int(ts_utc.hour),
        "ny_hour": int(ts_utc.tz_convert(_TZ_NY).hour),
        "london_hour": int(ts_utc.tz_convert(_TZ_LDN).hour),
        "tokyo_hour": int(ts_utc.tz_convert(_TZ_TKY).hour),
    }


def evaluate_session_match(session_key: str, clock_hours: Dict[str, int]) -> bool:
    """Exact deterministic session matching using harmonized UTC + SessionClock hours."""
    canon = normalize_session_name(session_key)
    u = clock_hours["utc_hour"]
    ny = clock_hours["ny_hour"]
    ldn = clock_hours["london_hour"]
    tky = clock_hours["tokyo_hour"]
    if canon == "ASIAN":
        return u >= 21 or u < 6
    if canon == "LONDON":
        return 6 <= u < 12
    if canon == "LONDON_NY_OVERLAP":
        return 12 <= u < 16
    if canon == "NY_AFTERNOON":
        return 16 <= u < 21
    if canon == "NY_SESSION":
        return 13 <= u < 21
    if canon == "NY_ROLLOVER":
        return ny == 17
    if canon == "LONDON_DST":
        return 8 <= ldn < 16
    if canon == "NEW_YORK_DST":
        return 8 <= ny < 17
    if canon == "TOKYO_DST":
        return 9 <= tky < 15
    return False


# ==============================================================================
# CAUSAL NUMBA / VECTORIZED INDICATOR KERNELS (ZERO LOOKAHEAD)
# ==============================================================================
@njit(fastmath=True)
def _compute_tr_and_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int):
    n = len(close)
    tr = np.empty(n, dtype=np.float64)
    atr = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return tr, atr

    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, max(hc, lc))

    if n >= period:
        acc = 0.0
        for i in range(period):
            acc += tr[i]
        atr[period - 1] = acc / period
        alpha = 1.0 / period
        for i in range(period, n):
            atr[i] = atr[i - 1] + alpha * (tr[i] - atr[i - 1])

    return tr, atr


@njit(fastmath=True)
def _compute_wilder_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, tr: np.ndarray, period: int = 14):
    n = len(close)
    adx = np.full(n, np.nan, dtype=np.float64)
    if n < 2 * period + 1:
        return adx

    pdm = np.zeros(n, dtype=np.float64)
    ndm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        if up > dn and up > 0.0:
            pdm[i] = up
        if dn > up and dn > 0.0:
            ndm[i] = dn

    dx = np.full(n, np.nan, dtype=np.float64)
    sm_tr = 0.0
    sm_pdm = 0.0
    sm_ndm = 0.0
    for i in range(1, period + 1):
        sm_tr += tr[i]
        sm_pdm += pdm[i]
        sm_ndm += ndm[i]

    for i in range(period, n):
        if i > period:
            sm_tr = sm_tr - (sm_tr / period) + tr[i]
            sm_pdm = sm_pdm - (sm_pdm / period) + pdm[i]
            sm_ndm = sm_ndm - (sm_ndm / period) + ndm[i]
        if sm_tr > 1e-12:
            pdi = 100.0 * sm_pdm / sm_tr
            ndi = 100.0 * sm_ndm / sm_tr
            s = pdi + ndi
            dx[i] = (100.0 * abs(pdi - ndi) / s) if s > 1e-12 else 0.0
        else:
            dx[i] = 0.0

    start_adx = 2 * period - 1
    if start_adx < n:
        acc = 0.0
        for i in range(period, start_adx + 1):
            acc += dx[i]
        adx[start_adx] = acc / period
        for i in range(start_adx + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return adx


@njit(fastmath=True)
def _compute_ker_and_chop(high: np.ndarray, low: np.ndarray, close: np.ndarray, tr: np.ndarray, n_ker: int, n_chop: int):
    n = len(close)
    ker = np.full(n, np.nan, dtype=np.float64)
    chop = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return ker, chop

    abs_diff = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        abs_diff[i] = abs(close[i] - close[i - 1])

    for i in range(n_ker, n):
        net_change = abs(close[i] - close[i - n_ker])
        path_sum = 0.0
        for j in range(i - n_ker + 1, i + 1):
            path_sum += abs_diff[j]
        ker[i] = net_change / path_sum if path_sum > 1e-12 else 0.0

    log10_n = np.log10(float(n_chop))
    for i in range(n_chop - 1, n):
        tr_sum = 0.0
        max_h = -1e18
        min_l = 1e18
        for j in range(i - n_chop + 1, i + 1):
            tr_sum += tr[j]
            if high[j] > max_h:
                max_h = high[j]
            if low[j] < min_l:
                min_l = low[j]
        span = max_h - min_l
        if span > 1e-12 and tr_sum > 1e-12:
            ratio = tr_sum / span
            chop[i] = 100.0 * np.log10(ratio if ratio > 1e-12 else 1e-12) / log10_n
        else:
            chop[i] = 50.0

    return ker, chop


@njit(fastmath=True)
def _compute_local_hurst_vr(close: np.ndarray, lag_q: int, window: int):
    n = len(close)
    hurst = np.full(n, np.nan, dtype=np.float64)
    vr_arr = np.full(n, np.nan, dtype=np.float64)
    if n <= window + lag_q:
        return hurst, vr_arr

    log_p = np.log(np.maximum(close, 1e-12))
    ret1 = np.zeros(n, dtype=np.float64)
    retq = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        ret1[i] = log_p[i] - log_p[i - 1]
    for i in range(lag_q, n):
        retq[i] = log_p[i] - log_p[i - lag_q]

    inv_2lnq = 1.0 / (2.0 * np.log(float(lag_q)))

    for i in range(window + lag_q, n):
        m1 = 0.0
        for j in range(i - window + 1, i + 1):
            m1 += ret1[j]
        m1 /= window
        v1 = 0.0
        for j in range(i - window + 1, i + 1):
            d = ret1[j] - m1
            v1 += d * d
        v1 /= (window - 1)

        mq = 0.0
        for j in range(i - window + 1, i + 1):
            mq += retq[j]
        mq /= window
        vq = 0.0
        for j in range(i - window + 1, i + 1):
            d = retq[j] - mq
            vq += d * d
        vq /= (window - 1)

        if v1 > 1e-16 and vq > 1e-16:
            vr = vq / (lag_q * v1)
            vr_arr[i] = vr
            hurst[i] = 0.5 + np.log(vr) * inv_2lnq

    return hurst, vr_arr


@njit(fastmath=True)
def _compute_bbw_and_bbwp(close: np.ndarray, bb_period: int, pct_window: int):
    n = len(close)
    bbw = np.full(n, np.nan, dtype=np.float64)
    bbwp = np.full(n, np.nan, dtype=np.float64)
    if n < bb_period:
        return bbw, bbwp

    for i in range(bb_period - 1, n):
        mean = 0.0
        for j in range(i - bb_period + 1, i + 1):
            mean += close[j]
        mean /= bb_period

        var = 0.0
        for j in range(i - bb_period + 1, i + 1):
            d = close[j] - mean
            var += d * d
        std = np.sqrt(var / bb_period)
        bbw[i] = (4.0 * std) / mean if mean > 1e-12 else 0.0

    start_idx = bb_period - 1 + pct_window - 1
    for i in range(start_idx, n):
        curr = bbw[i]
        count = 0
        for j in range(i - pct_window + 1, i + 1):
            if bbw[j] <= curr:
                count += 1
        bbwp[i] = (count / float(pct_window)) * 100.0

    return bbw, bbwp


def _causal_rolling_median(arr: np.ndarray, window: int = 288, min_periods: int = 96) -> np.ndarray:
    return pd.Series(arr, dtype=np.float64).rolling(window=window, min_periods=min_periods).median().to_numpy()


# ==============================================================================
# CLASSIFIER & PERMISSION ENGINE
# ==============================================================================
class MarketRegimeClassifier:
    """
    Production Causal 5-State Market Regime & Archetype Kill-Switch Engine.
    Guarantees identical strictly causal mathematics across vectorized array evaluation
    and single-bar live evaluation.
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
        self.ker_period = ker_period
        self.chop_period = chop_period
        self.adx_period = adx_period
        self.hurst_lag = hurst_lag
        self.hurst_window = hurst_window
        self.atr_fast = atr_fast
        self.atr_mid = atr_mid
        self.atr_slow = atr_slow
        self.bb_period = bb_period
        self.bbwp_window = bbwp_window
        self.baseline_window = baseline_window
        self.baseline_min_periods = baseline_min_periods
        self.thresholds = dict(DEFAULT_REGIME_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)

        self.min_bars = max(
            self.hurst_window + self.hurst_lag + 1,
            self.bbwp_window + self.bb_period - 1,
            self.atr_mid + self.baseline_min_periods - 1,
            2 * self.adx_period + 1,
            self.atr_slow,
        )

    @staticmethod
    def _pip_unit(symbol: Optional[str]) -> float:
        s = normalize_symbol(symbol or "")
        return 0.01 if s.endswith("JPY") else 0.0001

    def compute(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        spread: Optional[np.ndarray] = None,
        close_timestamps_ms: Optional[np.ndarray] = None,
        symbol: Optional[str] = None,
        bar_minutes: int = 15,
    ) -> Dict[str, Any]:
        """Vectorized strictly causal calculation across closed M15 bars."""
        h = np.ascontiguousarray(high, dtype=np.float64)
        l = np.ascontiguousarray(low, dtype=np.float64)
        c = np.ascontiguousarray(close, dtype=np.float64)
        n = len(c)
        sym = normalize_symbol(symbol or "EURUSD")
        pip_unit = self._pip_unit(sym)

        if spread is not None:
            sp = np.ascontiguousarray(spread, dtype=np.float64)
        else:
            sp = np.full(n, SYMBOL_BASELINE_SPREAD_PIPS.get(sym, 1.5), dtype=np.float64)

        tr, atr_fast = _compute_tr_and_atr(h, l, c, self.atr_fast)
        _, atr14 = _compute_tr_and_atr(h, l, c, self.atr_mid)
        _, atr_slow = _compute_tr_and_atr(h, l, c, self.atr_slow)
        adx = _compute_wilder_adx(h, l, c, tr, self.adx_period)
        ker, chop = _compute_ker_and_chop(h, l, c, tr, self.ker_period, self.chop_period)
        hurst, vr = _compute_local_hurst_vr(c, self.hurst_lag, self.hurst_window)
        bbw, bbwp = _compute_bbw_and_bbwp(c, self.bb_period, self.bbwp_window)

        rvr = np.divide(atr_fast, atr_slow, out=np.full_like(atr_fast, np.nan), where=atr_slow > 1e-12)
        atr5_pips = atr_fast / pip_unit
        atr14_pips = atr14 / pip_unit
        svp = np.divide(sp, atr5_pips, out=np.full_like(sp, np.nan), where=atr5_pips > 1e-12)
        svp14 = np.divide(sp, atr14_pips, out=np.full_like(sp, np.nan), where=atr14_pips > 1e-12)

        atr14_med = _causal_rolling_median(atr14, self.baseline_window, self.baseline_min_periods)
        atr_rel = np.divide(atr14, atr14_med, out=np.full_like(atr14, np.nan), where=atr14_med > 1e-12)

        if spread is not None:
            sp_med = _causal_rolling_median(sp, self.baseline_window, self.baseline_min_periods)
            floor_sp = 0.25 * SYMBOL_BASELINE_SPREAD_PIPS.get(sym, 1.0)
            sp_ref = np.where(np.isnan(sp_med), SYMBOL_BASELINE_SPREAD_PIPS.get(sym, 1.0), np.maximum(sp_med, floor_sp))
        else:
            sp_ref = np.full(n, SYMBOL_BASELINE_SPREAD_PIPS.get(sym, 1.5), dtype=np.float64)
        spread_ratio = np.divide(sp, sp_ref, out=np.ones_like(sp), where=sp_ref > 1e-12)

        is_rollover = np.zeros(n, dtype=bool)
        utc_hours = np.full(n, -1, dtype=np.int16)
        ny_hours = np.full(n, -1, dtype=np.int16)
        if close_timestamps_ms is not None and len(close_timestamps_ms) == n:
            close_dt = pd.to_datetime(close_timestamps_ms, unit="ms", utc=True)
            open_dt = close_dt - pd.Timedelta(minutes=bar_minutes)
            utc_hours = open_dt.hour.to_numpy(dtype=np.int16)
            ny_hours = open_dt.tz_convert(_TZ_NY).hour.to_numpy(dtype=np.int16)
            is_rollover = ny_hours == 17

        regimes = self._classify_regimes_vec(
            ker=ker,
            chop=chop,
            adx=adx,
            hurst=hurst,
            rvr=rvr,
            atr_rel=atr_rel,
            bbwp=bbwp,
            spread_ratio=spread_ratio,
            svp14=svp14,
            is_rollover=is_rollover,
        )
        gates = self._compute_archetype_gates(regimes)

        return {
            "ker": ker,
            "er": ker,
            "chop": chop,
            "adx": adx,
            "hurst": hurst,
            "vr": vr,
            "rvr": rvr,
            "atr14": atr14,
            "atr14_pips": atr14_pips,
            "atr_rel": atr_rel,
            "bb_width": bbw,
            "bbwp": bbwp,
            "spread_pips": sp,
            "spread_ratio": spread_ratio,
            "svp": svp,
            "svp14": svp14,
            "utc_hours": utc_hours,
            "ny_hours": ny_hours,
            "is_rollover": is_rollover,
            "regime": regimes,
            "gates": gates,
        }

    def _classify_regimes_vec(
        self,
        ker: np.ndarray,
        chop: np.ndarray,
        adx: np.ndarray,
        hurst: np.ndarray,
        rvr: np.ndarray,
        atr_rel: np.ndarray,
        bbwp: np.ndarray,
        spread_ratio: np.ndarray,
        svp14: np.ndarray,
        is_rollover: np.ndarray,
    ) -> np.ndarray:
        t = self.thresholds
        n = len(ker)
        regime = np.zeros(n, dtype=np.int8)

        valid = (
            ~np.isnan(bbwp)
            & ~np.isnan(hurst)
            & ~np.isnan(adx)
            & ~np.isnan(atr_rel)
            & ~np.isnan(ker)
            & ~np.isnan(chop)
            & ~np.isnan(rvr)
        )

        # Priority 1: Regime 4 — Dead / Low Liquidity / Wide Spread > 3x median
        r4 = valid & (
            (spread_ratio > t["r4_max_spread_ratio"])
            | (svp14 > t["r4_max_svp14"])
            | is_rollover
            | ((atr_rel < t["r4_dead_atr_rel"]) & (bbwp < t["r4_dead_bbwp"]))
        )

        # Priority 2: Regime 3 — Volatile / High-ATR Chop
        r3 = valid & (~r4) & (
            ((atr_rel > t["r3_min_atr_rel"]) | (rvr > t["r3_min_rvr"]))
            & (ker < t["r3_max_er"])
            & (chop > t["r3_min_chop"])
        )

        # Priority 3: Regime 1 — Strong Clean Trend
        r1 = valid & (~r4) & (~r3) & (
            (ker >= t["r1_min_er"])
            & (adx >= t["r1_min_adx"])
            & (chop < t["r1_max_chop"])
        )

        # Priority 4: Regime 2 — Calm Mean-Reversion
        r2 = valid & (~r4) & (~r3) & (~r1) & (
            (adx < t["r2_max_adx"])
            & (ker < t["r2_max_er"])
            & (atr_rel <= t["r2_max_atr_rel"])
            & ((hurst < t["r2_max_hurst"]) | (chop >= t["r2_min_chop"]))
        )

        # Priority 5: Regime 5 — Normal Balanced Market (healthy default!)
        regime[valid] = 5
        regime[r2] = 2
        regime[r1] = 1
        regime[r3] = 3
        regime[r4] = 4
        return regime

    def classify_regime(self, metrics: Dict[str, Any]) -> int:
        """Classifies single-bar metric dict into 5-state taxonomy (1..5, or 0 warmup)."""
        if not metrics or metrics.get("warmup", False):
            return 0
        t = self.thresholds
        ker = float(metrics.get("er", metrics.get("ker", 0.0)))
        chop = float(metrics.get("chop", 50.0))
        adx = float(metrics.get("adx", 20.0))
        hurst = float(metrics.get("hurst", 0.45))
        rvr = float(metrics.get("rvr", 1.0))
        atr_rel = float(metrics.get("atr_rel", 1.0))
        bbwp = float(metrics.get("bbwp", 50.0))
        spread_ratio = float(metrics.get("spread_ratio", 1.0))
        svp14 = float(metrics.get("svp14", 0.0))
        is_rollover = bool(metrics.get("is_rollover", False))

        if (
            spread_ratio > t["r4_max_spread_ratio"]
            or svp14 > t["r4_max_svp14"]
            or is_rollover
            or (atr_rel < t["r4_dead_atr_rel"] and bbwp < t["r4_dead_bbwp"])
        ):
            return 4

        if (
            (atr_rel > t["r3_min_atr_rel"] or rvr > t["r3_min_rvr"])
            and ker < t["r3_max_er"]
            and chop > t["r3_min_chop"]
        ):
            return 3

        if ker >= t["r1_min_er"] and adx >= t["r1_min_adx"] and chop < t["r1_max_chop"]:
            return 1

        if (
            adx < t["r2_max_adx"]
            and ker < t["r2_max_er"]
            and atr_rel <= t["r2_max_atr_rel"]
            and (hurst < t["r2_max_hurst"] or chop >= t["r2_min_chop"])
        ):
            return 2

        return 5

    @staticmethod
    def _compute_archetype_gates(regimes: np.ndarray) -> Dict[str, np.ndarray]:
        """Returns boolean masks: True = PERMITTED, False = BLOCKED."""
        valid = regimes > 0
        non_adverse = valid & (regimes != 3) & (regimes != 4)
        return {
            "trend_pullback": non_adverse,
            "mean_reversion": valid & ((regimes == 2) | (regimes == 5)),
            "volatility_breakout": non_adverse,
            "macro_retest": non_adverse,
            "macro_structure_retest": non_adverse,
            "session_sweep": non_adverse,
            "balanced": non_adverse,
        }

    def compute_single_bar_metrics(
        self,
        symbol: str,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        spread_pips: float,
        current_utc_dt: Optional[Union[pd.Timestamp, int, float, str]] = None,
        spread_history: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Computes strictly causal metrics for the latest closed M15 bar.
        Bounds tail length to 1,000 bars (RMA seed decay < 1e-8) for sub-ms live calls.
        """
        n = len(close)
        if n < self.min_bars:
            return {}

        tail = min(n, 1000)
        h = np.ascontiguousarray(high[-tail:], dtype=np.float64)
        l = np.ascontiguousarray(low[-tail:], dtype=np.float64)
        c = np.ascontiguousarray(close[-tail:], dtype=np.float64)
        sym = normalize_symbol(symbol)
        pip_unit = self._pip_unit(sym)

        tr, atr5 = _compute_tr_and_atr(h, l, c, self.atr_fast)
        _, atr14 = _compute_tr_and_atr(h, l, c, self.atr_mid)
        _, atr50 = _compute_tr_and_atr(h, l, c, self.atr_slow)
        adx14 = _compute_wilder_adx(h, l, c, tr, self.adx_period)

        atr5_val = float(atr5[-1])
        atr14_val = float(atr14[-1])
        atr50_val = float(atr50[-1])
        adx_val = float(adx14[-1])
        if np.isnan(atr14_val) or np.isnan(atr50_val) or np.isnan(adx_val):
            return {}

        # Causal ATR baseline median over trailing closed bars
        valid_atr14 = atr14[~np.isnan(atr14)]
        if len(valid_atr14) < self.baseline_min_periods:
            return {}
        atr14_med = float(np.median(valid_atr14[-self.baseline_window:]))
        atr_rel = float(atr14_val / atr14_med) if atr14_med > 1e-12 else 1.0
        rvr = float(atr5_val / atr50_val) if atr50_val > 1e-12 else 1.0

        # ER / KER over last ker_period bars
        net_change = abs(float(c[-1] - c[-self.ker_period - 1]))
        path_sum = float(np.sum(np.abs(np.diff(c[-(self.ker_period + 1):]))))
        ker = float(net_change / path_sum) if path_sum > 1e-12 else 0.0

        # CHOP over last chop_period bars
        tr_sum = float(np.sum(tr[-self.chop_period:]))
        span = float(np.max(h[-self.chop_period:]) - np.min(l[-self.chop_period:]))
        log10_n = np.log10(float(self.chop_period))
        chop = float(100.0 * np.log10(max(tr_sum / span, 1e-12)) / log10_n) if span > 1e-12 and tr_sum > 1e-12 else 50.0

        # Variance Ratio Hurst Exponent
        log_p = np.log(np.maximum(c[-(self.hurst_window + self.hurst_lag + 1):], 1e-12))
        ret1 = np.diff(log_p)
        retq = log_p[self.hurst_lag:] - log_p[:-self.hurst_lag]
        v1 = float(np.var(ret1[-self.hurst_window:], ddof=1))
        vq = float(np.var(retq[-self.hurst_window:], ddof=1))
        if v1 > 1e-16 and vq > 1e-16:
            vr = float(vq / (self.hurst_lag * v1))
            hurst = float(0.5 + np.log(vr) / (2.0 * np.log(float(self.hurst_lag))))
        else:
            vr = 1.0
            hurst = 0.45

        # Vectorized Bollinger Bandwidth & Percentile (zero Python loops)
        bb_tail = c[-(self.bbwp_window + self.bb_period - 1):]
        windows = np.lib.stride_tricks.sliding_window_view(bb_tail, self.bb_period)
        means = windows.mean(axis=1)
        stds = windows.std(axis=1, ddof=0)
        bbw_arr = np.where(means > 1e-12, (4.0 * stds) / means, 0.0)
        curr_bbw = float(bbw_arr[-1])
        bbwp = float(np.sum(bbw_arr <= curr_bbw) / float(len(bbw_arr)) * 100.0)

        # Spread friction & causal spread ratio
        sp = float(spread_pips)
        baseline_sp = SYMBOL_BASELINE_SPREAD_PIPS.get(sym, 1.5)
        if spread_history is not None and len(spread_history) >= self.baseline_min_periods:
            sp_hist = np.asarray(spread_history[-self.baseline_window:], dtype=np.float64)
            sp_hist = sp_hist[~np.isnan(sp_hist)]
            if len(sp_hist) >= self.baseline_min_periods:
                baseline_sp = max(float(np.median(sp_hist)), 0.25 * baseline_sp)
        spread_ratio = float(sp / baseline_sp) if baseline_sp > 1e-12 else 1.0

        atr5_pips = atr5_val / pip_unit
        atr14_pips = atr14_val / pip_unit
        svp = float(sp / atr5_pips) if atr5_pips > 1e-12 else 0.0
        svp14 = float(sp / atr14_pips) if atr14_pips > 1e-12 else 0.0

        is_rollover = False
        clock_hours = {"utc_hour": -1, "ny_hour": -1, "london_hour": -1, "tokyo_hour": -1}
        if current_utc_dt is not None:
            ts_utc = to_utc_timestamp(current_utc_dt)
            clock_hours = get_session_clock_hours(ts_utc)
            is_rollover = clock_hours["ny_hour"] == 17

        metrics: Dict[str, Any] = {
            "ker": round(ker, 4),
            "er": round(ker, 4),
            "chop": round(chop, 2),
            "adx": round(adx_val, 2),
            "hurst": round(hurst, 4),
            "vr": round(vr, 4),
            "rvr": round(rvr, 3),
            "atr14_pips": round(atr14_pips, 3),
            "atr_rel": round(atr_rel, 3),
            "bb_width": round(curr_bbw, 6),
            "bbwp": round(bbwp, 1),
            "spread_pips": round(sp, 3),
            "spread_ratio": round(spread_ratio, 3),
            "svp": round(svp, 4),
            "svp14": round(svp14, 4),
            "is_rollover": bool(is_rollover),
            "utc_hour": clock_hours["utc_hour"],
            "ny_hour": clock_hours["ny_hour"],
            "london_hour": clock_hours["london_hour"],
        }
        regime = self.classify_regime(metrics)
        metrics["regime"] = regime
        metrics["regime_label"] = REGIME_LABELS.get(regime, "Unknown")
        return metrics


def resolve_strategy_profile(
    strategy_name: str,
    symbol: str,
    registry: Dict[str, Dict[str, Any]],
    unique_name_registry: Dict[str, Dict[str, Any]],
    strategy_aliases: Optional[Dict[str, str]] = None,
    strategy_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Exact normalized registry lookup — zero substring matching."""
    sym = normalize_symbol(symbol)
    raw_id = strategy_id or strategy_name
    sid = normalize_strategy_id(raw_id, symbol=sym)

    exact_key = f"{sym}__{sid}"
    if exact_key in registry:
        return registry[exact_key]

    if strategy_aliases:
        alias_target = strategy_aliases.get(exact_key) or strategy_aliases.get(sid)
        if alias_target and alias_target in registry:
            return registry[alias_target]

    if sid in unique_name_registry:
        prof = unique_name_registry[sid]
        if normalize_symbol(prof.get("symbol", "")) == sym:
            return prof

    return None


def evaluate_permission_from_metrics(
    classifier: MarketRegimeClassifier,
    strategy_name: str,
    symbol: str,
    metrics: Dict[str, Any],
    current_utc_dt: Union[pd.Timestamp, int, float, str],
    registry: Dict[str, Dict[str, Any]],
    unique_name_registry: Dict[str, Dict[str, Any]],
    strategy_aliases: Optional[Dict[str, str]] = None,
    archetype: Optional[str] = None,
    strategy_id: Optional[str] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Shared deterministic kill-switch evaluation over precomputed/live metrics."""
    if not metrics:
        return True, "WARMUP_PASS", {"regime": 0, "regime_label": REGIME_LABELS[0]}

    sym = normalize_symbol(symbol)
    if sym == "USDJPY":
        return False, "KILL_EXCLUDED_SYMBOL_USDJPY", metrics

    ts_utc = to_utc_timestamp(current_utc_dt)
    clock_hours = get_session_clock_hours(ts_utc)
    day_name = ts_utc.day_name()
    is_rollover = clock_hours["ny_hour"] == 17

    m = dict(metrics)
    m["is_rollover"] = bool(is_rollover)
    m["utc_hour"] = clock_hours["utc_hour"]
    m["ny_hour"] = clock_hours["ny_hour"]
    m["london_hour"] = clock_hours["london_hour"]

    regime = classifier.classify_regime(m)
    m["regime"] = regime
    m["regime_label"] = REGIME_LABELS.get(regime, "Unknown")

    # Level 1: Universal Wide-Spread Guard (>3x causal median) & Extreme Liquidity Guard
    max_sr = classifier.thresholds["r4_max_spread_ratio"]
    if m.get("spread_ratio", 1.0) > max_sr:
        return (
            False,
            f"KILL_WIDE_SPREAD_GUARD (SpreadRatio={m['spread_ratio']:.2f} > {max_sr:.1f}x median)",
            m,
        )
    max_svp14 = classifier.thresholds["r4_max_svp14"]
    if m.get("svp14", 0.0) > max_svp14:
        return (
            False,
            f"KILL_EXTREME_SPREAD_TO_ATR (SVP14={m['svp14']:.2f} > {max_svp14:.2f})",
            m,
        )

    # Level 2: Exact Strategy Profile Lookup (zero substring matching)
    profile = resolve_strategy_profile(
        strategy_name=strategy_name,
        symbol=sym,
        registry=registry,
        unique_name_registry=unique_name_registry,
        strategy_aliases=strategy_aliases,
        strategy_id=strategy_id,
    )

    if profile is not None:
        if day_name in profile.get("toxic_days", []):
            return False, f"KILL_TAILORED_TOXIC_DAY ({profile['name']} on {day_name})", m

        for sess in profile.get("toxic_sessions", []):
            if evaluate_session_match(sess, clock_hours):
                return False, f"KILL_TAILORED_TOXIC_SESSION ({profile['name']} in {sess})", m

        prohibited = profile.get("prohibited_regimes", [3, 4])
        if regime in prohibited:
            return False, f"KILL_PROHIBITED_REGIME (Regime {regime}: {m['regime_label']})", m

        min_ker = profile.get("min_ker")
        if min_ker is not None and m.get("ker", 1.0) < float(min_ker):
            return False, f"KILL_LOW_ER (ER={m['ker']:.3f} < {min_ker})", m

        max_chop = profile.get("max_chop")
        if max_chop is not None and m.get("chop", 0.0) > float(max_chop):
            return False, f"KILL_HIGH_CHOP (CHOP={m['chop']:.1f} > {max_chop})", m

        min_adx = profile.get("min_adx")
        if min_adx is not None and m.get("adx", 100.0) < float(min_adx):
            return False, f"KILL_LOW_ADX (ADX={m['adx']:.1f} < {min_adx})", m

        max_svp14_prof = profile.get("max_svp14")
        if max_svp14_prof is not None and m.get("svp14", 0.0) > float(max_svp14_prof):
            return False, f"KILL_PROFILE_SVP14 (SVP14={m['svp14']:.3f} > {max_svp14_prof})", m

        return True, f"PERMITTED_TO_TRADE (Regime {regime}: {m['regime_label']})", m

    # Level 3: Explicit Archetype Rule (or Universal Extreme-Tail Safety Gate if unknown)
    norm_arch = normalize_archetype(archetype)
    if norm_arch and norm_arch in ARCHETYPE_RULES:
        prohibited = ARCHETYPE_RULES[norm_arch]["prohibited_regimes"]
        if regime in prohibited:
            return False, f"KILL_ARCHETYPE_PROHIBITED_REGIME ({norm_arch}: Regime {regime} {m['regime_label']})", m
        return True, f"PERMITTED_ARCHETYPE (Regime {regime}: {m['regime_label']})", m

    if regime in (3, 4):
        return False, f"KILL_ADVERSE_REGIME (Regime {regime}: {m['regime_label']})", m

    return True, f"PERMITTED_BALANCED (Regime {regime}: {m['regime_label']})", m
