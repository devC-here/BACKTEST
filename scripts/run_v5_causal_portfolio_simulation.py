#!/usr/bin/env python3
"""
================================================================================
  STRICTLY CAUSAL MARK-TO-MARKET (MTM) 20-SLOT PORTFOLIO SIMULATOR (v5.0)
================================================================================
Simulates a live multi-strategy FX portfolio across Quarantined Holdout Weeks
211..300 (strictly excluding USDJPY) with ZERO future leakage:

1. Causal M15 Bid/Ask Timeline Cache:
   - Built per symbol from `data/{symbol}` via `get_week_dataset(..., warmup_prev_weeks=0)`
     using strictly causal bar closes (`closed='left', label='right'`, so every
     bar timestamp T_close satisfies T_tick < T_close).
   - At any decision timestamp `e_time`, only bars with `T_close <= e_time` are
     visible (`np.searchsorted(..., side='right') - 1`).

2. Causal Mark-to-Market Accounting ($4.00/lot Round-Turn Commission):
   - Open Long trades mark to market and exit early on causal M15 `bid_close`.
   - Open Short trades mark to market and exit early on causal M15 `ask_close`.
   - Deducts exact pip commission (`get_friction_pips`) and $4.00/lot USD
     commission matching `tick_engine/core/engine.py` and `config.py`.
   - Never inspects future realized `exit_time`, `exit_price`, or `pips` for
     open trades (`exit_time > e_time`).

3. Execution Modes & Position Sizing:
   - Execution modes:
       * `no_cut`: Hard 20-slot concurrency cap (blocks when 20 open, 0 early cuts).
       * `causal_mtm_room_making`: 24h max-holding zombie close + causal MTM
         room-making (when 20/20 slots full, cuts weakest open loser past
         `timeout_hours`; strictly shields winning MTM trades).
       * `regime_filtered`: Same causal MTM room-making plus causal pre-entry
         permission check via `LiveMarketRegimeFilter` on closed M15 bars.
   - Position sizing modes:
       * `fixed_0.03`: Fixed 0.03 lots
       * `fixed_0.04`: Fixed 0.04 lots
       * `risk_0.5pct`: Dynamic 0.5% equity risk per trade:
         `round_down_0.01((equity * 0.005) / (sl_pips * pip_val_per_lot))`
         clamped to `[0.01, 5.0]`.
================================================================================
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import inspect
import json
import math
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.config import (
    DEFAULT_DOLLAR_PER_LOT,
    PIP_SIZES,
    PIP_VALUES_USD,
    get_friction_pips,
)
from tick_engine.core.dataloader import get_split_weeks, get_week_dataset

NON_USDJPY_SYMBOLS: Tuple[str, ...] = (
    "AUDCAD",
    "AUDNZD",
    "EURGBP",
    "EURNZD",
    "EURUSD",
    "GBPCHF",
    "GBPJPY",
    "NZDCAD",
    "USDCAD",
)
EXCLUDED_SYMBOLS = frozenset({"USDJPY"})
M15_MS = 15 * 60 * 1000
DEFAULT_CACHE_DIR = ROOT_DIR / "scratch" / "swarm_w4_m15_cache"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "results" / "v5_holdout_portfolio_simulation.json"


# ------------------------------------------------------------------------------
# Instrument & Sizing Helpers
# ------------------------------------------------------------------------------
def get_pip_size(symbol: str) -> float:
    s = symbol.upper()
    return float(PIP_SIZES.get(s, 0.01 if "JPY" in s else 0.0001))


def get_pip_value_usd(symbol: str) -> float:
    s = symbol.upper()
    return float(PIP_VALUES_USD.get(s, 10.0))


def get_commission_pips(
    symbol: str,
    commission_per_lot_usd: float = DEFAULT_DOLLAR_PER_LOT,
) -> float:
    comm_pips, _ = get_friction_pips(symbol, dollar_per_lot=float(commission_per_lot_usd))
    return float(comm_pips)


def compute_dynamic_risk_lots(
    equity: float,
    sl_pips: float,
    pip_val_per_lot: float,
    risk_fraction: float = 0.005,
    min_lots: float = 0.01,
    max_lots: float = 5.0,
) -> float:
    """
    Computes dynamic position size in lots:
      lots = round_down_0.01((equity * risk_fraction) / (sl_pips * pip_val_per_lot))
    clamped to [min_lots, max_lots].
    """
    if sl_pips <= 0.0:
        raise ValueError(f"sl_pips must be positive for dynamic sizing, got {sl_pips}")
    if pip_val_per_lot <= 0.0:
        raise ValueError(f"pip_val_per_lot must be positive, got {pip_val_per_lot}")
    if equity <= 0.0:
        return round(float(min_lots), 2)

    raw_lots = (float(equity) * float(risk_fraction)) / (float(sl_pips) * float(pip_val_per_lot))
    # Small 1e-9 guard prevents IEEE-754 underflow (e.g. 0.24999999999999997 -> 0.24)
    floored = math.floor((raw_lots * 100.0) + 1e-9) / 100.0
    clamped = max(float(min_lots), min(float(max_lots), floored))
    return round(clamped, 2)


def compute_position_lots(
    sizing_mode: str,
    equity: float,
    sl_pips: float,
    pip_val_per_lot: float,
) -> float:
    """Resolves trade lot size for fixed or dynamic equity-risk sizing modes."""
    mode = sizing_mode.strip().lower()
    if mode == "fixed_0.03":
        return 0.03
    if mode == "fixed_0.04":
        return 0.04
    if mode.startswith("fixed_"):
        val = float(mode.split("fixed_", 1)[1])
        return round(max(0.01, min(5.0, val)), 2)
    if mode == "risk_0.5pct":
        return compute_dynamic_risk_lots(
            equity=equity,
            sl_pips=sl_pips,
            pip_val_per_lot=pip_val_per_lot,
            risk_fraction=0.005,
        )
    if mode.startswith("risk_") and mode.endswith("pct"):
        pct = float(mode[len("risk_") : -len("pct")])
        return compute_dynamic_risk_lots(
            equity=equity,
            sl_pips=sl_pips,
            pip_val_per_lot=pip_val_per_lot,
            risk_fraction=pct / 100.0,
        )
    raise ValueError(f"Unsupported sizing_mode: {sizing_mode}")


# ------------------------------------------------------------------------------
# Causal M15 Bid/Ask Timeline & Disk Cache
# ------------------------------------------------------------------------------
@dataclass
class M15Timeline:
    """
    Strictly causal M15 timeline indexed by bar-close timestamp `timestamp_ms`.
    Every bar i aggregates ticks strictly in `[timestamp_ms[i] - 900_000, timestamp_ms[i])`.
    """

    symbol: str
    pip_size: float
    timestamp_ms: np.ndarray
    bid_close: np.ndarray
    ask_close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    spread_pips: np.ndarray

    def __post_init__(self) -> None:
        self.symbol = self.symbol.upper()
        self.pip_size = float(self.pip_size)
        self.timestamp_ms = np.ascontiguousarray(self.timestamp_ms, dtype=np.int64)
        self.bid_close = np.ascontiguousarray(self.bid_close, dtype=np.float64)
        self.ask_close = np.ascontiguousarray(self.ask_close, dtype=np.float64)
        self.high = np.ascontiguousarray(self.high, dtype=np.float64)
        self.low = np.ascontiguousarray(self.low, dtype=np.float64)
        self.close = np.ascontiguousarray(self.close, dtype=np.float64)
        self.spread_pips = np.ascontiguousarray(self.spread_pips, dtype=np.float64)

        if len(self.timestamp_ms) > 1 and np.any(self.timestamp_ms[1:] <= self.timestamp_ms[:-1]):
            order = np.argsort(self.timestamp_ms, kind="mergesort")
            self.timestamp_ms = self.timestamp_ms[order]
            self.bid_close = self.bid_close[order]
            self.ask_close = self.ask_close[order]
            self.high = self.high[order]
            self.low = self.low[order]
            self.close = self.close[order]
            self.spread_pips = self.spread_pips[order]

        self.unique_day_ints = np.unique(self.timestamp_ms // 86_400_000) if len(self.timestamp_ms) > 0 else np.empty(0, dtype=np.int64)

    def latest_bar_index(self, eval_time_ms: int) -> int:
        """Returns index of latest closed M15 bar with `T_close <= eval_time_ms`, or -1."""
        if len(self.timestamp_ms) == 0:
            return -1
        return int(np.searchsorted(self.timestamp_ms, int(eval_time_ms), side="right")) - 1

    def get_causal_bid_ask(self, eval_time_ms: int) -> Optional[Tuple[float, float, int]]:
        """Returns `(bid_close, ask_close, bar_close_ms)` for latest `T_close <= eval_time_ms`."""
        idx = self.latest_bar_index(eval_time_ms)
        if idx < 0:
            return None
        return (
            float(self.bid_close[idx]),
            float(self.ask_close[idx]),
            int(self.timestamp_ms[idx]),
        )

    def get_causal_regime_slice(
        self,
        eval_time_ms: int,
        lookback_bars: int = 1000,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int, int]]:
        """
        Returns strictly causal `(high, low, close, spread_history, spread_pips, bar_close_ms, bar_idx)`
        for bars ending at latest `T_close <= eval_time_ms`.
        """
        idx = self.latest_bar_index(eval_time_ms)
        if idx < 0:
            return None
        start = max(0, idx + 1 - int(lookback_bars))
        end = idx + 1
        return (
            self.high[start:end],
            self.low[start:end],
            self.close[start:end],
            self.spread_pips[start:end],
            float(self.spread_pips[idx]),
            int(self.timestamp_ms[idx]),
            idx,
        )

    def to_regime_dataframe(self) -> pd.DataFrame:
        """Converts causal M15 arrays to bar-close-indexed DataFrame for `precompute_m15_regimes`."""
        idx = pd.to_datetime(self.timestamp_ms, unit="ms", utc=True).as_unit("ms")
        return pd.DataFrame(
            {
                "high": self.high,
                "low": self.low,
                "close": self.close,
                "spread_pips": self.spread_pips,
            },
            index=idx,
        )


def ticks_to_m15_arrays(
    ticks: np.ndarray,
    pip_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregates structured tick array into causal M15 arrays matching
    `tick_engine.core.causal_bars.build_causal_time_bars(..., timeframe_min=15)`.
    """
    if len(ticks) == 0:
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float64)
        return empty_i, empty_f, empty_f, empty_f, empty_f, empty_f, empty_f, empty_f

    ts = np.asarray(ticks["timestamp"], dtype=np.int64)
    bid = np.asarray(ticks["bid"], dtype=np.float64)
    ask = np.asarray(ticks["ask"], dtype=np.float64)

    if len(ts) > 1 and np.any(ts[1:] < ts[:-1]):
        order = np.argsort(ts, kind="mergesort")
        ts = ts[order]
        bid = bid[order]
        ask = ask[order]

    bucket_close = ((ts // M15_MS) + 1) * M15_MS
    changes = np.flatnonzero(bucket_close[1:] != bucket_close[:-1]) + 1
    first_idx = np.r_[0, changes]
    last_idx = np.r_[changes - 1, len(ts) - 1]
    counts = (last_idx - first_idx + 1).astype(np.float64)

    mid = 0.5 * (bid + ask)
    tick_spread_pips = (ask - bid) / float(pip_size)

    ts_close = bucket_close[first_idx]
    bid_close = bid[last_idx]
    ask_close = ask[last_idx]
    high = np.maximum.reduceat(mid, first_idx)
    low = np.minimum.reduceat(mid, first_idx)
    close = mid[last_idx]
    spread_pips = np.add.reduceat(tick_spread_pips, first_idx) / counts
    return ts_close, bid_close, ask_close, high, low, close, spread_pips, counts


def build_m15_timeline_from_ticks(
    symbol: str,
    ticks: np.ndarray,
    pip_size: Optional[float] = None,
) -> M15Timeline:
    """Builds an in-memory `M15Timeline` directly from a structured tick array."""
    sym = symbol.upper()
    p_size = float(pip_size) if pip_size is not None else get_pip_size(sym)
    ts_close, bid_close, ask_close, high, low, close, spread_pips, _ = ticks_to_m15_arrays(ticks, p_size)
    return M15Timeline(
        symbol=sym,
        pip_size=p_size,
        timestamp_ms=ts_close,
        bid_close=bid_close,
        ask_close=ask_close,
        high=high,
        low=low,
        close=close,
        spread_pips=spread_pips,
    )


def _merge_sorted_bar_chunks(
    chunks: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not chunks:
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float64)
        return empty_i, empty_f, empty_f, empty_f, empty_f, empty_f, empty_f

    ts = np.concatenate([c[0] for c in chunks])
    bid_c = np.concatenate([c[1] for c in chunks])
    ask_c = np.concatenate([c[2] for c in chunks])
    high = np.concatenate([c[3] for c in chunks])
    low = np.concatenate([c[4] for c in chunks])
    close = np.concatenate([c[5] for c in chunks])
    spread = np.concatenate([c[6] for c in chunks])
    counts = np.concatenate([c[7] for c in chunks])

    if len(ts) <= 1:
        return ts, bid_c, ask_c, high, low, close, spread

    if np.any(ts[1:] < ts[:-1]):
        order = np.argsort(ts, kind="mergesort")
        ts = ts[order]
        bid_c = bid_c[order]
        ask_c = ask_c[order]
        high = high[order]
        low = low[order]
        close = close[order]
        spread = spread[order]
        counts = counts[order]

    # Fast path when no boundary bar overlaps between consecutive weeks
    if not np.any(ts[1:] == ts[:-1]):
        return ts, bid_c, ask_c, high, low, close, spread

    changes = np.flatnonzero(ts[1:] != ts[:-1]) + 1
    first_idx = np.r_[0, changes]
    last_idx = np.r_[changes - 1, len(ts) - 1]
    total_counts = np.maximum(np.add.reduceat(counts, first_idx), 1.0)
    weighted_spread = np.add.reduceat(spread * counts, first_idx) / total_counts
    return (
        ts[first_idx],
        bid_c[last_idx],
        ask_c[last_idx],
        np.maximum.reduceat(high, first_idx),
        np.minimum.reduceat(low, first_idx),
        close[last_idx],
        weighted_spread,
    )


def load_or_build_symbol_m15_timeline(
    symbol: str,
    weeks: Optional[Sequence[int]] = None,
    cache_dir: Optional[Path] = DEFAULT_CACHE_DIR,
    force_rebuild: bool = False,
) -> M15Timeline:
    """
    Loads or builds causal M15 Bid/Ask timeline for `symbol` over Holdout Weeks
    211..300 using `get_week_dataset(..., warmup_prev_weeks=0)`.
    """
    sym = symbol.upper()
    if sym in EXCLUDED_SYMBOLS:
        raise ValueError(f"Symbol {sym} is strictly excluded from v5.0 non-USDJPY simulation")

    p_size = get_pip_size(sym)
    raw_dir = ROOT_DIR / "data" / sym
    target_weeks = sorted(int(w) for w in (weeks if weeks is not None else get_split_weeks(raw_dir, split="test")))
    if not target_weeks:
        target_weeks = list(range(211, 301))

    week_arr = np.asarray(target_weeks, dtype=np.int32)
    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{sym}_m15_w{target_weeks[0]}_{target_weeks[-1]}_n{len(target_weeks)}.npz"
        if cache_path.exists() and not force_rebuild:
            with np.load(cache_path) as data:
                if "weeks" in data and np.array_equal(data["weeks"], week_arr):
                    return M15Timeline(
                        symbol=sym,
                        pip_size=p_size,
                        timestamp_ms=data["timestamp_ms"],
                        bid_close=data["bid_close"],
                        ask_close=data["ask_close"],
                        high=data["high"],
                        low=data["low"],
                        close=data["close"],
                        spread_pips=data["spread_pips"],
                    )

    chunks = []
    for w in target_weeks:
        ticks, _ = get_week_dataset(week=w, raw_dir=raw_dir, warmup_prev_weeks=0)
        if len(ticks) > 0:
            chunks.append(ticks_to_m15_arrays(ticks, p_size))

    ts, bid_c, ask_c, high, low, close, spread = _merge_sorted_bar_chunks(chunks)

    if cache_path is not None:
        np.savez_compressed(
            cache_path,
            weeks=week_arr,
            timestamp_ms=ts,
            bid_close=bid_c,
            ask_close=ask_c,
            high=high,
            low=low,
            close=close,
            spread_pips=spread,
        )

    return M15Timeline(
        symbol=sym,
        pip_size=p_size,
        timestamp_ms=ts,
        bid_close=bid_c,
        ask_close=ask_c,
        high=high,
        low=low,
        close=close,
        spread_pips=spread,
    )


def load_timelines_for_symbols(
    symbols: Iterable[str],
    weeks: Optional[Sequence[int]] = None,
    cache_dir: Optional[Path] = DEFAULT_CACHE_DIR,
    force_rebuild: bool = False,
    verbose: bool = True,
) -> Dict[str, M15Timeline]:
    timelines: Dict[str, M15Timeline] = {}
    unique_syms = sorted({s.upper() for s in symbols if s.upper() not in EXCLUDED_SYMBOLS})
    for sym in unique_syms:
        t0 = time.time()
        tl = load_or_build_symbol_m15_timeline(
            sym,
            weeks=weeks,
            cache_dir=cache_dir,
            force_rebuild=force_rebuild,
        )
        timelines[sym] = tl
        if verbose:
            print(f"  [M15 Cache] {sym:<7s}: {len(tl.timestamp_ms):>6,} causal M15 bars ({time.time() - t0:.2f}s)")
    return timelines


# ------------------------------------------------------------------------------
# Trade Log & Manifest Ingestion (Strict Non-USDJPY, Holdout Only)
# ------------------------------------------------------------------------------
def _extract_sl_from_strategy_file(file_path: Path) -> Optional[float]:
    if not file_path.exists():
        return None
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    for pattern in (
        r"sl_pips\s*:\s*float\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        r"self\.sl_pips\s*=\s*(?:float\()?\s*([0-9]+(?:\.[0-9]+)?)",
        r"SL_PIPS\s*=\s*([0-9]+(?:\.[0-9]+)?)",
    ):
        m = re.search(pattern, text)
        if m:
            val = float(m.group(1))
            if val > 0:
                return val
    return None


def _extract_manifest_sl_pips(manifest_data: Dict[str, Any]) -> Optional[float]:
    for candidate in (
        manifest_data.get("sl_pips"),
        manifest_data.get("parameters", {}).get("sl_pips") if isinstance(manifest_data.get("parameters"), dict) else None,
        (
            manifest_data.get("custom_metadata", {}).get("parameters", {}).get("sl_pips")
            if isinstance(manifest_data.get("custom_metadata"), dict)
            and isinstance(manifest_data.get("custom_metadata", {}).get("parameters"), dict)
            else None
        ),
    ):
        if candidate is not None:
            val = float(candidate)
            if val > 0:
                return val
    return None


def _infer_symbol_and_strategy(path: Path) -> Tuple[Optional[str], str]:
    stem = path.stem
    for suffix in ("_holdout_trades", "_trades", "_holdout_manifest", "_run_manifest"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    if "__" in stem:
        left, right = stem.split("__", 1)
        if left.upper() in PIP_SIZES:
            return left.upper(), right

    upper_stem = stem.upper()
    for sym in sorted(PIP_SIZES.keys(), key=len, reverse=True):
        if upper_stem.startswith(sym + "_"):
            return sym, stem[len(sym) + 1 :]
        if sym in upper_stem:
            return sym, stem

    return None, stem


def _resolve_sl_for_strategy(
    df: pd.DataFrame,
    entry_meta: Optional[Dict[str, Any]],
    manifest_path: Optional[Path],
    strategy_file: Optional[Path],
    comm_pips: float,
) -> Tuple[float, str]:
    if entry_meta:
        val = _extract_manifest_sl_pips(entry_meta)
        if val is not None:
            return val, "fleet_manifest"

    if manifest_path and manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                m_data = json.load(f)
            val = _extract_manifest_sl_pips(m_data)
            if val is not None:
                return val, "strategy_manifest"
        except Exception:
            pass

    if strategy_file:
        val = _extract_sl_from_strategy_file(strategy_file)
        if val is not None:
            return val, "strategy_source"

    if not df.empty and "exit_reason" in df.columns and "pips" in df.columns:
        sl_rows = df[(df["exit_reason"] == 1) & (df["pips"] < 0)]
        if not sl_rows.empty:
            inferred = float(np.median(np.abs(sl_rows["pips"].to_numpy(dtype=np.float64))) - comm_pips)
            if inferred >= 1.0:
                return round(inferred, 1), "inferred_sl_exits"

    return 20.0, "default_20p"


def _find_strategy_csv_and_manifest(
    entry: Dict[str, Any],
    search_dirs: Sequence[Path],
    manifest_dir: Optional[Path],
) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    sym = str(entry.get("symbol", "")).upper()
    name = str(
        entry.get("name")
        or entry.get("strategy_id")
        or entry.get("strategy_name")
        or entry.get("strategy")
        or entry.get("id")
        or ""
    )
    rel_path = (
        entry.get("relative_path")
        or entry.get("strategy_file")
        or entry.get("file_path")
        or entry.get("path")
        or entry.get("filename")
    )

    strat_file: Optional[Path] = None
    if rel_path:
        for base in [ROOT_DIR, manifest_dir] if manifest_dir else [ROOT_DIR]:
            cand = (base / str(rel_path)).resolve()
            if cand.exists() and cand.suffix == ".py":
                strat_file = cand
                break

    manifest_file: Optional[Path] = None
    mf_str = (
        entry.get("manifest_file")
        or entry.get("holdout_manifest")
        or entry.get("holdout_manifest_path")
        or entry.get("manifest_path")
    )
    if mf_str:
        for base in [ROOT_DIR, manifest_dir] if manifest_dir else [ROOT_DIR]:
            cand = (base / str(mf_str)).resolve()
            if cand.exists():
                manifest_file = cand
                break

    csv_file: Optional[Path] = None
    csv_str = (
        entry.get("holdout_trades_csv")
        or entry.get("holdout_csv_path")
        or entry.get("holdout_csv")
        or entry.get("trades_csv")
        or entry.get("csv_path")
    )
    if csv_str:
        for base in [ROOT_DIR, manifest_dir] if manifest_dir else [ROOT_DIR]:
            cand = (base / str(csv_str)).resolve()
            if cand.exists():
                csv_file = cand
                break

    if csv_file is None and manifest_file is not None:
        for old, new in (
            ("_holdout_manifest.json", "_holdout_trades.csv"),
            ("_run_manifest.json", "_holdout_trades.csv"),
            ("_run_manifest.json", "_trades.csv"),
        ):
            cand = manifest_file.with_name(manifest_file.name.replace(old, new))
            if cand.exists():
                csv_file = cand
                break

    if csv_file is None and sym and name:
        slug_cands = [name]
        if strat_file is not None:
            slug_cands.append(strat_file.parent.name if strat_file.name == "strategy.py" else strat_file.stem)
        for sdir in search_dirs:
            if not sdir.exists():
                continue
            for slug in slug_cands:
                for pat in (
                    f"{sym}__{slug}_holdout_trades.csv",
                    f"{sym}__{slug}_trades.csv",
                    f"{sym.lower()}_{slug}_holdout_trades.csv",
                    f"{sym.lower()}_{slug}_trades.csv",
                    f"{slug}_holdout_trades.csv",
                    f"{slug}_trades.csv",
                ):
                    cand = sdir / pat
                    if cand.exists():
                        csv_file = cand
                        break
                if csv_file is not None:
                    break
                # Safe prefix glob fallback (require '_' delimiter after slug so disc1 never matches disc10)
                for glob_pat in (
                    f"{sym}__{slug}_*_holdout_trades.csv",
                    f"{sym}__{slug}_*_trades.csv",
                    f"{sym.lower()}_{slug}_*_holdout_trades.csv",
                    f"{sym.lower()}_{slug}_*_trades.csv",
                ):
                    matches = [
                        m for m in sorted(sdir.glob(glob_pat))
                        if not m.name.endswith("_discovery_trades.csv")
                    ]
                    if matches:
                        csv_file = matches[0]
                        break
                if csv_file is not None:
                    break
            if csv_file is not None:
                break

    if manifest_file is None and sym and name:
        for sdir in search_dirs:
            if not sdir.exists():
                continue
            for pat in (
                f"{sym}__{name}_holdout_manifest.json",
                f"{sym}__{name}_run_manifest.json",
                f"{sym.lower()}_{name}_run_manifest.json",
            ):
                cand = sdir / pat
                if cand.exists():
                    manifest_file = cand
                    break
            if manifest_file is not None:
                break
            for glob_pat in (
                f"{sym}__{name}_*_holdout_manifest.json",
                f"{sym}__{name}_*_run_manifest.json",
            ):
                matches = sorted(sdir.glob(glob_pat))
                if matches:
                    manifest_file = matches[0]
                    break
            if manifest_file is not None:
                break

    return csv_file, manifest_file, strat_file


def _parse_manifest_entries(manifest_path: Path) -> List[Dict[str, Any]]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        for key in ("strategies", "certified_strategies", "candidates", "winners", "fleet"):
            if isinstance(data.get(key), list):
                return [d for d in data[key] if isinstance(d, dict)]
        # Support dict keyed by strategy id -> metadata dict
        if data and all(isinstance(v, dict) for v in data.values()):
            entries = []
            for k, v in data.items():
                item = dict(v)
                item.setdefault("name", k)
                entries.append(item)
            return entries
    raise ValueError(f"Unrecognized manifest JSON structure in {manifest_path}")


def _dataframe_to_trade_records(
    df: pd.DataFrame,
    symbol: Optional[str],
    strategy_id: str,
    archetype: str,
    default_sl_pips: float,
    default_pip_size: float,
    default_comm_pips: float,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if df.empty:
        return records

    p_col = "pips" if "pips" in df.columns else ("pnl_pips" if "pnl_pips" in df.columns else None)
    if p_col is None:
        raise ValueError(f"Trade CSV for {symbol}__{strategy_id} missing 'pips'/'pnl_pips' column")

    for row in df.itertuples(index=False):
        r = row._asdict()
        row_sym = str(r.get("symbol") or symbol or "").upper()
        if not row_sym:
            continue
        row_strat = str(r.get("strategy") or r.get("strategy_id") or r.get("strategy_name") or strategy_id)
        pip_size = get_pip_size(row_sym) if row_sym != symbol else default_pip_size
        comm_pips = get_commission_pips(row_sym) if row_sym != symbol else default_comm_pips

        entry_t = int(r["entry_time"])
        exit_t = int(r.get("exit_time", entry_t + 3600_000))
        sig_t = int(r.get("signal_time", entry_t))
        direction = int(r.get("direction", 1))
        entry_price = float(r.get("entry_price", 0.0))
        exit_price = float(r.get("exit_price", entry_price))
        pips_val = float(r[p_col])
        row_sl = r.get("sl_pips")
        sl_val = float(row_sl) if row_sl is not None and not pd.isna(row_sl) and float(row_sl) > 0 else float(default_sl_pips)
        row_tp = r.get("tp_pips")
        tp_val = float(row_tp) if row_tp is not None and not pd.isna(row_tp) and float(row_tp) > 0 else 0.0

        # Guard if entry_price was absent in synthetic logs
        if entry_price <= 0.0:
            raw_p = pips_val + comm_pips
            entry_price = 1.0
            exit_price = entry_price + (raw_p * pip_size * direction)

        records.append({
            "strategy": row_strat,
            "symbol": row_sym,
            "archetype": str(r.get("archetype", archetype or "")),
            "signal_time": sig_t,
            "entry_time": entry_t,
            "exit_time": exit_t,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pips": pips_val,
            "sl_pips": max(sl_val, 1.0),
            "tp_pips": tp_val,
            "exit_reason": int(r.get("exit_reason", 0)),
        })
    return records


def load_portfolio_trades(
    manifest_paths: Optional[Sequence[Path]] = None,
    trades_dirs: Optional[Sequence[Path]] = None,
    fleet_dir: Optional[Path] = None,
    include_unlisted_csvs: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Loads Holdout trades from v5.0 manifests and/or CSV folders, strictly
    excluding USDJPY and strictly excluding *_discovery_trades.csv files.
    """
    m_paths = [Path(p) for p in (manifest_paths or [])]
    t_dirs = [Path(d) for d in (trades_dirs or [])]

    if fleet_dir is not None:
        f_dir = Path(fleet_dir)
        for cand_m in ("FLEET_MANIFEST.json", "CERTIFIED_FLEET_MANIFEST.json", "summary.json"):
            if (f_dir / cand_m).exists() and (f_dir / cand_m) not in m_paths:
                m_paths.append(f_dir / cand_m)
        if f_dir not in t_dirs:
            t_dirs.append(f_dir)
        if (f_dir / "manifests").exists() and (f_dir / "manifests") not in t_dirs:
            t_dirs.append(f_dir / "manifests")

    if not m_paths and not t_dirs:
        raise ValueError("Provide at least one of --fleet-dir, --manifest, or --trades-dir")

    loaded_trades: List[Dict[str, Any]] = []
    seen_strategy_keys: set[Tuple[str, str]] = set()
    strategy_metadata: Dict[str, Dict[str, Any]] = {}
    excluded_usdjpy_strategies: List[str] = []
    excluded_usdjpy_trades = 0
    missing_csv_strategies: List[str] = []

    # 1. Load strategies declared in manifest JSON files
    for m_path in m_paths:
        if not m_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {m_path}")
        entries = _parse_manifest_entries(m_path)
        search_dirs = list(t_dirs) + [
            m_path.parent,
            m_path.parent / "manifests",
            ROOT_DIR / "results" / "certified_strategy_manifests",
        ]
        for entry in entries:
            sym = str(entry.get("symbol", "")).upper()
            name = str(
                entry.get("name")
                or entry.get("strategy_id")
                or entry.get("strategy_name")
                or entry.get("strategy")
                or entry.get("id")
                or ""
            )
            if not sym and entry.get("filename"):
                inferred_sym, inferred_name = _infer_symbol_and_strategy(Path(entry["filename"]))
                sym = inferred_sym or sym
                name = name or inferred_name
            if not sym:
                continue

            if sym in EXCLUDED_SYMBOLS:
                excluded_usdjpy_strategies.append(f"{sym}__{name}")
                continue

            key = (sym, name)
            if key in seen_strategy_keys:
                continue

            csv_file, strat_manifest, strat_file = _find_strategy_csv_and_manifest(entry, search_dirs, m_path.parent)
            if csv_file is None or not csv_file.exists():
                missing_csv_strategies.append(f"{sym}__{name}")
                continue

            df = pd.read_csv(csv_file)
            pip_size = get_pip_size(sym)
            comm_pips = get_commission_pips(sym)
            sl_pips, sl_source = _resolve_sl_for_strategy(df, entry, strat_manifest, strat_file, comm_pips)
            archetype = str(entry.get("archetype") or entry.get("category") or "")

            recs = _dataframe_to_trade_records(
                df=df,
                symbol=sym,
                strategy_id=name,
                archetype=archetype,
                default_sl_pips=sl_pips,
                default_pip_size=pip_size,
                default_comm_pips=comm_pips,
            )
            seen_strategy_keys.add(key)
            loaded_trades.extend(recs)
            strategy_metadata[f"{sym}__{name}"] = {
                "symbol": sym,
                "strategy": name,
                "archetype": archetype,
                "sl_pips": sl_pips,
                "sl_source": sl_source,
                "trades": len(recs),
                "csv_file": str(csv_file),
            }

    # 2. Scan trades_dirs for Holdout CSVs when no manifest was given (or when explicitly requested)
    if (not m_paths) or include_unlisted_csvs:
        for t_dir in t_dirs:
            if not t_dir.exists():
                raise FileNotFoundError(f"Trades directory not found: {t_dir}")
            holdout_csvs = sorted(t_dir.rglob("*_holdout_trades.csv"))
            other_csvs = [
                p for p in sorted(t_dir.rglob("*_trades.csv"))
                if not p.name.endswith("_discovery_trades.csv")
                and not p.name.endswith("_holdout_trades.csv")
            ]
            for csv_file in holdout_csvs + other_csvs:
                sym, strat_id = _infer_symbol_and_strategy(csv_file)
                df = pd.read_csv(csv_file)
                has_row_sym = "symbol" in df.columns and not df.empty
                if sym is None and has_row_sym:
                    unique_syms = {str(s).upper() for s in df["symbol"].dropna().unique()}
                    if len(unique_syms) == 1:
                        sym = next(iter(unique_syms))
                if sym is None and not has_row_sym:
                    continue

                if sym is not None and sym in EXCLUDED_SYMBOLS:
                    excluded_usdjpy_strategies.append(f"{sym}__{strat_id}")
                    excluded_usdjpy_trades += len(df)
                    continue

                if sym is not None:
                    key = (sym, strat_id)
                    if key in seen_strategy_keys:
                        continue

                # Look for matching holdout/run manifest beside CSV
                strat_manifest = None
                for pat in (
                    csv_file.name.replace("_holdout_trades.csv", "_holdout_manifest.json"),
                    csv_file.name.replace("_holdout_trades.csv", "_run_manifest.json"),
                    csv_file.name.replace("_trades.csv", "_run_manifest.json"),
                ):
                    cand = csv_file.with_name(pat)
                    if cand.exists():
                        strat_manifest = cand
                        break

                ref_sym = sym or str(df["symbol"].iloc[0]).upper()
                pip_size = get_pip_size(ref_sym)
                comm_pips = get_commission_pips(ref_sym)
                sl_pips, sl_source = _resolve_sl_for_strategy(df, None, strat_manifest, None, comm_pips)

                recs = _dataframe_to_trade_records(
                    df=df,
                    symbol=sym,
                    strategy_id=strat_id,
                    archetype="",
                    default_sl_pips=sl_pips,
                    default_pip_size=pip_size,
                    default_comm_pips=comm_pips,
                )
                for r in recs:
                    r_sym = r["symbol"]
                    r_strat = r["strategy"]
                    if r_sym in EXCLUDED_SYMBOLS:
                        excluded_usdjpy_strategies.append(f"{r_sym}__{r_strat}")
                        continue
                    r_key = (r_sym, r_strat)
                    if r_key in seen_strategy_keys:
                        continue
                    loaded_trades.append(r)
                    meta_key = f"{r_sym}__{r_strat}"
                    if meta_key not in strategy_metadata:
                        strategy_metadata[meta_key] = {
                            "symbol": r_sym,
                            "strategy": r_strat,
                            "archetype": r.get("archetype", ""),
                            "sl_pips": sl_pips,
                            "sl_source": sl_source,
                            "trades": 0,
                            "csv_file": str(csv_file),
                        }
                    strategy_metadata[meta_key]["trades"] += 1
                for r_sym, r_strat in {(r["symbol"], r["strategy"]) for r in recs if r["symbol"] not in EXCLUDED_SYMBOLS}:
                    seen_strategy_keys.add((r_sym, r_strat))

    # Final defensive USDJPY filter + deterministic chronological sort
    filtered_trades = []
    for t in loaded_trades:
        if str(t.get("symbol", "")).upper() in EXCLUDED_SYMBOLS:
            excluded_usdjpy_trades += 1
            continue
        filtered_trades.append(t)

    filtered_trades.sort(key=lambda x: (int(x["entry_time"]), int(x.get("signal_time", 0)), str(x["symbol"]), str(x["strategy"])))

    meta_summary = {
        "strategies_loaded": len(strategy_metadata),
        "total_holdout_trades": len(filtered_trades),
        "symbols_present": sorted({t["symbol"] for t in filtered_trades}),
        "excluded_usdjpy_strategies": sorted(set(excluded_usdjpy_strategies)),
        "excluded_usdjpy_trades": excluded_usdjpy_trades,
        "missing_csv_strategies": missing_csv_strategies,
        "strategy_sl_sources": {k: {"sl_pips": v["sl_pips"], "source": v["sl_source"], "trades": v["trades"]} for k, v in sorted(strategy_metadata.items())},
    }
    return filtered_trades, meta_summary


# ------------------------------------------------------------------------------
# Regime Filter Adapter (Causal Query at T_close <= e_time)
# ------------------------------------------------------------------------------
class CausalRegimeAdapter:
    """
    Wraps `@regime-architect`'s `LiveMarketRegimeFilter` and queries it strictly
    causally on closed M15 bars (`T_close <= e_time`).
    Uses O(N) `precompute_m15_regimes` + O(log N) `evaluate_at_timestamp` when
    available, falling back cleanly to `evaluate_live_permission` otherwise.
    """

    def __init__(
        self,
        regime_filter: Optional[Any] = None,
        lookback_bars: int = 1000,
        regime_module_path: Optional[Path] = None,
        use_precomputed: bool = True,
    ):
        self.lookback_bars = int(lookback_bars)
        if regime_filter is not None:
            self.filter = regime_filter
        elif regime_module_path is not None:
            import importlib.util

            mod_path = Path(regime_module_path)
            spec = importlib.util.spec_from_file_location(mod_path.stem, mod_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load regime module from {mod_path}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.filter = mod.LiveMarketRegimeFilter()
        else:
            from live_regime_filter import LiveMarketRegimeFilter
            self.filter = LiveMarketRegimeFilter()

        self._has_batch_api = bool(
            use_precomputed
            and hasattr(self.filter, "precompute_m15_regimes")
            and hasattr(self.filter, "evaluate_at_timestamp")
        )
        sig = inspect.signature(self.filter.evaluate_live_permission)
        self._accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        self._param_names = set(sig.parameters.keys())
        self._precomputed_by_symbol: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self._eval_cache: Dict[Tuple[str, int, str, int, int], Tuple[bool, str]] = {}

    def _get_precomputed(self, sym: str, tl: M15Timeline) -> Dict[str, Any]:
        key = (sym, id(tl))
        if key not in self._precomputed_by_symbol:
            df = tl.to_regime_dataframe()
            self._precomputed_by_symbol[key] = self.filter.precompute_m15_regimes(df, symbol=sym)
        return self._precomputed_by_symbol[key]

    def check_permission(
        self,
        trade: Dict[str, Any],
        eval_time_ms: int,
        timelines: Dict[str, M15Timeline],
    ) -> Tuple[bool, str]:
        sym = str(trade["symbol"]).upper()
        strat = str(trade.get("strategy", "unknown"))
        tl = timelines.get(sym)
        if tl is None:
            return True, "NO_TIMELINE_PASS"

        idx = tl.latest_bar_index(eval_time_ms)
        if idx < 0:
            return True, "WARMUP_NO_BARS_PASS"

        utc_dt = datetime.fromtimestamp(int(eval_time_ms) / 1000.0, tz=timezone.utc)
        cache_key = (sym, idx, strat, int(utc_dt.weekday()), int(utc_dt.hour))
        if cache_key in self._eval_cache:
            return self._eval_cache[cache_key]

        if self._has_batch_api:
            pre = self._get_precomputed(sym, tl)
            permitted, reason, _ = self.filter.evaluate_at_timestamp(
                strategy_name=strat,
                symbol=sym,
                precomputed=pre,
                decision_ts_ms=int(eval_time_ms),
                spread_pips=float(tl.spread_pips[idx]),
                archetype=trade.get("archetype") or None,
                strategy_id=strat,
            )
            result = (bool(permitted), str(reason))
            self._eval_cache[cache_key] = result
            return result

        sliced = tl.get_causal_regime_slice(eval_time_ms, lookback_bars=self.lookback_bars)
        if sliced is None:
            return True, "WARMUP_NO_BARS_PASS"

        high, low, close, spread_hist, spread_pips, _, _ = sliced
        ts_utc = pd.Timestamp(int(eval_time_ms), unit="ms", tz="UTC")
        kwargs: Dict[str, Any] = {
            "strategy_name": strat,
            "symbol": sym,
            "high": high,
            "low": low,
            "close": close,
            "spread_pips": spread_pips,
            "current_utc_dt": ts_utc,
        }
        if (self._accepts_kwargs or "archetype" in self._param_names) and trade.get("archetype"):
            kwargs["archetype"] = trade["archetype"]
        if self._accepts_kwargs or "strategy_id" in self._param_names:
            kwargs["strategy_id"] = strat
        if self._accepts_kwargs or "spread_history" in self._param_names:
            kwargs["spread_history"] = spread_hist

        permitted, reason, _ = self.filter.evaluate_live_permission(**kwargs)
        result = (bool(permitted), str(reason))
        self._eval_cache[cache_key] = result
        return result


# ------------------------------------------------------------------------------
# Core Portfolio Simulation Engine
# ------------------------------------------------------------------------------
@dataclass
class OpenPosition:
    seq_id: int
    symbol: str
    strategy: str
    archetype: str
    direction: int
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    csv_pips: float
    sl_pips: float
    lot_size: float
    pip_size: float
    pip_val_per_lot: float
    commission_pips: float
    commission_usd: float


def compute_mtm_at_timestamp(
    pos: OpenPosition,
    eval_time_ms: int,
    timelines: Dict[str, M15Timeline],
    commission_per_lot_usd: float,
) -> Tuple[float, float, float, int]:
    """
    Computes strictly causal unrealized MTM `(pips, usd, exit_price, bar_close_ms)`
    for `pos` at `eval_time_ms` using only M15 bars with `T_close <= eval_time_ms`.
    """
    tl = timelines.get(pos.symbol)
    ba = tl.get_causal_bid_ask(eval_time_ms) if tl is not None else None

    if ba is not None and ba[2] >= pos.entry_time:
        bid_c, ask_c, bar_close_ms = ba
        exit_price = bid_c if pos.direction == 1 else ask_c
        raw_pips = (
            (exit_price - pos.entry_price) / pos.pip_size
            if pos.direction == 1
            else (pos.entry_price - exit_price) / pos.pip_size
        )
    else:
        # No closed M15 bar at or after entry_time yet: mark causally at entry_price
        exit_price = pos.entry_price
        raw_pips = 0.0
        bar_close_ms = pos.entry_time

    unrealized_pips = raw_pips - pos.commission_pips
    unrealized_usd = (raw_pips * pos.pip_val_per_lot * pos.lot_size) - (commission_per_lot_usd * pos.lot_size)
    return float(unrealized_pips), float(unrealized_usd), float(exit_price), int(bar_close_ms)


class _PortfolioLedger:
    """Single source of truth for chronological cash/equity/DD/monthly accounting."""

    def __init__(self, initial_equity: float, commission_per_lot_usd: float):
        self.initial_equity = float(initial_equity)
        self.commission_per_lot_usd = float(commission_per_lot_usd)
        self.equity = float(initial_equity)
        self.peak_equity = float(initial_equity)
        self.max_dd_usd = 0.0
        self.max_dd_pct = 0.0
        self.closed_records: List[Dict[str, Any]] = []
        self.daily_pnl_usd: Dict[str, float] = defaultdict(float)

    def record_close(
        self,
        pos: OpenPosition,
        actual_exit_time: int,
        exit_price: float,
        realized_pips: float,
        realized_usd: float,
        close_reason: str,
        mtm_bar_close_ms: Optional[int] = None,
    ) -> None:
        self.equity += float(realized_usd)
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        dd_usd = self.peak_equity - self.equity
        dd_pct = (dd_usd / self.peak_equity) * 100.0 if self.peak_equity > 0 else 0.0
        if dd_usd > self.max_dd_usd:
            self.max_dd_usd = dd_usd
        if dd_pct > self.max_dd_pct:
            self.max_dd_pct = dd_pct

        exit_dt = datetime.fromtimestamp(int(actual_exit_time) / 1000.0, tz=timezone.utc)
        day_str = exit_dt.strftime("%Y-%m-%d")
        month_str = exit_dt.strftime("%Y-%m")
        self.daily_pnl_usd[day_str] += float(realized_usd)

        self.closed_records.append({
            "seq_id": pos.seq_id,
            "symbol": pos.symbol,
            "strategy": pos.strategy,
            "direction": pos.direction,
            "entry_time": pos.entry_time,
            "exit_time": int(actual_exit_time),
            "scheduled_exit_time": pos.exit_time,
            "lot_size": pos.lot_size,
            "sl_pips": pos.sl_pips,
            "entry_price": pos.entry_price,
            "exit_price": float(exit_price),
            "realized_pips": float(realized_pips),
            "realized_usd": float(realized_usd),
            "commission_usd": float(pos.commission_usd),
            "close_reason": close_reason,
            "mtm_bar_close_ms": mtm_bar_close_ms,
            "month": month_str,
        })

    def close_natural(self, pos: OpenPosition) -> None:
        raw_pips = pos.csv_pips + get_commission_pips(pos.symbol, DEFAULT_DOLLAR_PER_LOT)
        realized_pips = raw_pips - pos.commission_pips
        realized_usd = (raw_pips * pos.pip_val_per_lot * pos.lot_size) - pos.commission_usd
        self.record_close(
            pos=pos,
            actual_exit_time=pos.exit_time,
            exit_price=pos.exit_price,
            realized_pips=realized_pips,
            realized_usd=realized_usd,
            close_reason="NATURAL",
        )

    def close_causal_mtm(
        self,
        pos: OpenPosition,
        eval_time_ms: int,
        timelines: Dict[str, M15Timeline],
        close_reason: str,
    ) -> Tuple[float, float]:
        pips, usd, exit_price, bar_close_ms = compute_mtm_at_timestamp(
            pos=pos,
            eval_time_ms=eval_time_ms,
            timelines=timelines,
            commission_per_lot_usd=self.commission_per_lot_usd,
        )
        self.record_close(
            pos=pos,
            actual_exit_time=eval_time_ms,
            exit_price=exit_price,
            realized_pips=pips,
            realized_usd=usd,
            close_reason=close_reason,
            mtm_bar_close_ms=bar_close_ms,
        )
        return pips, usd


def _build_trading_days(
    closed_records: List[Dict[str, Any]],
    timelines: Dict[str, M15Timeline],
) -> List[str]:
    if not closed_records:
        return []
    min_ts = min(r["entry_time"] for r in closed_records)
    max_ts = max(r["exit_time"] for r in closed_records)
    min_day_int = int(min_ts // 86_400_000)
    max_day_int = int(max_ts // 86_400_000)

    day_ints = set()
    for tl in timelines.values():
        if len(tl.unique_day_ints) > 0:
            lo = int(np.searchsorted(tl.unique_day_ints, min_day_int, side="left"))
            hi = int(np.searchsorted(tl.unique_day_ints, max_day_int, side="right"))
            if hi > lo:
                day_ints.update(tl.unique_day_ints[lo:hi].tolist())

    timeline_days = {
        datetime.fromtimestamp(int(d) * 86400, tz=timezone.utc).strftime("%Y-%m-%d")
        for d in day_ints
    }
    for r in closed_records:
        timeline_days.add(datetime.fromtimestamp(int(r["exit_time"]) / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d"))

    if timeline_days:
        return sorted(timeline_days)

    start_d = pd.Timestamp(min_ts, unit="ms", tz="UTC").normalize()
    end_d = pd.Timestamp(max_ts, unit="ms", tz="UTC").normalize()
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start_d, end_d)]


def _compute_metrics_summary(
    ledger: _PortfolioLedger,
    timelines: Dict[str, M15Timeline],
    execution_mode: str,
    sizing_mode: str,
    max_slots: int,
    timeout_hours: float,
    max_holding_hours: Optional[float],
    counters: Dict[str, Any],
) -> Dict[str, Any]:
    records = ledger.closed_records
    n_closed = len(records)
    net_usd = ledger.equity - ledger.initial_equity
    return_pct = (net_usd / ledger.initial_equity) * 100.0 if ledger.initial_equity > 0 else 0.0
    net_pips = sum(r["realized_pips"] for r in records)

    wins_usd = [r["realized_usd"] for r in records if r["realized_usd"] > 0.0]
    losses_usd = [r["realized_usd"] for r in records if r["realized_usd"] < 0.0]
    gross_win_usd = sum(wins_usd)
    gross_loss_usd = abs(sum(losses_usd))
    pf = (gross_win_usd / gross_loss_usd) if gross_loss_usd > 0 else (99.99 if gross_win_usd > 0 else 0.0)
    wr_pct = (len(wins_usd) / n_closed * 100.0) if n_closed > 0 else 0.0

    # Daily percentage returns across all active trading days -> Annualized Daily Sharpe
    trading_days = _build_trading_days(records, timelines)
    daily_rets = []
    eq_cursor = ledger.initial_equity
    for d_str in trading_days:
        pnl_d = ledger.daily_pnl_usd.get(d_str, 0.0)
        ret_d = (pnl_d / eq_cursor) if eq_cursor > 0 else 0.0
        daily_rets.append(ret_d)
        eq_cursor += pnl_d

    daily_sharpe = 0.0
    if len(daily_rets) > 1:
        arr = np.asarray(daily_rets, dtype=np.float64)
        std_r = float(np.std(arr, ddof=1))
        if std_r > 1e-12:
            daily_sharpe = float((np.mean(arr) / std_r) * math.sqrt(252.0))

    # Monthly breakdown with compounding monthly return %
    monthly_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        monthly_groups[r["month"]].append(r)

    monthly_breakdown: Dict[str, Dict[str, Any]] = {}
    m_equity = ledger.initial_equity
    for m_key in sorted(monthly_groups.keys()):
        m_recs = monthly_groups[m_key]
        m_net_usd = sum(r["realized_usd"] for r in m_recs)
        m_net_pips = sum(r["realized_pips"] for r in m_recs)
        m_wins = [r["realized_usd"] for r in m_recs if r["realized_usd"] > 0.0]
        m_losses = [r["realized_usd"] for r in m_recs if r["realized_usd"] < 0.0]
        m_gw = sum(m_wins)
        m_gl = abs(sum(m_losses))
        m_pf = (m_gw / m_gl) if m_gl > 0 else (99.99 if m_gw > 0 else 0.0)
        m_ret_pct = (m_net_usd / m_equity * 100.0) if m_equity > 0 else 0.0
        m_equity += m_net_usd

        monthly_breakdown[m_key] = {
            "net_usd": round(m_net_usd, 2),
            "return_pct": round(m_ret_pct, 2),
            "net_pips": round(m_net_pips, 1),
            "trades": len(m_recs),
            "wins": len(m_wins),
            "win_rate_pct": round(len(m_wins) / len(m_recs) * 100.0, 1) if m_recs else 0.0,
            "profit_factor": round(m_pf, 2),
            "cut_trades": sum(1 for r in m_recs if r["close_reason"] == "ROOM_MAKING_CUT"),
            "zombie_closes": sum(1 for r in m_recs if r["close_reason"] == "ZOMBIE_24H_CLOSE"),
        }

    # Symbol breakdown
    sym_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        sym_groups[r["symbol"]].append(r)

    symbol_breakdown: Dict[str, Dict[str, Any]] = {}
    for sym in sorted(sym_groups.keys()):
        s_recs = sym_groups[sym]
        s_wins = [r["realized_usd"] for r in s_recs if r["realized_usd"] > 0.0]
        s_losses = [r["realized_usd"] for r in s_recs if r["realized_usd"] < 0.0]
        s_gw = sum(s_wins)
        s_gl = abs(sum(s_losses))
        s_pf = (s_gw / s_gl) if s_gl > 0 else (99.99 if s_gw > 0 else 0.0)
        symbol_breakdown[sym] = {
            "trades": len(s_recs),
            "net_usd": round(sum(r["realized_usd"] for r in s_recs), 2),
            "net_pips": round(sum(r["realized_pips"] for r in s_recs), 1),
            "win_rate_pct": round(len(s_wins) / len(s_recs) * 100.0, 1),
            "profit_factor": round(s_pf, 2),
            "cut_trades": sum(1 for r in s_recs if r["close_reason"] == "ROOM_MAKING_CUT"),
        }

    total_incoming = counters["total_incoming"]
    accepted = counters["accepted_direct"] + counters["accepted_via_room_making"]

    return {
        "execution_mode": execution_mode,
        "sizing_mode": sizing_mode,
        "max_slots": max_slots,
        "timeout_hours": timeout_hours,
        "max_holding_hours": max_holding_hours,
        "initial_equity": round(ledger.initial_equity, 2),
        "final_equity": round(ledger.equity, 2),
        "net_usd": round(net_usd, 2),
        "return_pct": round(return_pct, 2),
        "net_pips": round(net_pips, 1),
        "max_dd_usd": round(ledger.max_dd_usd, 2),
        "max_dd_pct": round(ledger.max_dd_pct, 2),
        "daily_sharpe": round(daily_sharpe, 2),
        "calmar_ratio": round(return_pct / ledger.max_dd_pct, 2) if ledger.max_dd_pct > 0 else 0.0,
        "win_rate_pct": round(wr_pct, 2),
        "profit_factor": round(pf, 2),
        "total_incoming_trades": total_incoming,
        "accepted_trades": accepted,
        "accepted_direct": counters["accepted_direct"],
        "accepted_via_room_making": counters["accepted_via_room_making"],
        "acceptance_rate_pct": round((accepted / total_incoming * 100.0), 2) if total_incoming > 0 else 0.0,
        "blocked_trades": counters["blocked_trades"],
        "regime_blocked_trades": counters["regime_blocked_trades"],
        "cut_trades": counters["cut_trades"],
        "zombie_closes": counters["zombie_closes"],
        "natural_closes": sum(1 for r in records if r["close_reason"] == "NATURAL"),
        "max_observed_open_slots": counters["max_observed_open_slots"],
        "total_commission_usd": round(sum(r["commission_usd"] for r in records), 2),
        "regime_block_reasons": dict(sorted(counters["regime_block_reasons"].items(), key=lambda kv: -kv[1])),
        "monthly_breakdown": monthly_breakdown,
        "symbol_breakdown": symbol_breakdown,
    }


def simulate_portfolio(
    trades: List[Dict[str, Any]],
    timelines: Dict[str, M15Timeline],
    execution_mode: str = "causal_mtm_room_making",
    sizing_mode: str = "fixed_0.04",
    initial_equity: float = 5000.0,
    max_slots: int = 20,
    timeout_hours: float = 4.0,
    max_holding_hours: Optional[float] = 24.0,
    commission_per_lot_usd: float = DEFAULT_DOLLAR_PER_LOT,
    regime_adapter: Optional[CausalRegimeAdapter] = None,
    allow_young_loss_cut: bool = False,
    return_closed_trades: bool = False,
) -> Dict[str, Any]:
    """
    Runs strictly causal MTM 20-slot portfolio simulation across `trades`.

    `execution_mode`:
      - `"no_cut"`: Hard `max_slots` cap; zero room-making cuts and zero zombie closes.
      - `"causal_mtm_room_making"`: 24h zombie closes + cuts weakest open loser past
        `timeout_hours` (using causal M15 Bid/Ask at `T_close <= e_time`) when slots
        are full, strictly protecting winning MTM trades.
      - `"regime_filtered"`: Causal `LiveMarketRegimeFilter` pre-entry gate + causal
        MTM room-making.
    """
    mode = execution_mode.strip().lower()
    if mode not in {"no_cut", "causal_mtm_room_making", "regime_filtered"}:
        raise ValueError(f"Unknown execution_mode: {execution_mode}")

    enable_room_making = mode in {"causal_mtm_room_making", "regime_filtered"}
    use_regime = mode == "regime_filtered"
    effective_max_holding_hours = None if mode == "no_cut" else max_holding_hours

    if use_regime and regime_adapter is None:
        regime_adapter = CausalRegimeAdapter()

    # Enforce non-USDJPY exclusion + deterministic chronological ordering
    clean_trades = [t for t in trades if str(t.get("symbol", "")).upper() not in EXCLUDED_SYMBOLS]
    sorted_trades = sorted(
        clean_trades,
        key=lambda x: (int(x["entry_time"]), int(x.get("signal_time", 0)), str(x["symbol"]), str(x["strategy"])),
    )

    timeout_ms = int(round(float(timeout_hours) * 3600 * 1000))
    max_holding_ms = (
        int(round(float(effective_max_holding_hours) * 3600 * 1000))
        if effective_max_holding_hours is not None
        else None
    )

    ledger = _PortfolioLedger(initial_equity=initial_equity, commission_per_lot_usd=commission_per_lot_usd)
    active_slots: List[OpenPosition] = []

    counters: Dict[str, Any] = {
        "total_incoming": len(sorted_trades),
        "accepted_direct": 0,
        "accepted_via_room_making": 0,
        "blocked_trades": 0,
        "regime_blocked_trades": 0,
        "cut_trades": 0,
        "zombie_closes": 0,
        "max_observed_open_slots": 0,
        "regime_block_reasons": defaultdict(int),
    }

    for seq_id, t in enumerate(sorted_trades):
        e_time = int(t["entry_time"])
        sym = str(t["symbol"]).upper()

        # 1. Close trades whose natural exit_time <= e_time in strict exit_time order
        natural_exits = [pos for pos in active_slots if pos.exit_time <= e_time]
        if natural_exits:
            natural_exits.sort(key=lambda p: (p.exit_time, p.seq_id))
            exited_ids = {p.seq_id for p in natural_exits}
            for pos in natural_exits:
                ledger.close_natural(pos)
            active_slots = [pos for pos in active_slots if pos.seq_id not in exited_ids]

        # 2. Close 24h zombie positions still open at e_time (strictly causal MTM Bid/Ask)
        if max_holding_ms is not None and active_slots:
            surviving: List[OpenPosition] = []
            for pos in active_slots:
                if (e_time - pos.entry_time) >= max_holding_ms:
                    ledger.close_causal_mtm(pos, e_time, timelines, close_reason="ZOMBIE_24H_CLOSE")
                    counters["zombie_closes"] += 1
                else:
                    surviving.append(pos)
            active_slots = surviving

        # 3. Optional causal regime gate BEFORE checking slots / cutting existing positions
        if use_regime and regime_adapter is not None:
            permitted, reason = regime_adapter.check_permission(t, e_time, timelines)
            if not permitted:
                counters["regime_blocked_trades"] += 1
                counters["regime_block_reasons"][reason] += 1
                continue

        # 4. Capacity Check & Causal MTM Room-Making
        admitted_via_room_making = False
        if len(active_slots) >= max_slots:
            if not enable_room_making:
                counters["blocked_trades"] += 1
                continue

            timed_out_losers: List[Tuple[float, int, int]] = []
            young_losers: List[Tuple[float, int, int]] = []

            for idx, pos in enumerate(active_slots):
                unrealized_pips, _, _, _ = compute_mtm_at_timestamp(
                    pos=pos,
                    eval_time_ms=e_time,
                    timelines=timelines,
                    commission_per_lot_usd=commission_per_lot_usd,
                )
                # Winning MTM trades (> 0) are strictly shielded
                if unrealized_pips > 0.0:
                    continue

                hold_ms = e_time - pos.entry_time
                if hold_ms >= timeout_ms:
                    timed_out_losers.append((unrealized_pips, -hold_ms, idx))
                elif allow_young_loss_cut and unrealized_pips < 0.0:
                    young_losers.append((unrealized_pips, -hold_ms, idx))

            cut_pool = timed_out_losers if timed_out_losers else young_losers
            if not cut_pool:
                counters["blocked_trades"] += 1
                continue

            cut_pool.sort(key=lambda item: (item[0], item[1], active_slots[item[2]].seq_id))
            cut_idx = cut_pool[0][2]
            victim = active_slots.pop(cut_idx)
            ledger.close_causal_mtm(victim, e_time, timelines, close_reason="ROOM_MAKING_CUT")
            counters["cut_trades"] += 1
            admitted_via_room_making = True

        # 5. Size and admit new trade
        pip_size = get_pip_size(sym)
        pip_val_per_lot = get_pip_value_usd(sym)
        comm_pips = get_commission_pips(sym, commission_per_lot_usd)
        sl_pips = float(t.get("sl_pips", 20.0))
        lots = compute_position_lots(
            sizing_mode=sizing_mode,
            equity=ledger.equity,
            sl_pips=sl_pips,
            pip_val_per_lot=pip_val_per_lot,
        )
        comm_usd = float(commission_per_lot_usd) * lots

        new_pos = OpenPosition(
            seq_id=seq_id,
            symbol=sym,
            strategy=str(t.get("strategy", "unknown")),
            archetype=str(t.get("archetype", "")),
            direction=int(t["direction"]),
            entry_time=e_time,
            exit_time=int(t["exit_time"]),
            entry_price=float(t["entry_price"]),
            exit_price=float(t["exit_price"]),
            csv_pips=float(t["pips"]),
            sl_pips=sl_pips,
            lot_size=lots,
            pip_size=pip_size,
            pip_val_per_lot=pip_val_per_lot,
            commission_pips=comm_pips,
            commission_usd=comm_usd,
        )
        active_slots.append(new_pos)
        if len(active_slots) > counters["max_observed_open_slots"]:
            counters["max_observed_open_slots"] = len(active_slots)

        if admitted_via_room_making:
            counters["accepted_via_room_making"] += 1
        else:
            counters["accepted_direct"] += 1

    # 6. Close remaining active trades at their natural exit_time in chronological order
    if active_slots:
        active_slots.sort(key=lambda p: (p.exit_time, p.seq_id))
        for pos in active_slots:
            ledger.close_natural(pos)
        active_slots.clear()

    summary = _compute_metrics_summary(
        ledger=ledger,
        timelines=timelines,
        execution_mode=mode,
        sizing_mode=sizing_mode,
        max_slots=max_slots,
        timeout_hours=timeout_hours,
        max_holding_hours=effective_max_holding_hours,
        counters=counters,
    )
    if return_closed_trades:
        summary["closed_trades"] = ledger.closed_records
    return summary


# ------------------------------------------------------------------------------
# CLI Matrix Runner
# ------------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly Causal MTM 20-Slot Portfolio Simulator for v5.0 Non-USDJPY Fleet"
    )
    parser.add_argument(
        "--fleet-dir",
        type=str,
        default=None,
        help="Fleet directory containing FLEET_MANIFEST.json and/or holdout CSVs (e.g. v5.0)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        nargs="+",
        default=None,
        help="One or more fleet manifest/summary JSON paths",
    )
    parser.add_argument(
        "--trades-dir",
        type=str,
        nargs="+",
        default=None,
        help="One or more directories containing *_holdout_trades.csv / *_trades.csv",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["no_cut", "causal_mtm_room_making", "regime_filtered"],
        choices=["no_cut", "causal_mtm_room_making", "regime_filtered"],
        help="Execution modes to simulate",
    )
    parser.add_argument(
        "--sizing",
        type=str,
        nargs="+",
        default=["fixed_0.03", "fixed_0.04", "risk_0.5pct"],
        help="Sizing modes (e.g. fixed_0.03 fixed_0.04 risk_0.5pct)",
    )
    parser.add_argument("--initial-equity", type=float, default=5000.0, help="Initial account equity USD")
    parser.add_argument("--max-slots", type=int, default=20, help="Maximum simultaneous open positions")
    parser.add_argument("--timeout-hours", type=float, default=4.0, help="Stagnant loss timeout before cut")
    parser.add_argument("--max-holding-hours", type=float, default=24.0, help="Hard zombie holding limit")
    parser.add_argument("--commission-per-lot", type=float, default=DEFAULT_DOLLAR_PER_LOT, help="USD per lot round-turn")
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR), help="M15 cache directory")
    parser.add_argument("--no-cache", action="store_true", help="Disable M15 disk cache")
    parser.add_argument("--rebuild-cache", action="store_true", help="Force rebuild M15 disk cache")
    parser.add_argument("--allow-young-loss-cut", action="store_true", help="Allow cutting losses younger than timeout")
    parser.add_argument(
        "--include-unlisted-csvs",
        action="store_true",
        help="When --manifest is given, also ingest holdout CSVs in --trades-dir not listed in the manifest",
    )
    parser.add_argument(
        "--allow-pre-holdout",
        action="store_true",
        help="Allow trades predating Week 211 M15 timeline start (default filters to Holdout window)",
    )
    parser.add_argument(
        "--regime-module",
        type=str,
        default=None,
        help="Optional path to live_regime_filter.py or MARKET_REGIME_SYSTEM.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output JSON report path",
    )
    return parser.parse_args(argv)


def _auto_discover_inputs(args: argparse.Namespace) -> None:
    if args.fleet_dir or args.manifest or args.trades_dir:
        return
    v5_dir = ROOT_DIR / "v5.0"
    if v5_dir.exists():
        args.fleet_dir = str(v5_dir)
        return
    cand_dirs = [
        d for d in (
            ROOT_DIR / "results" / "v5_candidates_majors" / "manifests",
            ROOT_DIR / "results" / "v5_candidates_crosses" / "manifests",
        )
        if d.exists()
    ]
    if cand_dirs:
        args.trades_dir = [str(d) for d in cand_dirs]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _auto_discover_inputs(args)
    t_start = time.time()

    print("=" * 96)
    print("  STRICTLY CAUSAL MTM 20-SLOT PORTFOLIO SIMULATOR (v5.0 NON-USDJPY FLEET)")
    print("=" * 96)

    trades, load_meta = load_portfolio_trades(
        manifest_paths=[Path(p) for p in args.manifest] if args.manifest else None,
        trades_dirs=[Path(d) for d in args.trades_dir] if args.trades_dir else None,
        fleet_dir=Path(args.fleet_dir) if args.fleet_dir else None,
        include_unlisted_csvs=args.include_unlisted_csvs,
    )
    print(
        f"Loaded {load_meta['total_holdout_trades']:,} holdout trades across "
        f"{load_meta['strategies_loaded']} non-USDJPY strategies "
        f"(Symbols: {', '.join(load_meta['symbols_present'])})"
    )
    if load_meta["excluded_usdjpy_strategies"] or load_meta["excluded_usdjpy_trades"]:
        print(
            f"Excluded USDJPY: {len(load_meta['excluded_usdjpy_strategies'])} strategies, "
            f"{load_meta['excluded_usdjpy_trades']} trades"
        )
    if load_meta["missing_csv_strategies"]:
        print(f"Warning: {len(load_meta['missing_csv_strategies'])} strategies lacked trade CSVs")

    if not trades:
        print("Error: No trades loaded to simulate.")
        return 1

    cache_dir = None if args.no_cache else Path(args.cache_dir)
    print("\nLoading causal M15 Bid/Ask timelines (Holdout Weeks 211..300)...")
    timelines = load_timelines_for_symbols(
        symbols=load_meta["symbols_present"],
        cache_dir=cache_dir,
        force_rebuild=args.rebuild_cache,
        verbose=True,
    )

    pre_holdout_filtered = 0
    if not args.allow_pre_holdout and timelines:
        non_empty_starts = [int(tl.timestamp_ms[0]) - M15_MS for tl in timelines.values() if len(tl.timestamp_ms) > 0]
        if non_empty_starts:
            holdout_start_ms = min(non_empty_starts)
            kept_trades = [t for t in trades if int(t["entry_time"]) >= holdout_start_ms]
            pre_holdout_filtered = len(trades) - len(kept_trades)
            if pre_holdout_filtered > 0:
                print(
                    f"Holdout Window Filter: Excluded {pre_holdout_filtered} pre-Week-211 trades "
                    f"(before {datetime.fromtimestamp(holdout_start_ms / 1000.0, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})."
                )
                trades = kept_trades
                load_meta["total_holdout_trades"] = len(trades)
    load_meta["pre_holdout_trades_filtered"] = pre_holdout_filtered

    if not trades:
        print("Error: No trades remaining inside Holdout window.")
        return 1

    regime_adapter: Optional[CausalRegimeAdapter] = None
    regime_available = True
    if "regime_filtered" in args.modes:
        try:
            regime_adapter = CausalRegimeAdapter(
                regime_module_path=Path(args.regime_module) if args.regime_module else None
            )
        except Exception as exc:
            regime_available = False
            print(f"Warning: Could not initialize LiveMarketRegimeFilter ({exc}); skipping regime_filtered mode.")

    scenarios: Dict[str, Dict[str, Any]] = {}
    print("\n" + "-" * 96)
    print(
        f"{'Scenario':<38s} | {'Net USD':>11s} | {'ROI %':>8s} | {'Max DD':>14s} | "
        f"{'Sharpe':>6s} | {'WR %':>6s} | {'PF':>5s} | {'Acc/Blk/Cut':>15s}"
    )
    print("-" * 96)

    for mode in args.modes:
        if mode == "regime_filtered" and not regime_available:
            continue
        for sizing in args.sizing:
            key = f"{mode}__{sizing}"
            res = simulate_portfolio(
                trades=trades,
                timelines=timelines,
                execution_mode=mode,
                sizing_mode=sizing,
                initial_equity=args.initial_equity,
                max_slots=args.max_slots,
                timeout_hours=args.timeout_hours,
                max_holding_hours=args.max_holding_hours,
                commission_per_lot_usd=args.commission_per_lot,
                regime_adapter=regime_adapter if mode == "regime_filtered" else None,
                allow_young_loss_cut=args.allow_young_loss_cut,
            )
            scenarios[key] = res
            abc = f"{res['accepted_trades']}/{res['blocked_trades']}/{res['cut_trades']}"
            dd_str = f"${res['max_dd_usd']:,.0f} ({res['max_dd_pct']:.2f}%)"
            print(
                f"{key:<38s} | ${res['net_usd']:>+10,.2f} | {res['return_pct']:>+7.2f}% | "
                f"{dd_str:>14s} | {res['daily_sharpe']:>6.2f} | {res['win_rate_pct']:>5.1f}% | "
                f"{res['profit_factor']:>5.2f} | {abc:>15s}"
            )

    report = {
        "metadata": {
            "simulator": "v5_strictly_causal_mtm_20_slot",
            "generated_utc": pd.Timestamp.now("UTC").isoformat(),
            "elapsed_sec": round(time.time() - t_start, 2),
            "initial_equity": args.initial_equity,
            "max_slots": args.max_slots,
            "timeout_hours": args.timeout_hours,
            "max_holding_hours": args.max_holding_hours,
            "commission_per_lot_usd": args.commission_per_lot,
            "allow_young_loss_cut": args.allow_young_loss_cut,
            "usdjpy_excluded": True,
        },
        "input_summary": load_meta,
        "scenarios": scenarios,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("-" * 96)
    print(f"Saved portfolio simulation report to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
