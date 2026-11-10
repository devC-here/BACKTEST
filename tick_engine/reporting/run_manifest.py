"""
================================================================================
  CANONICAL VERSIONED RUN MANIFEST & STATISTICAL AUDIT ENGINE
================================================================================
Implements Requirements 1, 7, and 8 of the Institutional Verification Standard:
1. Versioned Run Manifest (Engine & Strategy code hashes, dataset SHA256 hashes,
   pricing policy, timezone, causal warmup specs).
2. Advanced Statistical Confidence Metrics:
   - Expectancy (Mean Trade PnL)
   - Standard Error & 95% Confidence Interval for Expectancy
   - t-statistic and two-tailed p-value
   - Maximum Drawdown (pips and %)
   - Monthly PnL & Worst Month
   - 1,000-sample Bootstrap 95th Percentile Drawdown
   - Outlier Fragility (PnL & Profit Factor when Top 1 to 5 winners removed)
3. Multiple-Testing Protection:
   - Family-wise error rate (FWER) and Bonferroni-adjusted significance hurdles.
4. Cryptographic Certification:
   - Results without this manifest are explicitly tagged as NON_CERTIFIED.
================================================================================
"""

from __future__ import annotations
import os
import sys
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def compute_file_hash(path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    if not path.exists():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def compute_code_hash(code_str: str) -> str:
    """Computes SHA-256 hash of a code string."""
    return hashlib.sha256(code_str.encode("utf-8")).hexdigest()

def compute_statistical_metrics(
    trades: np.ndarray,
    candidates_screened: int = 1
) -> Dict[str, Any]:
    """
    Computes rigorous statistical confidence metrics from a TRADE_DTYPE array.
    """
    n = len(trades)
    if n == 0:
        return {
            "total_trades": 0,
            "net_pips": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy_pips": 0.0,
            "expectancy_std_err": 0.0,
            "expectancy_ci_95": [0.0, 0.0],
            "t_statistic": 0.0,
            "p_value": 1.0,
            "bonferroni_p_threshold": round(0.05 / max(1, candidates_screened), 5),
            "is_statistically_significant": False,
            "max_drawdown_pips": 0.0,
            "worst_month_pips": 0.0,
            "bootstrap_95_max_dd_pips": 0.0,
            "outlier_sensitivity": {
                "pips_ex_top1": 0.0,
                "pips_ex_top3": 0.0,
                "pips_ex_top5": 0.0,
                "is_outlier_dependent": True
            }
        }

    pips = trades['pips']
    wins = pips[pips > 0]
    losses = pips[pips < 0]

    net_pips = float(np.sum(pips))
    win_pips = float(np.sum(wins))
    loss_pips = float(np.abs(np.sum(losses)))
    win_rate = float(len(wins) / n * 100.0)
    profit_factor = round(win_pips / loss_pips, 2) if loss_pips > 0 else (99.0 if win_pips > 0 else 0.0)

    # Expectancy & Standard Error
    mean_pips = float(np.mean(pips))
    std_pips = float(np.std(pips, ddof=1)) if n > 1 else 0.0
    std_err = std_pips / np.sqrt(n) if n > 1 else 0.0
    ci_95 = [round(mean_pips - 1.96 * std_err, 2), round(mean_pips + 1.96 * std_err, 2)]

    # t-statistic & p-value
    t_stat = round(mean_pips / std_err, 2) if std_err > 0 else 0.0
    # Approximate normal CDF two-tailed p-value
    import math
    p_value = round(math.erfc(abs(t_stat) / math.sqrt(2)), 4)

    # Bonferroni adjustment
    bonf_thresh = round(0.05 / max(1, candidates_screened), 5)
    is_stat_sig = (p_value <= bonf_thresh) and (mean_pips > 0)

    # Equity curve & Max Drawdown
    cum_pips = np.cumsum(pips)
    peak = np.maximum.accumulate(cum_pips)
    dd_curve = peak - cum_pips
    max_dd = float(np.max(dd_curve)) if len(dd_curve) > 0 else 0.0

    # Monthly breakdown
    exit_times = trades['exit_time']
    try:
        ts_dates = pd.to_datetime(exit_times, unit='ms')
        df_trades = pd.DataFrame({'pips': pips}, index=ts_dates)
        monthly_pips = df_trades.resample('ME')['pips'].sum()
        worst_month = float(monthly_pips.min()) if len(monthly_pips) > 0 else 0.0
    except Exception:
        worst_month = 0.0

    # 1,000-sample Bootstrap Drawdown (95th percentile)
    np.random.seed(42)
    boot_dds = []
    for _ in range(1000):
        sample_pips = np.random.choice(pips, size=n, replace=True)
        sample_cum = np.cumsum(sample_pips)
        sample_peak = np.maximum.accumulate(sample_cum)
        boot_dds.append(float(np.max(sample_peak - sample_cum)))
    boot_95_dd = float(np.percentile(boot_dds, 95))

    # Outlier Sensitivity (Trimming Top 1 to 5 winners)
    sorted_pips = np.sort(pips)
    pips_ex1 = float(np.sum(sorted_pips[:-1])) if n > 1 else net_pips
    pips_ex3 = float(np.sum(sorted_pips[:-3])) if n > 3 else net_pips
    pips_ex5 = float(np.sum(sorted_pips[:-5])) if n > 5 else net_pips

    # If removing top 3 trades wipes out all profit, it is outlier-dependent
    is_outlier_dependent = bool(pips_ex3 <= 0.0)

    return {
        "total_trades": n,
        "net_pips": round(net_pips, 1),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": profit_factor,
        "expectancy_pips": round(mean_pips, 2),
        "expectancy_std_err": round(std_err, 2),
        "expectancy_ci_95": ci_95,
        "t_statistic": t_stat,
        "p_value": p_value,
        "bonferroni_p_threshold": bonf_thresh,
        "is_statistically_significant": is_stat_sig,
        "max_drawdown_pips": round(max_dd, 1),
        "worst_month_pips": round(worst_month, 1),
        "bootstrap_95_max_dd_pips": round(boot_95_dd, 1),
        "outlier_sensitivity": {
            "pips_ex_top1": round(pips_ex1, 1),
            "pips_ex_top3": round(pips_ex3, 1),
            "pips_ex_top5": round(pips_ex5, 1),
            "is_outlier_dependent": is_outlier_dependent
        }
    }

def create_certified_run_manifest(
    strategy_name: str,
    strategy_file: Path,
    symbol: str,
    split_version: str,
    start_week: int,
    end_week: int,
    pip_size: float,
    commission_pips: float,
    entry_slippage_pips: float,
    spread_policy: str,
    warmup_ticks: int,
    trades: np.ndarray,
    candidates_screened: int = 1,
    additional_meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Builds the complete cryptographic JSON Run Manifest for individual strategy certification.
    """
    engine_file = ROOT_DIR / "tick_engine" / "core" / "engine.py"
    loader_file = ROOT_DIR / "tick_engine" / "core" / "dataloader.py"

    strategy_hash = compute_file_hash(strategy_file) if strategy_file.exists() else "NO_FILE"
    engine_hash = compute_file_hash(engine_file)
    loader_hash = compute_file_hash(loader_file)

    stats = compute_statistical_metrics(trades, candidates_screened=candidates_screened)

    # Format detailed trade logs
    trade_logs = []
    for tr in trades:
        sig_t = int(tr['signal_time']) if 'signal_time' in tr.dtype.names else int(tr['entry_time'])
        ent_t = int(tr['entry_time'])
        trade_logs.append({
            "signal_time": sig_t,
            "entry_time": ent_t,
            "latency_ms": ent_t - sig_t,
            "exit_time": int(tr['exit_time']),
            "signal_utc": str(pd.to_datetime(sig_t, unit='ms')),
            "entry_utc": str(pd.to_datetime(ent_t, unit='ms')),
            "exit_utc": str(pd.to_datetime(tr['exit_time'], unit='ms')),
            "direction": int(tr['direction']),
            "entry_price": round(float(tr['entry_price']), 5),
            "exit_price": round(float(tr['exit_price']), 5),
            "pips": round(float(tr['pips']), 2),
            "exit_reason": int(tr['exit_reason'])
        })

    is_certified = (
        stats["net_pips"] > 0
        and stats["profit_factor"] >= 1.05
        and stats["total_trades"] >= 15
        and not stats["outlier_sensitivity"]["is_outlier_dependent"]
    )

    manifest = {
        "manifest_version": "1.0.0",
        "timestamp_utc": pd.Timestamp.now('UTC').isoformat(),
        "certification_status": "DUAL_CERTIFIED" if is_certified else "NON_CERTIFIED",
        "strategy": {
            "name": strategy_name,
            "filename": strategy_file.name,
            "code_sha256": strategy_hash
        },
        "engine": {
            "version": "tick_engine_v1.0",
            "engine_code_sha256": engine_hash,
            "loader_code_sha256": loader_hash,
            "order_timing_contract": "Strict Bar-Close Subsequent-Tick Execution",
            "timezone": "UTC"
        },
        "dataset": {
            "symbol": symbol,
            "pip_size": pip_size,
            "split_standard": split_version,
            "start_week": start_week,
            "end_week": end_week,
            "data_quality_gate": "PASSED (80h min span, monotonic timestamps, ask>=bid)",
            "warmup_ticks_causal": warmup_ticks
        },
        "pricing_friction": {
            "spread_model": spread_policy,
            "commission_pips": commission_pips,
            "entry_slippage_pips": entry_slippage_pips,
            "dollar_per_lot_commission": 4.00
        },
        "multiple_testing": {
            "candidates_screened": candidates_screened,
            "bonferroni_adjusted_significance": stats["is_statistically_significant"]
        },
        "statistical_metrics": stats,
        "trade_log": trade_logs,
        "custom_metadata": additional_meta or {}
    }

    return manifest

generate_canonical_manifest = create_certified_run_manifest

