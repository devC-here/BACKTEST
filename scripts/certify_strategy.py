#!/usr/bin/env python3
"""
================================================================================
  INSTITUTIONAL STRATEGY CERTIFICATION RUNNER (CLI)
================================================================================
The canonical verification harness enforcing all Institutional Reliability Gates:
1. Canonical Versioned Run Manifest with cryptographic SHA256 code & data hashes.
2. Proper Causal Warmup (2 prior weeks rolling warmup; exact warmup_tick_count gate).
3. Automated Dataset Quality Gate (strictly verified complete weeks >=80h).
4. Enforced Order-Timing Contract (signal_time vs entry_time, fill_ts > signal_ts).
5. Genuine 5-Profile Broker-Cost Stress Testing (Baseline, +50% Spread, 0.50p Slip, $6 Comm, Composite).
6. Mode-Isolated Manifests & Trade Logs (Discovery never overwrites Holdout).
7. Statistical Confidence (Expectancy 95% CI, t-stat, p-value, Bootstrap DD, Outliers).
8. Multiple-Testing Protection (Bonferroni / FWER hurdle calculation).
================================================================================
"""

from __future__ import annotations
import os
import sys
import json
import time
import argparse
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import (
    run_matching_loop,
    DEFAULT_COMMISSION_PIPS,
    DEFAULT_ENTRY_SLIPPAGE_PIPS
)
from tick_engine.core.dataloader import (
    get_week_dataset,
    get_complete_split_weeks
)
from tick_engine.stress.stress_suite import STRESS_PROFILES
from tick_engine.reporting.run_manifest import create_certified_run_manifest
from tick_engine.core.config import PIP_SIZES, get_friction_pips

DEFAULT_MANIFEST_DIR = ROOT_DIR / "results" / "certified_strategy_manifests"
DEFAULT_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
WARMUP_TICKS = 100000
WARMUP_PREV_WEEKS = 2


def load_strategy_module(file_path: Path):
    """Dynamically imports strategy module and extracts the strategy class."""
    if not file_path.exists():
        raise FileNotFoundError(f"Strategy file not found: {file_path}")

    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec from: {file_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    strat_cls = getattr(mod, "Strategy", None)
    if strat_cls is None or not hasattr(strat_cls, "generate_signals"):
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if isinstance(obj, type) and hasattr(obj, "generate_signals") and attr != "BaseStrategy":
                strat_cls = obj
                break

    if strat_cls is None or not hasattr(strat_cls, "generate_signals"):
        raise AttributeError(f"No valid Strategy class with generate_signals() found in {file_path}")

    return strat_cls


def _instantiate_strategy(strat_cls, symbol: str, pip_size: float):
    try:
        return strat_cls(asset=symbol, symbol=symbol, pip_size=pip_size)
    except Exception:
        try:
            return strat_cls(asset=symbol, pip_size=pip_size)
        except Exception:
            try:
                return strat_cls(pip_size=pip_size)
            except Exception:
                return strat_cls()


def evaluate_week_causal(
    strat_cls,
    symbol: str,
    week_num: int,
    pip_size: float,
    comm_pips: float,
    slip_pips: float,
    enforce_subsequent_tick: bool = True,
    warmup_prev_weeks: int = WARMUP_PREV_WEEKS,
    run_stress_profiles: bool = False,
) -> Any:
    """
    Evaluates one complete trading week with prior-weeks rolling warmup.
    When `run_stress_profiles=False`, returns contiguous TRADE_DTYPE array.
    When `run_stress_profiles=True`, returns `(trades, stress_dict)`.
    """
    raw_dir = ROOT_DIR / "data" / symbol
    strat_inst = _instantiate_strategy(strat_cls, symbol, pip_size)

    try:
        ticks, meta = get_week_dataset(
            week=week_num,
            raw_dir=raw_dir,
            warmup_prev_weeks=warmup_prev_weeks,
        )
    except Exception:
        empty = np.empty(0)
        return (empty, {}) if run_stress_profiles else empty

    w_cnt = int(meta.get("warmup_tick_count", 0))
    if len(ticks) <= 500 or len(ticks) <= w_cnt:
        empty = np.empty(0)
        return (empty, {}) if run_stress_profiles else empty

    res = strat_inst.generate_signals(ticks)
    sig_ts = None
    if isinstance(res, tuple):
        signals = res[0]
        sl_arr = res[1] if len(res) > 1 and res[1] is not None else np.full(len(ticks), getattr(strat_inst, "sl_pips", 20.0))
        tp_arr = res[2] if len(res) > 2 and res[2] is not None else np.full(len(ticks), getattr(strat_inst, "tp_pips", 40.0))
        if len(res) > 3 and res[3] is not None:
            sig_ts = res[3]
    else:
        signals = res
        sl_arr = np.full(len(ticks), getattr(strat_inst, "sl_pips", 20.0))
        tp_arr = np.full(len(ticks), getattr(strat_inst, "tp_pips", 40.0))

    min_entry_ts = int(ticks["timestamp"][w_cnt]) if w_cnt > 0 and w_cnt < len(ticks) else 0

    if not np.any(signals != 0):
        empty = np.empty(0)
        return (empty, {}) if run_stress_profiles else empty

    trades = run_matching_loop(
        ticks=ticks,
        signals=signals,
        pip_size=pip_size,
        sl_pips=sl_arr,
        tp_pips=tp_arr,
        use_slippage=True,
        close_at_end=True,
        commission_pips=comm_pips,
        entry_slippage_pips=slip_pips,
        signal_timestamps=sig_ts,
        min_entry_timestamp=min_entry_ts,
        enforce_subsequent_tick=enforce_subsequent_tick,
    )

    if not run_stress_profiles:
        return trades

    week_stress = {"BASELINE": trades}
    eval_ticks_15x = None
    for p_name, prof in STRESS_PROFILES.items():
        if p_name == "BASELINE":
            continue
        spr_mult = prof["spread_multiplier"]
        if spr_mult != 1.0:
            if eval_ticks_15x is None:
                eval_ticks_15x = ticks.copy()
                mid = 0.5 * (ticks["bid"] + ticks["ask"])
                half_sp = 0.5 * (ticks["ask"] - ticks["bid"]) * spr_mult
                eval_ticks_15x["bid"] = mid - half_sp
                eval_ticks_15x["ask"] = mid + half_sp
            t_in = eval_ticks_15x
        else:
            t_in = ticks

        c_p, _ = get_friction_pips(symbol, dollar_per_lot=prof["commission_dollar_per_lot"])
        s_p = float(prof["entry_slippage_pips"])
        tr_p = run_matching_loop(
            ticks=t_in,
            signals=signals,
            pip_size=pip_size,
            sl_pips=sl_arr,
            tp_pips=tp_arr,
            use_slippage=True,
            close_at_end=True,
            commission_pips=c_p,
            entry_slippage_pips=s_p,
            signal_timestamps=sig_ts,
            min_entry_timestamp=min_entry_ts,
            enforce_subsequent_tick=enforce_subsequent_tick,
        )
        week_stress[p_name] = tr_p

    return trades, week_stress


def _aggregate_stress_results(stress_lists: Dict[str, List[np.ndarray]]) -> Dict[str, Any]:
    profiles_out = {}
    survived_all = True
    for p_name, prof in STRESS_PROFILES.items():
        arrs = [a for a in stress_lists.get(p_name, []) if len(a) > 0]
        comb = np.concatenate(arrs) if arrs else np.empty(0)
        n_tr = len(comb)
        if n_tr > 0:
            pips = comb["pips"]
            net = float(np.sum(pips))
            win_p = float(np.sum(pips[pips > 0]))
            loss_p = float(np.abs(np.sum(pips[pips < 0])))
            pf = round(win_p / loss_p, 2) if loss_p > 0 else (99.0 if win_p > 0 else 0.0)
        else:
            net, pf = 0.0, 0.0
        is_pass = bool(net > 0.0 and pf >= 1.00)
        if not is_pass:
            survived_all = False
        profiles_out[p_name] = {
            "description": prof["description"],
            "trades": n_tr,
            "net_pips": round(net, 1),
            "profit_factor": pf,
            "survived": is_pass,
        }
    return {"survived_all_stress_profiles": survived_all, "profiles": profiles_out}


def certify_strategy(
    strategy_file: Path,
    symbol: Optional[str] = None,
    mode: str = "holdout",
    candidates_screened: int = 1,
    run_stress: bool = True,
    run_walkforward: bool = True,
    output_dir: Path = DEFAULT_MANIFEST_DIR,
) -> Dict[str, Any]:
    """
    Executes end-to-end institutional certification on an individual strategy.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    strat_cls = load_strategy_module(strategy_file)
    tmp_inst = strat_cls()

    sym = symbol
    if not sym:
        for part in [strategy_file.parent.name] + [p.name for p in strategy_file.parents]:
            p_upper = part.upper()
            for s in PIP_SIZES.keys():
                if s in p_upper:
                    sym = s
                    break
            if sym:
                break
    if not sym:
        sym = getattr(tmp_inst, "symbol", None) or getattr(tmp_inst, "asset", None)
    if not sym:
        fn = strategy_file.stem.upper()
        for s in PIP_SIZES.keys():
            if s in fn:
                sym = s
                break
    if not sym:
        sym = "EURUSD"

    strategy_id = strategy_file.parent.name if strategy_file.name == "strategy.py" else strategy_file.stem
    pip_size = PIP_SIZES.get(sym, 0.0001)
    comm_pips, slip_pips = get_friction_pips(sym)

    print("=" * 80)
    print("  INSTITUTIONAL STRATEGY CERTIFICATION HARNESS (STRICTLY CAUSAL)")
    print("=" * 80)
    print(f"  Strategy:           {strategy_file.name}")
    print(f"  Symbol:             {sym} (Pip Size: {pip_size})")
    print(f"  Evaluation Mode:    {mode.upper()}")
    print(f"  Pricing Friction:   {comm_pips:.2f}p Commission ($4/lot) + {slip_pips:.2f}p Adverse Slippage")
    print(f"  Order-Timing:       Strict Subsequent-Tick Causal Execution (T_fill > T_signal)")
    print(f"  Warmup Discipline:  {WARMUP_PREV_WEEKS}-Week Rolling Prior-Week Warmup (Zero Pre-window Fills)")
    print("=" * 80)

    if mode == "holdout":
        target_weeks = get_complete_split_weeks(sym, "holdout_test_30")
        split_version = "canonical_continuous_holdout_30 (Weeks 211..300)"
        start_w, end_w = 211, 300
    elif mode == "discovery":
        target_weeks = get_complete_split_weeks(sym, "discovery_70")
        split_version = "canonical_continuous_discovery_70 (Weeks 1..210)"
        start_w, end_w = 1, 210
    else:
        target_weeks = get_complete_split_weeks(sym, "discovery_70") + get_complete_split_weeks(sym, "holdout_test_30")
        split_version = "canonical_continuous_full_300 (Weeks 1..300)"
        start_w, end_w = 1, 300

    print(f"Evaluating across {len(target_weeks)} quality-certified trading weeks...")

    all_trades_list = []
    stress_lists: Dict[str, List[np.ndarray]] = {k: [] for k in STRESS_PROFILES}
    t0 = time.time()

    for w in target_weeks:
        res = evaluate_week_causal(
            strat_cls=strat_cls,
            symbol=sym,
            week_num=w,
            pip_size=pip_size,
            comm_pips=comm_pips,
            slip_pips=slip_pips,
            enforce_subsequent_tick=True,
            warmup_prev_weeks=WARMUP_PREV_WEEKS,
            run_stress_profiles=run_stress,
        )
        if run_stress:
            trades, w_stress = res
            for p_name, p_tr in w_stress.items():
                if len(p_tr) > 0:
                    stress_lists[p_name].append(p_tr)
        else:
            trades = res
        if len(trades) > 0:
            all_trades_list.append(trades)

    elapsed = time.time() - t0
    combined_trades = np.concatenate(all_trades_list) if all_trades_list else np.empty(0)
    print(f"Simulation completed in {elapsed:.1f}s. Total Trades: {len(combined_trades)}")

    stress_results = None
    if run_stress and len(combined_trades) > 0:
        print("\nAggregating 5-Profile Broker-Cost Stress Battery...")
        stress_results = _aggregate_stress_results(stress_lists)
        for p_name, p_info in stress_results["profiles"].items():
            tag = "PASS" if p_info["survived"] else "FAIL"
            print(f"  [{tag}] {p_name:<22}: {p_info['trades']:3d} tr | PF {p_info['profit_factor']:4.2f} | {p_info['net_pips']:+7.1f}p")

    manifest = create_certified_run_manifest(
        strategy_name=strat_cls.__name__,
        strategy_file=strategy_file,
        symbol=sym,
        split_version=split_version,
        start_week=start_w,
        end_week=end_w,
        pip_size=pip_size,
        commission_pips=comm_pips,
        entry_slippage_pips=slip_pips,
        spread_policy="Floating Raw Dukascopy Bid/Ask",
        warmup_ticks=WARMUP_PREV_WEEKS,
        trades=combined_trades,
        candidates_screened=candidates_screened,
        additional_meta={
            "warmup_policy": f"{WARMUP_PREV_WEEKS}_prior_weeks_rolling",
            "stress_battery": stress_results,
            "certified_clean_weeks_count": len(target_weeks),
        },
    )

    # Mode-isolated outputs (discovery NEVER overwrites holdout!)
    mode_manifest_path = output_dir / f"{sym.upper()}__{strategy_id}_{mode}_manifest.json"
    with open(mode_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if len(combined_trades) > 0:
        df_trades = pd.DataFrame(manifest["trade_log"])
        mode_csv_path = output_dir / f"{sym.upper()}__{strategy_id}_{mode}_trades.csv"
        df_trades.to_csv(mode_csv_path, index=False)
        print(f"Saved Mode Trade Log: {mode_csv_path.name}")

    # Canonical Holdout filenames written strictly on holdout runs only
    if mode == "holdout":
        manifest_path = output_dir / f"{sym.upper()}__{strategy_id}_run_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        if len(combined_trades) > 0:
            csv_path = output_dir / f"{sym.upper()}__{strategy_id}_trades.csv"
            df_trades.to_csv(csv_path, index=False)
            print(f"Saved Canonical Holdout Trade Log: {csv_path.name}")
        print(f"Saved Canonical Holdout Manifest:  {manifest_path.name}")

    stats = manifest["statistical_metrics"]
    status = manifest["certification_status"]

    print("\n" + "=" * 80)
    print(f"  OFFICIAL INSTITUTIONAL SCORECARD: {status} ({mode.upper()})")
    print("=" * 80)
    print(f"  Total Trades:           {stats['total_trades']}")
    print(f"  Net Pips:               {stats['net_pips']:+.1f} pips")
    print(f"  Profit Factor:          {stats['profit_factor']:.2f}")
    print(f"  Win Rate:               {stats['win_rate_pct']:.1f}%")
    print(f"  Expectancy (Mean PnL):  {stats['expectancy_pips']:+.2f} pips/trade")
    print(f"  Expectancy 95% CI:      [{stats['expectancy_ci_95'][0]:+.2f}, {stats['expectancy_ci_95'][1]:+.2f}] pips")
    print(f"  t-statistic (p-value):  {stats['t_statistic']:.2f} (p={stats['p_value']:.4f})")
    print(f"  Max Drawdown:           {stats['max_drawdown_pips']:.1f} pips")
    print(f"  Bootstrap 95th DD:      {stats['bootstrap_95_max_dd_pips']:.1f} pips")
    print(f"  Worst Month:            {stats['worst_month_pips']:+.1f} pips")
    print(f"  Top 1 Winner Removed:   {stats['outlier_sensitivity']['pips_ex_top1']:+.1f} pips")
    print(f"  Top 3 Winners Removed:  {stats['outlier_sensitivity']['pips_ex_top3']:+.1f} pips")
    print(f"  Outlier Dependent:      {stats['outlier_sensitivity']['is_outlier_dependent']}")
    print("=" * 80 + "\n")

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Institutional Strategy Certification Runner")
    parser.add_argument("--strategy", type=str, required=True, help="Path to strategy python file")
    parser.add_argument("--symbol", type=str, default=None, help="Symbol ticker (e.g. USDCAD)")
    parser.add_argument("--mode", type=str, choices=["holdout", "discovery", "full"], default="holdout", help="Partition mode")
    parser.add_argument("--candidates", type=int, default=1, help="Total candidates screened (multiple-testing penalty)")
    args = parser.parse_args()

    certify_strategy(
        strategy_file=Path(args.strategy),
        symbol=args.symbol,
        mode=args.mode,
        candidates_screened=args.candidates,
    )
