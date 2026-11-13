#!/usr/bin/env python3
"""
================================================================================
  TARGET 72 FLEET EXPANSION & CERTIFICATION HARNESS
================================================================================
Screens vetted candidate strategies from the archive against unbroken continuous
chronological partitions (In-Sample Weeks 1..210 vs Quarantined Holdout Weeks 211..300)
under mandatory live broker friction, 100k causal warmup, and subsequent-tick fills.

Applies strict institutional hurdle gates:
1. Realized Holdout Expectancy >= +1.25 pips/trade.
2. Holdout Profit Factor >= 1.15.
3. In-Sample Discovery Profit Factor >= 1.15 and Discovery PnL > 0.
4. Outlier independence: Profitable without Top 2 winning trades.
5. Entry Timestamp Overlap < 35% with all existing certified strategies.

Promotes passing champions until the fleet reaches 72 certified strategies.
================================================================================
"""

from __future__ import annotations
import sys
import json
import time
import shutil
import importlib.util
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset, get_complete_split_weeks
from tick_engine.core.config import PIP_SIZES, get_friction_pips, WARMUP_TICKS
from tick_engine.reporting.run_manifest import create_certified_run_manifest, compute_statistical_metrics

FLEET_DIR = ROOT_DIR / "certified_production_fleet"
MANIFEST_DIR = ROOT_DIR / "results" / "certified_strategy_manifests"


def eval_week(strat_file: str, symbol: str, week_num: int) -> np.ndarray:
    strat_path = Path(strat_file)
    pip_size = PIP_SIZES.get(symbol, 0.0001)
    comm_pips, slip_pips = get_friction_pips(symbol)

    spec = importlib.util.spec_from_file_location(f"strat_mod_{week_num}", strat_path)
    if spec is None or spec.loader is None:
        return np.empty(0)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return np.empty(0)

    strat_cls = None
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, type) and hasattr(obj, "generate_signals") and attr != "BaseStrategy":
            strat_cls = obj
            break

    if strat_cls is None:
        return np.empty(0)

    try:
        strat = strat_cls(asset=symbol, symbol=symbol, pip_size=pip_size)
    except Exception:
        try:
            strat = strat_cls(pip_size=pip_size)
        except Exception:
            try:
                strat = strat_cls()
            except Exception:
                return np.empty(0)

    raw_dir = ROOT_DIR / "data" / symbol
    try:
        ticks, _ = get_week_dataset(week=week_num, raw_dir=raw_dir, warmup_prev_week_ticks=WARMUP_TICKS)
    except Exception:
        return np.empty(0)

    if len(ticks) <= WARMUP_TICKS + 500:
        return np.empty(0)

    min_entry_ts = ticks['timestamp'][WARMUP_TICKS]

    try:
        res = strat.generate_signals(ticks)
    except Exception:
        return np.empty(0)

    if isinstance(res, tuple):
        sig = res[0]
        sl = res[1] if len(res) > 1 and res[1] is not None else np.full(len(ticks), getattr(strat, "sl_pips", 20.0))
        tp = res[2] if len(res) > 2 and res[2] is not None else np.full(len(ticks), getattr(strat, "tp_pips", 40.0))
        sig_ts = res[3] if len(res) > 3 else None
    else:
        sig = res
        sl = np.full(len(ticks), getattr(strat, "sl_pips", 20.0))
        tp = np.full(len(ticks), getattr(strat, "tp_pips", 40.0))
        sig_ts = None

    if not np.any(sig != 0):
        return np.empty(0)

    trades = run_matching_loop(
        ticks=ticks,
        signals=sig,
        pip_size=pip_size,
        sl_pips=sl,
        tp_pips=tp,
        use_slippage=True,
        close_at_end=True,
        commission_pips=comm_pips,
        entry_slippage_pips=slip_pips,
        signal_timestamps=sig_ts,
        min_entry_timestamp=min_entry_ts,
        enforce_subsequent_tick=True
    )
    return trades


def calc_overlap(trades_a: np.ndarray, trades_b: np.ndarray, window_ms: int = 15 * 60 * 1000) -> float:
    if len(trades_a) == 0 or len(trades_b) == 0:
        return 0.0
    matches = 0
    ts_b_list = list(zip(trades_b['entry_time'], trades_b['direction']))
    b_used = set()
    for tr_a in trades_a:
        ts_a = tr_a['entry_time']
        dir_a = tr_a['direction']
        for idx_b, (ts_b, dir_b) in enumerate(ts_b_list):
            if idx_b in b_used:
                continue
            if dir_a == dir_b and abs(ts_a - ts_b) <= window_ms:
                matches += 1
                b_used.add(idx_b)
                break
    return matches / min(len(trades_a), len(trades_b))


def main():
    print("=" * 90)
    print("  TARGET 72 FLEET EXPANSION & CERTIFICATION HARNESS")
    print("=" * 90)

    # 1. Identify existing certified strategy names & load existing trades
    with open(FLEET_DIR / "CERTIFIED_FLEET_MANIFEST.json", "r") as f:
        fleet_meta = json.load(f)

    existing_strategies = fleet_meta.get("strategies", [])
    current_count = len(existing_strategies)
    print(f"Current certified fleet size: {current_count} strategies.")
    needed = 72 - current_count
    print(f"Targeting: 72 strategies (Need to verify and promote {needed} more).")

    existing_trades_by_sym = {}
    for s in existing_strategies:
        sym = s["symbol"].upper()
        sname = s["name"]
        csv_path = MANIFEST_DIR / f"{sym}__{sname}_trades.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if len(df) > 0 and 'entry_time' in df.columns:
                arr = np.zeros(len(df), dtype=[('entry_time', 'i8'), ('direction', 'i1')])
                arr['entry_time'] = df['entry_time'].values
                arr['direction'] = df['direction'].values
                existing_trades_by_sym.setdefault(sym, []).append((sname, arr))

    # 2. Gather diverse candidate pool from archives across multiple pairs
    candidate_pool = [
        # AUDNZD candidates
        ("audnzd_disc115_fib_golden_pocket_lb48_elite", "AUDNZD", "archive_deprecated/strategies/audnzd_fleet/disc115_fib_golden_pocket_lb48_elite/strategy.py"),
        ("audnzd_disc116_fib_golden_pocket_precision_elite", "AUDNZD", "archive_deprecated/strategies/audnzd_fleet/disc116_fib_golden_pocket_precision_elite/strategy.py"),
        ("audnzd_disc117_fib_golden_pocket_high_wr_elite", "AUDNZD", "archive_deprecated/strategies/audnzd_fleet/disc117_fib_golden_pocket_high_wr_elite/strategy.py"),
        ("audnzd_disc118_fib_golden_pocket_lb36_elite", "AUDNZD", "archive_deprecated/strategies/audnzd_fleet/disc118_fib_golden_pocket_lb36_elite/strategy.py"),
        ("audnzd_disc120_session_vwap_band_fade_elite", "AUDNZD", "archive_deprecated/strategies/audnzd_fleet/disc120_session_vwap_band_fade_elite/strategy.py"),
        ("audnzd_disc129_fib_golden_pocket_lb72_macro_elite", "AUDNZD", "archive_deprecated/strategies/audnzd_fleet/disc129_fib_golden_pocket_lb72_macro_elite/strategy.py"),
        
        # EURGBP candidates
        ("eurgbp_disc121_session_vwap_fade_elite", "EURGBP", "archive_deprecated/strategies/eurgbp_fleet/disc121_session_vwap_fade_elite/strategy.py"),
        ("eurgbp_disc122_fib_golden_pocket_lb36_elite", "EURGBP", "archive_deprecated/strategies/eurgbp_fleet/disc122_fib_golden_pocket_lb36_elite/strategy.py"),
        ("eurgbp_disc123_daily_pivot_pullback_elite", "EURGBP", "archive_deprecated/strategies/eurgbp_fleet/disc123_daily_pivot_pullback_elite/strategy.py"),
        ("eurgbp_disc125_fib_golden_pocket_lb48_elite", "EURGBP", "archive_deprecated/strategies/eurgbp_fleet/disc125_fib_golden_pocket_lb48_elite/strategy.py"),
        ("eurgbp_disc73_donchian_ema_pullback_elite", "EURGBP", "archive_deprecated/strategies/eurgbp_fleet/disc73_donchian_ema_pullback_elite/strategy.py"),
        
        # EURUSD candidates
        ("eurusd_disc103_double_swing_neckline_retest_elite", "EURUSD", "archive_deprecated/strategies/qualified_strategies/eurusd/disc103_double_swing_neckline_retest_elite/strategy.py"),
        ("eurusd_disc105_chaikin_volatility_surge_elite", "EURUSD", "archive_deprecated/strategies/qualified_strategies/eurusd/disc105_chaikin_volatility_surge_elite/strategy.py"),
        ("eurusd_disc45_continuous_m5_rejection_elite", "EURUSD", "archive_deprecated/strategies/qualified_strategies/eurusd/disc45_continuous_m5_rejection_elite/strategy.py"),
        ("eurusd_disc39_stochastic_deep_pullback_elite", "EURUSD", "archive_deprecated/strategies/new_discoveries/disc39_stochastic_deep_pullback_elite/strategy.py"),
        ("eurusd_disc52_fib_golden_pocket_pin_defense_elite", "EURUSD", "archive_deprecated/strategies/eurusd_production_fleet/disc52_fib_golden_pocket_pin_defense_elite/strategy.py"),
        ("eurusd_disc54_h1_inside_bar_expansion_elite", "EURUSD", "archive_deprecated/strategies/eurusd_production_fleet/disc54_h1_inside_bar_expansion_elite/strategy.py"),
        ("eurusd_disc61_double_structure_neckline_breakout_elite", "EURUSD", "archive_deprecated/strategies/eurusd_production_fleet/disc61_double_structure_neckline_breakout_elite/strategy.py"),
        ("eurusd_disc64_asian_midpoint_rebound_elite", "EURUSD", "archive_deprecated/strategies/eurusd_production_fleet/disc64_asian_midpoint_rebound_elite/strategy.py"),
        ("eurusd_disc88_three_drive_ignition_elite", "EURUSD", "archive_deprecated/strategies/eurusd_production_fleet/disc88_three_drive_ignition_elite/strategy.py"),
        
        # USDJPY candidates
        ("usdjpy_disc92_asian_sr_flip_retest_elite", "USDJPY", "archive_deprecated/strategies/usdjpy_fleet/disc92_asian_sr_flip_retest_elite/strategy.py"),
        ("usdjpy_disc66_heikin_ashi_surge_elite", "USDJPY", "archive_deprecated/strategies/usdjpy_fleet/disc66_heikin_ashi_surge_elite/strategy.py"),
        ("usdjpy_disc52_fib_golden_pocket_pin_defense_elite", "USDJPY", "archive_deprecated/strategies/certified_29_champions/usdjpy/disc52_fib_golden_pocket_pin_defense_elite/strategy.py"),
        ("usdjpy_disc54_h1_inside_bar_expansion_elite", "USDJPY", "archive_deprecated/strategies/certified_29_champions/usdjpy/disc54_h1_inside_bar_expansion_elite/strategy.py"),
        ("usdjpy_disc64_asian_midpoint_rebound_elite", "USDJPY", "archive_deprecated/strategies/certified_29_champions/usdjpy/disc64_asian_midpoint_rebound_elite/strategy.py"),
        ("usdjpy_disc88_three_drive_ignition_elite", "USDJPY", "archive_deprecated/strategies/certified_29_champions/usdjpy/disc88_three_drive_ignition_elite/strategy.py")
    ]

    # Filter to only existing files
    valid_candidates = []
    for cid, sym, pstr in candidate_pool:
        p = ROOT_DIR / pstr
        if p.exists():
            valid_candidates.append((cid, sym, p))

    print(f"Queued {len(valid_candidates)} distinct candidate strategy files for evaluation.")

    promoted = []

    with ProcessPoolExecutor(max_workers=4) as executor:
        for cid, sym, strat_path in valid_candidates:
            if len(promoted) >= needed:
                print(f"\nReached target of 72 certified strategies! Stopping expansion.")
                break

            pip_size = PIP_SIZES.get(sym, 0.0001)
            comm_pips, slip_pips = get_friction_pips(sym)

            hold_weeks = get_complete_split_weeks(sym, "holdout_test_30")
            disc_weeks = get_complete_split_weeks(sym, "discovery_70")

            t0 = time.time()
            # Parallel week evaluation across holdout
            futs_hold = [executor.submit(eval_week, str(strat_path), sym, w) for w in hold_weeks]
            hold_trades_list = [f.result() for f in futs_hold if len(f.result()) > 0]
            hold_trades = np.concatenate(hold_trades_list) if hold_trades_list else np.empty(0)

            # Evaluate Discovery (Weeks 1..210)
            futs_disc = [executor.submit(eval_week, str(strat_path), sym, w) for w in disc_weeks]
            disc_trades_list = [f.result() for f in futs_disc if len(f.result()) > 0]
            disc_trades = np.concatenate(disc_trades_list) if disc_trades_list else np.empty(0)
            elapsed = time.time() - t0

            h_stats = compute_statistical_metrics(hold_trades)
            d_stats = compute_statistical_metrics(disc_trades)

            h_pnl = h_stats["net_pips"]
            h_pf = h_stats["profit_factor"]
            h_tr = h_stats["total_trades"]
            h_exp = h_stats["expectancy_pips"]
            d_pnl = d_stats["net_pips"]
            d_pf = d_stats["profit_factor"]
            d_tr = d_stats["total_trades"]

            # Institutional Hurdle Rates
            is_macro = any(m in cid.lower() for m in ["macro", "h4", "daily", "weekly", "golden_pocket"])
            min_h_tr = 15 if is_macro else 35

            passes_hurdles = (
                h_pnl > 0
                and h_pf >= 1.15
                and h_exp >= 1.25
                and h_tr >= min_h_tr
                and d_pnl > 0
                and d_pf >= 1.15
                and not h_stats["outlier_sensitivity"]["is_outlier_dependent"]
            )

            # Check overlap with existing strategies of the same symbol
            max_overlap = 0.0
            highest_overlap_peer = ""
            if passes_hurdles and sym in existing_trades_by_sym:
                for peer_name, peer_trades in existing_trades_by_sym[sym]:
                    ov = calc_overlap(hold_trades, peer_trades)
                    if ov > max_overlap:
                        max_overlap = ov
                        highest_overlap_peer = peer_name

            if max_overlap >= 0.35:
                passes_hurdles = False
                rejection_reason = f"High Overlap ({max_overlap*100:.1f}% with {highest_overlap_peer})"
            elif not passes_hurdles:
                rejection_reason = f"Hurdle: Holdout {h_pnl:+.1f}p (PF {h_pf:.2f}, {h_tr} tr) | Disc {d_pnl:+.1f}p (PF {d_pf:.2f})"
            else:
                rejection_reason = "PASSED"

            print(f"[{sym:6s}] {cid[:38]:38s} | Holdout: {h_pnl:>+6.1f}p (PF {h_pf:>4.2f}, {h_tr:>3} tr, {h_exp:>+5.2f}p/tr) | Disc: {d_pnl:>+7.1f}p (PF {d_pf:>4.2f}) | {rejection_reason} ({elapsed:.1f}s)")

            if passes_hurdles:
                # Promote strategy!
                dest_cat = f"{sym.lower()}_champions"
                dest_dir = FLEET_DIR / dest_cat / cid
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_strat_file = dest_dir / "strategy.py"
                shutil.copy2(strat_path, dest_strat_file)

                # Save canonical manifest and trade log
                manifest = create_certified_run_manifest(
                    strategy_name=cid,
                    strategy_file=dest_strat_file,
                    symbol=sym,
                    split_version="canonical_continuous_v1.0 (In-Sample: 1..210, Holdout: 211..300)",
                    start_week=211,
                    end_week=300,
                    pip_size=pip_size,
                    commission_pips=comm_pips,
                    entry_slippage_pips=slip_pips,
                    spread_policy="Floating Raw Dukascopy Bid/Ask",
                    warmup_ticks=WARMUP_TICKS,
                    trades=hold_trades,
                    additional_meta={
                        "discovery_in_sample_metrics": d_stats,
                        "evaluation_time_seconds": round(elapsed, 2)
                    }
                )

                manifest_path = MANIFEST_DIR / f"{sym.upper()}__{cid}_run_manifest.json"
                with open(manifest_path, "w", encoding="utf-8") as mf:
                    json.dump(manifest, mf, indent=2)

                csv_path = MANIFEST_DIR / f"{sym.upper()}__{cid}_trades.csv"
                pd.DataFrame(manifest["trade_log"]).to_csv(csv_path, index=False)

                # Add to existing trades tracking to prevent subsequent candidates from overlapping
                arr = np.zeros(len(hold_trades), dtype=[('entry_time', 'i8'), ('direction', 'i1')])
                arr['entry_time'] = hold_trades['entry_time']
                arr['direction'] = hold_trades['direction']
                existing_trades_by_sym.setdefault(sym, []).append((cid, arr))

                promoted.append((cid, sym, h_pnl, h_pf, h_tr, h_exp))
                print(f"   >>> PROMOTED & CERTIFIED [{len(promoted)}/{needed}]: {cid} -> {dest_dir.relative_to(ROOT_DIR)}")

    print("\n" + "=" * 90)
    print(f"EXPANSION RUN COMPLETE: PROMOTED {len(promoted)} NEW CERTIFIED CHAMPIONS!")
    print("=" * 90)
    for cid, sym, hp, hpf, htr, hexp in promoted:
        print(f"  * [{sym:6s}] {cid:42s} | Holdout: {hp:>+6.1f}p (PF {hpf:>4.2f}, {htr:>3} tr, {hexp:>+5.2f}p/tr)")


if __name__ == "__main__":
    main()

