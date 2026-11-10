#!/usr/bin/env python3
"""
================================================================================
  INSTITUTIONAL ZERO-BIAS ALPHA SEARCHER ENGINE (CONFIGURED FOR $4/LOT FRICTION)
================================================================================
Discovers and certifies robust algorithmic alphas across the 70% Discovery Partition
with exact $4.00/lot institutional ECN friction under strict zero-lookahead rules:
- Causal 100,000 tick historical warmup
- Zero .bfill() / Zero future data leakage
- Strict .shift(1) on all higher-timeframe bar aggregates
- Dynamic $4.00/lot friction calculation per currency pair
- Next-tick bar-close execution
- Dual-Partition Certification (Discovery 70% -> Holdout 30%)
================================================================================
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset
from data_engineering_suite.bars import build_time_bars

from alpha_searcher.config import (
    DEFAULT_DOLLAR_PER_LOT,
    WARMUP_TICKS,
    PIP_SIZES,
    MAX_SPREADS,
    LIVE_BROKER_SPREADS,
    SPLITS_FILE,
    RESULTS_DIR,
    get_friction_pips,
    get_total_friction_pips,
)
from alpha_searcher.archetypes import generate_candidate_battery


def get_partition_weeks(symbol: str) -> Tuple[List[int], List[int]]:
    with open(SPLITS_FILE, "r", encoding="utf-8") as f:
        splits = json.load(f)
    asset_data = splits["assets"].get(symbol.upper(), {})
    disc_weeks = sorted(asset_data.get("discovery_70", {}).get("weeks", []))
    hold_weeks = sorted(asset_data.get("holdout_test_30", {}).get("weeks", []))
    return disc_weeks, hold_weeks


def _evaluate_week_worker(args: Tuple[str, List[Dict[str, Any]], List[int], float, bool]) -> Dict[str, Dict[str, Any]]:
    symbol, candidates, weeks, dollar_per_lot, use_broker_spread = args
    pip_size = PIP_SIZES[symbol]
    max_spread = MAX_SPREADS.get(symbol, 2.0)
    comm_pips, slip_pips = get_friction_pips(symbol, dollar_per_lot)
    broker_spread_pips = LIVE_BROKER_SPREADS.get(symbol, 0.5) if use_broker_spread else None
    raw_dir = ROOT_DIR / "data" / symbol

    results = {
        c["name"]: {
            "cand": c,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "net_pips": 0.0,
            "win_pips": 0.0,
            "loss_pips": 0.0,
        }
        for c in candidates
    }

    for w in weeks:
        try:
            res = get_week_dataset(w, raw_dir=raw_dir, warmup_prev_week_ticks=WARMUP_TICKS)
            if res is None: continue
            ticks, _ = res
        except Exception:
            continue
        if len(ticks) < 1000: continue

        n_ticks = len(ticks)
        tick_ts = ticks["timestamp"]
        warmup_cutoff_ts = int(tick_ts[WARMUP_TICKS]) if (WARMUP_TICKS < n_ticks and w > 1) else 0

        # Precompute M15 bars once per week
        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=pip_size)
        if len(m15) < 30: continue

        hours = m15.index.hour.values
        dates = m15.index.date
        time_ms = m15.index.astype("datetime64[ms]").view("int64")
        dur_ms = 15 * 60 * 1000
        spread = m15["spread_pips"].values
        c = m15["close"].values
        h = m15["high"].values
        l = m15["low"].values
        o = m15["open"].values
        bar_rng = (h - l) / pip_size

        # Precompute H1 and H4 base frames once per week
        mid_series = pd.Series(0.5 * (ticks["bid"] + ticks["ask"]), index=pd.to_datetime(tick_ts, unit="ms"))
        h1_base = mid_series.resample("1h").agg(["first", "max", "min", "last"]).dropna()
        h4_base = mid_series.resample("4h").agg(["first", "max", "min", "last"]).dropna()

        h1_cache = {}
        h4_cache = {}

        for c_info in candidates:
            fam = c_info["family"]
            signals = np.zeros(n_ticks, dtype=np.int8)

            if fam in ("ASIAN_SWEEP", "LONDON_SWEEP", "NY_SWEEP"):
                if fam == "ASIAN_SWEEP":
                    s_end, t_start, t_end = 6, 6, 14
                elif fam == "LONDON_SWEEP":
                    s_end, t_start, t_end = 7, 7, 14
                else:  # NY_SWEEP
                    s_end, t_start, t_end = 12, 12, 17

                delta = c_info["delta"]
                min_r = c_info["min_rng"]
                max_r = c_info["max_rng"]
                last_d = None
                curr_d = None
                sh, sl = 0.0, 999999.0
                ready = False

                for i in range(len(m15)):
                    d = dates[i]
                    hr = hours[i]
                    if d != curr_d:
                        curr_d = d
                        sh, sl = 0.0, 999999.0
                        ready = False

                    if hr < s_end:
                        if h[i] > sh: sh = h[i]
                        if l[i] < sl: sl = l[i]
                    elif hr == s_end and not ready:
                        rng = (sh - sl) / pip_size
                        if min_r <= rng <= max_r:
                            ready = True

                    if ready and (t_start <= hr <= t_end):
                        if d == last_d or spread[i] > max_spread:
                            continue
                        sig = 0
                        if h[i] >= (sh + delta * pip_size) and c[i] < sh and c[i] < o[i]:
                            sig = -1
                        elif l[i] <= (sl - delta * pip_size) and c[i] > sl and c[i] > o[i]:
                            sig = 1

                        if sig != 0:
                            exec_t = time_ms[i] + dur_ms
                            idx = int(np.searchsorted(tick_ts, exec_t))
                            if idx < n_ticks and signals[idx] == 0:
                                signals[idx] = sig
                                last_d = d

            elif fam == "H1_RETEST":
                lb = c_info["lb"]
                if lb not in h1_cache:
                    if len(h1_base) >= lb + 5:
                        ema20 = h1_base["last"].ewm(span=20, adjust=False).mean()
                        bias = np.where(h1_base["last"] > ema20, 1, -1)
                        roll_h = h1_base["max"].rolling(lb).max().shift(1)
                        roll_l = h1_base["min"].rolling(lb).min().shift(1)
                        b_m15 = pd.Series(bias, index=h1_base.index).shift(1).reindex(m15.index, method="ffill").fillna(0).values
                        sp_m15 = roll_l.reindex(m15.index, method="ffill").values
                        rs_m15 = roll_h.reindex(m15.index, method="ffill").values
                        h1_cache[lb] = (b_m15, sp_m15, rs_m15)
                    else:
                        h1_cache[lb] = None

                cache_val = h1_cache[lb]
                if cache_val is not None:
                    b_m15, sp_m15, rs_m15 = cache_val
                    dist_p = c_info["dist"]
                    w_req = c_info["wick"]
                    last_d = None

                    for i in range(15, len(m15)):
                        hr = hours[i]
                        if not (6 <= hr <= 18): continue
                        d = dates[i]
                        if d == last_d or spread[i] > max_spread or bar_rng[i] < 2.0: continue
                        sp, rs, b_val = sp_m15[i], rs_m15[i], b_m15[i]
                        if np.isnan(sp) or np.isnan(rs): continue

                        sig = 0
                        if b_val > 0 and sp > 0:
                            dist_sp = abs(l[i] - sp) / pip_size
                            lw = (min(o[i], c[i]) - l[i]) / pip_size
                            if dist_sp <= dist_p and (lw / bar_rng[i]) >= w_req and c[i] > o[i]:
                                sig = 1
                        elif b_val < 0 and rs > 0:
                            dist_rs = abs(rs - h[i]) / pip_size
                            uw = (h[i] - max(o[i], c[i])) / pip_size
                            if dist_rs <= dist_p and (uw / bar_rng[i]) >= w_req and c[i] < o[i]:
                                sig = -1

                        if sig != 0:
                            exec_t = time_ms[i] + dur_ms
                            idx = int(np.searchsorted(tick_ts, exec_t))
                            if idx < n_ticks and signals[idx] == 0:
                                signals[idx] = sig
                                last_d = d

            elif fam == "H4_RETEST":
                lb = c_info["lb"]
                if lb not in h4_cache:
                    if len(h4_base) >= lb + 5:
                        ema20 = h4_base["last"].ewm(span=20, adjust=False).mean()
                        bias = np.where(h4_base["last"] > ema20, 1, -1)
                        roll_h = h4_base["max"].rolling(lb).max().shift(1)
                        roll_l = h4_base["min"].rolling(lb).min().shift(1)
                        b_m15 = pd.Series(bias, index=h4_base.index).shift(1).reindex(m15.index, method="ffill").fillna(0).values
                        sp_m15 = roll_l.reindex(m15.index, method="ffill").values
                        rs_m15 = roll_h.reindex(m15.index, method="ffill").values
                        h4_cache[lb] = (b_m15, sp_m15, rs_m15)
                    else:
                        h4_cache[lb] = None

                cache_val = h4_cache[lb]
                if cache_val is not None:
                    b_m15, sp_m15, rs_m15 = cache_val
                    dist_p = c_info["dist"]
                    w_req = c_info["wick"]
                    last_d = None

                    for i in range(15, len(m15)):
                        hr = hours[i]
                        if not (6 <= hr <= 20): continue
                        d = dates[i]
                        if d == last_d or spread[i] > max_spread or bar_rng[i] < 2.0: continue
                        sp, rs, b_val = sp_m15[i], rs_m15[i], b_m15[i]
                        if np.isnan(sp) or np.isnan(rs): continue

                        sig = 0
                        if b_val > 0 and sp > 0:
                            dist_sp = abs(l[i] - sp) / pip_size
                            lw = (min(o[i], c[i]) - l[i]) / pip_size
                            if dist_sp <= dist_p and (lw / bar_rng[i]) >= w_req and c[i] > o[i]:
                                sig = 1
                        elif b_val < 0 and rs > 0:
                            dist_rs = abs(rs - h[i]) / pip_size
                            uw = (h[i] - max(o[i], c[i])) / pip_size
                            if dist_rs <= dist_p and (uw / bar_rng[i]) >= w_req and c[i] < o[i]:
                                sig = -1

                        if sig != 0:
                            exec_t = time_ms[i] + dur_ms
                            idx = int(np.searchsorted(tick_ts, exec_t))
                            if idx < n_ticks and signals[idx] == 0:
                                signals[idx] = sig
                                last_d = d

            if not np.any(signals != 0):
                continue

            tr_list = run_matching_loop(
                ticks=ticks,
                signals=signals,
                pip_size=pip_size,
                sl_pips=c_info["sl"],
                tp_pips=c_info["tp"],
                commission_pips=comm_pips,
                entry_slippage_pips=slip_pips,
                max_trades=500,
                spread_pips=broker_spread_pips,
            )

            for tr in tr_list:
                if tr[0] < warmup_cutoff_ts and w > 1: continue
                pnl = float(tr[5])
                r = results[c_info["name"]]
                r["trades"] += 1
                r["net_pips"] += pnl
                if pnl > 0:
                    r["wins"] += 1
                    r["win_pips"] += pnl
                else:
                    r["losses"] += 1
                    r["loss_pips"] += abs(pnl)

    return results


def evaluate_candidates_on_weeks(
    symbol: str,
    candidates: List[Dict[str, Any]],
    weeks: List[int],
    dollar_per_lot: float = DEFAULT_DOLLAR_PER_LOT,
    is_holdout: bool = False,
    use_broker_spread: bool = True,
    n_workers: int = 8,
) -> Dict[str, Dict[str, Any]]:
    results = {
        c["name"]: {
            "cand": c,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "net_pips": 0.0,
            "win_pips": 0.0,
            "loss_pips": 0.0,
        }
        for c in candidates
    }

    t0 = time.time()
    n_weeks = len(weeks)
    tag = "Holdout" if is_holdout else "Discovery"

    if n_weeks <= 4 or n_workers <= 1:
        return _evaluate_week_worker((symbol, candidates, weeks, dollar_per_lot, use_broker_spread))

    # Split weeks across workers
    n_workers = min(n_workers, n_weeks)
    chunks = np.array_split(weeks, n_workers)
    worker_args = [(symbol, candidates, list(chunk), dollar_per_lot, use_broker_spread) for chunk in chunks if len(chunk) > 0]

    with ProcessPoolExecutor(max_workers=len(worker_args)) as executor:
        futures = [executor.submit(_evaluate_week_worker, arg) for arg in worker_args]
        for idx, fut in enumerate(as_completed(futures)):
            chunk_res = fut.result()
            for name, meta in chunk_res.items():
                r = results[name]
                r["trades"] += meta["trades"]
                r["wins"] += meta["wins"]
                r["losses"] += meta["losses"]
                r["net_pips"] += meta["net_pips"]
                r["win_pips"] += meta["win_pips"]
                r["loss_pips"] += meta["loss_pips"]
            print(f"  [{tag} Parallel] Worker chunk {idx+1}/{len(worker_args)} finished in {time.time() - t0:.1f}s")

    print(f"  [{tag} Complete] Finished {n_weeks} weeks across {len(worker_args)} cores in {time.time() - t0:.1f}s.")
    return results


def run_alpha_search(
    symbol: str,
    dollar_per_lot: float = DEFAULT_DOLLAR_PER_LOT,
    min_trades_per_year: float = 25.0,
    verify_holdout: bool = True,
    sample_discovery_rate: int = 4,
    use_broker_spread: bool = True,
) -> Dict[str, Any]:
    s = symbol.upper()
    disc_weeks, hold_weeks = get_partition_weeks(s)
    comm_pips, slip_pips = get_friction_pips(s, dollar_per_lot)
    tot_frict = get_total_friction_pips(s, dollar_per_lot)
    spr_model_val = LIVE_BROKER_SPREADS.get(s, 0.5) if use_broker_spread else None

    print("=" * 100)
    print(f"   ALPHA SEARCHER: {s} (Friction Configured to ${dollar_per_lot:.2f} / Standard Lot)   ".center(100))
    print("=" * 100)
    print(f"Symbol:                 {s} (Pip Size: {PIP_SIZES[s]})")
    print(f"Target Friction:        ${dollar_per_lot:.2f} / lot -> {tot_frict:.2f} pips total ({comm_pips:.2f}p comm + {slip_pips:.2f}p slippage)")
    if use_broker_spread:
        print(f"Spread Model:           LIVE BROKER SPREAD ({spr_model_val:.2f} pips / MT5 Market Watch)")
    else:
        print(f"Spread Model:           RAW DUKASCOPY FLOATING SPREAD (Stress Test)")
    print(f"Spread Filter Ceiling:  <= {MAX_SPREADS.get(s, 2.0):.1f} pips on M15 bars")
    print(f"Warmup Protocol:        {WARMUP_TICKS:,} ticks causal warmup")
    print(f"Discovery Dataset:      70% Partition ({len(disc_weeks)} weeks)")
    print(f"Holdout Dataset:        30% Quarantined ({len(hold_weeks)} weeks)")
    print("=" * 100)

    # 1. Generate candidate battery
    candidates = generate_candidate_battery(s, high_frequency_bias=True)
    print(f"Generated {len(candidates)} candidate configurations across 5 market archetypes.")

    # 2. Multi-regime discovery screening
    sample_disc = disc_weeks[::sample_discovery_rate]
    print(f"Screening on {len(sample_disc)} representative Discovery Weeks...")
    disc_results = evaluate_candidates_on_weeks(
        s,
        candidates,
        sample_disc,
        dollar_per_lot=dollar_per_lot,
        is_holdout=False,
        use_broker_spread=use_broker_spread,
    )

    ann_factor = 52.0 / len(sample_disc)
    qualified = []
    for name, r in disc_results.items():
        tr = r["trades"]
        ann_tr = tr * ann_factor
        if ann_tr < min_trades_per_year:
            continue
        net_p = r["net_pips"]
        if net_p <= 0:
            continue
        wr = (r["wins"] / tr * 100.0) if tr > 0 else 0.0
        pf = (r["win_pips"] / r["loss_pips"]) if r["loss_pips"] > 0 else 99.0
        ann_pips = net_p * ann_factor

        qualified.append({
            "name": name,
            "family": r["cand"]["family"],
            "cand": r["cand"],
            "disc_trades": tr,
            "ann_trades": round(ann_tr, 1),
            "disc_net_pips": round(net_p, 1),
            "ann_pips": round(ann_pips, 1),
            "win_rate": round(wr, 1),
            "profit_factor": round(pf, 2),
        })

    qualified.sort(key=lambda x: x["ann_pips"], reverse=True)
    print(f"\nTotal Qualifying Candidates on 70% Discovery Partition (>= {min_trades_per_year:.0f} tr/yr): {len(qualified)}")

    print("\n" + "=" * 115)
    print(f"{'Rank':4s} | {'Strategy Name':48s} | {'Fam':12s} | {'Trades/Yr':9s} | {'WR %':5s} | {'PF':5s} | {'Ann Net Pips':12s}")
    print("-" * 115)
    for idx, q in enumerate(qualified[:25]):
        print(f"{idx+1:4d} | {q['name']:48s} | {q['family']:12s} | {q['ann_trades']:9.1f} | {q['win_rate']:5.1f} | {q['profit_factor']:5.2f} | {q['ann_pips']:+12.1f}p")
    print("=" * 115)

    # 3. Dual-Certification Holdout Validation (if requested)
    holdout_champions = []
    if verify_holdout and qualified:
        # Pick top diverse candidates (up to 12)
        eval_cands = []
        seen = set()
        for q in qualified:
            c = q["cand"]
            key = (c["family"], c.get("delta", c.get("lb")), c["sl"], c["tp"])
            if key not in seen:
                seen.add(key)
                eval_cands.append(c)
            if len(eval_cands) >= 12:
                break

        print(f"\nValidating Top {len(eval_cands)} Candidates on 90-Week Quarantined Holdout ({len(hold_weeks)} weeks)...")
        hold_results = evaluate_candidates_on_weeks(
            s,
            eval_cands,
            hold_weeks,
            dollar_per_lot=dollar_per_lot,
            is_holdout=True,
            use_broker_spread=use_broker_spread,
        )

        n_hold_years = len(hold_weeks) / 52.0
        print("\n" + "=" * 120)
        print("                 DUAL-CERTIFICATION: 90-WEEK UNSEEN HOLDOUT PERFORMANCE SCORECARD                 ".center(120))
        print("=" * 120)
        print(f"{'Strategy Name':46s} | {'Fam':12s} | {'Trades':6s} | {'Trades/Yr':9s} | {'WR %':5s} | {'PF':5s} | {'Holdout Pips':12s} | {'Status':16s}")
        print("-" * 120)

        for c in eval_cands:
            r = hold_results[c["name"]]
            tr = r["trades"]
            net_p = r["net_pips"]
            wr = (r["wins"] / tr * 100.0) if tr > 0 else 0.0
            pf = (r["win_pips"] / r["loss_pips"]) if r["loss_pips"] > 0 else 99.0
            ann_tr = tr / n_hold_years
            status = "DUAL-CERTIFIED" if net_p > 0 else "FAILED HOLDOUT"

            if net_p > 0:
                holdout_champions.append({
                    "name": c["name"],
                    "cand": c,
                    "holdout_trades": tr,
                    "ann_trades": round(ann_tr, 1),
                    "holdout_net_pips": round(net_p, 1),
                    "win_rate": round(wr, 1),
                    "profit_factor": round(pf, 2),
                })

            print(f"{c['name']:46s} | {c['family']:12s} | {tr:6d} | {ann_tr:9.1f} | {wr:5.1f} | {pf:5.2f} | {net_p:+12.1f}p | {status:16s}")
        print("=" * 120)

    # Save complete findings to JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f"alpha_searcher_{s.lower()}_{int(dollar_per_lot)}usd.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "symbol": s,
            "dollar_per_lot": dollar_per_lot,
            "use_broker_spread": use_broker_spread,
            "broker_spread_pips": spr_model_val,
            "friction_commission_pips": comm_pips,
            "friction_slippage_pips": slip_pips,
            "total_friction_pips": tot_frict,
            "qualified_discovery_count": len(qualified),
            "dual_certified_holdout_count": len(holdout_champions),
            "discovery_candidates": qualified,
            "holdout_champions": holdout_champions,
        }, f, indent=2)

    print(f"\nAlpha search results successfully saved to: {out_file}")
    return {
        "symbol": s,
        "qualified_discovery": len(qualified),
        "dual_certified": len(holdout_champions),
        "results_file": str(out_file)
    }


def main():
    parser = argparse.ArgumentParser(description="Institutional Zero-Bias Alpha Searcher Engine ($4/lot Friction)")
    parser.add_argument("--symbol", type=str, default="EURUSD", help="Currency pair symbol (e.g. EURUSD, USDJPY)")
    parser.add_argument("--dollar-per-lot", type=float, default=DEFAULT_DOLLAR_PER_LOT, help="Total fixed friction in USD per standard lot (default: 4.0)")
    parser.add_argument("--min-trades", type=float, default=25.0, help="Minimum annualized trades required (default: 25.0)")
    parser.add_argument("--skip-holdout", action="store_true", help="Skip dual-certification holdout verification")
    parser.add_argument("--raw-spread", dest="use_broker_spread", action="store_false", default=True, help="Use raw historical Dukascopy floating spread instead of broker spread model")
    args = parser.parse_args()

    run_alpha_search(
        symbol=args.symbol,
        dollar_per_lot=args.dollar_per_lot,
        min_trades_per_year=args.min_trades,
        verify_holdout=not args.skip_holdout,
        use_broker_spread=args.use_broker_spread,
    )


if __name__ == "__main__":
    main()
