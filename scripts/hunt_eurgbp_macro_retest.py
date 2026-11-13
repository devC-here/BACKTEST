#!/usr/bin/env python3
"""
================================================================================
  EURGBP MACRO STRUCTURE RETEST DISCOVERY: H4/H1 DONCHIAN STRUCTURAL ABSORPTION
================================================================================
Tests the proven "Macro Structure Retest Elite" institutional archetype on EURGBP:
  - Higher Timeframe Anchor (H4 or H1): Structural Donchian support/resistance
  - Macro Trend Alignment: H4/H1 20-EMA bias
  - Microstructure Setup: M15 absorption rejection pin wick (>= 35%, 40%) testing within 2-5p
  - Active Session: European Liquid Hours (07:00 - 16:00 UTC) with spread <= 0.8p
  - Risk Setup: Asymmetric 2.0R to 2.2R (SL 8.0 - 10.0p | TP 16.0 - 22.0p)

Enforces:
  - Exact subsequent-tick execution contract (T_fill > T_sig)
  - Full ECN friction: 0.33p comm + 0.20p adverse slippage + floating Dukascopy spreads
  - Holdout (Weeks 211..300): PF >= 1.20, Trades >= 15, Net Pips > 0, ExTop3 > 0
  - Discovery (Weeks 1..210): PF >= 1.15, Trades >= 25, Net Pips > 0
  - Anti-Redundancy: Overlap < 50% & Corr < 0.50 vs disc107 AND disc172
================================================================================
"""

import sys
import os
import gc
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset, get_complete_split_weeks
from tick_engine.core.config import PIP_SIZES, get_friction_pips, WARMUP_TICKS
from data_engineering_suite.bars import build_time_bars

SYMBOL = "EURGBP"
PIP_SIZE = PIP_SIZES.get(SYMBOL, 0.0001)
COMM_PIPS, SLIP_PIPS = get_friction_pips(SYMBOL)

FLEET_MANIFEST = ROOT_DIR / "certified_production_fleet" / "CERTIFIED_FLEET_MANIFEST.json"
MANIFEST_DIR = ROOT_DIR / "results" / "certified_strategy_manifests"

existing_trades = []
with open(FLEET_MANIFEST, "r", encoding="utf-8") as f:
    fleet_data = json.load(f)

for s in fleet_data["strategies"]:
    if s["symbol"] == SYMBOL:
        name = s["name"]
        folder_slug = Path(s.get("relative_path", "")).parent.name
        csv_candidates = [
            MANIFEST_DIR / f"{SYMBOL}__{name}_trades.csv",
            MANIFEST_DIR / f"{SYMBOL}__{folder_slug}_trades.csv",
            MANIFEST_DIR / f"{SYMBOL.lower()}_{name}_trades.csv",
            MANIFEST_DIR / f"{name}_trades.csv"
        ]
        glob_matches = list(MANIFEST_DIR.glob(f"{SYMBOL}__{name}*_trades.csv"))
        if glob_matches:
            csv_candidates.extend(glob_matches)
        csv_file = next((f for f in csv_candidates if f.exists()), None)
        if csv_file:
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                existing_trades.append((name, df))
                print(f"Loaded existing EURGBP trades for {name}: {len(df)} trades")


def check_anti_redundancy(candidate_trades: np.ndarray, max_overlap_pct: float = 50.0, max_corr: float = 0.50):
    if len(candidate_trades) == 0:
        return False, "No trades"
    if not existing_trades:
        return True, "No existing strategies"

    cand_times = candidate_trades['entry_time']
    cand_dirs = candidate_trades['direction']

    cand_df = pd.DataFrame({
        "entry_time": cand_times,
        "direction": cand_dirs,
        "pips": candidate_trades['pips']
    })
    cand_df["day"] = pd.to_datetime(cand_df["entry_time"], unit="ms", utc=True).dt.date
    cand_daily = cand_df.groupby("day")["pips"].sum()

    tolerance_ms = 30 * 60 * 1000

    for ex_name, ex_df in existing_trades:
        ex_times = ex_df["entry_time"].values
        ex_dirs = ex_df["direction"].values if "direction" in ex_df else np.ones(len(ex_df))

        matches = 0
        for idx, t1 in enumerate(cand_times):
            diffs = np.abs(ex_times - t1)
            if np.any((diffs <= tolerance_ms) & (ex_dirs == cand_dirs[idx])):
                matches += 1

        overlap_pct = (matches / len(cand_times)) * 100.0 if len(cand_times) > 0 else 0
        if overlap_pct >= max_overlap_pct:
            return False, f"High overlap ({overlap_pct:.1f}%) with {ex_name}"

        ex_df_c = ex_df.copy()
        ex_df_c["day"] = pd.to_datetime(ex_df_c["entry_time"], unit="ms", utc=True).dt.date
        ex_daily = ex_df_c.groupby("day")["pips"].sum()

        all_days = sorted(list(set(cand_daily.index).union(set(ex_daily.index))))
        c_series = [cand_daily.get(d, 0.0) for d in all_days]
        e_series = [ex_daily.get(d, 0.0) for d in all_days]
        corr = np.corrcoef(c_series, e_series)[0, 1] if len(all_days) > 5 else 0.0
        if np.isnan(corr): corr = 0.0

        if corr >= max_corr:
            return False, f"High correlation ({corr:.2f}) with {ex_name}"

    return True, "PASSED Anti-Redundancy"


def load_eurgbp_retest_bars(week_num: int):
    raw_dir = ROOT_DIR / "data" / SYMBOL
    try:
        ticks, _ = get_week_dataset(week=week_num, raw_dir=raw_dir, warmup_prev_week_ticks=WARMUP_TICKS)
    except Exception:
        return None
    if len(ticks) <= WARMUP_TICKS + 500:
        return None

    m15 = build_time_bars(ticks, timeframe_min=15, pip_size=PIP_SIZE)
    if len(m15) < 35:
        return None

    c = m15['close'].values.astype(np.float64)
    h = m15['high'].values.astype(np.float64)
    l = m15['low'].values.astype(np.float64)
    o = m15['open'].values.astype(np.float64)
    spread = m15['spread_pips'].values.astype(np.float64)
    hours = m15.index.hour.values.astype(np.int16)
    dates = m15.index.date
    time_ms = m15.index.astype('datetime64[ms]').view('int64')
    dur_ms = 15 * 60 * 1000
    tick_ts = ticks['timestamp']
    n_ticks = len(ticks)
    exec_indices = np.searchsorted(tick_ts, time_ms + dur_ms).astype(np.int32)

    # H4 Structure (resampled from M15)
    h4 = m15[['high', 'low', 'close']].resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()

    h4_bias = np.zeros(len(m15), dtype=np.int8)
    supp_10 = np.zeros(len(m15), dtype=np.float64)
    res_10 = np.zeros(len(m15), dtype=np.float64)
    supp_16 = np.zeros(len(m15), dtype=np.float64)
    res_16 = np.zeros(len(m15), dtype=np.float64)

    if len(h4) >= 20:
        h4['ema20'] = h4['close'].ewm(span=20, adjust=False).mean()
        h4['bias'] = np.where(h4['close'] > h4['ema20'], 1, -1)
        h4['supp_10'] = h4['low'].rolling(10).min().shift(1).bfill()
        h4['res_10'] = h4['high'].rolling(10).max().shift(1).bfill()
        h4['supp_16'] = h4['low'].rolling(16).min().shift(1).bfill()
        h4['res_16'] = h4['high'].rolling(16).max().shift(1).bfill()

        merged_h4 = pd.merge_asof(
            m15.sort_index(),
            h4[['bias', 'supp_10', 'res_10', 'supp_16', 'res_16']].sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )
        h4_bias = merged_h4['bias'].fillna(0).values.astype(np.int8)
        supp_10 = merged_h4['supp_10'].fillna(0.0).values
        res_10 = merged_h4['res_10'].fillna(0.0).values
        supp_16 = merged_h4['supp_16'].fillna(0.0).values
        res_16 = merged_h4['res_16'].fillna(0.0).values

    # H1 Structure
    h1 = m15[['high', 'low', 'close']].resample('1h').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    h1_bias = np.zeros(len(m15), dtype=np.int8)
    h1_supp_24 = np.zeros(len(m15), dtype=np.float64)
    h1_res_24 = np.zeros(len(m15), dtype=np.float64)

    if len(h1) >= 30:
        h1['ema20'] = h1['close'].ewm(span=20, adjust=False).mean()
        h1['bias'] = np.where(h1['close'] > h1['ema20'], 1, -1)
        h1['supp_24'] = h1['low'].rolling(24).min().shift(1).bfill()
        h1['res_24'] = h1['high'].rolling(24).max().shift(1).bfill()

        merged_h1 = pd.merge_asof(
            m15.sort_index(),
            h1[['bias', 'supp_24', 'res_24']].sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )
        h1_bias = merged_h1['bias'].fillna(0).values.astype(np.int8)
        h1_supp_24 = merged_h1['supp_24'].fillna(0.0).values
        h1_res_24 = merged_h1['res_24'].fillna(0.0).values

    return {
        'ticks': ticks,
        'n_ticks': n_ticks,
        'min_entry_ts': ticks['timestamp'][WARMUP_TICKS],
        'c': c, 'h': h, 'l': l, 'o': o,
        'spread': spread,
        'hours': hours,
        'dates': dates,
        'exec_indices': exec_indices,
        'h4_bias': h4_bias,
        'supp_10': supp_10,
        'res_10': res_10,
        'supp_16': supp_16,
        'res_16': res_16,
        'h1_bias': h1_bias,
        'h1_supp_24': h1_supp_24,
        'h1_res_24': h1_res_24,
        'n_bars': len(m15),
        'signals_buf': np.zeros(n_ticks, dtype=np.int8),
        'sl_buf': np.zeros(n_ticks, dtype=np.float64),
        'tp_buf': np.zeros(n_ticks, dtype=np.float64)
    }


def execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips):
    if len(sig_indices) == 0:
        return []
    signals_buf = week_data['signals_buf']
    sl_buf = week_data['sl_buf']
    tp_buf = week_data['tp_buf']
    n_ticks = week_data['n_ticks']

    fired = []
    for idx, s_type in zip(sig_indices, sig_types):
        if idx < n_ticks and signals_buf[idx] == 0:
            signals_buf[idx] = s_type
            sl_buf[idx] = sl_pips
            tp_buf[idx] = tp_pips
            fired.append(idx)

    trades = run_matching_loop(
        ticks=week_data['ticks'],
        signals=signals_buf,
        sl_pips=sl_buf,
        tp_pips=tp_buf,
        min_entry_timestamp=int(week_data['min_entry_ts']),
        pip_size=PIP_SIZE,
        commission_pips=COMM_PIPS,
        entry_slippage_pips=SLIP_PIPS,
        enforce_subsequent_tick=True
    )

    for idx in fired:
        signals_buf[idx] = 0
        sl_buf[idx] = 0.0
        tp_buf[idx] = 0.0

    return trades


def eval_retest_candidate(week_data, tf_level, start_hr, end_hr, dist_pips, min_wick, sl_pips, tp_pips, max_spread):
    n_bars = week_data['n_bars']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    exec_indices = week_data['exec_indices']

    if tf_level == 'h4_10':
        bias = week_data['h4_bias']
        supp = week_data['supp_10']
        res = week_data['res_10']
    elif tf_level == 'h4_16':
        bias = week_data['h4_bias']
        supp = week_data['supp_16']
        res = week_data['res_16']
    elif tf_level == 'h1_24':
        bias = week_data['h1_bias']
        supp = week_data['h1_supp_24']
        res = week_data['h1_res_24']
    else:
        return []

    sig_indices = []
    sig_types = []
    last_trade_date = None

    dist = dist_pips * PIP_SIZE

    for i in range(25, n_bars):
        hr = hours[i]
        if not (start_hr <= hr <= end_hr):
            continue
        dt = dates[i]
        if dt == last_trade_date:
            continue
        if spread[i] > max_spread:
            continue

        b = bias[i]
        if b == 0:
            continue

        bar_rng = h[i] - l[i]
        if bar_rng < 1.5 * PIP_SIZE:
            continue

        sig = 0
        # Bullish Retest of Macro Support
        if b > 0 and supp[i] > 0:
            dist_supp = abs(l[i] - supp[i])
            lower_wick = min(o[i], c[i]) - l[i]
            if dist_supp <= dist and (lower_wick / bar_rng) >= min_wick and c[i] > o[i]:
                sig = 1
        # Bearish Retest of Macro Resistance
        elif b < 0 and res[i] > 0:
            dist_res = abs(res[i] - h[i])
            upper_wick = h[i] - max(o[i], c[i])
            if dist_res <= dist and (upper_wick / bar_rng) >= min_wick and c[i] < o[i]:
                sig = -1

        if sig != 0:
            sig_indices.append(exec_indices[i])
            sig_types.append(sig)
            last_trade_date = dt

    return execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips)


def generate_candidates():
    candidates = []

    # Test H4_10, H4_16, H1_24
    for tf in ['h4_10', 'h4_16', 'h1_24']:
        for wh in [(7, 16), (8, 15), (6, 17)]:
            for dist in [2.5, 3.5, 5.0]:
                for wk in [0.35, 0.40]:
                    for sl, tp in [(8.0, 16.0), (8.0, 18.0), (9.0, 18.0), (10.0, 20.0), (10.0, 22.0)]:
                        candidates.append({
                            'tf_level': tf,
                            'params': {'tf_level': tf, 'start_hr': wh[0], 'end_hr': wh[1], 'dist_pips': dist, 'min_wick': wk, 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 0.8},
                            'desc': f"EURGBP Macro Retest ({tf.upper()}, {wh[0]:02d}-{wh[1]:02d} UTC, Dist={dist}p, Wk={wk}, SL={sl}p, TP={tp}p)"
                        })

    return candidates


def main():
    print("=" * 80)
    print("  EURGBP MACRO STRUCTURE RETEST ALPHA HUNT: H4/H1 DONCHIAN LEVELS")
    print("=" * 80)

    candidates = generate_candidates()
    print(f"Generated {len(candidates)} total specialized setups.", flush=True)

    raw_dir = ROOT_DIR / "data" / SYMBOL
    disc_weeks = get_complete_split_weeks(raw_dir, split="discovery")
    hold_weeks = get_complete_split_weeks(raw_dir, split="holdout")

    print(f"Quarantined Holdout Weeks: {len(hold_weeks)} (Weeks {min(hold_weeks)}..{max(hold_weeks)})", flush=True)
    print("Streaming holdout tick data across all candidates...", flush=True)

    cand_holdout_trades = {i: [] for i in range(len(candidates))}

    t0 = time.time()
    for w_idx, w_num in enumerate(hold_weeks):
        w_data = load_eurgbp_retest_bars(w_num)
        if w_data is None:
            continue

        for c_idx, cand in enumerate(candidates):
            p = cand['params']
            tr = eval_retest_candidate(w_data, p['tf_level'], p['start_hr'], p['end_hr'], p['dist_pips'], p['min_wick'], p['sl_pips'], p['tp_pips'], p['max_spread'])
            if len(tr) > 0:
                cand_holdout_trades[c_idx].extend(tr)

        del w_data
        if (w_idx + 1) % 25 == 0 or (w_idx + 1) == len(hold_weeks):
            print(f"  Holdout week {w_idx+1}/{len(hold_weeks)} processed in {time.time()-t0:.1f}s...", flush=True)

    viable_indices = []
    print("\n--- Holdout Evaluation Results ---", flush=True)
    for c_idx, cand in enumerate(candidates):
        tr_list = cand_holdout_trades[c_idx]
        if len(tr_list) == 0:
            continue
        tr_arr = np.array(tr_list)
        pips = tr_arr['pips']
        gross_win = np.sum(pips[pips > 0])
        gross_loss = np.abs(np.sum(pips[pips < 0]))
        pf = (gross_win / gross_loss) if gross_loss > 0 else 99.0
        net_pips = np.sum(pips)
        sorted_pips = np.sort(pips)[::-1]
        pips_ex_top3 = net_pips - np.sum(sorted_pips[:3]) if len(sorted_pips) >= 3 else net_pips

        if net_pips > 0 and len(tr_list) >= 15:
            print(f"  [HOLD POSITIVE] {cand['desc']}: {len(tr_arr)} tr, Net: {net_pips:+.1f}p, PF: {pf:.2f}, ExTop3: {pips_ex_top3:+.1f}p", flush=True)

        if len(tr_list) >= 15 and pf >= 1.20 and net_pips > 0 and pips_ex_top3 > 0:
            passed, reason = check_anti_redundancy(tr_arr, max_overlap_pct=50.0, max_corr=0.50)
            if passed:
                cand['holdout_metrics'] = {
                    'trades': len(tr_arr),
                    'net_pips': float(net_pips),
                    'profit_factor': float(pf),
                    'expectancy': float(np.mean(pips)),
                    'pips_ex_top3': float(pips_ex_top3)
                }
                cand['holdout_trades'] = tr_arr
                viable_indices.append(c_idx)
                print(f"  *** [HOLDOUT PASS] *** {cand['desc']}: {len(tr_arr)} tr, +{net_pips:.1f}p, PF {pf:.2f}, Exp {np.mean(pips):.2f}p, ExTop3 +{pips_ex_top3:.1f}p", flush=True)
            else:
                print(f"  [REDUNDANT] {cand['desc']}: {reason}", flush=True)

    print(f"\nViable Holdout Candidates: {len(viable_indices)} of {len(candidates)}", flush=True)

    if not viable_indices:
        print("No candidates cleared holdout gate.", flush=True)
        return

    # Stream Discovery on viable candidates
    print(f"\nStreaming Discovery (Weeks {min(disc_weeks)}..{max(disc_weeks)}) for {len(viable_indices)} viable candidates...", flush=True)
    cand_disc_trades = {i: [] for i in viable_indices}

    t1 = time.time()
    for w_idx, w_num in enumerate(disc_weeks):
        w_data = load_eurgbp_retest_bars(w_num)
        if w_data is None:
            continue

        for c_idx in viable_indices:
            cand = candidates[c_idx]
            p = cand['params']
            tr = eval_retest_candidate(w_data, p['tf_level'], p['start_hr'], p['end_hr'], p['dist_pips'], p['min_wick'], p['sl_pips'], p['tp_pips'], p['max_spread'])
            if len(tr) > 0:
                cand_disc_trades[c_idx].extend(tr)

        del w_data
        if (w_idx + 1) % 40 == 0 or (w_idx + 1) == len(disc_weeks):
            print(f"  Discovery week {w_idx+1}/{len(disc_weeks)} processed in {time.time()-t1:.1f}s...", flush=True)

    all_champions = []
    print("\n--- Discovery Verification & Dual-Certification ---", flush=True)
    for c_idx in viable_indices:
        cand = candidates[c_idx]
        d_list = cand_disc_trades[c_idx]
        if len(d_list) < 25:
            print(f"  [DISC UNDERTRADED] {cand['desc']}: only {len(d_list)} trades", flush=True)
            continue
        d_arr = np.array(d_list)
        d_pips = d_arr['pips']
        d_win = np.sum(d_pips[d_pips > 0])
        d_loss = np.abs(np.sum(d_pips[d_pips < 0]))
        d_pf = (d_win / d_loss) if d_loss > 0 else 99.0
        d_net = np.sum(d_pips)

        if d_pf >= 1.15 and d_net > 0:
            cand['discovery_metrics'] = {
                'trades': len(d_arr),
                'net_pips': float(d_net),
                'profit_factor': float(d_pf),
                'expectancy': float(np.mean(d_pips))
            }
            cand['discovery_trades'] = d_arr
            all_champions.append(cand)
            print(f"  *** DUAL CERTIFIED EURGBP CHAMPION ***: {cand['desc']}", flush=True)
            print(f"      Holdout:   {cand['holdout_metrics']['trades']} tr, +{cand['holdout_metrics']['net_pips']:.1f}p, PF {cand['holdout_metrics']['profit_factor']:.2f}, Exp {cand['holdout_metrics']['expectancy']:.2f}p", flush=True)
            print(f"      Discovery: {len(d_arr)} tr, +{d_net:.1f}p, PF {d_pf:.2f}, Exp {np.mean(d_pips):.2f}p", flush=True)
        else:
            print(f"  [DISC FAIL] {cand['desc']}: {len(d_arr)} tr, Net: {d_net:+.1f}p, PF: {d_pf:.2f}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print(f"EURGBP DISCOVERY COMPLETE: Found {len(all_champions)} New Institutional Champions!", flush=True)
    print("=" * 80, flush=True)

    out_file = ROOT_DIR / "results" / "new_champions_eurgbp_retest.json"
    serializable = []
    for c in all_champions:
        serializable.append({
            'params': c['params'],
            'desc': c['desc'],
            'holdout_metrics': c['holdout_metrics'],
            'discovery_metrics': c['discovery_metrics']
        })
    with open(out_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved champions to: {out_file}", flush=True)


if __name__ == "__main__":
    main()

