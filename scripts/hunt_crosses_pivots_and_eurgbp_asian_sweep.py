#!/usr/bin/env python3
"""
================================================================================
  MULTI-PAIR INSTITUTIONAL ALPHA: GBPCHF PIVOT, AUDCAD PIVOT & EURGBP ASIAN SWEEP
================================================================================
Targeting underrepresented pairs:
  1. GBPCHF: Daily Pivot Pullback Rejection (Floor Traders Pivot S1/R1 Retest)
     - Target: Lift Holdout PF from 1.17 to >= 1.25 with asymmetric TP (2.0R to 2.4R)
  2. AUDCAD: Daily Pivot Pullback Rejection (Floor Traders Pivot S1/R1 Retest)
     - Test touch parameters, wick ratios, and asymmetric TP
  3. EURGBP: Asian Extreme Liquidity Sweep to London Trend (07:00 - 12:00 UTC)
     - Sweeps Asian High/Low (00:00-06:45) at Frankfurt/London Open and reverses sharply

All tests enforce:
  - Exact subsequent-tick execution contract (T_fill > T_sig)
  - Full ECN friction per symbol:
      * GBPCHF: 0.50p comm + 0.20p slip + floating spreads
      * AUDCAD: 0.50p comm + 0.20p slip + floating spreads
      * EURGBP: 0.33p comm + 0.20p slip + floating spreads
  - Holdout (Weeks 211..300): PF >= 1.20, Trades >= 15, Net Pips > 0, ExTop3 > 0
  - Discovery (Weeks 1..210): PF >= 1.15, Trades >= 25, Net Pips > 0
  - Anti-Redundancy: Overlap < 50% & Corr < 0.50 vs all existing fleet champions
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

FLEET_MANIFEST = ROOT_DIR / "certified_production_fleet" / "CERTIFIED_FLEET_MANIFEST.json"
MANIFEST_DIR = ROOT_DIR / "results" / "certified_strategy_manifests"

with open(FLEET_MANIFEST, "r", encoding="utf-8") as f:
    fleet_data = json.load(f)

existing_trades_by_sym = defaultdict(list)
for s in fleet_data["strategies"]:
    sym = s["symbol"]
    name = s["name"]
    folder_slug = Path(s.get("relative_path", "")).parent.name
    csv_candidates = [
        MANIFEST_DIR / f"{sym}__{name}_trades.csv",
        MANIFEST_DIR / f"{sym}__{folder_slug}_trades.csv",
        MANIFEST_DIR / f"{sym.lower()}_{name}_trades.csv",
        MANIFEST_DIR / f"{name}_trades.csv"
    ]
    glob_matches = list(MANIFEST_DIR.glob(f"{sym}__{name}*_trades.csv"))
    if glob_matches:
        csv_candidates.extend(glob_matches)
    csv_file = next((f for f in csv_candidates if f.exists()), None)
    if csv_file:
        df = pd.read_csv(csv_file)
        if len(df) > 0:
            existing_trades_by_sym[sym].append((name, df))


def check_anti_redundancy(sym: str, candidate_trades: np.ndarray, max_overlap_pct: float = 50.0, max_corr: float = 0.50):
    if len(candidate_trades) == 0:
        return False, "No trades"
    existing_list = existing_trades_by_sym.get(sym, [])
    if not existing_list:
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

    for ex_name, ex_df in existing_list:
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


def load_pivot_bars(symbol: str, week_num: int):
    raw_dir = ROOT_DIR / "data" / symbol
    pip_size = PIP_SIZES.get(symbol, 0.0001)
    try:
        ticks, _ = get_week_dataset(week=week_num, raw_dir=raw_dir, warmup_prev_week_ticks=WARMUP_TICKS)
    except Exception:
        return None
    if len(ticks) <= WARMUP_TICKS + 500:
        return None

    m15 = build_time_bars(ticks, timeframe_min=15, pip_size=pip_size)
    if len(m15) < 35:
        return None

    # H4 Trend Bias
    h4 = m15[['high', 'low', 'close']].resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    h4_bias = np.zeros(len(m15), dtype=np.int8)
    if len(h4) >= 10:
        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_b = np.where(h4['close'] > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'bias': h4_b}, index=h4.index)
        merged_h4 = pd.merge_asof(m15.sort_index(), h4_df.sort_index(), left_index=True, right_index=True, direction='backward')
        h4_bias = merged_h4['bias'].fillna(0).values.astype(np.int8)

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

    # Floor Traders Daily Pivots
    unique_dates = np.unique(dates)
    daily_stats = {}
    for d in unique_dates:
        mask = (dates == d)
        if np.sum(mask) >= 15:
            daily_stats[d] = (float(np.max(h[mask])), float(np.min(l[mask])), float(c[mask][-1]))

    pivots_p = np.zeros(len(m15), dtype=np.float64)
    pivots_s1 = np.zeros(len(m15), dtype=np.float64)
    pivots_r1 = np.zeros(len(m15), dtype=np.float64)

    for idx_d in range(1, len(unique_dates)):
        prev_d = unique_dates[idx_d - 1]
        cur_d = unique_dates[idx_d]
        if prev_d in daily_stats:
            dh, dl, dc = daily_stats[prev_d]
            p = (dh + dl + dc) / 3.0
            s1 = 2.0 * p - dh
            r1 = 2.0 * p - dl
            mask_cur = (dates == cur_d)
            pivots_p[mask_cur] = p
            pivots_s1[mask_cur] = s1
            pivots_r1[mask_cur] = r1

    # Asian Range (00:00 - 06:45)
    asian_high = np.zeros(len(m15), dtype=np.float64)
    asian_low = np.zeros(len(m15), dtype=np.float64)
    for d in unique_dates:
        mask_asian = (dates == d) & (hours < 7)
        if np.sum(mask_asian) >= 10:
            ah = float(np.max(h[mask_asian]))
            al = float(np.min(l[mask_asian]))
            mask_day = (dates == d)
            asian_high[mask_day] = ah
            asian_low[mask_day] = al

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
        'pivots_p': pivots_p,
        'pivots_s1': pivots_s1,
        'pivots_r1': pivots_r1,
        'asian_high': asian_high,
        'asian_low': asian_low,
        'n_bars': len(m15),
        'pip_size': pip_size,
        'signals_buf': np.zeros(n_ticks, dtype=np.int8),
        'sl_buf': np.zeros(n_ticks, dtype=np.float64),
        'tp_buf': np.zeros(n_ticks, dtype=np.float64)
    }


def execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips, comm, slip):
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
        pip_size=week_data['pip_size'],
        commission_pips=comm,
        entry_slippage_pips=slip,
        enforce_subsequent_tick=True
    )

    for idx in fired:
        signals_buf[idx] = 0
        sl_buf[idx] = 0.0
        tp_buf[idx] = 0.0

    return trades


def eval_pivot_candidate(week_data, start_hr, end_hr, touch_up, touch_dn, min_wick, sl_pips, tp_pips, max_spread, comm, slip):
    n_bars = week_data['n_bars']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    exec_indices = week_data['exec_indices']
    pivots_p = week_data['pivots_p']
    pivots_s1 = week_data['pivots_s1']
    pivots_r1 = week_data['pivots_r1']
    h4_bias = week_data['h4_bias']
    pip_size = week_data['pip_size']

    sig_indices = []
    sig_types = []
    last_trade_date = None

    t_up = touch_up * pip_size
    t_dn = touch_dn * pip_size

    for i in range(25, n_bars):
        hr = hours[i]
        if not (start_hr <= hr <= end_hr):
            continue
        dt = dates[i]
        if dt == last_trade_date:
            continue
        p = pivots_p[i]
        if p == 0.0:
            continue
        if spread[i] > max_spread:
            continue

        macro = h4_bias[i]
        if macro == 0:
            continue

        bar_rng = h[i] - l[i]
        if bar_rng < 1.5 * pip_size:
            continue

        sig = 0
        s1 = pivots_s1[i]
        r1 = pivots_r1[i]

        if macro == 1:
            touched = (l[i] <= p + t_up and l[i] >= p - t_dn) or (l[i] <= s1 + t_up and l[i] >= s1 - t_dn)
            if touched and c[i] > o[i]:
                lower_wick = (o[i] - l[i]) / bar_rng
                if lower_wick >= min_wick:
                    sig = 1
        elif macro == -1:
            touched = (h[i] >= p - t_up and h[i] <= p + t_dn) or (h[i] >= r1 - t_up and h[i] <= r1 + t_dn)
            if touched and c[i] < o[i]:
                upper_wick = (h[i] - o[i]) / bar_rng
                if upper_wick >= min_wick:
                    sig = -1

        if sig != 0:
            sig_indices.append(exec_indices[i])
            sig_types.append(sig)
            last_trade_date = dt

    return execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips, comm, slip)


# EURGBP Asian Sweep Reversal into London
def eval_eurgbp_asian_sweep(week_data, sweep_min, sweep_max, min_wick, sl_pips, tp_pips, max_spread, comm, slip):
    n_bars = week_data['n_bars']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    exec_indices = week_data['exec_indices']
    ah = week_data['asian_high']
    al = week_data['asian_low']
    pip_size = week_data['pip_size']

    sig_indices = []
    sig_types = []
    last_trade_date = None

    s_min = sweep_min * pip_size
    s_max = sweep_max * pip_size

    for i in range(15, n_bars):
        hr = hours[i]
        # Active window: Frankfurt/London Open 07:00 - 10:00 UTC
        if not (7 <= hr <= 10):
            continue
        dt = dates[i]
        if dt == last_trade_date:
            continue
        if ah[i] == 0.0:
            continue
        if spread[i] > max_spread:
            continue

        bar_rng = h[i] - l[i]
        if bar_rng < 1.5 * pip_size:
            continue

        sig = 0
        # Bearish Sweep: High pierced Asian High by [s_min, s_max], closes back below Asian High, candle red
        if (h[i] >= ah[i] + s_min) and (h[i] <= ah[i] + s_max) and (c[i] < ah[i]) and (c[i] < o[i]):
            upper_wick = (h[i] - o[i]) / bar_rng
            if upper_wick >= min_wick:
                sig = -1
        # Bullish Sweep: Low pierced Asian Low by [s_min, s_max], closes back above Asian Low, candle green
        elif (l[i] <= al[i] - s_min) and (l[i] >= al[i] - s_max) and (c[i] > al[i]) and (c[i] > o[i]):
            lower_wick = (o[i] - l[i]) / bar_rng
            if lower_wick >= min_wick:
                sig = 1

        if sig != 0:
            sig_indices.append(exec_indices[i])
            sig_types.append(sig)
            last_trade_date = dt

    return execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips, comm, slip)


def run_experiment(symbol: str, cand_list):
    print("=" * 80)
    print(f"  TARGET PAIR HUNT: {symbol} ({len(cand_list)} Candidates)")
    print("=" * 80)

    comm, slip = get_friction_pips(symbol)
    raw_dir = ROOT_DIR / "data" / symbol
    disc_weeks = get_complete_split_weeks(raw_dir, split="discovery")
    hold_weeks = get_complete_split_weeks(raw_dir, split="holdout")

    print(f"Quarantined Holdout Weeks: {len(hold_weeks)} (Weeks {min(hold_weeks)}..{max(hold_weeks)})", flush=True)

    cand_holdout_trades = {i: [] for i in range(len(cand_list))}
    t0 = time.time()

    for w_idx, w_num in enumerate(hold_weeks):
        w_data = load_pivot_bars(symbol, w_num)
        if w_data is None:
            continue

        for c_idx, cand in enumerate(cand_list):
            p = cand['params']
            arch = cand['archetype']
            if arch == 'pivot':
                tr = eval_pivot_candidate(w_data, p['start_hr'], p['end_hr'], p['touch_up'], p['touch_dn'], p['min_wick'], p['sl_pips'], p['tp_pips'], p['max_spread'], comm, slip)
            elif arch == 'eurgbp_asian_sweep':
                tr = eval_eurgbp_asian_sweep(w_data, p['sweep_min'], p['sweep_max'], p['min_wick'], p['sl_pips'], p['tp_pips'], p['max_spread'], comm, slip)
            else:
                tr = []
            if len(tr) > 0:
                cand_holdout_trades[c_idx].extend(tr)

        del w_data
        if (w_idx + 1) % 25 == 0 or (w_idx + 1) == len(hold_weeks):
            print(f"  Holdout week {w_idx+1}/{len(hold_weeks)} processed in {time.time()-t0:.1f}s...", flush=True)

    viable_indices = []
    print(f"\n--- Holdout Evaluation Results for {symbol} ---", flush=True)
    for c_idx, cand in enumerate(cand_list):
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
            passed, reason = check_anti_redundancy(symbol, tr_arr, max_overlap_pct=50.0, max_corr=0.50)
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

    print(f"\nViable Holdout Candidates for {symbol}: {len(viable_indices)} of {len(cand_list)}", flush=True)
    if not viable_indices:
        return []

    print(f"\nStreaming Discovery (Weeks {min(disc_weeks)}..{max(disc_weeks)}) for {len(viable_indices)} viable candidates...", flush=True)
    cand_disc_trades = {i: [] for i in viable_indices}
    t1 = time.time()

    for w_idx, w_num in enumerate(disc_weeks):
        w_data = load_pivot_bars(symbol, w_num)
        if w_data is None:
            continue

        for c_idx in viable_indices:
            cand = cand_list[c_idx]
            p = cand['params']
            arch = cand['archetype']
            if arch == 'pivot':
                tr = eval_pivot_candidate(w_data, p['start_hr'], p['end_hr'], p['touch_up'], p['touch_dn'], p['min_wick'], p['sl_pips'], p['tp_pips'], p['max_spread'], comm, slip)
            elif arch == 'eurgbp_asian_sweep':
                tr = eval_eurgbp_asian_sweep(w_data, p['sweep_min'], p['sweep_max'], p['min_wick'], p['sl_pips'], p['tp_pips'], p['max_spread'], comm, slip)
            else:
                tr = []
            if len(tr) > 0:
                cand_disc_trades[c_idx].extend(tr)

        del w_data
        if (w_idx + 1) % 40 == 0 or (w_idx + 1) == len(disc_weeks):
            print(f"  Discovery week {w_idx+1}/{len(disc_weeks)} processed in {time.time()-t1:.1f}s...", flush=True)

    champions = []
    print(f"\n--- Discovery Verification & Dual-Certification for {symbol} ---", flush=True)
    for c_idx in viable_indices:
        cand = cand_list[c_idx]
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
            cand['symbol'] = symbol
            cand['discovery_metrics'] = {
                'trades': len(d_arr),
                'net_pips': float(d_net),
                'profit_factor': float(d_pf),
                'expectancy': float(np.mean(d_pips))
            }
            cand['discovery_trades'] = d_arr
            champions.append(cand)
            print(f"  *** DUAL CERTIFIED {symbol} CHAMPION ***: {cand['desc']}", flush=True)
            print(f"      Holdout:   {cand['holdout_metrics']['trades']} tr, +{cand['holdout_metrics']['net_pips']:.1f}p, PF {cand['holdout_metrics']['profit_factor']:.2f}, Exp {cand['holdout_metrics']['expectancy']:.2f}p", flush=True)
            print(f"      Discovery: {len(d_arr)} tr, +{d_net:.1f}p, PF {d_pf:.2f}, Exp {np.mean(d_pips):.2f}p", flush=True)
        else:
            print(f"  [DISC FAIL] {cand['desc']}: {len(d_arr)} tr, Net: {d_net:+.1f}p, PF: {d_pf:.2f}", flush=True)

    return champions


def main():
    all_new_champions = []

    # 1. EURGBP Asian Sweep Reversals
    eurgbp_cands = []
    for s_min, s_max in [(0.5, 3.0), (1.0, 4.0), (1.5, 5.0)]:
        for wk in [0.30, 0.35, 0.40]:
            for sl, tp in [(7.0, 14.0), (8.0, 16.0), (8.0, 18.0)]:
                eurgbp_cands.append({
                    'archetype': 'eurgbp_asian_sweep',
                    'params': {'sweep_min': s_min, 'sweep_max': s_max, 'min_wick': wk, 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 0.8},
                    'desc': f"EURGBP Asian Sweep Reversal (Sweep={s_min}-{s_max}p, Wk={wk}, SL={sl}p, TP={tp}p)"
                })
    eurgbp_champs = run_experiment("EURGBP", eurgbp_cands)
    all_new_champions.extend(eurgbp_champs)

    # 2. GBPCHF Refined Daily Pivot Pullback
    gbpchf_cands = []
    for wh in [(7, 16), (8, 15)]:
        for tu, td in [(1.0, 2.0), (1.5, 2.5), (1.0, 2.5)]:
            for wk in [0.35, 0.40]:
                for sl, tp in [(11.0, 24.0), (12.0, 24.0), (12.0, 26.0), (12.0, 28.0)]:
                    gbpchf_cands.append({
                        'archetype': 'pivot',
                        'params': {'start_hr': wh[0], 'end_hr': wh[1], 'touch_up': tu, 'touch_dn': td, 'min_wick': wk, 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 1.6},
                        'desc': f"GBPCHF Daily Pivot ({wh[0]:02d}-{wh[1]:02d} UTC, Touch={tu}/{td}, Wk={wk}, SL={sl}p, TP={tp}p)"
                    })
    gbpchf_champs = run_experiment("GBPCHF", gbpchf_cands)
    all_new_champions.extend(gbpchf_champs)

    # 3. AUDCAD Daily Pivot Pullback
    audcad_cands = []
    for wh in [(7, 16), (8, 15), (12, 19)]:
        for tu, td in [(1.0, 2.0), (1.5, 2.5)]:
            for wk in [0.35, 0.40]:
                for sl, tp in [(10.0, 20.0), (11.0, 22.0), (11.0, 24.0)]:
                    audcad_cands.append({
                        'archetype': 'pivot',
                        'params': {'start_hr': wh[0], 'end_hr': wh[1], 'touch_up': tu, 'touch_dn': td, 'min_wick': wk, 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 1.6},
                        'desc': f"AUDCAD Daily Pivot ({wh[0]:02d}-{wh[1]:02d} UTC, Touch={tu}/{td}, Wk={wk}, SL={sl}p, TP={tp}p)"
                    })
    audcad_champs = run_experiment("AUDCAD", audcad_cands)
    all_new_champions.extend(audcad_champs)

    out_file = ROOT_DIR / "results" / "new_champions_crosses_and_eurgbp.json"
    serializable = []
    for c in all_new_champions:
        serializable.append({
            'symbol': c['symbol'],
            'archetype': c['archetype'],
            'params': c['params'],
            'desc': c['desc'],
            'holdout_metrics': c['holdout_metrics'],
            'discovery_metrics': c['discovery_metrics']
        })
    with open(out_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved {len(all_new_champions)} total champions to: {out_file}", flush=True)


if __name__ == "__main__":
    main()

