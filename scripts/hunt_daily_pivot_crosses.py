#!/usr/bin/env python3
"""
================================================================================
  DAILY PIVOT PULLBACK DISCOVERY ENGINE: CROSS-PAIRS & EURGBP
================================================================================
Explores the Daily Floor Traders Pivot Point Pullback & Rejection Archetype
across underrepresented currency pairs:
  - EURGBP
  - GBPCHF
  - USDCAD
  - AUDCAD
  - NZDCAD

Quantitative Setup:
  - Previous Day Floor Pivots: P = (H + L + C)/3, S1 = 2P - H, R1 = 2P - L.
  - Macro H4 20-EMA trend agreement.
  - M15 Pullback & Absorption Rejection Wick (>= 35%).
  - Asymmetric 1:2.0 to 1:2.5 Reward-to-Risk.

Certification Standards:
  - Holdout (Weeks 211..300): PF >= 1.20, Trades >= 15, Net Pips > 0, ExTop3 > 0
  - Discovery (Weeks 1..210): PF >= 1.15, Trades >= 30, Net Pips > 0
  - Anti-Redundancy: Overlap < 50% & Corr < 0.50 against existing fleet
  - Full ECN friction: $4.00/lot comm + 0.20p slippage + floating spreads
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

# Load existing fleet trades for anti-redundancy
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


def load_pivot_week_data(symbol: str, week_num: int):
    pip_size = PIP_SIZES.get(symbol, 0.0001)
    raw_dir = ROOT_DIR / "data" / symbol
    try:
        ticks, _ = get_week_dataset(week=week_num, raw_dir=raw_dir, warmup_prev_week_ticks=WARMUP_TICKS)
    except Exception:
        return None
    if len(ticks) <= WARMUP_TICKS + 500:
        return None

    m15 = build_time_bars(ticks, timeframe_min=15, pip_size=pip_size)
    if len(m15) < 35:
        return None

    # H4 bars directly from M15
    h4 = m15[['high', 'low', 'close']].resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    if len(h4) < 10:
        return None

    h4['ema20'] = h4['close'].ewm(span=20, adjust=False).mean()
    h4['bias'] = np.where(h4['close'] > h4['ema20'], 1, -1)
    h4_bias = h4['bias'].reindex(m15.index, method='ffill').fillna(0).values.astype(np.int8)

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

    # Compute daily pivots per day from previous day's H, L, C
    unique_dates = np.unique(dates)
    daily_stats = {}
    for d in unique_dates:
        mask = (dates == d)
        if np.sum(mask) >= 10:
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

    return {
        'ticks': ticks,
        'n_ticks': n_ticks,
        'pip_size': pip_size,
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
        'n_bars': len(m15),
        'signals_buf': np.zeros(n_ticks, dtype=np.int8),
        'sl_buf': np.zeros(n_ticks, dtype=np.float64),
        'tp_buf': np.zeros(n_ticks, dtype=np.float64)
    }


def eval_pivot_pullback(week_data, start_hr, end_hr, touch_up, touch_dn, min_wick, sl_pips, tp_pips, max_spread, comm, slip):
    n_bars = week_data['n_bars']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    h4_bias = week_data['h4_bias']
    pivots_p = week_data['pivots_p']
    pivots_s1 = week_data['pivots_s1']
    pivots_r1 = week_data['pivots_r1']
    pip_size = week_data['pip_size']
    exec_indices = week_data['exec_indices']
    n_ticks = week_data['n_ticks']
    signals_buf = week_data['signals_buf']
    sl_buf = week_data['sl_buf']
    tp_buf = week_data['tp_buf']

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
        if bar_rng < 2.0 * pip_size:
            continue

        sig = 0
        s1 = pivots_s1[i]
        r1 = pivots_r1[i]

        # Bullish setup: Macro up, test P or S1 from above, reject up
        if macro == 1:
            touched = (l[i] <= p + t_up and l[i] >= p - t_dn) or (l[i] <= s1 + t_up and l[i] >= s1 - t_dn)
            if touched and c[i] > o[i]:
                lower_wick = (o[i] - l[i]) / bar_rng
                if lower_wick >= min_wick:
                    sig = 1
        # Bearish setup: Macro down, test P or R1 from below, reject down
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

    if len(sig_indices) == 0:
        return []

    fired = []
    for idx, s_type in zip(sig_indices, sig_types):
        if idx < n_ticks:
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
        pip_size=pip_size,
        commission_pips=comm,
        entry_slippage_pips=slip,
        enforce_subsequent_tick=True
    )

    for idx in fired:
        signals_buf[idx] = 0
        sl_buf[idx] = 0.0
        tp_buf[idx] = 0.0

    return trades


def generate_pivot_candidates():
    candidates = []

    # EURGBP
    for sh, eh in [(7, 16), (6, 18)]:
        for sl, tp in [(7.0, 14.0), (8.0, 18.0), (9.0, 20.0)]:
            candidates.append({
                'symbol': 'EURGBP',
                'params': {'start_hr': sh, 'end_hr': eh, 'touch_up': 1.5, 'touch_dn': 2.5, 'min_wick': 0.35, 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 0.8},
                'desc': f"EURGBP Daily Pivot Pullback ({sh:02d}-{eh:02d} UTC, SL={sl}p, TP={tp}p)"
            })

    # GBPCHF (Refined Asymmetric Runners)
    for sh, eh in [(7, 16)]:
        for sl in [22.0, 25.0]:
            for tp in [55.0, 65.0, 75.0]:
                candidates.append({
                    'symbol': 'GBPCHF',
                    'params': {'start_hr': sh, 'end_hr': eh, 'touch_up': 3.0, 'touch_dn': 4.5, 'min_wick': 0.35, 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 3.0},
                    'desc': f"GBPCHF Daily Pivot Pullback ({sh:02d}-{eh:02d} UTC, SL={sl}p, TP={tp}p)"
                })

    # GBPJPY
    for sh, eh in [(7, 16), (12, 20)]:
        for sl, tp in [(25.0, 60.0), (30.0, 75.0), (35.0, 85.0)]:
            candidates.append({
                'symbol': 'GBPJPY',
                'params': {'start_hr': sh, 'end_hr': eh, 'touch_up': 3.5, 'touch_dn': 5.0, 'min_wick': 0.35, 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 2.5},
                'desc': f"GBPJPY Daily Pivot Pullback ({sh:02d}-{eh:02d} UTC, SL={sl}p, TP={tp}p)"
            })

    # EURNZD
    for sh, eh in [(7, 16), (6, 18)]:
        for sl, tp in [(25.0, 60.0), (30.0, 75.0), (35.0, 85.0)]:
            candidates.append({
                'symbol': 'EURNZD',
                'params': {'start_hr': sh, 'end_hr': eh, 'touch_up': 3.5, 'touch_dn': 5.0, 'min_wick': 0.35, 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 3.5},
                'desc': f"EURNZD Daily Pivot Pullback ({sh:02d}-{eh:02d} UTC, SL={sl}p, TP={tp}p)"
            })

    # USDCAD
    for sh, eh in [(12, 20), (7, 19)]:
        for sl, tp in [(15.0, 32.0), (18.0, 40.0), (20.0, 48.0)]:
            candidates.append({
                'symbol': 'USDCAD',
                'params': {'start_hr': sh, 'end_hr': eh, 'touch_up': 2.5, 'touch_dn': 4.0, 'min_wick': 0.35, 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 2.0},
                'desc': f"USDCAD Daily Pivot Pullback ({sh:02d}-{eh:02d} UTC, SL={sl}p, TP={tp}p)"
            })

    # AUDCAD
    for sh, eh in [(0, 16), (7, 17)]:
        for sl, tp in [(14.0, 28.0), (16.0, 36.0), (18.0, 40.0)]:
            candidates.append({
                'symbol': 'AUDCAD',
                'params': {'start_hr': sh, 'end_hr': eh, 'touch_up': 2.0, 'touch_dn': 3.5, 'min_wick': 0.35, 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 2.2},
                'desc': f"AUDCAD Daily Pivot Pullback ({sh:02d}-{eh:02d} UTC, SL={sl}p, TP={tp}p)"
            })

    # NZDCAD
    for sh, eh in [(0, 16), (7, 17)]:
        for sl, tp in [(14.0, 28.0), (16.0, 36.0), (18.0, 40.0)]:
            candidates.append({
                'symbol': 'NZDCAD',
                'params': {'start_hr': sh, 'end_hr': eh, 'touch_up': 2.0, 'touch_dn': 3.5, 'min_wick': 0.35, 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 2.5},
                'desc': f"NZDCAD Daily Pivot Pullback ({sh:02d}-{eh:02d} UTC, SL={sl}p, TP={tp}p)"
            })

    return candidates


def main():
    print("=" * 80)
    print("  DAILY PIVOT POINT PULLBACK DISCOVERY: CROSS-PAIRS & EURGBP")
    print("=" * 80)

    candidates = generate_pivot_candidates()
    print(f"Generated {len(candidates)} total daily pivot setups.")

    by_symbol = defaultdict(list)
    for c in candidates:
        by_symbol[c['symbol']].append(c)

    all_champions = []

    for sym in ['GBPCHF', 'GBPJPY', 'EURNZD']:
        sym_cands = by_symbol[sym]
        print(f"\n>>> Processing Symbol: {sym} ({len(sym_cands)} candidates) <<<")

        pip_size = PIP_SIZES.get(sym, 0.0001)
        comm, slip = get_friction_pips(sym)
        raw_dir = ROOT_DIR / "data" / sym
        disc_weeks = get_complete_split_weeks(raw_dir, split="discovery")
        hold_weeks = get_complete_split_weeks(raw_dir, split="holdout")

        print(f"[{sym}] Quarantined Holdout Weeks: {len(hold_weeks)} (Weeks {min(hold_weeks)}..{max(hold_weeks)})")
        print(f"[{sym}] Streaming holdout tick data...")

        cand_holdout_trades = {i: [] for i in range(len(sym_cands))}

        t0 = time.time()
        for w_idx, w_num in enumerate(hold_weeks):
            w_data = load_pivot_week_data(sym, w_num)
            if w_data is None:
                continue

            for c_idx, cand in enumerate(sym_cands):
                p = cand['params']
                tr = eval_pivot_pullback(w_data, p['start_hr'], p['end_hr'], p['touch_up'], p['touch_dn'], p['min_wick'], p['sl_pips'], p['tp_pips'], p['max_spread'], comm, slip)
                if len(tr) > 0:
                    cand_holdout_trades[c_idx].extend(tr)

            del w_data
            if (w_idx + 1) % 25 == 0 or (w_idx + 1) == len(hold_weeks):
                print(f"  Holdout week {w_idx+1}/{len(hold_weeks)} processed in {time.time()-t0:.1f}s...")

        viable_indices = []
        for c_idx, cand in enumerate(sym_cands):
            tr_list = cand_holdout_trades[c_idx]
            if len(tr_list) == 0:
                print(f"  [ZERO TRADES] {cand['desc']}")
                continue
            tr_arr = np.array(tr_list)
            pips = tr_arr['pips']
            gross_win = np.sum(pips[pips > 0])
            gross_loss = np.abs(np.sum(pips[pips < 0]))
            pf = (gross_win / gross_loss) if gross_loss > 0 else 99.0
            net_pips = np.sum(pips)
            sorted_pips = np.sort(pips)[::-1]
            pips_ex_top3 = net_pips - np.sum(sorted_pips[:3]) if len(sorted_pips) >= 3 else net_pips

            print(f"  [CANDIDATE STATS] {cand['desc']}: {len(tr_arr)} tr, Net: {net_pips:+.1f}p, PF: {pf:.2f}, ExTop3: {pips_ex_top3:+.1f}p")

            if len(tr_list) < 15:
                continue

            if pf >= 1.20 and net_pips > 0 and pips_ex_top3 > 0:
                passed, reason = check_anti_redundancy(sym, tr_arr, max_overlap_pct=50.0, max_corr=0.50)
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
                    print(f"  *** [HOLDOUT PASS] *** {cand['desc']}: {len(tr_arr)} tr, +{net_pips:.1f}p, PF {pf:.2f}, Exp {np.mean(pips):.2f}p, ExTop3 +{pips_ex_top3:.1f}p")
                else:
                    print(f"  [REDUNDANT] {cand['desc']}: {reason}")

        print(f"[{sym}] Viable Holdout Candidates: {len(viable_indices)} of {len(sym_cands)}")

        if not viable_indices:
            print(f"[{sym}] No candidates cleared holdout gate.")
            continue

        # Evaluate Discovery on viable candidates
        print(f"[{sym}] Streaming Discovery (Weeks {min(disc_weeks)}..{max(disc_weeks)}) for {len(viable_indices)} candidates...")
        cand_disc_trades = {i: [] for i in viable_indices}

        t1 = time.time()
        for w_idx, w_num in enumerate(disc_weeks):
            w_data = load_pivot_week_data(sym, w_num)
            if w_data is None:
                continue

            for c_idx in viable_indices:
                cand = sym_cands[c_idx]
                p = cand['params']
                tr = eval_pivot_pullback(w_data, p['start_hr'], p['end_hr'], p['touch_up'], p['touch_dn'], p['min_wick'], p['sl_pips'], p['tp_pips'], p['max_spread'], comm, slip)
                if len(tr) > 0:
                    cand_disc_trades[c_idx].extend(tr)

            del w_data
            if (w_idx + 1) % 40 == 0 or (w_idx + 1) == len(disc_weeks):
                print(f"  Discovery week {w_idx+1}/{len(disc_weeks)} processed in {time.time()-t1:.1f}s...")

        # Final Verification
        for c_idx in viable_indices:
            cand = sym_cands[c_idx]
            d_list = cand_disc_trades[c_idx]
            if len(d_list) < 30:
                print(f"  [DISC UNDERTRADED] {cand['desc']}: only {len(d_list)} trades")
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
                print(f"  *** DUAL CERTIFIED PIVOT CHAMPION ***: {cand['desc']}")
                print(f"      Holdout:   {cand['holdout_metrics']['trades']} tr, +{cand['holdout_metrics']['net_pips']:.1f}p, PF {cand['holdout_metrics']['profit_factor']:.2f}, Exp {cand['holdout_metrics']['expectancy']:.2f}p")
                print(f"      Discovery: {len(d_arr)} tr, +{d_net:.1f}p, PF {d_pf:.2f}, Exp {np.mean(d_pips):.2f}p")

    print("\n" + "=" * 80)
    print(f"PIVOT DISCOVERY COMPLETE: Found {len(all_champions)} New Institutional Champions!")
    print("=" * 80)

    out_file = ROOT_DIR / "results" / "new_champions_pivot_crosses.json"
    serializable = []
    for c in all_champions:
        serializable.append({
            'symbol': c['symbol'],
            'params': c['params'],
            'desc': c['desc'],
            'holdout_metrics': c['holdout_metrics'],
            'discovery_metrics': c['discovery_metrics']
        })
    with open(out_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved champions to: {out_file}")


if __name__ == "__main__":
    main()

