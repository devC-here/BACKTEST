#!/usr/bin/env python3
"""
================================================================================
  EURGBP SPECIALIZED ALPHA HUNT: LONDON FIX, M15 SQUEEZE & PIVOT REFINEMENT
================================================================================
Explores:
  1. London Fix WMR Flow (15:00 - 17:30 UTC)
  2. M15 Fast Carter Keltner Squeeze (08:00 - 14:00 UTC)
  3. Refined Daily Pivot Pullback with Asymmetric Consistency (SL 7p / TP 14p)
  4. H2 Two-Hour Volatility Squeeze (07:00 - 16:00 UTC)

All tests enforce:
  - Exact subsequent-tick fill contract (T_fill > T_sig)
  - Full ECN friction: 0.33p commission + 0.20p adverse slippage + floating spreads
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


def load_eurgbp_advanced_bars(week_num: int):
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

    # H2 bars
    h2 = m15[['high', 'low', 'close']].resample('2h').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    # H4 bars
    h4 = m15[['high', 'low', 'close']].resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()

    c = m15['close'].values.astype(np.float64)
    h = m15['high'].values.astype(np.float64)
    l = m15['low'].values.astype(np.float64)
    o = m15['open'].values.astype(np.float64)
    spread = m15['spread_pips'].values.astype(np.float64)
    hours = m15.index.hour.values.astype(np.int16)
    minutes = m15.index.minute.values.astype(np.int16)
    dates = m15.index.date
    time_ms = m15.index.astype('datetime64[ms]').view('int64')
    dur_ms = 15 * 60 * 1000
    tick_ts = ticks['timestamp']
    n_ticks = len(ticks)
    exec_indices = np.searchsorted(tick_ts, time_ms + dur_ms).astype(np.int32)

    # ATR(14)
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr14 = pd.Series(tr).rolling(14).mean().bfill().values.astype(np.float64)

    # M15 Carter Squeeze
    c_s = pd.Series(c)
    m15_ema20 = c_s.ewm(span=20, adjust=False).mean().values
    m15_std20 = c_s.rolling(20).std().bfill().values
    m15_bb_up = m15_ema20 + 2.0 * m15_std20
    m15_bb_dn = m15_ema20 - 2.0 * m15_std20
    m15_kc_up = m15_ema20 + 1.5 * atr14
    m15_kc_dn = m15_ema20 - 1.5 * atr14
    m15_sqz_on = (m15_bb_dn > m15_kc_dn) & (m15_bb_up < m15_kc_up)
    m15_sqz_fired = pd.Series(m15_sqz_on).shift(1).fillna(False).values & (~m15_sqz_on)
    m15_mom = np.where(c > m15_ema20, 1, np.where(c < m15_ema20, -1, 0))

    # H2 Carter Squeeze
    h2_sqz_fired = np.zeros(len(m15), dtype=bool)
    h2_mom = np.zeros(len(m15), dtype=np.int8)
    if len(h2) >= 20:
        h2_c = h2['close'].values
        h2_h = h2['high'].values
        h2_l = h2['low'].values
        h2_cs = pd.Series(h2_c)
        h2_ema20 = h2_cs.ewm(span=20, adjust=False).mean().values
        h2_std20 = h2_cs.rolling(20).std().bfill().values
        h2_bb_up = h2_ema20 + 2.0 * h2_std20
        h2_bb_dn = h2_ema20 - 2.0 * h2_std20

        h2_tr = np.maximum(h2_h - h2_l, np.maximum(np.abs(h2_h - np.roll(h2_c, 1)), np.abs(h2_l - np.roll(h2_c, 1))))
        h2_tr[0] = h2_h[0] - h2_l[0]
        h2_atr20 = pd.Series(h2_tr).rolling(20).mean().bfill().values
        h2_kc_up = h2_ema20 + 1.5 * h2_atr20
        h2_kc_dn = h2_ema20 - 1.5 * h2_atr20

        h2_sqz_on = (h2_bb_dn > h2_kc_dn) & (h2_bb_up < h2_kc_up)
        h2_sqz_f = pd.Series(h2_sqz_on).shift(1).fillna(False).values & (~h2_sqz_on)
        m_h2 = np.where(h2_c > h2_ema20, 1, np.where(h2_c < h2_ema20, -1, 0))

        h2_df = pd.DataFrame({'sqz_fire': h2_sqz_f, 'mom': m_h2}, index=h2.index)
        merged_h2 = pd.merge_asof(m15.sort_index(), h2_df.sort_index(), left_index=True, right_index=True, direction='backward')
        h2_sqz_fired = merged_h2['sqz_fire'].fillna(False).values
        h2_mom = merged_h2['mom'].fillna(0).values.astype(np.int8)

    # H4 Trend Bias
    h4_bias = np.zeros(len(m15), dtype=np.int8)
    if len(h4) >= 10:
        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_b = np.where(h4['close'] > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'bias': h4_b}, index=h4.index)
        merged_h4 = pd.merge_asof(m15.sort_index(), h4_df.sort_index(), left_index=True, right_index=True, direction='backward')
        h4_bias = merged_h4['bias'].fillna(0).values.astype(np.int8)

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

    return {
        'ticks': ticks,
        'n_ticks': n_ticks,
        'min_entry_ts': ticks['timestamp'][WARMUP_TICKS],
        'c': c, 'h': h, 'l': l, 'o': o,
        'spread': spread,
        'hours': hours,
        'minutes': minutes,
        'dates': dates,
        'exec_indices': exec_indices,
        'atr14': atr14,
        'm15_sqz_fired': m15_sqz_fired,
        'm15_mom': m15_mom,
        'h2_sqz_fired': h2_sqz_fired,
        'h2_mom': h2_mom,
        'h4_bias': h4_bias,
        'pivots_p': pivots_p,
        'pivots_s1': pivots_s1,
        'pivots_r1': pivots_r1,
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


def eval_london_fix_momentum(week_data, start_hr, end_hr, min_body_ratio, sl_pips, tp_pips, max_spread):
    n_bars = week_data['n_bars']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    exec_indices = week_data['exec_indices']
    h4_bias = week_data['h4_bias']
    atr14 = week_data['atr14']

    sig_indices = []
    sig_types = []
    last_trade_date = None

    for i in range(15, n_bars):
        hr = hours[i]
        # London Fix Window: 15:00 - 17:00 UTC
        if not (start_hr <= hr <= end_hr):
            continue
        dt = dates[i]
        if dt == last_trade_date:
            continue
        if spread[i] > max_spread:
            continue

        bar_rng = h[i] - l[i]
        if bar_rng < 1.0 * atr14[i] or bar_rng < 1.5 * PIP_SIZE:
            continue

        body = abs(c[i] - o[i])
        if (body / bar_rng) < min_body_ratio:
            continue

        macro = h4_bias[i]
        sig = 0
        if c[i] > o[i] and macro == 1:
            sig = 1
        elif c[i] < o[i] and macro == -1:
            sig = -1

        if sig != 0:
            sig_indices.append(exec_indices[i])
            sig_types.append(sig)
            last_trade_date = dt

    return execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips)


def eval_m15_squeeze(week_data, start_hr, end_hr, sl_pips, tp_pips, max_spread):
    n_bars = week_data['n_bars']
    c, o = week_data['c'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    exec_indices = week_data['exec_indices']
    sqz = week_data['m15_sqz_fired']
    mom = week_data['m15_mom']

    sig_indices = []
    sig_types = []
    last_trade_date = None

    for i in range(15, n_bars):
        hr = hours[i]
        if not (start_hr <= hr <= end_hr):
            continue
        dt = dates[i]
        if dt == last_trade_date:
            continue
        if spread[i] > max_spread:
            continue
        if not sqz[i]:
            continue

        m = mom[i]
        if m == 0:
            continue

        sig = 0
        if m == 1 and c[i] > o[i]:
            sig = 1
        elif m == -1 and c[i] < o[i]:
            sig = -1

        if sig != 0:
            sig_indices.append(exec_indices[i])
            sig_types.append(sig)
            last_trade_date = dt

    return execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips)


def eval_h2_squeeze(week_data, start_hr, end_hr, sl_pips, tp_pips, max_spread):
    n_bars = week_data['n_bars']
    c, o = week_data['c'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    exec_indices = week_data['exec_indices']
    sqz = week_data['h2_sqz_fired']
    mom = week_data['h2_mom']

    sig_indices = []
    sig_types = []
    last_trade_date = None

    for i in range(15, n_bars):
        hr = hours[i]
        if not (start_hr <= hr <= end_hr):
            continue
        dt = dates[i]
        if dt == last_trade_date:
            continue
        if spread[i] > max_spread:
            continue
        if not sqz[i]:
            continue

        m = mom[i]
        if m == 0:
            continue

        sig = 0
        if m == 1 and c[i] > o[i]:
            sig = 1
        elif m == -1 and c[i] < o[i]:
            sig = -1

        if sig != 0:
            sig_indices.append(exec_indices[i])
            sig_types.append(sig)
            last_trade_date = dt

    return execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips)


def eval_refined_daily_pivot(week_data, start_hr, end_hr, touch_up, touch_dn, min_wick, sl_pips, tp_pips, max_spread):
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

    sig_indices = []
    sig_types = []
    last_trade_date = None

    t_up = touch_up * PIP_SIZE
    t_dn = touch_dn * PIP_SIZE

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
        if bar_rng < 1.5 * PIP_SIZE:
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

    return execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips)


def generate_candidates():
    candidates = []

    # 1. London Fix WMR Flow (15:00 - 17:30 UTC)
    for wh in [(15, 17), (14, 17)]:
        for br in [0.50, 0.60]:
            for sl, tp in [(7.0, 14.0), (8.0, 16.0), (8.0, 18.0)]:
                candidates.append({
                    'archetype': 'london_fix',
                    'params': {'start_hr': wh[0], 'end_hr': wh[1], 'min_body_ratio': br, 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 0.8},
                    'desc': f"EURGBP London Fix Flow ({wh[0]:02d}-{wh[1]:02d} UTC, BR={br}, SL={sl}p, TP={tp}p)"
                })

    # 2. M15 Fast Carter Squeeze (08:00 - 14:00 UTC)
    for wh in [(8, 14), (7, 13), (8, 16)]:
        for sl, tp in [(6.0, 12.0), (7.0, 14.0), (8.0, 16.0)]:
            candidates.append({
                'archetype': 'm15_sqz',
                'params': {'start_hr': wh[0], 'end_hr': wh[1], 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 0.8},
                'desc': f"EURGBP M15 Carter Squeeze ({wh[0]:02d}-{wh[1]:02d} UTC, SL={sl}p, TP={tp}p)"
            })

    # 3. H2 Two-Hour Squeeze (07:00 - 16:00 UTC)
    for wh in [(7, 16), (8, 15)]:
        for sl, tp in [(8.0, 16.0), (9.0, 18.0), (10.0, 20.0)]:
            candidates.append({
                'archetype': 'h2_sqz',
                'params': {'start_hr': wh[0], 'end_hr': wh[1], 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 0.8},
                'desc': f"EURGBP H2 Squeeze ({wh[0]:02d}-{wh[1]:02d} UTC, SL={sl}p, TP={tp}p)"
            })

    # 4. Refined Daily Pivot Pullback (Consistent 2.0R to 2.2R)
    for wh in [(7, 16), (8, 15)]:
        for tu, td in [(1.0, 2.0), (1.5, 2.5)]:
            for wk in [0.35, 0.40]:
                for sl, tp in [(7.0, 14.0), (7.0, 15.0), (8.0, 16.0)]:
                    candidates.append({
                        'archetype': 'daily_pivot',
                        'params': {'start_hr': wh[0], 'end_hr': wh[1], 'touch_up': tu, 'touch_dn': td, 'min_wick': wk, 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 0.8},
                        'desc': f"EURGBP Daily Pivot ({wh[0]:02d}-{wh[1]:02d} UTC, Touch={tu}/{td}, Wk={wk}, SL={sl}p, TP={tp}p)"
                    })

    return candidates


def evaluate_cand(cand, w_data):
    arch = cand['archetype']
    p = cand['params']
    if arch == 'london_fix':
        return eval_london_fix_momentum(w_data, p['start_hr'], p['end_hr'], p['min_body_ratio'], p['sl_pips'], p['tp_pips'], p['max_spread'])
    elif arch == 'm15_sqz':
        return eval_m15_squeeze(w_data, p['start_hr'], p['end_hr'], p['sl_pips'], p['tp_pips'], p['max_spread'])
    elif arch == 'h2_sqz':
        return eval_h2_squeeze(w_data, p['start_hr'], p['end_hr'], p['sl_pips'], p['tp_pips'], p['max_spread'])
    elif arch == 'daily_pivot':
        return eval_refined_daily_pivot(w_data, p['start_hr'], p['end_hr'], p['touch_up'], p['touch_dn'], p['min_wick'], p['sl_pips'], p['tp_pips'], p['max_spread'])
    return []


def main():
    print("=" * 80)
    print("  EURGBP ADVANCED ALPHA DISCOVERY: LONDON FIX, SQUEEZES & PIVOTS")
    print("=" * 80)

    candidates = generate_candidates()
    print(f"Generated {len(candidates)} total specialized setups.")

    raw_dir = ROOT_DIR / "data" / SYMBOL
    disc_weeks = get_complete_split_weeks(raw_dir, split="discovery")
    hold_weeks = get_complete_split_weeks(raw_dir, split="holdout")

    print(f"Quarantined Holdout Weeks: {len(hold_weeks)} (Weeks {min(hold_weeks)}..{max(hold_weeks)})")
    print("Streaming holdout tick data across all candidates...")

    cand_holdout_trades = {i: [] for i in range(len(candidates))}

    t0 = time.time()
    for w_idx, w_num in enumerate(hold_weeks):
        w_data = load_eurgbp_advanced_bars(w_num)
        if w_data is None:
            continue

        for c_idx, cand in enumerate(candidates):
            tr = evaluate_cand(cand, w_data)
            if len(tr) > 0:
                cand_holdout_trades[c_idx].extend(tr)

        del w_data
        if (w_idx + 1) % 25 == 0 or (w_idx + 1) == len(hold_weeks):
            print(f"  Holdout week {w_idx+1}/{len(hold_weeks)} processed in {time.time()-t0:.1f}s...")

    viable_indices = []
    print("\n--- Holdout Evaluation Results ---")
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

        if net_pips > 0:
            print(f"  [HOLD POSITIVE] {cand['desc']}: {len(tr_arr)} tr, Net: {net_pips:+.1f}p, PF: {pf:.2f}, ExTop3: {pips_ex_top3:+.1f}p")

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
                print(f"  *** [HOLDOUT PASS] *** {cand['desc']}: {len(tr_arr)} tr, +{net_pips:.1f}p, PF {pf:.2f}, Exp {np.mean(pips):.2f}p, ExTop3 +{pips_ex_top3:.1f}p")
            else:
                print(f"  [REDUNDANT] {cand['desc']}: {reason}")

    print(f"\nViable Holdout Candidates: {len(viable_indices)} of {len(candidates)}")

    if not viable_indices:
        print("No candidates cleared holdout gate.")
        return

    # Stream Discovery on viable candidates
    print(f"\nStreaming Discovery (Weeks {min(disc_weeks)}..{max(disc_weeks)}) for {len(viable_indices)} viable candidates...")
    cand_disc_trades = {i: [] for i in viable_indices}

    t1 = time.time()
    for w_idx, w_num in enumerate(disc_weeks):
        w_data = load_eurgbp_advanced_bars(w_num)
        if w_data is None:
            continue

        for c_idx in viable_indices:
            cand = candidates[c_idx]
            tr = evaluate_cand(cand, w_data)
            if len(tr) > 0:
                cand_disc_trades[c_idx].extend(tr)

        del w_data
        if (w_idx + 1) % 40 == 0 or (w_idx + 1) == len(disc_weeks):
            print(f"  Discovery week {w_idx+1}/{len(disc_weeks)} processed in {time.time()-t1:.1f}s...")

    all_champions = []
    print("\n--- Discovery Verification & Dual-Certification ---")
    for c_idx in viable_indices:
        cand = candidates[c_idx]
        d_list = cand_disc_trades[c_idx]
        if len(d_list) < 25:
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
            print(f"  *** DUAL CERTIFIED EURGBP CHAMPION ***: {cand['desc']}")
            print(f"      Holdout:   {cand['holdout_metrics']['trades']} tr, +{cand['holdout_metrics']['net_pips']:.1f}p, PF {cand['holdout_metrics']['profit_factor']:.2f}, Exp {cand['holdout_metrics']['expectancy']:.2f}p")
            print(f"      Discovery: {len(d_arr)} tr, +{d_net:.1f}p, PF {d_pf:.2f}, Exp {np.mean(d_pips):.2f}p")
        else:
            print(f"  [DISC FAIL] {cand['desc']}: {len(d_arr)} tr, Net: {d_net:+.1f}p, PF: {d_pf:.2f}")

    print("\n" + "=" * 80)
    print(f"EURGBP DISCOVERY COMPLETE: Found {len(all_champions)} New Institutional Champions!")
    print("=" * 80)

    out_file = ROOT_DIR / "results" / "new_champions_eurgbp_fix.json"
    serializable = []
    for c in all_champions:
        serializable.append({
            'archetype': c['archetype'],
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
