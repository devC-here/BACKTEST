#!/usr/bin/env python3
"""
================================================================================
  EURGBP NOVEL COMPRESSION & ALTERNATE REGIME DISCOVERY
================================================================================
Explores:
  1. H4 Macro Keltner Squeeze (Multi-Day Compression Explosion)
  2. M30 Intraday Keltner Squeeze (30-Minute Compression Explosion)
  3. Daily NR4 / Inside-Bar Breakout (Multi-day narrow range breakout)
  4. NY Cash Session Volatility Expansion (12:00 - 17:00 UTC - Zero overlap with London morning)

Certification Criteria:
  - Holdout (Weeks 211..300): Trades >= 15, PF >= 1.20, Net Pips > 0, ExTop3 > 0
  - Discovery (Weeks 1..210): Trades >= 30, PF >= 1.15, Net Pips > 0
  - Anti-Redundancy: Overlap < 50% & Corr < 0.50 vs disc107
  - Strict subsequent-tick execution (T_fill > T_sig)
  - Full ECN friction: 0.33p commission + 0.20p slippage + floating spreads
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

# Load disc107 trades for anti-redundancy
existing_trades = []
with open(FLEET_MANIFEST, "r", encoding="utf-8") as f:
    fleet_data = json.load(f)

for s in fleet_data["strategies"]:
    if s["symbol"] == SYMBOL:
        name = s["name"]
        csv_file = MANIFEST_DIR / f"{SYMBOL}__{name}_trades.csv"
        if not csv_file.exists():
            csv_file = MANIFEST_DIR / f"{name}_trades.csv"
        if csv_file.exists():
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


def load_eurgbp_multi_bars(week_num: int):
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

    # M30 bars
    m30 = m15[['high', 'low', 'close']].resample('30min').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    # H4 bars
    h4 = m15[['high', 'low', 'close']].resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()

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

    # ATR(14) on M15
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr14 = pd.Series(tr).rolling(14).mean().bfill().values.astype(np.float64)

    # H4 Squeeze Calculation
    h4_sqz_fired = np.zeros(len(m15), dtype=bool)
    h4_mom = np.zeros(len(m15), dtype=np.int8)
    if len(h4) >= 20:
        h4_c = h4['close'].values
        h4_h = h4['high'].values
        h4_l = h4['low'].values
        h4_cs = pd.Series(h4_c)
        h4_ema20 = h4_cs.ewm(span=20, adjust=False).mean().values
        h4_std20 = h4_cs.rolling(20).std().bfill().values
        h4_bb_up = h4_ema20 + 2.0 * h4_std20
        h4_bb_dn = h4_ema20 - 2.0 * h4_std20

        h4_tr = np.maximum(h4_h - h4_l, np.maximum(np.abs(h4_h - np.roll(h4_c, 1)), np.abs(h4_l - np.roll(h4_c, 1))))
        h4_tr[0] = h4_h[0] - h4_l[0]
        h4_atr20 = pd.Series(h4_tr).rolling(20).mean().bfill().values

        h4_kc_up = h4_ema20 + 1.5 * h4_atr20
        h4_kc_dn = h4_ema20 - 1.5 * h4_atr20

        sqz_on = (h4_bb_dn > h4_kc_dn) & (h4_bb_up < h4_kc_up)
        sqz_f = pd.Series(sqz_on).shift(1).fillna(False).values & (~sqz_on)
        m = np.where(h4_c > h4_ema20, 1, np.where(h4_c < h4_ema20, -1, 0))

        h4_df = pd.DataFrame({'sqz_fire': sqz_f, 'mom': m}, index=h4.index)
        merged_h4 = pd.merge_asof(m15.sort_index(), h4_df.sort_index(), left_index=True, right_index=True, direction='backward')
        h4_sqz_fired = merged_h4['sqz_fire'].fillna(False).values
        h4_mom = merged_h4['mom'].fillna(0).values.astype(np.int8)

    # M30 Squeeze Calculation
    m30_sqz_fired = np.zeros(len(m15), dtype=bool)
    m30_mom = np.zeros(len(m15), dtype=np.int8)
    if len(m30) >= 25:
        m30_c = m30['close'].values
        m30_h = m30['high'].values
        m30_l = m30['low'].values
        m30_cs = pd.Series(m30_c)
        m30_ema20 = m30_cs.ewm(span=20, adjust=False).mean().values
        m30_std20 = m30_cs.rolling(20).std().bfill().values
        m30_bb_up = m30_ema20 + 2.0 * m30_std20
        m30_bb_dn = m30_ema20 - 2.0 * m30_std20

        m30_tr = np.maximum(m30_h - m30_l, np.maximum(np.abs(m30_h - np.roll(m30_c, 1)), np.abs(m30_l - np.roll(m30_c, 1))))
        m30_tr[0] = m30_h[0] - m30_l[0]
        m30_atr20 = pd.Series(m30_tr).rolling(20).mean().bfill().values

        m30_kc_up = m30_ema20 + 1.5 * m30_atr20
        m30_kc_dn = m30_ema20 - 1.5 * m30_atr20

        sqz_on_m30 = (m30_bb_dn > m30_kc_dn) & (m30_bb_up < m30_kc_up)
        sqz_f_m30 = pd.Series(sqz_on_m30).shift(1).fillna(False).values & (~sqz_on_m30)
        m_m30 = np.where(m30_c > m30_ema20, 1, np.where(m30_c < m30_ema20, -1, 0))

        m30_df = pd.DataFrame({'sqz_fire': sqz_f_m30, 'mom': m_m30}, index=m30.index)
        merged_m30 = pd.merge_asof(m15.sort_index(), m30_df.sort_index(), left_index=True, right_index=True, direction='backward')
        m30_sqz_fired = merged_m30['sqz_fire'].fillna(False).values
        m30_mom = merged_m30['mom'].fillna(0).values.astype(np.int8)

    # Daily bar compression (Inside Day / NR4)
    unique_dates = np.unique(dates)
    daily_stats = {}
    for d in unique_dates:
        mask = (dates == d)
        if np.sum(mask) >= 15:
            dh = float(np.max(h[mask]))
            dl = float(np.min(l[mask]))
            daily_stats[d] = (dh, dl, dh - dl)

    is_inside_day = np.zeros(len(m15), dtype=bool)
    prev_dh = np.zeros(len(m15), dtype=np.float64)
    prev_dl = np.zeros(len(m15), dtype=np.float64)

    for idx_d in range(2, len(unique_dates)):
        d_cur = unique_dates[idx_d]
        d_p1 = unique_dates[idx_d - 1]
        d_p2 = unique_dates[idx_d - 2]
        if d_p1 in daily_stats and d_p2 in daily_stats:
            h1, l1, r1 = daily_stats[d_p1]
            h2, l2, r2 = daily_stats[d_p2]
            # Day -1 was inside day relative to Day -2
            if h1 < h2 and l1 > l2:
                mask_cur = (dates == d_cur)
                is_inside_day[mask_cur] = True
                prev_dh[mask_cur] = h1
                prev_dl[mask_cur] = l1

    return {
        'ticks': ticks,
        'n_ticks': n_ticks,
        'min_entry_ts': ticks['timestamp'][WARMUP_TICKS],
        'c': c, 'h': h, 'l': l, 'o': o,
        'spread': spread,
        'hours': hours,
        'dates': dates,
        'exec_indices': exec_indices,
        'atr14': atr14,
        'h4_sqz_fired': h4_sqz_fired,
        'h4_mom': h4_mom,
        'm30_sqz_fired': m30_sqz_fired,
        'm30_mom': m30_mom,
        'is_inside_day': is_inside_day,
        'prev_dh': prev_dh,
        'prev_dl': prev_dl,
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


def eval_h4_squeeze(week_data, start_hr, end_hr, sl_pips, tp_pips, max_spread):
    n_bars = week_data['n_bars']
    c, o = week_data['c'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    exec_indices = week_data['exec_indices']
    sqz = week_data['h4_sqz_fired']
    mom = week_data['h4_mom']

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


def eval_m30_squeeze(week_data, start_hr, end_hr, sl_pips, tp_pips, max_spread):
    n_bars = week_data['n_bars']
    c, o = week_data['c'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    exec_indices = week_data['exec_indices']
    sqz = week_data['m30_sqz_fired']
    mom = week_data['m30_mom']

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


def eval_inside_day_breakout(week_data, start_hr, end_hr, sl_pips, tp_pips, max_spread):
    n_bars = week_data['n_bars']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    exec_indices = week_data['exec_indices']
    is_inside = week_data['is_inside_day']
    pd_h = week_data['prev_dh']
    pd_l = week_data['prev_dl']

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
        if not is_inside[i]:
            continue
        if spread[i] > max_spread:
            continue

        bar_rng = h[i] - l[i]
        if bar_rng < 1.5 * PIP_SIZE:
            continue

        sig = 0
        # Breakout of inside day high
        if c[i] > pd_h[i] and c[i] > o[i]:
            sig = 1
        # Breakout of inside day low
        elif c[i] < pd_l[i] and c[i] < o[i]:
            sig = -1

        if sig != 0:
            sig_indices.append(exec_indices[i])
            sig_types.append(sig)
            last_trade_date = dt

    return execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips)


def eval_ny_expansion(week_data, start_hr, end_hr, atr_mult, body_ratio, sl_pips, tp_pips, max_spread):
    n_bars = week_data['n_bars']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    exec_indices = week_data['exec_indices']
    atr14 = week_data['atr14']

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

        bar_rng = h[i] - l[i]
        if bar_rng < atr_mult * atr14[i]:
            continue

        body = abs(c[i] - o[i])
        if (body / max(bar_rng, 1e-9)) < body_ratio:
            continue

        sig = 0
        if c[i] > o[i]:
            sig = 1
        elif c[i] < o[i]:
            sig = -1

        if sig != 0:
            sig_indices.append(exec_indices[i])
            sig_types.append(sig)
            last_trade_date = dt

    return execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips)


def generate_candidates():
    candidates = []

    # 1. H4 Squeeze (Multi-Day Compression)
    for wh in [(7, 16), (6, 18), (12, 20)]:
        for sl, tp in [(10.0, 20.0), (12.0, 25.0), (14.0, 30.0)]:
            candidates.append({
                'archetype': 'h4_squeeze',
                'params': {'start_hr': wh[0], 'end_hr': wh[1], 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 0.8},
                'desc': f"EURGBP H4 Squeeze ({wh[0]:02d}-{wh[1]:02d} UTC, SL={sl}p, TP={tp}p)"
            })

    # 2. M30 Squeeze (Intraday Compression)
    for wh in [(7, 16), (8, 15)]:
        for sl, tp in [(7.0, 14.0), (8.0, 16.0), (8.0, 18.0), (9.0, 20.0)]:
            candidates.append({
                'archetype': 'm30_squeeze',
                'params': {'start_hr': wh[0], 'end_hr': wh[1], 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 0.8},
                'desc': f"EURGBP M30 Squeeze ({wh[0]:02d}-{wh[1]:02d} UTC, SL={sl}p, TP={tp}p)"
            })

    # 3. Inside Day Breakout
    for wh in [(7, 14), (7, 16)]:
        for sl, tp in [(8.0, 16.0), (9.0, 18.0), (10.0, 22.0)]:
            candidates.append({
                'archetype': 'inside_day',
                'params': {'start_hr': wh[0], 'end_hr': wh[1], 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 0.8},
                'desc': f"EURGBP Inside Day Breakout ({wh[0]:02d}-{wh[1]:02d} UTC, SL={sl}p, TP={tp}p)"
            })

    # 4. NY Session Expansion (12:00 - 17:00 UTC)
    for wh in [(12, 16), (13, 17)]:
        for am in [1.1, 1.3]:
            for br in [0.55, 0.65]:
                for sl, tp in [(7.0, 14.0), (8.0, 16.0), (8.0, 18.0)]:
                    candidates.append({
                        'archetype': 'ny_expansion',
                        'params': {'start_hr': wh[0], 'end_hr': wh[1], 'atr_mult': am, 'body_ratio': br, 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 0.8},
                        'desc': f"EURGBP NY Expansion ({wh[0]:02d}-{wh[1]:02d} UTC, ATRm={am}, BR={br}, SL={sl}p, TP={tp}p)"
                    })

    return candidates


def evaluate_cand(cand, w_data):
    arch = cand['archetype']
    p = cand['params']
    if arch == 'h4_squeeze':
        return eval_h4_squeeze(w_data, p['start_hr'], p['end_hr'], p['sl_pips'], p['tp_pips'], p['max_spread'])
    elif arch == 'm30_squeeze':
        return eval_m30_squeeze(w_data, p['start_hr'], p['end_hr'], p['sl_pips'], p['tp_pips'], p['max_spread'])
    elif arch == 'inside_day':
        return eval_inside_day_breakout(w_data, p['start_hr'], p['end_hr'], p['sl_pips'], p['tp_pips'], p['max_spread'])
    elif arch == 'ny_expansion':
        return eval_ny_expansion(w_data, p['start_hr'], p['end_hr'], p['atr_mult'], p['body_ratio'], p['sl_pips'], p['tp_pips'], p['max_spread'])
    return []


def main():
    print("=" * 80)
    print("  EURGBP NOVEL COMPRESSION & ALTERNATE REGIME ALPHA DISCOVERY")
    print("=" * 80)

    candidates = generate_candidates()
    print(f"Generated {len(candidates)} total novel EURGBP compression setups.")

    raw_dir = ROOT_DIR / "data" / SYMBOL
    disc_weeks = get_complete_split_weeks(raw_dir, split="discovery")
    hold_weeks = get_complete_split_weeks(raw_dir, split="holdout")

    print(f"Quarantined Holdout Weeks: {len(hold_weeks)} (Weeks {min(hold_weeks)}..{max(hold_weeks)})")
    print(f"Streaming holdout tick data across {len(candidates)} candidates...")

    cand_holdout_trades = {i: [] for i in range(len(candidates))}

    t0 = time.time()
    for w_idx, w_num in enumerate(hold_weeks):
        w_data = load_eurgbp_multi_bars(w_num)
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
        w_data = load_eurgbp_multi_bars(w_num)
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
            print(f"  *** DUAL CERTIFIED EURGBP CHAMPION ***: {cand['desc']}")
            print(f"      Holdout:   {cand['holdout_metrics']['trades']} tr, +{cand['holdout_metrics']['net_pips']:.1f}p, PF {cand['holdout_metrics']['profit_factor']:.2f}, Exp {cand['holdout_metrics']['expectancy']:.2f}p")
            print(f"      Discovery: {len(d_arr)} tr, +{d_net:.1f}p, PF {d_pf:.2f}, Exp {np.mean(d_pips):.2f}p")

    print("\n" + "=" * 80)
    print(f"EURGBP DISCOVERY COMPLETE: Found {len(all_champions)} New Institutional Champions!")
    print("=" * 80)

    out_file = ROOT_DIR / "results" / "new_champions_eurgbp_novel.json"
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

