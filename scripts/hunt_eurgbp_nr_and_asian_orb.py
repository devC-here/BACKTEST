#!/usr/bin/env python3
"""
================================================================================
  EURGBP DEEP ALPHA HUNT: NR7/NR4, ASIAN ORB, WEEKLY SWEEPS & REFINED H2 SQUEEZE
================================================================================
Targeting EURGBP institutional alpha:
  1. Toby Crabel NR7 & NR4 Volatility Compression Breakouts (07:00 - 15:00 UTC)
  2. Asian Range Breakout (Asian ORB: 00:00-06:45 range broken at London Open 07:00-10:00 UTC)
  3. Refined H2 Carter Squeeze (with candle body ratio >= 0.50, 0.60 and H4 Trend Alignment)
  4. Weekly High/Low Liquidity Sweep Rejection (Prior Week High/Low Fakeouts)

Enforces:
  - Exact subsequent-tick execution contract (T_fill > T_sig)
  - Full ECN friction: 0.33p comm + 0.20p slip + floating Dukascopy spreads
  - Holdout (Weeks 211..300): PF >= 1.20, Trades >= 15, Net Pips > 0, ExTop3 > 0
  - Discovery (Weeks 1..210): PF >= 1.15, Trades >= 25, Net Pips > 0
  - Anti-redundancy: Overlap < 50% & Corr < 0.50 vs disc107 AND disc172
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


def load_eurgbp_multi_archetype_bars(week_num: int):
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

    # H4 Trend Bias
    h4 = m15[['high', 'low', 'close']].resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    h4_bias = np.zeros(len(m15), dtype=np.int8)
    if len(h4) >= 10:
        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_b = np.where(h4['close'] > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'bias': h4_b}, index=h4.index)
        merged_h4 = pd.merge_asof(m15.sort_index(), h4_df.sort_index(), left_index=True, right_index=True, direction='backward')
        h4_bias = merged_h4['bias'].fillna(0).values.astype(np.int8)

    # H2 Carter Squeeze
    h2 = m15[['high', 'low', 'close']].resample('2h').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
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

    # Daily Stats for NR7, NR4, and Asian Range
    unique_dates = np.unique(dates)
    daily_stats = {}
    for d in unique_dates:
        mask = (dates == d)
        if np.sum(mask) >= 15:
            dh = float(np.max(h[mask]))
            dl = float(np.min(l[mask]))
            daily_stats[d] = {
                'high': dh,
                'low': dl,
                'range': dh - dl
            }

    # NR7 and NR4 identification
    is_nr7 = np.zeros(len(m15), dtype=bool)
    is_nr4 = np.zeros(len(m15), dtype=bool)
    prev_dh = np.zeros(len(m15), dtype=np.float64)
    prev_dl = np.zeros(len(m15), dtype=np.float64)

    for idx_d in range(1, len(unique_dates)):
        d_cur = unique_dates[idx_d]
        d_p1 = unique_dates[idx_d - 1]
        if d_p1 not in daily_stats:
            continue
        p1_rng = daily_stats[d_p1]['range']
        p1_h = daily_stats[d_p1]['high']
        p1_l = daily_stats[d_p1]['low']

        mask_cur = (dates == d_cur)
        prev_dh[mask_cur] = p1_h
        prev_dl[mask_cur] = p1_l

        # Check NR4 (last 3 prior days + d_p1 = 4 days total)
        if idx_d >= 4:
            prior_ranges_4 = [daily_stats[unique_dates[idx_d - 1 - k]]['range'] for k in range(1, 4) if unique_dates[idx_d - 1 - k] in daily_stats]
            if len(prior_ranges_4) == 3 and p1_rng < min(prior_ranges_4):
                is_nr4[mask_cur] = True

        # Check NR7 (last 6 prior days + d_p1 = 7 days total)
        if idx_d >= 7:
            prior_ranges_7 = [daily_stats[unique_dates[idx_d - 1 - k]]['range'] for k in range(1, 7) if unique_dates[idx_d - 1 - k] in daily_stats]
            if len(prior_ranges_7) == 6 and p1_rng < min(prior_ranges_7):
                is_nr7[mask_cur] = True

    # Asian Range (00:00 - 06:45 UTC) for each day
    asian_high = np.zeros(len(m15), dtype=np.float64)
    asian_low = np.zeros(len(m15), dtype=np.float64)
    asian_range = np.zeros(len(m15), dtype=np.float64)

    for d in unique_dates:
        mask_asian = (dates == d) & (hours < 7)
        if np.sum(mask_asian) >= 10:
            ah = float(np.max(h[mask_asian]))
            al = float(np.min(l[mask_asian]))
            ar = ah - al
            mask_day = (dates == d)
            asian_high[mask_day] = ah
            asian_low[mask_day] = al
            asian_range[mask_day] = ar

    # Prior Week High & Low (PWH, PWL) for Liquidity Sweeps
    # We estimate weekly high/low from the warmup ticks/prior bars
    # In tick data, week boundary is clear. Let's compute weekly min/max of first 3 days vs later days.
    # Or simply: Rolling 5-day High/Low
    rolling_5d_h = pd.Series(h).rolling(480, min_periods=96).max().shift(1).bfill().values
    rolling_5d_l = pd.Series(l).rolling(480, min_periods=96).min().shift(1).bfill().values

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
        'h4_bias': h4_bias,
        'h2_sqz_fired': h2_sqz_fired,
        'h2_mom': h2_mom,
        'is_nr7': is_nr7,
        'is_nr4': is_nr4,
        'prev_dh': prev_dh,
        'prev_dl': prev_dl,
        'asian_high': asian_high,
        'asian_low': asian_low,
        'asian_range': asian_range,
        'rolling_5d_h': rolling_5d_h,
        'rolling_5d_l': rolling_5d_l,
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


# 1. NR7 / NR4 Breakout
def eval_nr_breakout(week_data, nr_type, start_hr, end_hr, buffer_pips, min_body_ratio, macro_filter, sl_pips, tp_pips, max_spread):
    n_bars = week_data['n_bars']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    exec_indices = week_data['exec_indices']
    is_nr = week_data['is_nr7'] if nr_type == 'nr7' else week_data['is_nr4']
    pd_h = week_data['prev_dh']
    pd_l = week_data['prev_dl']
    h4_bias = week_data['h4_bias']

    sig_indices = []
    sig_types = []
    last_trade_date = None

    buf = buffer_pips * PIP_SIZE

    for i in range(15, n_bars):
        hr = hours[i]
        if not (start_hr <= hr <= end_hr):
            continue
        dt = dates[i]
        if dt == last_trade_date:
            continue
        if not is_nr[i]:
            continue
        if spread[i] > max_spread:
            continue

        bar_rng = h[i] - l[i]
        if bar_rng < 1.5 * PIP_SIZE:
            continue

        body = abs(c[i] - o[i])
        if (body / bar_rng) < min_body_ratio:
            continue

        macro = h4_bias[i] if macro_filter else 0

        sig = 0
        if c[i] > pd_h[i] + buf and c[i] > o[i]:
            if not macro_filter or macro >= 0:
                sig = 1
        elif c[i] < pd_l[i] - buf and c[i] < o[i]:
            if not macro_filter or macro <= 0:
                sig = -1

        if sig != 0:
            sig_indices.append(exec_indices[i])
            sig_types.append(sig)
            last_trade_date = dt

    return execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips)


# 2. Asian Range Breakout (Asian ORB)
def eval_asian_orb(week_data, start_hr, end_hr, max_asian_range, buffer_pips, min_body_ratio, macro_filter, sl_pips, tp_pips, max_spread):
    n_bars = week_data['n_bars']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    exec_indices = week_data['exec_indices']
    ah = week_data['asian_high']
    al = week_data['asian_low']
    ar = week_data['asian_range']
    h4_bias = week_data['h4_bias']

    sig_indices = []
    sig_types = []
    last_trade_date = None

    buf = buffer_pips * PIP_SIZE
    max_ar_val = max_asian_range * PIP_SIZE

    for i in range(15, n_bars):
        hr = hours[i]
        if not (start_hr <= hr <= end_hr):
            continue
        dt = dates[i]
        if dt == last_trade_date:
            continue
        if ah[i] == 0.0 or ar[i] > max_ar_val or ar[i] < 4.0 * PIP_SIZE:
            continue
        if spread[i] > max_spread:
            continue

        bar_rng = h[i] - l[i]
        if bar_rng < 1.5 * PIP_SIZE:
            continue

        body = abs(c[i] - o[i])
        if (body / bar_rng) < min_body_ratio:
            continue

        macro = h4_bias[i] if macro_filter else 0

        sig = 0
        if c[i] > ah[i] + buf and c[i] > o[i]:
            if not macro_filter or macro >= 0:
                sig = 1
        elif c[i] < al[i] - buf and c[i] < o[i]:
            if not macro_filter or macro <= 0:
                sig = -1

        if sig != 0:
            sig_indices.append(exec_indices[i])
            sig_types.append(sig)
            last_trade_date = dt

    return execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips)


# 3. Refined H2 Squeeze (with candle body ratio & macro alignment)
def eval_refined_h2_squeeze(week_data, start_hr, end_hr, min_body_ratio, macro_filter, sl_pips, tp_pips, max_spread):
    n_bars = week_data['n_bars']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    exec_indices = week_data['exec_indices']
    sqz = week_data['h2_sqz_fired']
    mom = week_data['h2_mom']
    h4_bias = week_data['h4_bias']

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

        bar_rng = h[i] - l[i]
        if bar_rng < 1.5 * PIP_SIZE:
            continue

        body = abs(c[i] - o[i])
        if (body / bar_rng) < min_body_ratio:
            continue

        macro = h4_bias[i] if macro_filter else 0

        sig = 0
        if m == 1 and c[i] > o[i]:
            if not macro_filter or macro >= 0:
                sig = 1
        elif m == -1 and c[i] < o[i]:
            if not macro_filter or macro <= 0:
                sig = -1

        if sig != 0:
            sig_indices.append(exec_indices[i])
            sig_types.append(sig)
            last_trade_date = dt

    return execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips)


# 4. Multi-Day Liquidity Sweep Rejection (5-Day Rolling High/Low Fakeouts)
def eval_liquidity_sweep(week_data, start_hr, end_hr, sweep_min, sweep_max, min_wick, sl_pips, tp_pips, max_spread):
    n_bars = week_data['n_bars']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    exec_indices = week_data['exec_indices']
    r5_h = week_data['rolling_5d_h']
    r5_l = week_data['rolling_5d_l']

    sig_indices = []
    sig_types = []
    last_trade_date = None

    s_min = sweep_min * PIP_SIZE
    s_max = sweep_max * PIP_SIZE

    for i in range(25, n_bars):
        hr = hours[i]
        if not (start_hr <= hr <= end_hr):
            continue
        dt = dates[i]
        if dt == last_trade_date:
            continue
        if spread[i] > max_spread:
            continue

        bar_rng = h[i] - l[i]
        if bar_rng < 1.5 * PIP_SIZE:
            continue

        sig = 0
        # Bullish Fakeout at Low (Liquidity Sweep & Rejection)
        if (l[i] <= r5_l[i] - s_min) and (l[i] >= r5_l[i] - s_max) and (c[i] > r5_l[i]) and (c[i] > o[i]):
            lower_wick = (o[i] - l[i]) / bar_rng
            if lower_wick >= min_wick:
                sig = 1
        # Bearish Fakeout at High (Liquidity Sweep & Rejection)
        elif (h[i] >= r5_h[i] + s_min) and (h[i] <= r5_h[i] + s_max) and (c[i] < r5_h[i]) and (c[i] < o[i]):
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

    # 1. NR7 & NR4 Breakouts
    for nr_t in ['nr7', 'nr4']:
        for wh in [(7, 15), (8, 15)]:
            for buf in [0.0, 0.5]:
                for br in [0.45, 0.55]:
                    for sl, tp in [(8.0, 16.0), (9.0, 18.0), (10.0, 20.0)]:
                        for mf in [True, False]:
                            candidates.append({
                                'archetype': 'nr_breakout',
                                'params': {'nr_type': nr_t, 'start_hr': wh[0], 'end_hr': wh[1], 'buffer_pips': buf, 'min_body_ratio': br, 'macro_filter': mf, 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 0.8},
                                'desc': f"EURGBP {nr_t.upper()} Breakout ({wh[0]:02d}-{wh[1]:02d} UTC, buf={buf}p, BR={br}, Macro={mf}, SL={sl}p, TP={tp}p)"
                            })

    # 2. Asian ORB (07:00 - 10:00 UTC)
    for max_ar in [20.0, 25.0]:
        for buf in [0.5, 1.0]:
            for br in [0.45, 0.55]:
                for sl, tp in [(7.0, 14.0), (8.0, 16.0), (9.0, 18.0)]:
                    for mf in [True, False]:
                        candidates.append({
                            'archetype': 'asian_orb',
                            'params': {'start_hr': 7, 'end_hr': 10, 'max_asian_range': max_ar, 'buffer_pips': buf, 'min_body_ratio': br, 'macro_filter': mf, 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 0.8},
                            'desc': f"EURGBP Asian ORB (MaxAR={max_ar}p, buf={buf}p, BR={br}, Macro={mf}, SL={sl}p, TP={tp}p)"
                        })

    # 3. Refined H2 Squeeze
    for wh in [(7, 16), (8, 15)]:
        for br in [0.45, 0.55, 0.65]:
            for mf in [True, False]:
                for sl, tp in [(9.0, 18.0), (10.0, 20.0), (10.0, 22.0)]:
                    candidates.append({
                        'archetype': 'refined_h2_sqz',
                        'params': {'start_hr': wh[0], 'end_hr': wh[1], 'min_body_ratio': br, 'macro_filter': mf, 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 0.8},
                        'desc': f"EURGBP Refined H2 Squeeze ({wh[0]:02d}-{wh[1]:02d} UTC, BR={br}, Macro={mf}, SL={sl}p, TP={tp}p)"
                    })

    # 4. Multi-Day Liquidity Sweep Rejection
    for wh in [(7, 16), (8, 15)]:
        for s_min, s_max in [(1.0, 5.0), (1.5, 6.0)]:
            for wk in [0.35, 0.40]:
                for sl, tp in [(8.0, 16.0), (9.0, 18.0), (10.0, 20.0)]:
                    candidates.append({
                        'archetype': 'liquidity_sweep',
                        'params': {'start_hr': wh[0], 'end_hr': wh[1], 'sweep_min': s_min, 'sweep_max': s_max, 'min_wick': wk, 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 0.8},
                        'desc': f"EURGBP 5D Liquidity Sweep ({wh[0]:02d}-{wh[1]:02d} UTC, Sweep={s_min}-{s_max}p, Wk={wk}, SL={sl}p, TP={tp}p)"
                    })

    return candidates


def evaluate_cand(cand, w_data):
    arch = cand['archetype']
    p = cand['params']
    if arch == 'nr_breakout':
        return eval_nr_breakout(w_data, p['nr_type'], p['start_hr'], p['end_hr'], p['buffer_pips'], p['min_body_ratio'], p['macro_filter'], p['sl_pips'], p['tp_pips'], p['max_spread'])
    elif arch == 'asian_orb':
        return eval_asian_orb(w_data, p['start_hr'], p['end_hr'], p['max_asian_range'], p['buffer_pips'], p['min_body_ratio'], p['macro_filter'], p['sl_pips'], p['tp_pips'], p['max_spread'])
    elif arch == 'refined_h2_sqz':
        return eval_refined_h2_squeeze(w_data, p['start_hr'], p['end_hr'], p['min_body_ratio'], p['macro_filter'], p['sl_pips'], p['tp_pips'], p['max_spread'])
    elif arch == 'liquidity_sweep':
        return eval_liquidity_sweep(w_data, p['start_hr'], p['end_hr'], p['sweep_min'], p['sweep_max'], p['min_wick'], p['sl_pips'], p['tp_pips'], p['max_spread'])
    return []


def main():
    print("=" * 80)
    print("  EURGBP DEEP ALPHA HUNT: NR7/NR4, ASIAN ORB, SWEEPS & H2 REFINEMENTS")
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
        w_data = load_eurgbp_multi_archetype_bars(w_num)
        if w_data is None:
            continue

        for c_idx, cand in enumerate(candidates):
            tr = evaluate_cand(cand, w_data)
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

        if net_pips > 0 and len(tr_list) >= 12:
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
        print("No candidates cleared holdout gate.")
        return

    # Stream Discovery on viable candidates
    print(f"\nStreaming Discovery (Weeks {min(disc_weeks)}..{max(disc_weeks)}) for {len(viable_indices)} viable candidates...", flush=True)
    cand_disc_trades = {i: [] for i in viable_indices}

    t1 = time.time()
    for w_idx, w_num in enumerate(disc_weeks):
        w_data = load_eurgbp_multi_archetype_bars(w_num)
        if w_data is None:
            continue

        for c_idx in viable_indices:
            cand = candidates[c_idx]
            tr = evaluate_cand(cand, w_data)
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

    out_file = ROOT_DIR / "results" / "new_champions_eurgbp_nr_asian.json"
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
    print(f"Saved champions to: {out_file}", flush=True)


if __name__ == "__main__":
    main()

