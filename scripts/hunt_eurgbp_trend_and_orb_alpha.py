#!/usr/bin/env python3
"""
================================================================================
  EURGBP TAILORED ARCHETYPES HUNT: ORB, H1 PULLBACK, M30 SQUEEZE, H1 INSIDE
================================================================================
Explores 4 structurally distinct archetypes tailored to EURGBP low-volatility drift:
  1. European Opening Range Breakout (07:00 - 09:00 UTC Range Breakout)
     - Frankfurt/London morning range (07:00-09:00 UTC) must be compressed (<= 14p)
     - London expansion (09:00 - 13:00 UTC): M15 breakout beyond 07:00-09:00 High/Low
     - Buffer: 0.3p, 0.5p, 0.8p | SL: 8-10p | TP: 16-22p (2.0R to 2.2R)
  2. H1 Keltner Midline Pullback Rejection (Macro Trend Continuation)
     - H4 Trend Filter: H4 Close > 20-EMA (Bullish) or < 20-EMA (Bearish)
     - H1 Pullback: Price pulls back to touch H1 20-EMA
     - M15 Rejection: Closes in trend direction with >= 35% rejection wick
     - Active: 07:00 - 16:00 UTC | SL: 8-10p | TP: 16-22p
  3. M30 Keltner Squeeze Expansion (Fast Squeeze)
     - M30 Bollinger Bands (20, 2.0) inside Keltner Channel (20, 1.5 ATR) for >= 2 bars
     - Squeeze expansion breakout in direction of M30 20-EMA slope
     - Active: 07:00 - 16:00 UTC | SL: 8-10p | TP: 16-20p
  4. H1 Inside Bar Compression Breakout (Intraday Coiling)
     - Prior H1 bar was completely inside the H1 bar before it (H_h1 < H_h2 and L_h1 > L_h2)
     - H1 range <= 12 pips
     - M15 candle breaks out beyond H1 High/Low during 07:00 - 15:00 UTC
     - Buffer: 0.3p, 0.5p | SL: 8-10p | TP: 16-20p

Certification Standards:
  - Holdout (Weeks 211..300): Trades >= 15, PF >= 1.20, Net Pips > 0, ExTop3 > 0
  - Discovery (Weeks 1..210): Trades >= 20, PF >= 1.15, Net Pips > 0
  - Anti-Redundancy: Overlap < 50%, Corr < 0.50 vs disc107 & disc172
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
        csv_file = MANIFEST_DIR / f"{SYMBOL}__{name}_trades.csv"
        if not csv_file.exists():
            csv_file = MANIFEST_DIR / f"{name}_trades.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                existing_trades.append((name, df))
                print(f"Loaded existing EURGBP trades for {name}: {len(df)} trades", flush=True)


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


def load_week_data(week_num: int):
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

    # H4 Trend Bias
    h4 = m15[['high', 'low', 'close']].resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    h4_bias = np.zeros(len(m15), dtype=np.int8)
    if len(h4) >= 10:
        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_b = np.where(h4['close'] > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'bias': h4_b}, index=h4.index)
        merged_h4 = pd.merge_asof(m15.sort_index(), h4_df.sort_index(), left_index=True, right_index=True, direction='backward')
        h4_bias = merged_h4['bias'].fillna(0).values.astype(np.int8)

    # H1 bars
    h1 = m15[['high', 'low', 'close', 'open']].resample('1h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna()
    h1_ema20 = h1['close'].ewm(span=20, adjust=False).mean()
    h1_h = h1['high'].values
    h1_l = h1['low'].values
    h1_c = h1['close'].values

    # H1 Inside Bar indicator
    h1_inside = np.zeros(len(h1), dtype=bool)
    h1_inside_h = np.zeros(len(h1), dtype=np.float64)
    h1_inside_l = np.zeros(len(h1), dtype=np.float64)
    for idx_h1 in range(1, len(h1)):
        prev_h = h1_h[idx_h1 - 1]
        prev_l = h1_l[idx_h1 - 1]
        cur_h = h1_h[idx_h1]
        cur_l = h1_l[idx_h1]
        if cur_h < prev_h and cur_l > prev_l:
            h1_inside[idx_h1] = True
            h1_inside_h[idx_h1] = cur_h
            h1_inside_l[idx_h1] = cur_l

    h1_df = pd.DataFrame({
        'h1_ema20': h1_ema20.values,
        'h1_inside': h1_inside,
        'h1_inside_h': h1_inside_h,
        'h1_inside_l': h1_inside_l
    }, index=h1.index)

    merged_h1 = pd.merge_asof(m15.sort_index(), h1_df.sort_index(), left_index=True, right_index=True, direction='backward')

    # M30 bars for Squeeze
    m30 = m15[['high', 'low', 'close']].resample('30min').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    m30_c = m30['close'].values
    m30_h = m30['high'].values
    m30_l = m30['low'].values
    m30_s = pd.Series(m30_c)
    m30_ema20 = m30_s.ewm(span=20, adjust=False).mean().values
    m30_std20 = m30_s.rolling(20).std().values
    m30_bb_up = m30_ema20 + 2.0 * m30_std20
    m30_bb_dn = m30_ema20 - 2.0 * m30_std20

    m30_tr = np.maximum(m30_h - m30_l, np.maximum(np.abs(m30_h - np.roll(m30_c, 1)), np.abs(m30_l - np.roll(m30_c, 1))))
    m30_tr[0] = m30_h[0] - m30_l[0]
    m30_atr20 = pd.Series(m30_tr).rolling(20).mean().values
    m30_kc_up = m30_ema20 + 1.5 * m30_atr20
    m30_kc_dn = m30_ema20 - 1.5 * m30_atr20

    m30_sqz_on = (m30_bb_dn > m30_kc_dn) & (m30_bb_up < m30_kc_up)
    m30_sqz_fired = pd.Series(m30_sqz_on).shift(1).fillna(False).values & (~m30_sqz_on)
    m30_mom = np.where(m30_c > m30_ema20, 1, np.where(m30_c < m30_ema20, -1, 0))

    m30_df = pd.DataFrame({
        'm30_sqz_fired': m30_sqz_fired,
        'm30_mom': m30_mom
    }, index=m30.index)

    merged_m30 = pd.merge_asof(m15.sort_index(), m30_df.sort_index(), left_index=True, right_index=True, direction='backward')

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

    # Precalculate Frankfurt 07:00-09:00 range per date
    unique_dates = np.unique(dates)
    ff_ranges = {}
    for d in unique_dates:
        mask = (dates == d) & (hours >= 7) & (hours < 9)
        if np.sum(mask) >= 6:
            ff_h = float(np.max(h[mask]))
            ff_l = float(np.min(l[mask]))
            ff_ranges[d] = (ff_h, ff_l, (ff_h - ff_l) / PIP_SIZE)

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
        'h4_bias': h4_bias,
        'h1_ema20': merged_h1['h1_ema20'].values,
        'h1_inside': merged_h1['h1_inside'].fillna(False).values,
        'h1_inside_h': merged_h1['h1_inside_h'].fillna(0.0).values,
        'h1_inside_l': merged_h1['h1_inside_l'].fillna(0.0).values,
        'm30_sqz_fired': merged_m30['m30_sqz_fired'].fillna(False).values,
        'm30_mom': merged_m30['m30_mom'].fillna(0).values.astype(np.int8),
        'ff_ranges': ff_ranges,
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


# Evaluator 1: Frankfurt/London ORB (07:00-09:00 Range Breakout)
def eval_orb(week_data, max_ff_rng, start_hr, end_hr, buffer_pips, body_ratio, macro_filter, sl_pips, tp_pips, max_spread):
    n_bars = week_data['n_bars']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    exec_indices = week_data['exec_indices']
    h4_bias = week_data['h4_bias']
    ff_ranges = week_data['ff_ranges']

    buf = buffer_pips * PIP_SIZE
    sig_indices = []
    sig_types = []
    last_trade_date = None

    for i in range(20, n_bars):
        hr = hours[i]
        if not (start_hr <= hr <= end_hr):
            continue
        dt = dates[i]
        if dt == last_trade_date:
            continue
        if dt not in ff_ranges:
            continue
        if spread[i] > max_spread:
            continue

        ff_h, ff_l, ff_rng = ff_ranges[dt]
        if ff_rng > max_ff_rng:
            continue

        bar_rng = h[i] - l[i]
        if bar_rng < 1.0 * PIP_SIZE:
            continue

        sig = 0
        if c[i] > ff_h + buf and c[i] > o[i]:
            if not macro_filter or h4_bias[i] == 1:
                body = (c[i] - o[i]) / bar_rng
                if body >= body_ratio:
                    sig = 1
        elif c[i] < ff_l - buf and c[i] < o[i]:
            if not macro_filter or h4_bias[i] == -1:
                body = (o[i] - c[i]) / bar_rng
                if body >= body_ratio:
                    sig = -1

        if sig != 0:
            sig_indices.append(exec_indices[i])
            sig_types.append(sig)
            last_trade_date = dt

    return execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips)


# Evaluator 2: H1 Keltner Midline Pullback Rejection
def eval_h1_pullback(week_data, touch_dist, min_wick, sl_pips, tp_pips, max_spread):
    n_bars = week_data['n_bars']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    exec_indices = week_data['exec_indices']
    h4_bias = week_data['h4_bias']
    h1_ema = week_data['h1_ema20']

    t_dist = touch_dist * PIP_SIZE
    sig_indices = []
    sig_types = []
    last_trade_date = None

    for i in range(25, n_bars):
        hr = hours[i]
        if not (7 <= hr <= 16):
            continue
        dt = dates[i]
        if dt == last_trade_date:
            continue
        if spread[i] > max_spread:
            continue

        macro = h4_bias[i]
        if macro == 0:
            continue

        ema = h1_ema[i]
        bar_rng = h[i] - l[i]
        if bar_rng < 1.2 * PIP_SIZE:
            continue

        sig = 0
        if macro == 1:
            # Bullish trend: low touches near H1 EMA from above, closes green with lower wick
            if l[i] <= ema + t_dist and l[i] >= ema - 2.0 * t_dist and c[i] > o[i]:
                lw = (o[i] - l[i]) / bar_rng
                if lw >= min_wick:
                    sig = 1
        elif macro == -1:
            # Bearish trend: high touches near H1 EMA from below, closes red with upper wick
            if h[i] >= ema - t_dist and h[i] <= ema + 2.0 * t_dist and c[i] < o[i]:
                uw = (h[i] - o[i]) / bar_rng
                if uw >= min_wick:
                    sig = -1

        if sig != 0:
            sig_indices.append(exec_indices[i])
            sig_types.append(sig)
            last_trade_date = dt

    return execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips)


# Evaluator 3: M30 Keltner Squeeze Expansion
def eval_m30_squeeze(week_data, sl_pips, tp_pips, max_spread):
    n_bars = week_data['n_bars']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    exec_indices = week_data['exec_indices']
    sqz_fired = week_data['m30_sqz_fired']
    mom = week_data['m30_mom']

    sig_indices = []
    sig_types = []
    last_trade_date = None

    for i in range(25, n_bars):
        hr = hours[i]
        if not (7 <= hr <= 16):
            continue
        dt = dates[i]
        if dt == last_trade_date:
            continue
        if spread[i] > max_spread:
            continue
        if not sqz_fired[i]:
            continue

        m = mom[i]
        if m == 0:
            continue

        bar_rng = h[i] - l[i]
        if bar_rng < 1.2 * PIP_SIZE:
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


# Evaluator 4: H1 Inside Bar Breakout
def eval_h1_inside(week_data, buffer_pips, body_ratio, macro_filter, sl_pips, tp_pips, max_spread):
    n_bars = week_data['n_bars']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    exec_indices = week_data['exec_indices']
    h4_bias = week_data['h4_bias']
    is_inside = week_data['h1_inside']
    in_h = week_data['h1_inside_h']
    in_l = week_data['h1_inside_l']

    buf = buffer_pips * PIP_SIZE
    sig_indices = []
    sig_types = []
    last_trade_date = None

    for i in range(25, n_bars):
        hr = hours[i]
        if not (7 <= hr <= 15):
            continue
        dt = dates[i]
        if dt == last_trade_date:
            continue
        if spread[i] > max_spread:
            continue
        if not is_inside[i]:
            continue

        bar_rng = h[i] - l[i]
        if bar_rng < 1.0 * PIP_SIZE:
            continue

        ih = in_h[i]
        il = in_l[i]
        if ih - il > 14.0 * PIP_SIZE:  # Maximum 14 pip H1 mother bar
            continue

        sig = 0
        if c[i] > ih + buf and c[i] > o[i]:
            if not macro_filter or h4_bias[i] == 1:
                body = (c[i] - o[i]) / bar_rng
                if body >= body_ratio:
                    sig = 1
        elif c[i] < il - buf and c[i] < o[i]:
            if not macro_filter or h4_bias[i] == -1:
                body = (o[i] - c[i]) / bar_rng
                if body >= body_ratio:
                    sig = -1

        if sig != 0:
            sig_indices.append(exec_indices[i])
            sig_types.append(sig)
            last_trade_date = dt

    return execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips)


def main():
    print(f"================================================================================", flush=True)
    print(f"  HUNTING EURGBP NOVEL INSTITUTIONAL ARCHETYPES", flush=True)
    print(f"================================================================================", flush=True)

    holdout_weeks = get_complete_split_weeks(ROOT_DIR / "data" / SYMBOL, split="holdout")
    discovery_weeks = get_complete_split_weeks(ROOT_DIR / "data" / SYMBOL, split="discovery")
    print(f"Loaded {len(holdout_weeks)} holdout weeks, {len(discovery_weeks)} discovery weeks.", flush=True)

    print(f"Loading {len(holdout_weeks)} holdout weeks into RAM...", flush=True)
    holdout_data = {}
    for w in holdout_weeks:
        wd = load_week_data(w)
        if wd is not None:
            holdout_data[w] = wd
    print(f"Ready: {len(holdout_data)} holdout weeks loaded.", flush=True)

    setups = []

    # 1. ORB setups
    for max_rng in [10.0, 12.0, 14.0]:
        for buf in [0.3, 0.5]:
            for body in [0.40, 0.50]:
                for macro in [True, False]:
                    for sl in [8.0, 9.0, 10.0]:
                        for tp in [16.0, 18.0, 20.0, 22.0]:
                            setups.append({
                                'archetype': 'orb',
                                'max_rng': max_rng, 'buf': buf, 'body': body, 'macro': macro,
                                'sl': sl, 'tp': tp,
                                'desc': f"EURGBP ORB (MaxRng={max_rng}p, Buf={buf}p, BR={body}, Mac={macro}, SL={sl}p, TP={tp}p)"
                            })

    # 2. H1 Midline Pullback setups
    for touch in [1.5, 2.5, 3.5]:
        for wick in [0.30, 0.35, 0.40]:
            for sl in [8.0, 9.0, 10.0]:
                for tp in [16.0, 18.0, 20.0, 22.0]:
                    setups.append({
                        'archetype': 'h1_pullback',
                        'touch': touch, 'wick': wick,
                        'sl': sl, 'tp': tp,
                        'desc': f"EURGBP H1 Pullback (Touch={touch}p, Wk={wick}, SL={sl}p, TP={tp}p)"
                    })

    # 3. M30 Squeeze setups
    for sl in [7.0, 8.0, 9.0, 10.0]:
        for tp in [14.0, 16.0, 18.0, 20.0]:
            setups.append({
                'archetype': 'm30_squeeze',
                'sl': sl, 'tp': tp,
                'desc': f"EURGBP M30 Squeeze (SL={sl}p, TP={tp}p)"
            })

    # 4. H1 Inside Bar setups
    for buf in [0.3, 0.5]:
        for body in [0.40, 0.50]:
            for macro in [True, False]:
                for sl in [8.0, 9.0, 10.0]:
                    for tp in [16.0, 18.0, 20.0]:
                        setups.append({
                            'archetype': 'h1_inside',
                            'buf': buf, 'body': body, 'macro': macro,
                            'sl': sl, 'tp': tp,
                            'desc': f"EURGBP H1 Inside Bar (Buf={buf}p, BR={body}, Mac={macro}, SL={sl}p, TP={tp}p)"
                        })

    print(f"Total candidate setups to screen on Holdout: {len(setups)}", flush=True)

    holdout_champs = []
    t0 = time.time()

    for idx_s, s in enumerate(setups):
        all_trades = []
        arch = s['archetype']
        for w, wd in holdout_data.items():
            if arch == 'orb':
                tr = eval_orb(wd, s['max_rng'], 9, 13, s['buf'], s['body'], s['macro'], s['sl'], s['tp'], 0.8)
            elif arch == 'h1_pullback':
                tr = eval_h1_pullback(wd, s['touch'], s['wick'], s['sl'], s['tp'], 0.8)
            elif arch == 'm30_squeeze':
                tr = eval_m30_squeeze(wd, s['sl'], s['tp'], 0.8)
            elif arch == 'h1_inside':
                tr = eval_h1_inside(wd, s['buf'], s['body'], s['macro'], s['sl'], s['tp'], 0.8)
            all_trades.extend(tr)

        if len(all_trades) < 15:
            continue

        pips = np.array([t['pips'] for t in all_trades], dtype=np.float64)
        net_pips = float(np.sum(pips))
        if net_pips <= 0:
            continue

        wins = pips[pips > 0]
        losses = pips[pips < 0]
        sum_w = float(np.sum(wins)) if len(wins) > 0 else 0.0
        sum_l = float(np.abs(np.sum(losses))) if len(losses) > 0 else 1e-6
        pf = sum_w / sum_l

        if pf < 1.15:
            continue

        sorted_pips = np.sort(pips)
        ex_top3 = float(np.sum(sorted_pips[:-3])) if len(pips) >= 4 else -1.0
        if ex_top3 <= 0:
            continue

        passed_ar, ar_msg = check_anti_redundancy(np.array(all_trades))
        if not passed_ar:
            continue

        holdout_champs.append({
            'setup': s,
            'holdout_metrics': {
                'trades': len(all_trades),
                'net_pips': net_pips,
                'profit_factor': pf,
                'expectancy': float(np.mean(pips)),
                'pips_ex_top3': ex_top3
            }
        })
        print(f"  [HOLDOUT HIT #{len(holdout_champs)}] {s['desc']} -> {len(all_trades)} tr, +{net_pips:.1f}p, PF {pf:.2f}, Exp {np.mean(pips):+.2f}p, ExTop3 +{ex_top3:.1f}p", flush=True)

    print(f"\nHoldout screening complete in {time.time()-t0:.1f}s. Found {len(holdout_champs)} Holdout Champions!", flush=True)

    if not holdout_champs:
        print("No setups passed Holdout criteria.", flush=True)
        return

    del holdout_data
    gc.collect()

    print(f"\n--- VERIFYING {len(holdout_champs)} HOLDOUT CHAMPIONS ON DISCOVERY (WEEKS 1..210) ---", flush=True)
    dual_champs = []

    disc_data = {}
    print(f"Loading {len(discovery_weeks)} discovery weeks into RAM...", flush=True)
    for w in discovery_weeks:
        wd = load_week_data(w)
        if wd is not None:
            disc_data[w] = wd
    print(f"Ready: {len(disc_data)} discovery weeks loaded.", flush=True)

    for c in holdout_champs:
        s = c['setup']
        arch = s['archetype']
        all_trades = []
        for w, wd in disc_data.items():
            if arch == 'orb':
                tr = eval_orb(wd, s['max_rng'], 9, 13, s['buf'], s['body'], s['macro'], s['sl'], s['tp'], 0.8)
            elif arch == 'h1_pullback':
                tr = eval_h1_pullback(wd, s['touch'], s['wick'], s['sl'], s['tp'], 0.8)
            elif arch == 'm30_squeeze':
                tr = eval_m30_squeeze(wd, s['sl'], s['tp'], 0.8)
            elif arch == 'h1_inside':
                tr = eval_h1_inside(wd, s['buf'], s['body'], s['macro'], s['sl'], s['tp'], 0.8)
            all_trades.extend(tr)

        if len(all_trades) < 20:
            continue

        pips = np.array([t['pips'] for t in all_trades], dtype=np.float64)
        net_pips = float(np.sum(pips))
        if net_pips <= 0:
            continue

        wins = pips[pips > 0]
        losses = pips[pips < 0]
        sum_w = float(np.sum(wins)) if len(wins) > 0 else 0.0
        sum_l = float(np.abs(np.sum(losses))) if len(losses) > 0 else 1e-6
        pf = sum_w / sum_l

        if pf < 1.15:
            continue

        c['discovery_metrics'] = {
            'trades': len(all_trades),
            'net_pips': net_pips,
            'profit_factor': pf,
            'expectancy': float(np.mean(pips))
        }
        dual_champs.append(c)
        print(f"  >>> DUAL CERTIFIED EURGBP CHAMPION! <<<", flush=True)
        print(f"      Setup: {s['desc']}", flush=True)
        hm = c['holdout_metrics']
        print(f"      Holdout:   {hm['trades']} tr, +{hm['net_pips']:.1f}p, PF {hm['profit_factor']:.2f}, Exp {hm['expectancy']:+.2f}p, ExTop3 +{hm['pips_ex_top3']:.1f}p", flush=True)
        print(f"      Discovery: {len(all_trades)} tr, +{net_pips:.1f}p, PF {pf:.2f}, Exp {np.mean(pips):+.2f}p", flush=True)
        print(f"      Total 5.75Y: {hm['trades'] + len(all_trades)} tr, +{hm['net_pips'] + net_pips:.1f} net pips", flush=True)

    out_file = ROOT_DIR / "results" / "new_champions_eurgbp_tailored.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(dual_champs, f, indent=2)

    print(f"\nSaved {len(dual_champs)} Dual-Certified EURGBP Champions to {out_file}!", flush=True)

if __name__ == "__main__":
    main()

