#!/usr/bin/env python3
"""
================================================================================
  DISCOVERY ENGINE: NOVEL ALPHA ARCHETYPES FOR UNDERREPRESENTED PAIRS
================================================================================
Targets underrepresented pairs with NOVEL institutional archetypes:
  1. EURGBP:
     - Archetype A: Asian Range Boundary Mean Reversion (21:00-05:00 UTC, BB/RSI Reversal)
     - Archetype B: London Lunch Session Fade (11:00-14:00 UTC, HOD/LOD VWAP Reversion)
     - Archetype C: Multi-Timeframe Keltner/Bollinger Volatility Squeeze (M30/H1 alternate)
  2. GBPCHF:
     - Archetype A: London Open Range Breakout (07:00-10:00 UTC, Pre-London ORB)
     - Archetype B: H1 Carter TTM Keltner Squeeze & Expansion
  3. AUDCAD & NZDCAD:
     - Archetype A: H1 Keltner Squeeze Breakout
     - Archetype B: Daily Pivot Point Bounce (Floor Traders S1/R1 Retest)

All tests enforce:
  - Exact subsequent-tick fill contract (T_fill > T_sig)
  - Full ECN friction: $4.00/lot comm + 0.20p slippage + floating Dukascopy spreads
  - Holdout (Weeks 211..300) PF >= 1.20, Trades >= 15, Net Pips > 0, ExTop3 > 0
  - Discovery (Weeks 1..210) PF >= 1.15, Trades >= 30, Net Pips > 0
  - Anti-redundancy: Overlap < 50% & Corr < 0.50 against existing fleet
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

# Load existing fleet trades for redundancy checks
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


def load_week_data(symbol: str, week_num: int):
    """Loads a week of tick data and builds multi-timeframe bar cache."""
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

    # H1 bars
    h1 = m15[['high', 'low', 'close']].resample('1h').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    # H4 bars
    h4 = m15[['high', 'low', 'close']].resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()

    # Precalculate indicators on M15
    c_m15 = m15['close'].values.astype(np.float64)
    h_m15 = m15['high'].values.astype(np.float64)
    l_m15 = m15['low'].values.astype(np.float64)
    o_m15 = m15['open'].values.astype(np.float64)
    spread = m15['spread_pips'].values.astype(np.float64)
    hours = m15.index.hour.values.astype(np.int16)
    dates = m15.index.date
    time_ms = m15.index.astype('datetime64[ms]').view('int64')
    dur_ms = 15 * 60 * 1000
    tick_ts = ticks['timestamp']
    n_ticks = len(ticks)
    exec_indices = np.searchsorted(tick_ts, time_ms + dur_ms).astype(np.int32)

    # RSI(14) on M15
    diff = np.diff(c_m15, prepend=c_m15[0])
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    avg_gain = pd.Series(gain).rolling(14).mean().bfill().values
    avg_loss = pd.Series(loss).rolling(14).mean().bfill().values
    rs = np.divide(avg_gain, np.maximum(avg_loss, 1e-9))
    rsi14 = 100.0 - (100.0 / (1.0 + rs))

    # Bollinger Bands (20, 2.0) on M15
    s_c = pd.Series(c_m15)
    bb_mean = s_c.rolling(20).mean().bfill().values
    bb_std = s_c.rolling(20).std().bfill().values
    bb_up = bb_mean + 2.0 * bb_std
    bb_dn = bb_mean - 2.0 * bb_std

    # ATR(14) on M15
    tr = np.maximum(h_m15 - l_m15, np.maximum(np.abs(h_m15 - np.roll(c_m15, 1)), np.abs(l_m15 - np.roll(c_m15, 1))))
    tr[0] = h_m15[0] - l_m15[0]
    atr14 = pd.Series(tr).rolling(14).mean().bfill().values.astype(np.float64)

    # H1 Squeeze Calculation
    h1_sqz_fired = np.zeros(len(m15), dtype=bool)
    h1_mom = np.zeros(len(m15), dtype=np.int8)
    if len(h1) >= 25:
        h1_c = h1['close'].values
        h1_h = h1['high'].values
        h1_l = h1['low'].values
        h1_cs = pd.Series(h1_c)
        h1_ema20 = h1_cs.ewm(span=20, adjust=False).mean().values
        h1_std20 = h1_cs.rolling(20).std().bfill().values
        h1_bb_up = h1_ema20 + 2.0 * h1_std20
        h1_bb_dn = h1_ema20 - 2.0 * h1_std20

        h1_tr = np.maximum(h1_h - h1_l, np.maximum(np.abs(h1_h - np.roll(h1_c, 1)), np.abs(h1_l - np.roll(h1_c, 1))))
        h1_tr[0] = h1_h[0] - h1_l[0]
        h1_atr20 = pd.Series(h1_tr).rolling(20).mean().bfill().values

        h1_kc_up = h1_ema20 + 1.5 * h1_atr20
        h1_kc_dn = h1_ema20 - 1.5 * h1_atr20

        sqz_on = (h1_bb_dn > h1_kc_dn) & (h1_bb_up < h1_kc_up)
        sqz_f = pd.Series(sqz_on).shift(1).fillna(False).values & (~sqz_on)
        m = np.where(h1_c > h1_ema20, 1, np.where(h1_c < h1_ema20, -1, 0))

        h1_df = pd.DataFrame({'sqz_fire': sqz_f, 'mom': m}, index=h1.index)
        merged = pd.merge_asof(m15.sort_index(), h1_df.sort_index(), left_index=True, right_index=True, direction='backward')
        h1_sqz_fired = merged['sqz_fire'].fillna(False).values
        h1_mom = merged['mom'].fillna(0).values.astype(np.int8)

    # H4 Trend Bias
    h4_bias = np.zeros(len(m15), dtype=np.int8)
    if len(h4) >= 10:
        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_b = np.where(h4['close'] > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'bias': h4_b}, index=h4.index)
        merged_h4 = pd.merge_asof(m15.sort_index(), h4_df.sort_index(), left_index=True, right_index=True, direction='backward')
        h4_bias = merged_h4['bias'].fillna(0).values.astype(np.int8)

    return {
        'ticks': ticks,
        'n_ticks': n_ticks,
        'pip_size': pip_size,
        'min_entry_ts': ticks['timestamp'][WARMUP_TICKS],
        'c': c_m15, 'h': h_m15, 'l': l_m15, 'o': o_m15,
        'spread': spread,
        'hours': hours,
        'dates': dates,
        'exec_indices': exec_indices,
        'atr14': atr14,
        'rsi14': rsi14,
        'bb_up': bb_up, 'bb_dn': bb_dn, 'bb_mean': bb_mean,
        'h1_sqz_fired': h1_sqz_fired,
        'h1_mom': h1_mom,
        'h4_bias': h4_bias,
        'n_bars': len(m15),
        'signals_buf': np.zeros(n_ticks, dtype=np.int8),
        'sl_buf': np.zeros(n_ticks, dtype=np.float64),
        'tp_buf': np.zeros(n_ticks, dtype=np.float64)
    }


def run_signals_and_match(week_data, sig_indices, sig_types, sl_pips, tp_pips, comm, slip):
    """Executes matching loop for given signal locations."""
    if len(sig_indices) == 0:
        return []
    signals_buf = week_data['signals_buf']
    sl_buf = week_data['sl_buf']
    tp_buf = week_data['tp_buf']
    n_ticks = week_data['n_ticks']

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
        pip_size=week_data['pip_size'],
        commission_pips=comm,
        entry_slippage_pips=slip,
        enforce_subsequent_tick=True
    )

    # Clean buffers
    for idx in fired:
        signals_buf[idx] = 0
        sl_buf[idx] = 0.0
        tp_buf[idx] = 0.0

    return trades

def eval_asian_mean_reversion(week_data, rsi_ob, rsi_os, sl_pips, tp_pips, max_spread, comm, slip):
    n_bars = week_data['n_bars']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    rsi14 = week_data['rsi14']
    bb_up, bb_dn = week_data['bb_up'], week_data['bb_dn']
    exec_indices = week_data['exec_indices']

    sig_indices = []
    sig_types = []
    last_trade_date = None

    for i in range(25, n_bars):
        hr = hours[i]
        # Asian night: 21:00 - 05:00 UTC
        if not (hr >= 21 or hr <= 5):
            continue
        dt = dates[i]
        if dt == last_trade_date:
            continue
        if spread[i] > max_spread:
            continue

        bar_rng = h[i] - l[i]
        if bar_rng < 1e-6:
            continue

        sig = 0
        # Long: pierced lower band, oversold RSI, bullish rejection
        if c[i] < bb_dn[i] and rsi14[i] < rsi_os:
            lower_wick = min(c[i], o[i]) - l[i]
            if c[i] > o[i] or (lower_wick / bar_rng) >= 0.30:
                sig = 1
        # Short: pierced upper band, overbought RSI, bearish rejection
        elif c[i] > bb_up[i] and rsi14[i] > rsi_ob:
            upper_wick = h[i] - max(c[i], o[i])
            if c[i] < o[i] or (upper_wick / bar_rng) >= 0.30:
                sig = -1

        if sig != 0:
            sig_indices.append(exec_indices[i])
            sig_types.append(sig)
            last_trade_date = dt

    return run_signals_and_match(week_data, sig_indices, sig_types, sl_pips, tp_pips, comm, slip)


def eval_london_lunch_fade(week_data, sl_pips, tp_pips, max_spread, comm, slip):
    n_bars = week_data['n_bars']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    pip_size = week_data['pip_size']
    exec_indices = week_data['exec_indices']

    sig_indices = []
    sig_types = []
    last_trade_date = None

    # Track morning high and low per day
    curr_date = None
    m_high = -1.0
    m_low = 1e9

    for i in range(15, n_bars):
        dt = dates[i]
        hr = hours[i]
        if dt != curr_date:
            curr_date = dt
            m_high = -1.0
            m_low = 1e9

        # Accumulate London morning range (07:00 - 10:45 UTC)
        if 7 <= hr <= 10:
            m_high = max(m_high, h[i])
            m_low = min(m_low, l[i])
            continue

        # Lunch fade window: 11:00 - 14:00 UTC
        if not (11 <= hr <= 14):
            continue
        if dt == last_trade_date or m_high < 0:
            continue
        if spread[i] > max_spread:
            continue

        bar_rng = h[i] - l[i]
        if bar_rng < 1e-6:
            continue

        sig = 0
        dist_high = abs(h[i] - m_high) / pip_size
        dist_low = abs(l[i] - m_low) / pip_size

        # Short fade near morning high with rejection
        if dist_high <= 3.0:
            upper_wick = h[i] - max(c[i], o[i])
            if c[i] < o[i] and (upper_wick / bar_rng) >= 0.30:
                sig = -1
        # Long fade near morning low with rejection
        elif dist_low <= 3.0:
            lower_wick = min(c[i], o[i]) - l[i]
            if c[i] > o[i] and (lower_wick / bar_rng) >= 0.30:
                sig = 1

        if sig != 0:
            sig_indices.append(exec_indices[i])
            sig_types.append(sig)
            last_trade_date = dt

    return run_signals_and_match(week_data, sig_indices, sig_types, sl_pips, tp_pips, comm, slip)


def eval_keltner_squeeze(week_data, start_hr, end_hr, sl_pips, tp_pips, max_spread, comm, slip):
    n_bars = week_data['n_bars']
    c, o = week_data['c'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    h1_sqz = week_data['h1_sqz_fired']
    h1_mom = week_data['h1_mom']
    exec_indices = week_data['exec_indices']

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
        if not h1_sqz[i]:
            continue

        mom = h1_mom[i]
        if mom == 0:
            continue

        sig = 0
        if mom == 1 and c[i] > o[i]:
            sig = 1
        elif mom == -1 and c[i] < o[i]:
            sig = -1

        if sig != 0:
            sig_indices.append(exec_indices[i])
            sig_types.append(sig)
            last_trade_date = dt

    return run_signals_and_match(week_data, sig_indices, sig_types, sl_pips, tp_pips, comm, slip)


def eval_premarket_breakout(week_data, sl_pips, tp_pips, max_spread, comm, slip):
    n_bars = week_data['n_bars']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    h4_bias = week_data['h4_bias']
    exec_indices = week_data['exec_indices']

    sig_indices = []
    sig_types = []
    last_trade_date = None

    curr_date = None
    pre_h = -1.0
    pre_l = 1e9

    for i in range(15, n_bars):
        dt = dates[i]
        hr = hours[i]
        if dt != curr_date:
            curr_date = dt
            pre_h = -1.0
            pre_l = 1e9

        # Pre-London range: 05:00 - 06:45 UTC
        if 5 <= hr <= 6:
            pre_h = max(pre_h, h[i])
            pre_l = min(pre_l, l[i])
            continue

        # Breakout window: 07:00 - 09:30 UTC
        if not (7 <= hr <= 9):
            continue
        if dt == last_trade_date or pre_h < 0:
            continue
        if spread[i] > max_spread:
            continue

        bar_rng = h[i] - l[i]
        if bar_rng < 1e-6:
            continue

        bias = h4_bias[i]
        sig = 0
        # Long breakout: close above pre-market high with bullish body aligned with H4 trend
        if c[i] > pre_h and (c[i] - o[i]) > 0.45 * bar_rng and bias >= 0:
            sig = 1
        # Short breakout: close below pre-market low with bearish body aligned with H4 trend
        elif c[i] < pre_l and (o[i] - c[i]) > 0.45 * bar_rng and bias <= 0:
            sig = -1

        if sig != 0:
            sig_indices.append(exec_indices[i])
            sig_types.append(sig)
            last_trade_date = dt

    return run_signals_and_match(week_data, sig_indices, sig_types, sl_pips, tp_pips, comm, slip)

def generate_candidates():
    candidates = []

    # --- EURGBP ---
    # 1. Asian Mean Reversion
    for rsi_ob, rsi_os in [(65, 35), (70, 30)]:
        for sl, tp in [(7.0, 8.0), (7.0, 10.0), (8.0, 12.0), (9.0, 14.0), (10.0, 15.0)]:
            candidates.append({
                'symbol': 'EURGBP',
                'archetype': 'asian_mean_rev',
                'params': {'rsi_ob': rsi_ob, 'rsi_os': rsi_os, 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 0.8},
                'desc': f"EURGBP Asian Mean Rev (OB={rsi_ob}/OS={rsi_os}, SL={sl}p, TP={tp}p)"
            })

    # 2. London Lunch Fade
    for sl, tp in [(7.0, 12.0), (8.0, 14.0), (9.0, 16.0), (8.0, 18.0)]:
        candidates.append({
            'symbol': 'EURGBP',
            'archetype': 'london_lunch_fade',
            'params': {'sl_pips': sl, 'tp_pips': tp, 'max_spread': 0.8},
            'desc': f"EURGBP London Lunch Fade (SL={sl}p, TP={tp}p)"
        })

    # 3. Keltner Squeeze (Alternate sessions / targets)
    for sh, eh in [(6, 14), (8, 17), (11, 18)]:
        for sl, tp in [(7.0, 14.0), (8.0, 18.0), (9.0, 20.0)]:
            candidates.append({
                'symbol': 'EURGBP',
                'archetype': 'keltner_squeeze',
                'params': {'start_hr': sh, 'end_hr': eh, 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 0.8},
                'desc': f"EURGBP H1 Squeeze ({sh:02d}-{eh:02d} UTC, SL={sl}p, TP={tp}p)"
            })

    # --- GBPCHF ---
    # 1. London Pre-Market Breakout (ORB)
    for sl, tp in [(18.0, 36.0), (20.0, 45.0), (22.0, 50.0), (25.0, 60.0)]:
        candidates.append({
            'symbol': 'GBPCHF',
            'archetype': 'premarket_breakout',
            'params': {'sl_pips': sl, 'tp_pips': tp, 'max_spread': 3.0},
            'desc': f"GBPCHF London ORB (SL={sl}p, TP={tp}p)"
        })

    # 2. H1 Keltner Squeeze
    for sh, eh in [(7, 16), (6, 14), (8, 17)]:
        for sl, tp in [(18.0, 40.0), (22.0, 50.0), (25.0, 60.0)]:
            candidates.append({
                'symbol': 'GBPCHF',
                'archetype': 'keltner_squeeze',
                'params': {'start_hr': sh, 'end_hr': eh, 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 3.0},
                'desc': f"GBPCHF H1 Squeeze ({sh:02d}-{eh:02d} UTC, SL={sl}p, TP={tp}p)"
            })

    # --- AUDCAD ---
    # H1 Keltner Squeeze
    for sh, eh in [(0, 16), (12, 20), (0, 23)]:
        for sl, tp in [(14.0, 28.0), (16.0, 36.0), (18.0, 40.0), (20.0, 45.0)]:
            candidates.append({
                'symbol': 'AUDCAD',
                'archetype': 'keltner_squeeze',
                'params': {'start_hr': sh, 'end_hr': eh, 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 2.2},
                'desc': f"AUDCAD H1 Squeeze ({sh:02d}-{eh:02d} UTC, SL={sl}p, TP={tp}p)"
            })

    # --- NZDCAD ---
    # H1 Keltner Squeeze
    for sh, eh in [(21, 15), (12, 20), (0, 23)]:
        for sl, tp in [(14.0, 28.0), (16.0, 36.0), (18.0, 40.0), (20.0, 45.0)]:
            candidates.append({
                'symbol': 'NZDCAD',
                'archetype': 'keltner_squeeze',
                'params': {'start_hr': sh, 'end_hr': eh, 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 2.5},
                'desc': f"NZDCAD H1 Squeeze ({sh:02d}-{eh:02d} UTC, SL={sl}p, TP={tp}p)"
            })

    return candidates


def evaluate_candidate_week(cand, week_data, comm, slip):
    arch = cand['archetype']
    p = cand['params']
    if arch == 'asian_mean_rev':
        return eval_asian_mean_reversion(week_data, p['rsi_ob'], p['rsi_os'], p['sl_pips'], p['tp_pips'], p['max_spread'], comm, slip)
    elif arch == 'london_lunch_fade':
        return eval_london_lunch_fade(week_data, p['sl_pips'], p['tp_pips'], p['max_spread'], comm, slip)
    elif arch == 'keltner_squeeze':
        return eval_keltner_squeeze(week_data, p['start_hr'], p['end_hr'], p['sl_pips'], p['tp_pips'], p['max_spread'], comm, slip)
    elif arch == 'premarket_breakout':
        return eval_premarket_breakout(week_data, p['sl_pips'], p['tp_pips'], p['max_spread'], comm, slip)
    return []


def main():
    print("=" * 80)
    print("  NOVEL ARCHETYPE ALPHA DISCOVERY: EURGBP, GBPCHF, AUDCAD, NZDCAD")
    print("=" * 80)

    candidates = generate_candidates()
    print(f"Generated {len(candidates)} total novel candidate setups across 4 underrepresented pairs.")

    by_symbol = defaultdict(list)
    for c in candidates:
        by_symbol[c['symbol']].append(c)

    all_champions = []

    for sym in ['GBPCHF', 'AUDCAD', 'NZDCAD']:
        sym_cands = by_symbol[sym]
        print(f"\n>>> Processing Symbol: {sym} ({len(sym_cands)} candidates) <<<")

        pip_size = PIP_SIZES.get(sym, 0.0001)
        comm, slip = get_friction_pips(sym)
        raw_dir = ROOT_DIR / "data" / sym
        disc_weeks = get_complete_split_weeks(raw_dir, split="discovery")
        hold_weeks = get_complete_split_weeks(raw_dir, split="holdout")

        print(f"[{sym}] Quarantined Holdout Weeks: {len(hold_weeks)} (Weeks {min(hold_weeks)}..{max(hold_weeks)})")
        print(f"[{sym}] Streaming holdout tick data...")

        # Accumulate holdout trades per candidate
        cand_holdout_trades = {i: [] for i in range(len(sym_cands))}

        t0 = time.time()
        for w_idx, w_num in enumerate(hold_weeks):
            w_data = load_week_data(sym, w_num)
            if w_data is None:
                continue

            for c_idx, cand in enumerate(sym_cands):
                tr = evaluate_candidate_week(cand, w_data, comm, slip)
                if len(tr) > 0:
                    cand_holdout_trades[c_idx].extend(tr)

            del w_data
            if (w_idx + 1) % 20 == 0 or (w_idx + 1) == len(hold_weeks):
                print(f"  Holdout week {w_idx+1}/{len(hold_weeks)} processed in {time.time()-t0:.1f}s...")

        # Analyze holdout performance
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
                # Anti-redundancy check
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
                    print(f"  [HOLDOUT PASS] {cand['desc']}: {len(tr_arr)} tr, +{net_pips:.1f}p, PF {pf:.2f}, Exp {np.mean(pips):.2f}p, ExTop3 +{pips_ex_top3:.1f}p")
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
            w_data = load_week_data(sym, w_num)
            if w_data is None:
                continue

            for c_idx in viable_indices:
                cand = sym_cands[c_idx]
                tr = evaluate_candidate_week(cand, w_data, comm, slip)
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
                print(f"  *** DUAL CERTIFIED CHAMPION ***: {cand['desc']}")
                print(f"      Holdout:   {cand['holdout_metrics']['trades']} tr, +{cand['holdout_metrics']['net_pips']:.1f}p, PF {cand['holdout_metrics']['profit_factor']:.2f}, Exp {cand['holdout_metrics']['expectancy']:.2f}p")
                print(f"      Discovery: {len(d_arr)} tr, +{d_net:.1f}p, PF {d_pf:.2f}, Exp {np.mean(d_pips):.2f}p")

    print("\n" + "=" * 80)
    print(f"DISCOVERY COMPLETE: Found {len(all_champions)} New Institutional Champions!")
    print("=" * 80)

    # Save champions
    out_file = ROOT_DIR / "results" / "new_champions_novel_crosses.json"
    serializable = []
    for c in all_champions:
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
    print(f"Saved champions to: {out_file}")


if __name__ == "__main__":
    main()
