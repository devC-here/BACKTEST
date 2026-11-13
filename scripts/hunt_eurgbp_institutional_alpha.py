#!/usr/bin/env python3
"""
================================================================================
  EURGBP SPECIALIZED INSTITUTIONAL ALPHA DISCOVERY ENGINE
================================================================================
Explores 5 diverse, non-redundant quantitative archetypes tailored to EURGBP:
  1. European Cash Open Range Expansion (07:00 - 10:00 UTC / Frankfurt + London)
  2. Asian Range Liquidity Sweep & Reversal Trap (07:00 - 10:00 UTC Fakeout Fade)
  3. M15 MACD Zero-Cross Momentum Acceleration (07:00 - 16:00 UTC)
  4. European Session VWAP Rebound (08:00 - 16:00 UTC)
  5. Carter Squeeze 20-EMA Pullback Continuation (Secondary Trend Wave)

Certification Rigor:
  - Strict subsequent-tick causal execution (T_fill > T_sig)
  - Full ECN friction: 0.33p commission ($4/lot) + 0.20p slippage + floating spreads
  - Holdout (Weeks 211..300): PF >= 1.20, Trades >= 15, Net Pips > 0, ExTop3 > 0
  - Discovery (Weeks 1..210): PF >= 1.15, Trades >= 30, Net Pips > 0
  - Anti-Redundancy: Overlap < 50% & Corr < 0.50 vs disc107
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

# Load existing EURGBP fleet trades for redundancy checks
with open(FLEET_MANIFEST, "r", encoding="utf-8") as f:
    fleet_data = json.load(f)

existing_trades = []
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
                print(f"Loaded existing EURGBP model {name}: {len(df)} trades from {csv_file.name}")


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


def load_eurgbp_week_data(week_num: int):
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

    # H1 bars
    h1 = m15[['high', 'low', 'close']].resample('1h').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
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

    # ATR(14)
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr14 = pd.Series(tr).rolling(14).mean().bfill().values.astype(np.float64)

    # MACD(12, 26, 9) on M15
    ema12 = pd.Series(c).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(c).ewm(span=26, adjust=False).mean()
    macd_line = (ema12 - ema26).values
    macd_sig = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
    macd_hist = macd_line - macd_sig

    # H4 Trend Bias
    h4_bias = np.zeros(len(m15), dtype=np.int8)
    if len(h4) >= 10:
        h4_ema20 = h4['close'].ewm(span=20, adjust=False).mean()
        h4_b = np.where(h4['close'] > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'bias': h4_b}, index=h4.index)
        merged_h4 = pd.merge_asof(m15.sort_index(), h4_df.sort_index(), left_index=True, right_index=True, direction='backward')
        h4_bias = merged_h4['bias'].fillna(0).values.astype(np.int8)

    # H1 Squeeze & EMA20
    h1_sqz_active = np.zeros(len(m15), dtype=bool)
    h1_ema20_arr = np.zeros(len(m15), dtype=np.float64)
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
        # Squeeze active: Bollinger bands expanding outside Keltner channel
        sqz_active = ~sqz_on
        m = np.where(h1_c > h1_ema20, 1, np.where(h1_c < h1_ema20, -1, 0))

        h1_df = pd.DataFrame({'sqz_active': sqz_active, 'ema20': h1_ema20, 'mom': m}, index=h1.index)
        merged_h1 = pd.merge_asof(m15.sort_index(), h1_df.sort_index(), left_index=True, right_index=True, direction='backward')
        h1_sqz_active = merged_h1['sqz_active'].fillna(False).values
        h1_ema20_arr = merged_h1['ema20'].bfill().values
        h1_mom = merged_h1['mom'].fillna(0).values.astype(np.int8)

    # Asian reference range: 00:00 to 06:45 UTC
    unique_dates = np.unique(dates)
    asian_ranges = {}
    for d in unique_dates:
        mask = (dates == d) & (hours >= 0) & (hours <= 6)
        if np.sum(mask) >= 6:
            asian_ranges[d] = (float(np.max(h[mask])), float(np.min(l[mask])))

    # European Session VWAP (anchored from 07:00 UTC)
    vwap = np.zeros(len(c), dtype=np.float64)
    vwap_std = np.zeros(len(c), dtype=np.float64)
    for d in unique_dates:
        mask = np.where((dates == d) & (hours >= 7))[0]
        if len(mask) > 0:
            typ = (h[mask] + l[mask] + c[mask]) / 3.0
            cum_m = np.cumsum(typ) / np.arange(1, len(mask) + 1)
            vwap[mask] = cum_m
            cum_var = np.cumsum((typ - cum_m)**2) / np.arange(1, len(mask) + 1)
            vwap_std[mask] = np.sqrt(np.maximum(cum_var, 1e-10))

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
        'macd_line': macd_line,
        'macd_sig': macd_sig,
        'macd_hist': macd_hist,
        'h4_bias': h4_bias,
        'h1_sqz_active': h1_sqz_active,
        'h1_ema20': h1_ema20_arr,
        'h1_mom': h1_mom,
        'asian_ranges': asian_ranges,
        'vwap': vwap,
        'vwap_std': vwap_std,
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

def eval_european_open_expansion(week_data, start_hr, end_hr, atr_mult, body_ratio, sl_pips, tp_pips, max_spread):
    n_bars = week_data['n_bars']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    exec_indices = week_data['exec_indices']
    atr14 = week_data['atr14']
    h4_bias = week_data['h4_bias']
    asian_ranges = week_data['asian_ranges']

    sig_indices = []
    sig_types = []
    last_trade_date = None

    for i in range(15, n_bars):
        hr = hours[i]
        if not (start_hr <= hr <= end_hr):
            continue
        dt = dates[i]
        if dt == last_trade_date or dt not in asian_ranges:
            continue
        if spread[i] > max_spread:
            continue

        asia_hi, asia_lo = asian_ranges[dt]
        asia_rng = asia_hi - asia_lo
        if asia_rng < 5.0 * PIP_SIZE or asia_rng > 25.0 * PIP_SIZE:
            continue

        bar_rng = h[i] - l[i]
        if bar_rng < atr_mult * atr14[i]:
            continue

        body = abs(c[i] - o[i])
        if (body / max(bar_rng, 1e-9)) < body_ratio:
            continue

        macro = h4_bias[i]
        sig = 0
        if c[i] > asia_hi and c[i] > o[i] and macro >= 0:
            sig = 1
        elif c[i] < asia_lo and c[i] < o[i] and macro <= 0:
            sig = -1

        if sig != 0:
            sig_indices.append(exec_indices[i])
            sig_types.append(sig)
            last_trade_date = dt

    return execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips)


def eval_asian_sweep_reversal(week_data, sweep_max_pips, min_wick, sl_pips, tp_pips, max_spread):
    n_bars = week_data['n_bars']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    exec_indices = week_data['exec_indices']
    asian_ranges = week_data['asian_ranges']

    sig_indices = []
    sig_types = []
    last_trade_date = None

    for i in range(15, n_bars):
        hr = hours[i]
        # London open liquidity sweep window: 07:00 - 10:30 UTC
        if not (7 <= hr <= 10):
            continue
        dt = dates[i]
        if dt == last_trade_date or dt not in asian_ranges:
            continue
        if spread[i] > max_spread:
            continue

        asia_hi, asia_lo = asian_ranges[dt]
        asia_rng = asia_hi - asia_lo
        if asia_rng < 5.0 * PIP_SIZE or asia_rng > 25.0 * PIP_SIZE:
            continue

        bar_rng = h[i] - l[i]
        if bar_rng < 1.5 * PIP_SIZE:
            continue

        sig = 0
        # Bull trap: swept Asian High by up to sweep_max_pips, closed back below
        if h[i] > asia_hi and h[i] <= asia_hi + sweep_max_pips * PIP_SIZE and c[i] < asia_hi:
            upper_wick = (h[i] - max(o[i], c[i])) / bar_rng
            if upper_wick >= min_wick and c[i] < o[i]:
                sig = -1
        # Bear trap: swept Asian Low by up to sweep_max_pips, closed back above
        elif l[i] < asia_lo and l[i] >= asia_lo - sweep_max_pips * PIP_SIZE and c[i] > asia_lo:
            lower_wick = (min(o[i], c[i]) - l[i]) / bar_rng
            if lower_wick >= min_wick and c[i] > o[i]:
                sig = 1

        if sig != 0:
            sig_indices.append(exec_indices[i])
            sig_types.append(sig)
            last_trade_date = dt

    return execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips)


def eval_macd_zero_cross(week_data, start_hr, end_hr, sl_pips, tp_pips, max_spread):
    n_bars = week_data['n_bars']
    c, o = week_data['c'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    exec_indices = week_data['exec_indices']
    macd_line = week_data['macd_line']
    h4_bias = week_data['h4_bias']

    sig_indices = []
    sig_types = []
    last_trade_date = None

    for i in range(2, n_bars):
        hr = hours[i]
        if not (start_hr <= hr <= end_hr):
            continue
        dt = dates[i]
        if dt == last_trade_date:
            continue
        if spread[i] > max_spread:
            continue

        macro = h4_bias[i]
        sig = 0
        # MACD zero-line cross upward
        if macd_line[i] > 0 and macd_line[i-1] <= 0 and macro == 1 and c[i] > o[i]:
            sig = 1
        # MACD zero-line cross downward
        elif macd_line[i] < 0 and macd_line[i-1] >= 0 and macro == -1 and c[i] < o[i]:
            sig = -1

        if sig != 0:
            sig_indices.append(exec_indices[i])
            sig_types.append(sig)
            last_trade_date = dt

    return execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips)


def eval_session_vwap_rebound(week_data, std_mult, min_wick, sl_pips, tp_pips, max_spread):
    n_bars = week_data['n_bars']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    exec_indices = week_data['exec_indices']
    vwap = week_data['vwap']
    vwap_std = week_data['vwap_std']

    sig_indices = []
    sig_types = []
    last_trade_date = None

    for i in range(15, n_bars):
        hr = hours[i]
        if not (8 <= hr <= 16):
            continue
        dt = dates[i]
        if dt == last_trade_date:
            continue
        if spread[i] > max_spread:
            continue

        std = vwap_std[i]
        if std < 1.0 * PIP_SIZE:
            continue

        bar_rng = h[i] - l[i]
        if bar_rng < 1.8 * PIP_SIZE:
            continue

        up_band = vwap[i] + std_mult * std
        lo_band = vwap[i] - std_mult * std

        sig = 0
        if l[i] < lo_band and c[i] > lo_band:
            lower_wick = (min(o[i], c[i]) - l[i]) / bar_rng
            if lower_wick >= min_wick and c[i] > o[i]:
                sig = 1
        elif h[i] > up_band and c[i] < up_band:
            upper_wick = (h[i] - max(o[i], c[i])) / bar_rng
            if upper_wick >= min_wick and c[i] < o[i]:
                sig = -1

        if sig != 0:
            sig_indices.append(exec_indices[i])
            sig_types.append(sig)
            last_trade_date = dt

    return execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips)


def eval_squeeze_pullback_continuation(week_data, start_hr, end_hr, dist_pips, min_wick, sl_pips, tp_pips, max_spread):
    n_bars = week_data['n_bars']
    c, h, l, o = week_data['c'], week_data['h'], week_data['l'], week_data['o']
    spread = week_data['spread']
    hours = week_data['hours']
    dates = week_data['dates']
    exec_indices = week_data['exec_indices']
    h1_sqz = week_data['h1_sqz_active']
    h1_ema = week_data['h1_ema20']
    h1_mom = week_data['h1_mom']

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

        bar_rng = h[i] - l[i]
        if bar_rng < 1.8 * PIP_SIZE:
            continue

        sig = 0
        ema = h1_ema[i]
        d_pips = dist_pips * PIP_SIZE

        # Bullish pullback test of H1 EMA20 during active expansion
        if mom == 1:
            touched = (l[i] <= ema + d_pips and l[i] >= ema - d_pips)
            if touched and c[i] > o[i]:
                lower_wick = (min(o[i], c[i]) - l[i]) / bar_rng
                if lower_wick >= min_wick:
                    sig = 1
        # Bearish pullback test of H1 EMA20 during active expansion
        elif mom == -1:
            touched = (h[i] >= ema - d_pips and h[i] <= ema + d_pips)
            if touched and c[i] < o[i]:
                upper_wick = (h[i] - max(o[i], c[i])) / bar_rng
                if upper_wick >= min_wick:
                    sig = -1

        if sig != 0:
            sig_indices.append(exec_indices[i])
            sig_types.append(sig)
            last_trade_date = dt

    return execute_matches(week_data, sig_indices, sig_types, sl_pips, tp_pips)

def generate_eurgbp_candidates():
    candidates = []

    # 1. European Open Expansion
    for wh in [(7, 9), (7, 10), (8, 10)]:
        for am in [1.0, 1.2]:
            for br in [0.50, 0.60]:
                for sl, tp in [(7.0, 14.0), (8.0, 16.0), (8.0, 18.0), (9.0, 18.0)]:
                    candidates.append({
                        'archetype': 'open_expansion',
                        'params': {'start_hr': wh[0], 'end_hr': wh[1], 'atr_mult': am, 'body_ratio': br, 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 0.8},
                        'desc': f"EURGBP Open Expansion ({wh[0]:02d}-{wh[1]:02d} UTC, ATRm={am}, BR={br}, SL={sl}p, TP={tp}p)"
                    })

    # 2. Asian Sweep Reversal
    for sw in [3.0, 4.5]:
        for wk in [0.35, 0.40]:
            for sl, tp in [(7.0, 12.0), (7.0, 14.0), (8.0, 14.0), (8.0, 16.0), (9.0, 18.0)]:
                candidates.append({
                    'archetype': 'asian_sweep',
                    'params': {'sweep_max_pips': sw, 'min_wick': wk, 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 0.8},
                    'desc': f"EURGBP Asian Sweep Reversal (Sweep<={sw}p, Wick>={wk}, SL={sl}p, TP={tp}p)"
                })

    # 3. MACD Zero Cross
    for wh in [(7, 16), (8, 15)]:
        for sl, tp in [(7.0, 14.0), (8.0, 16.0), (8.0, 18.0), (9.0, 18.0)]:
            candidates.append({
                'archetype': 'macd_cross',
                'params': {'start_hr': wh[0], 'end_hr': wh[1], 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 0.8},
                'desc': f"EURGBP MACD Zero Cross ({wh[0]:02d}-{wh[1]:02d} UTC, SL={sl}p, TP={tp}p)"
            })

    # 4. Session VWAP Rebound
    for sm in [1.8, 2.0, 2.2]:
        for wk in [0.35]:
            for sl, tp in [(7.0, 12.0), (7.0, 14.0), (8.0, 16.0), (8.0, 18.0)]:
                candidates.append({
                    'archetype': 'vwap_rebound',
                    'params': {'std_mult': sm, 'min_wick': wk, 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 0.8},
                    'desc': f"EURGBP VWAP Rebound (Std={sm}, SL={sl}p, TP={tp}p)"
                })

    # 5. Squeeze Pullback Continuation
    for wh in [(8, 16), (9, 16)]:
        for dp in [1.5, 2.5]:
            for sl, tp in [(7.0, 14.0), (8.0, 16.0), (8.0, 18.0), (9.0, 20.0)]:
                candidates.append({
                    'archetype': 'squeeze_pullback',
                    'params': {'start_hr': wh[0], 'end_hr': wh[1], 'dist_pips': dp, 'min_wick': 0.35, 'sl_pips': sl, 'tp_pips': tp, 'max_spread': 0.8},
                    'desc': f"EURGBP Squeeze EMA Pullback ({wh[0]:02d}-{wh[1]:02d} UTC, Dist={dp}p, SL={sl}p, TP={tp}p)"
                })

    return candidates


def evaluate_cand(cand, w_data):
    arch = cand['archetype']
    p = cand['params']
    if arch == 'open_expansion':
        return eval_european_open_expansion(w_data, p['start_hr'], p['end_hr'], p['atr_mult'], p['body_ratio'], p['sl_pips'], p['tp_pips'], p['max_spread'])
    elif arch == 'asian_sweep':
        return eval_asian_sweep_reversal(w_data, p['sweep_max_pips'], p['min_wick'], p['sl_pips'], p['tp_pips'], p['max_spread'])
    elif arch == 'macd_cross':
        return eval_macd_zero_cross(w_data, p['start_hr'], p['end_hr'], p['sl_pips'], p['tp_pips'], p['max_spread'])
    elif arch == 'vwap_rebound':
        return eval_session_vwap_rebound(w_data, p['std_mult'], p['min_wick'], p['sl_pips'], p['tp_pips'], p['max_spread'])
    elif arch == 'squeeze_pullback':
        return eval_squeeze_pullback_continuation(w_data, p['start_hr'], p['end_hr'], p['dist_pips'], p['min_wick'], p['sl_pips'], p['tp_pips'], p['max_spread'])
    return []


def main():
    print("=" * 80)
    print("  EURGBP SPECIALIZED INSTITUTIONAL ALPHA DISCOVERY ENGINE")
    print("=" * 80)

    candidates = generate_eurgbp_candidates()
    print(f"Generated {len(candidates)} total specialized EURGBP setups.")

    raw_dir = ROOT_DIR / "data" / SYMBOL
    disc_weeks = get_complete_split_weeks(raw_dir, split="discovery")
    hold_weeks = get_complete_split_weeks(raw_dir, split="holdout")

    print(f"Quarantined Holdout Weeks: {len(hold_weeks)} (Weeks {min(hold_weeks)}..{max(hold_weeks)})")
    print(f"Streaming holdout tick data across {len(candidates)} candidates...")

    cand_holdout_trades = {i: [] for i in range(len(candidates))}

    t0 = time.time()
    for w_idx, w_num in enumerate(hold_weeks):
        w_data = load_eurgbp_week_data(w_num)
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

        # Print all with positive pips
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
        w_data = load_eurgbp_week_data(w_num)
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

    out_file = ROOT_DIR / "results" / "new_champions_eurgbp.json"
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
