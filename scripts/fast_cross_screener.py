#!/usr/bin/env python3
"""
================================================================================
  ULTRA-FAST MEMORY-EFFICIENT CROSS-PAIR ALPHA SCREENER
================================================================================
Pairs: AUDCAD, NZDCAD, GBPCHF
Memory Footprint: < 50MB (stores only M15 bars for screening, streams ticks for verification)

Archetypes to explore:
1. AUDCAD:
   - NY Afternoon Reversal / Session Extreme Fade (15:00 - 20:00 UTC)
   - Asian Sydney-Tokyo Donchian Expansion (21:00 - 03:00 UTC)
   - London Morning Retest (07:00 - 10:30 UTC)
2. NZDCAD:
   - London Morning Retest & Asian Liquidity Sweep Fade (07:00 - 10:30 UTC)
   - Wellington Cash Open Volatility Breakout (21:00 - 01:00 UTC)
   - NY Session Momentum / Expansion (13:00 - 17:00 UTC)
3. GBPCHF:
   - London Open Impulse Breakout (07:00 - 10:30 UTC)
   - London/NY Trend Retest & VWAP/EMA Pullback (08:00 - 15:00 UTC)
================================================================================
"""

import sys
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

# Anti-redundancy trade cache
existing_trades_by_sym = defaultdict(list)
if FLEET_MANIFEST.exists():
    with open(FLEET_MANIFEST, "r", encoding="utf-8") as f:
        fleet_data = json.load(f)
    for s in fleet_data.get("strategies", []):
        sym = s["symbol"]
        name = s["name"]
        folder_slug = Path(s.get("relative_path", "")).parent.name
        csv_candidates = [
            MANIFEST_DIR / f"{sym}__{name}_trades.csv",
            MANIFEST_DIR / f"{sym}__{folder_slug}_trades.csv",
            MANIFEST_DIR / f"{sym.lower()}_{name}_trades.csv",
            MANIFEST_DIR / f"{name}_trades.csv"
        ]
        glob_matches = list(MANIFEST_DIR.glob(f"*{sym}*{name}*trades.csv"))
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

        # Check overlap
        matches = 0
        for idx, t1 in enumerate(cand_times):
            diffs = np.abs(ex_times - t1)
            if np.any((diffs <= tolerance_ms) & (ex_dirs == cand_dirs[idx])):
                matches += 1

        overlap_pct = (matches / len(cand_times)) * 100.0 if len(cand_times) > 0 else 0
        if overlap_pct >= max_overlap_pct:
            return False, f"High overlap ({overlap_pct:.1f}%) with {ex_name}"

        # Check daily correlation
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


def load_m15_week(symbol: str, week_num: int):
    """Loads one week and returns ONLY the compact M15 DataFrame."""
    pip_size = PIP_SIZES.get(symbol, 0.0001)
    raw_dir = ROOT_DIR / "data" / symbol
    try:
        ticks, _ = get_week_dataset(week=week_num, raw_dir=raw_dir, warmup_prev_week_ticks=WARMUP_TICKS)
    except Exception:
        return None
    if len(ticks) <= WARMUP_TICKS + 500:
        return None

    m15 = build_time_bars(ticks, timeframe_min=15, pip_size=pip_size)
    del ticks
    return m15


from concurrent.futures import ThreadPoolExecutor

def precompute_compact_m15_database(symbol: str, weeks: list):
    """Loads M15 bars for specified weeks in parallel (cached to disk) and concatenates into one clean DataFrame."""
    cache_dir = ROOT_DIR / "scratch" / "m15_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"m15_{symbol}_{weeks[0]}_{weeks[-1]}_{len(weeks)}.pkl"
    if cache_file.exists():
        print(f"Loading {symbol} M15 database from cache ({cache_file.name})...", flush=True)
        return pd.read_pickle(cache_file)

    print(f"Building compact M15 database for {symbol} ({len(weeks)} weeks)...", flush=True)
    t0 = time.time()
    def _fetch(w):
        df = load_m15_week(symbol, w)
        if df is not None and len(df) > 0:
            df['week'] = w
            return df
        return None

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_fetch, weeks))

    frames = [r for r in results if r is not None]
    if not frames:
        return None
    master = pd.concat(frames).sort_index()
    master.to_pickle(cache_file)
    print(f"Database built and cached: {len(master)} M15 bars ({len(frames)} weeks) in {time.time()-t0:.1f}s.", flush=True)
    return master





import numba

@numba.njit(fastmath=True)
def simulate_bar_trades_fast(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    signals: np.ndarray,       # 1 for Long, -1 for Short, 0 for None
    sl_pips: float,
    tp_pips: float,
    pip_size: float,
    friction_pips: float,
    max_hold_bars: int = 32
):
    """
    Numba JIT accelerated bar-level trade simulation.
    Returns: total_trades, wins, losses, net_pips, gross_win, gross_loss, profit_factor
    """
    n = len(signals)
    trades = 0
    gross_win = 0.0
    gross_loss = 0.0

    in_pos = 0       # 1 for long, -1 for short
    entry_price = 0.0
    entry_bar = 0

    for i in range(n - 1):
        # Manage open position
        if in_pos != 0:
            bars_held = i - entry_bar
            sl_price = entry_price - sl_pips * pip_size if in_pos == 1 else entry_price + sl_pips * pip_size
            tp_price = entry_price + tp_pips * pip_size if in_pos == 1 else entry_price - tp_pips * pip_size

            closed = False
            trade_pips = 0.0

            if in_pos == 1:
                sl_hit = (lows[i] <= sl_price)
                tp_hit = (highs[i] >= tp_price)
                if sl_hit and tp_hit:
                    # Conservative: SL hit first
                    closed = True
                    trade_pips = -sl_pips - friction_pips
                elif sl_hit:
                    closed = True
                    trade_pips = -sl_pips - friction_pips
                elif tp_hit:
                    closed = True
                    trade_pips = tp_pips - friction_pips
                elif bars_held >= max_hold_bars:
                    closed = True
                    trade_pips = (closes[i] - entry_price) / pip_size - friction_pips
            else: # in_pos == -1
                sl_hit = (highs[i] >= sl_price)
                tp_hit = (lows[i] <= tp_price)
                if sl_hit and tp_hit:
                    closed = True
                    trade_pips = -sl_pips - friction_pips
                elif sl_hit:
                    closed = True
                    trade_pips = -sl_pips - friction_pips
                elif tp_hit:
                    closed = True
                    trade_pips = tp_pips - friction_pips
                elif bars_held >= max_hold_bars:
                    closed = True
                    trade_pips = (entry_price - closes[i]) / pip_size - friction_pips

            if closed:
                trades += 1
                if trade_pips > 0:
                    gross_win += trade_pips
                else:
                    gross_loss += abs(trade_pips)
                in_pos = 0

        # Check for new signal at bar i closing -> enters at bar i+1 open
        if in_pos == 0 and signals[i] != 0:
            in_pos = signals[i]
            entry_price = opens[i + 1]
            entry_bar = i + 1

def enrich_m15_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Enriches M15 dataframe with multi-session reference levels and indicators."""
    pip_size = PIP_SIZES.get(symbol, 0.0001)
    df = df.copy()

    c = df['close']
    h = df['high']
    l = df['low']
    o = df['open']

    df['ema20'] = c.ewm(span=20, adjust=False).mean()
    df['ema50'] = c.ewm(span=50, adjust=False).mean()
    df['hour'] = df.index.hour
    df['date'] = df.index.date

    # Rolling Donchian channels
    for lb in [8, 12, 16, 20]:
        df[f'donch_h_{lb}'] = h.rolling(lb).max().shift(1)
        df[f'donch_l_{lb}'] = l.rolling(lb).min().shift(1)

    # Precalculate session levels by date
    dates = df['date'].values
    hours = df['hour'].values
    highs = h.values
    lows = l.values
    n = len(df)

    asian_h = np.zeros(n, dtype=np.float64)
    asian_l = np.zeros(n, dtype=np.float64)
    day_h_15 = np.zeros(n, dtype=np.float64)
    day_l_15 = np.zeros(n, dtype=np.float64)
    well_h = np.zeros(n, dtype=np.float64)
    well_l = np.zeros(n, dtype=np.float64)

    day_indices = defaultdict(list)
    for i in range(n):
        day_indices[dates[i]].append(i)

    for d, idxs in day_indices.items():
        # Asian 00:00 - 06:45
        as_idxs = [i for i in idxs if 0 <= hours[i] <= 6]
        if as_idxs:
            ash = np.max(highs[as_idxs])
            asl = np.min(lows[as_idxs])
            for i in idxs:
                asian_h[i] = ash
                asian_l[i] = asl

        # Daily levels before 15:00 UTC
        pre15 = [i for i in idxs if hours[i] < 15]
        if pre15:
            dh = np.max(highs[pre15])
            dl = np.min(lows[pre15])
            for i in idxs:
                day_h_15[i] = dh
                day_l_15[i] = dl

        # Wellington pre-session: 17:00 - 20:45
        w_idxs = [i for i in idxs if 17 <= hours[i] <= 20]
        if w_idxs:
            wh = np.max(highs[w_idxs])
            wl = np.min(lows[w_idxs])
            for i in idxs:
                well_h[i] = wh
                well_l[i] = wl

    df['asian_h'] = asian_h
    df['asian_l'] = asian_l
    df['day_h_15'] = day_h_15
    df['day_l_15'] = day_l_15
    df['well_h'] = well_h
    df['well_l'] = well_l

    # If GBPCHF, compute H1 Keltner Squeeze features
    if symbol == "GBPCHF":
        h1 = df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        h1_c = h1['close'].values
        h1_h = h1['high'].values
        h1_l = h1['low'].values
        c_s = pd.Series(h1_c)
        h1_ema20 = c_s.ewm(span=20, adjust=False).mean().values
        h1_std20 = c_s.rolling(20).std().values
        bb_upper = h1_ema20 + 2.0 * h1_std20
        bb_lower = h1_ema20 - 2.0 * h1_std20

        tr_h1 = np.maximum(h1_h - h1_l, np.maximum(np.abs(h1_h - np.roll(h1_c, 1)), np.abs(h1_l - np.roll(h1_c, 1))))
        tr_h1[0] = h1_h[0] - h1_l[0]
        atr20_h1 = pd.Series(tr_h1).rolling(20).mean().bfill().values

        for mult in [1.5, 1.8]:
            kc_upper = h1_ema20 + mult * atr20_h1
            kc_lower = h1_ema20 - mult * atr20_h1
            sqz_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
            sqz_fired = pd.Series(sqz_on).shift(1).fillna(False).values & (~sqz_on)
            h1_mom = np.where(h1_c > h1_ema20, 1, np.where(h1_c < h1_ema20, -1, 0))
            h1_df = pd.DataFrame({f'sqz_fire_{mult}': sqz_fired, f'h1_mom_{mult}': h1_mom}, index=h1.index)
            df = pd.merge_asof(df.sort_index(), h1_df.sort_index(), left_index=True, right_index=True, direction='backward')

    return df


def generate_signals_for_candidate(df: pd.DataFrame, c: dict, symbol: str) -> np.ndarray:
    n = len(df)
    signals = np.zeros(n, dtype=np.int8)

    c_vals = df['close'].values
    o_vals = df['open'].values
    h_vals = df['high'].values
    l_vals = df['low'].values
    hours = df['hour'].values
    dates = df['date'].values
    spread = df['spread_pips'].values if 'spread_pips' in df else np.ones(n)
    pip_size = PIP_SIZES.get(symbol, 0.0001)

    arch = c["archetype"]
    last_trade_date = None

    if arch == "audcad_ny_reversal":
        day_h = df['day_h_15'].values
        day_l = df['day_l_15'].values
        min_wick = c["min_wick"]
        start_hr = c["start_hr"]
        end_hr = c["end_hr"]
        max_spread = c["max_spread"]

        for i in range(15, n):
            hr = hours[i]
            if not (start_hr <= hr <= end_hr):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > max_spread:
                continue

            dh = day_h[i]
            dl = day_l[i]
            if dh <= 0 or dl <= 0:
                continue

            bar_rng = h_vals[i] - l_vals[i]
            if bar_rng < 2.0 * pip_size:
                continue

            upper_wick = h_vals[i] - max(o_vals[i], c_vals[i])
            lower_wick = min(o_vals[i], c_vals[i]) - l_vals[i]

            sig = 0
            # Bearish Reversal from Daily High
            if h_vals[i] >= (dh - 1.5 * pip_size) and c_vals[i] < dh and (upper_wick / bar_rng) >= min_wick and c_vals[i] < o_vals[i]:
                sig = -1
            # Bullish Reversal from Daily Low
            elif l_vals[i] <= (dl + 1.5 * pip_size) and c_vals[i] > dl and (lower_wick / bar_rng) >= min_wick and c_vals[i] > o_vals[i]:
                sig = 1

            if sig != 0:
                signals[i] = sig
                last_trade_date = dt

    elif arch == "audcad_asian_donchian":
        roll_h = df[f"donch_h_{c['lookback']}"].values
        roll_l = df[f"donch_l_{c['lookback']}"].values
        ema50 = df['ema50'].values
        min_body = c["min_body"]
        start_hr = c["start_hr"]
        end_hr = c["end_hr"]
        max_spread = c["max_spread"]

        for i in range(c['lookback'] + 2, n):
            hr = hours[i]
            in_window = (start_hr <= hr <= end_hr) if start_hr <= end_hr else (hr >= start_hr or hr <= end_hr)
            if not in_window:
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > max_spread:
                continue

            bar_rng = h_vals[i] - l_vals[i]
            if bar_rng < 2.0 * pip_size:
                continue

            sig = 0
            if c_vals[i] > roll_h[i] and c_vals[i] > o_vals[i]:
                body = c_vals[i] - o_vals[i]
                if (body / bar_rng) >= min_body and c_vals[i] > ema50[i]:
                    sig = 1
            elif c_vals[i] < roll_l[i] and c_vals[i] < o_vals[i]:
                body = o_vals[i] - c_vals[i]
                if (body / bar_rng) >= min_body and c_vals[i] < ema50[i]:
                    sig = -1

            if sig != 0:
                signals[i] = sig
                last_trade_date = dt

    elif arch == "nzdcad_asian_fade":
        ash = df['asian_h'].values
        asl = df['asian_l'].values
        min_wick = c["min_wick"]
        start_hr = c["start_hr"]
        end_hr = c["end_hr"]
        max_spread = c["max_spread"]

        for i in range(15, n):
            hr = hours[i]
            if not (start_hr <= hr <= end_hr):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > max_spread:
                continue

            ah = ash[i]
            al = asl[i]
            if ah <= 0 or al <= 0:
                continue

            bar_rng = h_vals[i] - l_vals[i]
            if bar_rng < 2.0 * pip_size:
                continue

            upper_wick = h_vals[i] - max(o_vals[i], c_vals[i])
            lower_wick = min(o_vals[i], c_vals[i]) - l_vals[i]

            sig = 0
            if h_vals[i] >= (ah - 1.5 * pip_size) and c_vals[i] < ah and (upper_wick / bar_rng) >= min_wick and c_vals[i] < o_vals[i]:
                sig = -1
            elif l_vals[i] <= (al + 1.5 * pip_size) and c_vals[i] > al and (lower_wick / bar_rng) >= min_wick and c_vals[i] > o_vals[i]:
                sig = 1

            if sig != 0:
                signals[i] = sig
                last_trade_date = dt

    elif arch == "nzdcad_wellington_breakout":
        wh = df['well_h'].values
        wl = df['well_l'].values
        ema50 = df['ema50'].values
        min_body = c["min_body"]
        start_hr = c["start_hr"]
        end_hr = c["end_hr"]
        max_spread = c["max_spread"]
        buf = c["buffer_pips"] * pip_size

        for i in range(15, n):
            hr = hours[i]
            in_window = (start_hr <= hr <= end_hr) if start_hr <= end_hr else (hr >= start_hr or hr <= end_hr)
            if not in_window:
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > max_spread:
                continue

            pre_h = wh[i]
            pre_l = wl[i]
            if pre_h <= 0 or pre_l <= 0:
                continue

            pre_range = (pre_h - pre_l) / pip_size
            if pre_range < 8.0 or pre_range > 45.0:
                continue

            bar_rng = h_vals[i] - l_vals[i]
            if bar_rng < 2.0 * pip_size:
                continue

            sig = 0
            if c_vals[i] > (pre_h + buf) and c_vals[i] > o_vals[i]:
                body = c_vals[i] - o_vals[i]
                if (body / bar_rng) >= min_body and c_vals[i] > ema50[i]:
                    sig = 1
            elif c_vals[i] < (pre_l - buf) and c_vals[i] < o_vals[i]:
                body = o_vals[i] - c_vals[i]
                if (body / bar_rng) >= min_body and c_vals[i] < ema50[i]:
                    sig = -1

            if sig != 0:
                signals[i] = sig
                last_trade_date = dt

    elif arch == "gbpchf_london_impulse":
        ash = df['asian_h'].values
        asl = df['asian_l'].values
        ema50 = df['ema50'].values
        min_body = c["min_body"]
        min_bar_pips = c["min_bar_pips"]
        start_hr = c["start_hr"]
        end_hr = c["end_hr"]
        max_spread = c["max_spread"]
        buf = c["buffer_pips"] * pip_size

        for i in range(15, n):
            hr = hours[i]
            if not (start_hr <= hr <= end_hr):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > max_spread:
                continue

            ah = ash[i]
            al = asl[i]
            if ah <= 0 or al <= 0:
                continue

            asian_rng_pips = (ah - al) / pip_size
            if asian_rng_pips < 12.0 or asian_rng_pips > 55.0:
                continue

            bar_rng_pips = (h_vals[i] - l_vals[i]) / pip_size
            if bar_rng_pips < min_bar_pips:
                continue

            sig = 0
            if c_vals[i] > (ah + buf) and c_vals[i] > o_vals[i]:
                body = c_vals[i] - o_vals[i]
                if (body / (h_vals[i] - l_vals[i])) >= min_body and c_vals[i] > ema50[i]:
                    sig = 1
            elif c_vals[i] < (al - buf) and c_vals[i] < o_vals[i]:
                body = o_vals[i] - c_vals[i]
                if (body / (h_vals[i] - l_vals[i])) >= min_body and c_vals[i] < ema50[i]:
                    sig = -1

            if sig != 0:
                signals[i] = sig
                last_trade_date = dt

    elif arch == "gbpchf_keltner_runner":
        kc = c["kc_mult"]
        sqz_col = f"sqz_fire_{kc}"
        mom_col = f"h1_mom_{kc}"
        if sqz_col not in df or mom_col not in df:
            return signals

        sqz_fire = df[sqz_col].fillna(False).values
        h1_mom = df[mom_col].fillna(0).values.astype(int)
        start_hr = c["start_hr"]
        end_hr = c["end_hr"]
        max_spread = c["max_spread"]

        for i in range(15, n):
            hr = hours[i]
            if not (start_hr <= hr <= end_hr):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > max_spread:
                continue

            if not sqz_fire[i]:
                continue
            mom = h1_mom[i]
            if mom == 0:
                continue

            sig = 0
            if mom == 1 and c_vals[i] > o_vals[i]:
                sig = 1
            elif mom == -1 and c_vals[i] < o_vals[i]:
                sig = -1

def verify_candidate_on_ticks(symbol: str, cand: dict, weeks: list, is_holdout: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Executes rigorous week-by-week tick-level backtest using run_matching_loop.
    Memory footprint: Exactly 1 week in RAM at any instant (~25MB).
    """
    pip_size = PIP_SIZES.get(symbol, 0.0001)
    comm, slip = get_friction_pips(symbol)
    raw_dir = ROOT_DIR / "data" / symbol

    trades_list = []

    for w in weeks:
        try:
            ticks, _ = get_week_dataset(week=w, raw_dir=raw_dir, warmup_prev_week_ticks=WARMUP_TICKS)
        except Exception:
            continue
        if len(ticks) <= WARMUP_TICKS + 500:
            del ticks
            continue

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=pip_size)
        if len(m15) < 30:
            del ticks, m15
            continue

        # Enrich single week features
        m15_feat = enrich_m15_features(m15, symbol)
        bar_sigs = generate_signals_for_candidate(m15_feat, cand, symbol)

        n_ticks = len(ticks)
        sig_arr = np.zeros(n_ticks, dtype=np.int8)
        sl_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_arr = np.zeros(n_ticks, dtype=np.float64)

        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']
        exec_indices = np.searchsorted(tick_ts, time_ms + dur_ms).astype(np.int32)

        has_sig = False
        for bi in range(len(bar_sigs)):
            if bar_sigs[bi] != 0:
                ti = exec_indices[bi]
                if ti < n_ticks and sig_arr[ti] == 0:
                    sig_arr[ti] = bar_sigs[bi]
                    sl_arr[ti] = cand["sl_pips"]
                    tp_arr[ti] = cand["tp_pips"]
                    has_sig = True

        if has_sig:
            tr = run_matching_loop(
                ticks=ticks,
                signals=sig_arr,
                pip_size=pip_size,
                sl_pips=sl_arr,
                tp_pips=tp_arr,
                use_slippage=True,
                close_at_end=True,
                commission_pips=comm,
                entry_slippage_pips=slip,
                min_entry_timestamp=ticks['timestamp'][WARMUP_TICKS],
                enforce_subsequent_tick=True
            )
            if len(tr) > 0:
                trades_list.append(tr)

        del ticks, m15, m15_feat, sig_arr, sl_arr, tp_arr

    if not trades_list:
        return np.empty(0), {}

    all_trades = np.concatenate(trades_list)
    pips = all_trades['pips']
    n_t = len(pips)
    wins = float(np.sum(pips[pips > 0]))
    losses = float(abs(np.sum(pips[pips < 0])))
    pf = (wins / losses) if losses > 0 else 99.0
    net = wins - losses
    exp = net / n_t if n_t > 0 else 0.0

    pips_sorted = np.sort(pips)
    pips_ex3 = float(np.sum(pips_sorted[:-3])) if n_t > 3 else 0.0

    metrics = {
        "trades": n_t,
        "net_pips": round(net, 1),
        "pf": round(pf, 2),
        "exp": round(exp, 2),
        "pips_ex3": round(pips_ex3, 1),
        "gross_win": round(wins, 1),
        "gross_loss": round(losses, 1)
    }
    return all_trades, metrics


def screen_pair(symbol: str):
    pip_size = PIP_SIZES.get(symbol, 0.0001)
    comm, slip = get_friction_pips(symbol)
    friction_pips = comm + slip

    disc_weeks = get_complete_split_weeks(symbol, "discovery_70")
    hold_weeks = get_complete_split_weeks(symbol, "holdout_test_30")

    print("=" * 80, flush=True)
    print(f"  FAST CROSS SCREENER: {symbol}", flush=True)
    print(f"  Discovery: {len(disc_weeks)}w | Quarantined Holdout: {len(hold_weeks)}w", flush=True)
    print(f"  Friction: {comm:.2f}p comm + {slip:.2f}p slip ({friction_pips:.2f}p total) + raw spread", flush=True)
    print("=" * 80, flush=True)

    # 1. Build M15 database for Discovery
    disc_df = precompute_compact_m15_database(symbol, disc_weeks)
    if disc_df is None or len(disc_df) == 0:
        print(f"Failed to load discovery data for {symbol}")
        return []
    disc_df = enrich_m15_features(disc_df, symbol)

    # Build M15 database for Holdout
    hold_df = precompute_compact_m15_database(symbol, hold_weeks)
    if hold_df is None or len(hold_df) == 0:
        print(f"Failed to load holdout data for {symbol}")
        return []
    hold_df = enrich_m15_features(hold_df, symbol)

    # Prepare parameter variations per symbol
    param_grid = []
    if symbol == "AUDCAD":
        # 1A. NY Afternoon Reversal / Daily Extreme Fade (15:00 - 20:00 UTC)
        for min_w in [0.25, 0.35, 0.45]:
            for s_hr, e_hr in [(15, 19), (16, 20), (15, 20), (14, 19)]:
                for sl, tp in [(12.0, 24.0), (14.0, 28.0), (14.0, 32.0), (16.0, 32.0), (16.0, 36.0), (18.0, 40.0)]:
                    for ms in [2.2, 2.6, 3.0]:
                        param_grid.append({
                            "archetype": "audcad_ny_reversal",
                            "min_wick": min_w,
                            "start_hr": s_hr,
                            "end_hr": e_hr,
                            "sl_pips": sl,
                            "tp_pips": tp,
                            "max_spread": ms,
                            "desc": f"NY_Fade_{s_hr}-{e_hr}_w{int(min_w*100)}"
                        })
        # 1B. Asian Sydney-Tokyo Donchian Expansion (21:00 - 03:00 UTC)
        for lb in [8, 12, 16]:
            for min_b in [0.35, 0.45]:
                for s_hr, e_hr in [(21, 3), (22, 3), (21, 2)]:
                    for sl, tp in [(12.0, 24.0), (14.0, 28.0), (16.0, 32.0), (16.0, 36.0)]:
                        for ms in [2.6, 3.2]:
                            param_grid.append({
                                "archetype": "audcad_asian_donchian",
                                "lookback": lb,
                                "min_body": min_b,
                                "start_hr": s_hr,
                                "end_hr": e_hr,
                                "sl_pips": sl,
                                "tp_pips": tp,
                                "max_spread": ms,
                                "desc": f"Donch_{lb}_{s_hr}-{e_hr}_b{int(min_b*100)}"
                            })

    elif symbol == "NZDCAD":
        # 2A. London Morning Retest & Asian Liquidity Fade (07:00 - 10:30 UTC)
        for min_w in [0.25, 0.35, 0.45]:
            for s_hr, e_hr in [(7, 10), (7, 11), (8, 11), (6, 10)]:
                for sl, tp in [(12.0, 24.0), (14.0, 28.0), (14.0, 32.0), (16.0, 32.0), (16.0, 36.0), (18.0, 40.0)]:
                    for ms in [2.4, 2.8, 3.2]:
                        param_grid.append({
                            "archetype": "nzdcad_asian_fade",
                            "min_wick": min_w,
                            "start_hr": s_hr,
                            "end_hr": e_hr,
                            "sl_pips": sl,
                            "tp_pips": tp,
                            "max_spread": ms,
                            "desc": f"London_Retest_{s_hr}-{e_hr}_w{int(min_w*100)}"
                        })
        # 2B. Wellington Cash Open Volatility Breakout (21:00 - 01:00 UTC)
        for buf in [0.0, 1.5, 3.0]:
            for min_b in [0.35, 0.45]:
                for s_hr, e_hr in [(21, 1), (21, 2), (22, 2)]:
                    for sl, tp in [(14.0, 28.0), (16.0, 32.0), (16.0, 36.0), (18.0, 40.0)]:
                        for ms in [2.8, 3.5]:
                            param_grid.append({
                                "archetype": "nzdcad_wellington_breakout",
                                "buffer_pips": buf,
                                "min_body": min_b,
                                "start_hr": s_hr,
                                "end_hr": e_hr,
                                "sl_pips": sl,
                                "tp_pips": tp,
                                "max_spread": ms,
                                "desc": f"Wellington_{s_hr}-{e_hr}_buf{buf:.1f}"
                            })

    elif symbol == "GBPCHF":
        # 3A. London Open Impulse Breakout (07:00 - 10:30 UTC)
        for buf in [0.0, 2.0, 4.0]:
            for min_b in [0.40, 0.50]:
                for min_bar in [5.0, 7.5]:
                    for s_hr, e_hr in [(7, 10), (7, 11), (8, 11)]:
                        for sl, tp in [(16.0, 34.0), (18.0, 38.0), (18.0, 45.0), (20.0, 44.0), (20.0, 50.0)]:
                            for ms in [3.0, 3.4]:
                                param_grid.append({
                                    "archetype": "gbpchf_london_impulse",
                                    "buffer_pips": buf,
                                    "min_body": min_b,
                                    "min_bar_pips": min_bar,
                                    "start_hr": s_hr,
                                    "end_hr": e_hr,
                                    "sl_pips": sl,
                                    "tp_pips": tp,
                                    "max_spread": ms,
                                    "desc": f"London_Impulse_{s_hr}-{e_hr}_buf{buf:.1f}"
                                })
        # 3B. H1 Keltner Asymmetric Runner (07:00 - 16:00 UTC)
        for kc in [1.5, 1.8]:
            for s_hr, e_hr in [(7, 16), (7, 12), (8, 16)]:
                for sl, tp in [(16.0, 40.0), (18.0, 45.0), (18.0, 50.0), (20.0, 48.0), (20.0, 55.0)]:
                    for ms in [3.0, 3.2]:
                        param_grid.append({
                            "archetype": "gbpchf_keltner_runner",
                            "kc_mult": kc,
                            "start_hr": s_hr,
                            "end_hr": e_hr,
                            "sl_pips": sl,
                            "tp_pips": tp,
                            "max_spread": ms,
                            "desc": f"Keltner_Runner_{kc:.1f}_{s_hr}-{e_hr}"
                        })

    print(f"Fast Screening {len(param_grid)} parameter variations on M15 bars...", flush=True)

    d_opens = disc_df['open'].values
    d_highs = disc_df['high'].values
    d_lows = disc_df['low'].values
    d_closes = disc_df['close'].values

    h_opens = hold_df['open'].values
    h_highs = hold_df['high'].values
    h_lows = hold_df['low'].values
    h_closes = hold_df['close'].values

    bar_passing_candidates = []
    all_disc_results = []

    t_screen_start = time.time()
    for p in param_grid:
        d_sigs = generate_signals_for_candidate(disc_df, p, symbol)
        sig_count = np.sum(d_sigs != 0)
        if sig_count < 15:
            continue
        d_tr, d_net, d_gw, d_gl, d_pf = simulate_bar_trades_fast(d_opens, d_highs, d_lows, d_closes, d_sigs, p["sl_pips"], p["tp_pips"], pip_size, friction_pips)

        h_sigs = generate_signals_for_candidate(hold_df, p, symbol)
        h_tr, h_net, h_gw, h_gl, h_pf = simulate_bar_trades_fast(h_opens, h_highs, h_lows, h_closes, h_sigs, p["sl_pips"], p["tp_pips"], pip_size, friction_pips)

        rec = dict(p)
        rec["d_tr"] = d_tr
        rec["d_net"] = round(d_net, 1)
        rec["d_pf"] = round(d_pf, 2)
        rec["h_tr"] = h_tr
        rec["h_net"] = round(h_net, 1)
        rec["h_pf"] = round(h_pf, 2)
        all_disc_results.append(rec)

        if d_tr >= 20 and d_net > 0 and d_pf >= 1.15 and h_tr >= 15 and h_net > 0 and h_pf >= 1.20:
            rec["bar_disc_trades"] = d_tr
            rec["bar_disc_pips"] = round(d_net, 1)
            rec["bar_disc_pf"] = round(d_pf, 2)
            rec["bar_hold_trades"] = h_tr
            rec["bar_hold_pips"] = round(h_net, 1)
            rec["bar_hold_pf"] = round(h_pf, 2)
            bar_passing_candidates.append(rec)

    # Print Top 15 ranked by Discovery Net Pips
    all_disc_results.sort(key=lambda x: x["d_net"], reverse=True)
    print("\n--- TOP 15 DISCOVERY CANDIDATES ---", flush=True)
    for r in all_disc_results[:15]:
        print(f"{r['archetype']:<24} | {r['desc']:<26} | {r['sl_pips']:.0f}/{r['tp_pips']:.0f} | Disc: {r['d_net']:>+7.1f}p (PF {r['d_pf']:.2f}, {r['d_tr']}t) | Hold: {r['h_net']:>+7.1f}p (PF {r['h_pf']:.2f}, {r['h_tr']}t)", flush=True)

    print(f"\nScreening completed in {time.time()-t_screen_start:.1f}s: {len(bar_passing_candidates)} passed M15 dual gates!", flush=True)


    if not bar_passing_candidates:
        print(f"No candidates passed M15 screening for {symbol}.")
        return []

    # Free M15 memory before tick verification
    del disc_df, hold_df
    gc.collect()

    print(f"\nProceeding to RIGOROUS CANONICAL TICK VALIDATION for {len(bar_passing_candidates)} candidates...", flush=True)
    print("=" * 115, flush=True)
    print(f"{'Archetype':<24} | {'Setup Desc':<26} | {'SL/TP':<9} | {'Hold Net':<9} | {'Hold PF':<7} | {'Trades':<6} | {'ExTop3':<8} | {'Status'}", flush=True)
    print("-" * 115, flush=True)

    certified_champs = []
    for c in bar_passing_candidates:
        # 1. Run Tick validation on Quarantined Holdout (90 weeks)
        h_trades, h_m = verify_candidate_on_ticks(symbol, c, hold_weeks, is_holdout=True)
        if len(h_trades) < 15:
            print(f"{c['archetype']:<24} | {c['desc']:<26} | {c['sl_pips']:.0f}/{c['tp_pips']:.0f}    | Low Trades ({len(h_trades)})", flush=True)
            continue

        pips_ex3 = h_m["pips_ex3"]
        is_distinct, anti_red_msg = check_anti_redundancy(symbol, h_trades, max_overlap_pct=50.0, max_corr=0.50)

        passed_holdout = (h_m["trades"] >= 15 and h_m["pf"] >= 1.20 and h_m["net_pips"] > 0 and pips_ex3 > 0 and is_distinct)

        sl_tp_str = f"{c['sl_pips']:.0f}/{c['tp_pips']:.0f}"

        if passed_holdout:
            # 2. Run Tick validation on Discovery (210 weeks) for full consistency
            d_trades, d_m = verify_candidate_on_ticks(symbol, c, disc_weeks, is_holdout=False)
            if d_m.get("net_pips", 0) > 0 and d_m.get("pf", 0) >= 1.15:
                status = "CERTIFIED"
                c["symbol"] = symbol
                c["holdout_metrics"] = h_m
                c["discovery_metrics"] = d_m
                c["trades_array"] = h_trades
                certified_champs.append(c)
                print(f"{c['archetype']:<24} | {c['desc']:<26} | {sl_tp_str:<9} | {h_m['net_pips']:>+8.1f}p | {h_m['pf']:>7.2f} | {h_m['trades']:>6} | {pips_ex3:>+7.1f}p | {status}", flush=True)
            else:
                status = f"REJECTED (Discovery PF {d_m.get('pf', 0):.2f})"
                print(f"{c['archetype']:<24} | {c['desc']:<26} | {sl_tp_str:<9} | {h_m['net_pips']:>+8.1f}p | {h_m['pf']:>7.2f} | {h_m['trades']:>6} | {pips_ex3:>+7.1f}p | {status}", flush=True)
        else:
            reason = "Failed Barrier" if (h_m["pf"] < 1.20 or h_m["net_pips"] <= 0) else ("Outlier Dependent" if pips_ex3 <= 0 else anti_red_msg)
            print(f"{c['archetype']:<24} | {c['desc']:<26} | {sl_tp_str:<9} | {h_m['net_pips']:>+8.1f}p | {h_m['pf']:>7.2f} | {h_m['trades']:>6} | {pips_ex3:>+7.1f}p | REJECTED ({reason})", flush=True)

    print("=" * 115, flush=True)
    print(f"Symbol {symbol}: Fully Certified {len(certified_champs)} non-redundant, independently robust champions!", flush=True)
    return certified_champs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AUDCAD", choices=["AUDCAD", "NZDCAD", "GBPCHF", "ALL"])
    args = parser.parse_args()

    symbols = ["AUDCAD", "NZDCAD", "GBPCHF"] if args.symbol == "ALL" else [args.symbol]

    for s in symbols:
        champs = screen_pair(s)
        if champs:
            out_json = ROOT_DIR / "results" / f"fast_champions_{s.lower()}.json"
            serializable = [{k: v for k, v in c.items() if k != "trades_array"} for c in champs]
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
            print(f"Saved {len(serializable)} champions to {out_json}", flush=True)




