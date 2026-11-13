import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset, get_complete_split_weeks
from tick_engine.core.config import PIP_SIZES, get_friction_pips, WARMUP_TICKS
from tick_engine.reporting.run_manifest import compute_statistical_metrics
from data_engineering_suite.bars import build_time_bars


def precompute_week_data(symbol: str, week_num: int):
    pip_size = PIP_SIZES[symbol]
    raw_dir = f"data/{symbol}"
    try:
        ticks, _ = get_week_dataset(week_num, raw_dir, warmup_prev_week_ticks=WARMUP_TICKS)
    except Exception:
        return None
    if len(ticks) <= WARMUP_TICKS + 500:
        return None

    m15 = build_time_bars(ticks, timeframe_min=15, pip_size=pip_size)
    if len(m15) < 30:
        return None

    ts = pd.to_datetime(ticks['timestamp'], unit='ms')
    mid = 0.5 * (ticks['bid'] + ticks['ask'])
    base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid}, index=ts)
    h4 = base.resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    if len(h4) < 25:
        return None

    h4['ema20'] = h4['close'].ewm(span=20, adjust=False).mean()
    h4['bias'] = np.where(h4['close'] > h4['ema20'], 1, -1)
    
    return {
        'ticks': ticks,
        'm15': m15,
        'h4': h4,
        'pip_size': pip_size,
        'min_entry_ts': ticks['timestamp'][WARMUP_TICKS]
    }


def eval_config_on_week(week_data, sl, tp, lookback_h4, dist_pips, min_wick, session_mode, max_spread, comm, slip):
    ticks = week_data['ticks']
    m15 = week_data['m15']
    h4 = week_data['h4'].copy()
    pip_size = week_data['pip_size']
    min_entry_ts = week_data['min_entry_ts']

    h4['roll_h'] = h4['high'].rolling(lookback_h4).max().shift(1).bfill()
    h4['roll_l'] = h4['low'].rolling(lookback_h4).min().shift(1).bfill()

    bias_m15 = h4['bias'].reindex(m15.index, method='ffill').fillna(0).values
    supp_m15 = h4['roll_l'].reindex(m15.index, method='ffill').values
    res_m15 = h4['roll_h'].reindex(m15.index, method='ffill').values

    c = m15['close'].values
    h = m15['high'].values
    l = m15['low'].values
    o = m15['open'].values
    spread = m15['spread_pips'].values
    hours = m15.index.hour.values
    dates = m15.index.date
    time_ms = m15.index.astype('datetime64[ms]').view('int64')
    dur_ms = 15 * 60 * 1000
    tick_ts = ticks['timestamp']
    n_ticks = len(ticks)

    signals = np.zeros(n_ticks, dtype=np.int8)
    sl_arr = np.zeros(n_ticks, dtype=np.float64)
    tp_arr = np.zeros(n_ticks, dtype=np.float64)

    last_trade_date = None

    for i in range(15, len(m15)):
        hr = hours[i]
        # Session filters:
        # 1: Asian + London (21-16 UTC)
        # 2: London + NY (06-18 UTC)
        # 3: All Day (0-23 UTC)
        if session_mode == 1:
            if not (hr >= 21 or hr <= 16):
                continue
        elif session_mode == 2:
            if not (6 <= hr <= 18):
                continue

        dt = dates[i]
        if dt == last_trade_date:
            continue
        if spread[i] > max_spread:
            continue

        bar_rng = (h[i] - l[i]) / pip_size
        if bar_rng < 2.5:
            continue

        b = bias_m15[i]
        sig = 0

        # Bullish Retest of H4 Support
        if b > 0 and supp_m15[i] > 0:
            dist_supp = abs(l[i] - supp_m15[i]) / pip_size
            lower_wick = (min(o[i], c[i]) - l[i]) / pip_size
            if dist_supp <= dist_pips and (lower_wick / bar_rng) >= min_wick and c[i] > o[i]:
                sig = 1

        # Bearish Retest of H4 Resistance
        elif b < 0 and res_m15[i] > 0:
            dist_res = abs(res_m15[i] - h[i]) / pip_size
            upper_wick = (h[i] - max(o[i], c[i])) / pip_size
            if dist_res <= dist_pips and (upper_wick / bar_rng) >= min_wick and c[i] < o[i]:
                sig = -1

        if sig != 0:
            exec_t = time_ms[i] + dur_ms
            idx = int(np.searchsorted(tick_ts, exec_t))
            if idx < n_ticks and signals[idx] == 0:
                signals[idx] = sig
                sl_arr[idx] = sl
                tp_arr[idx] = tp
                last_trade_date = dt

    if not np.any(signals != 0):
        return np.empty(0)

    tr = run_matching_loop(
        ticks=ticks,
        signals=signals,
        pip_size=pip_size,
        sl_pips=sl_arr,
        tp_pips=tp_arr,
        use_slippage=True,
        close_at_end=True,
        commission_pips=comm,
        entry_slippage_pips=slip,
        min_entry_timestamp=min_entry_ts,
        enforce_subsequent_tick=True
    )
    return tr


def search_symbol(symbol: str):
    print(f"\n=======================================================")
    print(f"  Searching Clean Macro Retest Champions for {symbol}")
    print(f"=======================================================")
    comm, slip = get_friction_pips(symbol)
    hold_weeks = get_complete_split_weeks(symbol, "holdout_test_30")
    disc_weeks = get_complete_split_weeks(symbol, "discovery_70")
    print(f"Loaded {len(hold_weeks)} holdout weeks and {len(disc_weeks)} discovery weeks.")

    # Precompute weeks in parallel
    print("Precomputing bar structures for Holdout...")
    with ProcessPoolExecutor(max_workers=4) as ex:
        hold_pre = [w for w in ex.map(precompute_week_data, [symbol]*len(hold_weeks), hold_weeks) if w is not None]
    print(f"Precomputed {len(hold_pre)} valid holdout weeks.")

    # Define candidate grid based on symbol volatility profile
    pip_size = PIP_SIZES[symbol]
    is_jpy = "JPY" in symbol
    
    if symbol in ["AUDCAD", "NZDCAD"]:
        sl_tp_pairs = [(12.0, 24.0), (14.0, 28.0), (14.0, 35.0), (16.0, 32.0), (16.0, 40.0), (18.0, 36.0)]
        lookbacks = [10, 12, 14, 16]
        dists = [4.0, 5.0, 6.0]
        wicks = [0.40, 0.45]
        sessions = [1, 2] # 1: Asian+London, 2: London+NY
        max_spread = 2.6
    elif symbol in ["EURNZD", "GBPJPY"]:
        if is_jpy:
            sl_tp_pairs = [(25.0, 50.0), (30.0, 60.0), (30.0, 75.0), (35.0, 70.0), (40.0, 80.0)]
            dists = [8.0, 10.0, 12.0]
            max_spread = 3.5
        else: # EURNZD
            sl_tp_pairs = [(22.0, 44.0), (25.0, 50.0), (25.0, 60.0), (28.0, 56.0), (30.0, 65.0)]
            dists = [8.0, 10.0, 12.0]
            max_spread = 3.8
        lookbacks = [10, 12, 14]
        wicks = [0.40, 0.45]
        sessions = [1, 2, 3]
    elif symbol in ["USDCAD", "EURGBP"]:
        sl_tp_pairs = [(10.0, 20.0), (12.0, 24.0), (12.0, 30.0), (14.0, 28.0), (15.0, 30.0)]
        lookbacks = [10, 12, 14, 16]
        dists = [3.5, 4.5, 5.5]
        wicks = [0.40, 0.45]
        sessions = [2, 3]
        max_spread = 2.4
    else:
        sl_tp_pairs = [(14.0, 28.0), (16.0, 32.0)]
        lookbacks = [10, 12]
        dists = [5.0]
        wicks = [0.45]
        sessions = [1, 2]
        max_spread = 2.5

    candidates_passed = []

    for sl, tp in sl_tp_pairs:
        for lb in lookbacks:
            for dst in dists:
                for wk in wicks:
                    for sm in sessions:
                        # Evaluate on Holdout first
                        hold_trades = []
                        for wd in hold_pre:
                            tr = eval_config_on_week(wd, sl, tp, lb, dst, wk, sm, max_spread, comm, slip)
                            if len(tr) > 0:
                                hold_trades.append(tr)
                        if not hold_trades:
                            continue
                        comb_h = np.concatenate(hold_trades)
                        m_h = compute_statistical_metrics(comb_h)

                        # Check Holdout hurdles: PF >= 1.15, Trades >= 15, Exp >= 1.25, Net pips > 0
                        if m_h['net_pips'] > 0 and m_h['profit_factor'] >= 1.15 and m_h['total_trades'] >= 15 and m_h['expectancy_pips'] >= 1.25 and not m_h['outlier_sensitivity']['is_outlier_dependent']:
                            candidates_passed.append({
                                'symbol': symbol,
                                'sl': sl, 'tp': tp, 'lb': lb, 'dist': dst, 'wick': wk, 'session': sm,
                                'holdout': m_h
                            })
                            print(f"  [CANDIDATE FOUND] SL/TP: {sl}/{tp} | LB: {lb} | Dist: {dst} | Wick: {wk} | Sm: {sm} -> Holdout: Net {m_h['net_pips']:>+6.1f}p | PF {m_h['profit_factor']:.2f} | Tr: {m_h['total_trades']} | Exp: {m_h['expectancy_pips']:>+5.2f}p")

    print(f"\nTotal holdout-passing candidates for {symbol}: {len(candidates_passed)}")
    if not candidates_passed:
        return []

    # Now precompute Discovery weeks (first 100 weeks is enough to verify In-Sample edge)
    print("Precomputing sample Discovery weeks (Weeks 1..100) to confirm in-sample alpha...")
    sample_disc_weeks = disc_weeks[:100]
    with ProcessPoolExecutor(max_workers=4) as ex:
        disc_pre = [w for w in ex.map(precompute_week_data, [symbol]*len(sample_disc_weeks), sample_disc_weeks) if w is not None]

    fully_verified = []
    for c in candidates_passed:
        disc_trades = []
        for wd in disc_pre:
            tr = eval_config_on_week(wd, c['sl'], c['tp'], c['lb'], c['dist'], c['wick'], c['session'], max_spread, comm, slip)
            if len(tr) > 0:
                disc_trades.append(tr)
        if not disc_trades:
            continue
        comb_d = np.concatenate(disc_trades)
        m_d = compute_statistical_metrics(comb_d)

        if m_d['net_pips'] > 0 and m_d['profit_factor'] >= 1.15 and m_d['total_trades'] >= 25:
            c['discovery'] = m_d
            fully_verified.append(c)
            print(f"  *** DUAL-CERTIFIED CHAMPION *** SL/TP: {c['sl']}/{c['tp']} | LB: {c['lb']} | Dist: {c['dist']} | Wick: {c['wick']} | Sm: {c['session']}")
            print(f"      Holdout (90w):   Net: {c['holdout']['net_pips']:>+6.1f}p | PF: {c['holdout']['profit_factor']:.2f} | Tr: {c['holdout']['total_trades']} | Exp: {c['holdout']['expectancy_pips']:>+5.2f}p")
            print(f"      Disc (100w):     Net: {m_d['net_pips']:>+6.1f}p | PF: {m_d['profit_factor']:.2f} | Tr: {m_d['total_trades']} | Exp: {m_d['expectancy_pips']:>+5.2f}p")

    return fully_verified


if __name__ == "__main__":
    import json
    results = {}
    for sym in ["AUDCAD", "NZDCAD", "EURNZD", "USDCAD"]:
        champs = search_symbol(sym)
        results[sym] = champs
        with open("results/clean_macro_champions_found.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
