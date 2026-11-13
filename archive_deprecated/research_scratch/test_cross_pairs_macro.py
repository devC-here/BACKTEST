import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset, get_complete_split_weeks
from tick_engine.core.config import PIP_SIZES, get_friction_pips, WARMUP_TICKS
from tick_engine.reporting.run_manifest import compute_statistical_metrics
from data_engineering_suite.bars import build_time_bars


def eval_macro_retest_pair(symbol, sl, tp, lookback, dist, min_wick, hrs, weeks):
    pip_size = PIP_SIZES[symbol]
    comm, slip = get_friction_pips(symbol)
    raw_dir = f"data/{symbol}"
    
    trades_list = []
    for w in weeks:
        ticks, _ = get_week_dataset(w, raw_dir, warmup_prev_week_ticks=WARMUP_TICKS)
        if len(ticks) <= WARMUP_TICKS + 500:
            continue
        
        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=pip_size)
        if len(m15) < 30:
            continue
            
        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid}, index=ts)
        h4 = base.resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(h4) < lookback + 5:
            continue
            
        h4['ema20'] = h4['close'].ewm(span=20, adjust=False).mean()
        h4['bias'] = np.where(h4['close'] > h4['ema20'], 1, -1)
        h4['roll_h'] = h4['high'].rolling(lookback).max().shift(1).bfill()
        h4['roll_l'] = h4['low'].rolling(lookback).min().shift(1).bfill()
        
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
        
        signals = np.zeros(len(ticks), dtype=np.int8)
        sl_arr = np.zeros(len(ticks), dtype=np.float64)
        tp_arr = np.zeros(len(ticks), dtype=np.float64)
        
        last_trade_date = None
        for i in range(15, len(m15)):
            hr = hours[i]
            if not (hrs[0] <= hr <= hrs[1] or (hrs[0] > hrs[1] and (hr >= hrs[0] or hr <= hrs[1]))):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > 2.5:
                continue
            bar_rng = (h[i] - l[i]) / pip_size
            if bar_rng < 2.5:
                continue
                
            b = bias_m15[i]
            sig = 0
            if b > 0:
                dist_supp = abs(l[i] - supp_m15[i]) / pip_size
                lower_wick = (min(o[i], c[i]) - l[i]) / pip_size
                if dist_supp <= dist and (lower_wick / bar_rng) >= min_wick and c[i] > o[i]:
                    sig = 1
            elif b < 0:
                dist_res = abs(res_m15[i] - h[i]) / pip_size
                upper_wick = (h[i] - max(o[i], c[i])) / pip_size
                if dist_res <= dist and (upper_wick / bar_rng) >= min_wick and c[i] < o[i]:
                    sig = -1
                    
            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < len(ticks) and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = sl
                    tp_arr[idx] = tp
                    last_trade_date = dt
                    
        tr = run_matching_loop(ticks, signals, pip_size, sl_arr, tp_arr, use_slippage=True, close_at_end=True, commission_pips=comm, entry_slippage_pips=slip, min_entry_timestamp=ticks['timestamp'][WARMUP_TICKS], enforce_subsequent_tick=True)
        if len(tr) > 0:
            trades_list.append(tr)
            
    comb = np.concatenate(trades_list) if trades_list else np.empty(0)
    return compute_statistical_metrics(comb)


def main():
    for sym in ["AUDCAD", "NZDCAD"]:
        hold_weeks = get_complete_split_weeks(sym, "holdout_test_30")
        disc_weeks = get_complete_split_weeks(sym, "discovery_70")
        print(f"\n--- Testing {sym} ({len(hold_weeks)} holdout weeks, {len(disc_weeks)} discovery weeks) ---")
        
        configs = [
            # (sl, tp, lookback, dist, min_wick, hrs)
            (14.0, 28.0, 10, 5.0, 0.40, (21, 16)),
            (12.0, 24.0, 12, 4.0, 0.45, (21, 16)),
            (16.0, 40.0, 14, 6.0, 0.40, (21, 16)),
            (15.0, 30.0, 16, 5.0, 0.45, (6, 17)),
            (12.0, 30.0, 10, 4.0, 0.45, (6, 17)),
        ]
        
        for sl, tp, lb, dst, wick, hrs in configs:
            h_res = eval_macro_retest_pair(sym, sl, tp, lb, dst, wick, hrs, hold_weeks)
            d_res = eval_macro_retest_pair(sym, sl, tp, lb, dst, wick, hrs, disc_weeks[:70])
            print(f"SL/TP: {sl}/{tp} | LB: {lb} | Dist: {dst} | Wick: {wick} | Hrs: {hrs}")
            print(f"  Holdout:   Net: {h_res['net_pips']:>+6.1f}p | PF: {h_res['profit_factor']:.2f} | Tr: {h_res['total_trades']:>2} | Exp: {h_res['expectancy_pips']:>+5.2f}p")
            print(f"  Discovery: Net: {d_res['net_pips']:>+6.1f}p | PF: {d_res['profit_factor']:.2f} | Tr: {d_res['total_trades']:>2} | Exp: {d_res['expectancy_pips']:>+5.2f}p")


if __name__ == "__main__":
    main()
