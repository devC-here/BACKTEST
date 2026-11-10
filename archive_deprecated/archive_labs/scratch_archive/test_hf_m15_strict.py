#!/usr/bin/env python3
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT

class HighFrequencyM15ScalperEliteStrict(BaseStrategy):
    def __init__(self, target_rr: float = 2.2, min_sl: float = 6.0, max_sl: float = 18.0, **kwargs):
        super().__init__(**kwargs)
        self.target_rr = target_rr
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.allowed_hours = {7, 8, 9, 11, 12, 13, 14, 15, 17}
        self.lookback = 16
        self.pip_size = 0.0001

    def generate_signals(self, ticks: np.ndarray):
        n_ticks = len(ticks)
        signals = np.zeros(n_ticks, dtype=np.int8)
        sl_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        tp_pips_arr = np.zeros(n_ticks, dtype=np.float64)
        if n_ticks < 500:
            return signals, sl_pips_arr, tp_pips_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base_df = pd.DataFrame({'open': mid, 'high': mid, 'low': mid, 'close': mid}, index=ts)

        m15 = base_df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(m15) < 40:
            return signals, sl_pips_arr, tp_pips_arr

        # STRICT ZERO-LOOKAHEAD H1: shift(1)
        h1 = base_df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        h1['h1_ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
        h1['h1_trend'] = np.where(h1['close'] > h1['h1_ema50'], 1, -1)
        m15['macro_trend'] = h1['h1_trend'].shift(1).reindex(m15.index, method='ffill').fillna(0)

        tr = np.maximum(m15['high'] - m15['low'], np.maximum(abs(m15['high'] - m15['close'].shift(1)), abs(m15['low'] - m15['close'].shift(1))))
        m15['atr'] = tr.rolling(14).mean().bfill()
        m15['ema_20'] = m15['close'].ewm(span=20, adjust=False).mean()

        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(25, len(m15)):
            row = m15.iloc[i]
            prev = m15.iloc[i - 1]
            prior = m15.iloc[i - 2]

            hour = row.name.hour
            weekday = row.name.dayofweek
            if hour not in self.allowed_hours:
                continue
            if weekday == 4 and hour >= 16:
                continue

            sig = 0
            risk_p = 0.0
            macro = row['macro_trend']
            atr_pips = row['atr'] / self.pip_size
            dyn_min_sl = max(self.min_sl, 0.8 * atr_pips)
            dyn_max_sl = max(self.max_sl, 1.8 * atr_pips)

            # Pattern A: 4-Hour Rolling Liquidity Sweep
            window = m15.iloc[i - self.lookback : i]
            roll_h = window['high'].max()
            roll_l = window['low'].min()

            sweep_l = (roll_l - row['low']) / self.pip_size
            if 0.8 <= sweep_l <= 6.0 and row['close'] > roll_l and row['close'] > row['open']:
                if macro >= 0:
                    sl_p = row['low'] - 1.2 * self.pip_size
                    r = max((row['close'] - sl_p) / self.pip_size, dyn_min_sl)
                    if r <= dyn_max_sl:
                        sig = SIGNAL_LONG
                        risk_p = r

            sweep_h = (row['high'] - roll_h) / self.pip_size
            if 0.8 <= sweep_h <= 6.0 and row['close'] < roll_h and row['close'] < row['open']:
                if macro <= 0:
                    sl_p = row['high'] + 1.2 * self.pip_size
                    r = max((sl_p - row['close']) / self.pip_size, dyn_min_sl)
                    if r <= dyn_max_sl:
                        sig = SIGNAL_SHORT
                        risk_p = r

            # Pattern B: Fair Value Gap Immediate Mitigation
            if sig == 0:
                if prev['close'] > prev['open'] and (prev['close'] - prev['open']) / self.pip_size >= 3.5:
                    if row['open'] > prior['high'] and row['low'] <= prior['high'] and row['close'] > prior['high'] and row['close'] > row['ema_20']:
                        if macro >= 0:
                            sl_p = prev['low'] - 1.2 * self.pip_size
                            r = max((row['close'] - sl_p) / self.pip_size, dyn_min_sl)
                            if r <= dyn_max_sl:
                                sig = SIGNAL_LONG
                                risk_p = r

                elif prev['close'] < prev['open'] and (prev['open'] - prev['close']) / self.pip_size >= 3.5:
                    if row['open'] < prior['low'] and row['high'] >= prior['low'] and row['close'] < prior['low'] and row['close'] < row['ema_20']:
                        if macro <= 0:
                            sl_p = prev['high'] + 1.2 * self.pip_size
                            r = max((sl_p - row['close']) / self.pip_size, dyn_min_sl)
                            if r <= dyn_max_sl:
                                sig = SIGNAL_SHORT
                                risk_p = r

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n_ticks and signals[idx] == 0:
                    signals[idx] = sig
                    sl_pips_arr[idx] = risk_p
                    tp_pips_arr[idx] = risk_p * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr

def run_week(w: int):
    ticks, meta = get_week_dataset(w)
    engine = BacktestEngine(pip_size=0.0001, use_slippage=True)
    strat = HighFrequencyM15ScalperEliteStrict()
    trades = engine.run(ticks, strat)
    return [float(t["pips"]) for t in trades]

def main():
    print("Testing HF_M15 with STRICT H1 shift(1)...")
    with ProcessPoolExecutor() as executor:
        week_res = list(executor.map(run_week, range(1, 53)))
    all_pips = []
    for r in week_res:
        all_pips.extend(r)
    pips = np.array(all_pips)
    wins = pips[pips > 0]
    losses = pips[pips < 0]
    wr = len(wins) / len(pips) * 100 if len(pips) > 0 else 0
    pf = np.sum(wins) / abs(np.sum(losses)) if len(losses) > 0 else 0
    rr = np.mean(wins) / abs(np.mean(losses)) if len(losses) > 0 else 0
    print(f"STRICT HF_M15: Trades={len(pips)}, WR={wr:.1f}%, Realized RR={rr:.2f}, PF={pf:.2f}, Net Pips={np.sum(pips):+.1f}")

if __name__ == "__main__":
    main()
