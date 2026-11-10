#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V18: HEIKIN-ASHI SURGE, PRE-FIX FADE, CMF EXPANSION & OPEN GAP
================================================================================
Evaluates 4 Quantitative Microstructure Architectures across 70% Discovery:
  1. DISC86_HEIKIN_ASHI_SURGE:
     M15 Heikin-Ashi calculation.
     3 consecutive directional HA bars with 0 counter-wick (strong institutional impulse).
     H4 20-EMA trend agreement. Active: 07:00 - 16:30 UTC. 2.0R target.
  2. DISC87_LONDON_PRE_FIX_EXHAUSTION:
     14:15 - 15:30 UTC window preceding the 16:00 UTC WMR London Fix.
     Intraday extension > 1.8x ATR. 2 consecutive M15 rejection candles against trend.
     Target: 1.5R.
  3. DISC88_CMF_MOMENTUM_EXPANSION:
     Chaikin Money Flow (20-period) on M15.
     CMF crosses above +0.10 in H4 uptrend or below -0.10 in H4 downtrend with expansion bar.
     Active: 07:00 - 16:00 UTC. 2.0R target.
  4. DISC89_EUROPEAN_OPEN_GAP_FADE:
     07:00 UTC European open gap relative to Asian close (06:45 UTC).
     Gap >= 4.0 pips. Enters fade towards gap fill during 07:00 - 08:30 UTC.
     Target: 1.5R.

Tested across EURUSD, EURGBP, and USDJPY on 70% Discovery Partition ONLY.
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine
from tick_engine.core.dataloader import get_week_dataset, get_discovery_weeks
from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT
from data_engineering_suite.bars import build_time_bars

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Model 1: Heikin-Ashi Institutional Impulse Surge (DISC86)
# ==============================================================================
class HeikinAshiImpulseSurge(BaseStrategy):
    """
    Operates during 07:00 - 16:30 UTC.
    Calculates Heikin-Ashi bars on M15.
    Bullish Impulse: 3 consecutive green HA bars where HA_Low == HA_Open (zero lower shadow).
    Bearish Impulse: 3 consecutive red HA bars where HA_High == HA_Open (zero upper shadow).
    H4 20-EMA trend agreement.
    Target: 2.0R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Heikin_Ashi_Surge"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m15) < 35 or len(h4) < 10:
            return signals, sl_arr, tp_arr

        h4_c = pd.Series(h4['close'].values)
        h4_ema20 = h4_c.ewm(span=20, adjust=False).mean().values
        h4_bias = np.where(h4_c.values > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'h4_bias': h4_bias}, index=h4.index)

        m15_df = pd.merge_asof(
            m15.sort_index(),
            h4_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )

        c = m15_df['close'].values
        h = m15_df['high'].values
        l = m15_df['low'].values
        o = m15_df['open'].values
        spread = m15_df['spread_pips'].values
        h4_trend = m15_df['h4_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        m = len(m15_df)
        ha_close = (o + h + l + c) / 4.0
        ha_open = np.zeros(m)
        ha_open[0] = (o[0] + c[0]) / 2.0
        for j in range(1, m):
            ha_open[j] = (ha_open[j-1] + ha_close[j-1]) / 2.0

        ha_high = np.maximum(h, np.maximum(ha_open, ha_close))
        ha_low = np.minimum(l, np.minimum(ha_open, ha_close))

        tol = 0.5 * self.pip_size
        last_trade_date = None

        for i in range(5, m):
            dt = dates[i]
            hr = hours[i]
            if not (7 <= hr <= 16):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            sig = 0
            # 3 consecutive green bars with no lower shadow
            if macro == 1:
                green3 = (ha_close[i] > ha_open[i]) and (ha_close[i-1] > ha_open[i-1]) and (ha_close[i-2] > ha_open[i-2])
                no_lower3 = (ha_open[i] - ha_low[i] <= tol) and (ha_open[i-1] - ha_low[i-1] <= tol)
                if green3 and no_lower3:
                    sig = SIGNAL_LONG
            # 3 consecutive red bars with no upper shadow
            elif macro == -1:
                red3 = (ha_close[i] < ha_open[i]) and (ha_close[i-1] < ha_open[i-1]) and (ha_close[i-2] < ha_open[i-2])
                no_upper3 = (ha_high[i] - ha_open[i] <= tol) and (ha_high[i-1] - ha_open[i-1] <= tol)
                if red3 and no_upper3:
                    sig = SIGNAL_SHORT

            if sig != 0:
                bar_close_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, bar_close_ms)
                if idx < n:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 2: London Pre-Fix Exhaustion Fade (DISC87)
# ==============================================================================
class LondonPreFixExhaustionFade(BaseStrategy):
    """
    Operates during 14:15 - 15:30 UTC.
    Detects if day's move from 07:00 UTC exceeds 1.8x ATR.
    Triggers fade on rejection wick >= 45% back towards day's midpoint.
    Target: 1.5R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Pre_Fix_Exhaustion"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 30:
            return signals, sl_arr, tp_arr

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        minutes = m15.index.minute.values
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # ATR 14
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        tr = np.insert(tr, 0, h[0] - l[0])
        atr = pd.Series(tr).rolling(14).mean().values

        last_trade_date = None
        for i in range(20, len(m15)):
            dt = dates[i]
            hr = hours[i]
            mn = minutes[i]

            in_window = (hr == 14 and mn >= 15) or (hr == 15 and mn <= 30)
            if not in_window:
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            # Day range from 07:00 UTC
            day_mask = (dates == dt) & (hours >= 7) & (np.arange(len(m15)) <= i)
            if np.sum(day_mask) < 8:
                continue

            day_hi = np.max(h[day_mask])
            day_lo = np.min(l[day_mask])
            day_open = o[np.where(day_mask)[0][0]]
            day_rng = day_hi - day_lo

            if np.isnan(atr[i]) or day_rng < 1.8 * atr[i] * 4:  # roughly daily ATR
                continue

            bar_rng = h[i] - l[i]
            if bar_rng < 3.0 * self.pip_size:
                continue

            sig = 0
            # Overextended up: price at day high, prints bearish pin bar
            if abs(h[i] - day_hi) <= 2.0 * self.pip_size and c[i] < o[i]:
                upper_wick = (h[i] - o[i]) / bar_rng
                if upper_wick >= 0.45:
                    sig = SIGNAL_SHORT
            # Overextended down: price at day low, prints bullish pin bar
            elif abs(l[i] - day_lo) <= 2.0 * self.pip_size and c[i] > o[i]:
                lower_wick = (o[i] - l[i]) / bar_rng
                if lower_wick >= 0.45:
                    sig = SIGNAL_LONG

            if sig != 0:
                bar_close_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, bar_close_ms)
                if idx < n:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 3: Chaikin Money Flow Expansion (DISC88)
# ==============================================================================
class CMFMomentumExpansion(BaseStrategy):
    """
    Operates during 07:00 - 16:00 UTC.
    Calculates Chaikin Money Flow (20-period) on M15.
    H4 20-EMA trend agreement.
    CMF crosses above +0.10 (bullish volume surge) or below -0.10 (bearish volume surge).
    Target: 2.0R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_CMF_Expansion"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        h4 = build_time_bars(ticks, timeframe_min=240, pip_size=self.pip_size)
        if len(m15) < 35 or len(h4) < 10:
            return signals, sl_arr, tp_arr

        h4_c = pd.Series(h4['close'].values)
        h4_ema20 = h4_c.ewm(span=20, adjust=False).mean().values
        h4_bias = np.where(h4_c.values > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'h4_bias': h4_bias}, index=h4.index)

        m15_df = pd.merge_asof(
            m15.sort_index(),
            h4_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )

        c = m15_df['close'].values
        h = m15_df['high'].values
        l = m15_df['low'].values
        spread = m15_df['spread_pips'].values
        h4_trend = m15_df['h4_bias'].fillna(0).values.astype(int)
        hours = m15_df.index.hour.values
        dates = m15_df.index.date
        time_ms = m15_df.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Chaikin Money Flow: Money Flow Multiplier * Volume
        # MFM = ((c - l) - (h - c)) / (h - l)
        hl = np.maximum(h - l, 1e-6)
        mfm = ((c - l) - (h - c)) / hl
        vol = m15_df['tick_count'].values if 'tick_count' in m15_df.columns else np.ones(len(m15_df))
        mfv = mfm * vol
        cmf = pd.Series(mfv).rolling(20).sum() / pd.Series(vol).rolling(20).sum()
        cmf_val = cmf.values

        last_trade_date = None
        for i in range(25, len(m15_df)):
            dt = dates[i]
            hr = hours[i]
            if not (7 <= hr <= 16):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0 or np.isnan(cmf_val[i]) or np.isnan(cmf_val[i-1]):
                continue

            sig = 0
            if macro == 1 and cmf_val[i-1] <= 0.10 and cmf_val[i] > 0.10:
                sig = SIGNAL_LONG
            elif macro == -1 and cmf_val[i-1] >= -0.10 and cmf_val[i] < -0.10:
                sig = SIGNAL_SHORT

            if sig != 0:
                bar_close_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, bar_close_ms)
                if idx < n:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 4: European Open Gap Fade (DISC89)
# ==============================================================================
class EuropeanOpenGapFade(BaseStrategy):
    """
    Operates during 07:00 - 08:30 UTC.
    Identifies gap between 07:00 open and 06:45 close.
    If gap >= 4.0 pips, enters mean-reversion fade towards gap fill.
    Target: 1.5R.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_European_Gap_Fade"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 30:
            return signals, sl_arr, tp_arr

        c = m15['close'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        minutes = m15.index.minute.values
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        last_trade_date = None
        for i in range(1, len(m15)):
            dt = dates[i]
            hr = hours[i]
            mn = minutes[i]

            # Trigger at 07:00 or 07:15 UTC
            if not ((hr == 7 and (mn == 0 or mn == 15))):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            # Gap relative to previous bar close
            gap_pips = (o[i] - c[i-1]) / self.pip_size
            if abs(gap_pips) < 4.0:
                continue

            sig = 0
            if gap_pips >= 4.0:
                # Gapped UP: Fade SHORT
                sig = SIGNAL_SHORT
            elif gap_pips <= -4.0:
                # Gapped DOWN: Fade LONG
                sig = SIGNAL_LONG

            if sig != 0:
                bar_close_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, bar_close_ms)
                if idx < n:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Discovery Worker for Alpha Discovery Lab V18
# ==============================================================================
def evaluate_v18_discovery_week(task: Tuple[str, int]) -> Dict[str, Any]:
    asset, week_num = task
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if ticks is None or len(ticks) < 500:
        return {"asset": asset, "week": week_num, "trades": {}}

    models = []
    if asset == "EURUSD":
        models.append(("HEIKIN_ASHI_SURGE", HeikinAshiImpulseSurge(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("PRE_FIX_FADE", LondonPreFixExhaustionFade(asset=asset, sl_pips=9.0, tp_pips=13.5, pip_size=0.0001, max_spread=0.6)))
        models.append(("CMF_EXPANSION", CMFMomentumExpansion(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("OPEN_GAP_FADE", EuropeanOpenGapFade(asset=asset, sl_pips=9.0, tp_pips=13.5, pip_size=0.0001, max_spread=0.6)))
    elif asset == "USDJPY":
        models.append(("HEIKIN_ASHI_SURGE", HeikinAshiImpulseSurge(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
        models.append(("PRE_FIX_FADE", LondonPreFixExhaustionFade(asset=asset, sl_pips=12.0, tp_pips=18.0, pip_size=0.01, max_spread=0.8)))
        models.append(("CMF_EXPANSION", CMFMomentumExpansion(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
        models.append(("OPEN_GAP_FADE", EuropeanOpenGapFade(asset=asset, sl_pips=12.0, tp_pips=18.0, pip_size=0.01, max_spread=0.8)))
    elif asset == "EURGBP":
        models.append(("HEIKIN_ASHI_SURGE", HeikinAshiImpulseSurge(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)))
        models.append(("PRE_FIX_FADE", LondonPreFixExhaustionFade(asset=asset, sl_pips=8.0, tp_pips=12.0, pip_size=0.0001, max_spread=0.8)))
        models.append(("CMF_EXPANSION", CMFMomentumExpansion(asset=asset, sl_pips=8.0, tp_pips=16.0, pip_size=0.0001, max_spread=0.8)))
        models.append(("OPEN_GAP_FADE", EuropeanOpenGapFade(asset=asset, sl_pips=8.0, tp_pips=12.0, pip_size=0.0001, max_spread=0.8)))

    week_trades = {}
    for m_name, model in models:
        try:
            raw_trades = engine.run(ticks, model)
            t_list = []
            for i in range(len(raw_trades)):
                row = raw_trades[i]
                t_list.append({
                    "strategy": m_name,
                    "asset": asset,
                    "week": week_num,
                    "pips": float(row["pips"]),
                    "direction": int(row["direction"])
                })
            week_trades[m_name] = t_list
        except Exception:
            week_trades[m_name] = []

    return {"asset": asset, "week": week_num, "trades": week_trades}


def run_discovery_v18():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("  LAUNCHING ALPHA DISCOVERY V18 EXPEDITION (70% DISCOVERY PARTITIONS)", flush=True)
    print("=" * 80, flush=True)

    assets = [os.environ["DISCOVERY_TARGET_SYMBOL"]] if "DISCOVERY_TARGET_SYMBOL" in os.environ else ["USDJPY", "EURGBP", "EURUSD"]
    all_tasks = []
    for asset in assets:
        disc_weeks = get_discovery_weeks(f"data/{asset}")
        print(f"[*] Queued {len(disc_weeks)} discovery weeks for {asset}...", flush=True)
        for w in disc_weeks:
            all_tasks.append((asset, w))

    print(f"\n[*] Total discovery tasks across all assets: {len(all_tasks)} weeks", flush=True)

    results_by_key: Dict[Tuple[str, str], List[float]] = {}
    completed = 0

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(evaluate_v18_discovery_week, t): t for t in all_tasks}
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            asset = res["asset"]
            for m_name, t_list in res["trades"].items():
                key = (asset, m_name)
                if key not in results_by_key:
                    results_by_key[key] = []
                results_by_key[key].extend([t["pips"] for t in t_list])

            if completed % 100 == 0 or completed == len(all_tasks):
                print(f"  [+] Processed {completed:3d}/{len(all_tasks)} discovery weeks in {time.time() - t_start:.1f}s...", flush=True)

    # Scorecard
    print("\n" + "=" * 90, flush=True)
    print(f"{'ASSET':<8} {'MODEL NAME':<26} {'TRADES':<8} {'WIN RATE':<10} {'PROFIT FACTOR':<15} {'NET PIPS':<12} {'STATUS'}", flush=True)
    print("=" * 90, flush=True)

    summary_rows = []
    for (asset, m_name), pips in sorted(results_by_key.items()):
        n_t = len(pips)
        if n_t == 0:
            print(f"{asset:<8} {m_name:<26} {'0':<8} {'0.0%':<10} {'0.00':<15} {'+0.0p':<12} {'NO TRADES'}", flush=True)
            continue

        wins = [p for p in pips if p > 0]
        losses = [p for p in pips if p < 0]
        wr = len(wins) / n_t * 100.0
        gross_win = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        pf = gross_win / gross_loss if gross_loss > 0 else 999.0
        net = sum(pips)

        status = "REJECTED"
        if pf >= 1.25 and net >= 500.0 and n_t >= 250:
            status = "PASS_HOLDOUT_CANDIDATE"

        summary_rows.append({
            "asset": asset,
            "model": m_name,
            "trades": n_t,
            "win_rate": round(wr, 1),
            "profit_factor": round(pf, 2),
            "net_pips": round(net, 1),
            "status": status
        })

        sign = "+" if net >= 0 else ""
        print(f"{asset:<8} {m_name:<26} {n_t:<8} {wr:5.1f}%    {pf:<15.2f} {sign}{net:.1f}p       {status}", flush=True)

    print("=" * 90, flush=True)
    elapsed = time.time() - t_start
    print(f"[*] Discovery Lab V18 complete in {elapsed:.2f} seconds.", flush=True)

    out_file = RESULTS_DIR / "alpha_discovery_lab_v18_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"[*] Results saved to {out_file}", flush=True)


if __name__ == "__main__":
    run_discovery_v18()
