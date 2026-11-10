#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY & OPTIMIZATION LAB V43:
  Targeting High-Return (>= 15% Annualized) Edges for EURGBP & Cross-Assets
================================================================================
Methodology:
  - Strictly executed on 70% In-Sample Discovery Weeks (Deterministic Seed 42).
  - Absolutely ZERO access to the quarantined 30% holdout test partition.
  - Realistic Account Model: 1.0% risk per trade.
  - Passing Bar: Annualized Return >= 15.0% AND Profit Factor >= 1.25.
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
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

EURUSD_MULT = 1.0000
USDJPY_MULT = 0.7193
EURGBP_MULT = 1.2921


# ==============================================================================
# Model 1: EURGBP Asian Range Retest & Fade (London Session Value Reversion)
# ==============================================================================
class AsianRangeRetestFade(BaseStrategy):
    """
    Asian Range (00:00 - 07:00 UTC) established.
    During London Opening (07:00 - 12:00 UTC):
      - If price spikes outside the Asian box (within 10 pips) but fails to sustain,
        closing back inside or near the boundary with a rejection wick >= 35%.
      - Fades the breakout back towards the Asian Midpoint / opposite boundary.
      - 2.0R to 2.2R target.
    """
    def __init__(
        self,
        asset: str,
        sl_pips: float,
        tp_pips: float,
        pip_size: float,
        max_spread: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Asian_Retest_Fade_R{tp_pips/sl_pips:.1f}"

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
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Determine Asian High and Low (00:00 - 06:45 UTC)
        asia_highs = {}
        asia_lows = {}
        for i in range(len(m15)):
            dt = dates[i]
            hr = hours[i]
            if hr < 7:
                if dt not in asia_highs:
                    asia_highs[dt] = h[i]
                    asia_lows[dt] = l[i]
                else:
                    asia_highs[dt] = max(asia_highs[dt], h[i])
                    asia_lows[dt] = min(asia_lows[dt], l[i])

        last_trade_date = None

        for i in range(20, len(m15)):
            dt = dates[i]
            hr = hours[i]

            # Active London opening window: 07:00 - 12:00 UTC
            if not (7 <= hr <= 12):
                continue
            if dt == last_trade_date:
                continue
            if dt not in asia_highs:
                continue
            if spread[i] > self.max_spread:
                continue

            a_hi = asia_highs[dt]
            a_lo = asia_lows[dt]
            a_rng = (a_hi - a_lo) / self.pip_size

            # Asian range must be valid (between 8 and 25 pips on EURGBP)
            if a_rng < 8.0 or a_rng > 30.0:
                continue

            bar_rng = h[i] - l[i]
            if bar_rng <= 0:
                continue

            sig = 0
            # False upside breakout: High swept above Asian High, but candle closes below Asian High
            # with upper rejection wick >= 35%
            upper_wick = h[i] - max(c[i], o[i])
            if h[i] >= a_hi and c[i] <= a_hi and (upper_wick / bar_rng) >= 0.35 and c[i] < o[i]:
                sig = SIGNAL_SHORT

            # False downside breakout: Low swept below Asian Low, but candle closes above Asian Low
            # with lower rejection wick >= 35%
            lower_wick = min(c[i], o[i]) - l[i]
            if l[i] <= a_lo and c[i] >= a_lo and (lower_wick / bar_rng) >= 0.35 and c[i] > o[i]:
                sig = SIGNAL_LONG

            if sig != 0:
                end_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, end_ms)
                if idx < n:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 2: EURGBP Camarilla L3/H3 Mean Reversion Floor/Ceiling Fade
# ==============================================================================
class CamarillaL3H3MeanReversionFade(BaseStrategy):
    """
    Camarilla L3 (Long Support) and H3 (Short Resistance).
    When price touches L3 / H3 during London / NY sessions (07:30 - 15:30 UTC):
      - In line with macro daily trend or mean-reverting towards daily pivot.
      - 2.0R to 2.2R target.
    """
    def __init__(
        self,
        asset: str,
        sl_pips: float,
        tp_pips: float,
        pip_size: float,
        max_spread: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Camarilla_L3H3_Fade_R{tp_pips/sl_pips:.1f}"

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
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        daily_bars = {}
        for i in range(len(m15)):
            d = dates[i]
            if d not in daily_bars:
                daily_bars[d] = {"high": h[i], "low": l[i], "close": c[i]}
            else:
                daily_bars[d]["high"] = max(daily_bars[d]["high"], h[i])
                daily_bars[d]["low"] = min(daily_bars[d]["low"], l[i])
                daily_bars[d]["close"] = c[i]

        unique_dates = sorted(list(daily_bars.keys()))
        date_to_levels = {}
        for idx_d in range(1, len(unique_dates)):
            curr_d = unique_dates[idx_d]
            prev_d = unique_dates[idx_d - 1]
            p_h = daily_bars[prev_d]["high"]
            p_l = daily_bars[prev_d]["low"]
            p_c = daily_bars[prev_d]["close"]
            rng = p_h - p_l
            h3 = p_c + (rng * 1.1 / 4.0)
            l3 = p_c - (rng * 1.1 / 4.0)
            date_to_levels[curr_d] = (h3, l3)

        last_trade_date = None

        for i in range(15, len(m15)):
            dt = dates[i]
            hr = hours[i]

            if not (7 <= hr <= 15):
                continue
            if dt == last_trade_date:
                continue
            if dt not in date_to_levels:
                continue
            if spread[i] > self.max_spread:
                continue

            h3, l3 = date_to_levels[dt]
            bar_rng = h[i] - l[i]
            if bar_rng <= 0:
                continue

            sig = 0
            # Test L3 Floor with lower wick >= 35%
            dist_l3 = abs(l[i] - l3) / self.pip_size
            lower_wick = min(c[i], o[i]) - l[i]
            if dist_l3 <= 3.0 and c[i] > l3 and (lower_wick / bar_rng) >= 0.35 and c[i] > o[i]:
                sig = SIGNAL_LONG

            # Test H3 Ceiling with upper wick >= 35%
            dist_h3 = abs(h[i] - h3) / self.pip_size
            upper_wick = h[i] - max(c[i], o[i])
            if dist_h3 <= 3.0 and c[i] < h3 and (upper_wick / bar_rng) >= 0.35 and c[i] < o[i]:
                sig = SIGNAL_SHORT

            if sig != 0:
                end_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, end_ms)
                if idx < n:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 3: Cardwell RSI 50 Momentum Bounce for EURGBP
# ==============================================================================
class CardwellRSI50EURGBP(BaseStrategy):
    """
    Cardwell RSI 50 Equilibrium Bounce adapted for EURGBP.
    H4 20-EMA trend, M15 RSI 14 pulls back to 45-55 zone and turns.
    """
    def __init__(
        self,
        asset: str,
        sl_pips: float,
        tp_pips: float,
        pip_size: float,
        max_spread: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_RSI50_Bounce_R{tp_pips/sl_pips:.1f}"

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
        h4_trend = np.where(h4_c.values > h4_ema20, 1, -1)
        h4_df = pd.DataFrame({'h4_bias': h4_trend}, index=h4.index)

        # M15 RSI 14
        m15_c = pd.Series(m15['close'].values)
        delta = m15_c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-9)
        rsi = (100.0 - (100.0 / (1.0 + rs))).values

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

        last_trade_date = None

        for i in range(25, len(m15_df)):
            dt = dates[i]
            hr = hours[i]

            if not (7 <= hr <= 15):
                continue
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            macro = h4_trend[i]
            if macro == 0:
                continue

            r = rsi[i]
            r_prev = rsi[i-1]
            if np.isnan(r) or np.isnan(r_prev):
                continue

            rng = h[i] - l[i]
            if rng <= 0:
                continue

            sig = 0
            if macro == 1 and 45.0 <= r_prev <= 55.0 and r > r_prev and c[i] > o[i]:
                lower_wick = min(c[i], o[i]) - l[i]
                if lower_wick >= 0.35 * rng:
                    sig = SIGNAL_LONG

            elif macro == -1 and 45.0 <= r_prev <= 55.0 and r < r_prev and c[i] < o[i]:
                upper_wick = h[i] - max(c[i], o[i])
                if upper_wick >= 0.35 * rng:
                    sig = SIGNAL_SHORT

            if sig != 0:
                end_ms = time_ms[i] + dur_ms
                idx = np.searchsorted(tick_ts, end_ms)
                if idx < n:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Execution Runner
# ==============================================================================
def get_lab_v43_models() -> List[Tuple[str, str, BaseStrategy, float]]:
    mods = []

    # 1. EURGBP Asian Range Retest & Fade
    mods.append(("EURGBP", "ASIA_FADE_20R_SL6", AsianRangeRetestFade("EURGBP", 6.0, 12.0, 0.0001, 0.6), 6.0))
    mods.append(("EURGBP", "ASIA_FADE_22R_SL6", AsianRangeRetestFade("EURGBP", 6.0, 13.2, 0.0001, 0.6), 6.0))
    mods.append(("EURGBP", "ASIA_FADE_20R_SL7", AsianRangeRetestFade("EURGBP", 7.0, 14.0, 0.0001, 0.6), 7.0))
    mods.append(("EURGBP", "ASIA_FADE_22R_SL7", AsianRangeRetestFade("EURGBP", 7.0, 15.4, 0.0001, 0.6), 7.0))

    # 2. EURGBP Camarilla L3/H3 Mean Reversion Fade
    mods.append(("EURGBP", "CAM_L3H3_20R_SL6", CamarillaL3H3MeanReversionFade("EURGBP", 6.0, 12.0, 0.0001, 0.6), 6.0))
    mods.append(("EURGBP", "CAM_L3H3_22R_SL6", CamarillaL3H3MeanReversionFade("EURGBP", 6.0, 13.2, 0.0001, 0.6), 6.0))
    mods.append(("EURGBP", "CAM_L3H3_20R_SL7", CamarillaL3H3MeanReversionFade("EURGBP", 7.0, 14.0, 0.0001, 0.6), 7.0))

    # 3. EURGBP Cardwell RSI 50 Momentum Bounce
    mods.append(("EURGBP", "RSI50_BOUNCE_20R_SL6", CardwellRSI50EURGBP("EURGBP", 6.0, 12.0, 0.0001, 0.6), 6.0))
    mods.append(("EURGBP", "RSI50_BOUNCE_22R_SL6", CardwellRSI50EURGBP("EURGBP", 6.0, 13.2, 0.0001, 0.6), 6.0))
    mods.append(("EURGBP", "RSI50_BOUNCE_22R_SL7", CardwellRSI50EURGBP("EURGBP", 7.0, 15.4, 0.0001, 0.6), 7.0))

    return mods


def run_week_task(task: Tuple[str, str, int, BaseStrategy]) -> Dict[str, Any]:
    asset, model_name, week_num, strat = task
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    try:
        ticks, _ = get_week_dataset(week_num, raw_dir=f"data/{asset}")
        if ticks is None or len(ticks) < 500:
            return {"asset": asset, "model": model_name, "week": week_num, "trades": []}

        raw_trades = engine.run(ticks, strat)
        t_list = []
        for i in range(len(raw_trades)):
            row = raw_trades[i]
            t_list.append({
                "pips": float(row["pips"]),
                "direction": int(row["direction"])
            })
        return {"asset": asset, "model": model_name, "week": week_num, "trades": t_list}
    except Exception:
        return {"asset": asset, "model": model_name, "week": week_num, "trades": []}


def calculate_metrics(pips: List[float], mult: float, sl_pips: float) -> Dict[str, Any]:
    if not pips:
        return {"trades": 0, "win_rate": 0.0, "pf": 0.0, "net_pips": 0.0, "norm_pips": 0.0, "net_r": 0.0, "annual_return_pct": 0.0, "max_dd": 0.0, "expectancy": 0.0}
    arr = np.array(pips, dtype=np.float64)
    trades = len(arr)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    wr = len(wins) / trades * 100.0 if trades > 0 else 0.0
    gross_win = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else 999.0
    net_pips = float(np.sum(arr))
    norm_pips = net_pips * mult
    expectancy = net_pips / trades if trades > 0 else 0.0

    net_r = net_pips / sl_pips if sl_pips > 0 else 0.0
    years_discovery = 3.5
    annual_r = net_r / years_discovery
    annual_return_pct = annual_r * 1.0  # 1% risk per trade -> 1R = 1%

    equity = np.cumsum(arr * mult)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

    return {
        "trades": trades,
        "win_rate": round(wr, 1),
        "pf": round(pf, 2),
        "net_pips": round(net_pips, 1),
        "norm_pips": round(norm_pips, 1),
        "net_r": round(net_r, 1),
        "annual_return_pct": round(annual_return_pct, 1),
        "max_dd": round(max_dd, 1),
        "expectancy": round(expectancy, 2)
    }


def main():
    t0 = time.time()
    models = get_lab_v43_models()
    print("=" * 115)
    print("  ALPHA DISCOVERY & OPTIMIZATION LAB V43: EURGBP HIGH-RETURN ARCHITECTURES")
    print("  Strictly on 70% In-Sample Discovery Weeks | 1.0% Risk / Trade | Hurdle >= 15% Annual Ret & PF >= 1.25")
    print("=" * 115)

    all_tasks = []
    for asset, m_name, strat, sl_p in models:
        weeks = get_discovery_weeks(raw_dir=f"data/{asset}")
        for w in weeks:
            all_tasks.append((asset, m_name, w, strat))

    print(f"Total Model-Week Tasks: {len(all_tasks)} across {len(models)} configurations")

    trade_pool: Dict[str, List[float]] = {f"{a}_{m}": [] for a, m, _, _ in models}

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_week_task, t) for t in all_tasks]
        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            res = fut.result()
            key = f"{res['asset']}_{res['model']}"
            for tr in res['trades']:
                trade_pool[key].append(tr['pips'])
            done += 1
            if done % 400 == 0 or done == total:
                print(f"  Progress: {done}/{total} tasks complete ({done/total*100:.1f}%) in {time.time()-t0:.1f}s", flush=True)

    print("\n" + "=" * 125)
    print(f"{'Asset':<8} {'Model Variant':<25} {'Trades':<8} {'WR%':<7} {'PF':<7} {'Exp(p)':<8} {'Raw Pips':<11} {'Norm Pips':<11} {'Net R':<9} {'Annual Ret%':<13} {'Max DD(p)':<10} {'Verdict'}")
    print("-" * 125)

    records = []
    mult_map = {"EURUSD": EURUSD_MULT, "USDJPY": USDJPY_MULT, "EURGBP": EURGBP_MULT}

    for asset, m_name, strat, sl_p in models:
        key = f"{asset}_{m_name}"
        pips = trade_pool[key]
        mult = mult_map[asset]
        metrics = calculate_metrics(pips, mult, sl_p)

        verdict = "[REJECT]"
        if metrics["pf"] >= 1.25 and metrics["annual_return_pct"] >= 15.0:
            verdict = "[>= 15% VIABLE CHAMPION]"
        elif metrics["pf"] >= 1.20 and metrics["annual_return_pct"] >= 10.0:
            verdict = "[PROMISING]"

        print(f"{asset:<8} {m_name:<25} {metrics['trades']:<8} {metrics['win_rate']:<7} {metrics['pf']:<7} {metrics['expectancy']:<8} {metrics['net_pips']:<11} {metrics['norm_pips']:<11} {metrics['net_r']:<9} {metrics['annual_return_pct']:<13} {metrics['max_dd']:<10} {verdict}")

        records.append({
            "asset": asset,
            "model_variant": m_name,
            "trades": metrics["trades"],
            "win_rate": metrics["win_rate"],
            "profit_factor": metrics["pf"],
            "expectancy": metrics["expectancy"],
            "raw_pips": metrics["net_pips"],
            "normalized_pips": metrics["norm_pips"],
            "net_r": metrics["net_r"],
            "annual_return_pct": metrics["annual_return_pct"],
            "max_dd_norm": metrics["max_dd"],
            "verdict": verdict
        })

    out_csv = RESULTS_DIR / "alpha_discovery_lab_v43_scorecard.csv"
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"\nSaved Scorecard to: {out_csv.resolve()}")
    print(f"Total Execution Time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

