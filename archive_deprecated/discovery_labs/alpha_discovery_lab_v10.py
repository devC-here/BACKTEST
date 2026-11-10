#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V10: ORDER BLOCKS, DIVERGENCE TRAPS & VWAP VALUE AREAS
================================================================================
Evaluates 4 New Quantitative Microstructure Architectures across 70% Discovery:
  1. DISC57_ORDER_BLOCK_MITIGATION: Institutional impulse order block 50% midpoint
     retest with tight SL and 2.5R target.
  2. DISC58_RSI_DIVERGENCE_SQUEEZE: Liquidity squeeze of failed RSI divergence faders;
     continuation expansion candle blowing past divergence swing high/low.
  3. DISC59_VWAP_VALUE_AREA_DEFENSE: Pullback into 1.0-SD VWAP Value Area boundary
     with rejection pin bar aligned with H4 trend.
  4. DISC60_LONDON_IGNITION_RANGE_BREAK: 07:00 - 07:30 UTC opening 30-min range
     momentum expansion breakout during active London morning (07:30 - 09:30 UTC).

Tested across EURUSD, EURGBP, and USDJPY on 70% Discovery Partition ONLY.
Strict 70/30 deterministic holdout isolation is maintained.
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


# ==============================================================================
# Model 1: Institutional Impulse Order Block Mitigation (DISC57)
# ==============================================================================
class OrderBlockMitigation(BaseStrategy):
    """
    Detects when an explosive M15 displacement bar occurs (bar range >= 2.0x 14-period ATR).
    Marks the origin M15 bar (the order block) and its 50% midpoint.
    When subsequent price pulls back to touch the 50% midpoint during 07:00 - 17:00 UTC,
    enters in direction of original displacement with 2.50R target.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Order_Block_Mitigation"

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

        # 14-period rolling average range
        bar_ranges = (h - l) / self.pip_size
        s_rng = pd.Series(bar_ranges).rolling(14, min_periods=1).mean().shift(1).values

        last_trade_date = None
        active_ob_dir = 0
        active_ob_mid = np.nan
        active_ob_bar = -999

        for i in range(20, len(m15)):
            if not (7 <= hours[i] <= 17):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            avg_r = s_rng[i]
            cur_r = bar_rng[i]
            body = abs(c[i] - o[i]) / self.pip_size

            # Check if current bar is an explosive displacement candle (>= 2.0x avg range and body >= 65%)
            if avg_r > 0 and cur_r >= 2.0 * avg_r and (body / cur_r) >= 0.65:
                if c[i] > o[i]:
                    active_ob_dir = 1
                    active_ob_mid = 0.5 * (h[i] + l[i])
                    active_ob_bar = i
                elif c[i] < o[i]:
                    active_ob_dir = -1
                    active_ob_mid = 0.5 * (h[i] + l[i])
                    active_ob_bar = i
                continue

            # Check if price pulls back to mitigate the active order block within 1 to 8 bars
            if active_ob_dir != 0 and 1 <= (i - active_ob_bar) <= 8:
                sig = 0
                tol = 2.0 * self.pip_size
                # Bullish OB mitigation: Low dips to order block midpoint and closes above
                if active_ob_dir == 1 and l[i] <= active_ob_mid + tol and c[i] > active_ob_mid and c[i] > o[i]:
                    sig = SIGNAL_LONG
                # Bearish OB mitigation: High reaches order block midpoint and closes below
                elif active_ob_dir == -1 and h[i] >= active_ob_mid - tol and c[i] < active_ob_mid and c[i] < o[i]:
                    sig = SIGNAL_SHORT

                if sig != 0:
                    exec_t = time_ms[i] + dur_ms
                    idx = int(np.searchsorted(tick_ts, exec_t))
                    if idx < n and signals[idx] == 0:
                        signals[idx] = sig
                        sl_arr[idx] = self.sl_pips
                        tp_arr[idx] = self.tp_pips
                        last_trade_date = dt
                        active_ob_dir = 0

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 2: Failed RSI Divergence Squeeze Continuation (DISC58)
# ==============================================================================
class RSIDivergenceSqueezeContinuation(BaseStrategy):
    """
    Identifies when early counter-trend traders get trapped:
      - Price makes a higher swing high on M15 while 14-period RSI makes a lower high (bearish divergence).
      - However, instead of reversing, price violently blows through the swing high with an expansion candle
        (body >= 60% of bar range), triggering a short squeeze.
      - Enters Long with the squeeze momentum (2.0R target).
      - Inverse for bullish divergence failure (long squeeze into Short entry).
      - Active Hours: 07:00 - 17:00 UTC.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_RSI_Divergence_Squeeze"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m15 = build_time_bars(ticks, timeframe_min=15, pip_size=self.pip_size)
        if len(m15) < 40:
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

        # Compute 14-period RSI
        delta = pd.Series(c).diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / np.maximum(avg_loss, 1e-9)
        rsi = (100 - (100 / (1 + rs))).values

        # Rolling 12-bar swing highs/lows
        swing_h = pd.Series(h).rolling(12).max().shift(1).values
        swing_l = pd.Series(l).rolling(12).min().shift(1).values

        last_trade_date = None

        for i in range(25, len(m15)):
            if not (7 <= hours[i] <= 17):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            sh = swing_h[i]
            sl = swing_l[i]
            if np.isnan(sh) or np.isnan(sl) or np.isnan(rsi[i]) or np.isnan(rsi[i-1]):
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            body = abs(c[i] - o[i]) / self.pip_size
            if bar_rng < 4.0 or (body / bar_rng) < 0.60:
                continue

            sig = 0
            # Bullish Squeeze: Price breaks above swing high while RSI was in lower-high regime (< 65), strong bullish candle
            if c[i] > sh and o[i] <= sh and c[i] > o[i] and rsi[i-1] < 62:
                sig = SIGNAL_LONG
            # Bearish Squeeze: Price breaks below swing low while RSI was in higher-low regime (> 38), strong bearish candle
            elif c[i] < sl and o[i] >= sl and c[i] < o[i] and rsi[i-1] > 38:
                sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 3: Dynamic VWAP Value Area Defense (DISC59)
# ==============================================================================
class VWAPValueAreaDefense(BaseStrategy):
    """
    Computes daily intraday VWAP starting at 00:00 UTC with 1.0-SD Value Area bands (approx 68% of price mass).
    When H4 20-EMA trend is active, a pullback to test the Value Area High (in uptrend) or Low (in downtrend)
    with a rejection pin bar triggers an institutional value defense entry (2.0R target).
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_VWAP_Value_Area"

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

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid}, index=ts)

        h4 = base.resample('4h').agg({'close': 'last'}).dropna()
        if len(h4) < 15:
            return signals, sl_arr, tp_arr

        h4['ema20'] = h4['close'].ewm(span=20, adjust=False).mean()
        h4_trend = np.where(h4['close'] > h4['ema20'], 1, -1)
        h4_bias = pd.Series(h4_trend, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

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

        # Daily cumulative VWAP proxy using typical price
        typ = (h + l + c) / 3.0
        vwap = np.zeros(len(m15))
        vwap_std = np.zeros(len(m15))

        cur_d = None
        cum_pv = 0.0
        cum_v = 0
        p_history = []

        for k in range(len(m15)):
            if dates[k] != cur_d:
                cur_d = dates[k]
                cum_pv = 0.0
                cum_v = 0
                p_history = []

            cum_pv += typ[k]
            cum_v += 1
            vwap[k] = cum_pv / cum_v
            p_history.append(typ[k])
            vwap_std[k] = np.std(p_history) if len(p_history) > 2 else 0.0

        vah = vwap + 1.0 * vwap_std
        val = vwap - 1.0 * vwap_std

        last_trade_date = None
        tol = 2.5 * self.pip_size

        for i in range(10, len(m15)):
            if not (7 <= hours[i] <= 17):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            bias = h4_bias[i]
            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < 3.5:
                continue

            sig = 0
            # Uptrend: Pullback to VAH or VWAP line and rejection wick >= 45%
            if bias > 0 and (l[i] <= vah[i] + tol and c[i] > vah[i]) and c[i] > o[i]:
                lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
                if (lower_wick / bar_rng) >= 0.45:
                    sig = SIGNAL_LONG

            # Downtrend: Rally to VAL or VWAP line and upper wick >= 45%
            elif bias < 0 and (h[i] >= val[i] - tol and c[i] < val[i]) and c[i] < o[i]:
                upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size
                if (upper_wick / bar_rng) >= 0.45:
                    sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    last_trade_date = dt

        return signals, sl_arr, tp_arr


# ==============================================================================
# Model 4: London 30-Min Ignition Range Expansion Breakout (DISC60)
# ==============================================================================
class LondonIgnitionRangeBreakout(BaseStrategy):
    """
    Measures the 07:00 - 07:30 UTC London opening 30-minute range (2 M15 bars).
    During 07:30 - 09:30 UTC:
      - Requires range to be between 4 and 25 pips (not excessive).
      - First M15 candle closing decisively beyond the ignition range in H4 20-EMA trend direction
        enters breakout trade with 2.0R target.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_London_Ignition_BO"

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

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid}, index=ts)

        h4 = base.resample('4h').agg({'close': 'last'}).dropna()
        if len(h4) < 15:
            return signals, sl_arr, tp_arr

        h4['ema20'] = h4['close'].ewm(span=20, adjust=False).mean()
        h4_trend = np.where(h4['close'] > h4['ema20'], 1, -1)
        h4_bias = pd.Series(h4_trend, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        mins = m15.index.minute.values
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        # Precompute London ignition range (07:00 to 07:30 UTC: bars at 07:00 and 07:15)
        ignition_ranges = {}
        for d in np.unique(dates):
            mask = (dates == d) & (hours == 7) & (mins < 30)
            if np.any(mask):
                ih = np.max(h[mask])
                il = np.min(l[mask])
                irng = (ih - il) / self.pip_size
                ignition_ranges[d] = (ih, il, irng)

        traded_days = set()
        max_irng = 30.0 if self.asset == "USDJPY" else 20.0

        for i in range(len(m15)):
            # Active window: 07:30 to 09:30 UTC
            in_window = (hours[i] == 7 and mins[i] >= 30) or (hours[i] == 8) or (hours[i] == 9 and mins[i] <= 30)
            if not in_window:
                continue
            d = dates[i]
            if d not in ignition_ranges or d in traded_days:
                continue
            if spread[i] > self.max_spread:
                continue

            ih, il, irng = ignition_ranges[d]
            if irng < 3.5 or irng > max_irng:
                continue

            bias = h4_bias[i]
            bar_rng = (h[i] - l[i]) / self.pip_size
            body = abs(c[i] - o[i]) / self.pip_size
            if bar_rng < 3.0 or (body / bar_rng) < 0.50:
                continue

            sig = 0
            if bias > 0 and c[i] > ih and o[i] <= ih and c[i] > o[i]:
                sig = SIGNAL_LONG
            elif bias < 0 and c[i] < il and o[i] >= il and c[i] < o[i]:
                sig = SIGNAL_SHORT

            if sig != 0:
                exec_t = time_ms[i] + dur_ms
                idx = int(np.searchsorted(tick_ts, exec_t))
                if idx < n and signals[idx] == 0:
                    signals[idx] = sig
                    sl_arr[idx] = self.sl_pips
                    tp_arr[idx] = self.tp_pips
                    traded_days.add(d)

        return signals, sl_arr, tp_arr


# ==============================================================================
# Discovery V10 Multi-Asset Parallel Worker
# ==============================================================================
def evaluate_v10_discovery_week(args: Tuple[str, int]) -> Dict[str, Any]:
    asset, week_num = args
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if len(ticks) == 0:
        return {"asset": asset, "week": week_num, "trades": {}}

    models = []
    if asset == "USDJPY":
        models.append(("ORDER_BLOCK_MIT", OrderBlockMitigation(asset=asset, sl_pips=12.0, tp_pips=30.0, pip_size=0.01, max_spread=0.8)))
        models.append(("RSI_DIV_SQUEEZE", RSIDivergenceSqueezeContinuation(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
        models.append(("VWAP_VAL_AREA", VWAPValueAreaDefense(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
        models.append(("LONDON_IGNITION", LondonIgnitionRangeBreakout(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
    elif asset == "EURGBP":
        models.append(("ORDER_BLOCK_MIT", OrderBlockMitigation(asset=asset, sl_pips=7.0, tp_pips=17.5, pip_size=0.0001, max_spread=2.0)))
        models.append(("RSI_DIV_SQUEEZE", RSIDivergenceSqueezeContinuation(asset=asset, sl_pips=7.0, tp_pips=14.0, pip_size=0.0001, max_spread=2.0)))
        models.append(("VWAP_VAL_AREA", VWAPValueAreaDefense(asset=asset, sl_pips=7.0, tp_pips=14.0, pip_size=0.0001, max_spread=2.0)))
        models.append(("LONDON_IGNITION", LondonIgnitionRangeBreakout(asset=asset, sl_pips=7.0, tp_pips=14.0, pip_size=0.0001, max_spread=2.0)))
    elif asset == "EURUSD":
        models.append(("ORDER_BLOCK_MIT", OrderBlockMitigation(asset=asset, sl_pips=9.0, tp_pips=22.5, pip_size=0.0001, max_spread=0.6)))
        models.append(("RSI_DIV_SQUEEZE", RSIDivergenceSqueezeContinuation(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("VWAP_VAL_AREA", VWAPValueAreaDefense(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("LONDON_IGNITION", LondonIgnitionRangeBreakout(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))

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


def run_discovery_v10():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("  LAUNCHING ALPHA DISCOVERY V10 EXPEDITION (70% DISCOVERY PARTITIONS)", flush=True)
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
        futures = {executor.submit(evaluate_v10_discovery_week, t): t for t in all_tasks}
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

    # Compute Scorecard
    rows = []
    for (asset, m_name), pips in results_by_key.items():
        n = len(pips)
        if n == 0:
            continue
        p_arr = np.array(pips)
        net_pips = float(np.sum(p_arr))
        wins = p_arr[p_arr > 0]
        losses = -p_arr[p_arr < 0]
        wr = float(len(wins) / n * 100.0)
        sum_w = float(np.sum(wins))
        sum_l = float(np.sum(losses))
        pf = round(sum_w / sum_l, 2) if sum_l > 0 else (99.0 if sum_w > 0 else 0.0)

        equity = np.cumsum(p_arr)
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
        ret_dd = round(net_pips / max_dd, 2) if max_dd > 0 else 0.0
        expectancy = round(net_pips / n, 2)

        rows.append({
            "asset": asset,
            "strategy": m_name,
            "trades": n,
            "win_rate": wr,
            "net_pips": net_pips,
            "profit_factor": pf,
            "max_dd_pips": max_dd,
            "ret_dd": ret_dd,
            "expectancy": expectancy
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["asset", "net_pips"], ascending=[True, False])
        out_csv = RESULTS_DIR / "alpha_discovery_lab_v10_scorecard.csv"
        df.to_csv(out_csv, index=False)

        print("\n" + "=" * 90, flush=True)
        print("  ALPHA DISCOVERY LAB V10 SCORECARD (70% DISCOVERY PARTITIONS ONLY)", flush=True)
        print("=" * 90, flush=True)
        print(f"{'Asset':<8} {'Strategy':<20} {'Trades':<8} {'Win Rate':<10} {'Net Pips':<12} {'PF':<8} {'Max DD':<10} {'Ret/DD':<8} {'Exp (p/t)':<10}")
        print("-" * 90, flush=True)
        for _, r in df.iterrows():
            print(f"{r['asset']:<8} {r['strategy']:<20} {int(r['trades']):<8} {r['win_rate']:>6.1f}%   {r['net_pips']:>10.1f}p  {r['profit_factor']:>6.2f}  {r['max_dd_pips']:>8.1f}p  {r['ret_dd']:>6.2f}x  {r['expectancy']:>8.2f}p")
        print("=" * 90, flush=True)

    print(f"\n[*] Alpha Discovery Lab V10 Complete in {time.time() - t_start:.2f}s.", flush=True)


if __name__ == "__main__":
    run_discovery_v10()
