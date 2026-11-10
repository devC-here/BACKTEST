#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V9: RANGE EXPANSIONS, ATR EXHAUSTIONS & LIQUIDITY TRAPS
================================================================================
Evaluates 4 New Quantitative Microstructure Models across the 70% Discovery Partition:
  1. DISC56_CALIBRATED_ASIAN_BREAKOUT_RETEST: Asian range (00:00 - 06:00 UTC) expansion
     breakout + boundary retest with volume/momentum confirmation (2.0R target).
  2. DISC57_ATR_DAILY_STRETCH_EXHAUSTION_FADE: Intraday extension exceeding 1.25x 14-day ATR
     during NY session (13:00 - 18:00 UTC) with M15 exhaustion pin fade.
  3. DISC58_DONCHIAN_SWEEP_LIQUIDITY_TRAP: False breakout / Turtle Soup trap where price
     sweeps 20-period H1 Donchian High/Low by 2-8 pips and snaps back inside.
  4. DISC59_TRIPLE_SCREEN_M15_IMPULSE: H4 20-EMA trend + H1 MACD Histogram slope agreement
     + M15 pullback to dynamic EMA13 value zone with reversal candle.

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
# Model 1: Calibrated Asian Range Breakout + Boundary Retest (DISC56)
# ==============================================================================
class CalibratedAsianBreakoutRetest(BaseStrategy):
    """
    Measures 00:00 - 06:00 UTC Asian High & Low.
    During 07:00 - 11:30 UTC:
      - Requires Asian range to be compact (< 30 pips on EU/EG, < 40 on UJ).
      - Price breaks out of Asian range with momentum.
      - Pulls back to retest the broken boundary level within 2.5 pips.
      - Prints a confirmation candle with rejection wick >= 35% in breakout direction.
      - 2.0R target.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Calib_Asian_Retest"

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
        mins = m15.index.minute.values
        dates = m15.index.date
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        asian_ranges = {}
        for d in np.unique(dates):
            mask = (dates == d) & (hours >= 0) & (hours < 6)
            if np.any(mask):
                ah = np.max(h[mask])
                al = np.min(l[mask])
                rng_p = (ah - al) / self.pip_size
                asian_ranges[d] = (ah, al, rng_p)

        traded_days = set()
        tol = 2.5 * self.pip_size
        max_asian_pips = 40.0 if self.asset == "USDJPY" else 28.0

        for i in range(len(m15)):
            if not ((hours[i] in {7, 8, 9, 10}) or (hours[i] == 11 and mins[i] <= 30)):
                continue
            d = dates[i]
            if d not in asian_ranges or d in traded_days:
                continue
            if spread[i] > self.max_spread:
                continue

            ah, al, arng = asian_ranges[d]
            if arng > max_asian_pips or arng < 5.0:
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < 3.0:
                continue

            sig = 0
            # Bullish: Retesting Asian High from above
            if l[i] <= ah + tol and c[i] > ah and c[i] > o[i]:
                lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
                if (lower_wick / bar_rng) >= 0.35:
                    sig = SIGNAL_LONG

            # Bearish: Retesting Asian Low from below
            elif h[i] >= al - tol and c[i] < al and c[i] < o[i]:
                upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size
                if (upper_wick / bar_rng) >= 0.35:
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
# Model 2: ATR Daily Stretch Exhaustion Fade (DISC57)
# ==============================================================================
class ATRDailyStretchExhaustionFade(BaseStrategy):
    """
    Measures 14-day rolling ATR from Daily bars.
    During 13:00 - 18:00 UTC (NY afternoon):
      - If current intraday move from the day's open exceeds 1.20x Daily ATR,
        the market is statistically overextended.
      - An M15 exhaustion pin bar (wick >= 50% pushing outside) triggers a fade trade
        targeting reversion back toward the day's median price (1.5R - 2.0R).
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_ATR_Stretch_Fade"

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

        daily = base.resample('1D').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(daily) < 2:
            return signals, sl_arr, tp_arr

        daily['tr'] = np.maximum(
            daily['high'] - daily['low'],
            np.maximum(
                abs(daily['high'] - daily['close'].shift(1)),
                abs(daily['low'] - daily['close'].shift(1))
            )
        )
        daily['atr14'] = daily['tr'].rolling(14, min_periods=1).mean().shift(1)
        daily_atr = daily['atr14'].reindex(m15.index, method='ffill').values

        # Daily open price
        dates = m15.index.date
        unique_dates = np.unique(dates)
        day_opens = {}
        for d in unique_dates:
            mask = dates == d
            if np.any(mask):
                day_opens[d] = m15['open'].values[np.where(mask)[0][0]]

        c = m15['close'].values
        h = m15['high'].values
        l = m15['low'].values
        o = m15['open'].values
        spread = m15['spread_pips'].values
        hours = m15.index.hour.values
        time_ms = m15.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        traded_days = set()

        for i in range(len(m15)):
            if not (13 <= hours[i] <= 18):
                continue
            d = dates[i]
            if d in traded_days or d not in day_opens:
                continue
            if spread[i] > self.max_spread:
                continue

            atr = daily_atr[i]
            if np.isnan(atr) or atr <= 0:
                continue

            d_open = day_opens[d]
            dist_from_open = (c[i] - d_open) / self.pip_size
            atr_pips = atr / self.pip_size

            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < 4.0:
                continue

            sig = 0
            # Upside stretch: price is > 1.15 ATR above day open, prints upper wick >= 50%
            if dist_from_open > 1.15 * atr_pips and c[i] < o[i]:
                upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size
                if (upper_wick / bar_rng) >= 0.50:
                    sig = SIGNAL_SHORT

            # Downside stretch: price is > 1.15 ATR below day open, prints lower wick >= 50%
            elif dist_from_open < -1.15 * atr_pips and c[i] > o[i]:
                lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
                if (lower_wick / bar_rng) >= 0.50:
                    sig = SIGNAL_LONG

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
# Model 3: Donchian Sweep Liquidity Trap / Turtle Soup (DISC58)
# ==============================================================================
class DonchianSweepLiquidityTrap(BaseStrategy):
    """
    Detects false breakouts beyond the 20-period H1 Donchian Channel:
      - Price penetrates the 20-period H1 Donchian High by 2 to 10 pips (triggering breakout buy stops).
      - Within the active M15 candle, sellers aggressively absorb and drive price back BELOW the Donchian High,
        printing an upper rejection wick >= 45%.
      - Enter Short with 2.0R target back toward the mean.
      - Inverse for Longs at Donchian Low.
      - Active Hours: 07:00 - 16:30 UTC.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Donchian_Sweep_Trap"

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

        h1 = base.resample('1h').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(h1) < 25:
            return signals, sl_arr, tp_arr

        h1['don_h20'] = h1['high'].rolling(20).max().shift(1)
        h1['don_l20'] = h1['low'].rolling(20).min().shift(1)

        don_h_m15 = h1['don_h20'].reindex(m15.index, method='ffill').values
        don_l_m15 = h1['don_l20'].reindex(m15.index, method='ffill').values

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

        traded_days = set()
        min_sweep_pips = 1.5
        max_sweep_pips = 10.0 if self.asset == "USDJPY" else 8.0

        for i in range(25, len(m15)):
            if not (7 <= hours[i] <= 16):
                continue
            d = dates[i]
            if d in traded_days:
                continue
            if spread[i] > self.max_spread:
                continue

            dh = don_h_m15[i]
            dl = don_l_m15[i]
            if np.isnan(dh) or np.isnan(dl):
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < 4.0:
                continue

            sig = 0

            # Bearish Liquidity Trap: High poked above Donchian High, but closed back below it with upper wick
            sweep_up = (h[i] - dh) / self.pip_size
            if min_sweep_pips <= sweep_up <= max_sweep_pips and c[i] < dh and c[i] < o[i]:
                upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size
                if (upper_wick / bar_rng) >= 0.45:
                    sig = SIGNAL_SHORT

            # Bullish Liquidity Trap: Low poked below Donchian Low, but closed back above it with lower wick
            sweep_dn = (dl - l[i]) / self.pip_size
            if min_sweep_pips <= sweep_dn <= max_sweep_pips and c[i] > dl and c[i] > o[i]:
                lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
                if (lower_wick / bar_rng) >= 0.45:
                    sig = SIGNAL_LONG

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
# Model 4: Triple Screen M15 Dynamic EMA Impulse (DISC59)
# ==============================================================================
class TripleScreenM15Impulse(BaseStrategy):
    """
    Elder Triple Screen Architecture adapted to FX tick microstructure:
      1. Screen 1 (H4): Macro Trend defined by H4 20-EMA.
      2. Screen 2 (H1): Wave Momentum defined by 12/26/9 MACD Histogram slope.
      3. Screen 3 (M15): Entry Ripple triggered when price pulls back to test M15 13-EMA
         and resumes trend with an expansion candle in H4/H1 direction.
      - Target: 2.0R.
      - Active Hours: 07:00 - 17:00 UTC.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Triple_Screen_Impulse"

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

        h1 = base.resample('1h').agg({'close': 'last'}).dropna()
        h4 = base.resample('4h').agg({'close': 'last'}).dropna()
        if len(h1) < 35 or len(h4) < 15:
            return signals, sl_arr, tp_arr

        # Screen 1: H4 20-EMA trend
        h4['ema20'] = h4['close'].ewm(span=20, adjust=False).mean()
        h4_trend = np.where(h4['close'] > h4['ema20'], 1, -1)
        h4_bias = pd.Series(h4_trend, index=h4.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        # Screen 2: H1 MACD Histogram slope
        h1['ema12'] = h1['close'].ewm(span=12, adjust=False).mean()
        h1['ema26'] = h1['close'].ewm(span=26, adjust=False).mean()
        h1['macd'] = h1['ema12'] - h1['ema26']
        h1['signal'] = h1['macd'].ewm(span=9, adjust=False).mean()
        h1['hist'] = h1['macd'] - h1['signal']
        h1['hist_slope'] = h1['hist'].diff()
        h1_slope_trend = np.where(h1['hist_slope'] > 0, 1, -1)
        h1_bias = pd.Series(h1_slope_trend, index=h1.index).shift(1).reindex(m15.index, method='ffill').fillna(0).values

        # Screen 3: M15 EMA 13 test
        m15['ema13'] = m15['close'].ewm(span=13, adjust=False).mean()
        ema13 = m15['ema13'].values

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

        last_trade_date = None
        tol = 2.5 * self.pip_size

        for i in range(25, len(m15)):
            if not (7 <= hours[i] <= 17):
                continue
            dt = dates[i]
            if dt == last_trade_date:
                continue
            if spread[i] > self.max_spread:
                continue

            bias_h4 = h4_bias[i]
            bias_h1 = h1_bias[i]
            if bias_h4 != bias_h1 or bias_h4 == 0:
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            body = abs(c[i] - o[i]) / self.pip_size
            if bar_rng < 3.5 or (body / bar_rng) < 0.50:
                continue

            e13 = ema13[i]
            sig = 0

            # Bullish: Low touched near EMA13 and closed strongly above
            if bias_h4 > 0 and l[i] <= e13 + tol and c[i] > e13 and c[i] > o[i]:
                sig = SIGNAL_LONG
            # Bearish: High touched near EMA13 and closed strongly below
            elif bias_h4 < 0 and h[i] >= e13 - tol and c[i] < e13 and c[i] < o[i]:
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
# Discovery V9 Multi-Asset Parallel Worker
# ==============================================================================
def evaluate_v9_discovery_week(args: Tuple[str, int]) -> Dict[str, Any]:
    asset, week_num = args
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if len(ticks) == 0:
        return {"asset": asset, "week": week_num, "trades": {}}

    models = []
    if asset == "USDJPY":
        models.append(("CALIB_ASIAN_BO", CalibratedAsianBreakoutRetest(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
        models.append(("ATR_STRETCH_FADE", ATRDailyStretchExhaustionFade(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
        models.append(("DONCHIAN_SWEEP", DonchianSweepLiquidityTrap(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
        models.append(("TRIPLE_SCREEN", TripleScreenM15Impulse(asset=asset, sl_pips=12.0, tp_pips=24.0, pip_size=0.01, max_spread=0.8)))
    elif asset == "EURGBP":
        models.append(("CALIB_ASIAN_BO", CalibratedAsianBreakoutRetest(asset=asset, sl_pips=7.0, tp_pips=14.0, pip_size=0.0001, max_spread=2.0)))
        models.append(("ATR_STRETCH_FADE", ATRDailyStretchExhaustionFade(asset=asset, sl_pips=7.0, tp_pips=14.0, pip_size=0.0001, max_spread=2.0)))
        models.append(("DONCHIAN_SWEEP", DonchianSweepLiquidityTrap(asset=asset, sl_pips=7.0, tp_pips=14.0, pip_size=0.0001, max_spread=2.0)))
        models.append(("TRIPLE_SCREEN", TripleScreenM15Impulse(asset=asset, sl_pips=7.0, tp_pips=14.0, pip_size=0.0001, max_spread=2.0)))
    elif asset == "EURUSD":
        models.append(("CALIB_ASIAN_BO", CalibratedAsianBreakoutRetest(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("ATR_STRETCH_FADE", ATRDailyStretchExhaustionFade(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("DONCHIAN_SWEEP", DonchianSweepLiquidityTrap(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("TRIPLE_SCREEN", TripleScreenM15Impulse(asset=asset, sl_pips=9.0, tp_pips=18.0, pip_size=0.0001, max_spread=0.6)))

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


def run_discovery_v9():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("  LAUNCHING ALPHA DISCOVERY V9 EXPEDITION (70% DISCOVERY PARTITIONS)", flush=True)
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
        futures = {executor.submit(evaluate_v9_discovery_week, t): t for t in all_tasks}
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
        out_csv = RESULTS_DIR / "alpha_discovery_lab_v9_scorecard.csv"
        df.to_csv(out_csv, index=False)

        print("\n" + "=" * 90, flush=True)
        print("  ALPHA DISCOVERY LAB V9 SCORECARD (70% DISCOVERY PARTITIONS ONLY)", flush=True)
        print("=" * 90, flush=True)
        print(f"{'Asset':<8} {'Strategy':<20} {'Trades':<8} {'Win Rate':<10} {'Net Pips':<12} {'PF':<8} {'Max DD':<10} {'Ret/DD':<8} {'Exp (p/t)':<10}")
        print("-" * 90, flush=True)
        for _, r in df.iterrows():
            print(f"{r['asset']:<8} {r['strategy']:<20} {int(r['trades']):<8} {r['win_rate']:>6.1f}%   {r['net_pips']:>10.1f}p  {r['profit_factor']:>6.2f}  {r['max_dd_pips']:>8.1f}p  {r['ret_dd']:>6.2f}x  {r['expectancy']:>8.2f}p")
        print("=" * 90, flush=True)

    print(f"\n[*] Alpha Discovery Lab V9 Complete in {time.time() - t_start:.2f}s.", flush=True)


if __name__ == "__main__":
    run_discovery_v9()
