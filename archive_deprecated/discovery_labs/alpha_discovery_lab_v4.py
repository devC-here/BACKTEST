#!/usr/bin/env python3
"""
================================================================================
  ALPHA DISCOVERY LAB V4: LOWER-TIMEFRAME PRECISION & CPR/PIVOT DEFENSE
================================================================================
Evaluates 3 New Quantitative Alpha Models across the 70% Discovery Partition:
  1. M30_M5_PRECISION_RETEST: M30 Donchian level + M5 rejection wick (>= 50%).
  2. CAMARILLA_PIN_REVERSAL: Camarilla L3/H3 floor/ceiling test with pin bar confirmation.
  3. DAILY_CPR_VIRGIN_BOUNCE: Daily Central Pivot Range test during London Open (07:00-11:00 UTC).

Tested across EURUSD, EURGBP, and USDJPY on 70% Discovery Partition ONLY.
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
# Model 1: M30 Donchian Level + M5 Rejection Wick
# ==============================================================================
class M30M5PrecisionRetest(BaseStrategy):
    """
    Scales macro structural retest to M30/M5.
    M30 20-period Donchian levels + M5 rejection wick (>= 50%).
    Trend filter: M30 20-EMA.
    Active window: 07:00 - 16:30 UTC (plus Tokyo Fix for USDJPY).
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_M30_M5_Precision_Retest"

    def generate_signals(self, ticks: np.ndarray):
        n = len(ticks)
        signals = np.zeros(n, dtype=np.int8)
        sl_arr = np.zeros(n, dtype=np.float64)
        tp_arr = np.zeros(n, dtype=np.float64)
        if n < 500:
            return signals, sl_arr, tp_arr

        m5 = build_time_bars(ticks, timeframe_min=5, pip_size=self.pip_size)
        if len(m5) < 60:
            return signals, sl_arr, tp_arr

        ts = pd.to_datetime(ticks['timestamp'], unit='ms')
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        base = pd.DataFrame({'close': mid, 'high': mid, 'low': mid}, index=ts)
        m30 = base.resample('30min').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(m30) < 25:
            return signals, sl_arr, tp_arr

        m30['ema20'] = m30['close'].ewm(span=20, adjust=False).mean()
        m30['bias'] = np.where(m30['close'] > m30['ema20'], 1, -1)
        m30['roll_h'] = m30['high'].rolling(20).max().shift(1).bfill()
        m30['roll_l'] = m30['low'].rolling(20).min().shift(1).bfill()

        bias_m5 = m30['bias'].reindex(m5.index, method='ffill').fillna(0).values
        supp_m5 = m30['roll_l'].reindex(m5.index, method='ffill').values
        res_m5 = m30['roll_h'].reindex(m5.index, method='ffill').values

        c = m5['close'].values
        h = m5['high'].values
        l = m5['low'].values
        o = m5['open'].values
        spread = m5['spread_pips'].values
        hours = m5.index.hour.values
        dates = m5.index.date
        time_ms = m5.index.astype('datetime64[ms]').view('int64')
        dur_ms = 5 * 60 * 1000
        tick_ts = ticks['timestamp']

        traded_days = set()
        valid_hours = {1, 2, 3, 7, 8, 9, 12, 13, 14, 15} if self.asset == "USDJPY" else set(range(7, 17))
        dist_thresh = 3.0 if self.asset == "USDJPY" else 2.5

        for i in range(20, len(m5)):
            if hours[i] not in valid_hours:
                continue
            d = dates[i]
            if d in traded_days:
                continue
            if spread[i] > self.max_spread:
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < (3.0 if self.asset == "USDJPY" else 2.2):
                continue

            b = bias_m5[i]
            sig = 0

            # Bullish Retest of M30 Support on M5
            if b > 0:
                dist_supp = abs(l[i] - supp_m5[i]) / self.pip_size
                lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
                if dist_supp <= dist_thresh and (lower_wick / bar_rng) >= 0.50 and c[i] > o[i]:
                    sig = SIGNAL_LONG

            # Bearish Retest of M30 Resistance on M5
            elif b < 0:
                dist_res = abs(res_m5[i] - h[i]) / self.pip_size
                upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size
                if dist_res <= dist_thresh and (upper_wick / bar_rng) >= 0.50 and c[i] < o[i]:
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
# Model 2: Camarilla Floor/Ceiling Pin Reversal (M15)
# ==============================================================================
class CamarillaPinReversal(BaseStrategy):
    """
    Camarilla L3 (Support) / H3 (Resistance) mean-reversion with M15 pin bar confirmation.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Camarilla_Pin_Reversal"

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

        # Camarilla Formulas
        prev_h = daily['high'].shift(1)
        prev_l = daily['low'].shift(1)
        prev_c = daily['close'].shift(1)
        rng = prev_h - prev_l

        h3 = prev_c + rng * 1.1 / 4.0
        l3 = prev_c - rng * 1.1 / 4.0

        h3_m15 = h3.reindex(m15.index, method='ffill').values
        l3_m15 = l3.reindex(m15.index, method='ffill').values

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
        dist_thresh = 4.0 if self.asset == "USDJPY" else 3.0

        for i in range(15, len(m15)):
            if not (7 <= hours[i] <= 16):
                continue
            d = dates[i]
            if d in traded_days:
                continue
            if spread[i] > self.max_spread:
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < 3.0:
                continue

            sig = 0
            # Long at L3 Support: Low tested L3, formed lower rejection wick >= 45%, closed green
            dist_l3 = abs(l[i] - l3_m15[i]) / self.pip_size
            lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
            if dist_l3 <= dist_thresh and (lower_wick / bar_rng) >= 0.45 and c[i] > o[i]:
                sig = SIGNAL_LONG

            # Short at H3 Resistance: High tested H3, formed upper rejection wick >= 45%, closed red
            dist_h3 = abs(h3_m15[i] - h[i]) / self.pip_size
            upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size
            if dist_h3 <= dist_thresh and (upper_wick / bar_rng) >= 0.45 and c[i] < o[i]:
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
# Model 3: Daily Central Pivot Range (CPR) Bounce
# ==============================================================================
class DailyCPRBounce(BaseStrategy):
    """
    Central Pivot Range (CPR) model:
    Pivot = (H + L + C) / 3
    BC = (H + L) / 2
    TC = (Pivot - BC) + Pivot
    During European hours (07:00-11:00 UTC), fades retests of the CPR boundary.
    """
    def __init__(self, asset: str, sl_pips: float, tp_pips: float, pip_size: float, max_spread: float, **kwargs):
        super().__init__(**kwargs)
        self.asset = asset
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.pip_size = float(pip_size)
        self.max_spread = float(max_spread)
        self.name = f"{asset}_Daily_CPR_Bounce"

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

        prev_h = daily['high'].shift(1)
        prev_l = daily['low'].shift(1)
        prev_c = daily['close'].shift(1)

        pivot = (prev_h + prev_l + prev_c) / 3.0
        bc = (prev_h + prev_l) / 2.0
        tc = (pivot - bc) + pivot

        cpr_top = np.maximum(tc, bc)
        cpr_bot = np.minimum(tc, bc)

        top_m15 = cpr_top.reindex(m15.index, method='ffill').values
        bot_m15 = cpr_bot.reindex(m15.index, method='ffill').values

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

        for i in range(15, len(m15)):
            if hours[i] not in {7, 8, 9, 10}:
                continue
            d = dates[i]
            if d in traded_days:
                continue
            if spread[i] > self.max_spread:
                continue

            bar_rng = (h[i] - l[i]) / self.pip_size
            if bar_rng < 3.0:
                continue

            sig = 0
            # Test CPR Top as Support
            dist_top = abs(l[i] - top_m15[i]) / self.pip_size
            lower_wick = (min(o[i], c[i]) - l[i]) / self.pip_size
            if dist_top <= 3.5 and (lower_wick / bar_rng) >= 0.45 and c[i] > o[i]:
                sig = SIGNAL_LONG

            # Test CPR Bottom as Resistance
            dist_bot = abs(bot_m15[i] - h[i]) / self.pip_size
            upper_wick = (h[i] - max(o[i], c[i])) / self.pip_size
            if dist_bot <= 3.5 and (upper_wick / bar_rng) >= 0.45 and c[i] < o[i]:
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
# Discovery V4 Execution Harness
# ==============================================================================
def evaluate_v4_discovery_week(args: Tuple[str, int]) -> Dict[str, Any]:
    asset, week_num = args
    pip_size = 0.01 if asset == "USDJPY" else 0.0001
    engine = BacktestEngine(pip_size=pip_size, use_slippage=True)

    ticks, meta = get_week_dataset(week_num, raw_dir=f"data/{asset}")
    if len(ticks) == 0:
        return {"asset": asset, "week": week_num, "trades": {}}

    models = []
    if asset == "USDJPY":
        models.append(("M30_M5_PRECISION", M30M5PrecisionRetest(asset=asset, sl_pips=10.0, tp_pips=16.0, pip_size=0.01, max_spread=0.8)))
        models.append(("CAMARILLA_PIN", CamarillaPinReversal(asset=asset, sl_pips=12.0, tp_pips=20.0, pip_size=0.01, max_spread=0.8)))
        models.append(("CPR_BOUNCE", DailyCPRBounce(asset=asset, sl_pips=12.0, tp_pips=20.0, pip_size=0.01, max_spread=0.8)))
    elif asset == "EURGBP":
        models.append(("M30_M5_PRECISION", M30M5PrecisionRetest(asset=asset, sl_pips=7.0, tp_pips=12.0, pip_size=0.0001, max_spread=2.0)))
        models.append(("CAMARILLA_PIN", CamarillaPinReversal(asset=asset, sl_pips=8.0, tp_pips=14.0, pip_size=0.0001, max_spread=2.0)))
        models.append(("CPR_BOUNCE", DailyCPRBounce(asset=asset, sl_pips=8.0, tp_pips=14.0, pip_size=0.0001, max_spread=2.0)))
    elif asset == "EURUSD":
        models.append(("M30_M5_PRECISION", M30M5PrecisionRetest(asset=asset, sl_pips=8.0, tp_pips=13.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("CAMARILLA_PIN", CamarillaPinReversal(asset=asset, sl_pips=9.0, tp_pips=15.0, pip_size=0.0001, max_spread=0.6)))
        models.append(("CPR_BOUNCE", DailyCPRBounce(asset=asset, sl_pips=9.0, tp_pips=15.0, pip_size=0.0001, max_spread=0.6)))

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


def run_discovery_v4():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("  LAUNCHING ALPHA DISCOVERY V4 EXPEDITION (70% DISCOVERY PARTITIONS)", flush=True)
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
        futures = {executor.submit(evaluate_v4_discovery_week, t): t for t in all_tasks}
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
            "Asset": asset,
            "Strategy": m_name,
            "Discovery_Trades": n,
            "Win_Rate_Pct": round(wr, 1),
            "Profit_Factor": pf,
            "Net_Pips": round(net_pips, 1),
            "Max_DD_Pips": round(max_dd, 1),
            "Ret_DD_Ratio": ret_dd,
            "Expectancy_pips": expectancy,
            "Discovery_Status": "🌟 VIABLE CANDIDATE" if net_pips > 100.0 and pf >= 1.35 and expectancy >= 1.5 else ("✅ POSITIVE" if net_pips > 0 and pf >= 1.15 else "❌ REJECT")
        })

    df = pd.DataFrame(rows).sort_values(by="Net_Pips", ascending=False)
    csv_out = RESULTS_DIR / "alpha_discovery_v4_scorecard.csv"
    df.to_csv(csv_out, index=False)
    print(f"\n[+] Saved Discovery V4 Scorecard to: {csv_out.resolve()}", flush=True)
    print("\n" + "=" * 80, flush=True)
    print(f"DISCOVERY V4 EXPEDITION COMPLETED IN {time.time() - t_start:.1f}s!", flush=True)
    print("=" * 80, flush=True)
    return df


if __name__ == "__main__":
    run_discovery_v4()
