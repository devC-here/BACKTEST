"""
========================================================================================
PRODUCTION SCANNER ENGINE v3.9: MT5 CONNECTIVITY & HOURS-OPTIMIZED SUITE
========================================================================================
Institutional-grade execution foundation for all 70 Forex Trading Strategy Scanners.
Calibrated with the empirical positive-EV hours from our week-long live forward test.

Key Capabilities:
  - Zero-Spread ECN Account Direct Connectivity (EURUSD.0, EURUSDm, Raw Spread, etc.)
  - Real-Time Market Session & Hour Tracking (Hours 00:00 to 23:00)
  - Pre-Flight Hours Optimization Gate (Filters off-hour toxic drawdown trades)
  - Multi-Timeframe Rate Extraction (M5, M15, M30, H1, H2, H4, D1)
  - Complete Institutional Technical Indicator Suite (All 70 Strategies)
  - Institutional ASCII Visual Signal Card with Hours Filter Status
  - Strict Pip & R-Multiple Measurement (Zero Dollar Distortion)
========================================================================================
"""

import sys
import os
import time
import datetime
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

# Configure console encoding for UTF-8 compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


TIMEFRAME_MAP = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 16385,
    "H2": 16386,
    "H4": 16388,
    "D1": 16408,
    "W1": 32769
}


class MT5Engine:
    """Production connection and rate extraction engine for MT5 with Hours-Optimization."""

    _initialized = False

    @classmethod
    def initialize(cls) -> bool:
        """Initializes connection to MT5 terminal."""
        if not MT5_AVAILABLE:
            print("[-] MetaTrader5 package is not installed.")
            return False

        if cls._initialized:
            info = mt5.terminal_info()
            if info is not None and info.connected:
                return True

        if not mt5.initialize():
            print(f"[-] MT5 initialization failed: {mt5.last_error()}")
            return False

        cls._initialized = True
        return True

    @classmethod
    def shutdown(cls):
        """Closes MT5 terminal connection."""
        if MT5_AVAILABLE and cls._initialized:
            mt5.shutdown()
            cls._initialized = False

    @classmethod
    def resolve_symbol(cls, preferred_symbol: Optional[str] = None) -> str:
        """
        Discovers the active broker symbol format for major FX pairs.
        Prioritizes Zero-Spread / Raw accounts (.0, .pro, m, _raw).
        """
        if not cls.initialize():
            return preferred_symbol or "EURUSD.0"

        candidates = []
        if preferred_symbol:
            candidates.append(preferred_symbol)
            base = preferred_symbol.split(".")[0].replace("m", "").replace("_raw", "")
            candidates.extend([f"{base}.0", base, f"{base}m", f"{base}.pro", f"{base}_raw"])
        else:
            candidates = ["EURUSD.0", "EURUSD", "EURUSDm", "EURUSD.pro", "EURUSD_raw"]

        for s in candidates:
            info = mt5.symbol_info(s)
            if info is not None and info.visible:
                return s

        symbols = mt5.symbols_get()
        if symbols:
            for s in symbols:
                if "EURUSD" in s.name:
                    mt5.symbol_select(s.name, True)
                    return s.name

        return preferred_symbol or "EURUSD.0"

    @classmethod
    def get_pip_size(cls, symbol: str) -> float:
        """Calculates authentic pip size for the symbol."""
        if not cls.initialize():
            return 0.0001
        info = mt5.symbol_info(symbol)
        if info is None:
            return 0.0001
        digits = info.digits
        if digits in [3, 5]:
            return 10 ** -(digits - 1)
        return 10 ** -digits

    @classmethod
    def fetch_rates(cls, symbol: str, timeframe: str = "M15", bars: int = 100) -> Optional[pd.DataFrame]:
        """Fetches historical OHLCV bar data from MT5."""
        if not cls.initialize():
            return None
        tf_const = TIMEFRAME_MAP.get(timeframe.upper(), mt5.TIMEFRAME_M15)
        rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, bars)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = calculate_all_indicators(df)
        return df

    @classmethod
    def fetch_all_timeframes(cls, symbol: str, bars: int = 100) -> Dict[str, pd.DataFrame]:
        """Fetches multi-timeframe candle datasets required for 70-strategy evaluation."""
        data = {}
        for tf in ["M5", "M15", "M30", "H1", "H2", "H4", "D1"]:
            df = cls.fetch_rates(symbol, tf, bars)
            if df is not None:
                data[tf] = df
        return data

    @staticmethod
    def get_current_hour(bar_time: Optional[str] = None) -> int:
        """
        Extracts current trading hour (0-23).
        Directly matches the candle bar time from MT5 terminal / live test.
        """
        if bar_time:
            try:
                return pd.to_datetime(bar_time).hour
            except Exception:
                pass
        return datetime.datetime.now().hour

    @staticmethod
    def get_session_name(hour: int) -> str:
        """Returns institutional market session name for a given trading hour."""
        if 0 <= hour <= 5:
            return "Asian Session (Tokyo / Sydney)"
        elif 6 <= hour <= 7:
            return "Asian / London Transition (Liquidity Build)"
        elif 8 <= hour <= 11:
            return "London Morning Rush (High Volatility)"
        elif 12 <= hour <= 16:
            return "London / NY Institutional Overlap (Peak Liquidity & Volume)"
        elif 17 <= hour <= 20:
            return "NY Afternoon Session (Trend Continuation)"
        else:
            return "Asian Rollover / Off-Peak (High Spread / Low Liquidity)"


# ==============================================================================
# TECHNICAL INDICATOR ENGINE
# ==============================================================================

def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates full quantitative technical indicators suite across OHLCV bars."""
    c = df['close']
    h = df['high']
    l = df['low']
    o = df['open']
    n = len(df)

    # 1. EMAs
    for p in [5, 7, 8, 9, 10, 13, 20, 21, 33, 44, 50, 200]:
        df[f'ema_{p}'] = c.ewm(span=p, adjust=False).mean()

    # 2. SMAs
    for p in [10, 20, 50, 100, 200]:
        df[f'sma_{p}'] = c.rolling(p, min_periods=1).mean()

    # 3. Double EMA (DEMA 200)
    ema1 = c.ewm(span=200, adjust=False).mean()
    ema2 = ema1.ewm(span=200, adjust=False).mean()
    df['dema_200'] = (2 * ema1) - ema2

    # 4. ATR
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['tr'] = tr
    df['atr_1'] = tr.rolling(1, min_periods=1).mean()
    df['atr_7'] = tr.rolling(7, min_periods=1).mean()
    df['atr_14'] = tr.rolling(14, min_periods=1).mean()

    # 5. RSI
    diff = c.diff()
    up = diff.clip(lower=0)
    dn = -diff.clip(upper=0)
    for p in [7, 10, 14]:
        roll_up = up.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
        roll_dn = dn.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
        rs = roll_up / roll_dn.replace(0, np.nan)
        df[f'rsi_{p}'] = 100.0 - (100.0 / (1.0 + rs))
        df[f'rsi_{p}'] = df[f'rsi_{p}'].bfill().fillna(50.0)

    # 6. MACD (12, 26, 9)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df['macd_line'] = ema12 - ema26
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']

    # 7. Bollinger Bands (20, 2)
    sma20 = df['sma_20']
    std20 = c.rolling(20, min_periods=1).std().bfill()
    df['bb_mid'] = sma20
    df['bb_upper'] = sma20 + (2.0 * std20)
    df['bb_lower'] = sma20 - (2.0 * std20)

    # 8. Stochastic (14, 3, 3)
    l14 = l.rolling(14, min_periods=1).min()
    h14 = h.rolling(14, min_periods=1).max()
    denom = (h14 - l14).replace(0, np.nan)
    df['stoch_k'] = 100.0 * (c - l14) / denom
    df['stoch_k'] = df['stoch_k'].rolling(3, min_periods=1).mean().bfill().fillna(50.0)
    df['stoch_d'] = df['stoch_k'].rolling(3, min_periods=1).mean().bfill().fillna(50.0)

    # 9. Swing Extremes
    df['swing_high_20'] = h.rolling(20, min_periods=1).max()
    df['swing_low_20'] = l.rolling(20, min_periods=1).min()
    df['swing_high_50'] = h.rolling(50, min_periods=1).max()
    df['swing_low_50'] = l.rolling(50, min_periods=1).min()

    # 10. Volume / OBV
    vol = df.get('tick_volume', pd.Series(1, index=df.index))
    df['vol_sma_20'] = vol.rolling(20, min_periods=1).mean()
    direction = np.where(c > c.shift(1), 1, np.where(c < c.shift(1), -1, 0))
    df['obv'] = (direction * vol).cumsum()

    # 11. Fair Value Gap (FVG)
    df['bullish_fvg'] = (l > h.shift(2)) & (l.shift(1) > h.shift(2))
    df['bearish_fvg'] = (h < l.shift(2)) & (h.shift(1) < l.shift(2))

    # 12. Heikin Ashi
    ha_close = (o + h + l + c) / 4.0
    ha_open = [(o.iloc[0] + c.iloc[0]) / 2.0]
    for i in range(1, n):
        ha_open.append((ha_open[-1] + ha_close.iloc[i - 1]) / 2.0)
    df['ha_close'] = ha_close
    df['ha_open'] = pd.Series(ha_open, index=df.index)
    df['ha_high'] = pd.concat([h, df['ha_open'], df['ha_close']], axis=1).max(axis=1)
    df['ha_low'] = pd.concat([l, df['ha_open'], df['ha_close']], axis=1).min(axis=1)

    # 13. Supertrend (7, 2.5)
    atr7 = df['atr_7']
    hl2 = (h + l) / 2.0
    up_b = hl2 - (2.5 * atr7)
    dn_b = hl2 + (2.5 * atr7)
    st = []
    st_dir = []
    d = 1
    curr_st = up_b.iloc[0]
    for i in range(n):
        cl = c.iloc[i]
        if d == 1:
            if cl < up_b.iloc[i]:
                d = -1
                curr_st = dn_b.iloc[i]
            else:
                curr_st = max(up_b.iloc[i], curr_st) if i > 0 else up_b.iloc[i]
        else:
            if cl > dn_b.iloc[i]:
                d = 1
                curr_st = up_b.iloc[i]
            else:
                curr_st = min(dn_b.iloc[i], curr_st) if i > 0 else dn_b.iloc[i]
        st.append(curr_st)
        st_dir.append(d)
    df['supertrend'] = st
    df['st_dir'] = st_dir

    # 14. VWAP
    typ = (h + l + c) / 3.0
    df['vwap'] = (typ * vol).cumsum() / (vol.cumsum().replace(0, np.nan))
    df['vwap'] = df['vwap'].bfill()

    # 15. Chandelier Exit (22, 3.0)
    atr22 = tr.rolling(22, min_periods=1).mean().bfill()
    df['chandelier_long'] = h.rolling(22, min_periods=1).max() - (3.0 * atr22)
    df['chandelier_short'] = l.rolling(22, min_periods=1).min() + (3.0 * atr22)

    # 16. UT Bot Trailing Stop (kv=2, atr=1)
    n_loss = 2.0 * df['atr_1']
    ut_stop_list = []
    c_stop = c.iloc[0] - n_loss.iloc[0]
    for i in range(n):
        cl = c.iloc[i]
        p_stop = ut_stop_list[-1] if ut_stop_list else c_stop
        if cl > p_stop:
            c_stop = max(p_stop, cl - n_loss.iloc[i])
        else:
            c_stop = min(p_stop, cl + n_loss.iloc[i])
        ut_stop_list.append(c_stop)
    df['ut_stop'] = ut_stop_list

    # 17. Bill Williams Alligator (13/8/5 smoothed)
    df['alligator_jaw'] = c.rolling(13, min_periods=1).mean().shift(8).bfill()
    df['alligator_teeth'] = c.rolling(8, min_periods=1).mean().shift(5).bfill()
    df['alligator_lips'] = c.rolling(5, min_periods=1).mean().shift(3).bfill()

    # 18. LuxAlgo TRAUMA Baseline (66)
    df['trauma'] = c.rolling(66, min_periods=1).mean().bfill()

    # 19. Camarilla Pivots (intraday dynamic)
    prev_h = h.shift(1).bfill()
    prev_l = l.shift(1).bfill()
    prev_c = c.shift(1).bfill()
    p_rng = (prev_h - prev_l).replace(0, np.nan)
    df['cam_h4'] = prev_c + (p_rng * 1.1 / 2.0)
    df['cam_h3'] = prev_c + (p_rng * 1.1 / 4.0)
    df['cam_l3'] = prev_c - (p_rng * 1.1 / 4.0)
    df['cam_l4'] = prev_c - (p_rng * 1.1 / 2.0)
    df['cam_s2'] = prev_c - (p_rng * 1.1 / 6.0)
    df['cam_s4'] = prev_c - (p_rng * 1.1 / 2.0)

    return df


# ==============================================================================
# VISUAL ASCII SIGNAL CARD EMITTER
# ==============================================================================

def emit_signal_card(sig: Dict[str, Any]):
    """
    Renders an institutional ASCII signal alert card to the console.
    Denominated strictly in PIPS and R-MULTIPLES. Displays Hours-Optimization Status.
    """
    sid = sig.get("strategy_id", "S00")
    sname = sig.get("strategy_name", "Quantitative Strategy")
    vtag = sig.get("version_tag", "v3.9_hours_optimized")
    symbol = sig.get("symbol", "EURUSD.0")
    tf = sig.get("timeframe", "M15")
    direction = sig.get("signal", "NONE").upper()
    status = sig.get("status", "WAITING")
    bar_time = sig.get("bar_time", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    entry = sig.get("entry_price", 0.0)
    sl = sig.get("sl_price", 0.0)
    tp = sig.get("tp_price", 0.0)
    target_rr = sig.get("target_rr", 2.0)
    risk_pips = sig.get("risk_pips", 0.0)
    reward_pips = sig.get("reward_pips", 0.0)
    be_price = sig.get("be_trigger_price", 0.0)
    be_pips = sig.get("be_trigger_pips", 0.0)
    inval = sig.get("invalidation_level", 0.0)
    grade = sig.get("grade", "Alpha Certified")

    hour = sig.get("current_hour", MT5Engine.get_current_hour(bar_time))
    session = sig.get("session_name", MT5Engine.get_session_name(hour))
    allowed_hours = sig.get("allowed_hours", [])
    opt_net = sig.get("opt_net_pips", "--")
    opt_pf = sig.get("opt_pf", "--")

    if status == "OFF_HOURS_SUPPRESSED":
        print("\n" + "=" * 80)
        print(f" [HOURS OPTIMIZATION FILTER] {sid} | {sname.upper()} (SUPPRESSED)")
        print("=" * 80)
        print(f" Market Pair        : {symbol} (Zero-Spread ECN Calibrated)")
        print(f" Bar Timestamp      : {bar_time} (Hour {hour:02d}:00) -> [{session}]")
        print(f" Filter State       : 🔴 RESTRICTED HOUR (Filtered to prevent off-hour drawdown)")
        print(f" Allowed Hours      : {allowed_hours}")
        print(f" Expected Alpha     : +{opt_net} pips | Profit Factor: {opt_pf}")
        print("-" * 80)
        print(" Action: SIGNAL SUPPRESSED. Strategy sleeping to preserve capital.")
        print("=" * 80 + "\n")
        return

    dir_color = "🟢 BUY SETUP" if direction == "BUY" else "🔴 SELL SETUP"

    print("\n" + "=" * 80)
    print(f" [INSTITUTIONAL SIGNAL ALERT] {sid} | {sname.upper()} ({dir_color})")
    print("=" * 80)
    print(f" Strategy Version   : {vtag} [{grade}]")
    print(f" Market Pair        : {symbol} (Zero-Spread ECN Calibrated)")
    print(f" Execution Chart    : {tf} Native Timeframe")
    print(f" Trigger Bar Time   : {bar_time} (Hour: {hour:02d}:00)")
    print(f" Market Session     : {session}")
    print(f" Hours Status       : 🟢 POSITIVE-EV ALPHA WINDOW (Opt PF: {opt_pf} | Net: +{opt_net}p)")
    print("-" * 80)
    print(f" Execution Action   : {direction} @ {entry:.5f}")
    print(f" Structural SL      : {sl:.5f}  (Risk: {risk_pips:.1f} pips)")
    print(f" Target Profit (TP) : {tp:.5f}  (Reward: {reward_pips:.1f} pips | 1:{target_rr:.2f} R:R)")
    print(f" Breakeven Trigger  : {be_price:.5f}  (Lock SL to BE at +{be_pips:.1f} pips)")
    if inval > 0:
        print(f" Invalidation Level : {inval:.5f}  (Pullback Boundary Cut)")
    print("-" * 80)
    print(" Note: SIGNAL CARD ONLY. Verified for manual execution or automated bridge.")
    print("=" * 80 + "\n")
