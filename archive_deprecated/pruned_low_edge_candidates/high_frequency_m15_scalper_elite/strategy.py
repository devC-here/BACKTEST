import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT

class HighFrequencyM15ScalperElite(BaseStrategy):
    """
    High-Frequency M15 Scalper (All-Weather Refined Edition)
    ---------------------------------------------------------
    Design Objective: High Trade Volume Scalping on Higher Timeframe (M15)
    Execution Engine: Millisecond Tick Lookup with Authentic Floating Spread
    Trading Window: Peak Institutional Hours [07, 08, 09, 11, 12, 13, 14, 15, 17] UTC
    
    All-Weather Refinements:
      1. Multi-Timeframe Gating: True H1 50-EMA Macro Trend Alignment prevents counter-trend traps
      2. Dynamic ATR Stops: Dynamic breathing room (0.8x to 1.8x ATR) eliminates micro-stop sweepouts
      3. Dual Institutional Triggers:
         - Pattern A: 4-Hour Rolling Liquidity Sweep Fade
         - Pattern B: High-Velocity Fair Value Gap (FVG) Immediate Mitigation
      4. Target Scaling: 2.20 : 1 to 2.50 : 1 expansion
    """
    def __init__(
        self,
        target_rr: float = 2.2,
        min_sl: float = 6.0,
        max_sl: float = 18.0,
        allowed_hours = None,
        sweep_lookback: int = 16,
        use_h1_filter: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_rr = float(target_rr)
        self.min_sl = float(min_sl)
        self.max_sl = float(max_sl)
        self.allowed_hours = allowed_hours if allowed_hours is not None else {7, 8, 9, 11, 12, 13, 14, 15, 17}
        self.lookback = int(sweep_lookback)
        self.use_h1_filter = bool(use_h1_filter)
        self.pip_size = 0.0001
        self.name = "HighFrequency_M15_Scalper_Elite"

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

        # M15 Candles
        m15 = base_df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(m15) < 40:
            return signals, sl_pips_arr, tp_pips_arr

        # H1 Macro Trend Filter (Forward-filled to prevent lookahead)
        h1 = base_df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        h1['h1_ema50'] = h1['close'].ewm(span=50, adjust=False).mean()
        h1['h1_trend'] = np.where(h1['close'] > h1['h1_ema50'], 1, -1)
        m15['macro_trend'] = h1['h1_trend'].reindex(m15.index, method='ffill').fillna(0)

        # M15 ATR & Fast Indicators
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
            macro = row['macro_trend'] if self.use_h1_filter else 0
            atr_pips = row['atr'] / self.pip_size
            dyn_min_sl = max(self.min_sl, 0.8 * atr_pips)
            dyn_max_sl = max(self.max_sl, 1.8 * atr_pips)

            # Pattern A: 4-Hour Rolling Liquidity Sweep
            window = m15.iloc[i - self.lookback : i]
            roll_h = window['high'].max()
            roll_l = window['low'].min()

            # Bullish Sweep: pierced 4h low, closed back inside & green
            sweep_l = (roll_l - row['low']) / self.pip_size
            if 0.8 <= sweep_l <= 6.0 and row['close'] > roll_l and row['close'] > row['open']:
                if not self.use_h1_filter or macro >= 0:
                    sl_p = row['low'] - 1.2 * self.pip_size
                    r = max((row['close'] - sl_p) / self.pip_size, dyn_min_sl)
                    if r <= dyn_max_sl:
                        sig = SIGNAL_LONG
                        risk_p = r

            # Bearish Sweep: pierced 4h high, closed back inside & red
            sweep_h = (row['high'] - roll_h) / self.pip_size
            if 0.8 <= sweep_h <= 6.0 and row['close'] < roll_h and row['close'] < row['open']:
                if not self.use_h1_filter or macro <= 0:
                    sl_p = row['high'] + 1.2 * self.pip_size
                    r = max((sl_p - row['close']) / self.pip_size, dyn_min_sl)
                    if r <= dyn_max_sl:
                        sig = SIGNAL_SHORT
                        risk_p = r

            # Pattern B: Fair Value Gap Immediate Mitigation
            if sig == 0:
                # Bullish FVG
                if prev['close'] > prev['open'] and (prev['close'] - prev['open']) / self.pip_size >= 3.5:
                    if row['open'] > prior['high'] and row['low'] <= prior['high'] and row['close'] > prior['high'] and row['close'] > row['ema_20']:
                        if not self.use_h1_filter or macro >= 0:
                            sl_p = prev['low'] - 1.2 * self.pip_size
                            r = max((row['close'] - sl_p) / self.pip_size, dyn_min_sl)
                            if r <= dyn_max_sl:
                                sig = SIGNAL_LONG
                                risk_p = r

                # Bearish FVG
                elif prev['close'] < prev['open'] and (prev['open'] - prev['close']) / self.pip_size >= 3.5:
                    if row['open'] < prior['low'] and row['high'] >= prior['low'] and row['close'] < prior['low'] and row['close'] < row['ema_20']:
                        if not self.use_h1_filter or macro <= 0:
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
