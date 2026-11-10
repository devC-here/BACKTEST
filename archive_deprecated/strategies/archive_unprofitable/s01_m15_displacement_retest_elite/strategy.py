import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

from tick_engine.strategies.base import BaseStrategy, SIGNAL_LONG, SIGNAL_SHORT

class S01M15DisplacementRetestElite(BaseStrategy):
    """
    S01 M15 Displacement & Structure Retest Scalper (Elite Edition)
    --------------------------------------------------------------
    Timeframe: M15
    Active Trading Window: 07:00 - 16:59 UTC (European & US active liquidity)
    Target RR: 2.8 : 1
    Minimum Displacement: 4.0 pips
    HTF Trend Filter: M15 EMA 50 vs EMA 200 (macro alignment)
    Execution: True candle-close millisecond timestamp lookup in tick stream
    """
    def __init__(
        self,
        target_rr: float = 2.8,
        min_displacement_pips: float = 4.0,
        use_h1_trend: bool = True,
        min_atr_pips: float = 5.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_rr = float(target_rr)
        self.min_disp = float(min_displacement_pips)
        self.use_h1_trend = bool(use_h1_trend)
        self.min_atr = float(min_atr_pips)
        self.pip_size = 0.0001
        self.allowed_hours = set(range(7, 17)) # 07:00 to 16:59 UTC

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
        candles = base_df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(candles) < 60:
            return signals, sl_pips_arr, tp_pips_arr

        c = candles['close']
        h = candles['high']
        l = candles['low']
        
        # M15 True Range / ATR calculation
        tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
        candles['atr'] = tr.rolling(14).mean()
        candles['ema_50'] = c.ewm(span=50, adjust=False).mean()
        candles['ema_200'] = c.ewm(span=200, adjust=False).mean()

        time_ms = candles.index.astype('datetime64[ms]').view('int64')
        dur_ms = 15 * 60 * 1000
        tick_ts = ticks['timestamp']

        for i in range(25, len(candles)):
            row = candles.iloc[i]
            c1 = candles.iloc[i - 1] # Displacement bar
            c2 = candles.iloc[i - 2] # Structure reference candle

            hour = row.name.hour
            weekday = row.name.dayofweek
            if hour not in self.allowed_hours:
                continue
            if weekday == 4 and hour >= 15:
                continue
            if pd.isna(row['atr']) or (row['atr'] / self.pip_size) < self.min_atr:
                continue

            # Trend bias alignment
            h1_bull = (row['ema_50'] > row['ema_200']) if self.use_h1_trend else True
            h1_bear = (row['ema_50'] < row['ema_200']) if self.use_h1_trend else True

            # Bullish Displacement & Retest:
            disp_pips = (c1['close'] - c1['open']) / self.pip_size
            if h1_bull and disp_pips >= self.min_disp and c1['close'] > c2['high']:
                retest_level = c2['high']
                if row['low'] <= retest_level and row['close'] > retest_level and row['close'] > row['open']:
                    sl_price = c1['low'] - 1.5 * self.pip_size
                    risk = (row['close'] - sl_price) / self.pip_size
                    if 5.0 <= risk <= 14.0:
                        exec_t = time_ms[i] + dur_ms
                        idx = int(np.searchsorted(tick_ts, exec_t))
                        if idx < n_ticks and signals[idx] == 0:
                            signals[idx] = SIGNAL_LONG
                            sl_pips_arr[idx] = risk
                            tp_pips_arr[idx] = risk * self.target_rr

            # Bearish Displacement & Retest:
            disp_pips = (c1['open'] - c1['close']) / self.pip_size
            if h1_bear and disp_pips >= self.min_disp and c1['close'] < c2['low']:
                retest_level = c2['low']
                if row['high'] >= retest_level and row['close'] < retest_level and row['close'] < row['open']:
                    sl_price = c1['high'] + 1.5 * self.pip_size
                    risk = (sl_price - row['close']) / self.pip_size
                    if 5.0 <= risk <= 14.0:
                        exec_t = time_ms[i] + dur_ms
                        idx = int(np.searchsorted(tick_ts, exec_t))
                        if idx < n_ticks and signals[idx] == 0:
                            signals[idx] = SIGNAL_SHORT
                            sl_pips_arr[idx] = risk
                            tp_pips_arr[idx] = risk * self.target_rr

        return signals, sl_pips_arr, tp_pips_arr
