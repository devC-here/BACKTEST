"""
High-Speed Numba JIT Order Matching and Simulation Engine.
Implements pip-denominated execution rules with spread awareness
and contiguous pre-allocated memory buffers.
"""

from __future__ import annotations
from typing import Any, Union, Tuple, Optional
import numpy as np
try:
    import numba
    _njit = numba.njit(fastmath=True, nogil=True)
except Exception:
    def _njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def dec(fn):
            return fn
        return dec

# Data schemas strictly adhering to architecture specification
TICK_DTYPE = np.dtype([
    ('timestamp', np.int64),    # Unix epoch (milliseconds or nanoseconds)
    ('bid', np.float64),        # Current bid price
    ('ask', np.float64)         # Current ask price
])

TRADE_DTYPE = np.dtype([
    ('signal_time', np.int64),
    ('entry_time', np.int64),
    ('exit_time', np.int64),
    ('direction', np.int8),     # 1 = Long, -1 = Short
    ('entry_price', np.float64),
    ('exit_price', np.float64),
    ('pips', np.float64),
    ('exit_reason', np.int8)    # 1 = SL, 2 = TP, 3 = Signal Exit
])

# Execution constants
EXIT_REASON_SL = 1
EXIT_REASON_TP = 2
EXIT_REASON_SIGNAL = 3

SIGNAL_HOLD = 0
SIGNAL_LONG = 1
SIGNAL_SHORT = -1
SIGNAL_FORCE_EXIT = 2

# Institutional Default Friction Constants ($4.00/lot flat commission + 0.20p adverse slippage)
DEFAULT_COMMISSION_PIPS = 0.40
DEFAULT_ENTRY_SLIPPAGE_PIPS = 0.20


@_njit
def run_matching_loop_signals(
    timestamps: np.ndarray,
    bids: np.ndarray,
    asks: np.ndarray,
    signals: np.ndarray,
    sl_pips_arr: np.ndarray,
    tp_pips_arr: np.ndarray,
    pip_size: float,
    use_slippage: bool,
    close_at_end: bool,
    trades_buffer: np.ndarray,
    commission_pips: float = 0.0,
    entry_slippage_pips: float = 0.0,
    signal_timestamps: np.ndarray = np.empty(0, dtype=np.int64),
    min_entry_timestamp: np.int64 = 0,
    enforce_subsequent_tick: bool = False
) -> int:
    """
    Numba JIT accelerated matching loop for pre-computed signal arrays.
    Supports fixed or dynamic per-tick Stop Loss and Take Profit levels,
    along with institutional commission penalties and entry slippage.
    
    Order-Timing Contract & Causal Warmup:
    - signal_timestamps: per-tick or per-bar signal generation timestamps.
    - min_entry_timestamp: entries before this timestamp are prohibited (warmup gate).
    - enforce_subsequent_tick: if True, signals at tick i cannot fill at tick i,
      they fill at tick i+1 or subsequent tick where ts > signal_time.
    - Every executed trade records both signal_time and entry_time in trades_buffer.
    """
    n = len(timestamps)
    max_trades = len(trades_buffer)
    trade_count = 0

    pos_dir = 0          # 0 = Flat, 1 = Long, -1 = Short
    entry_time = 0
    signal_time = 0
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0

    pending_dir = 0
    pending_sig_time = 0
    pending_sl_pips = 0.0
    pending_tp_pips = 0.0

    has_sig_ts = len(signal_timestamps) == n

    for i in range(n):
        ts = timestamps[i]
        b = bids[i]
        a = asks[i]

        # 0. Execute pending orders waiting for a subsequent executable tick
        if pending_dir != 0:
            if ts > pending_sig_time and ts >= min_entry_timestamp:
                if pending_dir == 1:
                    pos_dir = 1
                    signal_time = pending_sig_time
                    entry_time = ts
                    entry_price = a + (entry_slippage_pips * pip_size)
                    cur_sl = pending_sl_pips
                    cur_tp = pending_tp_pips
                    sl_price = entry_price - (cur_sl * pip_size) if cur_sl > 0.0 else 0.0
                    tp_price = entry_price + (cur_tp * pip_size) if cur_tp > 0.0 else 0.0
                elif pending_dir == -1:
                    pos_dir = -1
                    signal_time = pending_sig_time
                    entry_time = ts
                    entry_price = b - (entry_slippage_pips * pip_size)
                    cur_sl = pending_sl_pips
                    cur_tp = pending_tp_pips
                    sl_price = entry_price + (cur_sl * pip_size) if cur_sl > 0.0 else 0.0
                    tp_price = entry_price - (cur_tp * pip_size) if cur_tp > 0.0 else 0.0
                pending_dir = 0

        # 1. Check open SL / TP against current tick Bid/Ask
        if pos_dir == 1:
            # Active Long
            if sl_price > 0.0 and b <= (sl_price + 1e-9):
                exit_price = b if (use_slippage and b < sl_price) else sl_price
                pips = ((exit_price - entry_price) / pip_size) - commission_pips
                if trade_count < max_trades:
                    trades_buffer[trade_count]['signal_time'] = signal_time
                    trades_buffer[trade_count]['entry_time'] = entry_time
                    trades_buffer[trade_count]['exit_time'] = ts
                    trades_buffer[trade_count]['direction'] = 1
                    trades_buffer[trade_count]['entry_price'] = entry_price
                    trades_buffer[trade_count]['exit_price'] = exit_price
                    trades_buffer[trade_count]['pips'] = pips
                    trades_buffer[trade_count]['exit_reason'] = EXIT_REASON_SL
                    trade_count += 1
                pos_dir = 0

            elif tp_price > 0.0 and b >= (tp_price - 1e-9):
                exit_price = b if (use_slippage and b > tp_price) else tp_price
                pips = ((exit_price - entry_price) / pip_size) - commission_pips
                if trade_count < max_trades:
                    trades_buffer[trade_count]['signal_time'] = signal_time
                    trades_buffer[trade_count]['entry_time'] = entry_time
                    trades_buffer[trade_count]['exit_time'] = ts
                    trades_buffer[trade_count]['direction'] = 1
                    trades_buffer[trade_count]['entry_price'] = entry_price
                    trades_buffer[trade_count]['exit_price'] = exit_price
                    trades_buffer[trade_count]['pips'] = pips
                    trades_buffer[trade_count]['exit_reason'] = EXIT_REASON_TP
                    trade_count += 1
                pos_dir = 0

        elif pos_dir == -1:
            # Active Short
            if sl_price > 0.0 and a >= (sl_price - 1e-9):
                exit_price = a if (use_slippage and a > sl_price) else sl_price
                pips = ((entry_price - exit_price) / pip_size) - commission_pips
                if trade_count < max_trades:
                    trades_buffer[trade_count]['signal_time'] = signal_time
                    trades_buffer[trade_count]['entry_time'] = entry_time
                    trades_buffer[trade_count]['exit_time'] = ts
                    trades_buffer[trade_count]['direction'] = -1
                    trades_buffer[trade_count]['entry_price'] = entry_price
                    trades_buffer[trade_count]['exit_price'] = exit_price
                    trades_buffer[trade_count]['pips'] = pips
                    trades_buffer[trade_count]['exit_reason'] = EXIT_REASON_SL
                    trade_count += 1
                pos_dir = 0

            elif tp_price > 0.0 and a <= (tp_price + 1e-9):
                exit_price = a if (use_slippage and a < tp_price) else tp_price
                pips = ((entry_price - exit_price) / pip_size) - commission_pips
                if trade_count < max_trades:
                    trades_buffer[trade_count]['signal_time'] = signal_time
                    trades_buffer[trade_count]['entry_time'] = entry_time
                    trades_buffer[trade_count]['exit_time'] = ts
                    trades_buffer[trade_count]['direction'] = -1
                    trades_buffer[trade_count]['entry_price'] = entry_price
                    trades_buffer[trade_count]['exit_price'] = exit_price
                    trades_buffer[trade_count]['pips'] = pips
                    trades_buffer[trade_count]['exit_reason'] = EXIT_REASON_TP
                    trade_count += 1
                pos_dir = 0

        # 2. Invoke Strategy Signal (Flat / Long / Short / Force Exit)
        sig = signals[i]
        if sig != 0:
            sig_ts = signal_timestamps[i] if has_sig_ts else ts

            # Warmup exclusion: Discard any new entry signals originating before min_entry_timestamp
            if sig_ts < min_entry_timestamp and sig != SIGNAL_FORCE_EXIT:
                sig = 0

        if sig != 0:
            if pos_dir != 0:
                if sig == SIGNAL_FORCE_EXIT:
                    if pos_dir == 1:
                        exit_price = b
                        pips = ((exit_price - entry_price) / pip_size) - commission_pips
                    else:
                        exit_price = a
                        pips = ((entry_price - exit_price) / pip_size) - commission_pips

                    if trade_count < max_trades:
                        trades_buffer[trade_count]['signal_time'] = signal_time
                        trades_buffer[trade_count]['entry_time'] = entry_time
                        trades_buffer[trade_count]['exit_time'] = ts
                        trades_buffer[trade_count]['direction'] = pos_dir
                        trades_buffer[trade_count]['entry_price'] = entry_price
                        trades_buffer[trade_count]['exit_price'] = exit_price
                        trades_buffer[trade_count]['pips'] = pips
                        trades_buffer[trade_count]['exit_reason'] = EXIT_REASON_SIGNAL
                        trade_count += 1
                    pos_dir = 0

                elif sig == SIGNAL_SHORT and pos_dir == 1:
                    # Reversal: Close active Long at current Bid
                    exit_price = b
                    pips = ((exit_price - entry_price) / pip_size) - commission_pips
                    if trade_count < max_trades:
                        trades_buffer[trade_count]['signal_time'] = signal_time
                        trades_buffer[trade_count]['entry_time'] = entry_time
                        trades_buffer[trade_count]['exit_time'] = ts
                        trades_buffer[trade_count]['direction'] = 1
                        trades_buffer[trade_count]['entry_price'] = entry_price
                        trades_buffer[trade_count]['exit_price'] = exit_price
                        trades_buffer[trade_count]['pips'] = pips
                        trades_buffer[trade_count]['exit_reason'] = EXIT_REASON_SIGNAL
                        trade_count += 1
                    pos_dir = 0

                    # Open Short: Check if same-tick fill is permitted and strictly causal
                    can_fill_now = (not enforce_subsequent_tick) and ((not has_sig_ts) or (ts > sig_ts)) and (ts >= min_entry_timestamp)
                    if can_fill_now:
                        pos_dir = -1
                        signal_time = sig_ts
                        entry_time = ts
                        entry_price = b - (entry_slippage_pips * pip_size)
                        cur_sl = sl_pips_arr[i]
                        cur_tp = tp_pips_arr[i]
                        sl_price = entry_price + (cur_sl * pip_size) if cur_sl > 0.0 else 0.0
                        tp_price = entry_price - (cur_tp * pip_size) if cur_tp > 0.0 else 0.0
                    else:
                        pending_dir = -1
                        pending_sig_time = sig_ts
                        pending_sl_pips = sl_pips_arr[i]
                        pending_tp_pips = tp_pips_arr[i]

                elif sig == SIGNAL_LONG and pos_dir == -1:
                    # Reversal: Close active Short at current Ask
                    exit_price = a
                    pips = ((entry_price - exit_price) / pip_size) - commission_pips
                    if trade_count < max_trades:
                        trades_buffer[trade_count]['signal_time'] = signal_time
                        trades_buffer[trade_count]['entry_time'] = entry_time
                        trades_buffer[trade_count]['exit_time'] = ts
                        trades_buffer[trade_count]['direction'] = -1
                        trades_buffer[trade_count]['entry_price'] = entry_price
                        trades_buffer[trade_count]['exit_price'] = exit_price
                        trades_buffer[trade_count]['pips'] = pips
                        trades_buffer[trade_count]['exit_reason'] = EXIT_REASON_SIGNAL
                        trade_count += 1
                    pos_dir = 0

                    # Open Long: Check if same-tick fill is permitted and strictly causal
                    can_fill_now = (not enforce_subsequent_tick) and ((not has_sig_ts) or (ts > sig_ts)) and (ts >= min_entry_timestamp)
                    if can_fill_now:
                        pos_dir = 1
                        signal_time = sig_ts
                        entry_time = ts
                        entry_price = a + (entry_slippage_pips * pip_size)
                        cur_sl = sl_pips_arr[i]
                        cur_tp = tp_pips_arr[i]
                        sl_price = entry_price - (cur_sl * pip_size) if cur_sl > 0.0 else 0.0
                        tp_price = entry_price + (cur_tp * pip_size) if cur_tp > 0.0 else 0.0
                    else:
                        pending_dir = 1
                        pending_sig_time = sig_ts
                        pending_sl_pips = sl_pips_arr[i]
                        pending_tp_pips = tp_pips_arr[i]

            elif pos_dir == 0 and pending_dir == 0:
                can_fill_now = (not enforce_subsequent_tick) and ((not has_sig_ts) or (ts > sig_ts)) and (ts >= min_entry_timestamp)
                if can_fill_now:
                    if sig == SIGNAL_LONG:
                        pos_dir = 1
                        signal_time = sig_ts
                        entry_time = ts
                        entry_price = a + (entry_slippage_pips * pip_size)
                        cur_sl = sl_pips_arr[i]
                        cur_tp = tp_pips_arr[i]
                        sl_price = entry_price - (cur_sl * pip_size) if cur_sl > 0.0 else 0.0
                        tp_price = entry_price + (cur_tp * pip_size) if cur_tp > 0.0 else 0.0

                    elif sig == SIGNAL_SHORT:
                        pos_dir = -1
                        signal_time = sig_ts
                        entry_time = ts
                        entry_price = b - (entry_slippage_pips * pip_size)
                        cur_sl = sl_pips_arr[i]
                        cur_tp = tp_pips_arr[i]
                        sl_price = entry_price + (cur_sl * pip_size) if cur_sl > 0.0 else 0.0
                        tp_price = entry_price - (cur_tp * pip_size) if cur_tp > 0.0 else 0.0
                else:
                    pending_dir = sig
                    pending_sig_time = sig_ts
                    pending_sl_pips = sl_pips_arr[i]
                    pending_tp_pips = tp_pips_arr[i]

    # 3. Clean close-out at dataset end if configured
    if close_at_end and pos_dir != 0 and n > 0:
        last_ts = timestamps[n - 1]
        if pos_dir == 1:
            exit_price = bids[n - 1]
            pips = ((exit_price - entry_price) / pip_size) - commission_pips
        else:
            exit_price = asks[n - 1]
            pips = ((entry_price - exit_price) / pip_size) - commission_pips

        if trade_count < max_trades:
            trades_buffer[trade_count]['signal_time'] = signal_time
            trades_buffer[trade_count]['entry_time'] = entry_time
            trades_buffer[trade_count]['exit_time'] = last_ts
            trades_buffer[trade_count]['direction'] = pos_dir
            trades_buffer[trade_count]['entry_price'] = entry_price
            trades_buffer[trade_count]['exit_price'] = exit_price
            trades_buffer[trade_count]['pips'] = pips
            trades_buffer[trade_count]['exit_reason'] = EXIT_REASON_SIGNAL
            trade_count += 1

    return trade_count


def run_matching_loop(
    ticks: np.ndarray,
    signals: np.ndarray,
    pip_size: float = 0.0001,
    sl_pips: Any = 20.0,
    tp_pips: Any = 40.0,
    use_slippage: bool = True,
    close_at_end: bool = True,
    max_trades: int = 100_000,
    commission_pips: float = DEFAULT_COMMISSION_PIPS,
    entry_slippage_pips: float = DEFAULT_ENTRY_SLIPPAGE_PIPS,
    spread_pips: Optional[float] = None,
    signal_timestamps: Optional[np.ndarray] = None,
    min_entry_timestamp: int = 0,
    enforce_subsequent_tick: bool = False
) -> np.ndarray:
    """
    Python wrapper that allocates contiguous output buffer and triggers
    the fast Numba JIT simulation loop.
    Supports either scalar floats or per-tick arrays for sl_pips and tp_pips,
    with options for commission and slippage penalties.
    If spread_pips is specified, applies a realistic broker spread model around
    the historical tick mid-price (mid ± spread/2) instead of raw floating spread.
    
    Order-Timing Contract:
    - signal_timestamps: array of timestamps when signals were triggered.
    - min_entry_timestamp: entries before this timestamp are rejected (warmup exclusion).
    - enforce_subsequent_tick: if True, signals on tick i cannot fill on tick i.
    """
    n = len(ticks)
    timestamps = np.ascontiguousarray(ticks['timestamp'], dtype=np.int64)
    if spread_pips is not None and spread_pips > 0.0:
        spread_val = float(spread_pips) * float(pip_size)
        mid = 0.5 * (ticks['bid'] + ticks['ask'])
        bids = np.ascontiguousarray(mid - 0.5 * spread_val, dtype=np.float64)
        asks = np.ascontiguousarray(mid + 0.5 * spread_val, dtype=np.float64)
    else:
        bids = np.ascontiguousarray(ticks['bid'], dtype=np.float64)
        asks = np.ascontiguousarray(ticks['ask'], dtype=np.float64)
    signals = np.ascontiguousarray(signals, dtype=np.int8)

    if isinstance(sl_pips, np.ndarray) and len(sl_pips) == n:
        sl_pips_arr = np.ascontiguousarray(sl_pips, dtype=np.float64)
    else:
        sl_pips_arr = np.full(n, float(sl_pips), dtype=np.float64)

    if isinstance(tp_pips, np.ndarray) and len(tp_pips) == n:
        tp_pips_arr = np.ascontiguousarray(tp_pips, dtype=np.float64)
    else:
        tp_pips_arr = np.full(n, float(tp_pips), dtype=np.float64)

    if signal_timestamps is not None and len(signal_timestamps) == n:
        sig_ts_arr = np.ascontiguousarray(signal_timestamps, dtype=np.int64)
    else:
        sig_ts_arr = np.empty(0, dtype=np.int64)

    trades_buffer = np.zeros(max_trades, dtype=TRADE_DTYPE)

    count = run_matching_loop_signals(
        timestamps,
        bids,
        asks,
        signals,
        sl_pips_arr,
        tp_pips_arr,
        float(pip_size),
        bool(use_slippage),
        bool(close_at_end),
        trades_buffer,
        float(commission_pips),
        float(entry_slippage_pips),
        sig_ts_arr,
        np.int64(min_entry_timestamp),
        bool(enforce_subsequent_tick)
    )

    return trades_buffer[:count].copy()


class BacktestEngine:
    """
    Engine Controller that orchestrates tick ingestion, strategy execution,
    and performance logging.
    """
    def __init__(
        self,
        pip_size: float = 0.0001,
        sl_pips: float = 20.0,
        tp_pips: float = 40.0,
        use_slippage: bool = True,
        close_at_end: bool = True,
        max_trades: int = 200_000,
        commission_pips: float = DEFAULT_COMMISSION_PIPS,
        entry_slippage_pips: float = DEFAULT_ENTRY_SLIPPAGE_PIPS,
        spread_pips: Optional[float] = None,
        enforce_subsequent_tick: bool = False
    ):
        self.pip_size = pip_size
        self.sl_pips = sl_pips
        self.tp_pips = tp_pips
        self.use_slippage = use_slippage
        self.close_at_end = close_at_end
        self.max_trades = max_trades
        self.commission_pips = commission_pips
        self.entry_slippage_pips = entry_slippage_pips
        self.spread_pips = spread_pips
        self.enforce_subsequent_tick = enforce_subsequent_tick

    def run(
        self,
        ticks: np.ndarray,
        strategy,
        min_entry_timestamp: int = 0
    ) -> np.ndarray:
        """
        Executes strategy against ticks and returns contiguous TRADE_DTYPE array.
        Handles either scalar signals or tuple of (signals, sl_pips, tp_pips)
        or (signals, sl_pips, tp_pips, signal_timestamps).
        """
        res = strategy.generate_signals(ticks)
        sig_ts = None
        if isinstance(res, tuple):
            signals = res[0]
            sl = res[1] if len(res) > 1 and res[1] is not None else getattr(strategy, 'sl_pips', self.sl_pips)
            tp = res[2] if len(res) > 2 and res[2] is not None else getattr(strategy, 'tp_pips', self.tp_pips)
            if len(res) > 3 and res[3] is not None:
                sig_ts = res[3]
        else:
            signals = res
            sl = getattr(strategy, 'sl_pips', self.sl_pips)
            tp = getattr(strategy, 'tp_pips', self.tp_pips)

        comm = getattr(strategy, 'commission_pips', self.commission_pips)
        slip = getattr(strategy, 'entry_slippage_pips', self.entry_slippage_pips)
        spr = getattr(strategy, 'spread_pips', self.spread_pips)
        enf = getattr(strategy, 'enforce_subsequent_tick', self.enforce_subsequent_tick)

        return run_matching_loop(
            ticks=ticks,
            signals=signals,
            pip_size=self.pip_size,
            sl_pips=sl,
            tp_pips=tp,
            use_slippage=self.use_slippage,
            close_at_end=self.close_at_end,
            max_trades=self.max_trades,
            commission_pips=comm,
            entry_slippage_pips=slip,
            spread_pips=spr,
            signal_timestamps=sig_ts,
            min_entry_timestamp=min_entry_timestamp,
            enforce_subsequent_tick=enf
        )
