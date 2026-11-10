"""
Renko Brick Generator and Indicator Calculator.
Constructs authentic price-driven Renko bricks from raw tick bid/ask streams,
tracking tick volume, 10-period SMA, and On-Balance Volume (OBV).
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd


class RenkoBrick:
    __slots__ = ('brick_idx', 'tick_idx', 'timestamp', 'open', 'close', 'high', 'low', 'direction', 'volume')

    def __init__(self, brick_idx: int, tick_idx: int, timestamp: int, open_p: float, close_p: float, high_p: float, low_p: float, direction: int, volume: int):
        self.brick_idx = brick_idx
        self.tick_idx = tick_idx
        self.timestamp = timestamp
        self.open = open_p
        self.close = close_p
        self.high = high_p
        self.low = low_p
        self.direction = direction  # +1 for Up/Green, -1 for Down/Red
        self.volume = volume


def build_renko_bricks(
    ticks: np.ndarray,
    brick_size_pips: float = 5.0,
    pip_size: float = 0.0001
) -> List[RenkoBrick]:
    """
    Constructs standard Renko bricks from ticks:
    - Continuation: price advances by 1 brick_size beyond previous brick close
    - Reversal: price reverses by 2 brick_sizes from previous brick close (1 brick size past previous open)
    """
    n = len(ticks)
    if n == 0:
        return []

    brick_size = brick_size_pips * pip_size
    bids = ticks['bid']
    asks = ticks['ask']
    mids = 0.5 * (bids + asks)
    ts = ticks['timestamp']

    # Initial reference
    first_mid = mids[0]
    base_price = np.floor(first_mid / brick_size) * brick_size

    bricks: List[RenkoBrick] = []
    current_direction = 0  # 0: undefined, +1: up, -1: down
    last_close = base_price
    last_brick_tick = 0

    for i in range(n):
        price = mids[i]

        if current_direction == 0:
            # Establishing initial brick
            if price >= base_price + brick_size:
                vol = i - last_brick_tick + 1
                b = RenkoBrick(
                    brick_idx=len(bricks),
                    tick_idx=i,
                    timestamp=int(ts[i]),
                    open_p=base_price,
                    close_p=base_price + brick_size,
                    high_p=base_price + brick_size,
                    low_p=base_price,
                    direction=1,
                    volume=vol
                )
                bricks.append(b)
                current_direction = 1
                last_close = b.close
                last_brick_tick = i
            elif price <= base_price - brick_size:
                vol = i - last_brick_tick + 1
                b = RenkoBrick(
                    brick_idx=len(bricks),
                    tick_idx=i,
                    timestamp=int(ts[i]),
                    open_p=base_price,
                    close_p=base_price - brick_size,
                    high_p=base_price,
                    low_p=base_price - brick_size,
                    direction=-1,
                    volume=vol
                )
                bricks.append(b)
                current_direction = -1
                last_close = b.close
                last_brick_tick = i

        elif current_direction == 1:
            # In an Up trend
            # Check for Up continuation
            while price >= last_close + brick_size:
                vol = i - last_brick_tick + 1
                b = RenkoBrick(
                    brick_idx=len(bricks),
                    tick_idx=i,
                    timestamp=int(ts[i]),
                    open_p=last_close,
                    close_p=last_close + brick_size,
                    high_p=last_close + brick_size,
                    low_p=last_close,
                    direction=1,
                    volume=vol
                )
                bricks.append(b)
                last_close = b.close
                last_brick_tick = i

            # Check for Down reversal (requires 2 brick sizes below top close)
            if price <= last_close - 2.0 * brick_size:
                # Reversal brick
                new_open = last_close - brick_size
                new_close = last_close - 2.0 * brick_size
                vol = i - last_brick_tick + 1
                b = RenkoBrick(
                    brick_idx=len(bricks),
                    tick_idx=i,
                    timestamp=int(ts[i]),
                    open_p=new_open,
                    close_p=new_close,
                    high_p=new_open,
                    low_p=new_close,
                    direction=-1,
                    volume=vol
                )
                bricks.append(b)
                current_direction = -1
                last_close = b.close
                last_brick_tick = i

                # Possible further down continuation within same tick
                while price <= last_close - brick_size:
                    vol = 1
                    b_cont = RenkoBrick(
                        brick_idx=len(bricks),
                        tick_idx=i,
                        timestamp=int(ts[i]),
                        open_p=last_close,
                        close_p=last_close - brick_size,
                        high_p=last_close,
                        low_p=last_close - brick_size,
                        direction=-1,
                        volume=vol
                    )
                    bricks.append(b_cont)
                    last_close = b_cont.close

        elif current_direction == -1:
            # In a Down trend
            # Check for Down continuation
            while price <= last_close - brick_size:
                vol = i - last_brick_tick + 1
                b = RenkoBrick(
                    brick_idx=len(bricks),
                    tick_idx=i,
                    timestamp=int(ts[i]),
                    open_p=last_close,
                    close_p=last_close - brick_size,
                    high_p=last_close,
                    low_p=last_close - brick_size,
                    direction=-1,
                    volume=vol
                )
                bricks.append(b)
                last_close = b.close
                last_brick_tick = i

            # Check for Up reversal (requires 2 brick sizes above bottom close)
            if price >= last_close + 2.0 * brick_size:
                # Reversal brick
                new_open = last_close + brick_size
                new_close = last_close + 2.0 * brick_size
                vol = i - last_brick_tick + 1
                b = RenkoBrick(
                    brick_idx=len(bricks),
                    tick_idx=i,
                    timestamp=int(ts[i]),
                    open_p=new_open,
                    close_p=new_close,
                    high_p=new_close,
                    low_p=new_open,
                    direction=1,
                    volume=vol
                )
                bricks.append(b)
                current_direction = 1
                last_close = b.close
                last_brick_tick = i

                # Possible further up continuation within same tick
                while price >= last_close + brick_size:
                    vol = 1
                    b_cont = RenkoBrick(
                        brick_idx=len(bricks),
                        tick_idx=i,
                        timestamp=int(ts[i]),
                        open_p=last_close,
                        close_p=last_close + brick_size,
                        high_p=last_close + brick_size,
                        low_p=last_close,
                        direction=1,
                        volume=vol
                    )
                    bricks.append(b_cont)
                    last_close = b_cont.close

    return bricks


def compute_renko_indicators(
    bricks: List[RenkoBrick],
    sma_period: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes:
    - closes: array of brick close prices
    - sma: Simple Moving Average of brick closes
    - obv: On-Balance Volume based on brick direction and tick volume
    """
    n = len(bricks)
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    closes = np.array([b.close for b in bricks], dtype=np.float64)
    volumes = np.array([b.volume for b in bricks], dtype=np.float64)
    directions = np.array([b.direction for b in bricks], dtype=np.int8)

    # 1. 10 SMA of brick closes
    sma = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if i >= sma_period - 1:
            sma[i] = np.mean(closes[i - sma_period + 1 : i + 1])

    # 2. On-Balance Volume (OBV)
    obv = np.zeros(n, dtype=np.float64)
    cur_obv = 0.0
    for i in range(n):
        if directions[i] == 1:
            cur_obv += volumes[i]
        elif directions[i] == -1:
            cur_obv -= volumes[i]
        obv[i] = cur_obv

    return closes, sma, obv
