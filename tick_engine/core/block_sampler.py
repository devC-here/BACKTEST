#!/usr/bin/env python3
"""
SOP 6-Month Unit Block Sampler (SOP-TRD-VAL-5Y-001).
Divides the 5-Year dataset (262 weeks / 60 months) into 10 discrete 6-Month Units.
Selects 6 random units for Training & Building (3 Years) and 4 random units for Unseen Testing (2 Years).
"""

from __future__ import annotations
import random
from typing import List, Dict, Tuple, Any

# Discrete 10 Six-Month Units covering the full 5-Year Tick Dataset (262 Weeks)
# Each unit represents exactly ~6 months of continuous trading
BLOCK_UNITS: Dict[int, Dict[str, Any]] = {
    1: {
        "id": 1,
        "name": "B01 (2021-H2/22-Q1)",
        "start_week": 1,
        "end_week": 26,
        "weeks_count": 26,
        "start_date": "2021-10-01",
        "end_date": "2022-03-25",
        "regime": "Post-COVID recovery / Rate hike cycle onset",
        "tick_volume": 9_197_603
    },
    2: {
        "id": 2,
        "name": "B02 (2022-H1)",
        "start_week": 27,
        "end_week": 52,
        "weeks_count": 26,
        "start_date": "2022-03-28",
        "end_date": "2022-09-23",
        "regime": "Ukraine War / EUR Parity Crash (High Trend Volatility)",
        "tick_volume": 16_730_617
    },
    3: {
        "id": 3,
        "name": "B03 (2022-H2/23-Q1)",
        "start_week": 53,
        "end_week": 78,
        "weeks_count": 26,
        "start_date": "2022-09-26",
        "end_date": "2023-03-24",
        "regime": "Aggressive ECB/Fed Tightening / Major Volatility",
        "tick_volume": 21_306_649
    },
    4: {
        "id": 4,
        "name": "B04 (2023-H1)",
        "start_week": 79,
        "end_week": 104,
        "weeks_count": 26,
        "start_date": "2023-03-27",
        "end_date": "2023-09-22",
        "regime": "US Regional Banking Crisis / Consolidation",
        "tick_volume": 11_915_028
    },
    5: {
        "id": 5,
        "name": "B05 (2023-H2/24-Q1)",
        "start_week": 105,
        "end_week": 130,
        "weeks_count": 26,
        "start_date": "2023-09-25",
        "end_date": "2024-03-22",
        "regime": "Peak Policy Rates / Range-Bound Mean Reversion",
        "tick_volume": 11_594_685
    },
    6: {
        "id": 6,
        "name": "B06 (2024-H1)",
        "start_week": 131,
        "end_week": 156,
        "weeks_count": 26,
        "start_date": "2024-03-25",
        "end_date": "2024-09-20",
        "regime": "Disinflation Cycle / Macro Transition",
        "tick_volume": 8_925_193
    },
    7: {
        "id": 7,
        "name": "B07 (2024-H2/25-Q1)",
        "start_week": 157,
        "end_week": 182,
        "weeks_count": 26,
        "start_date": "2024-09-23",
        "end_date": "2025-03-21",
        "regime": "Global Rate Cuts Begin / Monetary Easing",
        "tick_volume": 12_663_910
    },
    8: {
        "id": 8,
        "name": "B08 (2025-H1)",
        "start_week": 183,
        "end_week": 208,
        "weeks_count": 26,
        "start_date": "2025-03-24",
        "end_date": "2025-09-19",
        "regime": "Balanced Volatility / Clean London Expansion",
        "tick_volume": 12_612_686
    },
    9: {
        "id": 9,
        "name": "B09 (2025-H2/26-Q1)",
        "start_week": 209,
        "end_week": 235,
        "weeks_count": 27,
        "start_date": "2025-09-22",
        "end_date": "2026-03-27",
        "regime": "Normalized Pullback Trends / Stable Ranges",
        "tick_volume": 9_595_360
    },
    10: {
        "id": 10,
        "name": "B10 (2026-H1)",
        "start_week": 236,
        "end_week": 262,
        "weeks_count": 27,
        "start_date": "2026-03-30",
        "end_date": "2026-10-01",
        "regime": "Modern Market Regime / High Liquidity",
        "tick_volume": 8_681_932
    }
}


def sample_train_test_blocks(seed: int = 42) -> Tuple[List[int], List[int]]:
    """
    Randomly splits the 10 six-month units into:
    - 6 Random Units for Training (36 Months / 3 Years)
    - 4 Random Units for Testing (24 Months / 2 Years)
    Uses a deterministic seed for reproducibility.
    """
    rng = random.Random(seed)
    all_blocks = list(range(1, 11))
    train_blocks = sorted(rng.sample(all_blocks, 6))
    test_blocks = sorted([b for b in all_blocks if b not in train_blocks])
    return train_blocks, test_blocks


def get_weeks_for_blocks(block_ids: List[int]) -> List[int]:
    """
    Returns sorted list of all week numbers included in the specified blocks.
    """
    weeks = set()
    for b in block_ids:
        if b in BLOCK_UNITS:
            info = BLOCK_UNITS[b]
            for w in range(info["start_week"], info["end_week"] + 1):
                weeks.add(w)
    return sorted(list(weeks))


def format_block_summary(block_ids: List[int]) -> str:
    """
    Formats a user-friendly string summary of the blocks.
    """
    names = [f"B{b:02d}" for b in block_ids]
    wks = get_weeks_for_blocks(block_ids)
    ticks = sum(BLOCK_UNITS[b]["tick_volume"] for b in block_ids if b in BLOCK_UNITS)
    return f"{', '.join(names)} ({len(block_ids)} Blocks / {len(wks)} Weeks / {ticks:,} Ticks)"
