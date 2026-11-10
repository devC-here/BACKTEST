"""
Alpha Searcher Package: Institutional Zero-Bias Alpha Discovery Suite
Configured for $4.00 per standard lot institutional friction.
"""

from alpha_searcher.config import (
    DEFAULT_DOLLAR_PER_LOT,
    WARMUP_TICKS,
    PIP_SIZES,
    MAX_SPREADS,
    get_friction_pips,
    get_total_friction_pips,
)
from alpha_searcher.archetypes import generate_candidate_battery
from alpha_searcher.searcher import run_alpha_search

__all__ = [
    "DEFAULT_DOLLAR_PER_LOT",
    "WARMUP_TICKS",
    "PIP_SIZES",
    "MAX_SPREADS",
    "get_friction_pips",
    "get_total_friction_pips",
    "generate_candidate_battery",
    "run_alpha_search",
]
