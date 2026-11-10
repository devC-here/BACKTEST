#!/usr/bin/env python3
"""
Tick Engine Core Configuration & Instrument Profiles
=====================================================
Live ECN Broker Profiles ($4.00/Lot Commission + Slippage + Spread)
"""

from pathlib import Path
from typing import Dict, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SPLITS_FILE = ROOT_DIR / "data_splits" / "master_chronological_splits.json"

DEFAULT_DOLLAR_PER_LOT = 4.0
DEFAULT_ENTRY_SLIPPAGE_PIPS = 0.20
WARMUP_TICKS = 100000

PIP_SIZES: Dict[str, float] = {
    "GBPJPY": 0.01,
    "USDJPY": 0.01,
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDCAD": 0.0001,
    "GBPCHF": 0.0001,
    "AUDCAD": 0.0001,
    "EURNZD": 0.0001,
    "AUDNZD": 0.0001,
    "EURGBP": 0.0001,
    "NZDCAD": 0.0001,
    "AUDUSD": 0.0001,
    "NZDUSD": 0.0001,
    "USDCHF": 0.0001,
}

PIP_VALUES_USD: Dict[str, float] = {
    "EURUSD": 10.00,
    "GBPUSD": 10.00,
    "AUDUSD": 10.00,
    "NZDUSD": 10.00,
    "USDJPY": 6.34,
    "GBPJPY": 6.34,
    "USDCAD": 7.12,
    "AUDCAD": 7.12,
    "NZDCAD": 7.12,
    "USDCHF": 12.25,
    "GBPCHF": 12.25,
    "EURGBP": 11.99,
    "EURNZD": 5.56,
    "AUDNZD": 5.56,
}


def get_friction_pips(symbol: str, dollar_per_lot: float = DEFAULT_DOLLAR_PER_LOT) -> Tuple[float, float]:
    """
    Computes (commission_pips, entry_slippage_pips) matching the institutional $4.00/lot standard:
    - Commission: exact pip equivalent of $4.00 per standard lot based on quote currency value
    - Slippage: 0.20 pips adverse entry fill penalty
    """
    s = symbol.upper()
    pip_val_usd = PIP_VALUES_USD.get(s, 10.0)
    comm_pips = dollar_per_lot / pip_val_usd
    return round(comm_pips, 2), DEFAULT_ENTRY_SLIPPAGE_PIPS
