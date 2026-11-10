#!/usr/bin/env python3
"""
Alpha Searcher Configuration: Live ECN Broker Profile ($4.00/Lot Commission + Live Spreads)
==========================================================================================
Calibrated directly from the live trading terminal market watch:
- Fixed Commission: $4.00 per standard lot round-turn ($2.00 per side)
- Raw ECN Spreads:
    * GBPUSD: 0.1 pips (1 point)
    * USDJPY: 0.1 pips (1 point)
    * EURUSD: 0.2 pips (2 points)
    * AUDUSD: 0.2 pips (2 points)
    * NZDUSD: 0.2 pips (2 points)
    * USDCAD: 0.3 pips (3 points)
    * USDCHF: 0.3 pips (3 points)
    * EURGBP: 0.5 pips (5 points)
    * AUDCAD: 1.0 pips (10 points)
    * NZDCAD: 1.0 pips (10 points)
    * GBPJPY: 1.2 pips (12 points)
    * EURNZD: 1.6 pips (16 points)
"""

from pathlib import Path
from typing import Dict, Tuple, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
SPLITS_FILE = ROOT_DIR / "data_splits" / "master_70_30_splits.json"
RESULTS_DIR = ROOT_DIR / "results"

DEFAULT_DOLLAR_PER_LOT = 4.0
WARMUP_TICKS = 100000

PIP_SIZES: Dict[str, float] = {
    "GBPJPY": 0.01,
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDCAD": 0.0001,
    "GBPCHF": 0.0001,
    "AUDCAD": 0.0001,
    "USDJPY": 0.01,
    "EURNZD": 0.0001,
    "AUDNZD": 0.0001,
    "EURGBP": 0.0001,
    "NZDCAD": 0.0001,
    "AUDUSD": 0.0001,
    "NZDUSD": 0.0001,
    "USDCHF": 0.0001,
}

# Live Broker Terminal Raw Spreads (in pips) directly from MT5 Market Watch:
LIVE_BROKER_SPREADS: Dict[str, float] = {
    "GBPUSD": 0.10,  # 1 point
    "USDCHF": 0.30,  # 3 points
    "USDJPY": 0.10,  # 1 point
    "AUDUSD": 0.20,  # 2 points
    "NZDUSD": 0.20,  # 2 points
    "USDCAD": 0.30,  # 3 points
    "AUDCAD": 1.00,  # 10 points
    "EURGBP": 0.50,  # 5 points
    "EURUSD": 0.20,  # 2 points
    "NZDCAD": 1.00,  # 10 points
    "GBPJPY": 1.00,  # 10 points
    "GBPCHF": 0.80,  # 8 points
    "EURNZD": 1.60,  # 16 points
    "AUDNZD": 1.60,  # 16 points
}

# Practical historical tick spread filter thresholds (to allow normal trading hours in historical Dukascopy data):
MAX_SPREADS: Dict[str, float] = {
    "EURUSD": 1.5,
    "GBPUSD": 1.5,
    "USDJPY": 1.8,
    "USDCAD": 2.0,
    "USDCHF": 2.0,
    "EURGBP": 1.8,
    "AUDCAD": 2.5,
    "NZDCAD": 2.5,
    "GBPJPY": 2.5,
    "GBPCHF": 2.5,
    "EURNZD": 3.5,
    "AUDNZD": 3.5,
}

# Standard 100k lot approximate pip values in USD (calibrated to terminal rates):
PIP_VALUES_USD: Dict[str, float] = {
    "EURUSD": 10.00,
    "GBPUSD": 10.00,
    "AUDUSD": 10.00,
    "NZDUSD": 10.00,
    "USDJPY": 6.34,   # 1,000 JPY @ 157.84
    "GBPJPY": 6.34,
    "USDCAD": 7.12,   # 10 CAD @ 1.4046
    "AUDCAD": 7.12,
    "NZDCAD": 7.12,
    "USDCHF": 12.25,  # 10 CHF @ 0.8166
    "GBPCHF": 12.25,
    "EURGBP": 11.99,  # 10 GBP @ 0.8340
    "EURNZD": 5.56,   # 10 NZD @ 0.5559
    "AUDNZD": 5.56,
}


def get_friction_pips(symbol: str, dollar_per_lot: float = DEFAULT_DOLLAR_PER_LOT) -> Tuple[float, float]:
    """
    Computes (commission_pips, entry_slippage_pips) matching the live broker's $4.00/lot:
    - Commission: exact pip equivalent of $4.00 per standard lot
    - Slippage: 0.20 pips institutional adverse ECN fill penalty
    """
    s = symbol.upper()
    pip_val_usd = PIP_VALUES_USD.get(s, 10.0)
    
    comm_pips = round(dollar_per_lot / pip_val_usd, 2)
    slip_pips = 0.20
    return comm_pips, slip_pips


def get_total_friction_pips(symbol: str, dollar_per_lot: float = DEFAULT_DOLLAR_PER_LOT) -> float:
    """Returns the total fixed friction penalty in pips (comm + slip)."""
    comm, slip = get_friction_pips(symbol, dollar_per_lot)
    return round(comm + slip, 2)


def get_live_broker_profile(symbol: str, dollar_per_lot: float = DEFAULT_DOLLAR_PER_LOT) -> Dict[str, Any]:
    """Returns full live broker cost specification (spread + comm + slip + dollar cost)."""
    s = symbol.upper()
    pip_val = PIP_VALUES_USD.get(s, 10.0)
    spread = LIVE_BROKER_SPREADS.get(s, 0.5)
    comm, slip = get_friction_pips(s, dollar_per_lot)
    total_cost_pips = round(spread + comm + slip, 2)
    total_cost_usd = round(total_cost_pips * pip_val, 2)
    
    return {
        "symbol": s,
        "spread_pips": spread,
        "comm_pips": comm,
        "slip_pips": slip,
        "total_cost_pips": total_cost_pips,
        "total_cost_usd": total_cost_usd,
        "pip_value_usd": pip_val
    }
