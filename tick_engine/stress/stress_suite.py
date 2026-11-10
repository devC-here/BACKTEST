"""
================================================================================
  BROKER-COST STRESS TESTING & ROLLING WALK-FORWARD SUITE
================================================================================
Implements Requirements 5 and 6 of the Institutional Verification Standard:
1. Broker-Cost Stress Testing:
   - Baseline Live ECN Profile ($4/lot comm, 0.20p slip, raw floating spread)
   - Spread Inflation Stress (+50% wider spreads)
   - Heavy Slippage Stress (0.50 pips adverse entry slippage)
   - High Commission Stress ($6.00/lot institutional turnover fee)
   - Severe Composite Stress (Spread +50% + 0.50p slip + $6/lot commission)
2. Rolling Chronological Walk-Forward Folds:
   - Multi-fold out-of-sample walk-forward matrix (Fold 1, Fold 2, Fold 3)
   - Aggregates strictly untouched out-of-sample performance across rolling regimes.
================================================================================
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.config import PIP_SIZES, get_friction_pips

STRESS_PROFILES = {
    "BASELINE": {
        "description": "Standard Live ECN Profile ($4.00/lot + 0.20p slippage + floating spread)",
        "spread_multiplier": 1.0,
        "commission_dollar_per_lot": 4.00,
        "entry_slippage_pips": 0.20
    },
    "SPREAD_INFLATION_50": {
        "description": "Spread Inflation Stress (+50% wider broker spread)",
        "spread_multiplier": 1.5,
        "commission_dollar_per_lot": 4.00,
        "entry_slippage_pips": 0.20
    },
    "HEAVY_SLIPPAGE_050": {
        "description": "Adverse Slippage Stress (0.50 pips entry slippage)",
        "spread_multiplier": 1.0,
        "commission_dollar_per_lot": 4.00,
        "entry_slippage_pips": 0.50
    },
    "HIGH_COMMISSION_6USD": {
        "description": "Turnover Fee Stress ($6.00/lot round-turn commission)",
        "spread_multiplier": 1.0,
        "commission_dollar_per_lot": 6.00,
        "entry_slippage_pips": 0.20
    },
    "SEVERE_COMPOSITE": {
        "description": "Worst-Case Composite Stress (Spread +50% AND 0.50p slippage AND $6/lot comm)",
        "spread_multiplier": 1.5,
        "commission_dollar_per_lot": 6.00,
        "entry_slippage_pips": 0.50
    }
}

def evaluate_friction_stress(
    ticks: np.ndarray,
    signals: np.ndarray,
    symbol: str = "EURUSD",
    sl_pips: Any = 20.0,
    tp_pips: Any = 40.0,
    pip_size: float = 0.0001
) -> Dict[str, Any]:
    """
    Evaluates a signal stream across all 5 institutional stress profiles.
    Returns scorecard showing survival and performance under adverse execution.
    """
    comm_base, slip_base = get_friction_pips(symbol, dollar_per_lot=4.00)
    comm_high, _ = get_friction_pips(symbol, dollar_per_lot=6.00)

    results = {}
    survived_all = True

    for p_name, prof in STRESS_PROFILES.items():
        spr_mult = prof["spread_multiplier"]
        slip = prof["entry_slippage_pips"]
        comm = comm_high if prof["commission_dollar_per_lot"] == 6.00 else comm_base

        eval_ticks = ticks
        if spr_mult != 1.0:
            # Scale spread around mid-price
            eval_ticks = ticks.copy()
            mid = 0.5 * (ticks['bid'] + ticks['ask'])
            half_spread = 0.5 * (ticks['ask'] - ticks['bid']) * spr_mult
            eval_ticks['bid'] = mid - half_spread
            eval_ticks['ask'] = mid + half_spread

        trades = run_matching_loop(
            eval_ticks,
            signals,
            pip_size=pip_size,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            commission_pips=comm,
            entry_slippage_pips=slip
        )

        n_tr = len(trades)
        net = float(np.sum(trades['pips'])) if n_tr > 0 else 0.0
        win_p = float(np.sum(trades['pips'][trades['pips'] > 0])) if n_tr > 0 else 0.0
        loss_p = float(np.abs(np.sum(trades['pips'][trades['pips'] < 0]))) if n_tr > 0 else 0.0
        pf = round(win_p / loss_p, 2) if loss_p > 0 else (99.0 if win_p > 0 else 0.0)
        is_pass = bool(net > 0 and pf >= 1.00)

        if not is_pass:
            survived_all = False

        results[p_name] = {
            "description": prof["description"],
            "trades": n_tr,
            "net_pips": round(net, 1),
            "profit_factor": pf,
            "survived": is_pass
        }

    return {
        "survived_all_stress_profiles": survived_all,
        "profiles": results
    }

evaluate_stress_battery = evaluate_friction_stress


def generate_rolling_walkforward_folds(total_weeks: int = 300) -> List[Dict[str, Any]]:
    """
    Generates institutional rolling walk-forward folds over continuous timeline.
    Fold 1: Train 1..120, Test 121..180 (60 out-of-sample weeks)
    Fold 2: Train 61..180, Test 181..240 (60 out-of-sample weeks)
    Fold 3: Train 121..240, Test 241..300 (60 out-of-sample weeks)
    """
    return [
        {
            "fold": 1,
            "train_weeks": list(range(1, 121)),
            "test_weeks": list(range(121, 181)),
            "train_range": "Weeks 1..120 (~2021-2023)",
            "test_range": "Weeks 121..180 (~2023-2024)"
        },
        {
            "fold": 2,
            "train_weeks": list(range(61, 181)),
            "test_weeks": list(range(181, 241)),
            "train_range": "Weeks 61..180 (~2022-2024)",
            "test_range": "Weeks 181..240 (~2024-2025)"
        },
        {
            "fold": 3,
            "train_weeks": list(range(121, 241)),
            "test_weeks": list(range(241, 301)),
            "train_range": "Weeks 121..240 (~2023-2025)",
            "test_range": "Weeks 241..300 (~2025-2026)"
        }
    ]
