#!/usr/bin/env python3
"""
Calculate exact 5-year historical exchange rates and pip value conversion multipliers
relative to EURUSD (EURUSD = 1.000 baseline).
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.dataloader import get_week_dataset


def main():
    print("=" * 80)
    print("  CALCULATING 5-YEAR HISTORICAL EXCHANGE RATES & PIP MULTIPLIERS (2021 - 2026)")
    print("=" * 80)

    # Sample representative weeks across 2021 - 2026
    sample_weeks = list(range(1, 301, 10))
    usdjpy_prices = []
    eurusd_prices = []
    eurgbp_prices = []

    for w in sample_weeks:
        try:
            t_uj, _ = get_week_dataset(w, raw_dir="data/USDJPY")
            if t_uj is not None and len(t_uj) > 0:
                usdjpy_prices.append(float(np.mean(t_uj['bid'])))

            t_eu, _ = get_week_dataset(w, raw_dir="data/EURUSD")
            if t_eu is not None and len(t_eu) > 0:
                eurusd_prices.append(float(np.mean(t_eu['bid'])))

            t_eg, _ = get_week_dataset(w, raw_dir="data/EURGBP")
            if t_eg is not None and len(t_eg) > 0:
                eurgbp_prices.append(float(np.mean(t_eg['bid'])))
        except Exception:
            pass

    mean_uj = float(np.mean(usdjpy_prices))
    mean_eu = float(np.mean(eurusd_prices))
    mean_eg = float(np.mean(eurgbp_prices))
    mean_gu = mean_eu / mean_eg

    # Multipliers relative to EURUSD ($10.00 / pip per standard lot)
    # USDJPY: 1 pip = 1,000 JPY = 1,000 / USDJPY USD.
    # Relative to EURUSD ($10.00): (1,000 / USDJPY) / 10.00 = 100 / USDJPY.
    mult_uj = 100.0 / mean_uj

    # EURGBP: 1 pip = 10 GBP = 10 * GBPUSD USD = 10 * (EURUSD / EURGBP) USD.
    # Relative to EURUSD ($10.00): [10 * (EURUSD / EURGBP)] / 10.00 = EURUSD / EURGBP.
    mult_eg = mean_gu

    print(f"Average EURUSD Price: {mean_eu:.4f}  -> Pip Value: $10.00 / lot  -> Multiplier: 1.000x")
    print(f"Average USDJPY Price: {mean_uj:.2f}   -> Pip Value: ${1000/mean_uj:.2f} / lot  -> Multiplier: {mult_uj:.4f}x ({mult_uj*100:.1f}%)")
    print(f"Average EURGBP Price: {mean_eg:.4f}  -> Derived GBPUSD: {mean_gu:.4f} -> Pip Value: ${10*mean_gu:.2f} / lot -> Multiplier: {mult_eg:.4f}x ({mult_eg*100:.1f}%)")

    print("\n" + "=" * 80)
    print("  IMPACT OF NORMALIZATION ON 1,000 RAW PIPS:")
    print("=" * 80)
    print(f"1,000 Raw EURUSD Pips = 1,000.0 EURUSD-Equivalent Pips ($10,000 USD)")
    print(f"1,000 Raw USDJPY Pips = {1000 * mult_uj:.1f} EURUSD-Equivalent Pips (${1000 * mult_uj * 10:.2f} USD)")
    print(f"1,000 Raw EURGBP Pips = {1000 * mult_eg:.1f} EURUSD-Equivalent Pips (${1000 * mult_eg * 10:.2f} USD)")


if __name__ == "__main__":
    main()
