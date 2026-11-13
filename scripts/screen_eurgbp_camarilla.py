#!/usr/bin/env python3
"""
================================================================================
  EURGBP CAMARILLA H4/L4 EXPANSION DISCOVERY
================================================================================
Tests Camarilla H4/L4 Breakout Expansion on EURGBP during Frankfurt/London Open (06:00-11:00 UTC).
================================================================================
"""

import sys
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.test_camarilla_expansion_crosses import test_symbol_camarilla

if __name__ == "__main__":
    print("Screening EURGBP Camarilla Expansion...")
    grid = [(7.0, 14.0), (8.0, 16.0), (9.0, 18.0), (10.0, 20.0), (10.0, 22.0)]
    res = test_symbol_camarilla("EURGBP", grid, start_hr=6, end_hr=11, max_spread=1.6)
    if res:
        out_path = ROOT_DIR / "results" / "new_camarilla_eurgbp.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"Saved {len(res)} champions to {out_path}")
    else:
        print("No EURGBP Camarilla champions found meeting the strict criteria.")

