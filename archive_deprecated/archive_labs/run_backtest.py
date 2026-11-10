#!/usr/bin/env python3
"""
Root entry point CLI runner for the Fast Minimal Tick-Level Strategy Tester.
Delegates directly to tick_engine.run_backtest.
"""

import sys
from pathlib import Path

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.run_backtest import main

if __name__ == '__main__':
    main()
