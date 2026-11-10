#!/usr/bin/env python3
"""
Convenience CLI launcher for the Master Symbol Alpha Discovery & Qualification Suite.
Delegates directly to data_engineering_suite/master_symbol_alpha_discovery_suite.py.
"""
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUITE = ROOT / "data_engineering_suite" / "master_symbol_alpha_discovery_suite.py"
PYTHON = sys.executable

if __name__ == "__main__":
    cmd = [PYTHON, str(SUITE)] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))

