#!/usr/bin/env python3
"""
Convenience CLI launcher for the Master Discovery Labs Orchestrator.
Delegates directly to data_engineering_suite/run_symbol_discovery_labs.py.
"""
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "data_engineering_suite" / "run_symbol_discovery_labs.py"
PYTHON = sys.executable

if __name__ == "__main__":
    cmd = [PYTHON, str(RUNNER)] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))

