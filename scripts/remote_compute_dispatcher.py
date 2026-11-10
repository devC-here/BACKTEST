#!/usr/bin/env python3
"""
================================================================================
  REMOTE VM COMPUTATION DISPATCHER FOR BACKTEST ENGINE
================================================================================
Dispatches high-speed tick backtesting, 5Y fleet audits, and alpha discovery
jobs from your local workstation to the Azure Cloud VM (azure-vm: C:\BACKTEST).

Features:
  1. Auto-syncs code, strategies, and discovery scripts to the VM.
  2. Executes remotely with BELOW_NORMAL process priority (protects live MT5 fleet).
  3. Real-time log streaming of progress, tick rates, and performance matrices.
  4. Automatic pull of results (CSVs, charts, JSON manifests) back to local results/.

Usage:
  # Master Symbol Alpha Discovery on EURUSD:
  python scripts/remote_compute_dispatcher.py --script run_symbol_alpha_discovery.py --args "--symbol EURUSD"

  # 5-Year Fleet Audit:
  python scripts/remote_compute_dispatcher.py --script run_5y_fleet_audit.py

  # Custom Hurdle Rates with Symbol Data Sync:
  python scripts/remote_compute_dispatcher.py --script run_symbol_alpha_discovery.py --args "--symbol GBPJPY" --sync-data GBPJPY
================================================================================
"""

import sys
import os
import subprocess
import argparse
import time
from pathlib import Path

# Paths
LOCAL_ROOT = Path(__file__).resolve().parent.parent
REMOTE_HOST = "azure-vm"
REMOTE_ROOT = "C:/BACKTEST"
REMOTE_PYTHON = "C:/Python311/python.exe"


def run_cmd(cmd: list[str], check: bool = True, shell: bool = False) -> subprocess.CompletedProcess:
    """Helper to run a shell command."""
    return subprocess.run(cmd, check=check, shell=shell)


def sync_codebase(host: str = REMOTE_HOST):
    """Syncs essential engine code, strategies, and suites to VM."""
    print(f"\n[*] [1/4] Synchronizing codebase to {host}:{REMOTE_ROOT}...")

    # Ensure remote dirs exist
    setup_cmd = [
        "ssh", host,
        f"powershell -Command \"New-Item -ItemType Directory -Force -Path {REMOTE_ROOT}, {REMOTE_ROOT}/data, {REMOTE_ROOT}/results, {REMOTE_ROOT}/scripts, {REMOTE_ROOT}/tick_engine/data/processed\""
    ]
    run_cmd(setup_cmd, check=False)

    # Sync python files, requirements, and directories
    items_to_copy = ["tick_engine", "data_engineering_suite"]
    for item in items_to_copy:
        p = LOCAL_ROOT / item
        if p.exists():
            print(f"  -> Uploading {item}/...")
            scp_cmd = ["scp", "-r", str(p), f"{host}:{REMOTE_ROOT}/"]
            run_cmd(scp_cmd, check=False)

    # Copy top-level python launcher scripts
    py_files = list(LOCAL_ROOT.glob("*.py"))
    if py_files:
        print(f"  -> Uploading {len(py_files)} root launcher scripts...")
        scp_cmd = ["scp"] + [str(f) for f in py_files] + [f"{host}:{REMOTE_ROOT}/"]
        run_cmd(scp_cmd, check=False)

    print("  [+] Codebase synchronization complete.")


def sync_symbol_data(symbol: str, host: str = REMOTE_HOST):
    """Syncs raw/processed tick data for a given symbol to VM."""
    sym = symbol.upper()
    print(f"\n[*] Synchronizing tick data for {sym} to {host}:{REMOTE_ROOT}/data/{sym}...")
    local_sym_dir = LOCAL_ROOT / "data" / sym
    if not local_sym_dir.exists():
        print(f"  [!] Warning: Local data directory {local_sym_dir} not found. Skipping data sync.")
        return

    # Ensure remote symbol dir exists
    run_cmd(["ssh", host, f"powershell -Command \"New-Item -ItemType Directory -Force -Path {REMOTE_ROOT}/data/{sym}\""], check=False)

    # Upload files
    print(f"  -> Uploading {sym} weekly tick data...")
    run_cmd(["scp", "-r", f"{str(local_sym_dir)}/*", f"{host}:{REMOTE_ROOT}/data/{sym}/"], check=False)
    print(f"  [+] Symbol {sym} data synchronization complete.")


def execute_remote_job(script_name: str, script_args: str = "", host: str = REMOTE_HOST):
    """Executes the specified script on the Azure VM and streams output in real time."""
    print(f"\n[*] [2/4] Launching {script_name} on {host}...")
    print(f"    Command: {REMOTE_PYTHON} {script_name} {script_args}")
    print("-" * 80)

    # PowerShell command to set process priority to BelowNormal (protecting live MT5) and execute
    remote_ps = (
        f"cd {REMOTE_ROOT} ; "
        f"$p = Start-Process -FilePath '{REMOTE_PYTHON}' -ArgumentList '{script_name} {script_args}' "
        f"-PassThru -NoNewWindow ; "
        f"$p.PriorityClass = 'BelowNormal' ; "
        f"$p.WaitForExit() ; "
        f"exit $p.ExitCode"
    )

    ssh_cmd = ["ssh", "-t", host, f"powershell -Command \"{remote_ps}\""]
    start_t = time.time()
    try:
        res = subprocess.run(ssh_cmd)
        elapsed = time.time() - start_t
        print("-" * 80)
        print(f"[*] [3/4] Execution finished in {elapsed:.1f}s (Exit code: {res.returncode})")
        return res.returncode == 0
    except KeyboardInterrupt:
        print("\n[!] User interrupted remote job streaming. Remote process may still be running in background.")
        return False


def pull_results(host: str = REMOTE_HOST):
    """Downloads all output files from VM results/ to local results/."""
    print(f"\n[*] [4/4] Pulling results from {host}:{REMOTE_ROOT}/results/...")
    local_results = LOCAL_ROOT / "results"
    local_results.mkdir(parents=True, exist_ok=True)

    scp_cmd = ["scp", "-r", f"{host}:{REMOTE_ROOT}/results/*", f"{str(local_results)}/"]
    run_cmd(scp_cmd, check=False)
    print(f"  [+] Output artifacts saved to: {local_results}")


def main():
    parser = argparse.ArgumentParser(description="H1-ONLINE Remote VM Computation Dispatcher")
    parser.add_argument("--script", type=str, default="run_symbol_alpha_discovery.py",
                        help="Script to run on VM (default: run_symbol_alpha_discovery.py)")
    parser.add_argument("--args", type=str, default="",
                        help="Arguments to pass to the script (e.g. '--symbol EURUSD')")
    parser.add_argument("--sync-data", type=str, default=None,
                        help="Optional symbol to sync tick data before execution (e.g. 'EURUSD')")
    parser.add_argument("--no-sync-code", action="store_true",
                        help="Skip code synchronization")
    parser.add_argument("--no-pull", action="store_true",
                        help="Skip pulling results after execution")
    parser.add_argument("--host", type=str, default=REMOTE_HOST,
                        help=f"Target SSH host alias (default: {REMOTE_HOST})")

    args = parser.parse_args()

    # 1. Sync code
    if not args.no_sync_code:
        sync_codebase(host=args.host)

    # 2. Sync data if requested
    if args.sync_data:
        sync_symbol_data(args.sync_data, host=args.host)

    # 3. Execute
    success = execute_remote_job(args.script, args.args, host=args.host)

    # 4. Pull results
    if not args.no_pull:
        pull_results(host=args.host)

    print("\n" + "=" * 80)
    print("  DISPATCH MISSION ACCOMPLISHED")
    print("=" * 80)


if __name__ == "__main__":
    main()

