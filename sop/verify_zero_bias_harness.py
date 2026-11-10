#!/usr/bin/env python3
"""
================================================================================
  PRE-FLIGHT ZERO-BIAS VERIFICATION & LINTING HARNESS
================================================================================
Automated CI/CD quality gate to prevent lookahead bias, backward-fill leakage,
and unshifted higher-timeframe bugs from ever entering production code.

Rules Enforced:
1. NO_BFILL_ON_ROLLING: Zero occurrences of .bfill() on rolling window series.
2. CAUSAL_HTF_SHIFT: All higher-timeframe resampled series must be shifted (.shift(1)).
3. BAR_CLOSE_TIMING: Execution timestamp must enforce bar-close latency (time_ms + dur_ms).
4. FACTORY_ENTRYPOINT: Strategy file must expose `Strategy = ...` or `class Strategy`.
================================================================================
"""

import sys
import re
import argparse
from pathlib import Path
from typing import List, Tuple

def audit_file(file_path: Path) -> List[Tuple[str, str, int]]:
    """
    Audits a single Python strategy file for lookahead biases and flaws.
    Returns list of (violation_code, message, line_number).
    """
    violations = []
    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    # Rule 1: NO .bfill() on rolling calculations
    for i, line in enumerate(lines, 1):
        if "rolling(" in line and ".bfill()" in line:
            violations.append((
                "FATAL_BFILL_LOOKAHEAD",
                f"Calling .bfill() on a rolling series leaks future price horizons backwards: `{line.strip()}`",
                i
            ))
        elif ".bfill()" in line and not line.strip().startswith("#"):
            # Check if bfill is used on price series
            if any(term in line for term in ["roll_", "h1", "h4", "d1", "fib", "donchian", "supp", "res"]):
                violations.append((
                    "SUSPICIOUS_BFILL",
                    f"Suspicious .bfill() call on financial indicator series: `{line.strip()}`",
                    i
                ))

    # Rule 2: Causal HTF Shift (.shift(1))
    # If resample is used, verify that .shift(1) is present
    if "resample('4h')" in content or 'resample("4h")' in content or "resample('1h')" in content or 'resample("1h")' in content:
        if "reindex(" in content and ".shift(1)" not in content:
            violations.append((
                "FATAL_UNSHIFTED_HTF_LEAK",
                "Higher-timeframe series is reindexed without .shift(1) causal alignment.",
                1
            ))

    # Rule 3: Bar Close Execution Timing
    has_timing = any(
        term in content for term in [
            "time_ms[i] + dur_ms",
            "time_ms + dur_ms",
            "exec_t = ",
            "np.searchsorted(tick_ts, exec_t)"
        ]
    )
    if not has_timing and "def generate_signals" in content:
        violations.append((
            "SUSPICIOUS_EXECUTION_TIMING",
            "Strategy generate_signals() does not appear to enforce bar-close execution latency (time_ms + dur_ms).",
            1
        ))

    # Rule 4: Factory Entrypoint
    if "Strategy =" not in content and "class Strategy" not in content:
        violations.append((
            "MISSING_FACTORY_ENTRYPOINT",
            "Strategy module lacks `Strategy = ...` or `class Strategy` entrypoint.",
            1
        ))

    return violations

def main():
    parser = argparse.ArgumentParser(description="Pre-flight Zero-Bias Verification Harness")
    parser.add_argument("--dir", type=str, default="v 4.3", help="Directory containing strategy files to audit")
    parser.add_argument("--file", type=str, default=None, help="Single strategy file to audit")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    if args.file:
        targets = [Path(args.file)]
    else:
        target_dir = root_dir / args.dir
        if not target_dir.exists():
            print(f"Directory not found: {target_dir}")
            sys.exit(1)
        targets = sorted(list(target_dir.glob("strategy_*.py")))

    print("=" * 100)
    print(f"              RUNNING PRE-FLIGHT ZERO-BIAS HARNESS ON {len(targets)} STRATEGIES              ".center(100))
    print("=" * 100)

    total_violations = 0
    flagged_files = 0

    for f in targets:
        violations = audit_file(f)
        if violations:
            flagged_files += 1
            total_violations += len(violations)
            print(f"\n[FAIL] {f.name} ({len(violations)} violation(s)):")
            for code, msg, line in violations:
                print(f"   Line {line:4d} | {code:25s} | {msg}")

    print("\n" + "=" * 100)
    if total_violations == 0:
        print(f"RESULT: ALL {len(targets)} STRATEGIES PASSED! (Zero Lookahead / Zero Bfill Detected)".center(100))
        print("=" * 100)
        sys.exit(0)
    else:
        print(f"RESULT: AUDIT FAILED! {flagged_files}/{len(targets)} files flagged with {total_violations} violation(s).".center(100))
        print("=" * 100)
        sys.exit(1)

if __name__ == "__main__":
    main()

