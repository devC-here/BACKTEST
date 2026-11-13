#!/usr/bin/env python3
"""
scripts/audit_fleet_integrity.py
================================================================================
Phase 1 Forensic Audit of Certified Production Fleet
Computes AST hash, SHA-256 code hash, hardcoded symbol/pip_size inspection,
and detects identical/cloned strategy files across all certified subdirectories.
================================================================================
"""

import ast
import hashlib
import json
import os
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
FLEET_DIR = ROOT_DIR / "certified_production_fleet"


def get_ast_hash(code_str: str) -> str:
    """Computes a normalized AST dump hash ignoring whitespace and comments."""
    try:
        tree = ast.parse(code_str)
        # ast.dump provides a structured canonical representation of AST
        dumped = ast.dump(tree, annotate_fields=True, include_attributes=False)
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()
    except Exception:
        # Fallback to normalized text hash
        norm = re.sub(r"\s+", " ", code_str)
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def get_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def audit_file(file_path: Path, expected_symbol: str) -> dict:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    raw_bytes = text.encode("utf-8")

    ast_h = get_ast_hash(text)
    sha_h = get_sha256(raw_bytes)

    # Inspect hardcoded symbols, pip sizes, spread limits
    symbols_found = set(re.findall(r'["\'](EURUSD|USDJPY|EURGBP|USDCAD|GBPJPY|GBPCHF|AUDCAD|AUDNZD|NZDCAD|EURNZD)["\']', text))
    pip_sizes_found = set(re.findall(r'pip_size\s*[:=]\s*([0-9\.]+)', text))
    sl_pips_found = re.findall(r'sl_pips\s*[:=]\s*([0-9\.]+)', text)
    tp_pips_found = re.findall(r'tp_pips\s*[:=]\s*([0-9\.]+)', text)

    # Check for hardcoded 0.0001 multiplication on prices
    has_hardcoded_pip_float = bool(re.search(r'\*\s*0\.0001', text))
    has_hardcoded_jpy_float = bool(re.search(r'\*\s*0\.01', text))

    return {
        "path": str(file_path.relative_to(ROOT_DIR)),
        "filename": file_path.name,
        "parent_dir": file_path.parent.name,
        "expected_symbol": expected_symbol,
        "file_size": len(raw_bytes),
        "sha256": sha_h,
        "ast_hash": ast_h,
        "symbols_found": sorted(list(symbols_found)),
        "pip_sizes_found": sorted(list(pip_sizes_found)),
        "sl_pips": sl_pips_found[0] if sl_pips_found else None,
        "tp_pips": tp_pips_found[0] if tp_pips_found else None,
        "has_hardcoded_0_0001": has_hardcoded_pip_float,
        "has_hardcoded_0_01": has_hardcoded_jpy_float,
    }


def main():
    print("=" * 95)
    print("  PHASE 1 FORENSIC AUDIT: CERTIFIED PRODUCTION FLEET INTEGRITY")
    print("=" * 95)

    all_strategies = []
    
    # Map subfolder to expected symbol
    subfolder_symbol_map = {
        "eurusd_champions": "EURUSD",
        "eurusd_ny_sweeps": "EURUSD",
        "usdjpy_champions": "USDJPY",
        "usdcad_asian_sweeps": "USDCAD",
        "audnzd_champions": "AUDNZD",
        "eurgbp_champions": "EURGBP",
        "eurgbp_london_sweeps": "EURGBP",
        "gbpchf_champions": "GBPCHF",
    }

    for subfolder, expected_sym in subfolder_symbol_map.items():
        sdir = FLEET_DIR / subfolder
        if not sdir.exists():
            continue
        for root, dirs, files in os.walk(sdir):
            for f in files:
                if f.endswith(".py") and not f.startswith("__"):
                    fpath = Path(root) / f
                    # Skip runner/test scripts if any
                    res = audit_file(fpath, expected_sym)
                    all_strategies.append(res)

    print(f"Total strategy python files audited: {len(all_strategies)}")

    # 1. Detect exact AST duplicates (exact code clones)
    ast_map = {}
    sha_map = {}
    for s in all_strategies:
        ast_map.setdefault(s["ast_hash"], []).append(s)
        sha_map.setdefault(s["sha256"], []).append(s)

    exact_sha_clones = {k: v for k, v in sha_map.items() if len(v) > 1}
    exact_ast_clones = {k: v for k, v in ast_map.items() if len(v) > 1 and k not in exact_sha_clones}

    print("\n--- 1. EXACT SHA-256 FILE DUPLICATES ---")
    if exact_sha_clones:
        for k, v in exact_sha_clones.items():
            print(f"Hash {k[:16]}... ({len(v)} files identical):")
            for item in v:
                print(f"   [{item['expected_symbol']}] {item['path']}")
    else:
        print("None found.")

    print("\n--- 2. AST CODE CLONES (Identical logic, possible variable name/comment changes) ---")
    if exact_ast_clones:
        for k, v in exact_ast_clones.items():
            print(f"AST Hash {k[:16]}... ({len(v)} files identical structure):")
            for item in v:
                print(f"   [{item['expected_symbol']}] {item['path']}")
    else:
        print("None found.")

    # 3. Detect Symbol Mismatches (e.g. USDJPY strategy containing only EURUSD or hardcoded 0.0001)
    print("\n--- 3. SYMBOL & PIP SIZE MISMATCH INSPECTION ---")
    mismatches = []
    for s in all_strategies:
        sym = s["expected_symbol"]
        is_jpy = "JPY" in sym
        issues = []
        if is_jpy and s["has_hardcoded_0_0001"]:
            issues.append("Contains hardcoded `* 0.0001` on a JPY pair!")
        if is_jpy and "0.0001" in s["pip_sizes_found"] and "0.01" not in s["pip_sizes_found"]:
            issues.append("pip_size defaults to 0.0001 instead of 0.01 on JPY pair!")
        if s["symbols_found"] and sym not in s["symbols_found"]:
            issues.append(f"Expected {sym} but file only references {s['symbols_found']}")
        if issues:
            mismatches.append((s, issues))
            print(f"WARNING: [{sym}] {s['path']}")
            for iss in issues:
                print(f"   -> {iss}")

    if not mismatches:
        print("No hardcoded symbol/pip_size mismatches detected.")

    # Save full audit JSON
    out_json = ROOT_DIR / "results" / "fleet_integrity_audit.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "total_files": len(all_strategies),
            "exact_sha_clones_count": len(exact_sha_clones),
            "exact_ast_clones_count": len(exact_ast_clones),
            "mismatch_count": len(mismatches),
            "strategies": all_strategies
        }, f, indent=2)

    print(f"\nFull integrity audit saved to: {out_json}")


if __name__ == "__main__":
    main()

