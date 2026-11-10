#!/usr/bin/env python3
"""
================================================================================
  MASTER DISCOVERY LABS ORCHESTRATOR & RUNNER
================================================================================
Indexes, catalogs, and systematically runs the complete suite of 53 Alpha
Discovery Labs (v1 through v45, plus AUDNZD and EURGBP specialist labs)
across any target symbol.

Commands:
  # List all 53 discovery labs in the catalog:
  .venv\\Scripts\\python.exe run_symbol_discovery_labs.py --list

  # Run a specific lab on a symbol:
  .venv\\Scripts\\python.exe run_symbol_discovery_labs.py --symbol EURGBP --lab v33

  # Run a batch of labs on a symbol:
  .venv\\Scripts\\python.exe run_symbol_discovery_labs.py --symbol AUDNZD --batch specialists
  .venv\\Scripts\\python.exe run_symbol_discovery_labs.py --symbol EURUSD --batch v1-v10

  # Run the full spectrum of all labs:
  .venv\\Scripts\\python.exe run_symbol_discovery_labs.py --symbol <SYMBOL> --batch all
================================================================================
"""

from __future__ import annotations
import sys
import os
import time
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
DISCOVERY_LABS_DIR = ROOT_DIR / "discovery_labs"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable


# ==============================================================================
# Complete Catalog of 53 Alpha Discovery Labs
# ==============================================================================

LAB_CATALOG: Dict[str, Dict[str, Any]] = {
    # Batch 1: Labs v1 - v10 (Foundation & Market Microstructure)
    "v1": {
        "script": "alpha_discovery_lab_v1.py",
        "batch": "v1-v10",
        "title": "Mining Quantitative Edges & Microstructure Filters",
        "architectures": "London open breakout, Asian sweep fade, Momentum pulse, EMA trend ride"
    },
    "v2": {
        "script": "alpha_discovery_lab_v2.py",
        "batch": "v1-v10",
        "title": "Advanced Systematic Architectures & Multi-Timeframe Regimes",
        "architectures": "Donchian trend pullbacks, H4 macro retests, Asian range compression"
    },
    "v3": {
        "script": "alpha_discovery_lab_v3.py",
        "batch": "v1-v10",
        "title": "Structural Level Mining & Tokyo Range Calibration",
        "architectures": "Tokyo fix breakout, London morning continuation, Donchian levels"
    },
    "v4": {
        "script": "alpha_discovery_lab_v4.py",
        "batch": "v1-v10",
        "title": "Lower-Timeframe Precision & CPR / Pivot Defense",
        "architectures": "Central Pivot Range (CPR) bounce, Daily pivot wall defense, M5 micro wick"
    },
    "v5": {
        "script": "alpha_discovery_lab_v5.py",
        "batch": "v1-v10",
        "title": "Pivot Extensions, Fibonacci Pullbacks & NR7 Ranges",
        "architectures": "Fibonacci golden pocket pin defense, Daily NR7 expansion, Pivot S1/R1"
    },
    "v6": {
        "script": "alpha_discovery_lab_v6.py",
        "batch": "v1-v10",
        "title": "Fibonacci Pin Defense, Tokyo Pivots & NR7 Volatility Expansion",
        "architectures": "Fibonacci 61.8% pin defense, Tokyo pivot wall, Narrow range 7 expansion"
    },
    "v7": {
        "script": "alpha_discovery_lab_v7.py",
        "batch": "v1-v10",
        "title": "Institutional Compression, Retests & Overlap Drives",
        "architectures": "Double structure neckline breakout, NY open overlap drive, Asian retests"
    },
    "v8": {
        "script": "alpha_discovery_lab_v8.py",
        "batch": "v1-v10",
        "title": "Asian Retests, Fix Reversals & Inside Bar Expansions",
        "architectures": "Asian high/low retest, London fix mean reversion, H1 inside bar expansion"
    },
    "v9": {
        "script": "alpha_discovery_lab_v9.py",
        "batch": "v1-v10",
        "title": "Range Expansions, ATR Exhaustions & Liquidity Traps",
        "architectures": "ATR daily stretch exhaustion fade, Asian liquidity sweep spring"
    },
    "v10": {
        "script": "alpha_discovery_lab_v10.py",
        "batch": "v1-v10",
        "title": "Order Blocks, Divergence Traps & VWAP Value Fades",
        "architectures": "RSI regime range shift, Stochastic deep pullbacks, Session VWAP value"
    },

    # Batch 2: Labs v11 - v20 (Squeezes, Momentum & Volume Expansions)
    "v11": {
        "script": "alpha_discovery_lab_v11.py",
        "batch": "v11-v20",
        "title": "Micro Blast, TTM Squeeze, Tokyo Sweep & Thursday Ignition",
        "architectures": "TTM Keltner squeeze, Tokyo range sweep, Thursday trend ignition"
    },
    "v12": {
        "script": "alpha_discovery_lab_v12.py",
        "batch": "v11-v20",
        "title": "MACD Zero Cross, Asia Compression & RSI 50 Regime Shift",
        "architectures": "M15 MACD zero-cross, Asian compression breakout, RSI 50 bounce"
    },
    "v13": {
        "script": "alpha_discovery_lab_v13.py",
        "batch": "v11-v20",
        "title": "Pin Runners, US Overlap & Double Structure Necklines",
        "architectures": "M15 50-EMA pin runner, US morning overlap, Double structure breakout"
    },
    "v14": {
        "script": "alpha_discovery_lab_v14.py",
        "batch": "v11-v20",
        "title": "Reversal Fade, Donchian Retest, Ribbon & London PM Expansion",
        "architectures": "Asian range reversal fade, Donchian retest, London afternoon drive"
    },
    "v15": {
        "script": "alpha_discovery_lab_v15.py",
        "batch": "v11-v20",
        "title": "Pre-Fix Expansion, Stoch-RSI, Tokyo Fix Trend & Midday MACD",
        "architectures": "Stoch-RSI pullback hook, Tokyo fix trend runner, London midday MACD"
    },
    "v16": {
        "script": "alpha_discovery_lab_v16.py",
        "batch": "v11-v20",
        "title": "3-Day Sweep, Asia Midpoint, SuperTrend & Tokyo Pullback",
        "architectures": "Asian midpoint rebound, SuperTrend M15 squeeze, Tokyo 20-EMA pullback"
    },
    "v17": {
        "script": "alpha_discovery_lab_v17.py",
        "batch": "v11-v20",
        "title": "Lunch Squeeze, Order Flow Retest, VWAP Bands & NY Continuation",
        "architectures": "London lunch expansion, Session VWAP 2.0-SD band fade, NY continuation"
    },
    "v18": {
        "script": "alpha_discovery_lab_v18.py",
        "batch": "v11-v20",
        "title": "Heikin-Ashi Surge, Pre-Fix Fade, CMF Expansion & 200-EMA Bounce",
        "architectures": "Heikin-Ashi impulse surge, CMF volume expansion, M15 200-EMA bounce"
    },
    "v19": {
        "script": "alpha_discovery_lab_v19.py",
        "batch": "v11-v20",
        "title": "Liquidity Retest, MTF Micro-Scalping & Asian Box Traps",
        "architectures": "Asian box retest, MTF micro-scalper, Liquidity sweep spring"
    },
    "v20": {
        "script": "alpha_discovery_lab_v20.py",
        "batch": "v11-v20",
        "title": "Volume Delta Absorption, Fix Continuation & H4 Wick Traps",
        "architectures": "Delta absorption scalper, London fix continuation, H4 wick rejection"
    },

    # Batch 3: Labs v21 - v30 (Pivots, Anchored VWAP & S/R Springs)
    "v21": {
        "script": "alpha_discovery_lab_v21.py",
        "batch": "v21-v30",
        "title": "Anchored VWAP Bands, Session Compression & Double Breakouts",
        "architectures": "Daily VWAP bands 2.0-SD, Session tight coil, Double top/bottom neckline"
    },
    "v22": {
        "script": "alpha_discovery_lab_v22.py",
        "batch": "v21-v30",
        "title": "Weekly Pivots, VWAP 1-SD Discount & Keltner Midline Expansion",
        "architectures": "Weekly pivot pullback (DISC75), VWAP 1-SD discount (DISC76), Keltner midline (DISC77)"
    },
    "v23": {
        "script": "alpha_discovery_lab_v23.py",
        "batch": "v21-v30",
        "title": "Directional ADX, London Opening 15M & Three-Bar Inside Breakout",
        "architectures": "Three-bar inside breakout (DISC78), London 15m ORB (DISC79), ADX DI cross"
    },
    "v24": {
        "script": "alpha_discovery_lab_v24.py",
        "batch": "v21-v30",
        "title": "SAR Acceleration, London Fix Reversion & Double Swing Neckline",
        "architectures": "Double swing neckline retest (DISC80), London fix reversion, SAR acceleration"
    },
    "v25": {
        "script": "alpha_discovery_lab_v25.py",
        "batch": "v21-v30",
        "title": "Fix Rebalance, Double Inside Bar & BB Squeeze Ignition",
        "architectures": "Bollinger squeeze ignition (DISC81), Double inside bar, London fix rebalance"
    },
    "v26": {
        "script": "alpha_discovery_lab_v26.py",
        "batch": "v21-v30",
        "title": "EMA-ATR Expansion, Pivot Breakout, 3-Drive Ignition & SAR Flip",
        "architectures": "Chaikin surge (DISC82), Parabolic SAR flip (DISC83), London 30m ORB (DISC84), EMA-ATR expansion (DISC86)"
    },
    "v27": {
        "script": "alpha_discovery_lab_v27.py",
        "batch": "v21-v30",
        "title": "London Fix Surge, SuperTrend Squeeze & Session Benchmark Impulse",
        "architectures": "Session open impulse (DISC85), Pivot R1/S1 breakout (DISC87), 3-drive ignition (DISC88)"
    },
    "v28": {
        "script": "alpha_discovery_lab_v28.py",
        "batch": "v21-v30",
        "title": "SuperTrend Squeeze Calibration & High-Volume Expansion",
        "architectures": "Calibrated SuperTrend squeeze, ATR expansion drives, Fixed pip targets"
    },
    "v29": {
        "script": "alpha_discovery_lab_v29.py",
        "batch": "v21-v30",
        "title": "Daily Pivot R2/S2 Extreme Reversion & Boundary Fades",
        "architectures": "Daily Pivot R2/S2 extreme boundary fade, Reversion oscillators"
    },
    "v30": {
        "script": "alpha_discovery_lab_v30.py",
        "batch": "v21-v30",
        "title": "Asian Range Liquidity Sweep & Spring Reversals",
        "architectures": "Asian liquidity sweep spring, False breakout re-entry, Range boundaries"
    },

    # Batch 4: Labs v31 - v40 (Mean Reversion, Pullbacks & Cardwell RSI)
    "v31": {
        "script": "alpha_discovery_lab_v31.py",
        "batch": "v31-v45",
        "title": "London Fix Overlap Spring & Multi-Session Fades",
        "architectures": "Fix overlap springs, Session transition liquidity traps, Volume absorption"
    },
    "v32": {
        "script": "alpha_discovery_lab_v32.py",
        "batch": "v31-v45",
        "title": "H1-M15 EMA Value Bounce & Trend Pullback Accelerators",
        "architectures": "H1-M15 20/50 EMA value bounce, Dynamic pullback accelerators"
    },
    "v33": {
        "script": "alpha_discovery_lab_v33.py",
        "batch": "v31-v45",
        "title": "EURGBP 2.5-SD BB Fade, London Lunch Pullback & Tokyo Expansion",
        "architectures": "Bollinger 2.5-SD extreme fade, London lunch pre-NY pullback, Tokyo continuation"
    },
    "v34": {
        "script": "alpha_discovery_lab_v34.py",
        "batch": "v31-v45",
        "title": "Prior Day Extreme Sweep & Multi-Day Boundary Springs",
        "architectures": "Prior day high/low liquidity sweep, False breakout springs"
    },
    "v35": {
        "script": "alpha_discovery_lab_v35.py",
        "batch": "v31-v45",
        "title": "Daily Pivot R1/S1 Retest & Trend Continuation Drives",
        "architectures": "Pivot R1/S1 breakout retests, Structural level flips, Trend continuation"
    },
    "v36": {
        "script": "alpha_discovery_lab_v36.py",
        "batch": "v31-v45",
        "title": "Camarilla H4/L4 Breakout vs L3/H3 Mean Reversion",
        "architectures": "Camarilla H4/L4 expansion drives, Camarilla L3/H3 boundary fades"
    },
    "v37": {
        "script": "alpha_discovery_lab_v37.py",
        "batch": "v31-v45",
        "title": "Cardwell RSI 50 Regime Pullback & Momentum Bounces",
        "architectures": "Cardwell RSI 50 regime support in bull trend, RSI 50 resistance in bear trend"
    },
    "v38": {
        "script": "alpha_optimization_lab_v38.py",
        "batch": "v31-v45",
        "title": "Multi-Asset Parameter Optimization & Spread Resistance",
        "architectures": "Spread resilience tuning, SL/TP parameter optimization"
    },
    "v39": {
        "script": "alpha_discovery_lab_v39.py",
        "batch": "v31-v45",
        "title": "Cardwell RSI & Camarilla Level Synergies",
        "architectures": "RSI 50 bounce at Camarilla pivot levels, High-probability confluence"
    },
    "v40": {
        "script": "alpha_discovery_lab_v40.py",
        "batch": "v31-v45",
        "title": "Alpha Correction & Return Optimization (>= 15% Annual Return)",
        "architectures": "Calibrated Camarilla, Asian compression, Double structure breakouts"
    },

    # Batch 5: Labs v41 - v45 (Realistic Return Hurdle Tuning)
    "v41": {
        "script": "alpha_optimization_lab_v41.py",
        "batch": "v31-v45",
        "title": "Return Optimization Lab: Asymmetric Risk Multipliers (2.0R - 2.5R)",
        "architectures": "2.0R to 2.5R asymmetric reward multipliers for 15% annual yield"
    },
    "v42": {
        "script": "alpha_discovery_lab_v42.py",
        "batch": "v31-v45",
        "title": "Realistic Return Lab: 15% Annual Net Hurdle Screening",
        "architectures": "M15 MACD zero-cross, Tokyo expansion ATR, Cardwell RSI 50, Camarilla expansion"
    },
    "v43": {
        "script": "alpha_discovery_lab_v43.py",
        "batch": "v31-v45",
        "title": "High-Yield Oscillator & Squeeze Refinements",
        "architectures": "Refined oscillator overbought/oversold, Directional squeeze timing"
    },
    "v44": {
        "script": "alpha_discovery_lab_v44.py",
        "batch": "v31-v45",
        "title": "Annual Consistency & Underwater Reduction Tuning",
        "architectures": "Underwater duration pruning, Multi-year consistency calibration"
    },
    "v45": {
        "script": "alpha_discovery_lab_v45.py",
        "batch": "v31-v45",
        "title": "Final 15% Annual Yield Calibration for Cross-Currency Pairs",
        "architectures": "Cross-currency volatility calibration, Asymmetric runner optimization"
    },

    # Batch 6: Cross-Currency Specialists (AUDNZD & EURGBP Expansion Labs)
    "audnzd_mean_reversion": {
        "script": "alpha_discovery_lab_audnzd_mean_reversion.py",
        "batch": "specialists",
        "title": "AUDNZD Mean-Reversion & Liquidity Sweep (High-Speed)",
        "architectures": "Sydney sweep fade (Turtle Soup), Multi-day Fib Golden Pocket (LB36-72), Macro H4 Donchian"
    },
    "audnzd_expansion": {
        "script": "alpha_discovery_lab_audnzd_expansion.py",
        "batch": "specialists",
        "title": "AUDNZD Breakout & Trend Expansion Screening",
        "architectures": "Sydney 30m ORB, Dual EMA pullbacks, Keltner midline expansions"
    },
    "audnzd_cross_specialists": {
        "script": "alpha_discovery_lab_audnzd_cross_specialists.py",
        "batch": "specialists",
        "title": "AUDNZD Cross-Currency Specialist Alpha Mining",
        "architectures": "Pacific session microstructure, Tokyo liquidity anchors, Spread resistance"
    },
    "audnzd_peer_fleet": {
        "script": "alpha_discovery_lab_audnzd_peer_fleet.py",
        "batch": "specialists",
        "title": "AUDNZD Systematic Peer Fleet Migration Audit",
        "architectures": "Testing 44 EURUSD and 42 USDJPY champions directly on AUDNZD data"
    },
    "audnzd_v2": {
        "script": "alpha_discovery_lab_audnzd_v2.py",
        "batch": "specialists",
        "title": "AUDNZD High-Edge Multi-Session Discovery",
        "architectures": "Pacific Open breakout, Tokyo overlap drive, Asian range mean reversion"
    },
    "audnzd_v3": {
        "script": "alpha_discovery_lab_audnzd_v3.py",
        "batch": "specialists",
        "title": "AUDNZD Precision Macro Structure Calibration",
        "architectures": "H4 macro structure wall retests with wide asymmetric runners (DISC111, DISC112)"
    },
    "eurgbp_expansion": {
        "script": "alpha_discovery_lab_eurgbp_expansion.py",
        "batch": "specialists",
        "title": "EURGBP Specialist Expansion & Deep Alpha Mining",
        "architectures": "Session VWAP 2.0-SD fade (DISC121), Fib Golden Pocket LB36/48 (DISC122, DISC125), Daily Pivot sniper (DISC123)"
    },
}


# ==============================================================================
# Execution Helpers
# ==============================================================================

def print_catalog():
    print("\n" + "=" * 135)
    print("  COMPLETE MASTER CATALOG: ALL 53 ALPHA DISCOVERY LABS (V1 TO V45 + SPECIALISTS)")
    print("=" * 135)
    print(f"{'Lab ID':<24} | {'Batch':<12} | {'Lab Title':<45} | {'Architectures Tested'}")
    print("-" * 135)

    current_batch = None
    for lab_id, info in LAB_CATALOG.items():
        if info["batch"] != current_batch:
            current_batch = info["batch"]
            print(f"--- BATCH: {current_batch.upper()} " + "-" * (115 - len(current_batch)))
        print(f"{lab_id:<24} | {info['batch']:<12} | {info['title'][:43]:<45} | {info['architectures'][:48]}")

    print("-" * 135)
    print(f"Total Available Discovery Labs: {len(LAB_CATALOG)}\n")


def run_single_lab(lab_id: str, symbol: str) -> bool:
    if lab_id not in LAB_CATALOG:
        print(f"ERROR: Unknown lab ID '{lab_id}'. Run with --list to view all available labs.")
        return False

    info = LAB_CATALOG[lab_id]
    script_path = DISCOVERY_LABS_DIR / info["script"]

    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return False

    print("\n" + "=" * 90)
    print(f"  EXECUTING LAB [{lab_id}]: {info['title']}")
    print(f"  Target Symbol: {symbol.upper()} | Script: {info['script']}")
    print("=" * 90)

    # Set environment variable so the lab targets the requested symbol
    env = os.environ.copy()
    env["DISCOVERY_TARGET_SYMBOL"] = symbol.upper()

    t0 = time.time()
    try:
        cmd = [PYTHON, str(script_path)]
        res = subprocess.run(cmd, env=env, cwd=str(ROOT_DIR))
        elapsed = time.time() - t0
        print(f"\n[LAB {lab_id}] Completed in {elapsed:.1f}s (Exit code: {res.returncode})")
        return res.returncode == 0
    except Exception as e:
        print(f"\n[LAB {lab_id}] Failed with exception: {e}")
        return False


def run_batch_labs(batch_name: str, symbol: str):
    batch_name = batch_name.lower()
    selected_labs = []

    if batch_name == "all":
        selected_labs = list(LAB_CATALOG.keys())
    else:
        for lab_id, info in LAB_CATALOG.items():
            if info["batch"] == batch_name or lab_id == batch_name:
                selected_labs.append(lab_id)

    if not selected_labs:
        print(f"ERROR: No labs found for batch '{batch_name}'. Available batches: v1-v10, v11-v20, v21-v30, v31-v45, specialists, all")
        return

    print("\n" + "=" * 90)
    print(f"  EXECUTING BATCH [{batch_name.upper()}] ACROSS {len(selected_labs)} LABS FOR {symbol.upper()}")
    print("=" * 90)

    start_time = time.time()
    successful = 0
    failed = 0

    for idx, lab_id in enumerate(selected_labs, 1):
        print(f"\n>>> Running [{idx}/{len(selected_labs)}]: {lab_id}...")
        ok = run_single_lab(lab_id, symbol)
        if ok:
            successful += 1
        else:
            failed += 1

    total_time = time.time() - start_time
    print("\n" + "=" * 90)
    print(f"  BATCH [{batch_name.upper()}] SUMMARY FOR {symbol.upper()}")
    print("=" * 90)
    print(f"Total Labs Run: {len(selected_labs)} | Successful: {successful} | Failed: {failed}")
    print(f"Total Execution Time: {total_time:.1f}s")
    print(f"Check results/ for all generated scorecards.")


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Master Discovery Labs Orchestrator & Runner")
    parser.add_argument("--list", action="store_true", help="List all 53 discovery labs in catalog")
    parser.add_argument("--symbol", type=str, help="Target symbol ticker (e.g. EURUSD, USDJPY, EURGBP, AUDNZD, USATECHIDXUSD)")
    parser.add_argument("--lab", type=str, help="Run a specific lab ID (e.g. v33, v12, audnzd_mean_reversion)")
    parser.add_argument("--batch", type=str, help="Run a batch of labs: v1-v10, v11-v20, v21-v30, v31-v45, specialists, all")

    args = parser.parse_args()

    if args.list:
        print_catalog()
        return

    if not args.symbol:
        print("ERROR: --symbol is required unless --list is specified.")
        parser.print_help()
        sys.exit(1)

    if args.lab:
        run_single_lab(args.lab, args.symbol)
    elif args.batch:
        run_batch_labs(args.batch, args.symbol)
    else:
        print("ERROR: Please specify either --lab <ID> or --batch <BATCH_NAME> or --list.")
        parser.print_help()


if __name__ == "__main__":
    main()

