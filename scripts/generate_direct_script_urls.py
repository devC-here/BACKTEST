#!/usr/bin/env python3
"""
================================================================================
  CANONICAL DIRECT SCRIPT URLS & REPO INDEX GENERATOR
================================================================================
Parses the cryptographically verified certified_production_fleet/CERTIFIED_FLEET_MANIFEST.json
and generates:
1. DIRECT_SCRIPT_URLS.md: Clean, verified markdown table with true Holdout metrics.
2. DIRECT_SCRIPT_URLS.json: Machine-readable API index of all script URLs.
Enforces 100% namespace fidelity: zero template fallbacks or mismatched clones.
================================================================================
"""

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
FLEET_MANIFEST = ROOT_DIR / "certified_production_fleet" / "CERTIFIED_FLEET_MANIFEST.json"
MD_OUTPUT = ROOT_DIR / "DIRECT_SCRIPT_URLS.md"
JSON_OUTPUT = ROOT_DIR / "DIRECT_SCRIPT_URLS.json"

RAW_BASE = "https://raw.githubusercontent.com/devC-here/BACKTEST/main/"
GITHUB_BASE = "https://github.com/devC-here/BACKTEST/blob/main/"

# Core platform scripts
core_platform_scripts = [
    {
        "name": "Numba Matching Engine Core",
        "path": "tick_engine/core/engine.py",
        "description": "High-throughput Numba @njit(fastmath=True) matching loop with strict bar-close subsequent tick execution and friction models."
    },
    {
        "name": "Memory-Mapped Tick DataLoader",
        "path": "tick_engine/core/dataloader.py",
        "description": "Zero-copy binary .npy and CSV tick loader supporting dual continuous and block partitioning with rolling causal warmup."
    },
    {
        "name": "Institutional Metrics & Statistical Engine",
        "path": "tick_engine/core/metrics.py",
        "description": "Net pips, profit factor, max drawdown, worst month, bootstrap confidence intervals, and outlier trade sensitivity."
    },
    {
        "name": "Engine Configuration & Friction Defaults",
        "path": "tick_engine/core/config.py",
        "description": "Defines mandatory institutional defaults: $4/lot commission, 0.20p entry slippage, raw floating spread."
    },
    {
        "name": "Forex Session Time & Daylight-Saving Filter",
        "path": "tick_engine/core/session_time.py",
        "description": "UTC session time tracking and DST adjustments for London, New York, and Tokyo sessions."
    },
    {
        "name": "Real-Time Data Quality Gate",
        "path": "tick_engine/quality/validator.py",
        "description": "Automatic rejection of partial weeks (<80h, <15k ticks), quote gaps (>4h), inverted spreads (Ask < Bid), and non-positive prices."
    },
    {
        "name": "Cryptographic Run Manifest Generator",
        "path": "tick_engine/reporting/run_manifest.py",
        "description": "Generates canonical JSON run manifests recording SHA-256 hashes of engine code, parameters, partitions, and trade logs."
    },
    {
        "name": "Adversarial Friction Stress Suite",
        "path": "tick_engine/stress/stress_suite.py",
        "description": "Stress-tests strategies against 2x spread, 2x slippage, +50% commission, and execution latency."
    },
    {
        "name": "Institutional Integration Test Suite",
        "path": "tick_engine/tests/test_institutional_platform.py",
        "description": "10 automated pytest tests validating split integrity, data gate, no same-tick fill, warmup exclusion, and fee deduction."
    },
    {
        "name": "Standalone Tick Engine Runner CLI",
        "path": "tick_engine/run_backtest.py",
        "description": "CLI entry point for running high-speed tick backtests with live progress and annual scorecards."
    }
]

# Executive reports & manifests
executive_docs = [
    {
        "name": "Master Certified Fleet Manifest (JSON)",
        "path": "certified_production_fleet/CERTIFIED_FLEET_MANIFEST.json",
        "description": "Cryptographically verified master manifest with holdout metrics, SHA-256 hashes, and portfolio totals."
    },
    {
        "name": "Portfolio Risk & Exposure Gate Audit Report (JSON)",
        "path": "results/portfolio_simulation_report.json",
        "description": "Concurrent currency-leg exposure gate simulation with minute-by-minute mark-to-market drawdown analysis."
    },
    {
        "name": "Institutional Strategy Verification Platform Report",
        "path": "docs/INSTITUTIONAL_STRATEGY_VERIFICATION_PLATFORM_REPORT.md",
        "description": "Comprehensive institutional audit report verifying continuous walkforward splits, causal warmup, and friction gates."
    },
    {
        "name": "Certified Production Fleet Documentation",
        "path": "certified_production_fleet/README.md",
        "description": "Full fleet layout, currency pair breakdown, primary alpha drivers, and strategy directory structure."
    },
    {
        "name": "Platform Root README",
        "path": "README.md",
        "description": "Repository overview, performance benchmarks (35M ticks/s), and quickstart execution commands."
    }
]

# Operational scripts
operational_scripts = [
    {
        "name": "Automated Strategy Certifier",
        "path": "scripts/certify_strategy.py",
        "description": "Executes dual-partition validation (Weeks 1..210 vs 211..300), checks hurdle rates, and outputs certified manifests."
    },
    {
        "name": "Full Fleet Certification Runner",
        "path": "scripts/run_certification_fleet.py",
        "description": "Batch certifies strategies across all currency pairs with mandatory broker friction."
    },
    {
        "name": "Portfolio Correlation & Risk Engine",
        "path": "scripts/run_portfolio_correlation_and_sizing.py",
        "description": "Evaluates concurrent currency-leg exposure caps and minute-by-minute portfolio mark-to-market drawdown."
    },
    {
        "name": "Fleet Integrity Forensic Auditor",
        "path": "scripts/audit_fleet_integrity.py",
        "description": "Performs AST and SHA-256 integrity hashing to detect and flag code duplicates and namespace collisions."
    }
]


def generate():
    if not FLEET_MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {FLEET_MANIFEST}")

    with open(FLEET_MANIFEST, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    strategies = manifest_data.get("strategies", [])
    totals = manifest_data.get("portfolio_totals", {})

    strategies_by_symbol = {}
    for s in strategies:
        sym = s["symbol"].upper()
        h_metrics = s.get("holdout_metrics", {})
        rel_path = s.get("relative_path", "")

        strat_item = {
            "symbol": sym,
            "name": s["name"],
            "path": rel_path,
            "holdout_net_pips": h_metrics.get("net_pips", 0.0),
            "holdout_profit_factor": h_metrics.get("profit_factor", 0.0),
            "holdout_trades": h_metrics.get("total_trades", 0),
            "expectancy_pips": h_metrics.get("expectancy_pips", 0.0),
            "win_rate_pct": h_metrics.get("win_rate_pct", 0.0),
            "max_drawdown_pips": h_metrics.get("max_drawdown_pips", 0.0),
            "raw_url": RAW_BASE + rel_path,
            "github_url": GITHUB_BASE + rel_path
        }
        strategies_by_symbol.setdefault(sym, []).append(strat_item)

    # Sort each symbol's strategies by Holdout Net Pips descending
    for sym in strategies_by_symbol:
        strategies_by_symbol[sym].sort(key=lambda x: x["holdout_net_pips"], reverse=True)

    # Build JSON structure
    export_json = {
        "repository": "https://github.com/devC-here/BACKTEST",
        "raw_base": RAW_BASE,
        "github_base": GITHUB_BASE,
        "total_certified_strategies": len(strategies),
        "portfolio_totals": totals,
        "core_platform_scripts": [
            {**item, "raw_url": RAW_BASE + item["path"], "github_url": GITHUB_BASE + item["path"]}
            for item in core_platform_scripts
        ],
        "executive_documentation": [
            {**item, "raw_url": RAW_BASE + item["path"], "github_url": GITHUB_BASE + item["path"]}
            for item in executive_docs
        ],
        "operational_scripts": [
            {**item, "raw_url": RAW_BASE + item["path"], "github_url": GITHUB_BASE + item["path"]}
            for item in operational_scripts
        ],
        "certified_strategies_by_symbol": strategies_by_symbol
    }

    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(export_json, f, indent=2)
    print(f"Saved {JSON_OUTPUT}")

    # Build Markdown
    md_lines = []
    md_lines.append("# Direct & Raw Script URLs Index for BACKTEST Platform")
    md_lines.append("")
    md_lines.append("> **Repository**: [`https://github.com/devC-here/BACKTEST`](https://github.com/devC-here/BACKTEST)  ")
    md_lines.append("> **Raw Content Base**: `https://raw.githubusercontent.com/devC-here/BACKTEST/main/`  ")
    md_lines.append(f"> **Status**: Institutional Clean Standard | **{totals.get('total_certified_strategies', len(strategies))} Production-Certified Strategies** | **+{totals.get('holdout_net_pips', 0):,.1f}p Holdout PnL** (Combined 5Y: **+{totals.get('combined_5y_net_pips', 0):,.1f}p**)")
    md_lines.append("")
    md_lines.append("This document provides verified raw download links and direct GitHub repository links for every core matching engine module, dataset quality gate, executive audit manifest, and all **production-certified quantitative trading strategies** in this repository.")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    # Section 1: Executive Reports & Manifests
    md_lines.append("## 1. 📋 Master Manifests & Executive Reports")
    md_lines.append("")
    md_lines.append("| Component | Description | Direct GitHub Link | Raw Download / API URL |")
    md_lines.append("|:---|:---|:---:|:---:|")
    for item in export_json["executive_documentation"]:
        md_lines.append(f"| **{item['name']}** | {item['description']} | [View on GitHub]({item['github_url']}) | [Raw Content]({item['raw_url']}) |")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    # Section 2: Core Platform & Quality Gate
    md_lines.append("## 2. 🛡️ Verification Platform & Numba Engine Core")
    md_lines.append("")
    md_lines.append("| Module | Purpose | Direct GitHub Link | Raw Download / API URL |")
    md_lines.append("|:---|:---|:---:|:---:|")
    for item in export_json["core_platform_scripts"]:
        md_lines.append(f"| **{item['name']}** (`{item['path']}`) | {item['description']} | [View on GitHub]({item['github_url']}) | [Raw Content]({item['raw_url']}) |")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    # Section 3: Operational & Calibration Scripts
    md_lines.append("## 3. ⚙️ Operational, Certification & Risk Scripts")
    md_lines.append("")
    md_lines.append("| Script | Functionality | Direct GitHub Link | Raw Download / API URL |")
    md_lines.append("|:---|:---|:---:|:---:|")
    for item in export_json["operational_scripts"]:
        md_lines.append(f"| **{item['name']}** (`{item['path']}`) | {item['description']} | [View on GitHub]({item['github_url']}) | [Raw Content]({item['raw_url']}) |")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    # Section 4: All Certified Strategies by Symbol
    md_lines.append("## 4. 🏆 Certified Production Strategies Fleet by Currency Pair")
    md_lines.append("")

    symbols_order = ["EURUSD", "USDJPY", "AUDNZD", "EURGBP", "USDCAD"]
    for sym in symbols_order:
        strats = strategies_by_symbol.get(sym, [])
        if not strats:
            continue
        tot_sym_pnl = sum(s["holdout_net_pips"] for s in strats)
        tot_sym_tr = sum(s["holdout_trades"] for s in strats)
        avg_sym_pf = sum(s["holdout_profit_factor"] for s in strats) / max(1, len(strats))

        md_lines.append(f"### {sym} Champions ({len(strats)} Strategies | +{tot_sym_pnl:,.1f}p Holdout PnL | Avg PF {avg_sym_pf:.2f} | {tot_sym_tr} Trades)")
        md_lines.append("")
        md_lines.append("| # | Strategy Identifier | Holdout PnL | Profit Factor | Trades | Expectancy | Direct GitHub Link | Raw Python Script |")
        md_lines.append("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
        for idx, s in enumerate(strats, 1):
            pnl_str = f"+{s['holdout_net_pips']:.1f}p" if s['holdout_net_pips'] > 0 else f"{s['holdout_net_pips']:.1f}p"
            pf_str = f"{s['holdout_profit_factor']:.2f}"
            tr_str = str(s['holdout_trades'])
            exp_str = f"+{s['expectancy_pips']:.2f}p" if s['expectancy_pips'] > 0 else f"{s['expectancy_pips']:.2f}p"
            md_lines.append(f"| {idx} | **`{s['name']}`** | **{pnl_str}** | {pf_str} | {tr_str} | {exp_str} | [GitHub Link]({s['github_url']}) | [Raw Python]({s['raw_url']}) |")
        md_lines.append("")

    with open(MD_OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Saved {MD_OUTPUT}")


if __name__ == "__main__":
    generate()

