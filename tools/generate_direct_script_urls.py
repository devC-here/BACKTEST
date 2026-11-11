import json
import os

RAW_BASE = "https://raw.githubusercontent.com/devC-here/BACKTEST/main/"
GITHUB_BASE = "https://github.com/devC-here/BACKTEST/blob/main/"

with open("certified_production_fleet/CERTIFIED_FLEET_MANIFEST.json", "r", encoding="utf-8") as f:
    manifest = json.load(f)

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
        "name": "Master 72-Strategy Fleet Manifest (JSON)",
        "path": "certified_production_fleet/CERTIFIED_FLEET_MANIFEST.json",
        "description": "Cryptographically verified master manifest with holdout metrics, SHA-256 hashes, and portfolio totals."
    },
    {
        "name": "Institutional Strategy Verification Platform Report",
        "path": "docs/INSTITUTIONAL_STRATEGY_VERIFICATION_PLATFORM_REPORT.md",
        "description": "Comprehensive 10-point audit report verifying all 72 dual-partition champions (+11,723.3p holdout PnL)."
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
        "description": "Batch certifies strategies across all 6 currency pairs with mandatory broker friction."
    },
    {
        "name": "Master Split Generator",
        "path": "scripts/build_master_continuous_splits.py",
        "description": "Constructs the continuous chronological partitions (Discovery 1..210 vs Quarantined Holdout 211..300)."
    },
    {
        "name": "Fleet Manifest Generator",
        "path": "scripts/generate_certified_fleet_manifest.py",
        "description": "Compiles individual strategy run manifests into the canonical CERTIFIED_FLEET_MANIFEST.json."
    }
]

# Collect all strategy files from certified_production_fleet
strategy_files = []
for root, dirs, files in os.walk("certified_production_fleet"):
    for f in files:
        if f.endswith(".py") and f != "__init__.py":
            rel_path = os.path.relpath(os.path.join(root, f), ".").replace("\\", "/")
            strategy_files.append(rel_path)

strategy_files.sort()

# Build mapping of strategy details from manifest
manifest_lookup = {}
for s in manifest["strategies"]:
    manifest_lookup[s["name"].lower()] = s

# Categorize strategies by symbol
strategies_by_symbol = {}
for sf in strategy_files:
    filename = os.path.basename(sf)
    folder = os.path.basename(os.path.dirname(sf))
    
    # determine symbol
    sym = "UNKNOWN"
    if "eurusd" in sf.lower():
        sym = "EURUSD"
    elif "usdjpy" in sf.lower():
        sym = "USDJPY"
    elif "usdcad" in sf.lower():
        sym = "USDCAD"
    elif "audnzd" in sf.lower():
        sym = "AUDNZD"
    elif "eurgbp" in sf.lower():
        sym = "EURGBP"
    elif "gbpchf" in sf.lower():
        sym = "GBPCHF"
    
    # find matching entry in manifest
    matched_entry = None
    for name, entry in manifest_lookup.items():
        clean_name = name.replace("eurusd_", "").replace("usdjpy_", "").replace("audnzd_", "").replace("eurgbp_", "").replace("gbpchf_", "").replace("usdcad_", "")
        if clean_name in sf.lower() or name in sf.lower() or os.path.splitext(filename)[0] in name:
            matched_entry = entry
            break
    
    holdout_pnl = matched_entry["holdout_metrics"]["net_pips"] if matched_entry else "N/A"
    holdout_pf = matched_entry["holdout_metrics"]["profit_factor"] if matched_entry else "N/A"
    trades = matched_entry["holdout_metrics"]["total_trades"] if matched_entry else "N/A"
    
    strat_item = {
        "symbol": sym,
        "path": sf,
        "name": folder if folder != "certified_production_fleet" and not folder.endswith("_sweeps") and not folder.endswith("_champions") else os.path.splitext(filename)[0],
        "holdout_net_pips": holdout_pnl,
        "holdout_profit_factor": holdout_pf,
        "holdout_trades": trades,
        "raw_url": RAW_BASE + sf,
        "github_url": GITHUB_BASE + sf
    }
    strategies_by_symbol.setdefault(sym, []).append(strat_item)

# Build full JSON data structure
export_json = {
    "repository": "https://github.com/devC-here/BACKTEST",
    "raw_base": RAW_BASE,
    "total_strategies": len(strategy_files),
    "core_platform_scripts": [
        {
            **item,
            "raw_url": RAW_BASE + item["path"],
            "github_url": GITHUB_BASE + item["path"]
        }
        for item in core_platform_scripts
    ],
    "executive_documentation": [
        {
            **item,
            "raw_url": RAW_BASE + item["path"],
            "github_url": GITHUB_BASE + item["path"]
        }
        for item in executive_docs
    ],
    "operational_scripts": [
        {
            **item,
            "raw_url": RAW_BASE + item["path"],
            "github_url": GITHUB_BASE + item["path"]
        }
        for item in operational_scripts
    ],
    "certified_strategies_by_symbol": strategies_by_symbol
}

# Write JSON
with open("DIRECT_SCRIPT_URLS.json", "w", encoding="utf-8") as f:
    json.dump(export_json, f, indent=2)

print("Saved DIRECT_SCRIPT_URLS.json successfully!")

# Write Markdown
md_lines = []
md_lines.append("# Direct & Raw Script URLs Index for BACKTEST Platform")
md_lines.append("")
md_lines.append("> **Repository**: [`https://github.com/devC-here/BACKTEST`](https://github.com/devC-here/BACKTEST)")
md_lines.append("> **Raw Content Base**: `https://raw.githubusercontent.com/devC-here/BACKTEST/main/`")
md_lines.append("> **Status**: Verified Public Repository | 72 Certified Strategies | +11,723.3p Holdout PnL")
md_lines.append("")
md_lines.append("This document provides immediate, one-click raw and direct GitHub links to every core engine script, verification module, executive audit, and all **72 production-certified quantitative trading strategies** in this repository.")
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
md_lines.append("## 3. ⚙️ Operational, Certification & Calibration Scripts")
md_lines.append("")
md_lines.append("| Script | Functionality | Direct GitHub Link | Raw Download / API URL |")
md_lines.append("|:---|:---|:---:|:---:|")
for item in export_json["operational_scripts"]:
    md_lines.append(f"| **{item['name']}** (`{item['path']}`) | {item['description']} | [View on GitHub]({item['github_url']}) | [Raw Content]({item['raw_url']}) |")
md_lines.append("")
md_lines.append("---")
md_lines.append("")

# Section 4: All 72 Certified Strategies by Symbol
md_lines.append("## 4. 🏆 Certified Production Strategies Fleet (72 Strategies)")
md_lines.append("")

symbols_order = ["EURUSD", "USDJPY", "USDCAD", "AUDNZD", "EURGBP", "GBPCHF"]
for sym in symbols_order:
    strats = strategies_by_symbol.get(sym, [])
    md_lines.append(f"### {sym} Champions ({len(strats)} Strategies)")
    md_lines.append("")
    md_lines.append("| # | Strategy Name | Holdout PnL | Profit Factor | Trades | Direct GitHub Link | Raw Python Script |")
    md_lines.append("|:---:|:---|:---:|:---:|:---:|:---:|:---:|")
    for idx, s in enumerate(strats, 1):
        pnl_str = f"+{s['holdout_net_pips']}p" if isinstance(s['holdout_net_pips'], (int, float)) and s['holdout_net_pips'] > 0 else f"{s['holdout_net_pips']}p"
        pf_str = f"{s['holdout_profit_factor']:.2f}" if isinstance(s['holdout_profit_factor'], (int, float)) else str(s['holdout_profit_factor'])
        tr_str = str(s['holdout_trades'])
        md_lines.append(f"| {idx} | **`{s['name']}`** | **{pnl_str}** | {pf_str} | {tr_str} | [GitHub Link]({s['github_url']}) | [Raw Python]({s['raw_url']}) |")
    md_lines.append("")

with open("DIRECT_SCRIPT_URLS.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print("Saved DIRECT_SCRIPT_URLS.md successfully!")
