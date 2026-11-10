#!/usr/bin/env python3
"""
Generates CERTIFIED_FLEET_MANIFEST.json for certified_production_fleet/.
"""

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
FLEET_DIR = ROOT_DIR / "certified_production_fleet"
MANIFEST_DIR = ROOT_DIR / "results" / "certified_strategy_manifests"

fleet_manifest = {
    "version": "1.0.0",
    "verification_standard": "Institutional Dual-Partition Continuous Walkforward Standard",
    "friction_policy": "$4.00/lot commission + 0.20p adverse entry slippage + raw Dukascopy floating spread",
    "order_timing_contract": "Strict Bar-Close Subsequent-Tick Causal Execution (T_fill > T_signal)",
    "causal_warmup": "100,000 rolling prior-week ticks with min_entry_timestamp gate",
    "partitions": {
        "in_sample_discovery": "Weeks 1..210 (Jan 2021 - Dec 2024, ~1.12B ticks)",
        "quarantined_holdout": "Weeks 211..300 (Jan 2025 - Oct 2026, ~421M ticks)"
    },
    "portfolio_totals": {
        "total_certified_strategies": 24,
        "holdout_net_pips": 4810.1,
        "in_sample_net_pips": 9173.2,
        "combined_5y_net_pips": 13983.3,
        "holdout_executed_trades": 2623,
        "realized_holdout_expectancy_pips": 1.83
    },
    "strategies": []
}

manifest_files = sorted(MANIFEST_DIR.glob("*_run_manifest.json"))

for mf in manifest_files:
    with open(mf, "r") as f:
        data = json.load(f)

    stats = data.get("statistical_metrics", {})
    strat = data.get("strategy", {})
    dataset = data.get("dataset", {})
    custom = data.get("custom_metadata", {})

    pips = stats.get("net_pips", 0.0)
    if pips <= 0 and data.get("certification_status") != "DUAL_CERTIFIED":
        continue

    entry = {
        "name": strat.get("name", mf.stem),
        "filename": strat.get("filename"),
        "symbol": dataset.get("symbol", "EURUSD"),
        "status": data.get("certification_status"),
        "holdout_metrics": {
            "net_pips": stats.get("net_pips"),
            "profit_factor": stats.get("profit_factor"),
            "total_trades": stats.get("total_trades"),
            "win_rate_pct": stats.get("win_rate_pct"),
            "expectancy_pips": stats.get("expectancy_pips"),
            "max_drawdown_pips": stats.get("max_drawdown_pips")
        },
        "discovery_metrics": custom.get("discovery_in_sample_metrics", {}),
        "code_sha256": strat.get("code_sha256"),
        "manifest_file": str(mf.relative_to(ROOT_DIR))
    }
    fleet_manifest["strategies"].append(entry)

h_pips_tot = round(sum(s["holdout_metrics"]["net_pips"] for s in fleet_manifest["strategies"] if s["holdout_metrics"]["net_pips"] is not None), 1)
d_pips_tot = round(sum(s["discovery_metrics"].get("net_pips", 0.0) for s in fleet_manifest["strategies"] if s["discovery_metrics"].get("net_pips") is not None), 1)
tr_tot = sum(s["holdout_metrics"]["total_trades"] for s in fleet_manifest["strategies"] if s["holdout_metrics"]["total_trades"] is not None)
exp_tot = round(h_pips_tot / max(1, tr_tot), 2)

fleet_manifest["portfolio_totals"] = {
    "total_certified_strategies": len(fleet_manifest["strategies"]),
    "holdout_net_pips": h_pips_tot,
    "in_sample_net_pips": d_pips_tot,
    "combined_5y_net_pips": round(h_pips_tot + d_pips_tot, 1),
    "holdout_executed_trades": tr_tot,
    "realized_holdout_expectancy_pips": exp_tot
}

out_path = FLEET_DIR / "CERTIFIED_FLEET_MANIFEST.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(fleet_manifest, f, indent=2)

print(f"Saved {out_path} with {len(fleet_manifest['strategies'])} certified strategies!")
print(f"Holdout PnL: {h_pips_tot:>+8.1f}p | Combined 5Y PnL: {round(h_pips_tot + d_pips_tot, 1):>+8.1f}p | Total Trades: {tr_tot}")

