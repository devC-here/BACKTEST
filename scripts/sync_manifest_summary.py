import json
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).resolve().parent.parent
manifest_path = ROOT_DIR / "certified_production_fleet" / "CERTIFIED_FLEET_MANIFEST.json"

with open(manifest_path, "r", encoding="utf-8") as f:
    data = json.load(f)

strats = data["strategies"]
pair_counts = dict(Counter([s["symbol"] for s in strats]))

tot_pips = sum(s.get("holdout_metrics", {}).get("net_pips", 0.0) for s in strats)
tot_trades = sum(s.get("holdout_metrics", {}).get("total_trades", 0) for s in strats)
pfs = [s.get("holdout_metrics", {}).get("profit_factor", 0.0) for s in strats if s.get("holdout_metrics", {}).get("profit_factor")]
avg_pf = round(sum(pfs) / len(pfs), 2) if pfs else 0.0
avg_exp = round(tot_pips / tot_trades, 2) if tot_trades > 0 else 0.0

data["fleet_version"] = "6.7.0_pristine_70_elite"
data["manifest_version"] = "6.7.0_pristine_70_elite"
data["total_strategies"] = len(strats)
data["strategy_count"] = len(strats)
data["last_updated"] = datetime.now(timezone.utc).isoformat()

data["fleet_summary"] = {
    "total_strategies": len(strats),
    "total_pairs": len(pair_counts),
    "total_holdout_pips": round(tot_pips, 1),
    "total_holdout_trades": tot_trades,
    "average_profit_factor": avg_pf,
    "average_expectancy_pips": avg_exp,
    "pair_breakdown": pair_counts
}

data["portfolio_totals"] = {
    "total_strategies": len(strats),
    "holdout_net_pips": round(tot_pips, 1),
    "holdout_total_trades": tot_trades,
    "holdout_expectancy_pips": avg_exp,
    "pruned_strategies_count": 15,
    "pruning_criteria": "Holdout Trades >= 15, Holdout PF >= 1.15, Pips ex Top 3 > 0, Discovery Pips > 0"
}

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"Updated {manifest_path} to version {data['manifest_version']} with {len(strats)} strategies across {len(pair_counts)} pairs.")

