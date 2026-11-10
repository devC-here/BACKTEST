import json
from pathlib import Path

manifest_dir = Path("results/certified_strategy_manifests")
manifest_files = sorted(manifest_dir.glob("*_run_manifest.json"))

print(f"{'Strategy Name':42s} | {'Symbol':6s} | {'Hold PnL':>9s} | {'Hold PF':>7s} | {'Trades':>6s} | {'Exp':>6s} | {'Max DD':>7s} | {'Status'}")
print("=" * 105)

total_hold_pips = 0.0
total_hold_trades = 0
certified_count = 0

for mf in manifest_files:
    with open(mf, "r") as f:
        data = json.load(f)
    
    name = data.get("strategy", {}).get("name", mf.stem.replace("_run_manifest", ""))
    sym = data.get("dataset", {}).get("symbol", "EURUSD")
    stats = data.get("statistical_metrics", {})
    pips = stats.get("net_pips", 0.0)
    pf = stats.get("profit_factor", 0.0)
    trades = stats.get("total_trades", 0)
    exp = stats.get("expectancy_pips", 0.0)
    dd = stats.get("max_drawdown_pips", 0.0)
    status = data.get("certification_status", "UNKNOWN")

    if status == "DUAL_CERTIFIED" or pips > 0:
        certified_count += 1
        total_hold_pips += pips
        total_hold_trades += trades
        print(f"{name[:42]:42s} | {sym:6s} | {pips:>8.1f}p | {pf:>7.2f} | {trades:>6d} | {exp:>5.2f}p | {dd:>6.1f}p | {status}")

print("=" * 105)
print(f"Total Certified/Profitable Strategies: {certified_count}")
print(f"Total Quarantined Holdout Profit:      {total_hold_pips:+.1f} net pips across {total_hold_trades} trades")

