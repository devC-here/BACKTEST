import json
from pathlib import Path

p = Path("results/continuous_walkforward_audits/v45_continuous_holdout_audit.json")
if not p.exists():
    print("File not found")
    exit(1)

with open(p, "r", encoding="utf-8") as f:
    data = json.load(f)

strategies = data.get("strategies", [])
survivors = [s for s in strategies if s.get("net_pips", 0) > 0]
print(f"Total evaluated strategies: {len(strategies)}")
print(f"Positive strategies on continuous holdout: {len(survivors)}")
print("-" * 80)
for s in sorted(survivors, key=lambda x: x.get("net_pips", 0), reverse=True):
    name = s.get("strategy_file") or s.get("name") or s.get("file") or str(s.get("slot"))
    print(f"{name:<45} | Net: {s['net_pips']:>+7.1f}p | PF: {s.get('profit_factor', 0):>4.2f} | Trades: {s['trades']:>3} | WR: {s.get('win_rate', 0):>4.1f}%")
