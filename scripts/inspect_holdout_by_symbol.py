import json
from collections import defaultdict

with open("results/continuous_walkforward_audits/v45_continuous_holdout_audit.json", "r") as f:
    data = json.load(f)

strats = data.get("strategies", [])

by_symbol = defaultdict(list)
for s in strats:
    by_symbol[s.get("symbol", "UNKNOWN")].append(s)

print(f"{'Symbol':8s} | {'Count':5s} | {'Profitable':10s} | {'Net Pips':>10s} | {'Avg PF':>6s} | {'Archetypes'}")
print("-" * 80)

for sym, items in sorted(by_symbol.items()):
    prof = [x for x in items if x.get("net_pips", 0) > 0]
    total_pips = sum(x.get("net_pips", 0) for x in items)
    pfs = [x.get("profit_factor", 0) for x in items if x.get("profit_factor", 0) > 0]
    avg_pf = sum(pfs) / len(pfs) if pfs else 0.0
    archs = set(x.get("archetype", "Unknown") for x in items)
    arch_str = ", ".join(archs)
    print(f"{sym:8s} | {len(items):5d} | {len(prof):10d} | {total_pips:>10.1f}p | {avg_pf:>6.2f} | {arch_str[:35]}")

