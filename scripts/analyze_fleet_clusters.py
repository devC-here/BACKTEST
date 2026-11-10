import json
from pathlib import Path
from collections import defaultdict

p = Path("results/continuous_walkforward_audits/v45_continuous_holdout_audit.json")
with open(p, "r", encoding="utf-8") as f:
    data = json.load(f)

strategies = data.get("strategies", [])

# Group by archetype/symbol
by_sym = defaultdict(list)
for s in strategies:
    sym = s.get("symbol", "UNKNOWN")
    by_sym[sym].append(s)

print("=" * 90)
print(f"{'SYMBOL':<10} | {'STRATS':<6} | {'TOTAL PIPS':<12} | {'WINNERS':<8} | {'TOP STRAT':<35} | {'TOP PIPS':<8}")
print("=" * 90)

for sym, strats in sorted(by_sym.items(), key=lambda x: sum(s.get("net_pips", 0) for s in x[1]), reverse=True):
    tot_pips = sum(s.get("net_pips", 0) for s in strats)
    winners = sum(1 for s in strats if s.get("net_pips", 0) > 0)
    top_s = max(strats, key=lambda x: x.get("net_pips", 0))
    top_name = top_s.get("file", top_s.get("name", "unk"))
    top_pips = top_s.get("net_pips", 0)
    print(f"{sym:<10} | {len(strats):<6} | {tot_pips:>+10.1f}p | {winners:>2}/{len(strats):<5} | {top_name:<35} | {top_pips:>+7.1f}p")

print("=" * 90)

# Detail breakdown by slot cluster
print("\n" + "=" * 90)
print("CLUSTER BREAKDOWN ACROSS 100 SLOTS")
print("=" * 90)

clusters = [
    ("Slots 01-08: GBPJPY H4 Macro Retest", 1, 8),
    ("Slots 09-16: EURUSD NY Sweep Fade", 9, 16),
    ("Slots 17-24: USDCAD Asian Sweep", 17, 24),
    ("Slots 25-32: GBPCHF Asian Sweep", 25, 32),
    ("Slots 33-40: AUDCAD H4 Macro Retest", 33, 40),
    ("Slots 41-48: AUDNZD Fleet", 41, 48),
    ("Slots 49-56: EURNZD H4 Macro Retest", 49, 56),
    ("Slots 57-61, 73-80: NZDCAD NY Sweep", 57, 80),
    ("Slots 62, 65-72, 87, 89, 90: EURGBP London Sweep", 62, 90),
    ("Slots 63, 64, 81-86, 91-100: USDJPY/GBPJPY Micro Trend/Breakout", 63, 100),
]

for label, start_s, end_s in clusters:
    matching = [s for s in strategies if start_s <= s.get("slot", 0) <= end_s]
    if not matching:
        continue
    tot = sum(s.get("net_pips", 0) for s in matching)
    pos = sum(1 for s in matching if s.get("net_pips", 0) > 0)
    avg_pf = sum(s.get("profit_factor", 0) for s in matching) / len(matching)
    print(f"{label:<55} | {pos:>2}/{len(matching):<2} win | Net: {tot:>+9.1f}p | Avg PF: {avg_pf:.2f}")

print("=" * 90)

