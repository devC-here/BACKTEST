import json
from pathlib import Path

p = Path("results/continuous_walkforward_audits/cross_pairs_screen.json")
with open(p, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data[25:]:
    sym = item["symbol"]
    name = item["name"]
    h = item.get("holdout", {})
    d = item.get("discovery", {})
    hp = h.get("net_pips", 0.0)
    dp = d.get("net_pips", 0.0)
    ht = h.get("total_trades", 0)
    dt = d.get("total_trades", 0)
    print(f"[{sym:6s}] {name:38s} | Hold: {hp:>+6.1f}p ({ht:>2} tr) | Disc: {dp:>+6.1f}p ({dt:>2} tr)")
