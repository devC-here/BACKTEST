#!/usr/bin/env python3
import re
from pathlib import Path

doc_path = Path(r"C:\Users\kdev2\.gemini\antigravity-ide\brain\094e6ad4-9870-426f-8e43-7458d9074e00\institutional_alpha_discovery_master_breakthrough.md")
with open(doc_path, "r", encoding="utf-8") as f:
    text = f.read()

lines = text.split("\n")
parsed = []
current_strat = None

for line in lines:
    m_strat = re.match(r"^## \d+\.\s+(DISC\d+.*?)(?:\s*\(.*?\))?$", line.strip())
    if m_strat:
        current_strat = m_strat.group(1).strip()

    if ("Full 5-Year" in line or "Full 1-Year" in line) and not line.strip().startswith("TOTAL"):
        parts = [p.strip() for p in line.split() if p.strip()]
        if len(parts) >= 7:
            asset = parts[0]
            # Find net pips
            pips_val = None
            trades_val = None
            wr_val = None
            pf_val = None
            for p in parts:
                if (p.startswith("+") or p.startswith("-")) and p.endswith("p"):
                    pips_val = float(p.replace("+", "").replace("p", "").replace(",", ""))
                elif p.endswith("%"):
                    wr_val = p

            for p in parts[2:]:
                if p.isdigit():
                    trades_val = int(p)
                    break

            for p in parts[4:]:
                try:
                    val = float(p.replace(",", ""))
                    if 0.5 <= val <= 25.0 and val != float(trades_val or 0):
                        pf_val = val
                        break
                except:
                    pass

            if pips_val is not None and current_strat:
                parsed.append({
                    "strategy": current_strat,
                    "asset": asset,
                    "trades": trades_val or 0,
                    "win_rate": wr_val or "N/A",
                    "profit_factor": pf_val or 0.0,
                    "net_pips": pips_val
                })

print(f"Total parsed individual strategy-asset instances: {len(parsed)}")
tot_pips = sum(r["net_pips"] for r in parsed)
print(f"Total Cumulative Pips in parsed instances: +{tot_pips:.1f} Pips")
print(f"Average Annual Pips in parsed instances:   +{tot_pips / 5.0:.1f} Pips / Year")

by_asset = {}
for r in parsed:
    a = r["asset"]
    if a not in by_asset:
        by_asset[a] = []
    by_asset[a].append(r)

for a, r_list in by_asset.items():
    a_pips = sum(x["net_pips"] for x in r_list)
    a_trades = sum(x["trades"] for x in r_list)
    print(f"\nAsset: {a} -> {len(r_list)} models | {a_trades} trades | +{a_pips:.1f} pips (+{a_pips/5.0:.1f} pips/yr)")
    for x in sorted(r_list, key=lambda k: -k["net_pips"])[:8]:
        print(f"   {x['strategy'][:40]:40s} | {x['trades']:5d} tr | {x['win_rate']:>6s} | {x['profit_factor']:4.2f} PF | +{x['net_pips']:8.1f}p")
