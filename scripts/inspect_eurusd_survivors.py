import json

with open("results/continuous_walkforward_audits/eurusd_older_strategies_screen.json", "r") as f:
    data = json.load(f)

# Filter for dual-partition profitable strategies (Holdout > 0 AND Discovery > 0)
dual_profitable = [
    x for x in data
    if x["holdout"]["net_pips"] > 0 and x["discovery"]["net_pips"] > 0 and x["holdout"]["pf"] >= 1.05 and x["discovery"]["pf"] >= 1.05
]

dual_profitable.sort(key=lambda x: (x["holdout"]["net_pips"] + x["discovery"]["net_pips"]), reverse=True)

print("=" * 110)
print(f"  CERTIFIED DUAL-PARTITION SURVIVORS: {len(dual_profitable)} OUT OF {len(data)} STRATEGIES")
print("=" * 110)
print(f"{'Rank':4s} | {'Strategy Name':38s} | {'Hold PnL':>9s} | {'Hold PF':>7s} | {'Hold Tr':>7s} | {'Disc PnL':>9s} | {'Disc PF':>7s} | {'Total PnL':>10s}")
print("-" * 110)

tot_hold_pnl = 0.0
tot_disc_pnl = 0.0
tot_hold_tr = 0

for idx, x in enumerate(dual_profitable, 1):
    h = x["holdout"]
    d = x["discovery"]
    tot = h["net_pips"] + d["net_pips"]
    tot_hold_pnl += h["net_pips"]
    tot_disc_pnl += d["net_pips"]
    tot_hold_tr += h["trades"]
    print(f"{idx:4d} | {x['name'][:38]:38s} | {h['net_pips']:>8.1f}p | {h['pf']:>7.2f} | {h['trades']:>7d} | {d['net_pips']:>8.1f}p | {d['pf']:>7.2f} | {tot:>9.1f}p")

print("-" * 110)
print(f"  TOTAL HOLDOUT NET PNL:    {tot_hold_pnl:>+9.1f} pips across {tot_hold_tr} trades")
print(f"  TOTAL DISCOVERY NET PNL:  {tot_disc_pnl:>+9.1f} pips")
print(f"  TOTAL COMBINED NET PNL:   {tot_hold_pnl + tot_disc_pnl:>+9.1f} pips")
print("=" * 110)

