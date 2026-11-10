import json
from pathlib import Path

with open("v 4.5/FLEET_MANIFEST.json", "r") as f:
    fleet_meta = json.load(f)

with open("results/continuous_walkforward_audits/v45_continuous_holdout_audit.json", "r") as f:
    audit_data = json.load(f)

audit_map = {x["filename"]: x for x in audit_data.get("strategies", [])}

eurgbp_slots = [x for x in fleet_meta if x.get("symbol") == "EURGBP"]

print(f"{'Slot':4s} | {'File':35s} | {'SL':>4s} | {'TP':>4s} | {'RR':>4s} | {'Trades':>6s} | {'Net Pips':>9s} | {'PF':>5s} | {'WR':>6s}")
print("-" * 85)

for s in eurgbp_slots:
    fn = s["filename"]
    res = audit_map.get(fn, {})
    tr = res.get("trades", 0)
    pips = res.get("net_pips", 0.0)
    pf = res.get("profit_factor", 0.0)
    wr = res.get("win_rate_pct", 0.0)
    sl = s.get("sl_pips", 0)
    tp = s.get("tp_pips", 0)
    rr = s.get("rr_ratio", 0)
    print(f"{s['slot']:4d} | {fn:35s} | {sl:4.1f} | {tp:4.1f} | {rr:4.1f} | {tr:6d} | {pips:>8.1f}p | {pf:5.2f} | {wr:5.1f}%")
