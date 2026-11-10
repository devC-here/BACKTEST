import json
from pathlib import Path

p = Path("results/continuous_walkforward_audits/v45_continuous_holdout_audit.json")
with open(p, "r", encoding="utf-8") as f:
    d = json.load(f)

for target_sym in ["NZDCAD", "GBPJPY", "GBPCHF", "AUDCAD", "AUDNZD", "EURNZD", "USDJPY"]:
    print("=" * 80)
    print(f"SYMBOL: {target_sym}")
    strats = [s for s in d["strategies"] if s.get("symbol") == target_sym]
    for s in strats[:5]:
        slot = s.get("slot")
        net = s.get("net_pips")
        pf = s.get("profit_factor", 0)
        tr = s.get("trades")
        wr = s.get("win_rate", 0)
        name = s.get("file", s.get("name", "unk"))
        print(f"  Slot {slot:02d}: Net={net:>+7.1f}p | PF={pf:>4.2f} | Trades={tr:>3} | WR={wr:>4.1f}% | {name}")

