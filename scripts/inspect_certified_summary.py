import json

with open("results/continuous_walkforward_audits/certified_survivors_fleet_summary.json", "r") as f:
    fleet = json.load(f)

print(f"{'Strategy File':38s} | {'Disc PnL':>9s} | {'Hold PnL':>9s} | {'Hold PF':>7s} | {'Trades':>6s} | {'Hold WR':>7s} | {'Status'}")
print("-" * 95)
total_hold_pnl = 0.0
total_hold_trades = 0

for x in fleet:
    disc_pnl = x.get("discovery", {}).get("net_pips", 0.0)
    hold_pnl = x.get("holdout", {}).get("net_pips", 0.0)
    hold_pf = x.get("holdout", {}).get("profit_factor", 0.0)
    hold_tr = x.get("holdout", {}).get("total_trades", 0)
    hold_wr = x.get("holdout", {}).get("win_rate_pct", 0.0)
    status = x.get("status", "UNKNOWN")
    print(f"{x['file'][:38]:38s} | {disc_pnl:>8.1f}p | {hold_pnl:>8.1f}p | {hold_pf:>7.2f} | {hold_tr:>6d} | {hold_wr:>6.1f}% | {status}")
    if status == "CERTIFIED_SURVIVOR":
        total_hold_pnl += hold_pnl
        total_hold_trades += hold_tr

print("-" * 95)
print(f"Total Certified Holdout PnL: {total_hold_pnl:+.1f} pips across {total_hold_trades} trades")

