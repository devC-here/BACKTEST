import json

with open("results/continuous_walkforward_audits/v45_continuous_holdout_audit.json", "r") as f:
    d = json.load(f)

print(f"{'Slot':4s} | {'Filename':40s} | {'Archetype':30s} | {'Trades':>6s} | {'Net Pips':>9s} | {'PF':>5s}")
print("-" * 105)

for x in d['strategies']:
    if x.get('symbol') == 'GBPJPY':
        print(f"{x['slot']:4d} | {x['filename'][:40]:40s} | {x['archetype'][:30]:30s} | {x['trades']:6d} | {x['net_pips']:>8.1f}p | {x['profit_factor']:5.2f}")

