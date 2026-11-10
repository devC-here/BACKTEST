import json

with open('results/continuous_walkforward_audits/v45_continuous_holdout_audit.json') as f:
    d = json.load(f)

winners = [s for s in d['strategies'] if s['net_pips'] > 0]
print(f"Total Profitable Strategies on Continuous Holdout (Weeks 211..300): {len(winners)} / {len(d['strategies'])}")
print("=" * 110)
for s in sorted(winners, key=lambda x: x['net_pips'], reverse=True):
    slot = s['slot']
    fname = s['filename']
    sym = s['symbol']
    tr = s['trades']
    wr = s['win_rate']
    pf = s['profit_factor']
    net = s['net_pips']
    arch = s.get('archetype', '')
    print(f"Slot {slot:02d} | {fname:42s} | {sym:7s} | Tr: {tr:3d} | WR: {wr:4.1f}% | PF: {pf:4.2f} | Net: {net:+7.1f}p | {arch}")

