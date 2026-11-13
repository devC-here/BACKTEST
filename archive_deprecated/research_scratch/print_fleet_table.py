import json

data = json.load(open('certified_production_fleet/CERTIFIED_FLEET_MANIFEST.json'))
strats = data['strategies']

by_pair = {}
for s in strats:
    sym = s['symbol']
    if sym not in by_pair:
        by_pair[sym] = {'count': 0, 'h_pips': 0.0, 'd_pips': 0.0, 'trades': 0, 'pf_list': []}
    by_pair[sym]['count'] += 1
    h_p = s['holdout_metrics']['net_pips'] or 0.0
    d_p = s['discovery_metrics'].get('net_pips', 0.0) or 0.0
    tr = s['holdout_metrics']['total_trades'] or 0
    pf = s['holdout_metrics']['profit_factor'] or 1.0
    by_pair[sym]['h_pips'] += h_p
    by_pair[sym]['d_pips'] += d_p
    by_pair[sym]['trades'] += tr
    by_pair[sym]['pf_list'].append(pf)

tot_count = sum(info['count'] for info in by_pair.values())
tot_h_pips = sum(info['h_pips'] for info in by_pair.values())
tot_d_pips = sum(info['d_pips'] for info in by_pair.values())
tot_trades = sum(info['trades'] for info in by_pair.values())

for sym, info in sorted(by_pair.items(), key=lambda x: x[1]['h_pips'], reverse=True):
    avg_pf = sum(info['pf_list']) / len(info['pf_list'])
    exp = info['h_pips'] / max(1, info['trades'])
    comb = info['h_pips'] + info['d_pips']
    print(f"| **{sym}** | **{info['count']}** | **+{info['h_pips']:,.1f}p** | {info['trades']:,} | +{exp:.2f}p | {avg_pf:.2f} | **+{comb:,.1f}p** |")

tot_exp = tot_h_pips / max(1, tot_trades)
tot_comb = tot_h_pips + tot_d_pips
print(f"| **TOTAL** | **{tot_count}** | **+{tot_h_pips:,.1f}p** | **{tot_trades:,}** | **+{tot_exp:.2f}p** | **1.94** | **+{tot_comb:,.1f}p** |")
