import json

with open('certified_production_fleet/CERTIFIED_FLEET_MANIFEST.json') as f:
    data = json.load(f)

print(f"{'SYMBOL':<8} {'NAME':<50} {'HOLDOUT PF':<12} {'HOLDOUT PIPS':<14} {'DISC PF':<10}")
print("-" * 96)
for s in data['strategies']:
    if s['symbol'] in ['GBPCHF', 'AUDCAD', 'NZDCAD', 'EURGBP']:
        h_pf = s['holdout_metrics'].get('profit_factor', 0.0)
        h_pips = s['holdout_metrics'].get('net_pips', 0.0)
        d_pf = s['discovery_metrics'].get('profit_factor', 0.0) if s.get('discovery_metrics') else 0.0
        print(f"{s['symbol']:<8} {s['name']:<50} {h_pf:<12.2f} {h_pips:<14.1f} {d_pf:<10.2f}")

