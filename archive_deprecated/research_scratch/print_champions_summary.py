import json

data = json.load(open('results/clean_macro_champions_found.json'))
for sym, champs in data.items():
    print(f"=== {sym}: {len(champs)} champions ===")
    sorted_champs = sorted(champs, key=lambda x: x['holdout']['net_pips'], reverse=True)
    for c in sorted_champs[:5]:
        h = c['holdout']
        d = c.get('discovery', {})
        print(f"  SL/TP {c['sl']}/{c['tp']} | LB {c['lb']} | Dist {c['dist']} | Wk {c['wick']} | Sm {c['session']}")
        print(f"     Holdout: Net {h['net_pips']:>+6.1f}p | PF {h['profit_factor']:.2f} | Tr {h['total_trades']} | Exp {h['expectancy_pips']:>+5.2f}p")
        if d:
            print(f"     Disc:    Net {d.get('net_pips', 0):>+6.1f}p | PF {d.get('profit_factor', 0):.2f} | Tr {d.get('total_trades', 0)} | Exp {d.get('expectancy_pips', 0):>+5.2f}p")
