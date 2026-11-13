import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent

data = json.load(open(ROOT_DIR / "results" / "new_champions_eurgbp_pivot_slope.json"))
print(f"Total Dual Champions Found: {len(data)}")

# Sort by Holdout PF
by_h_pf = sorted(data, key=lambda x: x['holdout']['pf'], reverse=True)
print("\n--- TOP 5 BY HOLDOUT PROFIT FACTOR ---")
for c in by_h_pf[:5]:
    s = c['setup']
    h = c['holdout']
    d = c['discovery']
    tot_pips = h['net_pips'] + d['net_pips']
    tot_tr = h['trades'] + d['trades']
    print(f"Overlap: {c['overlap_174_pct']}% | {s['desc']}:")
    print(f"   Holdout:   {h['trades']} tr, +{h['net_pips']:.1f}p, PF={h['pf']:.2f}, Exp={h['exp']:+.2f}p, ExTop3=+{h['ex_top3']:.1f}p")
    print(f"   Discovery: {d['trades']} tr, +{d['net_pips']:.1f}p, PF={d['pf']:.2f}, Exp={d['exp']:+.2f}p")
    print(f"   Total 5.75Y: {tot_tr} tr, +{tot_pips:.1f}p, Exp={tot_pips/tot_tr:+.2f}p\n")

# Sort by Discovery PF
by_d_pf = sorted(data, key=lambda x: x['discovery']['pf'], reverse=True)
print("\n--- TOP 5 BY DISCOVERY PROFIT FACTOR ---")
for c in by_d_pf[:5]:
    s = c['setup']
    h = c['holdout']
    d = c['discovery']
    tot_pips = h['net_pips'] + d['net_pips']
    tot_tr = h['trades'] + d['trades']
    print(f"Overlap: {c['overlap_174_pct']}% | {s['desc']}:")
    print(f"   Holdout:   {h['trades']} tr, +{h['net_pips']:.1f}p, PF={h['pf']:.2f}, Exp={h['exp']:+.2f}p, ExTop3=+{h['ex_top3']:.1f}p")
    print(f"   Discovery: {d['trades']} tr, +{d['net_pips']:.1f}p, PF={d['pf']:.2f}, Exp={d['exp']:+.2f}p")
    print(f"   Total 5.75Y: {tot_tr} tr, +{tot_pips:.1f}p, Exp={tot_pips/tot_tr:+.2f}p\n")

# Sort by Total Net Pips
by_tot = sorted(data, key=lambda x: x['holdout']['net_pips'] + x['discovery']['net_pips'], reverse=True)
print("\n--- TOP 5 BY TOTAL NET PIPS ---")
for c in by_tot[:5]:
    s = c['setup']
    h = c['holdout']
    d = c['discovery']
    tot_pips = h['net_pips'] + d['net_pips']
    tot_tr = h['trades'] + d['trades']
    print(f"Overlap: {c['overlap_174_pct']}% | {s['desc']}:")
    print(f"   Holdout:   {h['trades']} tr, +{h['net_pips']:.1f}p, PF={h['pf']:.2f}, Exp={h['exp']:+.2f}p, ExTop3=+{h['ex_top3']:.1f}p")
    print(f"   Discovery: {d['trades']} tr, +{d['net_pips']:.1f}p, PF={d['pf']:.2f}, Exp={d['exp']:+.2f}p")
    print(f"   Total 5.75Y: {tot_tr} tr, +{tot_pips:.1f}p, Exp={tot_pips/tot_tr:+.2f}p\n")

