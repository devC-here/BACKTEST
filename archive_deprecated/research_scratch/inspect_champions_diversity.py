import json
import pandas as pd

data = json.load(open('results/clean_macro_champions_found.json'))

for sym in ['AUDCAD', 'NZDCAD', 'USDCAD', 'EURNZD']:
    champs = data[sym]
    df = pd.DataFrame([{
        'sl': c['sl'],
        'tp': c['tp'],
        'rr': c['tp'] / c['sl'],
        'lb': c['lb'],
        'dist': c['dist'],
        'wick': c['wick'],
        'sm': c['session'],
        'hold_pips': c['holdout']['net_pips'],
        'hold_pf': c['holdout']['profit_factor'],
        'hold_tr': c['holdout']['total_trades'],
        'hold_exp': c['holdout']['expectancy_pips'],
        'disc_pips': c.get('discovery', {}).get('net_pips', 0),
        'disc_pf': c.get('discovery', {}).get('profit_factor', 0)
    } for c in champs])
    
    print(f"\n=================== {sym} ({len(df)} champions) ===================")
    print(f"Sessions available: {df['sm'].unique()}")
    print(f"Lookbacks available: {df['lb'].unique()}")
    print(f"SL/TP available: {df[['sl', 'tp']].drop_duplicates().values.tolist()}")
    print(df.sort_values(by='hold_pips', ascending=False).head(10).to_string())
