import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
FLEET_MANIFEST = ROOT_DIR / "certified_production_fleet" / "CERTIFIED_FLEET_MANIFEST.json"
MANIFEST_DIR = ROOT_DIR / "results" / "certified_strategy_manifests"

data = json.load(open(FLEET_MANIFEST))
strategies = data['strategies']
print(f"Total strategies: {len(strategies)}")

records = []
for s in strategies:
    strat_name = s['name']
    sym = s['symbol']
    cat = s['category']
    m_file = MANIFEST_DIR / f"{sym}__{strat_name}_run_manifest.json"
    if not m_file.exists():
        # Fallback
        m_file = MANIFEST_DIR / f"{sym.lower()}_{strat_name}_run_manifest.json"
    
    with open(m_file) as f:
        m = json.load(f)
    
    stats = m['statistical_metrics']
    custom = m.get('custom_metadata', {})
    params = custom.get('parameters', {})
    
    # Infer regime from name and archetype
    name_lower = strat_name.lower()
    regime = "Trend / Momentum"
    if "retest" in name_lower or "macro" in name_lower or "structure" in name_lower:
        regime = "Structural Retest & S/R"
    elif "vwap" in name_lower or "pivot" in name_lower or "bounce" in name_lower or "camarilla" in name_lower:
        regime = "Mean Reversion / Range"
    elif "sweep" in name_lower or "fade" in name_lower:
        regime = "Liquidity Sweep & Stop Hunt"
    elif "surge" in name_lower or "expansion" in name_lower or "squeeze" in name_lower or "orb" in name_lower or "breakout" in name_lower:
        regime = "Volatility Expansion & Breakout"
    
    # Session inference
    session = "24h Continuous / Multi-Session"
    if "asian" in name_lower:
        session = "Asian Session (21:00-07:00 UTC)"
    elif "london" in name_lower and "ny" in name_lower:
        session = "London/NY Overlap (12:00-16:00 UTC)"
    elif "london" in name_lower:
        session = "London Morning (06:00-12:00 UTC)"
    elif "ny" in name_lower:
        session = "New York Afternoon (16:00-21:00 UTC)"
    else:
        sm = params.get('session_mode')
        if sm == 1:
            session = "Asian + London (21:00-16:00 UTC)"
        elif sm == 2:
            session = "London + NY (06:00-18:00 UTC)"
        elif sm == 3:
            session = "24h Continuous Macro"

    records.append({
        'name': strat_name,
        'symbol': sym,
        'category': cat,
        'regime': regime,
        'session': session,
        'holdout_pips': stats.get('net_pips', 0.0),
        'holdout_trades': stats.get('total_trades', 0),
        'holdout_pf': stats.get('profit_factor', 0.0),
        'holdout_exp_pips': stats.get('expectancy_pips', 0.0),
        'sl_pips': params.get('sl_pips', 20.0),
        'max_dd_pips': stats.get('max_drawdown_pips', 0.0)
    })

df = pd.DataFrame(records)
print("\n=== REGIME COVERAGE ===")
print(df['regime'].value_counts())

print("\n=== SESSION COVERAGE ===")
print(df['session'].value_counts())
