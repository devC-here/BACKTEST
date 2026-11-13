import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
FLEET_MANIFEST = ROOT_DIR / "certified_production_fleet" / "CERTIFIED_FLEET_MANIFEST.json"
MANIFEST_DIR = ROOT_DIR / "results" / "certified_strategy_manifests"

data = json.load(open(FLEET_MANIFEST))
strategies = data['strategies']

# Authentic Pip Values for 1.0 Standard Lot ($100k notional)
PIP_VALUES_1LOT = {
    'EURUSD': 10.00,
    'USDJPY': 6.70,
    'USDCAD': 7.25,
    'AUDCAD': 7.25,
    'NZDCAD': 7.25,
    'AUDNZD': 6.00,
    'EURNZD': 6.00,
    'EURGBP': 13.00,
    'GBPCHF': 12.25,
    'GBPJPY': 6.70
}

RISK_PER_TRADE_USD = 350.0  # 0.35% risk on $100,000 account

records = []
for s in strategies:
    strat_name = s['name']
    sym = s['symbol']
    cat = s['category']
    m_file = MANIFEST_DIR / f"{sym}__{strat_name}_run_manifest.json"
    if not m_file.exists():
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

    sl_pips = float(params.get('sl_pips', 20.0))
    tp_pips = float(params.get('tp_pips', sl_pips * 2.0))
    net_pips = float(stats.get('net_pips', 0.0))
    total_trades = int(stats.get('total_trades', 0))
    pf = float(stats.get('profit_factor', 0.0))
    exp_pips = float(stats.get('expectancy_pips', 0.0))
    max_dd_pips = float(stats.get('max_drawdown_pips', 0.0))
    
    # 1. Standard 1.0 Lot Dollar Metrics
    pip_val = PIP_VALUES_1LOT.get(sym, 10.0)
    pnl_1lot_usd = net_pips * pip_val
    exp_1lot_usd = exp_pips * pip_val
    max_dd_1lot_usd = max_dd_pips * pip_val

    # 2. Institutional Risk-Adjusted Sizing ($350 risk per trade / 0.35% on $100k account)
    # Each pip = $350 / sl_pips
    dollar_per_pip_sized = RISK_PER_TRADE_USD / sl_pips
    pnl_sized_usd = net_pips * dollar_per_pip_sized
    exp_sized_usd = exp_pips * dollar_per_pip_sized
    max_dd_sized_usd = max_dd_pips * dollar_per_pip_sized
    
    # Holdout is 90 weeks ≈ 20.7 months
    monthly_pnl_usd = pnl_sized_usd / 20.7
    annual_pnl_usd = (pnl_sized_usd / 90.0) * 52.0

    # Viability Status
    # Viable if: Net Sized PnL > $1,000, Exp Sized > $20/trade, PF >= 1.15
    is_viable = (pnl_sized_usd >= 1000.0 and exp_sized_usd >= 20.0 and pf >= 1.15)
    viability_tier = "Tier 1: High Alpha Deployable" if (pnl_sized_usd >= 3000.0 and pf >= 1.50) else ("Tier 2: Solid Production" if is_viable else "Tier 3: Moderate Edge")

    records.append({
        'name': strat_name,
        'symbol': sym,
        'category': cat,
        'regime': regime,
        'session': session,
        'sl_pips': sl_pips,
        'tp_pips': tp_pips,
        'rr': round(tp_pips / sl_pips, 2),
        'net_pips': round(net_pips, 1),
        'trades': total_trades,
        'pf': round(pf, 2),
        'exp_pips': round(exp_pips, 2),
        'pnl_1lot_usd': round(pnl_1lot_usd, 2),
        'exp_1lot_usd': round(exp_1lot_usd, 2),
        'pnl_sized_usd': round(pnl_sized_usd, 2),
        'exp_sized_usd': round(exp_sized_usd, 2),
        'max_dd_sized_usd': round(max_dd_sized_usd, 2),
        'monthly_pnl_usd': round(monthly_pnl_usd, 2),
        'annual_pnl_usd': round(annual_pnl_usd, 2),
        'viability_tier': viability_tier
    })

df = pd.DataFrame(records)

print(f"\n=========================================================================")
print(f"  TOTAL FLEET DOLLAR ECONOMICS (72 STRATEGIES / 90-WEEK QUARANTINED HOLDOUT)")
print(f"=========================================================================")
tot_pnl_1lot = df['pnl_1lot_usd'].sum()
tot_pnl_sized = df['pnl_sized_usd'].sum()
tot_trades = df['trades'].sum()
avg_exp_sized = tot_pnl_sized / tot_trades

print(f"  Total Holdout Net Pips:              +{df['net_pips'].sum():,.1f} pips")
print(f"  Total Executed Trades:               {tot_trades:,} trades")
print(f"  Total PnL @ 1.0 Standard Lot:        +${tot_pnl_1lot:,.2f}")
print(f"  Total PnL @ 0.35% Risk Sizing:       +${tot_pnl_sized:,.2f} (+{tot_pnl_sized/1000:.1f}% on $100k)")
print(f"  Average Sized Expectancy:            +${avg_exp_sized:.2f} per trade")
print(f"  Annualized Fleet Return:             +${df['annual_pnl_usd'].sum():,.2f} / year")
print(f"  Monthly Fleet Return:                +${df['monthly_pnl_usd'].sum():,.2f} / month")
print(f"=========================================================================")

print(f"\n=== VIABILITY TIERS ===")
print(df['viability_tier'].value_counts())

print(f"\n=== DOLLAR PNL BY MARKET REGIME (0.35% SIZING) ===")
reg_summary = df.groupby('regime').agg({
    'name': 'count',
    'pnl_sized_usd': 'sum',
    'trades': 'sum',
    'monthly_pnl_usd': 'sum'
}).rename(columns={'name': 'Strategies', 'pnl_sized_usd': 'Total Sized USD', 'trades': 'Trades', 'monthly_pnl_usd': 'USD/Month'})
reg_summary['Exp USD/Tr'] = reg_summary['Total Sized USD'] / reg_summary['Trades']
print(reg_summary.to_string())

print(f"\n=== DOLLAR PNL BY TRADING SESSION (0.35% SIZING) ===")
sess_summary = df.groupby('session').agg({
    'name': 'count',
    'pnl_sized_usd': 'sum',
    'trades': 'sum',
    'monthly_pnl_usd': 'sum'
}).rename(columns={'name': 'Strategies', 'pnl_sized_usd': 'Total Sized USD', 'trades': 'Trades', 'monthly_pnl_usd': 'USD/Month'})
sess_summary['Exp USD/Tr'] = sess_summary['Total Sized USD'] / sess_summary['Trades']
print(sess_summary.to_string())

print(f"\n=== DOLLAR PNL BY CURRENCY PAIR (0.35% SIZING) ===")
pair_summary = df.groupby('symbol').agg({
    'name': 'count',
    'pnl_sized_usd': 'sum',
    'trades': 'sum',
    'monthly_pnl_usd': 'sum'
}).rename(columns={'name': 'Strategies', 'pnl_sized_usd': 'Total Sized USD', 'trades': 'Trades', 'monthly_pnl_usd': 'USD/Month'})
pair_summary['Exp USD/Tr'] = pair_summary['Total Sized USD'] / pair_summary['Trades']
print(pair_summary.to_string())

# Save complete JSON
df.to_json(ROOT_DIR / "results" / "strategy_dollar_viability_report.json", orient="records", indent=2)
print("\nSaved complete report to results/strategy_dollar_viability_report.json")

