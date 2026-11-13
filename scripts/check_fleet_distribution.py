import json
from pathlib import Path
from collections import Counter

ROOT_DIR = Path(__file__).resolve().parent.parent
manifest = json.load(open(ROOT_DIR / "certified_production_fleet" / "CERTIFIED_FLEET_MANIFEST.json"))

print(f"Manifest Version: {manifest['manifest_version']}")
print(f"Total Strategies: {manifest['total_strategies']}")
print(f"Holdout Net Pips: {manifest['portfolio_totals']['holdout_net_pips']} pips")
print(f"Holdout Trades:   {manifest['portfolio_totals']['holdout_total_trades']}")
print(f"Holdout Expectancy: {manifest['portfolio_totals']['holdout_expectancy_pips']} pips/trade")

counts = Counter([s['symbol'] for s in manifest['strategies']])
print("\n--- STRATEGY COUNT BY PAIR ---")
for sym, count in counts.most_common():
    print(f"  {sym:<8}: {count} strategies")

