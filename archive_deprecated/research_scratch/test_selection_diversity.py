import json
import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.dataloader import get_discovery_weeks, get_holdout_test_weeks
from tick_engine.core.config import PIP_SIZES, get_friction_pips, WARMUP_TICKS
from scripts.search_clean_macro_champions import precompute_week_data, eval_config_on_week
from scripts.promote_clean_champions import calc_overlap

data = json.load(open('results/clean_macro_champions_found.json'))

def get_trades_for_config(symbol, c, weeks):
    comm, slip = get_friction_pips(symbol)
    max_spread = 2.5 if symbol != 'EURNZD' else 5.0
    all_trades = []
    for w in weeks:
        wd = precompute_week_data(symbol, w)
        if wd is None:
            continue
        tr = eval_config_on_week(wd, c['sl'], c['tp'], c['lb'], c['dist'], c['wick'], c['session'], max_spread, comm, slip)
        if len(tr) > 0:
            all_trades.append(tr)
    return np.concatenate(all_trades) if all_trades else np.array([])

# Let's inspect pairwise overlap for AUDCAD top candidates
print("Checking AUDCAD candidates diversity...")
audcad_champs = sorted(data['AUDCAD'], key=lambda x: x['holdout']['net_pips'], reverse=True)
print(f"Total AUDCAD champs: {len(audcad_champs)}")
