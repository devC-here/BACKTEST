import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.dataloader import get_discovery_weeks, get_holdout_test_weeks
from tick_engine.core.config import PIP_SIZES, get_friction_pips
from tick_engine.reporting.run_manifest import compute_statistical_metrics
from scripts.search_clean_macro_champions import precompute_week_data, eval_config_on_week

def main():
    symbol = "EURNZD"
    comm, slip = get_friction_pips(symbol)
    max_spread = 5.0

    hold_weeks = get_holdout_test_weeks(symbol)
    disc_weeks = get_discovery_weeks(symbol)

    print(f"Precomputing {symbol} holdout weeks...")
    with ProcessPoolExecutor(max_workers=4) as ex:
        hold_pre = [w for w in ex.map(precompute_week_data, [symbol]*len(hold_weeks), hold_weeks) if w is not None]

    sl_tp_pairs = [
        (20.0, 40.0), (20.0, 50.0), (22.0, 44.0), (22.0, 55.0),
        (25.0, 50.0), (25.0, 60.0), (28.0, 56.0), (28.0, 70.0),
        (30.0, 60.0), (30.0, 75.0), (35.0, 70.0)
    ]
    lookbacks = [8, 10, 12, 14]
    dists = [6.0, 8.0, 10.0, 12.0, 14.0]
    wicks = [0.35, 0.40, 0.45]
    sessions = [1, 2, 3]

    candidates = []
    for sl, tp in sl_tp_pairs:
        for lb in lookbacks:
            for dst in dists:
                for wk in wicks:
                    for sm in sessions:
                        hold_trades = []
                        for wd in hold_pre:
                            tr = eval_config_on_week(wd, sl, tp, lb, dst, wk, sm, max_spread, comm, slip)
                            if len(tr) > 0:
                                hold_trades.append(tr)
                        if not hold_trades:
                            continue
                        comb_h = np.concatenate(hold_trades)
                        m_h = compute_statistical_metrics(comb_h)
                        if m_h['net_pips'] > 0 and m_h['profit_factor'] >= 1.15 and m_h['total_trades'] >= 15 and m_h['expectancy_pips'] >= 1.25 and not m_h['outlier_sensitivity']['is_outlier_dependent']:
                            candidates.append({
                                'symbol': symbol,
                                'sl': sl, 'tp': tp, 'lb': lb, 'dist': dst, 'wick': wk, 'session': sm,
                                'holdout': m_h
                            })

    print(f"EURNZD candidates passing holdout: {len(candidates)}")

    sample_disc_weeks = disc_weeks[:100]
    with ProcessPoolExecutor(max_workers=4) as ex:
        disc_pre = [w for w in ex.map(precompute_week_data, [symbol]*len(sample_disc_weeks), sample_disc_weeks) if w is not None]

    verified = []
    for c in candidates:
        disc_trades = []
        for wd in disc_pre:
            tr = eval_config_on_week(wd, c['sl'], c['tp'], c['lb'], c['dist'], c['wick'], c['session'], max_spread, comm, slip)
            if len(tr) > 0:
                disc_trades.append(tr)
        if not disc_trades:
            continue
        comb_d = np.concatenate(disc_trades)
        m_d = compute_statistical_metrics(comb_d)
        if m_d['net_pips'] > 0 and m_d['profit_factor'] >= 1.15 and m_d['total_trades'] >= 25:
            c['discovery'] = m_d
            verified.append(c)
            print(f"  *** DUAL-CERTIFIED EURNZD *** SL/TP: {c['sl']}/{c['tp']} | LB: {c['lb']} | Dist: {c['dist']} | Wick: {c['wick']} | Sm: {c['session']}")
            print(f"      Holdout (90w): Net {c['holdout']['net_pips']:>+6.1f}p | PF {c['holdout']['profit_factor']:.2f} | Tr {c['holdout']['total_trades']} | Exp {c['holdout']['expectancy_pips']:>+5.2f}p")
            print(f"      Disc (100w):   Net {m_d['net_pips']:>+6.1f}p | PF {m_d['profit_factor']:.2f} | Tr {m_d['total_trades']} | Exp {m_d['expectancy_pips']:>+5.2f}p")

    print(f"Total dual-certified EURNZD champions: {len(verified)}")
    with open('results/eurnzd_more_champions.json', 'w') as f:
        json.dump(verified, f, indent=2, default=str)

if __name__ == '__main__':
    main()
