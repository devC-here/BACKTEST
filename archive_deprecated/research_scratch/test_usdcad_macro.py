import sys
from pathlib import Path
import numpy as np
import pandas as pd
import importlib.util

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset, get_complete_split_weeks
from tick_engine.core.config import PIP_SIZES, get_friction_pips, WARMUP_TICKS
from tick_engine.reporting.run_manifest import compute_statistical_metrics

symbol = "USDCAD"
pip_size = PIP_SIZES[symbol]
comm, slip = get_friction_pips(symbol)

hold_weeks = get_complete_split_weeks(symbol, "holdout_test_30")
disc_weeks = get_complete_split_weeks(symbol, "discovery_70")

for strat_name in ["disc05_usdcad_h4_macro_structure_retest_elite", "disc06_usdcad_h4_macro_precision_retest_elite"]:
    p = Path(f"archive_deprecated/strategies/usdcad_fleet/{strat_name}/strategy.py")
    spec = importlib.util.spec_from_file_location("mod", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
    strat_cls = None
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, type) and hasattr(obj, "generate_signals") and attr != "BaseStrategy":
            strat_cls = obj
            break

    strat = strat_cls()

    # Holdout
    h_trades_list = []
    for w in hold_weeks:
        ticks, _ = get_week_dataset(w, f"data/{symbol}", warmup_prev_week_ticks=WARMUP_TICKS)
        if len(ticks) <= WARMUP_TICKS + 500:
            continue
        sig, sl, tp = strat.generate_signals(ticks)
        if not np.any(sig != 0):
            continue
        tr = run_matching_loop(ticks, sig, pip_size, sl, tp, use_slippage=True, close_at_end=True, commission_pips=comm, entry_slippage_pips=slip, min_entry_timestamp=ticks["timestamp"][WARMUP_TICKS], enforce_subsequent_tick=True)
        if len(tr) > 0:
            h_trades_list.append(tr)

    # Discovery
    d_trades_list = []
    for w in disc_weeks:
        ticks, _ = get_week_dataset(w, f"data/{symbol}", warmup_prev_week_ticks=WARMUP_TICKS)
        if len(ticks) <= WARMUP_TICKS + 500:
            continue
        sig, sl, tp = strat.generate_signals(ticks)
        if not np.any(sig != 0):
            continue
        tr = run_matching_loop(ticks, sig, pip_size, sl, tp, use_slippage=True, close_at_end=True, commission_pips=comm, entry_slippage_pips=slip, min_entry_timestamp=ticks["timestamp"][WARMUP_TICKS], enforce_subsequent_tick=True)
        if len(tr) > 0:
            d_trades_list.append(tr)

    comb_h = np.concatenate(h_trades_list) if h_trades_list else np.empty(0)
    comb_d = np.concatenate(d_trades_list) if d_trades_list else np.empty(0)
    
    h_stats = compute_statistical_metrics(comb_h)
    d_stats = compute_statistical_metrics(comb_d)

    print(f"\n{strat_name}:")
    print(f"  Holdout (90w):   Net: {h_stats['net_pips']:>+6.1f}p | PF: {h_stats['profit_factor']:.2f} | Tr: {h_stats['total_trades']} | Exp: {h_stats['expectancy_pips']:+.2f}p")
    print(f"  Discovery(210w): Net: {d_stats['net_pips']:>+6.1f}p | PF: {d_stats['profit_factor']:.2f} | Tr: {d_stats['total_trades']} | Exp: {d_stats['expectancy_pips']:+.2f}p")
