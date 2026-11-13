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

p = Path("archive_deprecated/archive_labs/collapsed_bfill_fleet/strategy_02_gbpjpy_h4_macro_retest.py")
spec = importlib.util.spec_from_file_location("mod", p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
strat_cls = mod.Strategy_02_gbpjpy_h4_macro_retest

symbol = "GBPJPY"
pip_size = 0.01
comm, slip = get_friction_pips(symbol)

hold_weeks = get_complete_split_weeks(symbol, "holdout_test_30")
strat = strat_cls()

hold_trades_list = []
for w in hold_weeks:
    ticks, _ = get_week_dataset(w, f"data/{symbol}", warmup_prev_week_ticks=WARMUP_TICKS)
    if len(ticks) <= WARMUP_TICKS + 500:
        continue
    sig, sl, tp = strat.generate_signals(ticks)
    if not np.any(sig != 0):
        continue
    tr = run_matching_loop(
        ticks, sig, pip_size, sl, tp,
        use_slippage=True, close_at_end=True,
        commission_pips=comm, entry_slippage_pips=slip,
        min_entry_timestamp=ticks["timestamp"][WARMUP_TICKS],
        enforce_subsequent_tick=True
    )
    if len(tr) > 0:
        hold_trades_list.append(tr)

comb = np.concatenate(hold_trades_list) if hold_trades_list else np.empty(0)
stats = compute_statistical_metrics(comb)
print(f"GBPJPY Strat 02 Holdout: Net={stats['net_pips']:+.1f}p | PF={stats['profit_factor']:.2f} | Tr={stats['total_trades']} | Exp={stats['expectancy_pips']:+.2f}p")
