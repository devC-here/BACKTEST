import sys
import importlib.util
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import run_matching_loop
from tick_engine.core.dataloader import get_week_dataset, get_complete_split_weeks
from tick_engine.core.config import PIP_SIZES, get_friction_pips, WARMUP_TICKS
from tick_engine.reporting.run_manifest import compute_statistical_metrics

def eval_candidate(file_path: Path, symbol: str, max_weeks: int = 90):
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
    strat_cls = None
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, type) and hasattr(obj, "generate_signals") and attr != "BaseStrategy":
            strat_cls = obj
            break
            
    if strat_cls is None:
        return None
        
    pip_size = PIP_SIZES.get(symbol, 0.0001)
    comm, slip = get_friction_pips(symbol)
    strat = strat_cls(asset=symbol, symbol=symbol, pip_size=pip_size)
    
    hold_weeks = get_complete_split_weeks(symbol, "holdout_test_30")[:max_weeks]
    disc_weeks = get_complete_split_weeks(symbol, "discovery_70")[:max_weeks]
    
    def run_weeks(weeks):
        trades = []
        for w in weeks:
            try:
                ticks, _ = get_week_dataset(w, f"data/{symbol}", warmup_prev_week_ticks=WARMUP_TICKS)
            except Exception:
                continue
            if len(ticks) <= WARMUP_TICKS + 500:
                continue
            res = strat.generate_signals(ticks)
            if isinstance(res, tuple):
                sig, sl, tp = res[0], res[1], res[2]
            else:
                sig = res
                sl = np.full(len(ticks), getattr(strat, "sl_pips", 20.0))
                tp = np.full(len(ticks), getattr(strat, "tp_pips", 40.0))
                
            if not np.any(sig != 0):
                continue
                
            tr = run_matching_loop(
                ticks=ticks,
                signals=sig,
                pip_size=pip_size,
                sl_pips=sl,
                tp_pips=tp,
                use_slippage=True,
                close_at_end=True,
                commission_pips=comm,
                entry_slippage_pips=slip,
                min_entry_timestamp=ticks['timestamp'][WARMUP_TICKS],
                enforce_subsequent_tick=True
            )
            if len(tr) > 0:
                trades.append(tr)
        return np.concatenate(trades) if trades else np.empty(0)

    h_trades = run_weeks(hold_weeks)
    d_trades = run_weeks(disc_weeks)
    
    h_stats = compute_statistical_metrics(h_trades)
    d_stats = compute_statistical_metrics(d_trades)
    return h_stats, d_stats

if __name__ == "__main__":
    v45 = Path("archive_deprecated/v 4.5")
    test_files = [
        (v45 / "strategy_33_audcad_h4_macro_retest.py", "AUDCAD"),
        (v45 / "strategy_34_audcad_h4_macro_retest.py", "AUDCAD"),
        (v45 / "strategy_49_eurnzd_h4_macro_retest.py", "EURNZD"),
        (v45 / "strategy_50_eurnzd_h4_macro_retest.py", "EURNZD"),
        (v45 / "strategy_53_nzdcad_h4_macro_retest.py", "NZDCAD"),
        (v45 / "strategy_54_nzdcad_h4_macro_retest.py", "NZDCAD"),
        (v45 / "strategy_01_gbpjpy_h4_macro_retest.py", "GBPJPY"),
        (v45 / "strategy_17_usdcad_asian_sweep.py", "USDCAD"),
    ]
    for p, sym in test_files:
        if p.exists():
            print(f"Testing {p.name} on {sym}...")
            res = eval_candidate(p, sym, max_weeks=45)
            if res:
                h, d = res
                print(f"  Holdout (45w): Net: {h['net_pips']:>+6.1f}p | PF: {h['profit_factor']:.2f} | Tr: {h['total_trades']:>2} | Exp: {h['expectancy_pips']:>+5.2f}p")
                print(f"  Disc    (45w): Net: {d['net_pips']:>+6.1f}p | PF: {d['profit_factor']:.2f} | Tr: {d['total_trades']:>2} | Exp: {d['expectancy_pips']:>+5.2f}p")
