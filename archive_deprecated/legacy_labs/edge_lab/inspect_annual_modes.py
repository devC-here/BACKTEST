import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tick_engine.core.engine import BacktestEngine, TRADE_DTYPE
from tick_engine.core.dataloader import get_week_dataset
from tick_engine.core.metrics import compute_metrics
from edge_lab.london_breakout import LondonBreakoutStrategy

engine = BacktestEngine(pip_size=0.0001)

modes_to_test = [
    ("v6_sweep_trend", "V6: Trend-Aligned Sweep Fade"),
    ("v0_naive", "V0: Naive Breakout Baseline"),
    ("v2_candle_close", "V2: Candle Confirmation Breakout")
]

for mode_key, title in modes_to_test:
    strat = LondonBreakoutStrategy(mode=mode_key)
    all_trades_list = []
    
    for w in range(1, 54):
        ticks, _ = get_week_dataset(w)
        trades = engine.run(ticks, strat)
        if len(trades) > 0:
            all_trades_list.append(trades)
            
    all_trades = np.concatenate(all_trades_list) if all_trades_list else np.empty(0, dtype=TRADE_DTYPE)
    m = compute_metrics(all_trades)
    
    print("=" * 70)
    print(f" {title.upper()} (1 FULL YEAR / 53 WEEKS) ")
    print("=" * 70)
    print(f"  Total Trades        : {m['total_trades']}")
    print(f"  Win Rate            : {m['win_rate_pct']}% ({m['win_count']} wins / {m['loss_count']} losses)")
    print(f"  Net Pips            : {m['net_pips']:+.1f} pips")
    print(f"  Gross Profit        : +{m['gross_profit_pips']:.1f} pips")
    print(f"  Gross Loss          : -{m['gross_loss_pips']:.1f} pips")
    print(f"  Profit Factor       : {m['profit_factor']}")
    print(f"  Max Drawdown        : {m['max_drawdown_pips']:.1f} pips")
    print(f"  Average Trade       : {m['avg_trade_pips']:+.2f} pips")
    print(f"  Avg Win / Avg Loss  : +{m['avg_win_pips']:.1f}p / -{m['avg_loss_pips']:.1f}p (Ratio: {m['win_loss_ratio']})")
    print(f"  Long Trades         : {m['long_trades']} ({m['long_win_rate_pct']}% win) -> {m['long_net_pips']:+.1f} pips")
    print(f"  Short Trades        : {m['short_trades']} ({m['short_win_rate_pct']}% win) -> {m['short_net_pips']:+.1f} pips")
    print(f"  Return / Max DD     : {m['return_to_dd']}")
    print()
