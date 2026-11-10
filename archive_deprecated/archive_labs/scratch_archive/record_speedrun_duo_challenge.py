#!/usr/bin/env python3
"""
================================================================================
  RECORD SPEEDRUN DUO (DISC42 + DISC43) PROP CHALLENGE TRADE LOG
================================================================================
Simulates the Precision Duo Speedrun on a $100k account:
  - DISC42: Macro Structure H4 Level + M30 Trigger (72.9% WR, PF 3.97)
  - DISC43: Intraday Precision H1 Level + M5 Trigger (48.3% WR, PF 1.47)
  - Risk per Trade: 0.90% flat ($900 on $100k)
  - Daily Stop: 2.0% ($2,000)
Records every trade: entry price, exit price, banked profit/loss, lot size, duration.
================================================================================
"""

from __future__ import annotations
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

sys.dont_write_bytecode = True

ROOT_DIR = Path(__file__).resolve().parent.parent
MASTER_FILE = ROOT_DIR / "results" / "master_7_prop_trades_5y.json"
RESULTS_DIR = ROOT_DIR / "results"

with open(MASTER_FILE, 'r') as f:
    all_trades = json.load(f)

strats = {"DISC42_H4_M30", "DISC43_H1_M5"}
trades = [t for t in all_trades if t['strategy'] in strats]
trades.sort(key=lambda x: x['entry_time'])

strat_losses = {}
for s in strats:
    sl = [abs(t['pips']) for t in trades if t['strategy'] == s and t['pips'] < 0]
    strat_losses[s] = float(np.mean(sl)) if sl else 8.5

INITIAL_CAPITAL = 100000.0
P1_TARGET = 8000.0
P2_TARGET = 5000.0

bal = INITIAL_CAPITAL
peak = bal
phase = 1
trade_records = []
current_day = None
daily_start_bal = bal
day_loss = 0.0

for i, tr in enumerate(trades):
    dt_entry = pd.to_datetime(tr['entry_time'], unit='ms')
    dt_exit = pd.to_datetime(tr['exit_time'], unit='ms')
    t_day = dt_entry.date()

    if t_day != current_day:
        current_day = t_day
        daily_start_bal = bal
        day_loss = 0.0

    if (day_loss / daily_start_bal) * 100.0 >= 2.0:
        continue

    avg_loss = strat_losses[tr['strategy']]
    r_mult = tr['pips'] / avg_loss
    dollar_risk = bal * 0.0090  # 0.90% risk
    pnl = r_mult * dollar_risk
    bal += pnl

    if pnl < 0:
        day_loss += abs(pnl)

    if bal > peak:
        peak = bal
    dd_pct = (peak - bal) / peak * 100.0

    lots = round(dollar_risk / (avg_loss * 10.0), 2)
    dur_min = int((tr['exit_time'] - tr['entry_time']) / 60000)
    dur_str = f"{dur_min//60}h {dur_min%60:02d}m" if dur_min >= 60 else f"{dur_min}m"
    reasons = {1: "Stop Loss", 2: "Take Profit", 3: "Signal/Time Exit"}

    rec = {
        'trade_id': len(trade_records) + 1,
        'phase': f"Phase {phase}",
        'strategy': tr['strategy'],
        'direction': "BUY" if tr['direction'] == 1 else "SELL",
        'entry_time': dt_entry.strftime('%Y-%m-%d %H:%M:%S'),
        'entry_price': round(tr['entry_price'], 5),
        'exit_time': dt_exit.strftime('%Y-%m-%d %H:%M:%S'),
        'exit_price': round(tr['exit_price'], 5),
        'duration': dur_str,
        'exit_reason': reasons.get(tr['exit_reason'], "Closed"),
        'pips': round(tr['pips'], 1),
        'lot_size': lots,
        'banked_pnl': round(pnl, 2),
        'balance': round(bal, 2),
        'drawdown_pct': round(dd_pct, 2)
    }
    trade_records.append(rec)

    if phase == 1 and bal >= INITIAL_CAPITAL + P1_TARGET:
        phase = 2
        bal = INITIAL_CAPITAL
        peak = bal
        day_loss = 0.0
    elif phase == 2 and bal >= INITIAL_CAPITAL + P2_TARGET:
        break

df_out = pd.DataFrame(trade_records)
csv_file = RESULTS_DIR / "speedrun_duo_challenge_trade_log.csv"
json_file = RESULTS_DIR / "speedrun_duo_challenge_trade_log.json"
df_out.to_csv(csv_file, index=False)
with open(json_file, 'w') as f:
    json.dump(trade_records, f, indent=2)

print(f"Speedrun Duo Challenge finished in {len(trade_records)} trades!")
p1_len = len([r for r in trade_records if r['phase'] == 'Phase 1'])
p2_len = len([r for r in trade_records if r['phase'] == 'Phase 2'])
print(f"Phase 1: {p1_len} trades | Phase 2: {p2_len} trades")
print(f"Saved to {csv_file}")
