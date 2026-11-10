#!/usr/bin/env python3
"""
Auditing Rolling Time-to-10% Target for 7-Star Prop Firm Fleet.
Simulates starting the challenge at ANY day of the year across 52 weeks (rolling windows).
"""

from __future__ import annotations
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from test_7_star_prop_fleet import run_week, FLEET


def main():
    print("Auditing Rolling Time to +10% Target for 7-Star Fleet across 52 Weeks...")
    with ProcessPoolExecutor() as executor:
        week_res = list(executor.map(run_week, range(1, 53)))

    all_trades = []
    for r in week_res:
        for k, v in r.items():
            all_trades.extend(v)

    s_trades = sorted(all_trades, key=lambda x: x["exit_time"])
    pips = np.array([t["pips"] for t in s_trades])
    losses = pips[pips < 0]
    avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 8.0

    print(f"Total trades: {len(s_trades)}, Avg Loss: {avg_loss:.2f} pips\n")

    for risk_pct in [0.35, 0.40, 0.50]:
        print("=" * 90)
        print(f"               RISK PER TRADE: {risk_pct:.2f}%  (Target: +10.0%, Capital: $100,000)")
        print("=" * 90)

        # Evaluate rolling starts across all trading days
        start_indices = list(range(0, len(s_trades) - 30, 5))
        times_to_target_days = []
        trades_to_target_count = []
        max_dds = []
        passed_count = 0

        for s_idx in start_indices:
            start_ts = s_trades[s_idx]["entry_time"]
            bal = 100000.0
            peak = bal
            max_dd = 0.0
            reached_target = False

            for k in range(s_idx, len(s_trades)):
                t = s_trades[k]
                r_pnl = (t["pips"] / avg_loss) * (bal * (risk_pct / 100.0))
                bal += r_pnl
                if bal > peak:
                    peak = bal
                dd = (peak - bal) / peak * 100.0
                if dd > max_dd:
                    max_dd = dd

                # Check +10% target ($110,000)
                if bal >= 110000.0:
                    time_days = (t["exit_time"] - start_ts) / (1000.0 * 86400.0)
                    trade_cnt = k - s_idx + 1
                    times_to_target_days.append(time_days)
                    trades_to_target_count.append(trade_cnt)
                    max_dds.append(max_dd)
                    reached_target = True
                    passed_count += 1
                    break

        if times_to_target_days:
            t_days = np.array(times_to_target_days)
            t_cnt = np.array(trades_to_target_count)
            t_dd = np.array(max_dds)

            print(f"Challenges Evaluated:     {len(start_indices)} rolling start dates")
            print(f"Pass Rate:                {passed_count / len(start_indices) * 100:.1f}%")
            print(f"Fastest Time to Pass:     {np.min(t_days):.1f} calendar days (~{np.min(t_days)/7:.1f} weeks) [{np.min(t_cnt)} trades]")
            print(f"Median Time to Pass:      {np.median(t_days):.1f} calendar days (~{np.median(t_days)/30.4:.1f} months) [{int(np.median(t_cnt))} trades]")
            print(f"Average Time to Pass:     {np.mean(t_days):.1f} calendar days (~{np.mean(t_days)/30.4:.1f} months) [{int(np.mean(t_cnt))} trades]")
            print(f"90th Percentile Time:     {np.percentile(t_days, 90):.1f} calendar days (~{np.percentile(t_days, 90)/30.4:.1f} months)")
            print(f"Max DD Encountered:       {np.max(t_dd):.2f}% (Average DD during phase: {np.mean(t_dd):.2f}%)")
        print()


if __name__ == "__main__":
    main()
