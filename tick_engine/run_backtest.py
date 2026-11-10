#!/usr/bin/env python3
"""
CLI Runner and Verification Benchmark Harness for Fast Minimal Tick-Level Strategy Tester.
Supports 1-by-1 weekly batch processing across 1 year (52/53 weeks) with real-time progress,
annual scorecard aggregation, CSV export, and complete Top 15 Fleet parallel evaluation.
"""

from __future__ import annotations
import sys
import os
import time
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd

# Ensure root directory and tick_engine are on sys.path
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tick_engine.core.engine import (
    TICK_DTYPE,
    TRADE_DTYPE,
    BacktestEngine,
    run_matching_loop
)
from tick_engine.core.dataloader import (
    load_ticks,
    csv_to_npy,
    batch_convert_csv_to_npy,
    load_manifest,
    get_week_dataset
)
from tick_engine.core.metrics import (
    compute_metrics,
    print_scorecard,
    print_weekly_progress_header,
    print_weekly_progress_row,
    compute_annual_metrics,
    print_annual_scorecard
)
from tick_engine.strategies import (
    get_strategy,
    list_strategies,
    EMACrossoverStrategy,
    Top15Strategy,
    TOP_15_FLEET,
    build_rates_by_timeframe
)


def parse_weeks_string(weeks_arg: str, max_weeks: int = 300) -> List[int]:
    """
    Parses strings like '1-156', '157-262', '1,2,5', '1-10,15-20', 'all' into a sorted list of week ints.
    """
    if weeks_arg.strip().lower() == "all":
        return list(range(1, max_weeks + 1))

    weeks = set()
    parts = [p.strip() for p in weeks_arg.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            start_s, end_s = p.split("-", 1)
            start, end = int(start_s), int(end_s)
            for w in range(min(start, end), max(start, end) + 1):
                if 1 <= w <= max_weeks:
                    weeks.add(w)
        else:
            w = int(p)
            if 1 <= w <= max_weeks:
                weeks.add(w)

    return sorted(list(weeks))



def export_weekly_csv(weekly_records: list[dict], annual_m: dict, strat_name: str, csv_path: Path) -> None:
    """
    Exports week-by-week audit table and annual totals to a CSV file.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "Week,StartDate,EndDate,Ticks,Trades,WinRatePct,NetPips,MaxDDPips,ProfitFactor,AvgTradePips,SimulationSeconds\n"
    ]

    for r in weekly_records:
        w = r['week']
        meta = r['meta']
        m = r['metrics']
        elapsed = r['elapsed']
        ticks = meta.get('tick_count', 0)
        s_date = meta.get('start_date', '')
        e_date = meta.get('end_date', '')
        pf = m['profit_factor']
        pf_val = pf if isinstance(pf, (int, float)) else ""

        line = (
            f"{w},{s_date},{e_date},{ticks},{m['total_trades']},{m['win_rate_pct']},"
            f"{m['net_pips']},{m['max_drawdown_pips']},{pf_val},{m['avg_trade_pips']},{elapsed:.4f}\n"
        )
        lines.append(line)

    with open(csv_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"[+] Weekly performance matrix exported to: {csv_path}")


def run_top15_fleet(
    weeks: List[int],
    raw_path: Path,
    proc_path: Path,
    enforce_hours: bool = True,
    pip_size: float = 0.0001,
    export_summary_csv: Optional[Path] = None,
    export_weekly_csv: Optional[Path] = None
) -> None:
    """
    Executes all 15 institutional strategies across authentic tick data week-by-week.
    Produces comprehensive side-by-side scorecard and portfolio metrics.
    """
    total_weeks = len(weeks)
    print("\n" + "=" * 95)
    print("      TOP 15 HIGH-CONVICTION STRATEGY FLEET - TICK ENGINE BACKTEST      ".center(95))
    print(f"      Execution Mode: 1-by-1 Weekly Batch ({total_weeks} Weeks) | Matching: Tick-by-Tick Bid/Ask      ".center(95))
    print(f"      Spread Friction: 0.8 pips | Accounting: Pip-Denominated | Hours Filter: {'ON' if enforce_hours else 'OFF'}      ".center(95))
    print("=" * 95 + "\n")

    engine = BacktestEngine(pip_size=pip_size)

    # Initialize strategy state containers
    fleet_data: Dict[str, Dict[str, Any]] = {}
    for item in TOP_15_FLEET:
        code = item["code"]
        strat = Top15Strategy(strategy_id=item["id"], enforce_hours=enforce_hours, spread_pips=0.8)
        fleet_data[code] = {
            "info": item,
            "strategy": strat,
            "all_trades": [],
            "weekly_metrics": []
        }

    total_ticks_eval = 0
    t_fleet_start = time.perf_counter()

    print(f"{'Week':<6} | {'Ticks':<9} | {'Sim(s)':<6} | {'Fleet Trades':<12} | {'Fleet Net Pips':<14} | {'Top Performer This Week':<32}")
    print("-" * 95)

    for idx, w in enumerate(weeks, start=1):
        t_w_start = time.perf_counter()
        ticks, meta = get_week_dataset(w, raw_dir=raw_path, processed_dir=proc_path, auto_convert=True)
        n_ticks = len(ticks)
        total_ticks_eval += n_ticks

        # Warm up candle resampling once for all 15 strategies
        _ = build_rates_by_timeframe(ticks)

        week_trades_count = 0
        week_net_pips = 0.0
        best_strat_code = "-"
        best_strat_pips = -999999.0

        for code, data in fleet_data.items():
            strat = data["strategy"]
            trades = engine.run(ticks, strat)
            m = compute_metrics(trades)

            data["all_trades"].append(trades)
            data["weekly_metrics"].append({
                "week": w,
                "ticks": n_ticks,
                "metrics": m
            })

            t_count = m["total_trades"]
            n_pips = m["net_pips"]
            week_trades_count += t_count
            week_net_pips += n_pips

            if n_pips > best_strat_pips and t_count > 0:
                best_strat_pips = n_pips
                best_strat_code = f"{code} ({n_pips:+.1f}p)"

        elapsed_w = time.perf_counter() - t_w_start
        best_desc = best_strat_code if best_strat_code != "-" else "No Trades"
        print(f"W{w:02d}    | {n_ticks:>9,d} | {elapsed_w:>6.2f} | {week_trades_count:>12d} | {week_net_pips:>+13.1f}p | {best_desc:<32}")

    t_fleet_total = time.perf_counter() - t_fleet_start
    throughput = total_ticks_eval / t_fleet_total if t_fleet_total > 0 else 0

    print("-" * 95)
    print(f"[*] Evaluated {total_ticks_eval:,} ticks across {total_weeks} weeks in {t_fleet_total:.2f}s ({throughput:,.0f} ticks/sec overall)")

    # Aggregate full metrics for all 15 strategies
    summary_rows = []
    portfolio_trades_list = []

    for code, data in fleet_data.items():
        info = data["info"]
        valid_trades = [t for t in data["all_trades"] if len(t) > 0]
        if valid_trades:
            strat_trades = np.concatenate(valid_trades)
            portfolio_trades_list.append(strat_trades)
        else:
            strat_trades = np.empty(0, dtype=TRADE_DTYPE)

        m = compute_metrics(strat_trades)
        summary_rows.append({
            "id": info["id"],
            "code": code,
            "name": info["name"],
            "tf": info["tf"],
            "trades": m["total_trades"],
            "win_rate": m["win_rate_pct"],
            "gross_profit": m["gross_profit_pips"],
            "gross_loss": m["gross_loss_pips"],
            "net_pips": m["net_pips"],
            "profit_factor": m["profit_factor"],
            "max_dd": m["max_drawdown_pips"],
            "avg_trade": m["avg_trade_pips"],
            "long_trades": m["long_trades"],
            "short_trades": m["short_trades"],
            "long_win_rate": m["long_win_rate_pct"],
            "short_win_rate": m["short_win_rate_pct"]
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary = df_summary.sort_values(by="net_pips", ascending=False).reset_index(drop=True)

    # Print Grand Top 15 Comparison Scorecard
    print("\n" + "=" * 115)
    print(f"              TOP 15 STRATEGY FLEET - OVERALL TICK PERFORMANCE SCORECARD ({total_weeks} WEEKS)              ".center(115))
    print("=" * 115)
    print(f"{'Rank':<4} | {'Code':<5} | {'TF':<3} | {'Strategy Name':<38} | {'Trades':<6} | {'WinRate':<7} | {'Net Pips':<10} | {'Max DD':<8} | {'PF':<6} | {'Avg Trade':<9}")
    print("-" * 115)

    for rank, row in df_summary.iterrows():
        pf_val = row['profit_factor']
        if isinstance(pf_val, (int, float)):
            pf_str = f"{pf_val:.2f}" if pf_val < 999 else "Inf"
        elif isinstance(pf_val, str):
            pf_str = pf_val
        else:
            pf_str = "-"
        print(f"{rank+1:>3d}  | {row['code']:<5} | {row['tf']:<3} | {row['name']:<38} | {row['trades']:>6d} | {row['win_rate']:>6.1f}% | {row['net_pips']:>+9.1f}p | {row['max_dd']:>7.1f}p | {pf_str:>6} | {row['avg_trade']:>+8.2f}p")

    print("=" * 115)

    # Portfolio combined performance
    if portfolio_trades_list:
        all_port_trades = np.concatenate(portfolio_trades_list)
        port_m = compute_metrics(all_port_trades)
        port_pf = f"{port_m['profit_factor']:.2f}" if isinstance(port_m['profit_factor'], (int, float)) and port_m['profit_factor'] < 999 else "-"
        print(f"FLEET PORTFOLIO TOTALS:")
        print(f"  - Total Fleet Trades  : {port_m['total_trades']:,}")
        print(f"  - Combined Net Pips   : {port_m['net_pips']:+.1f} pips")
        print(f"  - Portfolio Win Rate  : {port_m['win_rate_pct']:.1f}%")
        print(f"  - Portfolio Max DD    : {port_m['max_drawdown_pips']:.1f} pips")
        print(f"  - Portfolio PF        : {port_pf}")
        print("=" * 115 + "\n")

    # Export CSVs
    if export_summary_csv:
        export_summary_csv = Path(export_summary_csv)
        export_summary_csv.parent.mkdir(parents=True, exist_ok=True)
        df_summary.to_csv(export_summary_csv, index=False)
        print(f"[+] Exported fleet summary scorecard to: {export_summary_csv}")

    if export_weekly_csv:
        export_weekly_csv = Path(export_weekly_csv)
        export_weekly_csv.parent.mkdir(parents=True, exist_ok=True)
        matrix_rows = []
        for code, data in fleet_data.items():
            for wm in data["weekly_metrics"]:
                matrix_rows.append({
                    "week": wm["week"],
                    "strategy_code": code,
                    "strategy_name": data["info"]["name"],
                    "timeframe": data["info"]["tf"],
                    "ticks": wm["ticks"],
                    "trades": wm["metrics"]["total_trades"],
                    "win_rate_pct": wm["metrics"]["win_rate_pct"],
                    "net_pips": wm["metrics"]["net_pips"],
                    "max_drawdown_pips": wm["metrics"]["max_drawdown_pips"],
                    "profit_factor": wm["metrics"]["profit_factor"],
                    "avg_trade_pips": wm["metrics"]["avg_trade_pips"]
                })
        df_matrix = pd.DataFrame(matrix_rows)
        df_matrix.to_csv(export_weekly_csv, index=False)
        print(f"[+] Exported weekly matrix breakdown to: {export_weekly_csv}")


def run_1m_benchmark(target_seconds: float = 1.5) -> bool:
    """
    Executes a 1,000,000 tick speed benchmark to verify simulation throughput.
    Target: Under 1.5 seconds per million ticks.
    """
    print("\n" + "=" * 65)
    print("      1,000,000 TICK ENGINE SPEED BENCHMARK HARNESS      ".center(65))
    print("=" * 65)

    n_ticks = 1_000_000
    print(f"[*] Synthesizing {n_ticks:,} realistic EURUSD tick stream...")

    np.random.seed(42)
    timestamps = np.arange(1_700_000_000_000, 1_700_000_000_000 + n_ticks * 100, 100, dtype=np.int64)

    steps = np.random.choice([-0.00005, 0.0, 0.00005], size=n_ticks, p=[0.4, 0.2, 0.4])
    mid_prices = 1.0850 + np.cumsum(steps)
    half_spread = 0.00005  # 1 pip spread
    bids = mid_prices - half_spread
    asks = mid_prices + half_spread

    ticks = np.empty(n_ticks, dtype=TICK_DTYPE)
    ticks['timestamp'] = timestamps
    ticks['bid'] = bids
    ticks['ask'] = asks

    strat = EMACrossoverStrategy(fast_period=100, slow_period=400, sl_pips=15.0, tp_pips=30.0)
    engine = BacktestEngine(pip_size=0.0001)

    print("[*] JIT Warmup (1,000 ticks)...")
    _ = engine.run(ticks[:1000], strat)

    print(f"[*] Running full benchmark across {n_ticks:,} ticks...")
    t_start = time.perf_counter()
    trades = engine.run(ticks, strat)
    t_end = time.perf_counter()

    elapsed = t_end - t_start
    throughput = n_ticks / elapsed

    print("-" * 65)
    print(f"  Total Ticks Evaluated : {n_ticks:,}")
    print(f"  Simulation Elapsed    : {elapsed:.4f} seconds")
    print(f"  Throughput Speed      : {throughput:,.0f} ticks / sec")
    print(f"  Trades Executed       : {len(trades):,}")
    print(f"  Specification Target  : < {target_seconds:.2f} seconds per million ticks")

    passed = elapsed < target_seconds
    if passed:
        speedup = target_seconds / elapsed
        print(f"  Benchmark Status      : \033[92mPASSED\033[0m ({speedup:.1f}x faster than requirement!)")
    else:
        print(f"  Benchmark Status      : \033[91mFAILED\033[0m (Exceeded {target_seconds}s target)")
    print("=" * 65 + "\n")

    metrics = compute_metrics(trades)
    print_scorecard(metrics, title="BENCHMARK SIMULATION SCORECARD")
    return passed


def run_unit_tests() -> bool:
    """
    Runs the unit test suite verifying order fills, SL/TP rules, and metrics.
    """
    import unittest
    from tick_engine.tests.test_engine import TestTickEngineExecution

    suite = unittest.TestLoader().loadTestsFromTestCase(TestTickEngineExecution)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(
        description="Fast Minimal Tick-Level Strategy Tester - 52-Week Batch Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Mode selection
    parser.add_argument("--top15", action="store_true", help="Run full Top 15 Strategy Fleet side-by-side on authentic tick stream")
    parser.add_argument("--benchmark", action="store_true", help="Run 1,000,000 tick performance benchmark")
    parser.add_argument("--test", action="store_true", help="Run unit test suite validating execution rules")
    parser.add_argument("--all", action="store_true", help="Run full 5-year backtest (all 262 weeks in 1-by-1 batches)")
    parser.add_argument("--split", choices=["train", "test", "all"], default=None,
                        help="Standard SOP split: 'train' (3Y In-Sample, weeks 1-156), 'test' (2Y Out-of-Sample, weeks 157-262), or 'all' (5Y, weeks 1-262)")
    parser.add_argument("--random-blocks", action="store_true", help="Use SOP 10-Block (6-month units) randomized 6/4 train/test sampling")
    parser.add_argument("--seed", type=int, default=42, help="Seed for randomized 6/4 block sampling (default: 42)")
    parser.add_argument("--blocks", type=str, default=None, help="Comma list of 6-month block IDs (1 to 10) to run (e.g. '1,3,4,6,8,9')")
    parser.add_argument("--weeks", type=str, default=None, help="Range or comma list of weeks (e.g. '1-156', '157-262', '1,2,5')")
    parser.add_argument("--week", type=int, default=None, help="Single week number (1 to 262) to test")
    parser.add_argument("--data", type=str, default=None, help="Custom path to single tick data file (.npy, .parquet, .csv)")
    parser.add_argument("--convert", action="store_true", help="Batch convert raw CSVs into processed .npy files upfront")

    # Paths and options
    parser.add_argument("--raw-dir", type=str, default="eurusd_weekly_data", help="Directory of raw weekly CSV files")
    parser.add_argument("--processed-dir", type=str, default="tick_engine/data/processed", help="Directory for processed .npy files")
    parser.add_argument("--export-csv", type=str, default="results/tick_backtest_weekly_summary.csv", help="Path to export weekly CSV report")
    parser.add_argument("--verbose-scorecards", action="store_true", help="Print full scorecard for each individual week")
    parser.add_argument("--enforce-hours", action="store_true", default=True, help="Enforce strategy-specific active trading hours")
    parser.add_argument("--no-hours-filter", action="store_false", dest="enforce_hours", help="Disable strategy hours filter (run 24/5)")

    # Strategy configuration
    parser.add_argument("--strategy", type=str, default="ema_crossover", help=f"Strategy to run ({', '.join(list_strategies())})")
    parser.add_argument("--fast", type=int, default=150, help="Fast period in ticks (EMA)")
    parser.add_argument("--slow", type=int, default=600, help="Slow period in ticks (EMA)")
    parser.add_argument("--lookback", type=int, default=50, help="Lookback window in ticks (Momentum)")
    parser.add_argument("--threshold", type=float, default=3.0, help="Momentum threshold in pips")
    parser.add_argument("--sl", type=float, default=15.0, help="Stop loss in pips")
    parser.add_argument("--tp", type=float, default=30.0, help="Take profit in pips")
    parser.add_argument("--pip-size", type=float, default=0.0001, help="Pip value (0.0001 for EURUSD)")
    parser.add_argument("--slippage", action="store_true", help="Simulate gap slippage on SL/TP fills")

    args = parser.parse_args()

    if args.test:
        success = run_unit_tests()
        sys.exit(0 if success else 1)

    if args.benchmark:
        success = run_1m_benchmark(target_seconds=1.5)
        sys.exit(0 if success else 1)

    raw_path = Path(args.raw_dir)
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path

    proc_path = Path(args.processed_dir)
    if not proc_path.is_absolute():
        proc_path = PROJECT_ROOT / proc_path

    if args.convert:
        print(f"[*] Batch converting CSVs from {raw_path} to {proc_path}...")
        files = batch_convert_csv_to_npy(raw_path, proc_path)
        print(f"[+] Successfully converted {len(files)} files to contiguous .npy format.")
        sys.exit(0)

    # Determine weeks list to run
    manifest = load_manifest(raw_path)
    total_manifest_weeks = len(manifest) if manifest else 262
    TRAIN_WEEKS_END = 156

    from tick_engine.core.block_sampler import (
        sample_train_test_blocks,
        get_weeks_for_blocks,
        format_block_summary
    )

    weeks_to_run: Optional[List[int]] = None
    if args.blocks:
        b_ids = [int(x.strip()) for x in args.blocks.split(",") if x.strip().isdigit()]
        weeks_to_run = get_weeks_for_blocks(b_ids)
        print(f"[*] Running Specific 6-Month Units: {format_block_summary(b_ids)}")
    elif args.random_blocks:
        train_b, test_b = sample_train_test_blocks(seed=args.seed)
        if args.split == "test":
            weeks_to_run = get_weeks_for_blocks(test_b)
            print(f"[*] Running SOP 4-Block RANDOMIZED OUT-OF-SAMPLE TEST (Seed {args.seed}): {format_block_summary(test_b)}")
        else:
            # Default or train
            weeks_to_run = get_weeks_for_blocks(train_b)
            print(f"[*] Running SOP 6-Block RANDOMIZED IN-SAMPLE TRAIN (Seed {args.seed}): {format_block_summary(train_b)}")
    elif args.split == "train":
        weeks_to_run = list(range(1, min(TRAIN_WEEKS_END, total_manifest_weeks) + 1))
        print(f"[*] Running SOP 3-Year IN-SAMPLE TRAINING Split (Weeks 1 to {len(weeks_to_run)})...")
    elif args.split == "test":
        weeks_to_run = list(range(TRAIN_WEEKS_END + 1, total_manifest_weeks + 1))
        print(f"[*] Running SOP 2-Year OUT-OF-SAMPLE TESTING Split (Weeks {TRAIN_WEEKS_END + 1} to {total_manifest_weeks})...")
    elif args.split == "all" or args.all:
        weeks_to_run = sorted(list(manifest.keys())) if manifest else list(range(1, total_manifest_weeks + 1))
        print(f"[*] Running Full 5-Year Dataset (Weeks 1 to {len(weeks_to_run)})...")
    elif args.weeks:
        weeks_to_run = parse_weeks_string(args.weeks, max_weeks=total_manifest_weeks)
    elif args.week is not None:
        weeks_to_run = [args.week]



    # Mode 1: Top 15 Fleet Runner
    if args.top15:
        if weeks_to_run is None:
            # Default to all weeks if not specified
            manifest = load_manifest(raw_path)
            weeks_to_run = sorted(list(manifest.keys())) if manifest else list(range(1, 53))

        summary_csv = PROJECT_ROOT / "results" / "top_15_tick_backtest_summary.csv"
        matrix_csv = PROJECT_ROOT / "results" / "top_15_tick_weekly_matrix.csv"

        run_top15_fleet(
            weeks=weeks_to_run,
            raw_path=raw_path,
            proc_path=proc_path,
            enforce_hours=args.enforce_hours,
            pip_size=args.pip_size,
            export_summary_csv=summary_csv,
            export_weekly_csv=matrix_csv
        )
        sys.exit(0)

    # Mode 2: Single Strategy Weekly batch processing (1 by 1 weeks)
    if weeks_to_run is not None:
        strat_kwargs = {
            "sl_pips": args.sl,
            "tp_pips": args.tp,
            "fast_period": args.fast,
            "slow_period": args.slow,
            "lookback": args.lookback,
            "threshold_pips": args.threshold,
            "pip_size": args.pip_size,
            "enforce_hours": args.enforce_hours
        }
        strat = get_strategy(args.strategy, **{k: v for k, v in strat_kwargs.items() if k in ("sl_pips", "tp_pips", "fast_period", "slow_period", "lookback", "threshold_pips", "pip_size", "enforce_hours")})

        engine = BacktestEngine(
            pip_size=args.pip_size,
            sl_pips=args.sl,
            tp_pips=args.tp,
            use_slippage=args.slippage
        )

        total_weeks_count = len(weeks_to_run)
        print("\n" + "=" * 70)
        print(f"  TICK ENGINE BATCH SIMULATION: {total_weeks_count} WEEKS (1-by-1 Execution)")
        print(f"  Strategy : {strat.name} (SL={args.sl}p, TP={args.tp}p)")
        print(f"  Data Dir : {raw_path}")
        print("=" * 70)

        print_weekly_progress_header()

        weekly_records = []
        all_trades_list = []
        total_ticks_eval = 0
        jit_warmed = False
        t_batch_start = time.perf_counter()

        for idx, w in enumerate(weeks_to_run, start=1):
            t_load_start = time.perf_counter()
            ticks, meta = get_week_dataset(w, raw_dir=raw_path, processed_dir=proc_path, auto_convert=True)
            t_load = time.perf_counter() - t_load_start
            total_ticks_eval += len(ticks)

            # JIT Warmup on first batch
            if not jit_warmed:
                _ = engine.run(ticks[:1000], strat)
                jit_warmed = True

            t_sim_start = time.perf_counter()
            trades = engine.run(ticks, strat)
            t_sim = time.perf_counter() - t_sim_start

            m = compute_metrics(trades)
            weekly_records.append({
                "week": w,
                "meta": meta,
                "metrics": m,
                "elapsed": t_sim,
                "trades": trades
            })
            all_trades_list.append(trades)

            print_weekly_progress_row(w, meta, m, t_sim)

            if args.verbose_scorecards:
                print_scorecard(m, title=f"{strat.name} - Week {w:02d}")

        t_batch_total = time.perf_counter() - t_batch_start

        # Aggregate annual metrics
        if all_trades_list and any(len(t) > 0 for t in all_trades_list):
            valid_trades = [t for t in all_trades_list if len(t) > 0]
            all_trades = np.concatenate(valid_trades) if valid_trades else np.empty(0, dtype=TRADE_DTYPE)
        else:
            all_trades = np.empty(0, dtype=TRADE_DTYPE)

        annual_m = compute_annual_metrics(all_trades, weekly_records)

        title = f"ANNUAL {total_weeks_count}-WEEK TICK SCORECARD ({strat.name})"
        print_annual_scorecard(annual_m, title=title)

        overall_throughput = total_ticks_eval / t_batch_total if t_batch_total > 0 else 0
        print(f"[*] Total Ticks Evaluated : {total_ticks_eval:,} in {t_batch_total:.2f}s "
              f"({overall_throughput:,.0f} ticks/sec overall)")

        if args.export_csv:
            export_path = PROJECT_ROOT / args.export_csv if not Path(args.export_csv).is_absolute() else Path(args.export_csv)
            export_weekly_csv(weekly_records, annual_m, strat.name, export_path)

        sys.exit(0)

    # Mode 3: Custom single file data path
    if args.data:
        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = Path.cwd() / data_path
    else:
        # Default fallback
        print("[*] No weeks or file specified. Running 1-million tick speed benchmark.")
        run_1m_benchmark(target_seconds=1.5)
        sys.exit(0)

    print(f"[*] Loading ticks from: {data_path}")
    t0 = time.perf_counter()
    ticks = load_ticks(data_path, mmap=True)
    t_load = time.perf_counter() - t0
    print(f"[+] Loaded {len(ticks):,} ticks in {t_load*1000:.2f}ms (Memory-mapped).")

    strat_kwargs = {
        "sl_pips": args.sl,
        "tp_pips": args.tp,
        "fast_period": args.fast,
        "slow_period": args.slow,
        "lookback": args.lookback,
        "threshold_pips": args.threshold,
        "pip_size": args.pip_size
    }
    strat = get_strategy(args.strategy, **{k: v for k, v in strat_kwargs.items() if k in ("sl_pips", "tp_pips", "fast_period", "slow_period", "lookback", "threshold_pips", "pip_size")})
    engine = BacktestEngine(pip_size=args.pip_size, sl_pips=args.sl, tp_pips=args.tp, use_slippage=args.slippage)

    _ = engine.run(ticks[:1000], strat)

    print(f"[*] Running simulation with {strat.name}...")
    t0 = time.perf_counter()
    trades = engine.run(ticks, strat)
    t_sim = time.perf_counter() - t0

    throughput = len(ticks) / t_sim if t_sim > 0 else 0
    print(f"[+] Simulation completed in {t_sim:.4f}s ({throughput:,.0f} ticks/sec).")

    metrics = compute_metrics(trades)
    print_scorecard(metrics, title=f"{strat.name} ({data_path.stem})")


if __name__ == '__main__':
    main()
