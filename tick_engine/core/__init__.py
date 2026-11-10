"""
Core module for Fast Minimal Tick-Level Strategy Tester.
"""
from .engine import TICK_DTYPE, TRADE_DTYPE, BacktestEngine, run_matching_loop
from .dataloader import load_ticks, csv_to_npy, csv_to_parquet, batch_convert_csv_to_npy
from .metrics import compute_metrics, print_scorecard

__all__ = [
    "TICK_DTYPE",
    "TRADE_DTYPE",
    "BacktestEngine",
    "run_matching_loop",
    "load_ticks",
    "csv_to_npy",
    "csv_to_parquet",
    "batch_convert_csv_to_npy",
    "compute_metrics",
    "print_scorecard",
]
