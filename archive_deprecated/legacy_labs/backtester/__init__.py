"""
Forex Weekly Pip Backtest System.
High-speed, institutional-grade backtesting engine built for EURUSD weekly datasets.
"""

from backtester.models import Trade, Position
from backtester.data_loader import DataLoader
from backtester.strategy import Strategy
from backtester.engine import BacktestEngine
from backtester.metrics import PerformanceMetrics
from backtester.exporter import ResultsExporter
from backtester.strategies import AVAILABLE_STRATEGIES, get_strategy

__all__ = [
    "Trade",
    "Position",
    "DataLoader",
    "Strategy",
    "BacktestEngine",
    "PerformanceMetrics",
    "ResultsExporter",
    "AVAILABLE_STRATEGIES",
    "get_strategy"
]
