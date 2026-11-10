@echo off
title Top 15 Strategy Fleet Backtester
echo ===============================================================================
echo                TOP 15 HIGH-CONVICTION STRATEGY FLEET BACKTESTER
echo ===============================================================================
echo.
echo Running bar-by-bar backtest across all Top 15 strategies...
echo Data Source: sample_eurusd_m15.csv
echo Filter Mode: V3.8 Empirical Hours Filter
echo Friction:    0.8 pips spread
echo.

python backtester.py --strategy all --hours-filter --spread 0.8

echo.
echo ===============================================================================
echo Backtest completed!
echo Detailed trade log exported to: top_15_backtest_trades.csv
echo ===============================================================================
pause
