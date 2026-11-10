@echo off
title EUR/USD Volatility Chart GUI - Time-Invariant Terminal
echo ===================================================================
echo     EUR/USD VOLATILITY CHART TERMINAL (TIME-INVARIANT)
echo ===================================================================
echo Launching GUI with Python Virtual Environment...
echo.
"%~dp0.venv\Scripts\python.exe" "%~dp0eurusd_volatility_gui.py"
if %errorlevel% neq 0 (
    echo.
    echo [!] Attempting with system python...
    python "%~dp0eurusd_volatility_gui.py"
)
pause
