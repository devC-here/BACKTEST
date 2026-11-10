@echo off
setlocal enabledelayedexpansion
title BACKTEST: Azure VM Remote Compute Dispatcher
color 0b

echo ===============================================================================
echo        BACKTEST ENGINE - AZURE CLOUD VM REMOTE COMPUTE DISPATCHER
echo ===============================================================================

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b 1
)

:: If argument provided, pass directly
if not "%~1"=="" (
    echo [*] Dispatching: python scripts\remote_compute_dispatcher.py --script run_symbol_alpha_discovery.py --args "--symbol %~1"
    python scripts\remote_compute_dispatcher.py --script run_symbol_alpha_discovery.py --args "--symbol %~1"
    goto DONE
)

:: Interactive Menu
echo Select job to dispatch to Azure VM:
echo   [1] Run Master Alpha Discovery on EURUSD
echo   [2] Run Master Alpha Discovery on USDJPY
echo   [3] Run Master Alpha Discovery on GBPJPY
echo   [4] Run Master Alpha Discovery on EURGBP
echo   [5] Run 5-Year Fleet Audit (run_5y_fleet_audit.py)
echo   [6] Custom Script or Command
echo.
set /p CHOICE="Enter choice [1-6]: "

if "%CHOICE%"=="1" (
    python scripts\remote_compute_dispatcher.py --script run_symbol_alpha_discovery.py --args "--symbol EURUSD"
) else if "%CHOICE%"=="2" (
    python scripts\remote_compute_dispatcher.py --script run_symbol_alpha_discovery.py --args "--symbol USDJPY"
) else if "%CHOICE%"=="3" (
    python scripts\remote_compute_dispatcher.py --script run_symbol_alpha_discovery.py --args "--symbol GBPJPY"
) else if "%CHOICE%"=="4" (
    python scripts\remote_compute_dispatcher.py --script run_symbol_alpha_discovery.py --args "--symbol EURGBP"
) else if "%CHOICE%"=="5" (
    python scripts\remote_compute_dispatcher.py --script run_5y_fleet_audit.py
) else if "%CHOICE%"=="6" (
    set /p CUST_SCRIPT="Enter script name (e.g. run_symbol_alpha_discovery.py): "
    set /p CUST_ARGS="Enter script arguments (e.g. --symbol AUDCAD): "
    python scripts\remote_compute_dispatcher.py --script !CUST_SCRIPT! --args "!CUST_ARGS!"
) else (
    echo [!] Invalid selection.
)

:DONE
echo.
echo ===============================================================================
echo Job finished or terminated.
echo ===============================================================================
pause

