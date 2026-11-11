@echo off
title Push BACKTEST Platform to GitHub
echo ================================================================================
echo    Pushing BACKTEST Institutional Platform to https://github.com/devC-here/BACKTEST
echo ================================================================================
cd /d "%~dp0\.."
git push -u origin main
echo ================================================================================
echo    Push complete! Repository is live at:
echo    https://github.com/devC-here/BACKTEST
echo ================================================================================
pause
