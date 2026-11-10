@echo off
title VS Code Remote SSH - Azure VM (C:\BACKTEST)
echo ================================================================================
echo    Launching VS Code Remote SSH to Azure Cloud VM (azure-vm)
echo    Target Directory: C:\BACKTEST
echo ================================================================================

:: Ensure remote C:\BACKTEST exists before opening
ssh azure-vm "if not exist C:\BACKTEST mkdir C:\BACKTEST"

:: Launch VS Code Remote SSH session directly attached to C:\BACKTEST
code --folder-uri "vscode-remote://ssh-remote+azure-vm/C:/BACKTEST"

