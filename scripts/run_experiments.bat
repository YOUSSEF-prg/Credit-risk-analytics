@echo off
REM Run all experiments in configs\models.yaml (tuning, validation, comparisons, selection).
REM Output: logs\experiments_run.log. Double-click, or run from Anaconda Prompt.
REM Arguments are passed on, e.g.  run_experiments.bat --resume   (skip finished models).
cd /d "%~dp0.."
if not exist logs mkdir logs
set LOG=logs\experiments_run.log
set PYTHONUNBUFFERED=1
set MLFLOW_DISABLE_AGENT_HINT=1
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" credit-risk
if errorlevel 1 (
    echo Could not activate the conda environment "credit-risk". > %LOG%
    exit /b 1
)
title Credit risk - experiments (do not close)
echo === Experiments started %DATE% %TIME% (args: %*) === > %LOG%
python scripts\run_experiments.py %* >> %LOG% 2>&1
if errorlevel 1 (
    echo EXPERIMENTS_FAILED: see traceback above >> %LOG%
    exit /b 1
)
echo EXPERIMENTS_DONE %DATE% %TIME% >> %LOG%
