@echo off
REM Re-run the test suite and artifact checks on this PC. Output: logs\verify_run.log
REM Double-click this file. Keep the window open until the log ends with VERIFY_DONE.
cd /d "%~dp0.."
if not exist logs mkdir logs
set LOG=logs\verify_run.log
set PYTHONUNBUFFERED=1
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" credit-risk
if errorlevel 1 (
    echo Could not activate the conda environment "credit-risk". > %LOG%
    exit /b 1
)
title Credit risk - verification (do not close)
echo === Verification started %DATE% %TIME% === > %LOG%
python -m pip install -q -r requirements.txt >> %LOG% 2>&1
python -m pip install -q -e . >> %LOG% 2>&1
echo === Recompute SHAP on this PC (frozen model, no re-selection) === >> %LOG%
python scripts\compute_shap.py >> %LOG% 2>&1
echo === Artifact checks === >> %LOG%
python scripts\verify_artifacts.py >> %LOG% 2>&1
set CHECKS=%ERRORLEVEL%
echo === Test suite === >> %LOG%
python -m pytest -q -p no:cacheprovider >> %LOG% 2>&1
set TESTS=%ERRORLEVEL%
echo VERIFY_DONE checks=%CHECKS% tests=%TESTS% %DATE% %TIME% >> %LOG%
