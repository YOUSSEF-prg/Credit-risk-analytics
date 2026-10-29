@echo off
REM Launch the Streamlit dashboard (opens http://localhost:8501 in your browser).
cd /d "%~dp0.."
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" credit-risk
title Credit risk - dashboard (close this window to stop)
streamlit run app.py
