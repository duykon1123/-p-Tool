@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo [1/2] Tao moi truong ao .venv ...
  py -3 -m venv .venv || python -m venv .venv || (echo Chua cai Python 3.10+ & pause & exit /b 1)
)
echo [2/2] Cai thu vien (lan dau hoi lau) ...
.venv\Scripts\python.exe -m pip install -q -r requirements.txt pywebview
echo Khoi dong Dop Tool ...
.venv\Scripts\python.exe app_window.py
pause
