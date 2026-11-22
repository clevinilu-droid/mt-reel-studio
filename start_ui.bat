@echo off
cd /d "%~dp0"
start "" http://127.0.0.1:7860
"%~dp0.venv\Scripts\python.exe" "%~dp0_engine\run_ui.py" --host 127.0.0.1 --port 7860
pause
