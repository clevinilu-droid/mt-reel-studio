@echo off
cd /d "%~dp0"
start "" "http://127.0.0.1:8787"
.venv\Scripts\python.exe -m uvicorn ui.app:app --host 127.0.0.1 --port 8787
