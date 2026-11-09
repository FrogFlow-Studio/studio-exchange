@echo off
rem Открыть демо в браузере (страница на этом компьютере, интернет не нужен)
chcp 65001 >nul
cd /d "%~dp0"
set PY=python
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe
%PY% app.py
pause
