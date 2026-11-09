@echo off
rem Открыть демо в браузере (страница на этом компьютере, интернет не нужен)
chcp 65001 >nul
cd /d "%~dp0"
python app.py
pause
