@echo off
rem Демо с ИИ: ключ спрашивается при запуске и живёт только в этом окне (никуда не сохраняется)
chcp 65001 >nul
cd /d "%~dp0"
set PY=python
py -3.12 --version >nul 2>&1 && set PY=py -3.12
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe
rem после обновления (git pull) могли появиться новые библиотеки: доставляем недостающие
%PY% -m pip install -q -r requirements.txt >nul 2>&1 || echo Не удалось доставить библиотеки: запустите install.bat
set /p ANTHROPIC_API_KEY=Вставьте ключ ИИ и нажмите Enter (пусто - без ИИ): 
if "%ANTHROPIC_API_KEY%"=="" set ANTHROPIC_API_KEY=
cls
%PY% app.py
pause
