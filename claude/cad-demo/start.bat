@echo off
rem Рабочее окно в браузере (страница на этом компьютере, интернет не нужен). Старая страница: python app.py
chcp 65001 >nul
cd /d "%~dp0"
set PY=python
py -3.12 --version >nul 2>&1 && set PY=py -3.12
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe
rem после обновления (git pull) могли появиться новые библиотеки: доставляем недостающие
%PY% -m pip install -q -r requirements.txt >nul 2>&1 || echo Не удалось доставить библиотеки: запустите install.bat
%PY% workspace\server.py
pause
