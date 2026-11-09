@echo off
rem Установка один раз. Нужен Python 3.10-3.12 с python.org (галочка "Add to PATH").
chcp 65001 >nul
cd /d "%~dp0"
python --version || (echo Python не найден. Поставьте с https://www.python.org/downloads/ и отметьте "Add python.exe to PATH". & pause & exit /b 1)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt || (echo Установка не удалась, пришлите этот экран. & pause & exit /b 1)
echo.
echo Готово. Запуск: start.bat
pause
