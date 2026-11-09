@echo off
rem Одна команда: run.bat [скан.stl] ["заметки"]
chcp 65001 >nul
cd /d "%~dp0"
set SCAN=%~1
if "%SCAN%"=="" set SCAN=samples\bracket_scan.stl
set NOTES=%~2
if "%NOTES%"=="" set NOTES=угол плиты отколот, отверстие в нём потеряно
python demo.py "%SCAN%" --notes "%NOTES%"
pause
