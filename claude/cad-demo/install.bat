@echo off
rem Установка один раз (и после обновления, если что-то сломалось).
rem Нужен Python 3.11 или 3.12 с python.org (галочка "Add python.exe to PATH").
rem Ставит библиотеки в отдельную папку .venv рядом и кладёт ярлык "CAD Intelligence" на рабочий стол.
chcp 65001 >nul
cd /d "%~dp0"
set PY=
py -3.12 --version >nul 2>&1 && set PY=py -3.12
if not defined PY (py -3.11 --version >nul 2>&1 && set PY=py -3.11)
if not defined PY (python --version >nul 2>&1 && set PY=python)
if not defined PY (
  echo Python не найден. Поставьте Python 3.12 с https://www.python.org/downloads/
  echo и отметьте "Add python.exe to PATH", потом запустите install.bat ещё раз.
  pause
  exit /b 1
)
echo [1/4] Python: %PY%
if not exist .venv\Scripts\python.exe (
  %PY% -m venv .venv || (echo Не удалось создать папку .venv, пришлите этот экран. & pause & exit /b 1)
)
echo [2/4] Ставлю библиотеки, это 5-15 минут при первой установке...
.venv\Scripts\python.exe -m pip install -q --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt || (echo Установка библиотек не удалась, пришлите этот экран. & pause & exit /b 1)
echo [3/4] Проверяю, что всё на месте...
.venv\Scripts\python.exe -c "import cadquery, trimesh, manifold3d, pymeshfix, fast_simplification, psutil; print('библиотеки на месте')" || (echo Библиотеки не загрузились, пришлите этот экран. & pause & exit /b 1)
.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'workspace'); import server" >nul 2>&1 || echo Внимание: рабочее окно не запускается. Нужна папка chatgpt-1 рядом (полная копия studio-exchange).
echo [4/4] Ярлык на рабочем столе...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$d=[Environment]::GetFolderPath('Desktop'); $s=(New-Object -ComObject WScript.Shell).CreateShortcut($d+'\CAD Intelligence.lnk'); $s.TargetPath='%~dp0start.bat'; $s.WorkingDirectory='%~dp0'; $s.Save()" >nul 2>&1 && echo Ярлык "CAD Intelligence" создан. || echo Ярлык создать не удалось: запускайте start.bat из этой папки.
echo.
echo Готово. Запуск: ярлык "CAD Intelligence" или start.bat (с ИИ: start_ai.bat).
pause
