; Установщик CAD Intelligence для Windows 10/11 (64 бит).
; Внутри всё: Python 3.12, библиотеки, программа и проверка Севера. Интернет не нужен,
; права администратора не нужны (ставится в папку пользователя).
; Собирается на Linux: installer/build_installer.sh (makensis).

Unicode true
SetCompressor /SOLID lzma
SetCompressorDictSize 64
ManifestDPIAware true
RequestExecutionLevel user

!ifndef VERSION
  !define VERSION "0.0.0"
!endif
!ifndef STAGE
  !define STAGE "stage"
!endif
!ifndef OUTFILE
  !define OUTFILE "CAD-Intelligence-Setup.exe"
!endif

!define APP "CAD Intelligence"
!define COMPANY "FrogFlow Studio"
!define UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\CADIntelligence"
!define LAUNCH '"$INSTDIR\python\pythonw.exe" "$INSTDIR\app\claude\cad-demo\workspace\launcher.pyw"'

Name "${APP}"
Caption "${APP} — установка"
BrandingText "${COMPANY} · версия ${VERSION}"
OutFile "${OUTFILE}"
InstallDir "$LOCALAPPDATA\Programs\${APP}"
InstallDirRegKey HKCU "${UNINST_KEY}" "InstallLocation"

VIProductVersion "1.0.0.0"
VIAddVersionKey /LANG=1049 "ProductName" "${APP}"
VIAddVersionKey /LANG=1049 "CompanyName" "${COMPANY}"
VIAddVersionKey /LANG=1049 "FileDescription" "Установка ${APP}"
VIAddVersionKey /LANG=1049 "FileVersion" "${VERSION}"
VIAddVersionKey /LANG=1049 "ProductVersion" "${VERSION}"
VIAddVersionKey /LANG=1049 "LegalCopyright" "${COMPANY}"

!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "x64.nsh"

!define MUI_ICON "${STAGE}/art/icon.ico"
!define MUI_UNICON "${STAGE}/art/icon.ico"
!define MUI_WELCOMEFINISHPAGE_BITMAP "${STAGE}/art/welcome.bmp"
!define MUI_UNWELCOMEFINISHPAGE_BITMAP "${STAGE}/art/welcome.bmp"
!define MUI_HEADERIMAGE
!define MUI_HEADERIMAGE_RIGHT
!define MUI_HEADERIMAGE_BITMAP "${STAGE}/art/header.bmp"
!define MUI_HEADERIMAGE_UNBITMAP "${STAGE}/art/header.bmp"
!define MUI_ABORTWARNING
!define MUI_ABORTWARNING_TEXT "Прервать установку ${APP}?"

!define MUI_WELCOMEPAGE_TITLE "Установка ${APP}"
!define MUI_WELCOMEPAGE_TEXT "Программа строит CAD-модель (STEP) по 3D-скану детали: восстанавливает сколы, находит плоскости, отверстия и скругления и сверяет модель со сканом.$\r$\n$\r$\nВсё нужное уже внутри установщика: интернет и права администратора не нужны.$\r$\n$\r$\nНажмите «Далее», чтобы продолжить."
!insertmacro MUI_PAGE_WELCOME
!define MUI_DIRECTORYPAGE_TEXT_TOP "Программа будет установлена в эту папку. Нужно около 2 ГБ свободного места."
!insertmacro MUI_PAGE_DIRECTORY
!define MUI_PAGE_CUSTOMFUNCTION_SHOW HideDetails
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_TITLE "${APP} установлен"
!define MUI_FINISHPAGE_TEXT "Ярлык «${APP}» на рабочем столе и в меню «Пуск» открывает рабочее окно в браузере. Окно работает на этом компьютере, сканы никуда не отправляются.$\r$\n$\r$\nПервый запуск занимает до минуты."
!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_TEXT "Запустить ${APP}"
!define MUI_FINISHPAGE_RUN_FUNCTION LaunchApp
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

!insertmacro MUI_LANGUAGE "Russian"

Function .onInit
  ${IfNot} ${RunningX64}
    MessageBox MB_ICONSTOP "${APP} работает только на 64-битной Windows 10 или 11."
    Abort
  ${EndIf}
FunctionEnd

Function HideDetails
  SetDetailsView hide
FunctionEnd

Function LaunchApp
  Exec '${LAUNCH}'
FunctionEnd

; Запущенная программа держит файлы открытыми: перед установкой поверх и удалением её
; останавливаем (только процессы Python из папки программы, чужие не трогаются).
!macro StopApp
  nsExec::Exec `powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Process python,pythonw -ErrorAction SilentlyContinue | Where-Object { $$_.Path -like '$INSTDIR\*' } | Stop-Process -Force"`
  Pop $0
  Sleep 500
!macroend

Section "Программа" SecMain
  SectionIn RO
  !insertmacro StopApp
  SetOutPath "$INSTDIR"
  ; старая копия программы удаляется целиком (данные расчётов лежат отдельно и остаются)
  RMDir /r "$INSTDIR\app"
  ; «File /r папка» ищет это имя по всему дереву, поэтому копируется содержимое через *.*
  DetailPrint "Копирую Python и библиотеки…"
  SetOutPath "$INSTDIR\python"
  File /r "${STAGE}/python/*.*"
  DetailPrint "Копирую программу…"
  SetOutPath "$INSTDIR\app"
  File /r "${STAGE}/app/*.*"
  SetOutPath "$INSTDIR"
  File "${STAGE}/art/icon.ico"

  DetailPrint "Готовлю библиотеки к быстрому запуску (до двух минут)…"
  nsExec::ExecToLog '"$INSTDIR\python\python.exe" -m compileall -q -j 0 "$INSTDIR\python\Lib" "$INSTDIR\app"'
  Pop $0
  DetailPrint "Проверяю, что всё на месте…"
  nsExec::ExecToStack '"$INSTDIR\python\python.exe" -c "import cadquery, trimesh, manifold3d, pymeshfix, fast_simplification, psutil, scipy; print(42)"'
  Pop $0
  Pop $1
  ${If} $0 != 0
    MessageBox MB_ICONEXCLAMATION "Библиотеки не загрузились (код $0).$\r$\n$\r$\n$1$\r$\nПришлите снимок этого окна разработчику."
  ${EndIf}

  CreateShortcut "$DESKTOP\${APP}.lnk" "$INSTDIR\python\pythonw.exe" '"$INSTDIR\app\claude\cad-demo\workspace\launcher.pyw"' "$INSTDIR\icon.ico" 0 SW_SHOWNORMAL "" "Открыть рабочее окно ${APP}"
  CreateDirectory "$SMPROGRAMS\${APP}"
  CreateShortcut "$SMPROGRAMS\${APP}\${APP}.lnk" "$INSTDIR\python\pythonw.exe" '"$INSTDIR\app\claude\cad-demo\workspace\launcher.pyw"' "$INSTDIR\icon.ico" 0 SW_SHOWNORMAL "" "Открыть рабочее окно ${APP}"
  CreateShortcut "$SMPROGRAMS\${APP}\Удалить ${APP}.lnk" "$INSTDIR\uninstall.exe"

  WriteUninstaller "$INSTDIR\uninstall.exe"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayName" "${APP}"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "${UNINST_KEY}" "Publisher" "${COMPANY}"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayIcon" "$INSTDIR\icon.ico"
  WriteRegStr HKCU "${UNINST_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNINST_KEY}" "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegDWORD HKCU "${UNINST_KEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINST_KEY}" "NoRepair" 1
  WriteRegDWORD HKCU "${UNINST_KEY}" "EstimatedSize" 1900000
SectionEnd

Section "Uninstall"
  !insertmacro StopApp
  Delete "$DESKTOP\${APP}.lnk"
  RMDir /r "$SMPROGRAMS\${APP}"
  RMDir /r "$INSTDIR\python"
  RMDir /r "$INSTDIR\app"
  Delete "$INSTDIR\icon.ico"
  Delete "$INSTDIR\uninstall.exe"
  RMDir "$INSTDIR"
  DeleteRegKey HKCU "${UNINST_KEY}"
  ; расчёты и сканы пользователя (%LOCALAPPDATA%\FrogFlow\CAD Workspace) не удаляются
SectionEnd
