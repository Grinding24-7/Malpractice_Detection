@echo off
setlocal
title IntelliProctor Setup
set "APPDIR=%~dp0"
set "APP=%APPDIR%IntelliProctor.exe"
set "IP_APP=%APP%"
set "IP_DIR=%APPDIR%"

if not exist "%APP%" (
    echo [ERROR] IntelliProctor.exe was not found in:
    echo         %APPDIR%
    echo.
    echo Keep this installer in the same folder as IntelliProctor.exe.
    echo.
    pause
    exit /b 1
)

echo Creating an IntelliProctor shortcut on your Desktop...
powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $ws = New-Object -ComObject WScript.Shell; $desktop = [Environment]::GetFolderPath([Environment+SpecialFolder]::DesktopDirectory); $lnk = Join-Path $desktop 'IntelliProctor.lnk'; $s = $ws.CreateShortcut($lnk); $s.TargetPath = $env:IP_APP; $s.WorkingDirectory = $env:IP_DIR; $s.IconLocation = $env:IP_APP + ',0'; $s.Description = 'IntelliProctor - AI-assisted exam proctoring and malpractice detection'; $s.WindowStyle = 1; $s.Save(); Write-Host ('Created: ' + $lnk)"
if errorlevel 1 (
    echo [ERROR] Could not create the shortcut.
    echo.
    pause
    exit /b 1
)

echo.
echo Done. You can now start IntelliProctor from your Desktop.
echo To uninstall, delete the Desktop shortcut and this folder.
echo.
choice /C YN /N /M "Launch IntelliProctor now? [Y/N] "
if errorlevel 2 goto :end
start "" "%APP%"
:end
endlocal
