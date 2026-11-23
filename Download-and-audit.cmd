@echo off
setlocal
cd /d "%~dp0" || exit /b 1
set "PROJECT_PYTHON=python"
if exist "..\.venv\Scripts\python.exe" set "PROJECT_PYTHON=..\.venv\Scripts\python.exe"

echo Downloading and checking the three Stanford archives...
"%PROJECT_PYTHON%" "scripts\download_stanford.py"
if errorlevel 1 goto download_failed

echo Auditing speeches and speaker-party links...
"%PROJECT_PYTHON%" "scripts\audit_stanford.py"
if errorlevel 1 goto audit_failed

echo Finished. The audit tables are in "%~dp0..\stanford_raw\audit".
pause
exit /b 0

:download_failed
echo Download or archive validation failed. See the error above.
echo Run this file again to resume an interrupted download.
pause
exit /b 1

:audit_failed
echo The archives passed validation, but the audit failed. See the error above.
pause
exit /b 1
