@echo off
REM Double-click launcher for auto_cut.
REM UTF-8 is forced because WhisperX prints Filipino transcript text to stdout
REM and the Windows cp1252 default crashes the run mid-file without it.

setlocal
cd /d "%~dp0auto_cut"

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

python app.py

REM Only pause if something went wrong, so a normal close doesn't leave a
REM console window sitting there.
if errorlevel 1 (
    echo.
    echo Auto-Cut exited with an error ^(code %errorlevel%^).
    pause
)
endlocal
