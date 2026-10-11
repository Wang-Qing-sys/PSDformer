@echo off
setlocal
cd /d "%~dp0"
python -m psdformer demo --output runs/demo --device cpu
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" (
    echo Demo failed. Check the error above and activate the Python environment from README.md.
) else (
    echo Demo completed. Outputs are in runs\demo.
)
pause
exit /b %RESULT%
