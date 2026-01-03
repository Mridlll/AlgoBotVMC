@echo off
REM VMC Trading Bot - Auto-restart wrapper
REM This script keeps the bot running by restarting it when it exits

echo ============================================
echo VMC Trading Bot - Forever Runner
echo ============================================
echo This wrapper will automatically restart the bot if it crashes
echo Press Ctrl+C twice to fully stop
echo ============================================

cd /d "%~dp0"

:loop
echo.
echo [%date% %time%] Starting bot...
python run_production.py --config config/config.yaml
echo.
echo [%date% %time%] Bot exited with code %ERRORLEVEL%
echo Waiting 10 seconds before restart...
timeout /t 10 /nobreak
goto loop
