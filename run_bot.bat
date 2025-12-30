@echo off
REM VMC Trading Bot - Windows Run Script
REM =====================================

echo.
echo ========================================
echo   VMC Trading Bot
echo ========================================
echo.

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [ERROR] Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

REM Determine config file
set CONFIG_FILE=config\config.yaml

if "%1"=="production" (
    set CONFIG_FILE=config\config_production.yaml
    echo [*] Using PRODUCTION config
) else if "%1"=="prod" (
    set CONFIG_FILE=config\config_production.yaml
    echo [*] Using PRODUCTION config
) else if "%1"=="v6" (
    set CONFIG_FILE=config\config_v6_production.yaml
    echo [*] Using V6 PRODUCTION config (15 strategies)
) else if not "%1"=="" (
    set CONFIG_FILE=%1
    echo [*] Using custom config: %1
) else (
    echo [*] Using default config
)

REM Check if config exists
if not exist "%CONFIG_FILE%" (
    echo [ERROR] Config file not found: %CONFIG_FILE%
    echo Run setup.bat first or specify a valid config file.
    pause
    exit /b 1
)

echo [*] Config: %CONFIG_FILE%
echo.

REM Run the bot
echo [*] Starting VMC Trading Bot...
echo [*] Press Ctrl+C to stop
echo.
echo ----------------------------------------

python src\main.py --config %CONFIG_FILE%

echo.
echo ----------------------------------------
echo [*] Bot stopped
pause
