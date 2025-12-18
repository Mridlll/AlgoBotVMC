@echo off
REM VMC Trading Bot - Windows Setup Script
REM ========================================

echo.
echo ========================================
echo   VMC Trading Bot - Setup
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [*] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)
echo.

REM Activate virtual environment
echo [*] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM Upgrade pip
echo [*] Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo [OK] pip upgraded
echo.

REM Install dependencies
echo [*] Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed
echo.

REM Create data directory for SQLite database
if not exist "data" (
    echo [*] Creating data directory...
    mkdir data
    echo [OK] Data directory created
)
echo.

REM Create config from example if it doesn't exist
if not exist "config\config.yaml" (
    if exist "config\config.example.yaml" (
        echo [*] Creating config.yaml from example...
        copy config\config.example.yaml config\config.yaml >nul
        echo [OK] Config file created at config\config.yaml
        echo.
        echo [!] IMPORTANT: Edit config\config.yaml with your credentials before running!
    )
) else (
    echo [OK] Config file already exists
)
echo.

REM Check if production config exists
if exist "config\config_production.yaml" (
    echo [OK] Production config available at config\config_production.yaml
)
echo.

echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Edit config\config.yaml (or config\config_production.yaml)
echo      - Add your Hyperliquid private key
echo      - Add your wallet address
echo      - Keep testnet: true for initial testing
echo.
echo   2. Run the bot:
echo      run_bot.bat
echo.
echo   3. Or run with production config:
echo      run_bot.bat production
echo.
pause
