#!/bin/bash
# VMC Trading Bot - Unix Setup Script
# ====================================

set -e

echo ""
echo "========================================"
echo "  VMC Trading Bot - Setup"
echo "========================================"
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 is not installed"
    echo "Please install Python 3.10+ using your package manager"
    exit 1
fi

echo "[OK] Python found"
python3 --version
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[*] Creating virtual environment..."
    python3 -m venv venv
    echo "[OK] Virtual environment created"
else
    echo "[OK] Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "[*] Activating virtual environment..."
source venv/bin/activate
echo "[OK] Virtual environment activated"
echo ""

# Upgrade pip
echo "[*] Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "[OK] pip upgraded"
echo ""

# Install dependencies
echo "[*] Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt
echo "[OK] Dependencies installed"
echo ""

# Create data directory for SQLite database
if [ ! -d "data" ]; then
    echo "[*] Creating data directory..."
    mkdir -p data
    echo "[OK] Data directory created"
fi
echo ""

# Create config from example if it doesn't exist
if [ ! -f "config/config.yaml" ]; then
    if [ -f "config/config.example.yaml" ]; then
        echo "[*] Creating config.yaml from example..."
        cp config/config.example.yaml config/config.yaml
        echo "[OK] Config file created at config/config.yaml"
        echo ""
        echo "[!] IMPORTANT: Edit config/config.yaml with your credentials before running!"
    fi
else
    echo "[OK] Config file already exists"
fi
echo ""

# Check if production config exists
if [ -f "config/config_production.yaml" ]; then
    echo "[OK] Production config available at config/config_production.yaml"
fi
echo ""

echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Edit config/config.yaml (or config/config_production.yaml)"
echo "     - Add your Hyperliquid private key"
echo "     - Add your wallet address"
echo "     - Keep testnet: true for initial testing"
echo ""
echo "  2. Run the bot:"
echo "     ./run_bot.sh"
echo ""
echo "  3. Or run with production config:"
echo "     ./run_bot.sh production"
echo ""
