#!/bin/bash
# VMC Trading Bot - Unix Run Script
# ==================================

echo ""
echo "========================================"
echo "  VMC Trading Bot"
echo "========================================"
echo ""

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "[ERROR] Virtual environment not found. Run setup.sh first."
    exit 1
fi

# Determine config file
CONFIG_FILE="config/config.yaml"

if [ "$1" == "production" ] || [ "$1" == "prod" ]; then
    CONFIG_FILE="config/config_production.yaml"
    echo "[*] Using PRODUCTION config"
elif [ -n "$1" ]; then
    CONFIG_FILE="$1"
    echo "[*] Using custom config: $1"
else
    echo "[*] Using default config"
fi

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] Config file not found: $CONFIG_FILE"
    echo "Run setup.sh first or specify a valid config file."
    exit 1
fi

echo "[*] Config: $CONFIG_FILE"
echo ""

# Run the bot
echo "[*] Starting VMC Trading Bot..."
echo "[*] Press Ctrl+C to stop"
echo ""
echo "----------------------------------------"

python src/main.py --config "$CONFIG_FILE"

echo ""
echo "----------------------------------------"
echo "[*] Bot stopped"
