#!/bin/bash
# VMC Trading Bot V6 - Service Installation Script
# This script installs the bot as a systemd service for auto-start on boot

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "  VMC Trading Bot V6 - Service Installer"
echo "=============================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Error: Please run as root (sudo)${NC}"
    echo "Usage: sudo ./install_service.sh"
    exit 1
fi

# Get the actual user (not root)
if [ -n "$SUDO_USER" ]; then
    ACTUAL_USER="$SUDO_USER"
else
    echo -e "${RED}Error: Could not determine the actual user${NC}"
    echo "Please run with: sudo ./install_service.sh"
    exit 1
fi

# Get installation directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo -e "${GREEN}Installation directory:${NC} $SCRIPT_DIR"
echo -e "${GREEN}Service user:${NC} $ACTUAL_USER"
echo ""

# Verify required files exist
if [ ! -f "$SCRIPT_DIR/run_production.py" ]; then
    echo -e "${RED}Error: run_production.py not found${NC}"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/venv/bin/python" ]; then
    echo -e "${RED}Error: Virtual environment not found${NC}"
    echo "Please run setup.sh first to create the virtual environment"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/config/config.yaml" ]; then
    echo -e "${YELLOW}Warning: config/config.yaml not found${NC}"
    echo "Please run 'python setup_wizard.py' to create configuration"
fi

# Create service file from template
SERVICE_FILE="/etc/systemd/system/vmc-bot.service"

echo "Creating systemd service file..."

cat > "$SERVICE_FILE" << EOF
[Unit]
Description=VMC Trading Bot V6
Documentation=https://github.com/YourUsername/vmc_trading_bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$ACTUAL_USER
Group=$ACTUAL_USER
WorkingDirectory=$SCRIPT_DIR
ExecStart=$SCRIPT_DIR/venv/bin/python run_production.py --config config/config.yaml

# Restart policy
Restart=always
RestartSec=30

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=$SCRIPT_DIR/logs
ReadWritePaths=$SCRIPT_DIR/data
PrivateTmp=true

# Logging
StandardOutput=append:$SCRIPT_DIR/logs/systemd.log
StandardError=append:$SCRIPT_DIR/logs/systemd.log

# Environment
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

echo -e "${GREEN}Service file created:${NC} $SERVICE_FILE"

# Create logs directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/logs"
chown "$ACTUAL_USER:$ACTUAL_USER" "$SCRIPT_DIR/logs"

# Create data directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/data"
chown "$ACTUAL_USER:$ACTUAL_USER" "$SCRIPT_DIR/data"

# Reload systemd
echo "Reloading systemd daemon..."
systemctl daemon-reload

# Enable service
echo "Enabling service to start on boot..."
systemctl enable vmc-bot

echo ""
echo "=============================================="
echo -e "${GREEN}  Installation Complete!${NC}"
echo "=============================================="
echo ""
echo "Commands:"
echo "  Start bot:    sudo systemctl start vmc-bot"
echo "  Stop bot:     sudo systemctl stop vmc-bot"
echo "  Restart bot:  sudo systemctl restart vmc-bot"
echo "  Check status: sudo systemctl status vmc-bot"
echo "  View logs:    sudo journalctl -u vmc-bot -f"
echo "  View app log: tail -f $SCRIPT_DIR/logs/production.log"
echo ""
echo -e "${YELLOW}Important:${NC}"
echo "  1. Make sure you've run 'python setup_wizard.py' first"
echo "  2. Test manually before starting service:"
echo "     cd $SCRIPT_DIR && ./venv/bin/python run_production.py"
echo "  3. Start with: sudo systemctl start vmc-bot"
echo ""
