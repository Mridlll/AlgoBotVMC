#!/usr/bin/env python3
"""
VMC Trading Bot V6 - Interactive Setup Wizard

Guides the user through complete configuration:
- Hyperliquid wallet setup
- Risk settings
- Strategy selection
- Discord notifications

Usage:
    python setup_wizard.py
"""

import os
import sys
import re
from pathlib import Path
from typing import Optional, Dict, Any


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    """Print the welcome banner."""
    banner = r"""
================================================================================
                     __      ____  __  _____
                     \ \    / /  \/  |/ ____|
                      \ \  / /| \  / | |
                       \ \/ / | |\/| | |
                        \  /  | |  | | |____
                         \/   |_|  |_|\_____|

                   VMC TRADING BOT V6 - SETUP WIZARD
================================================================================

    BACKTEST PERFORMANCE (1 Year, $10K Start, 3% Risk):

    +------------------+---------------+
    | Total PnL        | $974,457      |
    | Average Sharpe   | 2.79          |
    | Win Rate         | 31-47%        |
    +------------------+---------------+
    | BTC              | $99,716       |
    | ETH              | $237,340      |
    | SOL              | $485,072      |
    +------------------+---------------+

    15 Optimized Strategies on Hyperliquid Exchange

================================================================================
"""
    print(banner)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70 + "\n")


def print_info(msg: str):
    """Print info message."""
    print(f"  [*] {msg}")


def print_warning(msg: str):
    """Print warning message."""
    print(f"  [!] {msg}")


def print_error(msg: str):
    """Print error message."""
    print(f"  [ERROR] {msg}")


def print_success(msg: str):
    """Print success message."""
    print(f"  [OK] {msg}")


def get_input(prompt: str, default: str = None, required: bool = True) -> str:
    """Get user input with optional default value."""
    if default:
        display_prompt = f"  {prompt} [{default}]: "
    else:
        display_prompt = f"  {prompt}: "

    while True:
        value = input(display_prompt).strip()

        if not value and default:
            return default
        if not value and required:
            print_warning("This field is required. Please enter a value.")
            continue
        return value


def get_hidden_input(prompt: str) -> str:
    """Get password/secret input (hidden)."""
    try:
        import getpass
        value = getpass.getpass(f"  {prompt}: ")
    except:
        print_warning("Hidden input not supported. Input will be visible.")
        value = input(f"  {prompt}: ").strip()
    return value


def get_float(prompt: str, default: float = None, min_val: float = None, max_val: float = None) -> float:
    """Get a float input with validation."""
    while True:
        default_str = str(default) if default is not None else None
        value_str = get_input(prompt, default_str, required=True)
        try:
            value = float(value_str)
            if min_val is not None and value < min_val:
                print_warning(f"Value must be at least {min_val}")
                continue
            if max_val is not None and value > max_val:
                print_warning(f"Value must be at most {max_val}")
                continue
            return value
        except ValueError:
            print_warning("Please enter a valid number.")


def get_int(prompt: str, default: int = None, min_val: int = None, max_val: int = None) -> int:
    """Get an integer input with validation."""
    return int(get_float(prompt, default, min_val, max_val))


def get_yes_no(prompt: str, default: bool = True) -> bool:
    """Get a yes/no input."""
    default_str = "Y" if default else "N"
    while True:
        value = get_input(f"{prompt} (Y/N)", default_str, required=True).upper()
        if value in ['Y', 'YES']:
            return True
        if value in ['N', 'NO']:
            return False
        print_warning("Please enter Y or N.")


def validate_eth_address(address: str) -> bool:
    """Validate Ethereum address format."""
    if not address:
        return False
    if not address.startswith('0x'):
        return False
    if len(address) != 42:
        return False
    try:
        int(address[2:], 16)
        return True
    except ValueError:
        return False


def validate_private_key(key: str) -> bool:
    """Validate private key format."""
    if not key:
        return False
    clean_key = key[2:] if key.startswith('0x') else key
    if len(clean_key) != 64:
        return False
    try:
        int(clean_key, 16)
        return True
    except ValueError:
        return False


def step_wallet_setup() -> Dict[str, Any]:
    """Step 1: Hyperliquid wallet configuration."""
    print_section("STEP 1: HYPERLIQUID WALLET SETUP")

    print("""
    Hyperliquid uses a two-wallet system for API trading:

    1. MAIN WALLET   - Your primary wallet that holds funds (USDC)
    2. API WALLET    - A separate wallet that signs trades

    The API wallet can only trade - it cannot withdraw funds.
    This protects your assets if the API key is compromised.

    TO SET UP:
    1. Go to https://app.hyperliquid.xyz
    2. Connect your wallet and deposit USDC
    3. Go to the API page: https://app.hyperliquid.xyz/API
    4. Click "Generate API Wallet"
    5. SAVE THE PRIVATE KEY - you will only see it once!
    6. Authorize the API wallet to trade

    """)

    input("  Press ENTER when ready to continue...")
    print()

    # Main wallet address
    while True:
        main_address = get_input("Enter your MAIN wallet address (0x...)")
        if validate_eth_address(main_address):
            break
        print_error("Invalid address. Must start with 0x and be 42 characters.")

    # API wallet address
    while True:
        api_address = get_input("Enter your API wallet address (0x...)")
        if validate_eth_address(api_address):
            break
        print_error("Invalid address. Must start with 0x and be 42 characters.")

    # Private key
    print()
    print_warning("SECURITY WARNING:")
    print("  Your private key will be stored in config/config.yaml")
    print("  Keep this file secure and NEVER share it!")
    print()

    while True:
        private_key = get_hidden_input("Enter your API wallet PRIVATE KEY")
        if validate_private_key(private_key):
            # Normalize format
            if not private_key.startswith('0x'):
                private_key = '0x' + private_key
            break
        print_error("Invalid private key. Must be 64 hex characters.")

    # Testnet
    print()
    use_testnet = get_yes_no("Start with TESTNET? (Recommended for first run)", default=True)

    if use_testnet:
        print_info("Testnet mode selected. No real funds at risk.")
        print_info("Get testnet funds at: https://app.hyperliquid.xyz (testnet)")
    else:
        print_warning("MAINNET mode selected. Real funds will be used!")
        confirm = get_yes_no("Are you SURE you want to trade with real funds?", default=False)
        if not confirm:
            use_testnet = True
            print_info("Switched to testnet mode for safety.")

    return {
        'main_address': main_address,
        'api_address': api_address,
        'private_key': private_key,
        'testnet': use_testnet
    }


def step_risk_settings() -> Dict[str, Any]:
    """Step 2: Risk management settings."""
    print_section("STEP 2: RISK SETTINGS")

    print("""
    Configure how much risk you're comfortable with:

    RISK PER TRADE:
      - 1-2%: Conservative (recommended for beginners)
      - 2-3%: Moderate (backtest used 3%)
      - 3-5%: Aggressive (higher risk, higher reward)

    LEVERAGE:
      - 1x: No leverage (safest)
      - 2x: Moderate leverage (recommended)
      - 3-5x: Higher leverage (more risk)

    MAX POSITIONS:
      - Default is 3 (one per asset: BTC, ETH, SOL)

    """)

    risk_percent = get_float(
        "Risk per trade (%)",
        default=2.0,
        min_val=0.5,
        max_val=10.0
    )

    leverage = get_float(
        "Leverage (1-5x)",
        default=2.0,
        min_val=1.0,
        max_val=5.0
    )

    max_positions = get_int(
        "Max simultaneous positions",
        default=3,
        min_val=1,
        max_val=9
    )

    return {
        'risk_percent': risk_percent,
        'leverage': leverage,
        'max_positions': max_positions
    }


def step_strategy_config() -> Dict[str, bool]:
    """Step 3: Strategy selection."""
    print_section("STEP 3: STRATEGY CONFIGURATION")

    print("""
    V6 includes 15 optimized strategies (5 per asset).
    All strategies are ENABLED by default for maximum diversification.

    TOP PERFORMERS:
      - SOL 5m SIMPLE NY_HOURS:     $166,586 | Sharpe 1.26
      - SOL 5m ENHANCED_60:         $152,788 | Sharpe 1.39
      - ETH 5m SIMPLE NY_HOURS:     $78,946  | Sharpe 0.93
      - SOL 1h ENHANCED_70 WKND:    $74,115  | Sharpe 6.88 (Best!)
      - BTC 5m ENHANCED_60:         $54,339  | Sharpe 2.82

    You can disable individual strategies in config/config.yaml later.

    """)

    enable_all = get_yes_no("Enable ALL 15 strategies? (Recommended)", default=True)

    if enable_all:
        print_success("All 15 strategies will be enabled.")
        return {'enable_all': True}

    # Let user choose per asset
    print()
    print("  Select how many strategies per asset (0-5):")
    print()

    btc_count = get_int("  BTC strategies", default=5, min_val=0, max_val=5)
    eth_count = get_int("  ETH strategies", default=5, min_val=0, max_val=5)
    sol_count = get_int("  SOL strategies", default=5, min_val=0, max_val=5)

    return {
        'enable_all': False,
        'btc_count': btc_count,
        'eth_count': eth_count,
        'sol_count': sol_count
    }


def step_discord_setup() -> Dict[str, Any]:
    """Step 4: Discord notifications."""
    print_section("STEP 4: DISCORD NOTIFICATIONS (Optional)")

    print("""
    The bot can send trade notifications to Discord:
      - New signals detected
      - Trades opened (with entry price, size, stop loss)
      - Trades closed (with PnL)
      - Errors and warnings

    TO SET UP:
    1. Open Discord and go to your server
    2. Edit a channel -> Integrations -> Webhooks
    3. Create a new webhook
    4. Copy the webhook URL

    """)

    enable_discord = get_yes_no("Enable Discord notifications?", default=False)

    if not enable_discord:
        return {'enabled': False, 'webhook_url': ''}

    webhook_url = get_input("Discord Webhook URL")

    # Validate URL format
    if not webhook_url.startswith('https://discord.com/api/webhooks/'):
        print_warning("URL doesn't look like a Discord webhook. Continuing anyway.")

    return {
        'enabled': True,
        'webhook_url': webhook_url
    }


def generate_config(
    wallet: Dict[str, Any],
    risk: Dict[str, Any],
    strategies: Dict[str, Any],
    discord: Dict[str, Any]
) -> str:
    """Generate the YAML configuration content."""

    # Determine which strategies to enable
    strategy_enabled = {}
    if strategies.get('enable_all', True):
        # All enabled
        pass
    else:
        # Custom selection - we'll handle this when writing
        btc_count = strategies.get('btc_count', 5)
        eth_count = strategies.get('eth_count', 5)
        sol_count = strategies.get('sol_count', 5)

        btc_strategies = ['btc_5m_enhanced60_ny', 'btc_5m_simple_ny', 'btc_4h_enhanced60_all', 'btc_30m_enhanced70_ny', 'btc_4h_simple_all']
        eth_strategies = ['eth_5m_simple_ny', 'eth_30m_simple_weekends', 'eth_15m_enhanced70_weekends', 'eth_5m_enhanced60_ny', 'eth_30m_enhanced70_ny']
        sol_strategies = ['sol_5m_simple_ny', 'sol_5m_enhanced60_ny', 'sol_1h_enhanced70_weekends', 'sol_15m_simple_ny', 'sol_15m_enhanced60_ny']

        for i, s in enumerate(btc_strategies):
            strategy_enabled[s] = i < btc_count
        for i, s in enumerate(eth_strategies):
            strategy_enabled[s] = i < eth_count
        for i, s in enumerate(sol_strategies):
            strategy_enabled[s] = i < sol_count

    # Read template
    template_path = Path(__file__).parent / "config" / "config_v6_production.yaml"

    if not template_path.exists():
        # Try V6_RELEASE folder
        template_path = Path(__file__).parent / "V6_RELEASE" / "config_v6_production.yaml"

    if not template_path.exists():
        print_error(f"Template config not found at {template_path}")
        return None

    with open(template_path, 'r') as f:
        content = f.read()

    # Replace exchange settings
    content = re.sub(
        r'api_secret: ""',
        f'api_secret: "{wallet["private_key"]}"',
        content,
        count=1
    )
    content = re.sub(
        r'wallet_address: ""',
        f'wallet_address: "{wallet["api_address"]}"',
        content,
        count=1
    )
    content = re.sub(
        r'account_address: ""',
        f'account_address: "{wallet["main_address"]}"',
        content,
        count=1
    )
    content = re.sub(
        r'testnet: true',
        f'testnet: {"true" if wallet["testnet"] else "false"}',
        content,
        count=1
    )

    # Replace risk settings
    content = re.sub(
        r'risk_percent: \d+\.?\d*',
        f'risk_percent: {risk["risk_percent"]}',
        content,
        count=1
    )
    content = re.sub(
        r'leverage: \d+\.?\d*',
        f'leverage: {risk["leverage"]}',
        content,
        count=1
    )
    content = re.sub(
        r'max_positions: \d+',
        f'max_positions: {risk["max_positions"]}',
        content,
        count=1
    )

    # Replace Discord settings
    content = re.sub(
        r'discord:\n  enabled: true',
        f'discord:\n  enabled: {"true" if discord["enabled"] else "false"}',
        content,
        count=1
    )
    if discord['webhook_url']:
        content = re.sub(
            r'webhook_url: ""',
            f'webhook_url: "{discord["webhook_url"]}"',
            content,
            count=1
        )

    # Handle strategy selection if not all enabled
    if strategy_enabled:
        for strategy_name, enabled in strategy_enabled.items():
            # Find the strategy block and update enabled status
            pattern = rf'({strategy_name}:\n\s+enabled:) (true|false)'
            replacement = rf'\1 {"true" if enabled else "false"}'
            content = re.sub(pattern, replacement, content)

    return content


def save_config(content: str) -> Path:
    """Save the configuration to file."""
    config_dir = Path(__file__).parent / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / "config.yaml"

    # Backup existing config
    if config_path.exists():
        backup_path = config_dir / "config.yaml.backup"
        config_path.rename(backup_path)
        print_info(f"Backed up existing config to {backup_path.name}")

    with open(config_path, 'w') as f:
        f.write(content)

    return config_path


def print_summary(wallet: Dict, risk: Dict, strategies: Dict, discord: Dict):
    """Print configuration summary."""
    print_section("CONFIGURATION SUMMARY")

    testnet_status = "TESTNET (safe)" if wallet['testnet'] else "MAINNET (live funds!)"
    discord_status = "Enabled" if discord['enabled'] else "Disabled"

    strategy_count = 15 if strategies.get('enable_all', True) else (
        strategies.get('btc_count', 0) +
        strategies.get('eth_count', 0) +
        strategies.get('sol_count', 0)
    )

    print(f"""
    WALLET CONFIGURATION:
      Main Wallet:   {wallet['main_address'][:10]}...{wallet['main_address'][-6:]}
      API Wallet:    {wallet['api_address'][:10]}...{wallet['api_address'][-6:]}
      Network:       {testnet_status}

    RISK SETTINGS:
      Risk/Trade:    {risk['risk_percent']}%
      Leverage:      {risk['leverage']}x
      Max Positions: {risk['max_positions']}

    STRATEGIES:
      Enabled:       {strategy_count} of 15

    NOTIFICATIONS:
      Discord:       {discord_status}
    """)


def print_next_steps(config_path: Path, testnet: bool):
    """Print next steps after setup."""
    print_section("SETUP COMPLETE!")

    if testnet:
        print("""
    NEXT STEPS:

    1. GET TESTNET FUNDS:
       Go to https://app.hyperliquid.xyz (switch to testnet)
       Get testnet USDC from the faucet

    2. START THE BOT:
       Windows: run_bot.bat
       Linux:   ./run_bot.sh

       Or for production with auto-restart:
       python run_production.py

    3. MONITOR:
       - Check logs/production.log for activity
       - View positions at https://app.hyperliquid.xyz
       - Discord notifications (if enabled)

    4. GO LIVE:
       When ready, edit config/config.yaml:
       Change 'testnet: true' to 'testnet: false'

    """)
    else:
        print("""
    NEXT STEPS:

    1. START THE BOT:
       Windows: run_bot.bat
       Linux:   ./run_bot.sh

       Or for production with auto-restart:
       python run_production.py

    2. MONITOR:
       - Check logs/production.log for activity
       - View positions at https://app.hyperliquid.xyz
       - Discord notifications (if enabled)

    *** WARNING: You are in MAINNET mode. Real funds will be used! ***

    """)

    print(f"    Configuration saved to: {config_path}")
    print()
    print("    To reconfigure, run this wizard again: python setup_wizard.py")
    print()


def main():
    """Main setup wizard flow."""
    try:
        clear_screen()
        print_banner()

        print("  This wizard will guide you through configuring the bot.")
        print("  Press ENTER to start...")
        input()

        # Step 1: Wallet setup
        wallet = step_wallet_setup()

        # Step 2: Risk settings
        risk = step_risk_settings()

        # Step 3: Strategy selection
        strategies = step_strategy_config()

        # Step 4: Discord notifications
        discord = step_discord_setup()

        # Show summary
        clear_screen()
        print_banner()
        print_summary(wallet, risk, strategies, discord)

        # Confirm and save
        if get_yes_no("Save this configuration?", default=True):
            content = generate_config(wallet, risk, strategies, discord)

            if content:
                config_path = save_config(content)
                print_success(f"Configuration saved to {config_path}")
                print_next_steps(config_path, wallet['testnet'])
            else:
                print_error("Failed to generate configuration. Check template file exists.")
        else:
            print()
            print_info("Setup cancelled. No changes made.")

    except KeyboardInterrupt:
        print("\n\n  Setup cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print_error(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
