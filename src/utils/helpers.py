"""Helper functions for VMC Trading Bot."""

from decimal import Decimal, ROUND_DOWN
from typing import Union

Number = Union[int, float, Decimal]


def round_to_tick(value: Number, tick_size: Number) -> float:
    """
    Round a value down to the nearest tick size.

    Args:
        value: The value to round
        tick_size: The tick size to round to

    Returns:
        Rounded value as float
    """
    if tick_size <= 0:
        return float(value)

    decimal_value = Decimal(str(value))
    decimal_tick = Decimal(str(tick_size))

    rounded = (decimal_value / decimal_tick).to_integral_value(rounding=ROUND_DOWN) * decimal_tick
    return float(rounded)


def round_to_precision(value: Number, precision: int) -> float:
    """
    Round a value to a specific decimal precision.

    Args:
        value: The value to round
        precision: Number of decimal places

    Returns:
        Rounded value as float
    """
    decimal_value = Decimal(str(value))
    factor = Decimal(10) ** -precision
    rounded = decimal_value.quantize(factor, rounding=ROUND_DOWN)
    return float(rounded)


def calculate_pnl(
    entry_price: Number,
    exit_price: Number,
    size: Number,
    side: str
) -> float:
    """
    Calculate profit/loss for a trade.

    Args:
        entry_price: Entry price
        exit_price: Exit price
        size: Position size
        side: Trade side ('long' or 'short')

    Returns:
        PnL amount (positive = profit, negative = loss)
    """
    entry = float(entry_price)
    exit_p = float(exit_price)
    sz = float(size)

    if side.lower() == 'long':
        return (exit_p - entry) * sz
    else:  # short
        return (entry - exit_p) * sz


def calculate_pnl_percent(
    entry_price: Number,
    exit_price: Number,
    side: str,
    leverage: Number = 1
) -> float:
    """
    Calculate percentage profit/loss for a trade.

    Args:
        entry_price: Entry price
        exit_price: Exit price
        side: Trade side ('long' or 'short')
        leverage: Leverage used (default 1)

    Returns:
        PnL percentage
    """
    entry = float(entry_price)
    exit_p = float(exit_price)
    lev = float(leverage)

    if side.lower() == 'long':
        return ((exit_p - entry) / entry) * 100 * lev
    else:  # short
        return ((entry - exit_p) / entry) * 100 * lev


def calculate_position_size(
    account_balance: Number,
    risk_percent: Number,
    entry_price: Number,
    stop_loss_price: Number,
    leverage: Number = 1
) -> float:
    """
    Calculate position size based on risk management.

    Args:
        account_balance: Total account balance
        risk_percent: Percentage of account to risk (e.g., 2 for 2%)
        entry_price: Planned entry price
        stop_loss_price: Stop loss price
        leverage: Leverage to use

    Returns:
        Position size in base currency
    """
    balance = float(account_balance)
    risk = float(risk_percent) / 100
    entry = float(entry_price)
    sl = float(stop_loss_price)
    lev = float(leverage)

    # Risk amount in USD
    risk_amount = balance * risk

    # Price difference (risk per unit)
    price_diff = abs(entry - sl)

    if price_diff == 0:
        return 0.0

    # Position size calculation
    # risk_amount = position_size * price_diff
    # But with leverage, we can hold a larger position
    position_size = (risk_amount / price_diff) * lev

    return position_size


def format_price(price: Number, decimals: int = 2) -> str:
    """
    Format price for display.

    Args:
        price: Price value
        decimals: Number of decimal places

    Returns:
        Formatted price string
    """
    return f"{float(price):,.{decimals}f}"


def format_percent(value: Number, decimals: int = 2) -> str:
    """
    Format percentage for display.

    Args:
        value: Percentage value
        decimals: Number of decimal places

    Returns:
        Formatted percentage string with sign
    """
    v = float(value)
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.{decimals}f}%"
