"""Time filter utility for V6 strategy time-based trading restrictions.

V6 supports three time filter modes:
- ALL_HOURS: Trade any time (no restriction)
- NY_HOURS_ONLY: Only trade during NY market hours (14:00-21:00 UTC)
- WEEKENDS_ONLY: Only trade on weekends (Saturday/Sunday)
"""

from datetime import datetime, timezone
from typing import Optional

from config.config import TimeFilterType, StrategyTimeFilterConfig


# NY market hours in UTC (9:30 AM - 4:00 PM EST = 14:30 - 21:00 UTC)
# We use 14:00 - 21:00 for simplicity
NY_HOURS_START_UTC = 14  # 14:00 UTC = 9:00 AM EST
NY_HOURS_END_UTC = 21    # 21:00 UTC = 4:00 PM EST


def is_ny_market_hours(dt: Optional[datetime] = None) -> bool:
    """Check if the given time is during NY market hours.

    NY market hours: 14:00 - 21:00 UTC (9:00 AM - 4:00 PM EST)

    Args:
        dt: Datetime to check (default: current UTC time)

    Returns:
        True if within NY market hours
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    elif dt.tzinfo is None:
        # Assume UTC if no timezone
        dt = dt.replace(tzinfo=timezone.utc)

    hour = dt.hour
    return NY_HOURS_START_UTC <= hour < NY_HOURS_END_UTC


def is_weekend(dt: Optional[datetime] = None) -> bool:
    """Check if the given time is on a weekend.

    Weekend: Saturday (5) or Sunday (6)

    Args:
        dt: Datetime to check (default: current UTC time)

    Returns:
        True if Saturday or Sunday
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # Monday = 0, Sunday = 6
    return dt.weekday() >= 5  # Saturday (5) or Sunday (6)


def should_trade_now(time_filter: StrategyTimeFilterConfig, dt: Optional[datetime] = None) -> bool:
    """Check if trading is allowed based on the time filter configuration.

    Args:
        time_filter: Time filter configuration for the strategy
        dt: Datetime to check (default: current UTC time)

    Returns:
        True if trading is allowed at this time
    """
    # If time filter is disabled, always allow trading
    if not time_filter.enabled:
        return True

    if dt is None:
        dt = datetime.now(timezone.utc)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    mode = time_filter.mode

    if mode == TimeFilterType.ALL_HOURS:
        return True
    elif mode == TimeFilterType.NY_HOURS_ONLY:
        return is_ny_market_hours(dt)
    elif mode == TimeFilterType.WEEKENDS_ONLY:
        return is_weekend(dt)
    else:
        # Unknown mode, default to allow
        return True


def get_time_filter_status(time_filter: StrategyTimeFilterConfig, dt: Optional[datetime] = None) -> str:
    """Get a human-readable status of the time filter.

    Args:
        time_filter: Time filter configuration
        dt: Datetime to check (default: current UTC time)

    Returns:
        Status string describing current filter state
    """
    if not time_filter.enabled:
        return "disabled (trading always allowed)"

    if dt is None:
        dt = datetime.now(timezone.utc)

    mode = time_filter.mode
    can_trade = should_trade_now(time_filter, dt)

    if mode == TimeFilterType.ALL_HOURS:
        return "all_hours (trading allowed)"
    elif mode == TimeFilterType.NY_HOURS_ONLY:
        is_ny = is_ny_market_hours(dt)
        status = "active" if is_ny else "inactive"
        return f"ny_hours_only ({status}, NY hours: {NY_HOURS_START_UTC}:00-{NY_HOURS_END_UTC}:00 UTC)"
    elif mode == TimeFilterType.WEEKENDS_ONLY:
        is_wknd = is_weekend(dt)
        status = "active (weekend)" if is_wknd else "inactive (weekday)"
        return f"weekends_only ({status})"
    else:
        return f"unknown mode: {mode}"


def get_next_trading_window(time_filter: StrategyTimeFilterConfig, dt: Optional[datetime] = None) -> Optional[datetime]:
    """Calculate when the next trading window starts.

    Args:
        time_filter: Time filter configuration
        dt: Current time (default: now UTC)

    Returns:
        Datetime of next trading window start, or None if always allowed
    """
    if not time_filter.enabled or time_filter.mode == TimeFilterType.ALL_HOURS:
        return None  # Always allowed

    if dt is None:
        dt = datetime.now(timezone.utc)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    if should_trade_now(time_filter, dt):
        return None  # Already in trading window

    mode = time_filter.mode

    if mode == TimeFilterType.NY_HOURS_ONLY:
        # Next NY hours start
        if dt.hour >= NY_HOURS_END_UTC:
            # After NY close, next window is tomorrow
            next_day = dt.replace(hour=NY_HOURS_START_UTC, minute=0, second=0, microsecond=0)
            from datetime import timedelta
            next_day += timedelta(days=1)
            return next_day
        else:
            # Before NY open, next window is today
            return dt.replace(hour=NY_HOURS_START_UTC, minute=0, second=0, microsecond=0)

    elif mode == TimeFilterType.WEEKENDS_ONLY:
        # Next weekend (Saturday)
        days_until_saturday = (5 - dt.weekday()) % 7
        if days_until_saturday == 0 and dt.weekday() != 5:
            days_until_saturday = 7  # It's not Saturday, wait for next one
        from datetime import timedelta
        next_saturday = dt + timedelta(days=days_until_saturday)
        return next_saturday.replace(hour=0, minute=0, second=0, microsecond=0)

    return None
