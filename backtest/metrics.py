"""Performance metrics for backtesting."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pathlib import Path


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    # Returns
    total_return: float
    annualized_return: float
    monthly_returns: List[float]

    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # in bars

    # Trade metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    expectancy: float  # Expected value per trade

    # Consistency
    winning_months: int
    losing_months: int
    best_month: float
    worst_month: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "winning_months": self.winning_months,
            "losing_months": self.losing_months,
        }

    def summary(self) -> str:
        """Generate text summary of metrics."""
        return f"""
========== Performance Summary ==========
Total Return:        {self.total_return:.2f}%
Annualized Return:   {self.annualized_return:.2f}%
Max Drawdown:        {self.max_drawdown:.2f}%

Sharpe Ratio:        {self.sharpe_ratio:.2f}
Sortino Ratio:       {self.sortino_ratio:.2f}
Calmar Ratio:        {self.calmar_ratio:.2f}

Total Trades:        {self.total_trades}
Win Rate:            {self.win_rate:.1f}%
Profit Factor:       {self.profit_factor:.2f}
Expectancy:          ${self.expectancy:.2f}

Avg Win:             ${self.avg_win:.2f}
Avg Loss:            ${self.avg_loss:.2f}

Winning Months:      {self.winning_months}
Losing Months:       {self.losing_months}
Best Month:          {self.best_month:.2f}%
Worst Month:         {self.worst_month:.2f}%
=========================================
"""


def calculate_metrics(
    equity_curve: List[float],
    trades: List[Any],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 2190  # ~6 candles/day * 365
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics.

    Args:
        equity_curve: List of equity values
        trades: List of trade objects with pnl attribute
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year

    Returns:
        PerformanceMetrics object
    """
    equity = pd.Series(equity_curve)
    returns = equity.pct_change().dropna()

    # Total return
    total_return = ((equity.iloc[-1] / equity.iloc[0]) - 1) * 100

    # Annualized return
    n_periods = len(equity)
    annualized_return = ((equity.iloc[-1] / equity.iloc[0]) ** (periods_per_year / n_periods) - 1) * 100

    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(periods_per_year) * 100

    # Sharpe ratio
    excess_returns = returns - (risk_free_rate / periods_per_year)
    sharpe = (excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)) if returns.std() > 0 else 0

    # Sortino ratio (using downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else returns.std()
    sortino = (excess_returns.mean() / downside_std * np.sqrt(periods_per_year)) if downside_std > 0 else 0

    # Max drawdown
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax * 100
    max_dd = drawdown.min()

    # Max drawdown duration
    in_drawdown = drawdown < 0
    dd_groups = (~in_drawdown).cumsum()
    dd_lengths = in_drawdown.groupby(dd_groups).sum()
    max_dd_duration = int(dd_lengths.max()) if len(dd_lengths) > 0 else 0

    # Calmar ratio
    calmar = (annualized_return / abs(max_dd)) if max_dd != 0 else 0

    # Trade metrics
    total_trades = len(trades)
    winners = [t for t in trades if hasattr(t, 'pnl') and t.pnl > 0]
    losers = [t for t in trades if hasattr(t, 'pnl') and t.pnl <= 0]

    win_rate = (len(winners) / total_trades * 100) if total_trades > 0 else 0

    gross_profit = sum(t.pnl for t in winners) if winners else 0
    gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0

    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

    avg_win = (gross_profit / len(winners)) if winners else 0
    avg_loss = (gross_loss / len(losers)) if losers else 0

    # Expectancy (expected value per trade)
    if total_trades > 0:
        win_prob = len(winners) / total_trades
        loss_prob = len(losers) / total_trades
        expectancy = (win_prob * avg_win) - (loss_prob * avg_loss)
    else:
        expectancy = 0

    # Monthly returns (approximate)
    bars_per_month = periods_per_year / 12
    monthly_returns = []
    for i in range(0, len(equity) - 1, int(bars_per_month)):
        end_idx = min(i + int(bars_per_month), len(equity) - 1)
        month_return = ((equity.iloc[end_idx] / equity.iloc[i]) - 1) * 100
        monthly_returns.append(month_return)

    winning_months = len([r for r in monthly_returns if r > 0])
    losing_months = len([r for r in monthly_returns if r <= 0])
    best_month = max(monthly_returns) if monthly_returns else 0
    worst_month = min(monthly_returns) if monthly_returns else 0

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        monthly_returns=monthly_returns,
        volatility=volatility,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_duration,
        total_trades=total_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        expectancy=expectancy,
        winning_months=winning_months,
        losing_months=losing_months,
        best_month=best_month,
        worst_month=worst_month
    )


def plot_equity_curve(
    equity_curve: List[float],
    trades: Optional[List[Any]] = None,
    title: str = "Equity Curve",
    save_path: Optional[str] = None
) -> None:
    """
    Plot equity curve with optional trade markers.

    Args:
        equity_curve: List of equity values
        trades: Optional list of trades
        title: Plot title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Equity curve
    ax1 = axes[0]
    ax1.plot(equity_curve, label='Equity', color='blue', linewidth=1.5)
    ax1.set_title(title)
    ax1.set_ylabel('Equity ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Drawdown
    equity = pd.Series(equity_curve)
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax * 100

    ax2 = axes[1]
    ax2.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
    ax2.set_title('Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Bars')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_monthly_returns(
    monthly_returns: List[float],
    title: str = "Monthly Returns",
    save_path: Optional[str] = None
) -> None:
    """
    Plot monthly returns bar chart.

    Args:
        monthly_returns: List of monthly return percentages
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ['green' if r > 0 else 'red' for r in monthly_returns]
    ax.bar(range(len(monthly_returns)), monthly_returns, color=colors, alpha=0.7)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel('Month')
    ax.set_ylabel('Return (%)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
