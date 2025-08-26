"""High-performance backtester using Numba compiled logic.

The backtester implements a simple moving-average crossover strategy.  Core
loops are JIT compiled via Numba which translates Python code into efficient
machine code (LLVM/C), satisfying the requirement for a low-level backend while
retaining a clean Python interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numba import njit


@njit
def _ma_crossover_backtest(close: np.ndarray, fast: int, slow: int, cash: float, fee: float) -> np.ndarray:
    """Return the equity curve for an SMA crossover strategy."""

    n = close.shape[0]
    fast_ma = np.empty(n)
    slow_ma = np.empty(n)
    fast_ma[:] = np.nan
    slow_ma[:] = np.nan

    # rolling means
    for i in range(fast - 1, n):
        fast_ma[i] = close[i - fast + 1 : i + 1].mean()
    for i in range(slow - 1, n):
        slow_ma[i] = close[i - slow + 1 : i + 1].mean()

    equity = cash
    position = 0.0  # number of shares held
    equity_curve = np.empty(n)
    equity_curve[0] = cash

    for i in range(1, n):
        if np.isnan(fast_ma[i - 1]) or np.isnan(slow_ma[i - 1]) or np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]):
            equity_curve[i] = equity
            continue

        # crossover logic
        if position == 0 and fast_ma[i - 1] <= slow_ma[i - 1] and fast_ma[i] > slow_ma[i]:
            position = (equity * (1 - fee)) / close[i]
            equity = position * close[i]
        elif position > 0 and fast_ma[i - 1] >= slow_ma[i - 1] and fast_ma[i] < slow_ma[i]:
            equity = position * close[i] * (1 - fee)
            position = 0.0
        else:
            if position > 0:
                equity = position * close[i]

        equity_curve[i] = equity

    return equity_curve


@dataclass
class BacktestStats:
    final_equity: float
    cagr: float
    max_drawdown: float
    sharpe: float


class BacktestAgent:
    """Run the compiled backtest and compute performance metrics."""

    def run(self, ohlcv: pd.DataFrame, *, strategy_config: dict[str, Any] | None = None) -> dict[str, Any]:
        if not isinstance(ohlcv.index, pd.DatetimeIndex):
            ohlcv = ohlcv.copy()
            ohlcv.index = pd.to_datetime(ohlcv.index)

        cfg = strategy_config or {}
        fast = int(cfg.get("fast", 20))
        slow = int(cfg.get("slow", 50))
        cash = float(cfg.get("cash", 100_000))
        fee = float(cfg.get("commission", 0.001))

        close = ohlcv["Close"].to_numpy(dtype=np.float64)
        equity_curve = _ma_crossover_backtest(close, fast, slow, cash, fee)

        returns = np.diff(equity_curve) / equity_curve[:-1]
        if equity_curve.size > 1:
            cagr = (equity_curve[-1] / equity_curve[0]) ** (252 / (len(equity_curve) - 1)) - 1
        else:
            cagr = 0.0
        drawdown = np.maximum.accumulate(equity_curve) - equity_curve
        max_dd = drawdown.max() / np.maximum.accumulate(equity_curve).max()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0.0

        stats = BacktestStats(
            final_equity=float(equity_curve[-1]),
            cagr=float(cagr),
            max_drawdown=float(max_dd),
            sharpe=float(sharpe),
        )
        return stats.__dict__

