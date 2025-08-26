"""Vectorbt-powered SMA crossover backtester."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import vectorbt as vbt


@dataclass
class BacktestStats:
    final_equity: float
    total_return: float
    cagr: float
    max_drawdown: float
    sharpe: float
    sortino: float
    win_rate: float


class BacktestAgent:
    """Run an SMA crossover backtest using vectorbt."""

    def run(
        self, ohlcv: pd.DataFrame, *, strategy_config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if not isinstance(ohlcv.index, pd.DatetimeIndex):
            ohlcv = ohlcv.copy()
            ohlcv.index = pd.to_datetime(ohlcv.index)

        cfg = strategy_config or {}
        fast = int(cfg.get("fast", 20))
        slow = int(cfg.get("slow", 50))
        cash = float(cfg.get("cash", 100_000))
        fee = float(cfg.get("commission", 0.001))

        close = ohlcv["Close"]

        fast_ma = vbt.MA.run(close, window=fast).ma
        slow_ma = vbt.MA.run(close, window=slow).ma
        entries = fast_ma > slow_ma
        exits = fast_ma < slow_ma

        pf = vbt.Portfolio.from_signals(
            close, entries, exits, fees=fee, init_cash=cash, freq="1D"
        )

        stats_series = pf.stats()
        trades_stats = pf.trades.stats()

        final_equity = float(stats_series.get("End Value", cash))
        start_value = float(stats_series.get("Start Value", cash))
        periods = len(close)
        cagr = (
            (final_equity / start_value) ** (252 / max(periods - 1, 1)) - 1
            if periods > 1
            else 0.0
        )

        stats = BacktestStats(
            final_equity=final_equity,
            total_return=float(stats_series.get("Total Return [%]", 0.0)),
            cagr=float(cagr * 100),
            max_drawdown=float(stats_series.get("Max Drawdown [%]", 0.0)),
            sharpe=float(stats_series.get("Sharpe Ratio", 0.0)),
            sortino=float(stats_series.get("Sortino Ratio", 0.0)),
            win_rate=float(trades_stats.get("Win Rate [%]", 0.0)),
        )
        return stats.__dict__
