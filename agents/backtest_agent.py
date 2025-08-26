"""Lightweight backtester built on vectorbt."""

from __future__ import annotations

from typing import Any

import pandas as pd
import vectorbt as vbt


class BacktestAgent:
    """Run a vectorised backtest over OHLCV data.

    Uses a simple moving-average crossover strategy by default. Parameters can
    be tweaked via ``strategy_config`` which supports ``fast``, ``slow``,
    ``cash`` and ``commission`` keys.
    """

    def run(
        self, ohlcv: pd.DataFrame, *, strategy_config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute the backtest and return key performance statistics."""

        if not isinstance(ohlcv.index, pd.DatetimeIndex):
            ohlcv = ohlcv.copy()
            ohlcv.index = pd.to_datetime(ohlcv.index)

        cfg = strategy_config or {}
        fast = int(cfg.get("fast", 20))
        slow = int(cfg.get("slow", 50))
        cash = float(cfg.get("cash", 100_000))
        commission = float(cfg.get("commission", 0.001))

        price = ohlcv["Close"]
        fast_ma = price.rolling(window=fast).mean()
        slow_ma = price.rolling(window=slow).mean()

        entries = fast_ma > slow_ma
        exits = fast_ma < slow_ma

        portfolio = vbt.Portfolio.from_signals(
            price,
            entries,
            exits,
            init_cash=cash,
            fees=commission,
        )

        return portfolio.stats().to_dict()

