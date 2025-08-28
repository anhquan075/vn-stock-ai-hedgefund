"""Long/short portfolio backtester using the project's multi-agent workflow."""

import argparse
import asyncio
import re
from datetime import datetime, timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd

from agents.data_agent import DataAgent
from agents.researchers.research_team import ResearchTeam
from agents.trading.decision_team import DecisionTeam


def get_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily price data via DataAgent with tolerant symbol handling."""

    data_agent = DataAgent()
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    try:
        df = data_agent.fetch(ticker, start=start_dt, end=end_dt, interval="1d")
    except Exception:
        return pd.DataFrame()
    return df


def _parse_action(markdown: str) -> str:
    match = re.search(r"Action:\s*(BUY|SELL|HOLD)", markdown, re.IGNORECASE)
    return match.group(1).lower() if match else "hold"


def multi_agent_decision_agent(
    *, tickers: list[str], start_date: str, end_date: str, portfolio: dict, **_: Any
) -> dict[str, dict[str, Any]]:
    data_agent = DataAgent()
    research_team = ResearchTeam()
    decision_team = DecisionTeam()

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    decisions: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        try:
            ohlcv = data_agent.fetch(ticker, start=start_dt, end=end_dt)
            if ohlcv.empty:
                raise ValueError
            bull, bear = asyncio.run(research_team.run(ticker, ohlcv))
            decision_md = asyncio.run(decision_team.run(bull, bear))
            act = _parse_action(decision_md)
        except Exception:
            act = "hold"

        pos = portfolio["positions"][ticker]
        if act == "buy" and pos["short"]:
            act = "cover"
        elif act == "sell" and not pos["long"]:
            act = "short"
        decisions[ticker] = {"action": act, "quantity": 1 if act != "hold" else 0}

    return {"decisions": decisions, "analyst_signals": {}}


class Backtester:
    def __init__(
        self,
        agent: Callable[..., dict[str, dict[str, Any]]],
        tickers: list[str],
        start_date: str,
        end_date: str,
        initial_capital: float,
        initial_margin_requirement: float = 0.0,
    ) -> None:
        self.agent = agent
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = float(initial_capital)

        self.portfolio = {
            "cash": float(initial_capital),
            "margin_used": 0.0,
            "margin_requirement": float(initial_margin_requirement),
            "positions": {
                t: {
                    "long": 0,
                    "short": 0,
                    "long_cost_basis": 0.0,
                    "short_cost_basis": 0.0,
                    "short_margin_used": 0.0,
                }
                for t in tickers
            },
            "realized_gains": {t: {"long": 0.0, "short": 0.0} for t in tickers},
        }

        self.portfolio_values: list[dict[str, Any]] = []

    def execute_trade(
        self, ticker: str, action: str, quantity: float, current_price: float
    ) -> int:
        """Update the portfolio for buy, sell, short or cover actions."""

        if quantity <= 0:
            return 0

        quantity = int(quantity)
        position = self.portfolio["positions"][ticker]

        if action == "buy":
            cost = quantity * current_price
            if cost > self.portfolio["cash"]:
                quantity = int(self.portfolio["cash"] / current_price)
                cost = quantity * current_price
            if quantity <= 0:
                return 0

            old_shares = position["long"]
            old_cost = position["long_cost_basis"] * old_shares
            new_total = old_shares + quantity
            position["long_cost_basis"] = (old_cost + cost) / new_total
            position["long"] = new_total
            self.portfolio["cash"] -= cost
            return quantity

        if action == "sell":
            quantity = min(quantity, position["long"])
            if quantity <= 0:
                return 0
            avg_cost = position["long_cost_basis"]
            realized = (current_price - avg_cost) * quantity
            position["long"] -= quantity
            self.portfolio["cash"] += quantity * current_price
            self.portfolio["realized_gains"][ticker]["long"] += realized
            if position["long"] == 0:
                position["long_cost_basis"] = 0.0
            return quantity

        if action == "short":
            proceeds = current_price * quantity
            margin = proceeds * self.portfolio["margin_requirement"]
            if margin > self.portfolio["cash"]:
                max_q = int(
                    self.portfolio["cash"]
                    / (current_price * self.portfolio["margin_requirement"])
                )
                quantity = max_q
                proceeds = current_price * quantity
                margin = proceeds * self.portfolio["margin_requirement"]
            if quantity <= 0:
                return 0
            old_shares = position["short"]
            old_cost = position["short_cost_basis"] * old_shares
            new_total = old_shares + quantity
            position["short_cost_basis"] = (
                old_cost + current_price * quantity
            ) / new_total
            position["short"] = new_total
            position["short_margin_used"] += margin
            self.portfolio["margin_used"] += margin
            self.portfolio["cash"] += proceeds - margin
            return quantity

        if action == "cover":
            quantity = min(quantity, position["short"])
            if quantity <= 0:
                return 0
            cover_cost = quantity * current_price
            avg_short = position["short_cost_basis"]
            realized = (avg_short - current_price) * quantity

            portion = quantity / position["short"] if position["short"] > 0 else 1.0
            margin_release = portion * position["short_margin_used"]

            position["short"] -= quantity
            position["short_margin_used"] -= margin_release
            self.portfolio["margin_used"] -= margin_release
            self.portfolio["cash"] += margin_release - cover_cost
            self.portfolio["realized_gains"][ticker]["short"] += realized

            if position["short"] == 0:
                position["short_cost_basis"] = 0.0
                position["short_margin_used"] = 0.0
            return quantity

        return 0

    def calculate_portfolio_value(self, current_prices: dict[str, float]) -> float:
        total = self.portfolio["cash"]
        for t in self.tickers:
            pos = self.portfolio["positions"][t]
            price = current_prices[t]
            total += pos["long"] * price
            if pos["short"] > 0:
                total -= pos["short"] * price
        return total

    def run_backtest(self) -> dict[str, float]:
        dates = pd.date_range(self.start_date, self.end_date, freq="B")
        if dates.empty:
            raise ValueError("No trading days in the specified range")

        self.portfolio_values = [
            {"Date": dates[0], "Portfolio Value": self.initial_capital}
        ]

        for current_date in dates:
            lookback_start = (current_date - timedelta(days=30)).strftime("%Y-%m-%d")
            current_str = current_date.strftime("%Y-%m-%d")
            prev_str = (current_date - timedelta(days=1)).strftime("%Y-%m-%d")

            current_prices: dict[str, float] = {}
            missing = False
            for t in self.tickers:
                data = get_price_data(t, prev_str, current_str)
                if data.empty:
                    missing = True
                    break
                # DataAgent ensures canonical 'Close' column
                current_prices[t] = float(data.iloc[-1]["Close"])
            if missing:
                continue

            output = self.agent(
                tickers=self.tickers,
                start_date=lookback_start,
                end_date=current_str,
                portfolio=self.portfolio,
            )
            decisions = output.get("decisions", {})

            for t in self.tickers:
                decision = decisions.get(t, {"action": "hold", "quantity": 0})
                self.execute_trade(
                    t,
                    decision.get("action", "hold"),
                    decision.get("quantity", 0),
                    current_prices[t],
                )

            total_value = self.calculate_portfolio_value(current_prices)
            self.portfolio_values.append(
                {"Date": current_date, "Portfolio Value": total_value}
            )

        performance = self._update_performance_metrics()
        return performance

    def _update_performance_metrics(self) -> dict[str, float]:
        df = pd.DataFrame(self.portfolio_values).set_index("Date")
        df["Daily Return"] = df["Portfolio Value"].pct_change()
        returns = df["Daily Return"].dropna()
        metrics: dict[str, float] = {}

        if not returns.empty:
            daily_rf = 0.0434 / 252
            excess = returns - daily_rf
            mean = excess.mean()
            std = excess.std()
            metrics["sharpe_ratio"] = (
                float(np.sqrt(252) * (mean / std)) if std > 1e-12 else 0.0
            )
            neg = excess[excess < 0]
            if len(neg) > 0:
                d_std = neg.std()
                metrics["sortino_ratio"] = (
                    float(np.sqrt(252) * (mean / d_std))
                    if d_std > 1e-12
                    else float("inf")
                )
            else:
                metrics["sortino_ratio"] = float("inf") if mean > 0 else 0.0
            roll_max = df["Portfolio Value"].cummax()
            drawdown = (df["Portfolio Value"] - roll_max) / roll_max
            metrics["max_drawdown"] = float(drawdown.min() * 100)
        else:
            metrics["sharpe_ratio"] = metrics["sortino_ratio"] = metrics[
                "max_drawdown"
            ] = 0.0

        self.performance_metrics = metrics
        return metrics

    def analyze_performance(self) -> pd.DataFrame:
        if not self.portfolio_values:
            raise RuntimeError("run_backtest must be called before analyze_performance")

        df = pd.DataFrame(self.portfolio_values).set_index("Date")
        start_val = df["Portfolio Value"].iloc[0]
        end_val = df["Portfolio Value"].iloc[-1]
        total_return = (end_val / start_val - 1) * 100

        print("Backtest Performance\n")
        print(f"Start value : {start_val:,.2f}")
        print(f"End value   : {end_val:,.2f}")
        print(f"Total return: {total_return:,.2f}%")
        print(f"Sharpe      : {self.performance_metrics.get('sharpe_ratio', 0.0):.2f}")
        print(f"Sortino     : {self.performance_metrics.get('sortino_ratio', 0.0):.2f}")
        print(f"Max DD      : {self.performance_metrics.get('max_drawdown', 0.0):.2f}%")

        return df


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Run backtest over VN stocks")
    parser.add_argument(
        "--tickers", required=True, help="Comma separated list of tickers"
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--cash", type=float, default=5_000_000, help="Starting capital"
    )
    parser.add_argument(
        "--margin", type=float, default=0.0, help="Margin requirement for shorts"
    )
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    backtester = Backtester(
        agent=multi_agent_decision_agent,
        tickers=tickers,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.cash,
        initial_margin_requirement=args.margin,
    )

    backtester.run_backtest()
    backtester.analyze_performance()
