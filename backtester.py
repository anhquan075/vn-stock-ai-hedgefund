from __future__ import annotations

import argparse
from typing import Any

import pandas as pd
from colorama import Fore, Style, init

from vnstock import Company

from config.settings import settings

from agents.backtest_agent import BacktestAgent
from utils.logging import print_backtest_stats
from utils.telegram import send_telegram_message

init(autoreset=True)


def fetch_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily OHLCV data using ``vnstock.Company.history``."""

    src = settings.VNSTOCK_SOURCE.upper()
    df = Company(symbol=symbol, source=src).history(start=start, end=end)
    df = df.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    df.set_index("time", inplace=True)
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run SMA crossover backtest")
    parser.add_argument("symbol", help="Ticker symbol, e.g. VCB")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--fast", type=int, default=20, help="Fast SMA length")
    parser.add_argument("--slow", type=int, default=50, help="Slow SMA length")
    parser.add_argument("--cash", type=float, default=100_000, help="Starting capital")
    parser.add_argument("--commission", type=float, default=0.001, help="Trading fee")
    args = parser.parse_args(argv)

    try:
        ohlcv = fetch_ohlcv(args.symbol, args.start, args.end)
    except Exception as err:  # pragma: no cover - network errors
        print(f"{Fore.RED}Failed to fetch data: {err}{Style.RESET_ALL}")
        return

    agent = BacktestAgent()
    stats: dict[str, Any] = agent.run(
        ohlcv,
        strategy_config={
            "fast": args.fast,
            "slow": args.slow,
            "cash": args.cash,
            "commission": args.commission,
        },
    )

    print(f"\n{Fore.WHITE}{Style.BRIGHT}Backtest results for {args.symbol}{Style.RESET_ALL}\n")
    print_backtest_stats(stats)

    # Telegram notification (optional)
    summary = "\n".join(f"{k}: {v:.2f}" for k, v in stats.items())
    send_telegram_message(f"Backtest {args.symbol}\n{summary}")


if __name__ == "__main__":  # pragma: no cover
    main()
