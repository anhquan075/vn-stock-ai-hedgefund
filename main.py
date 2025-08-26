import argparse
import asyncio

from agents.orchestration import Orchestrator


def main():
    """Entry point for running the multi-agent pipeline from the CLI."""

    parser = argparse.ArgumentParser(
        description="Multi-agent VN stock analysis & backtest"
    )
    parser.add_argument("symbol", help="Ticker symbol, e.g. VNM.VN or VIC.VN")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--source",
        choices=["yfinance", "vnstock"],
        help="Data source provider",
        default="vnstock",
    )
    args = parser.parse_args()

    orchestrator = Orchestrator(data_source=args.source)
    asyncio.run(orchestrator.run(symbol=args.symbol, start=args.start, end=args.end))


if __name__ == "__main__":
    main()
