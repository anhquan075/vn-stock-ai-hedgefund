"""Agno tool(s) for interacting with the backtest engine.

Provides a simple callable tool that agents can use to run a backtest on an
OHLCV dataset and return key statistics.
"""

from typing import Any, Literal

import pandas as pd
from agno.tools import tool
from vnstock import Company, Finance

from config.settings import settings

from .strategy_agent import StrategyAgent


@tool
def run_backtest_tool(
    ohlcv: pd.DataFrame, *, strategy_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Execute a backtest on the provided OHLCV data.

    Args:
        ohlcv: Historical OHLCV data with columns ``Open, High, Low, Close, Volume``.
        strategy_config: Optional configuration for the strategy factory.

    Returns:
        Backtest statistics as a dictionary
        (e.g., CAGR, max drawdown, Sharpe ratio, win rate).
    """
    from .backtest_agent import (
        BacktestAgent,  # local import to avoid heavy deps at import time
    )

    engine = BacktestAgent(strategy_agent=None)
    return engine.run(ohlcv, strategy_config=strategy_config)


@tool
def vn_company_overview(symbol: str, source: str | None = None) -> dict[str, Any]:
    """Fetch company overview via vnstock.Company.

    Args:
        symbol: Ticker symbol, e.g., "ACB".
        source: Data source (VCI|TCBS|MSN). Defaults to settings.VNSTOCK_SOURCE.

    Returns:
        A dict with keys: columns, records
    """
    src = (source or settings.VNSTOCK_SOURCE).upper()
    df = Company(symbol=symbol, source=src).overview(to_df=True)  # type: ignore[arg-type]
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return {"columns": list(df.columns), "records": df.to_dict(orient="records")}


ReportType = Literal["balance_sheet", "income_statement", "cash_flow", "ratio"]
PeriodType = Literal["quarter", "annual", "year"]


@tool
def vn_finance_report(
    symbol: str,
    report_type: ReportType,
    *,
    period: PeriodType = "annual",
    lang: Literal["vi", "en"] | None = None,
    dropna: bool = True,
    source: str | None = None,
) -> dict[str, Any]:
    """Fetch financial statements via vnstock.Finance.

    Args:
        symbol: Ticker symbol, e.g., "VCI".
        report_type: One of balance_sheet | income_statement | cash_flow | ratio.
        period: "quarter" | "annual" ("year" is accepted and mapped to "annual").
        lang: Optional language code where supported (e.g., "vi" or "en").
        dropna: Whether to drop NA rows/columns if supported by the source.
        source: Data source (VCI|TCBS). Defaults to settings.VNSTOCK_SOURCE.

    Returns:
        A dict with keys: columns, records
    """
    src = (source or settings.VNSTOCK_SOURCE).upper()
    mapped_period = "annual" if period == "year" else period

    fin = Finance(symbol=symbol, source=src)
    fn = getattr(fin, report_type)
    kwargs: dict[str, Any] = {"period": mapped_period}
    if lang is not None:
        kwargs["lang"] = lang
    kwargs["dropna"] = dropna

    df = fn(**kwargs)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return {"columns": list(df.columns), "records": df.to_dict(orient="records")}


@tool
def vn_company_news(
    symbol: str,
    *,
    page_size: int = 15,
    page: int = 0,
    source: str | None = None,
) -> dict[str, Any]:
    """Fetch recent company news via ``vnstock.Company``.

    Args:
        symbol: Ticker symbol, e.g., "VCB".
        page_size: Number of news items to fetch.
        page: Page index for pagination.
        source: Data source (VCI|TCBS|MSN). Defaults to settings.VNSTOCK_SOURCE.

    Returns:
        A dict with keys ``columns`` and ``records`` representing the news
        table.
    """

    src = (source or settings.VNSTOCK_SOURCE).upper()
    df = Company(symbol=symbol, source=src).news(
        page_size=page_size, page=page
    )  # type: ignore[arg-type]
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return {"columns": list(df.columns), "records": df.to_dict(orient="records")}
