"""Agno tool(s) for interacting with the backtest engine.

Provides a simple callable tool that agents can use to run a backtest on an
OHLCV dataset and return key statistics.
"""

from typing import Any, Literal, Tuple

import pandas as pd
from agno.tools import tool
from vnstock import Company, Finance

from config.settings import settings


@tool
def run_backtest_tool(
    ohlcv: pd.DataFrame, *, strategy_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Execute a vectorised backtest on the provided OHLCV data.

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

    engine = BacktestAgent()
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


ReportType = Literal[
    "balance_sheet", "income_statement", "cash_flow", "ratio", "profit_loss"
]
PeriodType = Literal["quarter", "annual", "year"]


def _parse_period_input(
    period: PeriodType | dict[str, Any],
) -> Tuple[PeriodType, dict[str, Any]]:
    """Normalize period input.

    Accepts either a simple period string (``"quarter"`` or ``"annual"``/``"year"``)
    or a mapping that may include additional details such as ``year`` or
    ``quarter``. Returns the mapped period string and any extra keyword
    arguments that should be forwarded to the underlying ``vnstock`` call.
    """

    if isinstance(period, dict):
        if "quarter" in period:
            extra = {k: v for k, v in period.items() if k in {"quarter", "year"}}
            return "quarter", extra
        if "annual" in period:
            return "annual", {"year": period["annual"]}
        if "year" in period:
            return "annual", {"year": period["year"]}
    return period, {}


def vn_finance_report(
    symbol: str,
    report_type: ReportType,
    *,
    period: PeriodType | dict[str, Any] = "annual",
    lang: Literal["vi", "en"] | None = None,
    dropna: bool = True,
    source: str | None = None,
    **kwargs_extra: Any,
) -> dict[str, Any]:
    """Fetch financial statements via vnstock.Finance.

    Args:
        symbol: Ticker symbol, e.g., "VCI".
        report_type: One of balance_sheet | income_statement | cash_flow | ratio.
        period: Either a simple period string or a mapping providing
            ``year``/``quarter`` details. ``"year"`` is accepted and mapped to
            ``"annual"``.
        lang: Optional language code where supported (e.g., "vi" or "en").
        dropna: Whether to drop NA rows/columns if supported by the source.
        source: Data source (VCI|TCBS). Defaults to settings.VNSTOCK_SOURCE.

    Returns:
        A dict with keys: columns, records
    """
    src = (source or settings.VNSTOCK_SOURCE).upper()
    period_val, extra = _parse_period_input(period)
    mapped_period = "annual" if period_val == "year" else period_val

    fin = Finance(symbol=symbol, source=src)
    fn = getattr(fin, report_type)
    kwargs: dict[str, Any] = {"period": mapped_period, "dropna": dropna}
    if lang is not None:
        kwargs["lang"] = lang
    kwargs.update(extra)
    kwargs.update(kwargs_extra)

    df = fn(**kwargs)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return {"columns": list(df.columns), "records": df.to_dict(orient="records")}


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
    df = Company(symbol=symbol, source=src).news(page_size=page_size, page=page)  # type: ignore[arg-type]
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return {"columns": list(df.columns), "records": df.to_dict(orient="records")}


@tool
def vn_news_data(
    symbol: str,
    *,
    page_size: int = 5,  # Reduced default from 15 to 5
    page: int = 0,
    source: str | None = None,
) -> dict[str, Any]:
    """Fetch recent company news with limited size to manage context."""

    src = (source or settings.VNSTOCK_SOURCE).upper()
    df = Company(symbol=symbol, source=src).news(page_size=page_size, page=page)  # type: ignore[arg-type]
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return {"columns": list(df.columns), "records": df.to_dict(orient="records")}


@tool
def vn_sec_filings(
    symbol: str,
    *,
    page_size: int = 5,  # Reduced default from 20 to 5
    page: int = 0,
    source: str | None = None,
) -> dict[str, Any]:
    """Fetch company events/filings with limited size to manage context."""

    src = (source or settings.VNSTOCK_SOURCE).upper()
    df = Company(symbol=symbol, source=src).events(
        page_size=min(page_size, 5), page=page
    )  # type: ignore[arg-type]
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return {"columns": list(df.columns), "records": df.to_dict(orient="records")}


@tool
def vn_financials_as_reported(
    symbol: str,
    report_type: ReportType = "balance_sheet",
    *,
    period: PeriodType | dict[str, Any] = "quarter",
    lang: Literal["vi", "en"] | None = None,
    dropna: bool = True,
    source: str | None = None,
) -> dict[str, Any]:
    """Fetch financial statements as reported via ``vnstock.Finance``.

    Args:
        symbol: Ticker symbol, e.g., "VCI".
        report_type: Type of financial report to fetch. Must be one of:
            'balance_sheet', 'income_statement', 'cash_flow', 'ratio', 'profit_loss'.
        period: Either a simple period string or a mapping providing
            ``year``/``quarter`` details.
        lang: Optional language code where supported (e.g., "vi" or "en").
        dropna: Whether to drop NA rows/columns if supported by the source.
        source: Data source (VCI|TCBS). Defaults to settings.VNSTOCK_SOURCE.

    Returns:
        A dict with keys: columns, records
    """
    extra_kwargs: dict[str, Any] = {}
    if isinstance(period, dict):
        period_val, extra = _parse_period_input(period)
        extra_kwargs.update(extra)
        period = period_val

    return vn_finance_report(
        symbol,
        report_type,
        period=period,
        lang=lang,
        dropna=dropna,
        source=source,
        **extra_kwargs,
    )


@tool
def vn_company_shareholders(
    symbol: str,
    *,
    page_size: int = 10,  # Reduced default from 20 to 10
    page: int = 0,
    source: str | None = None,
) -> dict[str, Any]:
    """Fetch major shareholders with limited size to manage context."""

    src = (source or settings.VNSTOCK_SOURCE).upper()
    df = Company(symbol=symbol, source=src).shareholders(page_size=page_size, page=page)  # type: ignore[arg-type]
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return {"columns": list(df.columns), "records": df.to_dict(orient="records")}


@tool
def vn_finance_ratio(
    symbol: str,
    *,
    period: PeriodType | dict[str, Any] = "annual",
    lang: Literal["vi", "en"] | None = None,
    dropna: bool = True,
    source: str | None = None,
) -> dict[str, Any]:
    """Fetch financial ratios via ``vnstock.Finance.ratio``.

    The ``period`` argument accepts either a simple string (``"quarter``",
    ``"annual"`` or ``"year"``) or a mapping with ``year``/``quarter`` keys to
    specify the exact reporting period.
    """

    src = (source or settings.VNSTOCK_SOURCE).upper()
    fin = Finance(symbol=symbol, source=src)
    period_val, extra = _parse_period_input(period)
    kwargs: dict[str, Any] = {"period": period_val, "dropna": dropna}
    if lang is not None:
        kwargs["lang"] = lang
    kwargs.update(extra)
    df = fin.ratio(**kwargs)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return {"columns": list(df.columns), "records": df.to_dict(orient="records")}
