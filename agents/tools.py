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


ReportType = Literal["balance_sheet", "income_statement", "cash_flow", "ratio"]
PeriodType = Literal["quarter", "annual", "year"]


def _parse_period_input(period: PeriodType | dict[str, Any]) -> Tuple[PeriodType, dict[str, Any]]:
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


@tool
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


@tool
def vn_news_data(
    symbol: str,
    *,
    page_size: int = 15,
    page: int = 0,
    source: str | None = None,
) -> dict[str, Any]:
    """Alias for :func:`vn_company_news` for compatibility with TradingAgents."""

    return vn_company_news(
        symbol, page_size=page_size, page=page, source=source
    )


@tool
def vn_sec_filings(
    symbol: str,
    *,
    page_size: int = 20,
    page: int = 0,
    source: str | None = None,
) -> dict[str, Any]:
    """Fetch company events/filings via ``vnstock.Company.events``."""

    src = (source or settings.VNSTOCK_SOURCE).upper()
    df = Company(symbol=symbol, source=src).events(
        page_size=page_size, page=page
    )  # type: ignore[arg-type]
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return {"columns": list(df.columns), "records": df.to_dict(orient="records")}


@tool
def vn_financials_as_reported(
    symbol: str,
    report_type: ReportType | dict[str, Any],
    *,
    period: PeriodType | dict[str, Any] = "quarter",
    lang: Literal["vi", "en"] | None = None,
    dropna: bool = True,
    source: str | None = None,
) -> dict[str, Any]:
    """Fetch financial statements as reported via ``vnstock.Finance``.

    ``report_type`` may be provided either as a simple string or as a mapping
    with keys ``type`` and ``period``. Likewise ``period`` itself may be a
    mapping (e.g., ``{"quarter": 1, "year": 2023}``) to request a specific
    reporting window.
    """

    extra_kwargs: dict[str, Any] = {}
    if isinstance(report_type, dict):
        period_info = report_type.get("period")
        report_type = report_type.get("type", "balance_sheet")
        if period_info is not None:
            period, extra = _parse_period_input(period_info)
            extra_kwargs.update(extra)
    if isinstance(period, dict):
        period, extra = _parse_period_input(period)
        extra_kwargs.update(extra)

    return vn_finance_report(
        symbol,
        report_type,  # type: ignore[arg-type]
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
    page_size: int = 20,
    page: int = 0,
    source: str | None = None,
) -> dict[str, Any]:
    """Fetch major shareholders via ``vnstock.Company.shareholders``."""

    src = (source or settings.VNSTOCK_SOURCE).upper()
    df = Company(symbol=symbol, source=src).shareholders(
        page_size=page_size, page=page
    )  # type: ignore[arg-type]
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
