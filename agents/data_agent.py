"""Agent responsible for fetching Vietnamese stock data using *yfinance*.

The agent exposes :func:`fetch` which utilises the ``yfinance.Ticker`` API to
pull historical OHLCV data. The raw ``pandas.DataFrame`` is returned so that
other agents (analysis, backtest, etc.) can consume it.
"""

import asyncio
import re
from datetime import datetime, timedelta
from typing import Any, Literal

import pandas as pd
import yfinance as yf
from vnstock import Quote

from config.settings import settings

from .base_agent import BaseAgent

DEFAULT_INTERVAL: Literal[
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
] = "1d"


SourceType = Literal["yfinance", "vnstock"]


class DataAgent(BaseAgent):
    """Agent for retrieving VN stock data using *yfinance*.

    Example
    -------
    >>> agent = DataAgent()
    >>> df = agent.fetch("VNM", period="1y", interval="1d")
    >>> print(df.head())
    """

    def __init__(self, *, source: SourceType | None = None) -> None:  # noqa: D401  (simple init description)
        super().__init__(
            model=None,
            tools=[],
            instructions="Data fetching agent",
            name="data-agent",
            agent_id="data-agent",
            description="Fetches VN stock data via yfinance",
            monitoring=False,
        )
        self._source: SourceType = (source or settings.DATA_SOURCE).lower()  # type: ignore[assignment]

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    @staticmethod
    def _normalize_symbols(symbol: str) -> tuple[str, str]:
        """Return (vnstock_symbol, yfinance_symbol) from a raw input symbol.

        The function is tolerant of inputs like "$HPG", "hpg", "HPG.VN", etc.
        """
        # Keep only alphanumerics and dots first
        raw = re.sub(r"[^A-Z0-9.]", "", symbol.strip().upper())
        # Strip common VN suffixes if present
        if raw.endswith(".VN"):
            base = raw[:-3]
        elif raw.endswith("VN"):
            base = raw[:-2]
        else:
            base = raw
        # Ensure the base contains only alphanumerics (no residual dots)
        base = re.sub(r"[^A-Z0-9]", "", base)
        return base, f"{base}.VN"

    def fetch(
        self,
        symbol: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        interval: Literal[
            "1m",
            "2m",
            "5m",
            "15m",
            "30m",
            "60m",
            "90m",
            "1h",
            "1d",
            "5d",
            "1wk",
            "1mo",
            "3mo",
        ] = DEFAULT_INTERVAL,
        period: str | None = "1y",
        auto_adjust: bool = True,
        progress: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data for *symbol*.

        Args:
            symbol: Ticker symbol. For Yahoo! Finance, Vietnamese stocks are
                often suffixed with ``.VN`` (e.g. ``VNM.VN``). For vnstock,
                use the bare symbol (e.g. ``VNM``).
            start: Optional start datetime for historical download.
            end: Optional end datetime for historical download.
            interval: Candle timeframe. Defaults to daily.
            period: Look-back period (e.g. ``"6mo"``, ``"1y"``). When *start*
                and *end* are provided, *period* is ignored.
            auto_adjust: Whether to auto-adjust OHLC prices for splits/dividends.
            progress: Display a CLI progress bar during download.
            **kwargs: Additional keyword arguments forwarded to
                :meth:`yfinance.Ticker.history`.

        Returns
        -------
        pandas.DataFrame
            Historical OHLCV data indexed by *DatetimeIndex*.
        """
        if self._source == "vnstock":
            if start is None or end is None:
                raise ValueError(
                    "vnstock source requires explicit start and end datetimes"
                )
            # Map our interval to vnstock interval tokens (string minutes or 1D/1W/1M)
            interval_map: dict[str, str] = {
                "1m": "1",
                "2m": "1",
                "5m": "5",
                "15m": "15",
                "30m": "30",
                "60m": "60",
                "90m": "60",
                "1h": "60",
                "1d": "1D",
                "5d": "1D",
                "1wk": "1W",
                "1mo": "1M",
                "3mo": "3M",
            }
            vn_interval = interval_map.get(interval, "1D")  # type: ignore[arg-type]
            start_str = start.strftime("%Y-%m-%d")
            end_str = end.strftime("%Y-%m-%d")

            vn_symbol, yf_symbol = self._normalize_symbols(symbol)

            df: pd.DataFrame | None = None
            # Use vnstock Quote API
            if Quote is not None:
                try:
                    q = Quote(symbol=vn_symbol, source=settings.VNSTOCK_SOURCE)
                    df = q.history(
                        start=start_str,
                        end=end_str,
                        interval=vn_interval,
                        to_df=True,
                    )
                except Exception:
                    df = None
            if df is None or df.empty:
                # Fallback to Yahoo if vnstock has no data
                ticker = yf.Ticker(yf_symbol)
                # yfinance end is exclusive for daily-and-above intervals
                end_for_yf = end
                if interval in {"1d", "5d", "1wk", "1mo", "3mo"} and end is not None:
                    end_for_yf = end + timedelta(days=1)
                df = ticker.history(
                    start=start,
                    end=end_for_yf,
                    interval=interval,
                    period=None,
                    auto_adjust=auto_adjust,
                    **kwargs,
                )
            # Normalize columns
            col_map = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "time": "Datetime",
                "date": "Datetime",
            }
            df = df.rename(
                columns={k: v for k, v in col_map.items() if k in df.columns}
            )
            # Ensure datetime index
            if "Datetime" in df.columns:
                df["Datetime"] = pd.to_datetime(
                    df["Datetime"], utc=True, errors="coerce"
                )
                df = df.dropna(subset=["Datetime"])  # drop unparsable rows
                df = df.set_index("Datetime")
            # Remove timezone info for consistency
            if isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = df.index.tz_convert(None)
                except Exception:
                    df.index = df.index.tz_localize(None)
            # Keep only standard columns if available
            for required in ["Open", "High", "Low", "Close", "Volume"]:
                if required not in df.columns:
                    raise ValueError(
                        "vnstock response missing required column: " + required
                    )
        else:
            _, yf_symbol = self._normalize_symbols(symbol)
            ticker = yf.Ticker(yf_symbol)
            # yfinance end is exclusive for daily-and-above intervals
            end_for_yf = end
            if interval in {"1d", "5d", "1wk", "1mo", "3mo"} and end is not None:
                end_for_yf = end + timedelta(days=1)
            df = ticker.history(
                start=start,
                end=end_for_yf,
                interval=interval,
                period=None if start or end else period,
                auto_adjust=auto_adjust,
                **kwargs,
            )
        # Ensure sorted index and uniform dtype
        if not df.empty and isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = df.index.tz_convert(None)
            except Exception:
                df.index = df.index.tz_localize(None)
            df = df.sort_index()
        if df.empty:
            raise ValueError(f"No data returned for symbol '{symbol}'.")
        return df

    async def afetch(
        self,
        symbol: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        interval: Literal[
            "1m",
            "2m",
            "5m",
            "15m",
            "30m",
            "60m",
            "90m",
            "1h",
            "1d",
            "5d",
            "1wk",
            "1mo",
            "3mo",
        ] = DEFAULT_INTERVAL,
        period: str | None = "1y",
        auto_adjust: bool = True,
        progress: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Async wrapper around fetch (runs in thread)."""
        return await asyncio.to_thread(
            self.fetch,
            symbol,
            start=start,
            end=end,
            interval=interval,
            period=period,
            auto_adjust=auto_adjust,
            progress=progress,
            **kwargs,
        )
