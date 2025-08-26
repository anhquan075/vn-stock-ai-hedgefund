"""Agent responsible for running technical analysis on OHLCV data.

This agent consumes a :class:`pandas.DataFrame` containing historical market
prices and computes a wide range of indicators and candlestick patterns using
the C-backed `TA-Lib` library. The resulting snapshot is then explained via
the reasoning capabilities of the underlying LLM through Agno.
"""

import asyncio
from functools import partial
from textwrap import dedent
from typing import Sequence

import pandas as pd
from agno.tools.reasoning import ReasoningTools  # type: ignore

from utils import technical_analysis as ta_utils
from utils.logging import log_info, log_markdown
from utils.model_factory import build_default_model

# Project imports
from ..base_agent import BaseAgent


def _build_model() -> object:
    return build_default_model()


class AnalysisAgent(BaseAgent):
    """Agent that performs and explains technical analysis."""

    def __init__(self) -> None:
        expected_output = dedent(
            """
            A markdown-formatted technical analysis report:

            # Technical Analysis Report for {symbol}

            ## Latest Indicator Readings
            {indicator_table}

            ## Interpretation
            {analysis_summary}

            ---
            _Report generated on {current_date}_
            """
        )

        super().__init__(
            model=_build_model(),
            tools=[ReasoningTools(add_instructions=False)],
            instructions=dedent(
                """
                You are an expert technical analyst. Provide a clean, concise markdown report with the following sections only:

                ### Market Structure
                - Brief trend description (bullish/bearish/sideways)
                - Price vs SMA_20 and EMA_20

                ### Momentum
                - RSI_14 interpretation (overbought/oversold/neutral)
                - MACD signal (above/below signal) and implication

                ### Volatility
                - Bollinger Bands context (near BB_high/BB_low or mid)

                ### Volume
                - Volume context vs recent average

                ### Key Levels
                - Support: <level>
                - Resistance: <level>

                ### TL;DR
                - Action Bias: BUY | SELL | HOLD
                - Confidence: <0..1>

                Currency: All price levels MUST be expressed in Vietnamese dong with thousands separators and the "₫" suffix (e.g., 75,300 ₫).
                Keep it under 12 bullet points total. Do not include any internal reasoning, steps, tool outputs, or system text.
                """
            ),
            markdown=True,
            name="analysis-agent",
            agent_id="analysis-agent",
            description="LLM-powered technical analysis agent",
            monitoring=False,
            expected_output=expected_output,
            show_tool_calls=True,
            add_datetime_to_instructions=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self, ohlcv: pd.DataFrame, *, indicators: Sequence[str] | None = None, **kwargs
    ) -> str:  # noqa: D401
        """Run technical analysis on *ohlcv* and return markdown explanation.

        Args:
            ohlcv: Historical OHLCV data.
            indicators: Optional subset of indicator names to compute. When
                *None*, a default indicator set (SMA, EMA, RSI, MACD, Bollinger
                Bands) is used.

        Returns
        -------
        str
            A markdown-formatted explanation ready for display or further
            processing by downstream agents.
        """
        # Compute indicators using the utility class
        log_info("[TA] Computing indicators...")
        enriched_df = ta_utils.compute_indicators(ohlcv, indicators=indicators)
        # Convert DataFrame (last row) to markdown table for the LLM
        latest = enriched_df.tail(1).T.reset_index()
        latest.columns = ["Indicator", "Value"]

        # Format values: price-like indicators in VND, others concise floats/ints
        def _is_price_indicator(name: str) -> bool:
            upper = name.upper()
            return (
                upper in {"OPEN", "HIGH", "LOW", "CLOSE"}
                or upper.startswith("SMA")
                or upper.startswith("EMA")
                or upper.startswith("BB")
            )

        def _format_value(indicator: str, value: object) -> str:
            try:
                num = float(value)  # type: ignore[arg-type]
            except Exception:
                return str(value)
            if _is_price_indicator(indicator):
                # VN quotes often come in thousands; convert to VND units without rounding
                vnd = int(num * 1000.0)
                return f"{vnd:,} ₫"
            if indicator.lower().startswith("volume"):
                return f"{int(round(num)):,}"
            # default concise float
            return f"{num:.4f}"

        latest["Value"] = [
            _format_value(indicator, val)
            for indicator, val in zip(
                latest["Indicator"].tolist(), latest["Value"].tolist()
            )
        ]

        table_md = latest.to_markdown(index=False)  # type: ignore[arg-type]

        prompt = (
            "Using the indicator snapshot below, produce ONLY the requested sections in the system instructions.\n"
            "Do NOT include any internal steps or tool metadata.\n\n" + table_md
        )
        log_markdown("## Indicator Snapshot\n\n" + table_md)
        log_info("[TA] Querying analysis model...")
        response = super().run(prompt)
        log_info("[TA] Model response received.")
        content = getattr(response, "content", None)
        text = content if isinstance(content, str) else str(content or "")
        return text

    async def arun(
        self, ohlcv: pd.DataFrame, *, indicators: Sequence[str] | None = None, **kwargs
    ) -> str:
        """Async variant of run using threads for compute and model call."""
        log_info("[TA] Computing indicators (async)...")
        enriched_df = await asyncio.to_thread(
            ta_utils.compute_indicators, ohlcv, indicators=indicators
        )
        latest = enriched_df.tail(1).T.reset_index()
        latest.columns = ["Indicator", "Value"]
        # Use partial to pass keyword argument index=False; avoid treating bool as buf
        table_md = await asyncio.to_thread(partial(latest.to_markdown, index=False))
        log_markdown("## Indicator Snapshot\n\n" + table_md)
        prompt = (
            "Using the indicator snapshot below, produce ONLY the requested sections in the system instructions.\n"
            "Do NOT include any internal steps or tool metadata.\n\n" + table_md
        )
        log_info("[TA] Querying analysis model (async)...")
        response = await asyncio.to_thread(super().run, prompt)
        log_info("[TA] Model response received (async).")
        content = getattr(response, "content", None)
        return content if isinstance(content, str) else str(content or "")
