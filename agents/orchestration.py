"""Orchestration layer gluing together all project agents."""

import asyncio
from typing import Any, Sequence

import pandas as pd
from agno.team import Team
from rich.console import Console

from utils import technical_analysis as ta_utils
from utils.logging import log_dataframe, log_info, log_markdown_panel

from .analysts import (
    AnalysisAgent,
    DataAgent,
    FundamentalAnalyst,
    NewsAnalyst,
    SentimentAnalyst,
)
from .researchers import ResearchTeam
from .trading import PortfolioManager, TraderAgent
from .backtest_agent import BacktestAgent


class Orchestrator:
    """High-level façade for the multi-agent workflow."""

    def __init__(self, *, data_source: str | None = None) -> None:  # noqa: D401
        self._data_agent = DataAgent(source=data_source)  # type: ignore[arg-type]

        # Define Analyst Team
        self.analyst_team = Team(
            name="Analyst Team",
            members=[
                FundamentalAnalyst(),
                NewsAnalyst(),
                SentimentAnalyst(),
            ],
            mode="collaborate",
            show_members_responses=True,
            markdown=True,
            telemetry=False,
        )
        self._technical_analyst = AnalysisAgent()

        self._research_team = ResearchTeam()
        self._trader = TraderAgent()
        self._portfolio_manager = PortfolioManager()

    async def run(
        self,
        symbol: str,
        *,
        start: str | None = None,
        end: str | None = None,
        indicators: Sequence[str] | None = None,
        strategy_config: dict[str, Any] | None = None,
    ) -> None:
        """Execute the full multi-agent workflow.

        Shows a single-line spinner for the current step only. We avoid
        percentage-based progress bars because step duration is unknown.
        """

        console = Console()

        # 1. Data Gathering
        with console.status("[bold cyan]1/7 Fetching data...", spinner="dots"):
            log_info("[Orchestrator] Fetching OHLCV data...")
            ohlcv = await asyncio.to_thread(
                self._data_agent.fetch,
                symbol,
                start=pd.to_datetime(start) if start else None,
                end=pd.to_datetime(end) if end else None,
            )
        log_info("[Orchestrator] Data fetched.")

        # 2. Analyst Team Reports (LLM-enabled)
        with console.status("[bold cyan]2/7 Analyst Team...", spinner="dots"):
            log_info("[Orchestrator] Running Analyst Team...")
            analyst_response = await asyncio.to_thread(
                self.analyst_team.run,
                f"Generate a comprehensive analysis for {symbol}.",
            )
            analyst_text_obj = getattr(analyst_response, "content", None)
            analyst_text = (
                analyst_text_obj
                if isinstance(analyst_text_obj, str)
                else str(analyst_response or "")
            )
        log_info("[Orchestrator] Analyst Team completed.")

        # 3. Technical Analyst (needs OHLCV)
        with console.status("[bold cyan]3/7 Technical Analysis...", spinner="dots"):
            technical_report = await self._technical_analyst.arun(
                ohlcv, indicators=indicators
            )
            # Compute numeric anchors for consistency with trader plan
            enriched_df = await asyncio.to_thread(
                ta_utils.compute_indicators, ohlcv, indicators=indicators
            )
            latest_row = enriched_df.iloc[-1]

            def _to_vnd(value: float) -> str:
                vnd = int(float(value) * 1000.0)
                return f"{vnd:,} ₫"

            last_close_vnd = _to_vnd(float(latest_row.get("Close")))
            sma20_vnd = (
                _to_vnd(float(latest_row["SMA_20"]))
                if "SMA_20" in enriched_df.columns
                else "N/A"
            )
            ema20_vnd = (
                _to_vnd(float(latest_row["EMA_20"]))
                if "EMA_20" in enriched_df.columns
                else "N/A"
            )
            bb_high_vnd = (
                _to_vnd(float(latest_row["BB_high"]))
                if "BB_high" in enriched_df.columns
                else "N/A"
            )
            bb_low_vnd = (
                _to_vnd(float(latest_row["BB_low"]))
                if "BB_low" in enriched_df.columns
                else "N/A"
            )

        compiled_research = (
            "## Analyst Team\n"
            f"{analyst_text}"
            "\n\n"
            "## Technical Analysis (Clean)\n"
            f"{technical_report}"
        )
        log_markdown_panel("Analyst Team Reports", compiled_research)

        # 4. Research Team (Agno Team) + Debate
        with console.status("[bold cyan]4/7 Research Team...", spinner="dots"):
            bull_case, bear_case = await self._research_team.run(symbol, ohlcv)
        debate_summary = f"## Bullish Case\n{bull_case}\n\n## Bearish Case\n{bear_case}"
        log_markdown_panel("Researcher Team Debate", debate_summary)

        # 5. Trader decision
        # Provide numeric anchors from technicals to align price levels
        ta_context = (
            "\n\n### Market Anchors (from Technicals)\n"
            f"- Latest Close: {last_close_vnd}\n"
            f"- SMA_20: {sma20_vnd}\n"
            f"- EMA_20: {ema20_vnd}\n"
            f"- Support (BB_low): {bb_low_vnd}\n"
            f"- Resistance (BB_high): {bb_high_vnd}\n"
            "- Constraint: Entry should be within ±1% of Latest Close unless justified; "
            "Stop below Support for BUY and above Resistance for SELL; Target near the opposite band."
        )
        with console.status("[bold cyan]5/7 Trader Plan...", spinner="dots"):
            trade_plan = await asyncio.to_thread(
                self._trader.run, debate_summary + ta_context
            )
            trade_plan_text = (
                trade_plan if isinstance(trade_plan, str) else str(trade_plan or "")
            )
        log_markdown_panel("Trader's Plan", trade_plan_text)

        # 6. Portfolio Manager Final Decision
        with console.status("[bold cyan]6/7 Portfolio Decision...", spinner="dots"):
            final_decision = await asyncio.to_thread(
                self._portfolio_manager.run, trade_plan
            )
        log_markdown_panel("Portfolio Manager's Final Decision", str(final_decision))

        # 7. Backtest the default strategy for reference
        with console.status("[bold cyan]7/7 Backtest...", spinner="dots"):
            stats = await asyncio.to_thread(BacktestAgent().run, ohlcv)
        log_dataframe(pd.DataFrame([stats]), title="Backtest Results")
