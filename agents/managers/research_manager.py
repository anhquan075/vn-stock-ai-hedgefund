"""High-level research manager orchestrating analyst synthesis and debate."""

from __future__ import annotations

import asyncio

import pandas as pd
from agno.team import Team

from utils import technical_analysis as ta_utils
from utils.logging import log_info
from ..researchers.research_team import (
    BearishResearcher,
    BullishResearcher,
    FundamentalsAgent,
    NewsAgent,
    SentimentAgent,
    SocialMediaAgent,
    TechnicalResearchAgent,
)


def _text(resp: object) -> str:
    content = getattr(resp, "content", None)
    return content if isinstance(content, str) else str(content or "")


ANALYSTS = [
    FundamentalsAgent(),
    SentimentAgent(),
    NewsAgent(),
    TechnicalResearchAgent(),
    SocialMediaAgent(),
]

DEBATERS = [BullishResearcher(), BearishResearcher()]


TEAM_PROMPT = (
    "You are a financial research team analyzing {symbol}.\n"
    "Coordinate among members to produce a concise synthesis with this exact structure (markdown):\n\n"
    "### Synthesis\n"
    "- Fundamentals: <2 short bullets>\n"
    "- Sentiment: <2 short bullets>\n"
    "- News/Catalysts: <2 short bullets>\n"
    "- Technicals: <2 short bullets>\n"
    "- Social Media: <2 short bullets>\n\n"
    "### Key Risks\n"
    "- bullet\n- bullet\n\n"
    "### Watchlist\n"
    "- bullet\n- bullet\n\n"
    "Do not include any internal steps or tool metadata.\n\n"
    "Technical Snapshot (latest):\n{tech}"
)

DEBATE_PROMPT = (
    "You are the Research Debate Team (Bullish Researcher, Bearish Researcher).\n"
    "Work in order: Bullish -> Bearish. Produce ONE consolidated markdown output with these sections:\n\n"
    "## Bullish Case\n"
    "- bullet 1\n- bullet 2\n- bullet 3\n\n"
    "## Bearish Case\n"
    "- bullet 1\n- bullet 2\n- bullet 3\n\n"
    "Do not include internal steps or tool calls.\n\n"
    "Context:\n{report}"
)


def _split_cases(text: str) -> tuple[str, str]:
    bull_marker, bear_marker = "## Bullish Case", "## Bearish Case"
    if bull_marker in text and bear_marker in text:
        bull = text.split(bull_marker, 1)[1].split(bear_marker)[0].strip()
        bear = text.split(bear_marker, 1)[1].strip()
        return bull, bear
    return text, ""


class ResearchManager:
    """Coordinates analyst synthesis and bull/bear debate using Agno Teams."""

    def __init__(self) -> None:
        self.analyst_team = Team(
            name="Research Team",
            mode="coordinate",
            members=ANALYSTS,
            show_members_responses=False,
            markdown=True,
            telemetry=False,
        )
        self.debate_team = Team(
            name="Research Debate Team",
            mode="coordinate",
            members=DEBATERS,
            show_members_responses=True,
            markdown=True,
            telemetry=False,
        )

    async def run(self, symbol: str, ohlcv: pd.DataFrame) -> tuple[str, str]:
        """Run synthesis then debate, returning bull and bear cases."""
        log_info("[ResearchManager] Computing technical snapshot for context...")
        enriched = await asyncio.to_thread(ta_utils.compute_indicators, ohlcv)
        latest = enriched.tail(1).T.reset_index()
        latest.columns = ["Indicator", "Value"]
        tech_table = latest.to_markdown(index=False)  # type: ignore[arg-type]

        prompt = TEAM_PROMPT.format(symbol=symbol, tech=tech_table)
        synth = _text(await asyncio.to_thread(self.analyst_team.run, prompt))
        report = f"--- Team Synthesis ---\n{synth}\n\n--- Technicals (latest) ---\n{tech_table}"

        debate = DEBATE_PROMPT.format(report=report)
        result = _text(await asyncio.to_thread(self.debate_team.run, debate))
        return _split_cases(result)
