"""Risk manager coordinating conservative, neutral, and aggressive debates."""

from __future__ import annotations

import asyncio

from agno.team import Team
from agno.tools.reasoning import ReasoningTools  # type: ignore

from utils.model_factory import build_default_model
from ..base_agent import BaseAgent


def _text(resp: object) -> str:
    content = getattr(resp, "content", None)
    return content if isinstance(content, str) else str(content or "")


def _make_debater(role: str, instructions: str) -> BaseAgent:
    return BaseAgent(
        model=build_default_model(),
        tools=[ReasoningTools(add_instructions=True)],
        instructions=instructions,
        name=f"{role.lower()}-debater",
        agent_id=f"{role.lower()}-debater",
        description=f"{role} risk debater",
        monitoring=False,
    )


RISK_VIEWS = {
    "Conservative": (
        "You are a conservative risk analyst for Vietnamese equities. "
        "Highlight worst-case scenarios, regulatory or liquidity risks, and position-sizing limits. Provide 2 concise bullets."
    ),
    "Neutral": (
        "You are a neutral risk analyst bridging bullish and bearish views for Vietnamese equities. "
        "Weigh upside versus downside and note catalysts that could shift the balance. Provide 2 concise bullets."
    ),
    "Aggressive": (
        "You are an aggressive risk analyst for Vietnamese markets. "
        "Argue why potential rewards outweigh the dangers and where risks can be hedged. Provide 2 concise bullets."
    ),
}


RISK_PROMPT = (
    "You are the Risk Debate Team (Conservative, Neutral, Aggressive).\n"
    "Work in order: Conservative -> Neutral -> Aggressive. Produce ONE consolidated markdown output with these sections:\n\n"
    "### Risk Debate\n"
    "#### Conservative View\n- bullet 1\n- bullet 2\n\n"
    "#### Neutral View\n- bullet 1\n- bullet 2\n\n"
    "#### Aggressive View\n- bullet 1\n- bullet 2\n\n"
    "#### Summary\n- One sentence jointly summarizing the risk stance.\n\n"
    "Do not include internal steps or tool calls.\n\n"
    "Context:\n{trade}"
)


class RiskManager:
    """Coordinates a risk debate team to assess the trade plan."""

    def __init__(self) -> None:
        self.debaters = [
            _make_debater(role, instr) for role, instr in RISK_VIEWS.items()
        ]
        self.team = Team(
            name="Risk Debate Team",
            members=self.debaters,
            mode="coordinate",
            show_members_responses=True,
            markdown=True,
            telemetry=False,
        )

    async def run(self, trade_plan: str, **kwargs) -> str:
        """Run the risk debate and return structured markdown."""
        prompt = RISK_PROMPT.format(trade=trade_plan)
        resp = await asyncio.to_thread(self.team.run, prompt, **kwargs)
        return _text(resp)
