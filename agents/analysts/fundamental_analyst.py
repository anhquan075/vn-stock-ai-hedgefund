from utils.model_factory import build_default_model

from ..base_agent import BaseAgent
from ..tools import vn_company_overview, vn_finance_report


class FundamentalAnalyst(BaseAgent):
    """Analyzes company financials and fundamentals (Agno Agent)."""

    def __init__(self) -> None:  # noqa: D401
        super().__init__(
            model=build_default_model(),
            tools=[vn_company_overview, vn_finance_report],
            instructions=(
                "You are a fundamentals researcher focused on Vietnamese equities. "
                "Use the provided tools to retrieve company profiles and financial statements. "
                "Write a concise markdown report covering profitability, growth, leverage and cash flow. "
                "Conclude with an overall view: Bullish, Bearish or Neutral, and include a small markdown table of key ratios."
            ),
            name="fundamental-analyst",
            agent_id="fundamental-analyst",
            description="Fundamentals analyst",
            monitoring=False,
        )
