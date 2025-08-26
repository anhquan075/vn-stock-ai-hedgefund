from agno.tools.googlesearch import GoogleSearchTools

from utils.model_factory import build_default_model

from ..base_agent import BaseAgent
from ..tools import vn_company_news


class NewsAnalyst(BaseAgent):
    """Analyzes news articles for sentiment and impact."""

    def __init__(self) -> None:  # noqa: D401
        super().__init__(
            model=build_default_model(),
            tools=[vn_company_news, GoogleSearchTools(fixed_language="vi")],
            instructions=(
                "You are a news researcher covering Vietnamese equities. "
                "Review recent company news and broader macro headlines. "
                "Summarize key catalysts and note whether each is bullish or bearish for the ticker. "
                "Finish with a short markdown table of headline and sentiment."
            ),
            name="news-analyst",
            agent_id="news-analyst",
            description="News analyst",
            monitoring=False,
        )
