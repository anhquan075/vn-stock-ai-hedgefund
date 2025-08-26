from agno.tools.googlesearch import GoogleSearchTools

from utils.model_factory import build_default_model

from ..base_agent import BaseAgent
from ..tools import vn_insider_sentiment


class SentimentAnalyst(BaseAgent):
    """Analyzes social media sentiment using web searches."""

    def __init__(self) -> None:  # noqa: D401
        super().__init__(
            model=build_default_model(),
            tools=[vn_insider_sentiment, GoogleSearchTools(fixed_language="vi")],
            instructions=(
                "You are a sentiment analyst monitoring Vietnamese social media and forums (Facebook, Reddit, Voz, etc.). "
                "Search these sources to gauge public mood on the ticker, incorporate insider trading sentiment, "
                "highlight prevailing themes and classify sentiment as Bullish, Bearish or Neutral. "
                "End with a small markdown table of source and tone."
            ),
            name="sentiment-analyst",
            agent_id="sentiment-analyst",
            description="Sentiment analyst",
            monitoring=False,
        )
