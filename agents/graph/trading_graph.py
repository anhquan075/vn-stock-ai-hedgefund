"""Simple trading graph using research and risk managers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pandas as pd

from ..decision_team import PortfolioManagerAgent, TraderAgent
from ..managers import ResearchManager, RiskManager


def _text(resp: object) -> str:
    content = getattr(resp, "content", None)
    return content if isinstance(content, str) else str(content or "")


@dataclass
class TradeState:
    """State container flowing through the trading graph."""

    symbol: str
    bull_case: str = ""
    bear_case: str = ""
    trade_plan: str = ""
    risk_debate: str = ""
    final_decision: str = ""


class TradingGraph:
    """Orchestrates research, decision, and risk review."""

    def __init__(self) -> None:  # noqa: D401
        self.research = ResearchManager()
        self.risk = RiskManager()
        self.trader = TraderAgent()
        self.pm = PortfolioManagerAgent()

    async def run(self, symbol: str, ohlcv: pd.DataFrame) -> TradeState:
        """Execute the trading workflow and return accumulated state."""
        state = TradeState(symbol=symbol)

        # Research and debate
        state.bull_case, state.bear_case = await self.research.run(symbol, ohlcv)
        debate = f"## Bullish Case\n{state.bull_case}\n\n## Bearish Case\n{state.bear_case}"

        # Trader plan
        trade_resp = await asyncio.to_thread(self.trader.decide, debate)
        state.trade_plan = _text(trade_resp)

        # Risk debate
        state.risk_debate = await self.risk.run(state.trade_plan)

        # Portfolio manager decision
        pm_input = f"{state.trade_plan}\n\n### Risk Debate\n{state.risk_debate}"
        pm_resp = await asyncio.to_thread(self.pm.approve, pm_input)
        state.final_decision = _text(pm_resp)

        return state
