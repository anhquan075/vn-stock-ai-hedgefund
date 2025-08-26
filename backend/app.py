from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vnstock import Vnstock

from agents.backtest_agent import BacktestAgent

app = FastAPI(title="VN Stock Backtester")


class BacktestRequest(BaseModel):
    symbol: str
    start: str
    end: str
    fast: int | None = 20
    slow: int | None = 50
    cash: float | None = 100_000
    commission: float | None = 0.001


def fetch_ohlcv(symbol: str, start: str, end: str):
    client = Vnstock().stock(symbol=symbol)
    df = client.quote.history(start=start, end=end, interval="1D")
    df = df.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    df.set_index("time", inplace=True)
    return df


@app.post("/backtest")
def run_backtest(req: BacktestRequest):
    try:
        ohlcv = fetch_ohlcv(req.symbol, req.start, req.end)
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))

    agent = BacktestAgent()
    stats = agent.run(
        ohlcv,
        strategy_config={
            "fast": req.fast or 20,
            "slow": req.slow or 50,
            "cash": req.cash or 100_000,
            "commission": req.commission or 0.001,
        },
    )
    return stats


@app.get("/")
def health():
    return {"status": "ok"}
