"""Technical analysis helpers powered by TA-Lib.

This module centralises indicator calculations so analyst agents can remain
lightweight.  The heavy lifting is delegated to the C-backed `TA-Lib` Python
bindings, providing both speed and access to an extensive catalogue of
indicators and candlestick pattern recognitions.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
import talib as ta  # type: ignore

# Default indicator set offering a broad technical snapshot.  Additional
# indicators can be requested via the ``indicators`` argument to
# :func:`compute_indicators`.
_DEFAULT_INDICATORS: list[str] = [
    "sma",
    "ema",
    "rsi",
    "macd",
    "bbands",
    "atr",
    "adx",
    "stoch",
    "cci",
    "obv",
    "mfi",
    "roc",
    "williams_r",
    "cmo",
]

# Common candlestick patterns worth highlighting for Vietnamese equities.
_CANDLE_PATTERNS: list[str] = [
    "CDLDOJI",
    "CDLENGULFING",
    "CDLHAMMER",
    "CDLSHOOTINGSTAR",
    "CDLMORNINGSTAR",
    "CDLEVENINGSTAR",
]


def compute_indicators(
    ohlcv: pd.DataFrame,
    *,
    indicators: Sequence[str] | None = None,
    patterns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return *ohlcv* enriched with TA-Lib indicators and candle patterns."""

    if indicators is None:
        indicators = _DEFAULT_INDICATORS
    if patterns is None:
        patterns = _CANDLE_PATTERNS

    df = ohlcv.copy()
    open_, high, low, close, volume = (
        df["Open"].astype(float).values,
        df["High"].astype(float).values,
        df["Low"].astype(float).values,
        df["Close"].astype(float).values,
        df["Volume"].astype(float).values,
    )

    # --- Trend & momentum indicators -------------------------------------------------
    if "sma" in indicators:
        df["SMA_20"] = ta.SMA(close, timeperiod=20)
    if "ema" in indicators:
        df["EMA_20"] = ta.EMA(close, timeperiod=20)
    if "rsi" in indicators:
        df["RSI_14"] = ta.RSI(close, timeperiod=14)
    if "macd" in indicators:
        macd, macdsig, _ = ta.MACD(close)
        df["MACD"] = macd
        df["MACD_signal"] = macdsig
    if "bbands" in indicators:
        upper, middle, lower = ta.BBANDS(close)
        df["BB_high"] = upper
        df["BB_mid"] = middle
        df["BB_low"] = lower
    if "atr" in indicators:
        df["ATR_14"] = ta.ATR(high, low, close, timeperiod=14)
    if "adx" in indicators:
        df["ADX_14"] = ta.ADX(high, low, close, timeperiod=14)
    if "stoch" in indicators:
        k, d = ta.STOCH(high, low, close)
        df["STOCH_%K"] = k
        df["STOCH_%D"] = d
    if "cci" in indicators:
        df["CCI_20"] = ta.CCI(high, low, close, timeperiod=20)
    if "obv" in indicators:
        df["OBV"] = ta.OBV(close, volume)
    if "mfi" in indicators:
        df["MFI_14"] = ta.MFI(high, low, close, volume, timeperiod=14)
    if "roc" in indicators:
        df["ROC_10"] = ta.ROC(close, timeperiod=10)
    if "williams_r" in indicators:
        df["WILLR_14"] = ta.WILLR(high, low, close, timeperiod=14)
    if "cmo" in indicators:
        df["CMO_14"] = ta.CMO(close, timeperiod=14)

    # --- Candlestick pattern recognition ---------------------------------------------
    for pattern in patterns:
        func = getattr(ta, pattern)
        df[pattern] = func(open_, high, low, close)

    return df

