"""Simple Telegram notification helper."""
from __future__ import annotations

import os
from typing import Optional

import httpx


def send_telegram_message(message: str, *, token: Optional[str] = None, chat_id: Optional[str] = None) -> None:
    """Send ``message`` via Telegram bot if credentials are configured.

    Parameters
    ----------
    message: The text to send.
    token: Bot token. Falls back to ``TELEGRAM_BOT_TOKEN`` env var.
    chat_id: Chat id. Falls back to ``TELEGRAM_CHAT_ID`` env var.
    """

    token = token or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:  # pragma: no cover - network/credentials issues are ignored
        httpx.post(url, data={"chat_id": chat_id, "text": message}, timeout=10.0)
    except Exception:
        pass
