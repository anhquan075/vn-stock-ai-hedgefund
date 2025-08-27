"""Factory utilities for constructing Agno model instances from environment.

Supported providers via ``AGNO_MODEL_PROVIDER``:
- openai (default) → uses ``AGNO_MODEL_ID`` (default: ``gpt-4o-mini``)
- anthropic → uses ``AGNO_MODEL_ID`` (default: ``claude-3-5-sonnet-20240620``)
- groq → uses ``AGNO_MODEL_ID`` (default: ``llama-3.1-70b-versatile``)
"""

from importlib import import_module

from config.settings import settings


def build_default_model() -> object:
    """Return a configured Agno model instance based on environment vars."""
    provider = settings.AGNO_MODEL_PROVIDER.lower()
    model_id = settings.AGNO_MODEL_ID

    if provider == "anthropic":
        Claude = import_module("agno.models.anthropic").Claude  # type: ignore[attr-defined]
        return Claude(id=model_id or "claude-3-5-sonnet-20240620")
    if provider == "groq":
        Groq = import_module("agno.models.groq").Groq  # type: ignore[attr-defined]
        return Groq(id=model_id or "llama-3.1-70b-versatile")
    OpenAIChat = import_module("agno.models.openai").OpenAIChat  # type: ignore[attr-defined]
    return OpenAIChat(id=model_id or "gpt-4o", api_key=settings.OPENAI_API_KEY)
