from __future__ import annotations

from typing import Literal

import google.genai as genai
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from v2a_inspect.clients import DEFAULT_GEMINI_MODEL
from v2a_inspect.workflows import InspectRuntime


def build_genai_client(*, api_key: str | None = None) -> genai.Client:
    """Build a Gemini SDK client for file uploads and file lookup."""

    return genai.Client(api_key=api_key or _require_gemini_api_key())


def build_llm(
    *,
    provider: Literal["gemini", "openai"] = "gemini",
    model: str = DEFAULT_GEMINI_MODEL,
    api_key: str | None = None,
    max_retries: int = 3,
    timeout_seconds: float | None = None,
    temperature: float = 0.1,
) -> BaseChatModel:
    """Build a LangChain chat model for the given provider."""

    if provider == "openai":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key or _require_openai_api_key(),
            max_retries=max(1, max_retries),
            timeout=timeout_seconds,
        )

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        api_key=api_key or _require_gemini_api_key(),
        max_retries=max(1, max_retries),
        timeout=timeout_seconds,
    )


def build_inspect_runtime(
    *,
    provider: Literal["gemini", "openai"] = "gemini",
    model: str = DEFAULT_GEMINI_MODEL,
    api_key: str | None = None,
    max_retries: int = 3,
    timeout_seconds: float | None = None,
    temperature: float = 0.1,
    llm: BaseChatModel | None = None,
    genai_client: genai.Client | None = None,
) -> InspectRuntime:
    """Build workflow runtime dependencies for the inspect graph."""

    if llm is None:
        if provider == "gemini":
            resolved_api_key = api_key or _require_gemini_api_key()
        else:
            resolved_api_key = api_key or _require_openai_api_key()
        llm = build_llm(
            provider=provider,
            model=model,
            api_key=resolved_api_key,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
        )

    resolved_genai_client = genai_client
    if provider == "gemini" and resolved_genai_client is None:
        resolved_genai_client = build_genai_client(
            api_key=api_key or _require_gemini_api_key()
        )

    return InspectRuntime(
        llm=llm,
        provider=provider,
        genai_client=resolved_genai_client,
    )


def _require_gemini_api_key() -> str:
    from v2a_inspect.settings import settings

    if settings.gemini_api_key is not None:
        return settings.gemini_api_key.get_secret_value()
    raise ValueError("GEMINI_API_KEY must be set for Gemini provider.")


def _require_openai_api_key() -> str:
    from v2a_inspect.settings import settings

    if settings.openai_api_key is not None:
        return settings.openai_api_key.get_secret_value()
    raise ValueError("OPENAI_API_KEY must be set for OpenAI provider.")
