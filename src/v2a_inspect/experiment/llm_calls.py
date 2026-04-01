"""Direct LLM call functions for experiments, bypassing the LangGraph pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

import google.genai as genai
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from v2a_inspect.clients.video import upload_video, build_uploaded_video_content_block
from v2a_inspect.experiment.frames import extract_frames, strip_audio

T = TypeVar("T", bound=BaseModel)


# --------------------------------------------------------------------------- #
# Token pricing per 1M tokens (USD)
# --------------------------------------------------------------------------- #
PRICING: dict[str, dict[str, float]] = {
    "gemini-3.1-pro-preview": {"input": 1.25, "output": 10.0},
    "gpt-5.4": {"input": 2.5, "output": 10.0},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.5},
}

# Supported models per provider
GEMINI_MODELS = ["gemini-3.1-pro-preview"]
OPENAI_MODELS = ["gpt-5.4", "gpt-5.4-mini"]
ALL_MODELS = GEMINI_MODELS + OPENAI_MODELS


@dataclass
class LLMCallResult:
    """Result of a single LLM call."""

    result: Any
    elapsed_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    raw_response: Any = field(default=None, repr=False)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = PRICING.get(model, {"input": 0.0, "output": 0.0})
    return (
        input_tokens * prices["input"] + output_tokens * prices["output"]
    ) / 1_000_000


# --------------------------------------------------------------------------- #
# Gemini: native video upload
# --------------------------------------------------------------------------- #


def call_gemini(
    video_path: str,
    system_prompt: str,
    user_prompt: str,
    schema: type[T],
    *,
    fps: float = 3.0,
    model: str = "gemini-3.1-pro-preview",
    api_key: str | None = None,
    temperature: float = 0.0,
) -> LLMCallResult:
    """Call Gemini with native video upload and structured output.

    Supported models: gemini-3.1-pro-preview
    """

    resolved_key = api_key or _require_gemini_key()

    # Strip audio to prevent audio leakage into analysis
    silent_path = strip_audio(video_path)

    # Upload silent video
    client = genai.Client(api_key=resolved_key)
    file_obj = upload_video(client, silent_path)

    # Build LLM
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        api_key=resolved_key,
        max_retries=3,
    )
    structured_llm = llm.with_structured_output(schema, method="json_schema")

    # Build messages
    messages: list[BaseMessage] = []
    if system_prompt.strip():
        messages.append(SystemMessage(content=system_prompt))
    messages.append(
        HumanMessage(
            content=[
                {"type": "text", "text": user_prompt},
                build_uploaded_video_content_block(file_obj, fps=fps),
            ]
        )
    )

    # Invoke
    start = time.monotonic()
    try:
        result = structured_llm.invoke(messages)
        elapsed = time.monotonic() - start

        if not isinstance(result, schema):
            result = schema.model_validate(result)

        return LLMCallResult(
            result=result,
            elapsed_seconds=round(elapsed, 2),
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        return LLMCallResult(
            result=None,
            elapsed_seconds=round(elapsed, 2),
            error=str(exc),
        )


# --------------------------------------------------------------------------- #
# OpenAI: frame extraction → multi-image input
# --------------------------------------------------------------------------- #


def call_openai(
    video_path: str,
    system_prompt: str,
    user_prompt: str,
    schema: type[T],
    *,
    fps: float = 3.0,
    model: str,
    api_key: str | None = None,
    temperature: float = 0.0,
) -> LLMCallResult:
    """Call OpenAI with extracted frames and structured output.

    Supported models: gpt-5.4, gpt-5.4-mini
    """

    resolved_key = api_key or _require_openai_key()

    # Extract frames
    frames = extract_frames(video_path, fps=fps)

    # Build LLM
    llm = ChatOpenAI(
        model=model,  # type: ignore[unknown-argument]
        temperature=temperature,
        api_key=resolved_key,  # type: ignore[unknown-argument]
        max_retries=3,
    )
    structured_llm = llm.with_structured_output(schema, method="json_schema")

    # Build messages with frame images
    content_blocks: list[dict[str, Any]] = []

    # Add frame context header
    content_blocks.append(
        {
            "type": "text",
            "text": (
                f"Below are {len(frames)} frames extracted from a video at {fps} fps. "
                f"Frame timestamps are shown in the order they appear.\n\n"
            ),
        }
    )

    # Add each frame as an image with timestamp
    for timestamp, b64_jpeg in frames:
        content_blocks.append(
            {
                "type": "text",
                "text": f"[Frame at {timestamp:.1f}s]",
            }
        )
        content_blocks.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_jpeg}",
                    "detail": "low",
                },
            }
        )

    # Add the actual prompt
    content_blocks.append(
        {
            "type": "text",
            "text": f"\n{user_prompt}",
        }
    )

    messages: list[BaseMessage] = []
    if system_prompt.strip():
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=content_blocks))  # type: ignore[call-overload]

    # Invoke
    start = time.monotonic()
    try:
        response = structured_llm.invoke(messages)
        elapsed = time.monotonic() - start

        if not isinstance(response, schema):
            response = schema.model_validate(response)

        # Try to extract token usage from the raw response
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "response_metadata"):
            usage = getattr(response, "response_metadata", {}).get("token_usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

        return LLMCallResult(
            result=response,
            elapsed_seconds=round(elapsed, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=estimate_cost(model, input_tokens, output_tokens),
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        return LLMCallResult(
            result=None,
            elapsed_seconds=round(elapsed, 2),
            error=str(exc),
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _require_gemini_key() -> str:
    from v2a_inspect.settings import settings

    if settings.gemini_api_key is not None:
        return settings.gemini_api_key.get_secret_value()
    raise ValueError("GEMINI_API_KEY must be set.")


def _require_openai_key() -> str:
    from v2a_inspect.settings import settings

    if settings.openai_api_key is not None:  # type: ignore[attr-defined]
        return settings.openai_api_key.get_secret_value()  # type: ignore[attr-defined]
    raise ValueError("OPENAI_API_KEY must be set.")
