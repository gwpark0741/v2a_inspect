from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from v2a_inspect.clients import DEFAULT_GEMINI_MODEL
from v2a_inspect.pipeline.response_models import (
    GroupedAnalysis,
    RawTrack,
    TrackGroup,
    VideoSceneAnalysis,
)


class InspectOptions(BaseModel):
    """User-configurable options for the inspection workflow."""

    provider: Literal["gemini", "openai"] = "gemini"
    model_name: str = DEFAULT_GEMINI_MODEL
    fps: float = Field(default=2.0, gt=0.0)
    scene_analysis_mode: Literal[
        "default", "extended", "v2", "v3", "object_based", "constrained"
    ] = "default"
    enable_vlm_verify: bool = True
    enable_model_select: bool = False
    upload_timeout_seconds: int = Field(default=300, ge=1)
    text_timeout_ms: int = Field(default=120_000, ge=1)
    video_timeout_ms: int = Field(default=180_000, ge=1)
    max_retries: int = Field(default=3, ge=0)
    poll_interval_seconds: float = Field(default=2.0, gt=0.0)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    analyze_only: bool = False  # True이면 extract 완료 후 그루핑 없이 종료


class InspectState(TypedDict, total=False):
    """Shared LangGraph state for the inspection workflow."""

    video_path: str
    options: InspectOptions
    gemini_file: Any
    video_frames: list[tuple[float, str]]  # OpenAI용 (timestamp, base64_jpeg)
    silent_video_path: str  # 오디오 제거된 영상 경로
    scene_analysis: VideoSceneAnalysis
    raw_tracks: list[RawTrack]
    text_groups: list[TrackGroup]
    verified_groups: list[TrackGroup]
    final_groups: list[TrackGroup]
    grouped_analysis: GroupedAnalysis
    trace_id: str
    root_observation_id: str
    errors: list[str]
    warnings: list[str]
    progress_messages: list[str]
