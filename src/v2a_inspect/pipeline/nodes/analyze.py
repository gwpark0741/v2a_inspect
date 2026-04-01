from __future__ import annotations

from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from v2a_inspect.workflows.state import InspectState

from ..response_models import VideoSceneAnalysis, ObjectBasedVideoSceneAnalysis
from ..response_models.scenes import AudioEvent, Scene
from ._shared import (
    append_state_message,
    get_scene_analysis_prompt,
    invoke_structured_video,
)


def _convert_object_to_event(
    obj_analysis: ObjectBasedVideoSceneAnalysis,
) -> VideoSceneAnalysis:
    """Convert object-based analysis to event-based format for downstream compatibility."""
    scenes = []
    for s in obj_analysis.scenes:
        events = [
            AudioEvent(
                description=obj.description,
                source_visible=True,
                time_range=obj.time_range,
                event_timestamps=obj.event_timestamps,
            )
            for obj in s.objects
        ]
        scenes.append(
            Scene(
                scene_index=s.scene_index,
                time_range=s.time_range,
                background_ambience=s.background_sound,
                audio_events=events,
            )
        )
    return VideoSceneAnalysis(total_duration=obj_analysis.total_duration, scenes=scenes)


def analyze_scenes(
    state: InspectState,
    *,
    llm: BaseChatModel,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    """Run scene analysis for the uploaded video (supports Gemini and OpenAI)."""

    options = state.get("options")
    if options is None:
        raise ValueError("analyze_scenes requires 'options' in state.")

    resolved_prompt = get_scene_analysis_prompt(options)
    is_object_based = options.scene_analysis_mode == "object_based"
    schema = ObjectBasedVideoSceneAnalysis if is_object_based else VideoSceneAnalysis

    if options.provider == "openai":
        video_frames = state.get("video_frames")
        if video_frames is None:
            raise ValueError(
                "analyze_scenes requires 'video_frames' for OpenAI provider."
            )
        raw_result = invoke_structured_video(
            llm,
            frames=video_frames,
            fps=options.fps,
            prompt=resolved_prompt,
            schema=schema,
            timeout_ms=options.video_timeout_ms,
            max_retries=options.max_retries,
            label=f"scene_analysis_{options.scene_analysis_mode}",
            config=config,
        )
        provider_label = "OpenAI"
    else:
        gemini_file = state.get("gemini_file")
        if gemini_file is None:
            raise ValueError(
                "analyze_scenes requires 'gemini_file' for Gemini provider."
            )
        raw_result = invoke_structured_video(
            llm,
            file_obj=gemini_file,
            fps=options.fps,
            prompt=resolved_prompt,
            schema=schema,
            model=options.model_name,
            timeout_ms=options.video_timeout_ms,
            max_retries=options.max_retries,
            label=f"scene_analysis_{options.scene_analysis_mode}",
            config=config,
        )
        provider_label = "Gemini"

    if is_object_based:
        scene_analysis = _convert_object_to_event(
            cast(ObjectBasedVideoSceneAnalysis, raw_result)
        )
    else:
        scene_analysis = cast(VideoSceneAnalysis, raw_result)

    message = f"Analyzed {len(scene_analysis.scenes)} scenes with {provider_label}."
    return {
        "scene_analysis": scene_analysis,
        "progress_messages": append_state_message(state, "progress_messages", message),
    }
