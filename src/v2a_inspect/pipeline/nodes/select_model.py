from __future__ import annotations

import math
from collections import Counter
from typing import Literal

import google.genai as genai
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from ..prompt_templates import resolve_prompt
from ..response_models import ModelSelectResponse, ModelSelection
from v2a_inspect.workflows.state import InspectState
from ._shared import (
    append_state_message,
    build_model_select_segment_list,
    get_active_groups,
    invoke_structured_video,
)
from ._video import clip_and_extract_frames, clip_and_upload


def select_models(
    state: InspectState,
    *,
    llm: BaseChatModel,
    genai_client: genai.Client | None = None,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    """Assign TTA or VTA to track groups using CoT reasoning (Gemini or OpenAI)."""

    options = state.get("options")
    if options is None:
        raise ValueError("select_models requires 'options' in state.")

    raw_tracks = state.get("raw_tracks")
    if raw_tracks is None:
        raise ValueError("select_models requires 'raw_tracks' in state.")

    groups = [group.model_copy(deep=True) for group in get_active_groups(state)]
    copied_tracks = [track.model_copy(deep=True) for track in raw_tracks]

    if not groups:
        return {
            "final_groups": [],
            "raw_tracks": copied_tracks,
            "progress_messages": append_state_message(
                state, "progress_messages", "Skipped model selection: no groups."
            ),
        }

    provider = options.provider
    gemini_file = state.get("gemini_file")
    video_frames = state.get("video_frames")

    if gemini_file is None and video_frames is None:
        warnings = append_state_message(
            state, "warnings", "Skipped model selection: no video input."
        )
        return {
            "final_groups": groups,
            "raw_tracks": copied_tracks,
            "warnings": warnings,
        }

    video_path = state.get("video_path", "")
    tracks_by_id = {track.track_id: track for track in copied_tracks}
    warnings = list(state.get("warnings", []))

    for group in groups:
        member_tracks = [
            tracks_by_id[track_id]
            for track_id in group.member_ids
            if track_id in tracks_by_id
        ]
        if not member_tracks:
            continue

        # Background tracks: always TTA, deterministic
        if all(track.kind == "background" for track in member_tracks):
            bg_selection = _background_model_selection()
            for track in member_tracks:
                track.model_selection = bg_selection.model_copy(deep=True)
            group.model_selection = bg_selection
            continue

        # Clip video to group range
        clip_start = min(t.start for t in member_tracks)
        clip_end = max(t.end for t in member_tracks)

        resolved_prompt = resolve_prompt("model_select").render(
            segment_list=build_model_select_segment_list(member_tracks)
        )

        invoke_kwargs: dict = {
            "fps": options.fps,
            "prompt": resolved_prompt,
            "schema": ModelSelectResponse,
            "timeout_ms": options.video_timeout_ms,
            "max_retries": options.max_retries,
            "label": f"model_select_{group.group_id}",
            "config": config,
        }

        if provider == "openai" and video_path:
            try:
                clip_frames = clip_and_extract_frames(
                    video_path, clip_start, clip_end, options.fps
                )
                invoke_kwargs["frames"] = clip_frames
            except Exception as clip_exc:  # noqa: BLE001
                warnings.append(
                    f"Frame extraction failed for {group.group_id}: {clip_exc}"
                )
                invoke_kwargs["frames"] = video_frames
        elif video_path and genai_client is not None:
            try:
                file_to_use = clip_and_upload(
                    video_path, clip_start, clip_end, genai_client
                )
                invoke_kwargs["file_obj"] = file_to_use
            except Exception as clip_exc:  # noqa: BLE001
                warnings.append(f"Clip failed for {group.group_id}: {clip_exc}")
                invoke_kwargs["file_obj"] = gemini_file
        else:
            invoke_kwargs["file_obj"] = gemini_file

        if provider == "gemini":
            invoke_kwargs["model"] = options.model_name

        try:
            response = invoke_structured_video(llm, **invoke_kwargs)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Model selection failed for {group.group_id}: {exc}")
            continue

        # Apply per-track CoT results
        for segment in response.segments:
            idx = segment.segment_index
            if idx is None or idx >= len(member_tracks):
                continue
            member_tracks[idx].model_selection = ModelSelection(
                model_type=segment.selected_model,
                confidence=segment.confidence,
                reasoning=segment.reasoning,
                rule_based=False,
            )

        # Aggregate to group level: majority voting + geometric mean confidence
        decided = [t for t in member_tracks if t.model_selection is not None]
        if not decided:
            continue

        decided_selections = [
            t.model_selection for t in decided if t.model_selection is not None
        ]
        vote_counts = Counter(s.model_type for s in decided_selections)
        group_model: Literal["TTA", "VTA"] = vote_counts.most_common(1)[0][0]
        confidences = [s.confidence for s in decided_selections]
        group_confidence = math.exp(
            sum(math.log(max(c, 1e-6)) for c in confidences) / len(confidences)
        )
        reasoning = "; ".join(s.reasoning for s in decided_selections if s.reasoning)
        group.model_selection = ModelSelection(
            model_type=group_model,
            confidence=round(group_confidence, 3),
            reasoning=reasoning[:200],
            rule_based=False,
        )

    updates: dict[str, object] = {
        "final_groups": groups,
        "raw_tracks": copied_tracks,
        "progress_messages": append_state_message(
            state, "progress_messages", "Assigned CoT model selections to track groups."
        ),
    }
    if warnings != state.get("warnings", []):
        updates["warnings"] = warnings
    return updates


def _background_model_selection() -> ModelSelection:
    return ModelSelection(
        model_type="TTA",
        confidence=0.90,
        reasoning=(
            "Background track: TTA preferred to avoid foreground sound bleed-through "
            "that VTA may introduce."
        ),
        rule_based=True,
    )
