from __future__ import annotations

from pathlib import Path

import google.genai as genai

from v2a_inspect.clients import upload_video as upload_gemini_video
from v2a_inspect.experiment.frames import extract_frames, strip_audio
from v2a_inspect.workflows.state import InspectState

from ._shared import append_state_message


def upload_video(
    state: InspectState,
    *,
    genai_client: genai.Client | None = None,
) -> dict[str, object]:
    """Preprocess video (strip audio) and prepare for the target provider.

    - Gemini: upload silent video to Gemini file API.
    - OpenAI: extract frames as base64 images.
    """

    video_path = state.get("video_path")
    if not video_path:
        raise ValueError("upload_video requires 'video_path' in state.")

    options = state.get("options")
    if options is None:
        raise ValueError("upload_video requires 'options' in state.")

    # 1. Strip audio (common for both providers)
    silent_path = strip_audio(video_path)

    video_name = Path(video_path).name
    updates: dict[str, object] = {"silent_video_path": silent_path}

    if options.provider == "openai":
        # 2b. OpenAI: extract frames
        frames = extract_frames(silent_path, fps=options.fps)
        updates["video_frames"] = frames
        message = (
            f"Extracted {len(frames)} frames from {video_name} (fps={options.fps})."
        )
    else:
        # 2a. Gemini: upload to file API
        if genai_client is None:
            raise ValueError("Gemini provider requires genai_client.")
        gemini_file = upload_gemini_video(
            genai_client,
            silent_path,
            poll_interval_seconds=options.poll_interval_seconds,
            max_wait_seconds=options.upload_timeout_seconds,
        )
        updates["gemini_file"] = gemini_file
        message = f"Uploaded {video_name} to Gemini."

    updates["progress_messages"] = append_state_message(
        state, "progress_messages", message
    )
    return updates
