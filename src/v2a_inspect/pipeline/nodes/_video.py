from __future__ import annotations

import os
import tempfile
from typing import Any

import google.genai as genai


def clip_and_upload(
    video_path: str,
    start: float,
    end: float,
    genai_client: genai.Client,
    *,
    padding: float = 0.5,
) -> Any:
    """Clip video segment and upload to Gemini. Returns a Gemini file object."""
    clip_start = max(0.0, start - padding)
    clip_end = end + padding
    tmp_path = _clip_to_temp(video_path, clip_start, clip_end)
    try:
        return _upload_and_wait(tmp_path, genai_client)
    finally:
        _cleanup(tmp_path)


def _clip_to_temp(video_path: str, start: float, end: float) -> str:
    from moviepy import VideoFileClip

    source = VideoFileClip(video_path)
    actual_end = min(end, float(source.duration or end))
    clip = source.subclipped(start, actual_end)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()
    clip.write_videofile(tmp_path, logger=None)
    clip.close()
    source.close()
    return tmp_path


def _upload_and_wait(path: str, client: genai.Client) -> Any:
    from v2a_inspect.clients.video import upload_video

    return upload_video(client, path)


def clip_and_extract_frames(
    video_path: str,
    start: float,
    end: float,
    fps: float,
    *,
    padding: float = 0.5,
) -> list[tuple[float, str]]:
    """Clip video segment and extract frames as base64 images (for OpenAI)."""
    from v2a_inspect.experiment.frames import extract_frames
    from v2a_inspect.observability import start_observation

    clip_start = max(0.0, start - padding)
    clip_end = end + padding

    with start_observation(
        name="openai.clip_and_extract_frames",
        as_type="tool",
        input={"start": clip_start, "end": clip_end, "fps": fps},
    ) as obs:
        tmp_path = _clip_to_temp(video_path, clip_start, clip_end)
        try:
            frames = extract_frames(tmp_path, fps=fps)
        finally:
            _cleanup(tmp_path)

        if obs is not None:
            obs.update(output={"frame_count": len(frames)})
        return frames


def _cleanup(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass
