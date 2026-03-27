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


def _cleanup(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass
