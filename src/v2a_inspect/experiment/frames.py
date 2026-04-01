"""Frame extraction and video preprocessing utilities for experiments."""

from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path

from moviepy import VideoFileClip
from PIL import Image


# --------------------------------------------------------------------------- #
# Audio stripping — prevents audio leakage into Gemini's video analysis
# --------------------------------------------------------------------------- #


def strip_audio(video_path: str, *, output_dir: str | None = None) -> str:
    """Remove audio track from a video file.

    Returns the path to the silent video (a new temp file).
    If the video already has no audio, returns the original path unchanged.
    """
    from v2a_inspect.observability import start_observation

    with start_observation(
        name="preprocess.strip_audio",
        as_type="tool",
        input={"video_path": str(Path(video_path).name)},
    ) as obs:
        with VideoFileClip(video_path) as clip:
            if clip.audio is None:
                if obs is not None:
                    obs.update(output={"skipped": True, "reason": "no audio track"})
                return video_path

            silent = clip.without_audio()
            if output_dir:
                out = Path(output_dir)
                out.mkdir(parents=True, exist_ok=True)
                out_path = str(out / f"{Path(video_path).stem}_silent.mp4")
            else:
                fd, out_path = tempfile.mkstemp(suffix=".mp4", prefix="silent_")
                import os

                os.close(fd)

            silent.write_videofile(out_path, logger=None, audio=False)

        if obs is not None:
            obs.update(
                output={"skipped": False, "output_path": str(Path(out_path).name)}
            )

    return out_path


def extract_frames(
    video_path: str,
    fps: float = 3.0,
    *,
    max_width: int = 512,
    jpeg_quality: int = 85,
) -> list[tuple[float, str]]:
    """Extract frames from a video at the given fps.

    Returns a list of (timestamp_seconds, base64_jpeg_string) tuples.
    Frames are resized to *max_width* to keep token usage reasonable.
    """
    from v2a_inspect.observability import start_observation

    with start_observation(
        name="preprocess.extract_frames",
        as_type="tool",
        input={
            "video_path": str(Path(video_path).name),
            "fps": fps,
            "max_width": max_width,
        },
    ) as obs:
        frames: list[tuple[float, str]] = []
        interval = 1.0 / fps

        with VideoFileClip(video_path) as clip:
            duration = clip.duration
            t = 0.0
            while t < duration:
                frame_array = clip.get_frame(t)
                img = Image.fromarray(frame_array)

                if img.width > max_width:
                    ratio = max_width / img.width
                    img = img.resize(
                        (max_width, int(img.height * ratio)),
                        Image.Resampling.LANCZOS,
                    )

                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=jpeg_quality)
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")

                frames.append((round(t, 1), b64))
                t += interval

        if obs is not None:
            obs.update(output={"frame_count": len(frames), "duration": duration})

    return frames


def extract_thumbnail_frames(
    video_path: str,
    fps: float = 3.0,
    *,
    max_width: int = 200,
    jpeg_quality: int = 70,
) -> list[tuple[float, bytes]]:
    """Extract small thumbnail frames for HTML reports.

    Returns (timestamp, raw_jpeg_bytes) tuples.
    """
    thumbnails: list[tuple[float, bytes]] = []
    interval = 1.0 / fps

    with VideoFileClip(video_path) as clip:
        duration = clip.duration
        t = 0.0
        while t < duration:
            frame_array = clip.get_frame(t)
            img = Image.fromarray(frame_array)
            ratio = max_width / img.width
            img = img.resize(
                (max_width, int(img.height * ratio)),
                Image.Resampling.LANCZOS,
            )
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_quality)
            thumbnails.append((round(t, 1), buf.getvalue()))
            t += interval

    return thumbnails


def save_frames_to_dir(
    video_path: str,
    output_dir: str,
    fps: float = 3.0,
    *,
    max_width: int = 200,
) -> list[str]:
    """Extract frames and save as JPEG files. Returns list of saved paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    interval = 1.0 / fps

    with VideoFileClip(video_path) as clip:
        duration = clip.duration
        t = 0.0
        while t < duration:
            frame_array = clip.get_frame(t)
            img = Image.fromarray(frame_array)
            if img.width > max_width:
                ratio = max_width / img.width
                img = img.resize(
                    (max_width, int(img.height * ratio)),
                    Image.Resampling.LANCZOS,
                )
            filename = f"frame_{t:06.1f}s.jpg"
            path = out / filename
            img.save(path, format="JPEG", quality=80)
            saved.append(str(path))
            t += interval

    return saved
