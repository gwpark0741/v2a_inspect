from __future__ import annotations

from v2a_inspect.workflows.state import InspectState
from ..response_models import RawTrack
from ._shared import append_state_message


def extract_raw_tracks(state: InspectState) -> dict[str, object]:
    """Flatten scene analysis output into ordered raw tracks (background + events)."""

    scene_analysis = state.get("scene_analysis")
    if scene_analysis is None:
        raise ValueError("extract_raw_tracks requires 'scene_analysis' in state.")

    tracks: list[RawTrack] = []
    for scene in scene_analysis.scenes:
        scene_index = scene.scene_index
        n_events = len(scene.audio_events)

        tracks.append(
            RawTrack(
                track_id=f"s{scene_index}_bg",
                scene_index=scene_index,
                kind="background",
                description=scene.background_ambience,
                source_visible=None,
                start=scene.time_range.start,
                end=scene.time_range.end,
                event_timestamps=[],
                n_scene_events=n_events,
            )
        )
        for event_index, event in enumerate(scene.audio_events):
            tracks.append(
                RawTrack(
                    track_id=f"s{scene_index}_ev{event_index}",
                    scene_index=scene_index,
                    kind="event",
                    description=event.description,
                    source_visible=event.source_visible,
                    start=event.time_range.start,
                    end=event.time_range.end,
                    event_timestamps=event.event_timestamps,
                    n_scene_events=n_events,
                )
            )

    return {
        "raw_tracks": tracks,
        "progress_messages": append_state_message(
            state,
            "progress_messages",
            f"Extracted {len(tracks)} raw tracks from {len(scene_analysis.scenes)} scenes.",
        ),
    }
