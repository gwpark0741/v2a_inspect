import re
from pydantic import BaseModel, Field
from typing import Optional, Literal, List, Dict

from .scenes import EventTimestamp, VideoSceneAnalysis


class ModelSelection(BaseModel):
    """TTA/VTA model selection result for a single track or group."""

    reasoning: str
    model_type: Literal["TTA", "VTA"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    rule_based: bool = (
        False  # True = deterministic rule (background, etc.), False = LLM judgment
    )


class RawTrack(BaseModel):
    """One track extracted from a Scene (background or event)."""

    track_id: str  # format: "s{scene_index}_bg" or "s{scene_index}_ev{event_index}"
    scene_index: int
    kind: Literal["background", "event"]
    description: str
    source_visible: Optional[bool] = None  # None for background tracks
    start: float
    end: float
    event_timestamps: List[EventTimestamp] = Field(default_factory=list)
    n_scene_events: int = 0  # number of event tracks in the same scene
    model_selection: Optional[ModelSelection] = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    @classmethod
    def validate_track_id(cls, track_id: str) -> str:
        pattern = r"^s\d+_(bg|ev\d+)$"
        if not re.match(pattern, track_id):
            raise ValueError(
                f"Invalid track_id format: '{track_id}'. Expected 's{{i}}_bg' or 's{{i}}_ev{{j}}'."
            )
        return track_id


class TrackGroup(BaseModel):
    """A set of RawTracks that represent the same real-world audio entity."""

    group_id: str
    canonical_description: str  # description used for audio generation
    member_ids: List[str]  # track_ids belonging to this group
    vlm_verified: bool = False
    model_selection: Optional[ModelSelection] = None  # group-level representative


class GroupedAnalysis(BaseModel):
    """VideoSceneAnalysis annotated with group assignments."""

    scene_analysis: VideoSceneAnalysis  # VideoSceneAnalysis (annotated copy)
    raw_tracks: List[RawTrack]
    groups: List[TrackGroup]
    track_to_group: Dict[str, str]  # track_id -> group_id
