from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

NonNegativeInt = Annotated[int, Field(ge=0)]


class GroupingResponseGroup(BaseModel):
    member_indices: list[NonNegativeInt] = Field(default_factory=list)
    canonical_index: NonNegativeInt | None = None
    reasoning: str = ""


class GroupingResponse(BaseModel):
    groups: list[GroupingResponseGroup] = Field(default_factory=list)


class VLMVerifyResponse(BaseModel):
    same_entity: bool | Literal["uncertain"] = True
    confirmed_groups: list[list[NonNegativeInt]] | None = None
    reasoning: str = ""


class CoTModelSelectSegmentResponse(BaseModel):
    segment_index: NonNegativeInt | None = None
    reasoning: str = ""
    selected_model: Literal["TTA", "VTA"] = "TTA"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ModelSelectResponse(BaseModel):
    segments: list[CoTModelSelectSegmentResponse] = Field(default_factory=list)
