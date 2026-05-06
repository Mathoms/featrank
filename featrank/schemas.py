"""Pydantic data models that flow through the entire pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class RawRequest(BaseModel):
    id: str
    text: str
    source: str
    user_id: Optional[str] = None
    mrr: float = 0.0
    created_at: Optional[datetime] = None
    metadata: dict = Field(default_factory=dict)


class ClusteredRequest(RawRequest):
    cluster_id: int
    embedding: list[float]
    cleaned_text: str


class FeatureCluster(BaseModel):
    cluster_id: int
    label: str
    requests: list[ClusteredRequest]
    request_count: int
    unique_users: int
    total_mrr: float
    sample_requests: list[str]
    llm_diagnosis: Optional[str] = None


class PrioritizedCluster(FeatureCluster):
    score_frequency: float = Field(ge=0.0, le=1.0)
    score_segment_value: float = Field(ge=0.0, le=1.0)
    score_github_signal: float = Field(ge=0.0, le=1.0)
    score_roadmap_fit: float = Field(ge=0.0, le=1.0)
    priority_score: float = Field(ge=0.0, le=100.0)
    priority_rank: int = Field(ge=1)
    recommendation: str = ""
