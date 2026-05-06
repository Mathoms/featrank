"""Pydantic request/response models for the REST API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from featrank.schemas import FeatureCluster, PrioritizedCluster, RawRequest


class IngestRequest(BaseModel):
    requests: list[RawRequest]


class IngestResponse(BaseModel):
    job_id: str
    request_count: int


class ClusterRequest(BaseModel):
    job_id: Optional[str] = None
    requests: Optional[list[RawRequest]] = None


class ClusterResponse(BaseModel):
    clusters: list[FeatureCluster]
    noise_count: int


class RankRequest(BaseModel):
    clusters: list[FeatureCluster]
    roadmap_text: Optional[str] = None
    github_repo: Optional[str] = None
    weights: Optional[dict[str, float]] = None


class RankResponse(BaseModel):
    ranked: list[PrioritizedCluster]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
