"""POST /rank — score and rank feature clusters."""

from __future__ import annotations

from fastapi import APIRouter
from loguru import logger

from featrank.api.schemas import RankRequest, RankResponse
from featrank.scoring.ranker import CompositeRanker

router = APIRouter()


@router.post("", response_model=RankResponse)
async def rank(body: RankRequest) -> RankResponse:
    weights = body.weights or {}
    ranker = CompositeRanker(
        weight_frequency=weights.get("frequency"),
        weight_segment_value=weights.get("segment_value"),
        weight_github_signal=weights.get("github_signal"),
        weight_roadmap_fit=weights.get("roadmap_fit"),
        github_repo=body.github_repo,
    )
    ranked = ranker.rank(body.clusters, roadmap_text=body.roadmap_text)
    logger.info(f"[api/rank] {len(ranked)} clusters ranked")
    return RankResponse(ranked=ranked)
