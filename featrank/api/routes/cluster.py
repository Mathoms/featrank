"""POST /cluster — run clustering pipeline on ingested requests."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from loguru import logger

from featrank.api.routes.ingest import get_job
from featrank.api.schemas import ClusterRequest, ClusterResponse
from featrank.pipeline.cleaner import TextCleaner
from featrank.pipeline.clusterer import HDBSCANClusterer
from featrank.pipeline.embedder import Embedder
from featrank.pipeline.labeler import ClusterLabeler
from featrank.schemas import RawRequest

router = APIRouter()

_cleaner = TextCleaner()
_embedder = Embedder()
_clusterer = HDBSCANClusterer()
_labeler = ClusterLabeler()


@router.post("", response_model=ClusterResponse)
async def cluster(body: ClusterRequest) -> ClusterResponse:
    if body.requests:
        requests = body.requests
    elif body.job_id:
        raw = get_job(body.job_id)
        if raw is None:
            raise HTTPException(status_code=404, detail=f"job_id '{body.job_id}' not found")
        requests = [RawRequest(**r) for r in raw]
    else:
        raise HTTPException(status_code=422, detail="Provide either job_id or requests")

    kept, cleaned_texts = _cleaner.clean_requests(requests)
    embeddings = _embedder.embed(cleaned_texts)
    clustered = _clusterer.fit_predict(kept, embeddings, cleaned_texts)

    noise_count = sum(1 for r in clustered if r.cluster_id == -1)
    clusters = _labeler.build_clusters(clustered)

    logger.info(f"[api/cluster] {len(clusters)} clusters, {noise_count} noise")
    return ClusterResponse(clusters=clusters, noise_count=noise_count)
