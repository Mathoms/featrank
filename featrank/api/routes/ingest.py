"""POST /ingest — accept raw feature requests."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter
from loguru import logger

from featrank.api.schemas import IngestRequest, IngestResponse

router = APIRouter()

_store: dict[str, list[Any]] = {}


@router.post("", response_model=IngestResponse)
async def ingest(body: IngestRequest) -> IngestResponse:
    job_id = str(uuid.uuid4())
    _store[job_id] = [r.model_dump() for r in body.requests]
    logger.info(f"[api/ingest] job_id={job_id}, count={len(body.requests)}")
    return IngestResponse(job_id=job_id, request_count=len(body.requests))


def get_job(job_id: str) -> list[Any] | None:
    return _store.get(job_id)
