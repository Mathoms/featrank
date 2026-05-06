"""GET /health endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from featrank import __version__
from featrank.api.schemas import HealthResponse

router = APIRouter()

_model_loaded = False


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_model_loaded,
        version=__version__,
    )
