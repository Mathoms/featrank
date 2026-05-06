"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from featrank import __version__
from featrank.api.routes import ingest, cluster, rank, report, health

app = FastAPI(
    title="featrank API",
    description="Semantic feature request deduplication + priority ranking",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(cluster.router, prefix="/cluster", tags=["cluster"])
app.include_router(rank.router, prefix="/rank", tags=["rank"])
app.include_router(report.router, prefix="/report", tags=["report"])
