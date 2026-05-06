"""HDBSCAN clustering over sentence embeddings."""

from __future__ import annotations

import time
from typing import Sequence

import hdbscan
import numpy as np
from loguru import logger

from featrank.config import settings
from featrank.schemas import ClusteredRequest, RawRequest


class HDBSCANClusterer:
    """Cluster embeddings with HDBSCAN.

    cluster_id == -1 → noise / outlier (no cluster).
    """

    def __init__(
        self,
        min_cluster_size: int | None = None,
        min_samples: int | None = None,
        metric: str | None = None,
        cluster_selection_epsilon: float | None = None,
    ) -> None:
        self.min_cluster_size = min_cluster_size or settings.hdbscan_min_cluster_size
        self.min_samples = min_samples or settings.hdbscan_min_samples
        self.metric = metric or settings.hdbscan_metric
        self.cluster_selection_epsilon = (
            cluster_selection_epsilon or settings.hdbscan_cluster_selection_epsilon
        )

    def fit_predict(
        self,
        requests: Sequence[RawRequest],
        embeddings: np.ndarray,
        cleaned_texts: Sequence[str],
    ) -> list[ClusteredRequest]:
        """Assign cluster IDs to each request and return ClusteredRequest list."""
        t0 = time.perf_counter()
        assert len(requests) == len(embeddings) == len(cleaned_texts), (
            "requests, embeddings, cleaned_texts must all have the same length"
        )

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            prediction_data=True,
        )
        labels: np.ndarray = clusterer.fit_predict(embeddings)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        elapsed = time.perf_counter() - t0

        logger.info(
            f"[clusterer] {len(requests)} requests → "
            f"{n_clusters} clusters, {n_noise} noise ({n_noise / len(labels):.1%}) "
            f"— {elapsed:.2f}s"
        )

        clustered: list[ClusteredRequest] = []
        for req, emb, cleaned, label in zip(requests, embeddings, cleaned_texts, labels):
            clustered.append(
                ClusteredRequest(
                    **req.model_dump(),
                    cluster_id=int(label),
                    embedding=emb.tolist(),
                    cleaned_text=cleaned,
                )
            )

        return clustered
