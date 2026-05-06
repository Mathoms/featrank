"""Exact and semantic deduplication of raw requests."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
from loguru import logger

from featrank.schemas import RawRequest


class Deduplicator:
    """Remove exact duplicates and near-duplicate requests.

    Two passes:
    1. Exact dedup — same normalized text
    2. Semantic dedup — cosine similarity above threshold
    """

    def __init__(self, similarity_threshold: float = 0.95) -> None:
        self.similarity_threshold = similarity_threshold

    def exact_dedup(self, requests: Sequence[RawRequest]) -> list[RawRequest]:
        seen: set[str] = set()
        unique: list[RawRequest] = []
        for req in requests:
            key = req.text.strip().lower()
            if key not in seen:
                seen.add(key)
                unique.append(req)
        dropped = len(requests) - len(unique)
        logger.info(f"[dedup] Exact: {len(unique)} kept, {dropped} dropped")
        return unique

    def semantic_dedup(
        self,
        requests: Sequence[RawRequest],
        embeddings: np.ndarray,
    ) -> tuple[list[RawRequest], np.ndarray]:
        """Remove semantically near-duplicate requests.

        Greedy: keep the first occurrence; drop subsequent requests
        whose max cosine similarity to any kept embedding exceeds threshold.
        """
        t0 = time.perf_counter()
        requests = list(requests)
        n = len(requests)
        kept_indices: list[int] = []
        kept_embeddings: list[np.ndarray] = []

        for i in range(n):
            emb = embeddings[i]
            if kept_embeddings:
                kept_matrix = np.stack(kept_embeddings)
                sims = kept_matrix @ emb
                if sims.max() >= self.similarity_threshold:
                    continue
            kept_indices.append(i)
            kept_embeddings.append(emb)

        kept_requests = [requests[i] for i in kept_indices]
        kept_emb_array = np.stack(kept_embeddings) if kept_embeddings else np.array([])
        dropped = n - len(kept_indices)

        logger.info(
            f"[dedup] Semantic (threshold={self.similarity_threshold}): "
            f"{len(kept_indices)} kept, {dropped} dropped "
            f"— {time.perf_counter() - t0:.2f}s"
        )
        return kept_requests, kept_emb_array
