"""RAG retrieval using FAISS index over cluster embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import faiss
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from featrank.config import settings
from featrank.schemas import FeatureCluster


class RAGIndex:
    """Build a FAISS index over cluster centroids for semantic retrieval."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None
        self._index: faiss.Index | None = None
        self._clusters: list[FeatureCluster] = []

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build(self, clusters: Sequence[FeatureCluster]) -> None:
        """Build FAISS index from cluster centroids."""
        self._clusters = list(clusters)
        centroids: list[np.ndarray] = []

        for cluster in self._clusters:
            embs = np.array([r.embedding for r in cluster.requests], dtype=np.float32)
            centroid = embs.mean(axis=0)
            norm = np.linalg.norm(centroid)
            centroids.append(centroid / norm if norm > 0 else centroid)

        matrix = np.stack(centroids).astype(np.float32)
        dim = matrix.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(matrix)
        logger.info(f"[rag] FAISS index built: {len(self._clusters)} clusters, dim={dim}")

    def search(self, query: str, top_k: int = 5) -> list[tuple[FeatureCluster, float]]:
        """Return top-k clusters most relevant to the query."""
        if self._index is None or not self._clusters:
            raise RuntimeError("Call build() before search()")

        q_emb = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        distances, indices = self._index.search(q_emb.astype(np.float32), top_k)

        results: list[tuple[FeatureCluster, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self._clusters):
                results.append((self._clusters[idx], float(dist)))
        return results

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))
        logger.info(f"[rag] Index saved to {path}")

    def load(self, path: str | Path) -> None:
        self._index = faiss.read_index(str(path))
        logger.info(f"[rag] Index loaded from {path}")
