"""Tests for featrank.pipeline.clusterer."""

from __future__ import annotations

import numpy as np
import pytest

from featrank.pipeline.clusterer import HDBSCANClusterer
from featrank.schemas import ClusteredRequest, RawRequest


def make_requests(n: int) -> list[RawRequest]:
    return [RawRequest(id=str(i), text=f"request {i}", source="test") for i in range(n)]


def make_embeddings(n: int, dim: int = 16) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((n, dim)).astype(np.float32)


class TestHDBSCANClusterer:
    def setup_method(self):
        self.clusterer = HDBSCANClusterer(
            min_cluster_size=2,
            min_samples=1,
            metric="euclidean",
            cluster_selection_epsilon=0.0,
        )

    def test_returns_clustered_requests(self):
        n = 20
        reqs = make_requests(n)
        embs = make_embeddings(n)
        texts = [f"text {i}" for i in range(n)]
        result = self.clusterer.fit_predict(reqs, embs, texts)
        assert len(result) == n
        assert all(isinstance(r, ClusteredRequest) for r in result)

    def test_cluster_ids_assigned(self):
        n = 30
        reqs = make_requests(n)
        embs = make_embeddings(n)
        texts = [f"text {i}" for i in range(n)]
        result = self.clusterer.fit_predict(reqs, embs, texts)
        cluster_ids = {r.cluster_id for r in result}
        assert -1 in cluster_ids or len(cluster_ids) >= 1

    def test_raises_on_length_mismatch(self):
        reqs = make_requests(5)
        embs = make_embeddings(3)
        texts = ["t"] * 5
        with pytest.raises(AssertionError):
            self.clusterer.fit_predict(reqs, embs, texts)

    def test_cleaned_text_stored(self):
        n = 10
        reqs = make_requests(n)
        embs = make_embeddings(n)
        texts = [f"cleaned text {i}" for i in range(n)]
        result = self.clusterer.fit_predict(reqs, embs, texts)
        for r, t in zip(result, texts):
            assert r.cleaned_text == t
