"""Tests for featrank.scoring.ranker."""

from __future__ import annotations

import pytest

from featrank.schemas import ClusteredRequest, FeatureCluster, PrioritizedCluster
from featrank.scoring.ranker import CompositeRanker


def make_cluster(
    cluster_id: int,
    label: str,
    request_count: int,
    unique_users: int,
    total_mrr: float,
) -> FeatureCluster:
    req = ClusteredRequest(
        id="1",
        text="sample",
        source="test",
        cluster_id=cluster_id,
        embedding=[0.1] * 10,
        cleaned_text="sample",
    )
    return FeatureCluster(
        cluster_id=cluster_id,
        label=label,
        requests=[req] * request_count,
        request_count=request_count,
        unique_users=unique_users,
        total_mrr=total_mrr,
        sample_requests=["sample"],
    )


@pytest.fixture
def clusters():
    return [
        make_cluster(0, "dark mode", 50, 40, 5000.0),
        make_cluster(1, "csv export", 10, 8, 800.0),
        make_cluster(2, "bulk actions", 30, 25, 15000.0),
    ]


class TestCompositeRanker:
    def setup_method(self):
        self.ranker = CompositeRanker(
            weight_frequency=0.25,
            weight_segment_value=0.25,
            weight_github_signal=0.25,
            weight_roadmap_fit=0.25,
            github_repo="",
        )

    def test_returns_prioritized_clusters(self, clusters):
        result = self.ranker.rank(clusters)
        assert len(result) == len(clusters)
        assert all(isinstance(c, PrioritizedCluster) for c in result)

    def test_ranks_are_sequential(self, clusters):
        result = self.ranker.rank(clusters)
        ranks = [c.priority_rank for c in result]
        assert sorted(ranks) == list(range(1, len(clusters) + 1))

    def test_scores_are_sorted_descending(self, clusters):
        result = self.ranker.rank(clusters)
        scores = [c.priority_score for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_scores_are_in_range(self, clusters):
        result = self.ranker.rank(clusters)
        for c in result:
            assert 0 <= c.priority_score <= 100
            assert 0 <= c.score_frequency <= 1
            assert 0 <= c.score_segment_value <= 1

    def test_high_mrr_cluster_ranks_higher(self, clusters):
        result = self.ranker.rank(clusters)
        bulk_rank = next(c.priority_rank for c in result if c.label == "bulk actions")
        csv_rank = next(c.priority_rank for c in result if c.label == "csv export")
        assert bulk_rank < csv_rank
