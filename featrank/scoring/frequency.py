"""Raw request count signal scorer."""

from __future__ import annotations

from typing import Sequence

from featrank.schemas import FeatureCluster


class FrequencyScorer:
    """Score clusters by normalized raw request count."""

    def score(self, clusters: Sequence[FeatureCluster]) -> list[float]:
        """Return list of [0, 1] scores in the same order as input clusters."""
        counts = [c.request_count for c in clusters]
        max_count = max(counts) if counts else 1
        return [c / max_count for c in counts]
