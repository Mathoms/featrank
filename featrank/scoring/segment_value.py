"""MRR-weighted segment value scorer."""

from __future__ import annotations

from typing import Sequence

from featrank.schemas import FeatureCluster


class SegmentValueScorer:
    """Score clusters by total MRR of requestors.

    Falls back to unique user count if no MRR data is present.

    Core insight: 1 enterprise request at $50K MRR outweighs
    50 free-tier requests in business-priority ranking.
    """

    def score(self, clusters: Sequence[FeatureCluster]) -> list[float]:
        mrrs = [c.total_mrr for c in clusters]
        has_mrr = any(m > 0 for m in mrrs)

        if has_mrr:
            max_val = max(mrrs) or 1
            return [m / max_val for m in mrrs]

        users = [c.unique_users for c in clusters]
        max_val = max(users) if users else 1
        return [u / max_val for u in users]
