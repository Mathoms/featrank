"""Auto-label clusters using TF-IDF keywords + LLM fallback."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Sequence

from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer

from featrank.schemas import ClusteredRequest, FeatureCluster


class ClusterLabeler:
    """Generate human-readable labels for each cluster.

    Strategy:
    1. TF-IDF over cluster texts → top 5 keywords
    2. Attempt LLM label generation (imported lazily to avoid circular deps)
    3. Fall back to TF-IDF keywords if LLM is unavailable
    """

    def __init__(self, top_k_keywords: int = 5, top_k_samples: int = 3) -> None:
        self.top_k_keywords = top_k_keywords
        self.top_k_samples = top_k_samples

    def _tfidf_keywords(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        vectorizer = TfidfVectorizer(
            max_features=self.top_k_keywords,
            stop_words="english",
            ngram_range=(1, 2),
        )
        try:
            vectorizer.fit(texts)
            return list(vectorizer.get_feature_names_out())
        except Exception:
            return []

    def _tfidf_label(self, texts: list[str]) -> str:
        keywords = self._tfidf_keywords(texts)
        return " / ".join(keywords[:3]) if keywords else "unlabeled cluster"

    def _llm_label(self, texts: list[str], keywords: list[str]) -> str | None:
        """Attempt to call the LLM layer; return None on any failure."""
        try:
            from featrank.llm.extractor import label_cluster

            return label_cluster(texts, keywords)
        except Exception as exc:
            logger.debug(f"[labeler] LLM label failed: {exc}")
            return None

    def build_clusters(
        self, clustered_requests: Sequence[ClusteredRequest]
    ) -> list[FeatureCluster]:
        """Group clustered requests into FeatureCluster objects with labels."""
        t0 = time.perf_counter()
        groups: dict[int, list[ClusteredRequest]] = defaultdict(list)
        for req in clustered_requests:
            if req.cluster_id != -1:
                groups[req.cluster_id].append(req)

        clusters: list[FeatureCluster] = []
        for cluster_id, reqs in sorted(groups.items()):
            texts = [r.cleaned_text for r in reqs]
            keywords = self._tfidf_keywords(texts)
            label = self._llm_label(texts, keywords) or self._tfidf_label(texts)

            unique_users = len({r.user_id for r in reqs if r.user_id})
            total_mrr = sum(r.mrr for r in reqs)
            samples = [r.text for r in reqs[: self.top_k_samples]]

            clusters.append(
                FeatureCluster(
                    cluster_id=cluster_id,
                    label=label,
                    requests=list(reqs),
                    request_count=len(reqs),
                    unique_users=unique_users,
                    total_mrr=total_mrr,
                    sample_requests=samples,
                )
            )

        logger.info(
            f"[labeler] {len(clusters)} clusters labeled "
            f"— {time.perf_counter() - t0:.2f}s"
        )
        return clusters
