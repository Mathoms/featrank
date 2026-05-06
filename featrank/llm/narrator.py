"""Generate PM-style diagnosis narratives for feature clusters."""

from __future__ import annotations

import time
from typing import Sequence

from loguru import logger

from featrank.llm._client import get_llm_client
from featrank.schemas import FeatureCluster, PrioritizedCluster

_DIAGNOSIS_SYSTEM = (
    "You are a senior product manager analyzing feature requests. "
    "Be specific. No filler. Think like a PM who reads P&L statements."
)

_DIAGNOSIS_TEMPLATE = """\
Feature cluster: "{label}"
Request count: {count}
MRR affected: ${mrr:.0f}
Sample requests:
{samples}

Write a 2-3 sentence diagnosis covering:
1. What users actually need (not what they asked for)
2. Business urgency signal
3. Recommended action (build now / investigate / defer)"""


class Narrator:
    """Generate LLM-powered diagnosis for each cluster."""

    def narrate(self, cluster: FeatureCluster) -> str:
        samples = "\n".join(f"- {s}" for s in cluster.sample_requests)
        prompt = _DIAGNOSIS_TEMPLATE.format(
            label=cluster.label,
            count=cluster.request_count,
            mrr=cluster.total_mrr,
            samples=samples,
        )
        try:
            client = get_llm_client()
            return client.chat(system=_DIAGNOSIS_SYSTEM, user=prompt).strip()
        except Exception as exc:
            logger.warning(f"[narrator] Narration failed for '{cluster.label}': {exc}")
            return (
                f"Cluster '{cluster.label}' has {cluster.request_count} requests "
                f"from {cluster.unique_users} users."
            )

    def narrate_all(
        self, clusters: Sequence[FeatureCluster | PrioritizedCluster]
    ) -> list[FeatureCluster | PrioritizedCluster]:
        t0 = time.perf_counter()
        result = []
        for cluster in clusters:
            diagnosis = self.narrate(cluster)
            updated = cluster.model_copy(update={"llm_diagnosis": diagnosis})
            result.append(updated)
        logger.info(
            f"[narrator] {len(result)} clusters narrated "
            f"— {time.perf_counter() - t0:.2f}s"
        )
        return result
