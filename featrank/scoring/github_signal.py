"""GitHub issue/PR correlation scorer."""

from __future__ import annotations

import time
from typing import Sequence

import requests
from loguru import logger

from featrank.config import settings
from featrank.schemas import FeatureCluster

_GITHUB_SEARCH = "https://api.github.com/search/issues"
_ISSUE_WEIGHT = 1.0
_PR_WEIGHT = 2.0


class GitHubSignalScorer:
    """Score clusters based on correlated GitHub issues + PRs.

    PR weight > issue weight because open PRs signal active engineering work.
    Gracefully degrades to score=0 if GitHub token or repo is missing.
    """

    def __init__(
        self,
        repo: str | None = None,
        token: str | None = None,
    ) -> None:
        self.repo = repo or settings.github_repo
        self.token = token or settings.github_token
        self._cache: dict[str, float] = {}

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Accept": "application/vnd.github+json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _search_score(self, query: str) -> float:
        if query in self._cache:
            return self._cache[query]

        if not self.repo:
            return 0.0

        try:
            resp = requests.get(
                _GITHUB_SEARCH,
                headers=self._headers(),
                params={"q": f"{query} repo:{self.repo}", "per_page": 10},
                timeout=10,
            )
            resp.raise_for_status()
            items = resp.json().get("items", [])
            raw = sum(
                _PR_WEIGHT if i.get("pull_request") else _ISSUE_WEIGHT for i in items
            )
            self._cache[query] = raw
            time.sleep(0.1)
            return raw
        except Exception as exc:
            logger.warning(f"[github_signal] Search failed for '{query}': {exc}")
            self._cache[query] = 0.0
            return 0.0

    def score(self, clusters: Sequence[FeatureCluster]) -> list[float]:
        if not self.repo:
            logger.info("[github_signal] No GITHUB_REPO configured — returning zeros")
            return [0.0] * len(clusters)

        raw_scores: list[float] = []
        for cluster in clusters:
            query = cluster.label.replace(" / ", " ").replace("/", " ")
            raw_scores.append(self._search_score(query))

        max_val = max(raw_scores) if raw_scores else 1
        if max_val == 0:
            return [0.0] * len(clusters)
        return [s / max_val for s in raw_scores]
