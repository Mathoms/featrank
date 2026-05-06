"""Benchmark featrank composite ranker vs frequency-only and random baselines."""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.table import Table


def _dcg(relevances: Sequence[float], k: int) -> float:
    return sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(relevances[:k])
    )


def _ndcg(ranked_relevances: Sequence[float], ideal_relevances: Sequence[float], k: int) -> float:
    dcg = _dcg(ranked_relevances, k)
    idcg = _dcg(sorted(ideal_relevances, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


def load_clusters(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def assign_relevance(clusters: list[dict]) -> dict[int, float]:
    """Assign ground-truth relevance using a combination of mrr + request_count."""
    max_mrr = max(c["total_mrr"] for c in clusters) or 1
    max_count = max(c["request_count"] for c in clusters) or 1
    relevance: dict[int, float] = {}
    for c in clusters:
        rel = 0.6 * (c["total_mrr"] / max_mrr) + 0.4 * (c["request_count"] / max_count)
        relevance[c["cluster_id"]] = round(rel * 3)
    return relevance


def frequency_rank(clusters: list[dict]) -> list[dict]:
    return sorted(clusters, key=lambda c: c["request_count"], reverse=True)


def random_rank(clusters: list[dict], seed: int = 42) -> list[dict]:
    shuffled = list(clusters)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def featrank_rank(clusters: list[dict]) -> list[dict]:
    return sorted(clusters, key=lambda c: c.get("priority_score", 0), reverse=True)


def run_benchmark(clusters_path: Path) -> None:
    console = Console()
    clusters = load_clusters(clusters_path)

    if not clusters:
        console.print("[red]No clusters found in file[/red]")
        sys.exit(1)

    relevance = assign_relevance(clusters)
    console.print(f"\n[bold]Benchmark:[/bold] {len(clusters)} clusters\n")

    methods = {
        "Random": random_rank(clusters),
        "Frequency-only": frequency_rank(clusters),
        "featrank": featrank_rank(clusters),
    }

    table = Table(title="Benchmark Results")
    table.add_column("Method", style="bold")
    table.add_column("NDCG@5", justify="right")
    table.add_column("NDCG@10", justify="right")

    for name, ranked in methods.items():
        rels = [relevance.get(c["cluster_id"], 0) for c in ranked]
        ideal = list(relevance.values())
        ndcg5 = _ndcg(rels, ideal, k=5)
        ndcg10 = _ndcg(rels, ideal, k=10)
        table.add_row(name, f"{ndcg5:.3f}", f"{ndcg10:.3f}")

    console.print(table)


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/ranked.json")
    run_benchmark(path)
