"""Markdown priority report formatter."""

from __future__ import annotations

from typing import Sequence

from featrank.schemas import PrioritizedCluster


class MarkdownFormatter:
    """Render a ranked cluster list as a Markdown report."""

    def format(self, clusters: Sequence[PrioritizedCluster]) -> str:
        lines: list[str] = [
            "# featrank Priority Report",
            "",
            f"**{len(clusters)} feature clusters ranked by business value**",
            "",
            "| Rank | Feature | Score | Requests | Users | MRR | Freq | Seg | GH | Road |",
            "|------|---------|-------|----------|-------|-----|------|-----|----|------|",
        ]

        for c in clusters:
            lines.append(
                f"| {c.priority_rank} "
                f"| {c.label} "
                f"| **{c.priority_score:.1f}** "
                f"| {c.request_count} "
                f"| {c.unique_users} "
                f"| ${c.total_mrr:,.0f} "
                f"| {c.score_frequency:.2f} "
                f"| {c.score_segment_value:.2f} "
                f"| {c.score_github_signal:.2f} "
                f"| {c.score_roadmap_fit:.2f} |"
            )

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Cluster Details")
        lines.append("")

        for c in clusters:
            lines.append(f"### #{c.priority_rank} — {c.label}")
            lines.append("")
            lines.append(f"**Priority Score:** {c.priority_score:.1f}/100")
            lines.append(f"**Requests:** {c.request_count} | **Users:** {c.unique_users} | **MRR:** ${c.total_mrr:,.0f}")
            lines.append("")
            lines.append("**Sample Requests:**")
            for sample in c.sample_requests:
                lines.append(f"- _{sample}_")
            if c.llm_diagnosis:
                lines.append("")
                lines.append(f"**PM Diagnosis:** {c.llm_diagnosis}")
            if c.recommendation:
                lines.append("")
                lines.append(f"**Recommendation:** {c.recommendation}")
            lines.append("")

        return "\n".join(lines)
