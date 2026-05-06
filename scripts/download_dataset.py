"""Pull feature request data from public GitHub repos for development."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from loguru import logger


REPOS = [
    "microsoft/vscode",
    "vercel/next.js",
    "supabase/supabase",
]

OUTPUT = Path("data/raw/github_issues.csv")


def fetch_github_issues(repo: str, token: str | None = None, max_issues: int = 200) -> list[dict]:
    from github import Github, GithubException

    gh = Github(token) if token else Github()
    try:
        r = gh.get_repo(repo)
    except GithubException as e:
        logger.error(f"Cannot access {repo}: {e}")
        return []

    rows = []
    for issue in r.get_issues(state="open", labels=["feature request"])[:max_issues]:
        body = issue.body or ""
        rows.append({
            "id": str(issue.number),
            "text": f"{issue.title}. {body[:300]}".strip(),
            "source": "github",
            "user_id": issue.user.login if issue.user else "",
            "mrr": 0.0,
            "created_at": issue.created_at.isoformat(),
            "repo": repo,
        })
    logger.info(f"{repo}: {len(rows)} issues")
    return rows


def main() -> None:
    import os

    token = os.getenv("GITHUB_TOKEN")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for repo in REPOS:
        all_rows.extend(fetch_github_issues(repo, token=token))

    if not all_rows:
        logger.warning("No issues fetched. Check GITHUB_TOKEN and repo labels.")
        sys.exit(1)

    fieldnames = ["id", "text", "source", "user_id", "mrr", "created_at", "repo"]
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    logger.info(f"Saved {len(all_rows)} issues → {OUTPUT}")


if __name__ == "__main__":
    main()
