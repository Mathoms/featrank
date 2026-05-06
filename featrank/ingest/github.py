"""GitHub issues connector."""

from __future__ import annotations

from typing import Iterator

from github import Github, GithubException
from loguru import logger

from featrank.config import settings
from featrank.ingest.base import BaseConnector
from featrank.schemas import RawRequest


class GitHubConnector(BaseConnector):
    """Fetch feature-request issues from a GitHub repository."""

    def __init__(self, repo: str | None = None, token: str | None = None) -> None:
        self.repo = repo or settings.github_repo
        self.token = token or settings.github_token
        if not self.repo:
            raise ValueError("GITHUB_REPO (owner/repo) is required for GitHubConnector")
        self._gh = Github(self.token) if self.token else Github()

    def fetch(self) -> Iterator[RawRequest]:
        logger.info(f"Fetching issues from GitHub: {self.repo}")
        try:
            repo = self._gh.get_repo(self.repo)
        except GithubException as exc:
            logger.error(f"GitHub repo not found: {exc}")
            return

        count = 0
        for issue in repo.get_issues(state="open", labels=["feature request"]):
            body = issue.body or ""
            text = f"{issue.title}. {body}".strip()
            yield RawRequest(
                id=str(issue.number),
                text=text,
                source="github",
                user_id=issue.user.login if issue.user else None,
                metadata={
                    "github_number": issue.number,
                    "reactions": issue.reactions.total_count,
                    "comments": issue.comments,
                    "labels": [lbl.name for lbl in issue.labels],
                },
            )
            count += 1

        logger.info(f"Loaded {count} GitHub issues from {self.repo}")
