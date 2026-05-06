"""Zendesk API connector."""

from __future__ import annotations

from typing import Iterator

import requests
from loguru import logger

from featrank.config import settings
from featrank.ingest.base import BaseConnector
from featrank.schemas import RawRequest


class ZendeskConnector(BaseConnector):
    """Fetch tickets from Zendesk."""

    def __init__(
        self,
        subdomain: str | None = None,
        email: str | None = None,
        token: str | None = None,
    ) -> None:
        self.subdomain = subdomain or settings.zendesk_subdomain
        self.email = email or settings.zendesk_email
        self.token = token or settings.zendesk_api_token
        if not all([self.subdomain, self.email, self.token]):
            raise ValueError(
                "ZENDESK_SUBDOMAIN, ZENDESK_EMAIL, ZENDESK_API_TOKEN are all required"
            )
        self._base = f"https://{self.subdomain}.zendesk.com/api/v2"
        self._auth = (f"{self.email}/token", self.token)

    def fetch(self) -> Iterator[RawRequest]:
        logger.info(f"Fetching tickets from Zendesk ({self.subdomain})")
        url = f"{self._base}/tickets.json?per_page=100"
        count = 0
        while url:
            resp = requests.get(url, auth=self._auth, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            for ticket in data.get("tickets", []):
                desc = ticket.get("description", "")
                if not desc:
                    continue
                yield RawRequest(
                    id=str(ticket["id"]),
                    text=desc,
                    source="zendesk",
                    user_id=str(ticket.get("requester_id", "")),
                    metadata={
                        "zendesk_id": ticket["id"],
                        "status": ticket.get("status"),
                        "priority": ticket.get("priority"),
                    },
                )
                count += 1

            url = data.get("next_page")

        logger.info(f"Loaded {count} tickets from Zendesk")
