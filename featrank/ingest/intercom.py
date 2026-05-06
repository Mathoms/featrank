"""Intercom API connector."""

from __future__ import annotations

from typing import Iterator

import requests
from loguru import logger

from featrank.config import settings
from featrank.ingest.base import BaseConnector
from featrank.schemas import RawRequest

_INTERCOM_BASE = "https://api.intercom.io"


class IntercomConnector(BaseConnector):
    """Fetch conversations / notes from Intercom."""

    def __init__(self, token: str | None = None) -> None:
        self.token = token or settings.intercom_token
        if not self.token:
            raise ValueError("INTERCOM_TOKEN is required for IntercomConnector")
        self._headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }

    def fetch(self) -> Iterator[RawRequest]:
        logger.info("Fetching conversations from Intercom")
        page = 1
        count = 0
        while True:
            resp = requests.get(
                f"{_INTERCOM_BASE}/conversations",
                headers=self._headers,
                params={"page": page, "per_page": 50},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            conversations = data.get("conversations", [])
            if not conversations:
                break

            for conv in conversations:
                body = (
                    conv.get("conversation_message", {}).get("body", "")
                    or conv.get("source", {}).get("body", "")
                )
                if not body:
                    continue
                yield RawRequest(
                    id=str(conv.get("id", count)),
                    text=body,
                    source="intercom",
                    user_id=str(conv.get("user", {}).get("id", "")),
                    metadata={"intercom_id": conv.get("id")},
                )
                count += 1

            if not data.get("pages", {}).get("next"):
                break
            page += 1

        logger.info(f"Loaded {count} conversations from Intercom")
