"""Text normalization and cleaning for raw feature requests."""

from __future__ import annotations

import re
import time
from typing import Sequence

from loguru import logger

from featrank.schemas import RawRequest


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_SPECIAL_CHARS_RE = re.compile(r"[^\w\s\.\!\?\,\'\-]")


class TextCleaner:
    """Normalize and clean raw request text.

    Steps applied in order:
    1. Strip HTML tags
    2. Strip URLs
    3. Lowercase
    4. Remove non-printable / special characters (keep basic punctuation)
    5. Collapse whitespace
    6. Strip leading/trailing whitespace
    """

    def __init__(self, min_length: int = 5) -> None:
        self.min_length = min_length

    def clean(self, text: str) -> str:
        text = _HTML_TAG_RE.sub(" ", text)
        text = _URL_RE.sub(" ", text)
        text = text.lower()
        text = _SPECIAL_CHARS_RE.sub(" ", text)
        text = _WHITESPACE_RE.sub(" ", text)
        return text.strip()

    def clean_requests(
        self, requests: Sequence[RawRequest]
    ) -> tuple[list[RawRequest], list[str]]:
        """Return (kept_requests, cleaned_texts).

        Requests with cleaned text shorter than `min_length` are dropped.
        """
        t0 = time.perf_counter()
        kept: list[RawRequest] = []
        texts: list[str] = []
        dropped = 0

        for req in requests:
            cleaned = self.clean(req.text)
            if len(cleaned) < self.min_length:
                dropped += 1
                continue
            kept.append(req)
            texts.append(cleaned)

        elapsed = time.perf_counter() - t0
        logger.info(
            f"[cleaner] {len(kept)} kept, {dropped} dropped "
            f"(min_length={self.min_length}) — {elapsed:.2f}s"
        )
        return kept, texts
