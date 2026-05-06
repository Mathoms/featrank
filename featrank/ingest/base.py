"""Abstract base class for all source connectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from featrank.schemas import RawRequest


class BaseConnector(ABC):
    """All connectors must implement `fetch()` which yields `RawRequest` objects."""

    @abstractmethod
    def fetch(self) -> Iterator[RawRequest]:
        """Yield raw feature requests from the source."""
        ...

    def fetch_all(self) -> list[RawRequest]:
        return list(self.fetch())
