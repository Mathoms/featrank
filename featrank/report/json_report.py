"""JSON report formatter."""

from __future__ import annotations

import json
from typing import Sequence

from featrank.schemas import PrioritizedCluster


class JSONFormatter:
    """Serialize ranked clusters to pretty-printed JSON."""

    def format(self, clusters: Sequence[PrioritizedCluster]) -> str:
        data = [c.model_dump() for c in clusters]
        return json.dumps(data, indent=2, default=str)
