"""CSV / spreadsheet connector."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd
from loguru import logger

from featrank.ingest.base import BaseConnector
from featrank.schemas import RawRequest


class CSVConnector(BaseConnector):
    """Load feature requests from a CSV file.

    Expected columns: id, text, source, user_id, mrr, created_at
    Only `text` is strictly required; all others are optional.
    """

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def fetch(self) -> Iterator[RawRequest]:
        logger.info(f"Reading CSV: {self.file_path}")
        df = pd.read_csv(self.file_path)

        required = {"text"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        count = 0
        for i, row in df.iterrows():
            yield RawRequest(
                id=str(row.get("id", i)),
                text=str(row["text"]),
                source=str(row.get("source", "csv")),
                user_id=str(row["user_id"]) if pd.notna(row.get("user_id")) else None,
                mrr=float(row["mrr"]) if pd.notna(row.get("mrr")) else 0.0,
                created_at=row.get("created_at"),
                metadata={},
            )
            count += 1

        logger.info(f"Loaded {count} requests from {self.file_path.name}")
