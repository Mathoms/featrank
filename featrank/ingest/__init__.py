"""Source connectors for ingesting feature requests."""

from featrank.ingest.base import BaseConnector
from featrank.ingest.csv_connector import CSVConnector

__all__ = ["BaseConnector", "CSVConnector"]
