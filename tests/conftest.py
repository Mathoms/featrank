"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_csv_path() -> Path:
    return FIXTURES_DIR / "sample_requests.csv"


@pytest.fixture
def sample_roadmap_path() -> Path:
    return FIXTURES_DIR / "sample_roadmap.txt"


@pytest.fixture
def raw_requests():
    from featrank.schemas import RawRequest

    return [
        RawRequest(id="1", text="I need dark mode support", source="intercom", user_id="u1", mrr=299.0),
        RawRequest(id="2", text="Please add dark theme for night use", source="slack", user_id="u2", mrr=599.0),
        RawRequest(id="3", text="Dark mode would be great", source="zendesk", user_id="u3", mrr=99.0),
        RawRequest(id="4", text="CSV export functionality is needed", source="csv", user_id="u4", mrr=0.0),
        RawRequest(id="5", text="Export my data to spreadsheet please", source="intercom", user_id="u5", mrr=49.0),
        RawRequest(id="6", text="Need bulk export to CSV", source="zendesk", user_id="u6", mrr=199.0),
    ]
