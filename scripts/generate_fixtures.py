"""Generate synthetic feature request fixtures for testing."""

from __future__ import annotations

import csv
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

SOURCES = ["intercom", "zendesk", "csv", "slack", "github"]

TEMPLATES = [
    # Dark mode cluster
    "Would love dark mode support",
    "Please add a dark theme option",
    "Dark mode would be amazing for night use",
    "Can you add dark mode to reduce eye strain?",
    "Dark theme support is really needed",
    "Please implement dark mode ASAP",
    "Night mode / dark theme please",
    "My eyes hurt at night, dark mode?",
    # CSV export cluster
    "Please add CSV export functionality",
    "I need to export my data to a spreadsheet",
    "CSV download for reports would be great",
    "Need bulk export to CSV",
    "Export data as Excel or CSV",
    "Allow me to export all records to CSV",
    "Data export feature is missing, need CSV",
    # Bulk actions cluster
    "Bulk delete would save so much time",
    "Please add multi-select and bulk actions",
    "Bulk edit / delete for records",
    "Need to select multiple items at once",
    "Mass actions on selected rows please",
    "Support bulk operations on tasks",
    # Mobile app cluster
    "The mobile app needs a lot of work",
    "Mobile experience is really poor",
    "Please improve the iOS app",
    "Android app crashes frequently",
    "Mobile app is too slow",
    "Need push notifications on mobile",
    "Mobile version lags behind desktop",
    # SSO / Auth cluster
    "Please support SSO login",
    "SAML SSO integration is needed for our company",
    "Google SSO would be great",
    "Need Okta integration for enterprise",
    "Two-factor authentication support please",
    "Single sign-on via SAML is a blocker",
    # Webhooks cluster
    "Webhook support would unlock many integrations",
    "Please add outgoing webhooks",
    "Need webhook events for automation",
    "REST webhooks for status changes",
    "Zapier/Webhook integration please",
    # Analytics cluster
    "Better reporting and analytics dashboard",
    "I need more advanced analytics",
    "Custom reports with date range filters",
    "Analytics for team performance would be useful",
    "Visual charts for usage data",
    "More detailed usage statistics",
    # API limits cluster
    "The API rate limits are too low",
    "Please increase API rate limits",
    "We keep hitting the API rate limit",
    "Higher API quotas needed for enterprise",
    # Audit log cluster
    "Need an audit log for compliance",
    "Activity history and audit trail",
    "Who did what and when — audit log please",
    "Compliance requires full audit history",
    # Onboarding cluster
    "The onboarding flow is confusing",
    "Better getting started guide please",
    "Interactive onboarding tutorial would help new users",
    "Setup wizard for first-time users",
    # Noise / outliers
    "The color of the button is slightly off",
    "Can you change the font size?",
    "Great product overall!",
    "Random feature idea: integrate with fax machines",
    "Not sure what I need but something is missing",
]

MRR_BUCKETS = [
    (0.0, 0.55),
    (49.0, 0.10),
    (99.0, 0.10),
    (199.0, 0.08),
    (299.0, 0.07),
    (599.0, 0.05),
    (999.0, 0.03),
    (4999.0, 0.015),
    (9999.0, 0.005),
]


def _sample_mrr() -> float:
    vals, weights = zip(*MRR_BUCKETS)
    return float(random.choices(vals, weights=weights, k=1)[0])


def generate(count: int = 500, output: Path | None = None) -> Path:
    if output is None:
        output = Path(__file__).parent.parent / "tests" / "fixtures" / "sample_requests.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    base_date = datetime(2024, 1, 1)
    rows = []

    for i in range(1, count + 1):
        text = rng.choice(TEMPLATES)
        if rng.random() < 0.15:
            text = text + " " + rng.choice(["PLEASE", "asap", "!!!", "this is critical"])
        source = rng.choice(SOURCES)
        user_id = f"u_{rng.randint(1, int(count * 0.6)):04d}"
        mrr = _sample_mrr()
        created_at = base_date + timedelta(days=rng.randint(0, 365))
        rows.append({
            "id": str(i),
            "text": text,
            "source": source,
            "user_id": user_id,
            "mrr": mrr,
            "created_at": created_at.isoformat(),
        })

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "source", "user_id", "mrr", "created_at"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {count} synthetic requests → {output}")
    return output


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    generate(count=count)
