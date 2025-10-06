"""Base configuration and shared utilities for BigQuery tools."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from google.cloud import bigquery

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "mgm-digitalconcierge")
DEFAULT_DATASET = os.getenv("DEFAULT_DATASET", "security_insights")
DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "security_findings")
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "100"))
ENABLE_EXPLORATION = os.getenv("ENABLE_EXPLORATION", "true").lower() == "true"


# Initialize BigQuery client
try:
    bq_client = bigquery.Client(project=PROJECT_ID)
    logger.info("✅ BigQuery client initialized for project: %s", PROJECT_ID)
except Exception as exc:  # pragma: no cover - executed only when client init fails
    logger.error("❌ Failed to initialize BigQuery client: %s", exc)
    bq_client = None


def check_client():
    """Ensure the BigQuery client is available before making requests."""

    if not bq_client:
        raise Exception("BigQuery client not initialized")
    return bq_client


@dataclass
class StructuredToolResponse:
    """Container that preserves human-readable and structured tool results."""

    summary: str
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return the structured payload for downstream consumers."""

        payload: Dict[str, Any] = {"summary": self.summary, "data": self.data}
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    def __str__(self) -> str:  # pragma: no cover - exercised via agent runtime
        """Preserve backwards compatibility with text-centric tool usage."""

        return self.summary
