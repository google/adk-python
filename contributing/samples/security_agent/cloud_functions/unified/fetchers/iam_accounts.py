"""
IAM accounts fetcher module
"""

from typing import List, Dict, Any
from datetime import datetime
from google.cloud import bigquery
import json

from .base import BaseFetcher
from shared import Config


class IAMAccountsFetcher(BaseFetcher):
    """Fetcher for IAM bindings at project level"""

    @property
    def table_name(self) -> str:
        return 'iam_bindings'

    @property
    def schema(self) -> List[bigquery.SchemaField]:
        return [
            bigquery.SchemaField("binding_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("member", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("member_type", "STRING"),
            bigquery.SchemaField("role", "STRING"),
            bigquery.SchemaField("condition", "JSON"),
            bigquery.SchemaField("project_id", "STRING"),
            bigquery.SchemaField("ingestion_time", "TIMESTAMP")
        ]

    def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch IAM bindings - simplified for now"""
        if Config.ENABLE_SAMPLE_DATA:
            return self.get_sample_data()
        return []

    def get_sample_data(self) -> List[Dict[str, Any]]:
        """Get sample IAM bindings data"""
        return [
            {
                "binding_id": f"{Config.PROJECT_ID}_user_editor_001",
                "member": "user:admin@example.com",
                "member_type": "user",
                "role": "roles/editor",
                "condition": json.dumps({}),
            }
        ]
