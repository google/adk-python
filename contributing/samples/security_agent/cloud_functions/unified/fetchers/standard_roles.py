"""
Standard roles fetcher module
"""

from typing import List, Dict, Any
from datetime import datetime
from google.cloud import bigquery
import json

from .base import BaseFetcher
from shared import Config


class StandardRolesFetcher(BaseFetcher):
    """Fetcher for predefined GCP IAM roles"""

    @property
    def table_name(self) -> str:
        return 'iam_standard_roles'

    @property
    def schema(self) -> List[bigquery.SchemaField]:
        return [
            bigquery.SchemaField("role_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("title", "STRING"),
            bigquery.SchemaField("description", "STRING"),
            bigquery.SchemaField("stage", "STRING"),
            bigquery.SchemaField("included_permissions", "JSON"),
            bigquery.SchemaField("permission_count", "INTEGER"),
            bigquery.SchemaField("project_id", "STRING"),
            bigquery.SchemaField("ingestion_time", "TIMESTAMP")
        ]

    def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch standard roles - simplified for now"""
        if Config.ENABLE_SAMPLE_DATA:
            return self.get_sample_data()
        return []

    def get_sample_data(self) -> List[Dict[str, Any]]:
        """Get sample standard roles data"""
        return [
            {
                "role_id": "roles/viewer",
                "title": "Viewer",
                "description": "Read access to all resources",
                "stage": "GA",
                "included_permissions": json.dumps(["resourcemanager.projects.get", "resourcemanager.projects.list"]),
                "permission_count": 2
            }
        ]
