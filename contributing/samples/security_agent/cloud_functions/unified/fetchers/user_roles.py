"""
User roles fetcher module
"""

from typing import List, Dict, Any
from datetime import datetime
from google.cloud import bigquery
import json

from .base import BaseFetcher
from shared import Config


class UserRolesFetcher(BaseFetcher):
    """Fetcher for user IAM role assignments"""

    @property
    def table_name(self) -> str:
        return 'user_roles'

    @property
    def schema(self) -> List[bigquery.SchemaField]:
        return [
            bigquery.SchemaField("user_email", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("role", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("resource", "STRING"),
            bigquery.SchemaField("condition", "JSON"),
            bigquery.SchemaField("project_id", "STRING"),
            bigquery.SchemaField("ingestion_time", "TIMESTAMP")
        ]

    def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch user role assignments - simplified for now"""
        if Config.ENABLE_SAMPLE_DATA:
            return self.get_sample_data()
        return []

    def get_sample_data(self) -> List[Dict[str, Any]]:
        """Get sample user roles data"""
        return [
            {
                "user_email": "admin@example.com",
                "role": "roles/owner",
                "resource": f"projects/{Config.PROJECT_ID}",
                "condition": json.dumps({})
            }
        ]
