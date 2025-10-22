"""
Service account roles fetcher module
"""

from typing import List, Dict, Any
from datetime import datetime
from google.cloud import bigquery
import json

from .base import BaseFetcher
from shared import Config


class ServiceAccountRolesFetcher(BaseFetcher):
    """Fetcher for service account role assignments"""

    @property
    def table_name(self) -> str:
        return 'service_account_roles'

    @property
    def schema(self) -> List[bigquery.SchemaField]:
        return [
            bigquery.SchemaField("service_account", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("role", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("resource", "STRING"),
            bigquery.SchemaField("condition", "JSON"),
            bigquery.SchemaField("project_id", "STRING"),
            bigquery.SchemaField("ingestion_time", "TIMESTAMP")
        ]

    def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch service account role assignments - simplified for now"""
        if Config.ENABLE_SAMPLE_DATA:
            return self.get_sample_data()
        return []

    def get_sample_data(self) -> List[Dict[str, Any]]:
        """Get sample service account roles data"""
        return [
            {
                "service_account": f"compute@{Config.PROJECT_ID}.iam.gserviceaccount.com",
                "role": "roles/compute.instanceAdmin",
                "resource": f"projects/{Config.PROJECT_ID}",
                "condition": json.dumps({})
            }
        ]
