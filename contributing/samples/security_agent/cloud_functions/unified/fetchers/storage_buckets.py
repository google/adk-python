"""
Storage buckets fetcher module
"""

from typing import List, Dict, Any
from datetime import datetime
from google.cloud import bigquery
import json

from .base import BaseFetcher
from shared import Config


class StorageBucketsFetcher(BaseFetcher):
    """Fetcher for Cloud Storage buckets"""

    @property
    def table_name(self) -> str:
        return 'storage_buckets'

    @property
    def schema(self) -> List[bigquery.SchemaField]:
        return [
            bigquery.SchemaField("bucket_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("location", "STRING"),
            bigquery.SchemaField("storage_class", "STRING"),
            bigquery.SchemaField("versioning_enabled", "BOOLEAN"),
            bigquery.SchemaField("lifecycle_rules", "JSON"),
            bigquery.SchemaField("iam_configuration", "JSON"),
            bigquery.SchemaField("labels", "JSON"),
            bigquery.SchemaField("encryption", "JSON"),
            bigquery.SchemaField("retention_policy", "JSON"),
            bigquery.SchemaField("cors", "JSON"),
            bigquery.SchemaField("creation_time", "TIMESTAMP"),
            bigquery.SchemaField("project_id", "STRING"),
            bigquery.SchemaField("ingestion_time", "TIMESTAMP")
        ]

    def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch storage buckets - simplified for now"""
        if Config.ENABLE_SAMPLE_DATA:
            return self.get_sample_data()
        return []

    def get_sample_data(self) -> List[Dict[str, Any]]:
        """Get sample storage buckets data"""
        return [
            {
                "bucket_name": f"{Config.PROJECT_ID}-data",
                "location": "US",
                "storage_class": "STANDARD",
                "versioning_enabled": True,
                "lifecycle_rules": json.dumps([]),
                "iam_configuration": json.dumps({"uniformBucketLevelAccess": {"enabled": True}}),
                "labels": json.dumps({"env": "production"}),
                "encryption": json.dumps({"defaultKmsKeyName": None}),
                "retention_policy": json.dumps({}),
                "cors": json.dumps([]),
                "creation_time": datetime.utcnow().isoformat()
            }
        ]
