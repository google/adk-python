"""
Firewall rules fetcher module
"""

from typing import List, Dict, Any
from datetime import datetime
from google.cloud import bigquery
import json

from .base import BaseFetcher
from shared import Config


class FirewallRulesFetcher(BaseFetcher):
    """Fetcher for VPC firewall rules"""

    @property
    def table_name(self) -> str:
        return 'firewall_rules'

    @property
    def schema(self) -> List[bigquery.SchemaField]:
        return [
            bigquery.SchemaField("rule_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("network", "STRING"),
            bigquery.SchemaField("priority", "INTEGER"),
            bigquery.SchemaField("direction", "STRING"),
            bigquery.SchemaField("source_ranges", "JSON"),
            bigquery.SchemaField("destination_ranges", "JSON"),
            bigquery.SchemaField("allowed", "JSON"),
            bigquery.SchemaField("denied", "JSON"),
            bigquery.SchemaField("source_tags", "JSON"),
            bigquery.SchemaField("target_tags", "JSON"),
            bigquery.SchemaField("source_service_accounts", "JSON"),
            bigquery.SchemaField("target_service_accounts", "JSON"),
            bigquery.SchemaField("disabled", "BOOLEAN"),
            bigquery.SchemaField("description", "STRING"),
            bigquery.SchemaField("creation_timestamp", "TIMESTAMP"),
            bigquery.SchemaField("project_id", "STRING"),
            bigquery.SchemaField("ingestion_time", "TIMESTAMP")
        ]

    def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch firewall rules - simplified for now"""
        # This is a placeholder implementation
        if Config.ENABLE_SAMPLE_DATA:
            return self.get_sample_data()
        return []

    def get_sample_data(self) -> List[Dict[str, Any]]:
        """Get sample firewall rules data"""
        return [
            {
                "rule_id": "sample-firewall-001",
                "name": "default-allow-http",
                "network": "default",
                "priority": 1000,
                "direction": "INGRESS",
                "source_ranges": json.dumps(["0.0.0.0/0"]),
                "destination_ranges": json.dumps([]),
                "allowed": json.dumps([{"IPProtocol": "tcp", "ports": ["80"]}]),
                "denied": json.dumps([]),
                "source_tags": json.dumps([]),
                "target_tags": json.dumps(["http-server"]),
                "source_service_accounts": json.dumps([]),
                "target_service_accounts": json.dumps([]),
                "disabled": False,
                "description": "Allow HTTP traffic",
                "creation_timestamp": datetime.utcnow().isoformat()
            }
        ]
