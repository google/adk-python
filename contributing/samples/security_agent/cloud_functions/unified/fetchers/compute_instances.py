"""
Compute instances fetcher module
"""

from typing import List, Dict, Any
from datetime import datetime
from google.cloud import bigquery
import json

from .base import BaseFetcher
from shared import get_authenticated_client, Config


class ComputeInstancesFetcher(BaseFetcher):
    """Fetcher for Compute Engine instances"""

    @property
    def table_name(self) -> str:
        return 'compute_instances'

    @property
    def schema(self) -> List[bigquery.SchemaField]:
        return [
            bigquery.SchemaField("instance_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("zone", "STRING"),
            bigquery.SchemaField("machine_type", "STRING"),
            bigquery.SchemaField("status", "STRING"),
            bigquery.SchemaField("network_interfaces", "JSON"),
            bigquery.SchemaField("disks", "JSON"),
            bigquery.SchemaField("service_accounts", "JSON"),
            bigquery.SchemaField("labels", "JSON"),
            bigquery.SchemaField("metadata", "JSON"),
            bigquery.SchemaField("tags", "JSON"),
            bigquery.SchemaField("creation_timestamp", "TIMESTAMP"),
            bigquery.SchemaField("project_id", "STRING"),
            bigquery.SchemaField("ingestion_time", "TIMESTAMP")
        ]

    def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch compute instances"""
        instances = []

        try:
            compute_client = get_authenticated_client('compute')

            # List all instances across all zones
            aggregated_list = compute_client.aggregated_list(project=Config.PROJECT_ID)

            for zone, response in aggregated_list:
                if response.instances:
                    for instance in response.instances:
                        instances.append(self._convert_instance_to_dict(instance, zone))

        except Exception as e:
            self.logger.warning(f"Failed to fetch compute instances: {e}")
            if Config.ENABLE_SAMPLE_DATA:
                instances = self.get_sample_data()

        return instances

    def _convert_instance_to_dict(self, instance, zone: str) -> Dict[str, Any]:
        """Convert instance object to dictionary"""
        zone_name = zone.split('/')[-1] if '/' in zone else zone

        return {
            "instance_id": str(instance.id) if hasattr(instance, 'id') else instance.name,
            "name": instance.name,
            "zone": zone_name,
            "machine_type": instance.machine_type.split('/')[-1] if instance.machine_type else None,
            "status": instance.status,
            "network_interfaces": json.dumps([{
                "name": ni.name,
                "network": ni.network,
                "network_ip": ni.network_i_p if hasattr(ni, 'network_i_p') else None,
                "access_configs": [{
                    "name": ac.name,
                    "nat_ip": ac.nat_i_p if hasattr(ac, 'nat_i_p') else None
                } for ac in (ni.access_configs or [])]
            } for ni in (instance.network_interfaces or [])]),
            "disks": json.dumps([{
                "device_name": d.device_name,
                "source": d.source.split('/')[-1] if d.source else None,
                "boot": d.boot
            } for d in (instance.disks or [])]),
            "service_accounts": json.dumps([{
                "email": sa.email,
                "scopes": list(sa.scopes or [])
            } for sa in (instance.service_accounts or [])]),
            "labels": json.dumps(dict(instance.labels or {})),
            "metadata": json.dumps({}),
            "tags": json.dumps(list(instance.tags.items or []) if instance.tags else []),
            "creation_timestamp": instance.creation_timestamp if hasattr(instance, 'creation_timestamp') else None
        }

    def get_sample_data(self) -> List[Dict[str, Any]]:
        """Get sample compute instances data"""
        return [
            {
                "instance_id": "sample-instance-001",
                "name": "web-server-1",
                "zone": "us-central1-a",
                "machine_type": "n1-standard-1",
                "status": "RUNNING",
                "network_interfaces": json.dumps([{
                    "name": "nic0",
                    "network": "default",
                    "network_ip": "10.128.0.2",
                    "access_configs": [{"name": "External NAT", "nat_ip": "34.123.45.67"}]
                }]),
                "disks": json.dumps([{
                    "device_name": "boot",
                    "source": "web-server-1-boot",
                    "boot": True
                }]),
                "service_accounts": json.dumps([{
                    "email": f"default@{Config.PROJECT_ID}.iam.gserviceaccount.com",
                    "scopes": ["https://www.googleapis.com/auth/cloud-platform"]
                }]),
                "labels": json.dumps({"env": "production", "team": "backend"}),
                "metadata": json.dumps({}),
                "tags": json.dumps(["http-server", "https-server"]),
                "creation_timestamp": datetime.utcnow().isoformat()
            }
        ]
