#!/usr/bin/env python3
"""
Cloud Function to fetch Compute Engine instances and load to BigQuery
Runs independently on a schedule (every 2 hours)
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any
from google.cloud import compute_v1
from google.cloud import bigquery
from google.api_core import exceptions


def get_external_ip(instance) -> str:
    """Extract external IP from instance if available"""
    for interface in instance.network_interfaces:
        for config in interface.access_configs or []:
            if config.nat_i_p:
                return config.nat_i_p
    return None


def get_internal_ip(instance) -> str:
    """Extract internal IP from instance"""
    for interface in instance.network_interfaces:
        if interface.network_i_p:
            return interface.network_i_p
    return None


def fetch_compute_instances(request):
    """
    Cloud Function entry point - fetches all compute instances
    and refreshes BigQuery table

    Args:
        request: HTTP request object (can contain force_refresh flag)

    Returns:
        JSON response with status and record count
    """

    # Initialize clients
    compute_client = compute_v1.InstancesClient()
    zones_client = compute_v1.ZonesClient()
    bq_client = bigquery.Client()

    # Get configuration from environment
    project_id = os.environ.get('PROJECT_ID', 'mgm-digitalconcierge')
    dataset_id = os.environ.get('BQ_DATASET_ID', 'security_insights')

    print(f"Starting compute instances refresh for project: {project_id}")

    try:
        # Fetch all instances across all zones
        instances_data = []
        zones = zones_client.list(project=project_id)

        for zone in zones:
            try:
                zone_instances = compute_client.list(
                    project=project_id,
                    zone=zone.name
                )

                for instance in zone_instances:
                    # Extract security-relevant information
                    instance_record = {
                        'instance_id': str(instance.id),
                        'name': instance.name,
                        'zone': zone.name,
                        'region': zone.name.rsplit('-', 1)[0],  # Extract region from zone
                        'machine_type': instance.machine_type.split('/')[-1],
                        'status': instance.status,
                        'external_ip': get_external_ip(instance),
                        'internal_ip': get_internal_ip(instance),
                        'created_at': instance.creation_timestamp,
                        'labels': json.dumps(dict(instance.labels) if instance.labels else {}),
                        'network_tags': list(instance.tags.items) if instance.tags else [],
                        'service_accounts': [sa.email for sa in (instance.service_accounts or [])],
                        'disks': [disk.source.split('/')[-1] for disk in instance.disks],
                        'can_ip_forward': instance.can_ip_forward,
                        'deletion_protection': instance.deletion_protection,
                        'shielded_instance': json.dumps({
                            'enable_secure_boot': instance.shielded_instance_config.enable_secure_boot if instance.shielded_instance_config else False,
                            'enable_vtpm': instance.shielded_instance_config.enable_vtpm if instance.shielded_instance_config else False,
                            'enable_integrity_monitoring': instance.shielded_instance_config.enable_integrity_monitoring if instance.shielded_instance_config else False
                        }),
                        'metadata_items': json.dumps({
                            item.key: item.value for item in (instance.metadata.items or [])
                        }),
                        'last_refreshed': datetime.utcnow().isoformat(),
                        'refresh_job': 'scheduled_2h'  # Track which job updated this
                    }
                    instances_data.append(instance_record)

            except exceptions.PermissionDenied as e:
                print(f"Permission denied for zone {zone.name}: {e}")
                continue
            except Exception as e:
                print(f"Error fetching instances in zone {zone.name}: {e}")
                continue

        # Load data to BigQuery
        if instances_data:
            table_id = f"{project_id}.{dataset_id}.compute_instances"

            # Define schema for the table
            schema = [
                bigquery.SchemaField("instance_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("name", "STRING"),
                bigquery.SchemaField("zone", "STRING"),
                bigquery.SchemaField("region", "STRING"),
                bigquery.SchemaField("machine_type", "STRING"),
                bigquery.SchemaField("status", "STRING"),
                bigquery.SchemaField("external_ip", "STRING"),
                bigquery.SchemaField("internal_ip", "STRING"),
                bigquery.SchemaField("created_at", "TIMESTAMP"),
                bigquery.SchemaField("labels", "JSON"),
                bigquery.SchemaField("network_tags", "STRING", mode="REPEATED"),
                bigquery.SchemaField("service_accounts", "STRING", mode="REPEATED"),
                bigquery.SchemaField("disks", "STRING", mode="REPEATED"),
                bigquery.SchemaField("can_ip_forward", "BOOLEAN"),
                bigquery.SchemaField("deletion_protection", "BOOLEAN"),
                bigquery.SchemaField("shielded_instance", "JSON"),
                bigquery.SchemaField("metadata_items", "JSON"),
                bigquery.SchemaField("last_refreshed", "TIMESTAMP"),
                bigquery.SchemaField("refresh_job", "STRING"),
            ]

            # Configure load job
            job_config = bigquery.LoadJobConfig(
                schema=schema,
                write_disposition="WRITE_TRUNCATE",  # Replace entire table
                create_disposition="CREATE_IF_NEEDED",
                schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION]
            )

            # Load data
            job = bq_client.load_table_from_json(
                instances_data,
                table_id,
                job_config=job_config
            )
            job.result()  # Wait for job to complete

            print(f"Successfully loaded {len(instances_data)} instances to BigQuery")

            # Log refresh metadata
            metadata_table_id = f"{project_id}.{dataset_id}.refresh_metadata"
            metadata_record = [{
                'table_name': 'compute_instances',
                'refresh_time': datetime.utcnow().isoformat(),
                'record_count': len(instances_data),
                'status': 'success',
                'refresh_type': 'scheduled',
                'error_message': None
            }]

            try:
                metadata_job = bq_client.load_table_from_json(
                    metadata_record,
                    metadata_table_id,
                    job_config=bigquery.LoadJobConfig(
                        write_disposition="WRITE_APPEND",
                        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION]
                    )
                )
                metadata_job.result()
            except Exception as e:
                print(f"Warning: Could not update refresh metadata: {e}")

            return {
                'status': 'success',
                'records': len(instances_data),
                'table': table_id,
                'timestamp': datetime.utcnow().isoformat()
            }
        else:
            print("No instances found to load")
            return {
                'status': 'success',
                'records': 0,
                'message': 'No instances found',
                'timestamp': datetime.utcnow().isoformat()
            }

    except Exception as e:
        error_msg = f"Error in fetch_compute_instances: {str(e)}"
        print(error_msg)

        # Try to log error to metadata table
        try:
            metadata_table_id = f"{project_id}.{dataset_id}.refresh_metadata"
            error_record = [{
                'table_name': 'compute_instances',
                'refresh_time': datetime.utcnow().isoformat(),
                'record_count': 0,
                'status': 'failed',
                'refresh_type': 'scheduled',
                'error_message': str(e)[:1000]  # Truncate error message
            }]

            bq_client.load_table_from_json(
                error_record,
                metadata_table_id,
                job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
            ).result()
        except:
            pass  # Silent fail on metadata logging

        return {
            'status': 'error',
            'error': error_msg,
            'timestamp': datetime.utcnow().isoformat()
        }, 500


# For local testing
if __name__ == "__main__":
    class MockRequest:
        def __init__(self):
            self.json = {}

    result = fetch_compute_instances(MockRequest())
    print(json.dumps(result, indent=2))