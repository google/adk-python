#!/usr/bin/env python3
"""
Cloud Function to fetch Firewall Rules and analyze security risks
Runs independently on a schedule (every 4 hours)
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any
from google.cloud import compute_v1
from google.cloud import bigquery
import ipaddress


def analyze_firewall_risk(rule) -> Dict[str, Any]:
    """Analyze the security risk of a firewall rule"""
    risk_score = 0
    risk_factors = []

    # Check for overly permissive source ranges
    if rule.source_ranges:
        for source in rule.source_ranges:
            if source == '0.0.0.0/0':
                risk_score += 10
                risk_factors.append('Open to entire internet')
            elif source == '0.0.0.0/8':
                risk_score += 8
                risk_factors.append('Very broad source range')
            else:
                try:
                    network = ipaddress.ip_network(source)
                    if network.num_addresses > 65536:  # /16 or larger
                        risk_score += 5
                        risk_factors.append(f'Large source range: {source}')
                except:
                    pass

    # Check for risky protocols and ports
    if rule.allowed:
        for allowed_rule in rule.allowed:
            protocol = allowed_rule.I_p_protocol.lower() if allowed_rule.I_p_protocol else 'all'

            # Check for allow all protocols
            if protocol == 'all':
                risk_score += 8
                risk_factors.append('All protocols allowed')

            # Check for risky ports
            if allowed_rule.ports:
                for port_range in allowed_rule.ports:
                    if '-' in port_range:
                        start, end = port_range.split('-')
                        if int(end) - int(start) > 1000:
                            risk_score += 3
                            risk_factors.append(f'Large port range: {port_range}')

                    # Check for commonly exploited ports
                    risky_ports = {
                        '22': 'SSH',
                        '23': 'Telnet',
                        '3389': 'RDP',
                        '445': 'SMB',
                        '1433': 'MSSQL',
                        '3306': 'MySQL',
                        '5432': 'PostgreSQL',
                        '27017': 'MongoDB'
                    }

                    for risky_port, service in risky_ports.items():
                        if risky_port in port_range:
                            if '0.0.0.0/0' in (rule.source_ranges or []):
                                risk_score += 5
                                risk_factors.append(f'{service} open to internet')
                            else:
                                risk_score += 2
                                risk_factors.append(f'{service} port exposed')
            else:
                # No specific ports means all ports
                risk_score += 5
                risk_factors.append('All ports allowed')

    # Determine risk level based on score
    if risk_score >= 15:
        risk_level = 'CRITICAL'
    elif risk_score >= 10:
        risk_level = 'HIGH'
    elif risk_score >= 5:
        risk_level = 'MEDIUM'
    else:
        risk_level = 'LOW'

    return {
        'risk_level': risk_level,
        'risk_score': risk_score,
        'risk_factors': risk_factors
    }


def fetch_firewall_rules(request):
    """
    Cloud Function entry point - fetches all firewall rules
    and refreshes BigQuery table with security analysis

    Args:
        request: HTTP request object

    Returns:
        JSON response with status and record count
    """

    # Initialize clients
    firewall_client = compute_v1.FirewallsClient()
    bq_client = bigquery.Client()

    # Get configuration from environment
    project_id = os.environ.get('PROJECT_ID', 'mgm-digitalconcierge')
    dataset_id = os.environ.get('BQ_DATASET_ID', 'security_insights')

    print(f"Starting firewall rules refresh for project: {project_id}")

    try:
        # Fetch all firewall rules
        firewall_rules_data = []
        rules = firewall_client.list(project=project_id)

        for rule in rules:
            # Analyze security risk
            risk_analysis = analyze_firewall_risk(rule)

            # Extract rule details
            rule_record = {
                'rule_id': str(rule.id) if rule.id else rule.name,
                'name': rule.name,
                'description': rule.description or '',
                'direction': rule.direction,
                'priority': rule.priority,
                'source_ranges': rule.source_ranges if rule.source_ranges else [],
                'destination_ranges': rule.destination_ranges if rule.destination_ranges else [],
                'source_tags': rule.source_tags if rule.source_tags else [],
                'target_tags': rule.target_tags if rule.target_tags else [],
                'source_service_accounts': rule.source_service_accounts if rule.source_service_accounts else [],
                'target_service_accounts': rule.target_service_accounts if rule.target_service_accounts else [],
                'allowed': json.dumps([{
                    'protocol': a.I_p_protocol,
                    'ports': a.ports if a.ports else []
                } for a in (rule.allowed or [])]),
                'denied': json.dumps([{
                    'protocol': d.I_p_protocol,
                    'ports': d.ports if d.ports else []
                } for d in (rule.denied or [])]),
                'network': rule.network.split('/')[-1] if rule.network else 'default',
                'disabled': rule.disabled if hasattr(rule, 'disabled') else False,
                'log_config': json.dumps({
                    'enable': rule.log_config.enable if rule.log_config else False,
                    'metadata': rule.log_config.metadata if rule.log_config else None
                }),
                'created_at': rule.creation_timestamp,
                'risk_level': risk_analysis['risk_level'],
                'risk_score': risk_analysis['risk_score'],
                'risk_factors': json.dumps(risk_analysis['risk_factors']),
                'is_ingress': rule.direction == 'INGRESS',
                'is_egress': rule.direction == 'EGRESS',
                'allows_all_traffic': '0.0.0.0/0' in (rule.source_ranges or []),
                'last_refreshed': datetime.utcnow().isoformat(),
                'refresh_job': 'scheduled_4h'
            }

            firewall_rules_data.append(rule_record)

        # Load data to BigQuery
        if firewall_rules_data:
            table_id = f"{project_id}.{dataset_id}.firewall_rules"

            # Define schema
            schema = [
                bigquery.SchemaField("rule_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("name", "STRING"),
                bigquery.SchemaField("description", "STRING"),
                bigquery.SchemaField("direction", "STRING"),
                bigquery.SchemaField("priority", "INTEGER"),
                bigquery.SchemaField("source_ranges", "STRING", mode="REPEATED"),
                bigquery.SchemaField("destination_ranges", "STRING", mode="REPEATED"),
                bigquery.SchemaField("source_tags", "STRING", mode="REPEATED"),
                bigquery.SchemaField("target_tags", "STRING", mode="REPEATED"),
                bigquery.SchemaField("source_service_accounts", "STRING", mode="REPEATED"),
                bigquery.SchemaField("target_service_accounts", "STRING", mode="REPEATED"),
                bigquery.SchemaField("allowed", "JSON"),
                bigquery.SchemaField("denied", "JSON"),
                bigquery.SchemaField("network", "STRING"),
                bigquery.SchemaField("disabled", "BOOLEAN"),
                bigquery.SchemaField("log_config", "JSON"),
                bigquery.SchemaField("created_at", "TIMESTAMP"),
                bigquery.SchemaField("risk_level", "STRING"),
                bigquery.SchemaField("risk_score", "INTEGER"),
                bigquery.SchemaField("risk_factors", "JSON"),
                bigquery.SchemaField("is_ingress", "BOOLEAN"),
                bigquery.SchemaField("is_egress", "BOOLEAN"),
                bigquery.SchemaField("allows_all_traffic", "BOOLEAN"),
                bigquery.SchemaField("last_refreshed", "TIMESTAMP"),
                bigquery.SchemaField("refresh_job", "STRING"),
            ]

            # Configure load job
            job_config = bigquery.LoadJobConfig(
                schema=schema,
                write_disposition="WRITE_TRUNCATE",
                create_disposition="CREATE_IF_NEEDED",
                schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION]
            )

            # Load data
            job = bq_client.load_table_from_json(
                firewall_rules_data,
                table_id,
                job_config=job_config
            )
            job.result()

            print(f"Successfully loaded {len(firewall_rules_data)} firewall rules to BigQuery")

            # Create security alerts view
            create_firewall_alerts_view(bq_client, project_id, dataset_id)

            # Calculate statistics
            critical_rules = sum(1 for r in firewall_rules_data if r['risk_level'] == 'CRITICAL')
            high_risk_rules = sum(1 for r in firewall_rules_data if r['risk_level'] == 'HIGH')
            open_to_internet = sum(1 for r in firewall_rules_data if r['allows_all_traffic'])

            # Log refresh metadata
            metadata_table_id = f"{project_id}.{dataset_id}.refresh_metadata"
            metadata_record = [{
                'table_name': 'firewall_rules',
                'refresh_time': datetime.utcnow().isoformat(),
                'record_count': len(firewall_rules_data),
                'status': 'success',
                'refresh_type': 'scheduled',
                'details': json.dumps({
                    'total_rules': len(firewall_rules_data),
                    'critical_rules': critical_rules,
                    'high_risk_rules': high_risk_rules,
                    'open_to_internet': open_to_internet,
                    'disabled_rules': sum(1 for r in firewall_rules_data if r['disabled'])
                }),
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
                'records': len(firewall_rules_data),
                'table': table_id,
                'security_summary': {
                    'total_rules': len(firewall_rules_data),
                    'critical_rules': critical_rules,
                    'high_risk_rules': high_risk_rules,
                    'open_to_internet': open_to_internet
                },
                'timestamp': datetime.utcnow().isoformat()
            }
        else:
            return {
                'status': 'success',
                'records': 0,
                'message': 'No firewall rules found',
                'timestamp': datetime.utcnow().isoformat()
            }

    except Exception as e:
        error_msg = f"Error in fetch_firewall_rules: {str(e)}"
        print(error_msg)

        # Log error
        try:
            metadata_table_id = f"{project_id}.{dataset_id}.refresh_metadata"
            error_record = [{
                'table_name': 'firewall_rules',
                'refresh_time': datetime.utcnow().isoformat(),
                'record_count': 0,
                'status': 'failed',
                'refresh_type': 'scheduled',
                'error_message': str(e)[:1000]
            }]

            bq_client.load_table_from_json(
                error_record,
                metadata_table_id,
                job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
            ).result()
        except:
            pass

        return {
            'status': 'error',
            'error': error_msg,
            'timestamp': datetime.utcnow().isoformat()
        }, 500


def create_firewall_alerts_view(bq_client, project_id, dataset_id):
    """Create a view for critical firewall security alerts"""
    view_id = f"{project_id}.{dataset_id}.firewall_security_alerts"

    view_query = f"""
    CREATE OR REPLACE VIEW `{view_id}` AS
    SELECT
        name,
        risk_level,
        risk_score,
        ARRAY_TO_STRING(source_ranges, ', ') as source_ranges,
        JSON_EXTRACT_SCALAR(allowed, '$[0].protocol') as protocol,
        JSON_EXTRACT_SCALAR(allowed, '$[0].ports[0]') as ports,
        risk_factors,
        CASE
            WHEN risk_level = 'CRITICAL' THEN 'IMMEDIATE ACTION REQUIRED'
            WHEN risk_level = 'HIGH' THEN 'Review and remediate soon'
            WHEN risk_level = 'MEDIUM' THEN 'Schedule for review'
            ELSE 'Monitor'
        END as recommended_action,
        last_refreshed
    FROM `{project_id}.{dataset_id}.firewall_rules`
    WHERE risk_level IN ('CRITICAL', 'HIGH')
        AND NOT disabled
    ORDER BY risk_score DESC
    """

    try:
        bq_client.query(view_query).result()
        print(f"Created/updated firewall alerts view: {view_id}")
    except Exception as e:
        print(f"Warning: Could not create alerts view: {e}")


# For local testing
if __name__ == "__main__":
    class MockRequest:
        def __init__(self):
            self.json = {}

    result = fetch_firewall_rules(MockRequest())
    print(json.dumps(result, indent=2))