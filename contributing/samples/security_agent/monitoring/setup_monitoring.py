#!/usr/bin/env python3
"""
Setup Cloud Monitoring for the Security Agent System
Creates dashboards, alerts, and uptime checks
"""

import json
import os
from google.cloud import monitoring_v3
from google.cloud import monitoring_dashboard_v1
from google.api_core import exceptions
import time

# Project configuration
PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT', 'mgm-digitalconcierge')
NOTIFICATION_CHANNEL = os.environ.get('ALERT_EMAIL', 'stuart.gano@example.com')

class MonitoringSetup:
    def __init__(self, project_id=PROJECT_ID):
        self.project_id = project_id
        self.project_name = f"projects/{project_id}"

        # Initialize clients
        self.metrics_client = monitoring_v3.MetricServiceClient()
        self.alert_client = monitoring_v3.AlertPolicyServiceClient()
        self.dashboard_client = monitoring_dashboard_v1.DashboardsServiceClient()
        self.uptime_client = monitoring_v3.UptimeCheckServiceClient()
        self.notification_client = monitoring_v3.NotificationChannelServiceClient()

    def create_custom_metrics(self):
        """Create custom metrics for monitoring"""
        from google.cloud.monitoring_v3 import types

        custom_metrics = [
            {
                "type": "custom.googleapis.com/security/findings_count",
                "display_name": "Security Findings Count",
                "description": "Number of security findings in the system",
                "metric_kind": types.MetricDescriptor.MetricKind.GAUGE,
                "value_type": types.MetricDescriptor.ValueType.INT64
            },
            {
                "type": "custom.googleapis.com/service_discovery/api_calls",
                "display_name": "Service Discovery API Calls",
                "description": "Number of service discovery API calls",
                "metric_kind": types.MetricDescriptor.MetricKind.CUMULATIVE,
                "value_type": types.MetricDescriptor.ValueType.INT64
            },
            {
                "type": "custom.googleapis.com/agent/health_status",
                "display_name": "ADK Agent Health Status",
                "description": "Health status of the ADK agent (0=down, 1=up)",
                "metric_kind": types.MetricDescriptor.MetricKind.GAUGE,
                "value_type": types.MetricDescriptor.ValueType.DOUBLE
            },
            {
                "type": "custom.googleapis.com/confluence/sync_status",
                "display_name": "Confluence Sync Status",
                "description": "Status of Confluence synchronization",
                "metric_kind": types.MetricDescriptor.MetricKind.GAUGE,
                "value_type": types.MetricDescriptor.ValueType.INT64
            },
            {
                "type": "custom.googleapis.com/url_learning/success_rate",
                "display_name": "URL Learning Success Rate",
                "description": "Success rate of URL documentation learning",
                "metric_kind": types.MetricDescriptor.MetricKind.GAUGE,
                "value_type": types.MetricDescriptor.ValueType.DOUBLE
            }
        ]

        for metric_def in custom_metrics:
            descriptor = types.MetricDescriptor()
            descriptor.type = metric_def["type"]
            descriptor.display_name = metric_def["display_name"]
            descriptor.description = metric_def["description"]
            descriptor.metric_kind = metric_def["metric_kind"]
            descriptor.value_type = metric_def["value_type"]

            try:
                descriptor = self.metrics_client.create_metric_descriptor(
                    name=self.project_name,
                    metric_descriptor=descriptor
                )
                print(f"✅ Created metric: {descriptor.type}")
            except exceptions.AlreadyExists:
                print(f"ℹ️ Metric already exists: {metric_def['type']}")
            except Exception as e:
                print(f"❌ Error creating metric {metric_def['type']}: {e}")

    def create_dashboard(self):
        """Create the monitoring dashboard"""
        try:
            # Load dashboard configuration
            with open('monitoring/dashboard_config.json', 'r') as f:
                dashboard_config = json.load(f)

            dashboard = monitoring_dashboard_v1.Dashboard()
            dashboard.display_name = dashboard_config['displayName']
            dashboard.mosaic_layout = dashboard_config['mosaicLayout']

            # Create the dashboard
            created = self.dashboard_client.create_dashboard(
                parent=self.project_name,
                dashboard=dashboard
            )
            print(f"✅ Created dashboard: {created.name}")
            return created.name
        except Exception as e:
            print(f"❌ Error creating dashboard: {e}")
            return None

    def create_alert_policies(self):
        """Create alert policies for critical conditions"""
        policies = [
            {
                "display_name": "Cloud Function Errors",
                "conditions": [{
                    "display_name": "Error rate too high",
                    "condition_threshold": {
                        "filter": 'resource.type="cloud_function" '
                                 'metric.type="cloudfunctions.googleapis.com/function/user_errors"',
                        "aggregations": [{
                            "alignment_period": {"seconds": 300},
                            "per_series_aligner": monitoring_v3.types.Aggregation.Aligner.ALIGN_RATE,
                            "cross_series_reducer": monitoring_v3.types.Aggregation.Reducer.REDUCE_SUM,
                            "group_by_fields": ["resource.function_name"]
                        }],
                        "comparison": monitoring_v3.types.ComparisonType.COMPARISON_GT,
                        "threshold_value": 1.0,
                        "duration": {"seconds": 60}
                    }
                }],
                "documentation": {
                    "content": "Cloud Function error rate is above threshold. Check logs for details."
                }
            },
            {
                "display_name": "BigQuery Query Failures",
                "conditions": [{
                    "display_name": "Query failures detected",
                    "condition_threshold": {
                        "filter": 'resource.type="bigquery_project" '
                                 'metric.type="bigquery.googleapis.com/job/num_failed_jobs"',
                        "aggregations": [{
                            "alignment_period": {"seconds": 300},
                            "per_series_aligner": monitoring_v3.types.Aggregation.Aligner.ALIGN_RATE
                        }],
                        "comparison": monitoring_v3.types.ComparisonType.COMPARISON_GT,
                        "threshold_value": 0.1,
                        "duration": {"seconds": 60}
                    }
                }]
            },
            {
                "display_name": "ADK Agent Down",
                "conditions": [{
                    "display_name": "Agent not responding",
                    "condition_threshold": {
                        "filter": 'metric.type="custom.googleapis.com/agent/health_status"',
                        "aggregations": [{
                            "alignment_period": {"seconds": 60},
                            "per_series_aligner": monitoring_v3.types.Aggregation.Aligner.ALIGN_MAX
                        }],
                        "comparison": monitoring_v3.types.ComparisonType.COMPARISON_LT,
                        "threshold_value": 0.5,
                        "duration": {"seconds": 180}
                    }
                }],
                "documentation": {
                    "content": "ADK Agent is not responding. Check if the service is running."
                }
            },
            {
                "display_name": "High Security Findings",
                "conditions": [{
                    "display_name": "Critical security findings detected",
                    "condition_threshold": {
                        "filter": 'metric.type="custom.googleapis.com/security/findings_count"',
                        "aggregations": [{
                            "alignment_period": {"seconds": 300},
                            "per_series_aligner": monitoring_v3.types.Aggregation.Aligner.ALIGN_MAX
                        }],
                        "comparison": monitoring_v3.types.ComparisonType.COMPARISON_GT,
                        "threshold_value": 100,
                        "duration": {"seconds": 60}
                    }
                }]
            }
        ]

        # Get or create notification channel
        notification_channel = self.get_or_create_notification_channel()

        for policy_config in policies:
            policy = monitoring_v3.types.AlertPolicy()
            policy.display_name = policy_config["display_name"]

            for condition_config in policy_config["conditions"]:
                condition = monitoring_v3.types.AlertPolicy.Condition()
                condition.display_name = condition_config["display_name"]
                condition.condition_threshold = condition_config["condition_threshold"]
                policy.conditions.append(condition)

            if "documentation" in policy_config:
                policy.documentation = policy_config["documentation"]

            if notification_channel:
                policy.notification_channels = [notification_channel]

            policy.combiner = monitoring_v3.types.AlertPolicy.ConditionCombinerType.AND

            try:
                created_policy = self.alert_client.create_alert_policy(
                    name=self.project_name,
                    alert_policy=policy
                )
                print(f"✅ Created alert policy: {created_policy.display_name}")
            except Exception as e:
                print(f"❌ Error creating alert policy {policy.display_name}: {e}")

    def get_or_create_notification_channel(self):
        """Get or create an email notification channel"""
        try:
            # List existing channels
            channels = self.notification_client.list_notification_channels(
                name=self.project_name
            )

            for channel in channels:
                if channel.type_ == "email" and channel.labels.get("email_address") == NOTIFICATION_CHANNEL:
                    print(f"ℹ️ Using existing notification channel: {channel.display_name}")
                    return channel.name

            # Create new channel
            channel = monitoring_v3.types.NotificationChannel()
            channel.type_ = "email"
            channel.display_name = "Security Agent Alerts"
            channel.labels = {"email_address": NOTIFICATION_CHANNEL}
            channel.enabled = True

            created = self.notification_client.create_notification_channel(
                name=self.project_name,
                notification_channel=channel
            )
            print(f"✅ Created notification channel: {created.display_name}")
            return created.name

        except Exception as e:
            print(f"❌ Error with notification channel: {e}")
            return None

    def create_uptime_checks(self):
        """Create uptime checks for critical endpoints"""
        uptime_checks = [
            {
                "display_name": "Flask API Health Check",
                "monitored_resource": {
                    "type": "uptime_url",
                    "labels": {
                        "project_id": self.project_id,
                        "host": "localhost:5000"
                    }
                },
                "http_check": {
                    "path": "/health",
                    "port": 5000,
                    "request_method": monitoring_v3.types.UptimeCheckConfig.HttpCheck.RequestMethod.GET
                },
                "period": {"seconds": 60},
                "timeout": {"seconds": 10}
            },
            {
                "display_name": "ADK Agent Health Check",
                "monitored_resource": {
                    "type": "uptime_url",
                    "labels": {
                        "project_id": self.project_id,
                        "host": "localhost:8000"
                    }
                },
                "http_check": {
                    "path": "/health",
                    "port": 8000,
                    "request_method": monitoring_v3.types.UptimeCheckConfig.HttpCheck.RequestMethod.GET
                },
                "period": {"seconds": 60},
                "timeout": {"seconds": 10}
            }
        ]

        for check_config in uptime_checks:
            check = monitoring_v3.types.UptimeCheckConfig()
            check.display_name = check_config["display_name"]
            check.monitored_resource = check_config["monitored_resource"]
            check.http_check = check_config["http_check"]
            check.period = check_config["period"]
            check.timeout = check_config["timeout"]

            try:
                created_check = self.uptime_client.create_uptime_check_config(
                    parent=self.project_name,
                    uptime_check_config=check
                )
                print(f"✅ Created uptime check: {created_check.display_name}")
            except Exception as e:
                print(f"❌ Error creating uptime check {check.display_name}: {e}")

    def write_metrics_sample_data(self):
        """Write sample data to custom metrics for testing"""
        from google.cloud import monitoring_v3
        import random

        client = monitoring_v3.MetricServiceClient()

        # Create time series data
        series = monitoring_v3.types.TimeSeries()
        series.metric.type = "custom.googleapis.com/security/findings_count"
        series.resource.type = "global"
        series.resource.labels["project_id"] = self.project_id

        now = time.time()
        seconds = int(now)
        nanos = int((now - seconds) * 10**9)
        interval = monitoring_v3.types.TimeInterval(
            {"end_time": {"seconds": seconds, "nanos": nanos}}
        )
        point = monitoring_v3.types.Point(
            {"interval": interval, "value": {"int64_value": random.randint(10, 100)}}
        )
        series.points = [point]

        try:
            client.create_time_series(name=self.project_name, time_series=[series])
            print("✅ Wrote sample metric data")
        except Exception as e:
            print(f"❌ Error writing metrics: {e}")


def main():
    print("\n🔧 Setting up Cloud Monitoring for Security Agent System")
    print("=" * 60)

    setup = MonitoringSetup()

    print("\n📊 Creating Custom Metrics...")
    setup.create_custom_metrics()

    print("\n📈 Creating Dashboard...")
    dashboard_name = setup.create_dashboard()

    print("\n🚨 Creating Alert Policies...")
    setup.create_alert_policies()

    print("\n🔍 Creating Uptime Checks...")
    setup.create_uptime_checks()

    print("\n📝 Writing Sample Metrics...")
    setup.write_metrics_sample_data()

    print("\n" + "=" * 60)
    print("✅ Monitoring Setup Complete!")
    print(f"\nView your dashboard at:")
    print(f"https://console.cloud.google.com/monitoring/dashboards")

    print(f"\nView alerts at:")
    print(f"https://console.cloud.google.com/monitoring/alerting/policies")

    print(f"\nView uptime checks at:")
    print(f"https://console.cloud.google.com/monitoring/uptime")


if __name__ == "__main__":
    main()