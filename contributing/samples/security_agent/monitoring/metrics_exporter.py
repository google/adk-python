#!/usr/bin/env python3
"""
Export metrics from the Security Agent System to Cloud Monitoring
This runs periodically to send system metrics
"""

import os
import time
import json
import sqlite3
from datetime import datetime, timedelta
from google.cloud import monitoring_v3
from google.cloud import bigquery
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT', 'mgm-digitalconcierge')

class MetricsExporter:
    def __init__(self):
        self.project_id = PROJECT_ID
        self.project_name = f"projects/{self.project_id}"
        self.metrics_client = monitoring_v3.MetricServiceClient()
        self.bq_client = bigquery.Client(project=self.project_id)

    def export_security_findings_count(self):
        """Export count of security findings to Cloud Monitoring"""
        try:
            # Query BigQuery for security findings count
            query = """
                SELECT COUNT(*) as findings_count
                FROM `security_insights.security_findings`
                WHERE created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
            """

            result = self.bq_client.query(query).result()
            count = list(result)[0].findings_count if result else 0

            # Write to Cloud Monitoring
            self._write_metric(
                metric_type="custom.googleapis.com/security/findings_count",
                value=count,
                value_type="int64"
            )
            logger.info(f"Exported security findings count: {count}")

        except Exception as e:
            logger.error(f"Error exporting security findings: {e}")

    def export_service_discovery_metrics(self):
        """Export service discovery API call metrics"""
        try:
            # Check if Flask API is running
            response = requests.get("http://localhost:5000/api/services/categories", timeout=5)
            if response.status_code == 200:
                api_status = 1.0
            else:
                api_status = 0.0

            self._write_metric(
                metric_type="custom.googleapis.com/service_discovery/api_calls",
                value=api_status,
                value_type="double"
            )
            logger.info(f"Service Discovery API status: {api_status}")

        except Exception as e:
            logger.warning(f"Service Discovery API check failed: {e}")
            self._write_metric(
                metric_type="custom.googleapis.com/service_discovery/api_calls",
                value=0.0,
                value_type="double"
            )

    def export_agent_health(self):
        """Export ADK agent health status"""
        try:
            # Check ADK agent health
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                health_status = 1.0
                logger.info("ADK Agent is healthy")
            else:
                health_status = 0.0
                logger.warning(f"ADK Agent returned status {response.status_code}")

        except requests.exceptions.ConnectionError:
            health_status = 0.0
            logger.warning("ADK Agent is not responding")
        except Exception as e:
            health_status = 0.0
            logger.error(f"Error checking agent health: {e}")

        self._write_metric(
            metric_type="custom.googleapis.com/agent/health_status",
            value=health_status,
            value_type="double"
        )

    def export_confluence_sync_status(self):
        """Export Confluence synchronization status"""
        try:
            # Check Confluence cache database
            cache_db = "cache/confluence_cache.db"
            if os.path.exists(cache_db):
                conn = sqlite3.connect(cache_db)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM documents")
                doc_count = cursor.fetchone()[0]
                conn.close()

                sync_status = 1 if doc_count > 0 else 0
                logger.info(f"Confluence cache has {doc_count} documents")
            else:
                sync_status = 0
                logger.warning("Confluence cache database not found")

        except Exception as e:
            sync_status = 0
            logger.error(f"Error checking Confluence sync: {e}")

        self._write_metric(
            metric_type="custom.googleapis.com/confluence/sync_status",
            value=sync_status,
            value_type="int64"
        )

    def export_url_learning_metrics(self):
        """Export URL learning success rate"""
        try:
            # Check learned services cache
            cache_db = "cache/service_docs/parsed_services.db"
            if os.path.exists(cache_db):
                conn = sqlite3.connect(cache_db)
                cursor = conn.cursor()

                # Count successful and total parses
                cursor.execute("""
                    SELECT
                        COUNT(*) as total,
                        COUNT(CASE WHEN service_name IS NOT NULL THEN 1 END) as successful
                    FROM parsed_services
                    WHERE parse_date >= datetime('now', '-1 day')
                """)

                result = cursor.fetchone()
                total, successful = result if result else (0, 0)
                conn.close()

                success_rate = (successful / total) if total > 0 else 1.0
                logger.info(f"URL learning success rate: {success_rate:.2%} ({successful}/{total})")
            else:
                success_rate = 0.0
                logger.warning("URL learning cache not found")

        except Exception as e:
            success_rate = 0.0
            logger.error(f"Error checking URL learning metrics: {e}")

        self._write_metric(
            metric_type="custom.googleapis.com/url_learning/success_rate",
            value=success_rate,
            value_type="double"
        )

    def export_cloud_function_metrics(self):
        """Export Cloud Function execution metrics"""
        try:
            # Query BigQuery for function execution logs
            query = """
                SELECT
                    function_name,
                    COUNT(*) as execution_count,
                    COUNTIF(status = 'ERROR') as error_count,
                    AVG(execution_time_ms) as avg_execution_time
                FROM `system_logs.cloud_function_logs`
                WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
                GROUP BY function_name
            """

            results = self.bq_client.query(query).result()

            for row in results:
                # Write execution count metric
                self._write_metric(
                    metric_type=f"custom.googleapis.com/function/{row.function_name}/executions",
                    value=row.execution_count,
                    value_type="int64"
                )

                # Write error count metric
                self._write_metric(
                    metric_type=f"custom.googleapis.com/function/{row.function_name}/errors",
                    value=row.error_count,
                    value_type="int64"
                )

                logger.info(f"Function {row.function_name}: {row.execution_count} executions, {row.error_count} errors")

        except Exception as e:
            logger.error(f"Error exporting Cloud Function metrics: {e}")

    def _write_metric(self, metric_type: str, value: any, value_type: str = "double"):
        """Write a single metric to Cloud Monitoring"""
        try:
            series = monitoring_v3.TimeSeries()
            series.metric.type = metric_type
            series.resource.type = "global"
            series.resource.labels["project_id"] = self.project_id

            now = time.time()
            seconds = int(now)
            nanos = int((now - seconds) * 10**9)
            interval = monitoring_v3.TimeInterval(
                {"end_time": {"seconds": seconds, "nanos": nanos}}
            )

            point = monitoring_v3.Point()
            point.interval = interval

            if value_type == "int64":
                point.value.int64_value = int(value)
            elif value_type == "double":
                point.value.double_value = float(value)
            elif value_type == "bool":
                point.value.bool_value = bool(value)
            else:
                point.value.string_value = str(value)

            series.points = [point]

            self.metrics_client.create_time_series(
                name=self.project_name,
                time_series=[series]
            )

        except Exception as e:
            logger.error(f"Error writing metric {metric_type}: {e}")

    def export_all_metrics(self):
        """Export all metrics"""
        print("\n📊 Exporting System Metrics to Cloud Monitoring")
        print("=" * 60)

        print("Exporting security findings count...")
        self.export_security_findings_count()

        print("Exporting service discovery metrics...")
        self.export_service_discovery_metrics()

        print("Exporting agent health status...")
        self.export_agent_health()

        print("Exporting Confluence sync status...")
        self.export_confluence_sync_status()

        print("Exporting URL learning metrics...")
        self.export_url_learning_metrics()

        print("Exporting Cloud Function metrics...")
        self.export_cloud_function_metrics()

        print("\n✅ Metrics export complete")


def main():
    """Main function to export metrics periodically"""
    exporter = MetricsExporter()

    # Run once if called directly
    exporter.export_all_metrics()

    # Optionally run in a loop
    if os.environ.get('RUN_CONTINUOUS', 'false').lower() == 'true':
        print("\n🔄 Running in continuous mode (exporting every 60 seconds)")
        while True:
            time.sleep(60)
            exporter.export_all_metrics()


if __name__ == "__main__":
    main()