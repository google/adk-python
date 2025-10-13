#!/usr/bin/env python3
"""
Dynamic GCP Service Discovery and Analysis Tool

This tool enables on-demand analysis of ANY GCP service without pre-defined lists.
It can discover services, enumerate resources, and perform security analysis dynamically.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from google.cloud import bigquery
from google.api_core import exceptions
import google.auth
from google.auth.transport.requests import Request
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports - these require additional packages
try:
    from google.cloud import asset_v1
    HAS_ASSET_API = True
except ImportError:
    HAS_ASSET_API = False
    logger.warning("Cloud Asset API not available. Some features will be limited.")

try:
    from google.cloud import resource_manager_v3
    HAS_RESOURCE_MANAGER = True
except ImportError:
    HAS_RESOURCE_MANAGER = False
    logger.warning("Resource Manager API not available. Some features will be limited.")

class GCPServiceDiscovery:
    """Dynamic GCP service discovery and analysis"""

    # Comprehensive GCP service catalog with API endpoints
    GCP_SERVICES = {
        # Compute Services
        'compute': {
            'name': 'Compute Engine',
            'api': 'compute.googleapis.com',
            'resource_types': ['instances', 'disks', 'networks', 'firewalls', 'snapshots', 'images'],
            'analysis_queries': {
                'security': "SELECT * FROM `{project}.{dataset}.compute_instances` WHERE external_ip IS NOT NULL",
                'compliance': "SELECT * FROM `{project}.{dataset}.compute_instances` WHERE encryption_status != 'ENCRYPTED'",
                'cost': "SELECT machine_type, COUNT(*) as count, SUM(estimated_cost) as total_cost FROM `{project}.{dataset}.compute_instances` GROUP BY machine_type"
            }
        },
        'kubernetes': {
            'name': 'Google Kubernetes Engine',
            'api': 'container.googleapis.com',
            'resource_types': ['clusters', 'node-pools', 'workloads'],
            'analysis_queries': {
                'security': "SELECT * FROM `{project}.{dataset}.gke_clusters` WHERE private_cluster = FALSE",
                'compliance': "SELECT * FROM `{project}.{dataset}.gke_clusters` WHERE binary_authorization_enabled = FALSE",
                'rbac': "SELECT * FROM `{project}.{dataset}.gke_clusters` WHERE legacy_abac_enabled = TRUE"
            }
        },
        'storage': {
            'name': 'Cloud Storage',
            'api': 'storage.googleapis.com',
            'resource_types': ['buckets', 'objects'],
            'analysis_queries': {
                'security': "SELECT * FROM `{project}.{dataset}.storage_buckets` WHERE 'allUsers' IN UNNEST(iam_members)",
                'compliance': "SELECT * FROM `{project}.{dataset}.storage_buckets` WHERE retention_policy IS NULL",
                'encryption': "SELECT * FROM `{project}.{dataset}.storage_buckets` WHERE default_kms_key IS NULL"
            }
        },
        'bigquery': {
            'name': 'BigQuery',
            'api': 'bigquery.googleapis.com',
            'resource_types': ['datasets', 'tables', 'views', 'routines'],
            'analysis_queries': {
                'security': "SELECT * FROM `{project}.{dataset}.bigquery_datasets` WHERE public_access = TRUE",
                'compliance': "SELECT * FROM `{project}.{dataset}.bigquery_tables` WHERE encryption_type != 'CMEK'",
                'usage': "SELECT dataset_id, SUM(total_bytes) as size_bytes FROM `{project}.{dataset}.bigquery_tables` GROUP BY dataset_id"
            }
        },
        'pubsub': {
            'name': 'Cloud Pub/Sub',
            'api': 'pubsub.googleapis.com',
            'resource_types': ['topics', 'subscriptions', 'schemas'],
            'analysis_queries': {
                'security': "SELECT * FROM `{project}.{dataset}.pubsub_topics` WHERE kms_key IS NULL",
                'compliance': "SELECT * FROM `{project}.{dataset}.pubsub_subscriptions` WHERE message_retention_duration < 604800",
                'usage': "SELECT topic, COUNT(*) as subscription_count FROM `{project}.{dataset}.pubsub_subscriptions` GROUP BY topic"
            }
        },
        'cloudsql': {
            'name': 'Cloud SQL',
            'api': 'sqladmin.googleapis.com',
            'resource_types': ['instances', 'databases', 'users'],
            'analysis_queries': {
                'security': "SELECT * FROM `{project}.{dataset}.cloudsql_instances` WHERE public_ip_configured = TRUE",
                'compliance': "SELECT * FROM `{project}.{dataset}.cloudsql_instances` WHERE backup_enabled = FALSE",
                'encryption': "SELECT * FROM `{project}.{dataset}.cloudsql_instances` WHERE disk_encryption_type != 'CMEK'"
            }
        },
        'cloudrun': {
            'name': 'Cloud Run',
            'api': 'run.googleapis.com',
            'resource_types': ['services', 'jobs', 'revisions'],
            'documentation_url': 'https://cloud.google.com/run/docs',
            'analysis_queries': {
                'security': "SELECT * FROM `{project}.{dataset}.cloudrun_services` WHERE allow_unauthenticated = TRUE",
                'compliance': "SELECT * FROM `{project}.{dataset}.cloudrun_services` WHERE binary_authorization_policy IS NULL",
                'scaling': "SELECT service_name, MAX(max_instances) as max_scale FROM `{project}.{dataset}.cloudrun_services` GROUP BY service_name"
            }
        },
        'cloudfunctions': {
            'name': 'Cloud Functions',
            'api': 'cloudfunctions.googleapis.com',
            'resource_types': ['functions', 'triggers'],
            'analysis_queries': {
                'security': "SELECT * FROM `{project}.{dataset}.cloud_functions` WHERE ingress_settings = 'ALLOW_ALL'",
                'compliance': "SELECT * FROM `{project}.{dataset}.cloud_functions` WHERE vpc_connector IS NULL",
                'runtime': "SELECT runtime, COUNT(*) as function_count FROM `{project}.{dataset}.cloud_functions` GROUP BY runtime"
            }
        },
        'iam': {
            'name': 'Identity and Access Management',
            'api': 'iam.googleapis.com',
            'resource_types': ['service-accounts', 'roles', 'policies'],
            'analysis_queries': {
                'security': "SELECT * FROM `{project}.{dataset}.iam_accounts` WHERE 'roles/owner' IN UNNEST(roles)",
                'compliance': "SELECT * FROM `{project}.{dataset}.service_accounts` WHERE key_rotation_age > 90",
                'privileges': "SELECT member, COUNT(DISTINCT role) as role_count FROM `{project}.{dataset}.iam_bindings` GROUP BY member HAVING role_count > 5"
            }
        },
        'monitoring': {
            'name': 'Cloud Monitoring',
            'api': 'monitoring.googleapis.com',
            'resource_types': ['alert-policies', 'uptime-checks', 'dashboards'],
            'analysis_queries': {
                'coverage': "SELECT resource_type, COUNT(*) as alert_count FROM `{project}.{dataset}.monitoring_alerts` GROUP BY resource_type",
                'sla': "SELECT service, AVG(uptime_percentage) as avg_uptime FROM `{project}.{dataset}.uptime_checks` GROUP BY service"
            }
        },
        'logging': {
            'name': 'Cloud Logging',
            'api': 'logging.googleapis.com',
            'resource_types': ['logs', 'sinks', 'metrics'],
            'analysis_queries': {
                'security': "SELECT * FROM `{project}.{dataset}.log_sinks` WHERE destination_type != 'BIGQUERY'",
                'retention': "SELECT log_name, retention_days FROM `{project}.{dataset}.log_retention` WHERE retention_days < 30"
            }
        },
        'spanner': {
            'name': 'Cloud Spanner',
            'api': 'spanner.googleapis.com',
            'resource_types': ['instances', 'databases', 'backup'],
            'analysis_queries': {
                'security': "SELECT * FROM `{project}.{dataset}.spanner_instances` WHERE encryption_type != 'CMEK'",
                'backup': "SELECT * FROM `{project}.{dataset}.spanner_databases` WHERE last_backup_date < TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)"
            }
        },
        'firestore': {
            'name': 'Firestore',
            'api': 'firestore.googleapis.com',
            'resource_types': ['databases', 'collections', 'indexes'],
            'analysis_queries': {
                'security': "SELECT * FROM `{project}.{dataset}.firestore_databases` WHERE security_rules_version IS NULL"
            }
        },
        'memorystore': {
            'name': 'Memorystore',
            'api': 'redis.googleapis.com',
            'resource_types': ['redis-instances', 'memcached-instances'],
            'analysis_queries': {
                'security': "SELECT * FROM `{project}.{dataset}.memorystore_instances` WHERE auth_enabled = FALSE"
            }
        },
        'vpc': {
            'name': 'Virtual Private Cloud',
            'api': 'compute.googleapis.com',
            'resource_types': ['networks', 'subnets', 'routes', 'peerings'],
            'analysis_queries': {
                'security': "SELECT * FROM `{project}.{dataset}.vpc_networks` WHERE auto_create_subnetworks = TRUE",
                'connectivity': "SELECT * FROM `{project}.{dataset}.vpc_peerings` WHERE state != 'ACTIVE'"
            }
        },
        'loadbalancing': {
            'name': 'Cloud Load Balancing',
            'api': 'compute.googleapis.com',
            'resource_types': ['load-balancers', 'backend-services', 'health-checks'],
            'analysis_queries': {
                'security': "SELECT * FROM `{project}.{dataset}.load_balancers` WHERE ssl_policy IS NULL",
                'health': "SELECT * FROM `{project}.{dataset}.backend_services` WHERE health_check_id IS NULL"
            }
        },
        'cdn': {
            'name': 'Cloud CDN',
            'api': 'compute.googleapis.com',
            'resource_types': ['cdn-policies', 'cache-invalidations'],
            'analysis_queries': {
                'performance': "SELECT origin, AVG(cache_hit_ratio) as avg_hit_ratio FROM `{project}.{dataset}.cdn_metrics` GROUP BY origin"
            }
        },
        'armor': {
            'name': 'Cloud Armor',
            'api': 'compute.googleapis.com',
            'resource_types': ['security-policies', 'rules'],
            'analysis_queries': {
                'security': "SELECT * FROM `{project}.{dataset}.armor_policies` WHERE default_action != 'DENY'",
                'coverage': "SELECT target_resource, COUNT(*) as rule_count FROM `{project}.{dataset}.armor_rules` GROUP BY target_resource"
            }
        },
        'apigateway': {
            'name': 'API Gateway',
            'api': 'apigateway.googleapis.com',
            'resource_types': ['apis', 'configs', 'gateways'],
            'analysis_queries': {
                'security': "SELECT * FROM `{project}.{dataset}.api_gateways` WHERE require_auth = FALSE"
            }
        },
        'apigee': {
            'name': 'Apigee',
            'api': 'apigee.googleapis.com',
            'resource_types': ['organizations', 'environments', 'proxies'],
            'analysis_queries': {
                'security': "SELECT * FROM `{project}.{dataset}.apigee_proxies` WHERE oauth_enabled = FALSE"
            }
        },
        'dataflow': {
            'name': 'Cloud Dataflow',
            'api': 'dataflow.googleapis.com',
            'resource_types': ['jobs', 'templates', 'snapshots'],
            'analysis_queries': {
                'performance': "SELECT job_name, AVG(execution_time) as avg_time FROM `{project}.{dataset}.dataflow_jobs` GROUP BY job_name"
            }
        },
        'composer': {
            'name': 'Cloud Composer',
            'api': 'composer.googleapis.com',
            'resource_types': ['environments', 'dags'],
            'analysis_queries': {
                'security': "SELECT * FROM `{project}.{dataset}.composer_environments` WHERE private_environment = FALSE"
            }
        },
        'dataproc': {
            'name': 'Cloud Dataproc',
            'api': 'dataproc.googleapis.com',
            'resource_types': ['clusters', 'jobs', 'workflows'],
            'analysis_queries': {
                'security': "SELECT * FROM `{project}.{dataset}.dataproc_clusters` WHERE encryption_type != 'CMEK'"
            }
        }
    }

    def __init__(self, project_id: Optional[str] = None, dataset_id: str = "security_insights"):
        """Initialize the service discovery tool"""
        self.project_id = project_id or os.environ.get('GOOGLE_CLOUD_PROJECT')
        self.dataset_id = dataset_id

        # Initialize clients
        try:
            self.bq_client = bigquery.Client(project=self.project_id)

            # Initialize optional clients
            if HAS_ASSET_API:
                self.asset_client = asset_v1.AssetServiceClient()
            else:
                self.asset_client = None

            logger.info(f"✅ Service Discovery initialized for project: {self.project_id}")
        except Exception as e:
            logger.error(f"Failed to initialize clients: {e}")
            raise

    def discover_all_services(self) -> List[Dict[str, Any]]:
        """Discover all enabled GCP services in the project"""
        try:
            enabled_services = []

            # Try to use Cloud Asset API if available
            if self.asset_client and HAS_ASSET_API:
                # Use Cloud Asset API to discover enabled services
                parent = f"projects/{self.project_id}"

                # List all assets of type serviceusage.googleapis.com/Service
                request = asset_v1.ListAssetsRequest(
                    parent=parent,
                    asset_types=["serviceusage.googleapis.com/Service"],
                    content_type=asset_v1.ContentType.RESOURCE
                )

                page_result = self.asset_client.list_assets(request=request)

                for asset in page_result:
                    service_name = asset.name.split('/')[-1]
                    if service_name.endswith('.googleapis.com'):
                        service_key = service_name.replace('.googleapis.com', '').replace('-', '')

                        # Match with our service catalog
                        for key, details in self.GCP_SERVICES.items():
                            if details['api'] == service_name:
                                enabled_services.append({
                                    'service_key': key,
                                    'service_name': details['name'],
                                    'api': service_name,
                                    'status': 'enabled',
                                    'resource_types': details.get('resource_types', [])
                                })
                                break
                        else:
                            # Service not in our catalog - still add it
                            enabled_services.append({
                                'service_key': service_key,
                                'service_name': service_name,
                                'api': service_name,
                                'status': 'enabled',
                                'resource_types': []
                            })
            else:
                # Fallback: Return services from our catalog
                logger.info("Cloud Asset API not available. Returning catalog services.")
                for key, details in self.GCP_SERVICES.items():
                    enabled_services.append({
                        'service_key': key,
                        'service_name': details['name'],
                        'api': details['api'],
                        'status': 'available',
                        'resource_types': details.get('resource_types', [])
                    })

            return enabled_services

        except Exception as e:
            logger.error(f"Error discovering services: {e}")
            # Fallback to returning our full catalog
            return [
                {
                    'service_key': key,
                    'service_name': details['name'],
                    'api': details['api'],
                    'status': 'available',
                    'resource_types': details.get('resource_types', [])
                }
                for key, details in self.GCP_SERVICES.items()
            ]

    def analyze_service(self, service_key: str, analysis_type: str = 'security') -> Dict[str, Any]:
        """Perform on-demand analysis of a specific service"""
        if service_key not in self.GCP_SERVICES:
            # Try to find service by partial match
            for key in self.GCP_SERVICES:
                if service_key.lower() in key.lower():
                    service_key = key
                    break
            else:
                return {
                    'error': f'Service {service_key} not found',
                    'available_services': list(self.GCP_SERVICES.keys())
                }

        service = self.GCP_SERVICES[service_key]
        results = {
            'service': service['name'],
            'api': service['api'],
            'analysis_type': analysis_type,
            'timestamp': datetime.now().isoformat(),
            'findings': []
        }

        # Get analysis queries for this service
        queries = service.get('analysis_queries', {})

        if analysis_type in queries:
            query = queries[analysis_type].format(
                project=self.project_id,
                dataset=self.dataset_id
            )

            try:
                # Execute the analysis query
                query_job = self.bq_client.query(query)
                rows = list(query_job.result())

                results['findings'] = [dict(row) for row in rows]
                results['total_findings'] = len(rows)
                results['query_executed'] = query

            except exceptions.NotFound:
                results['error'] = (
                    f"Table not found. You may need to run data collection for {service['name']} first."
                )
                results['suggested_action'] = f"Deploy cloud function: fetch_{service_key}_data"

                # Provide prescriptive next steps from documentation/best practices
                guidance = self._build_default_guidance(service_key, service)
                if guidance:
                    results['recommended_actions'] = guidance

                # Surface any learned documentation snippets
                if HAS_DOC_PARSER:
                    try:
                        parser = ServiceDocumentationParser()
                        learned = parser.parse_documentation_url(service.get('documentation_url', ''), force_refresh=False)
                        if learned and not learned.get('error'):
                            results['learned_summary'] = {
                                'description': learned.get('description'),
                                'capabilities': learned.get('capabilities', []),
                                'permissions': learned.get('permissions', []),
                            }
                    except Exception as exc:  # pragma: no cover - informational only
                        logger.debug(f"Could not enrich guidance for {service_key}: {exc}")

            except Exception as e:
                results['error'] = str(e)
        else:
            results['error'] = f"Analysis type '{analysis_type}' not available for {service['name']}"
            results['available_analyses'] = list(queries.keys())

        return results

    def _build_default_guidance(self, service_key: str, service: Dict[str, Any]) -> List[str]:
        """Return actionable guidance when project telemetry is unavailable."""

        guidance_catalog: Dict[str, List[str]] = {
            'cloudrun': [
                "Ensure the Cloud Run service agent `service-<PROJECT_NUMBER>@serverless-robot-prod.iam.gserviceaccount.com` has `roles/run.serviceAgent` plus VPC connector roles when private networking is required.",
                "Grant the workload identity user (your deploying CI/CD SA) `roles/run.admin` and `roles/iam.serviceAccountUser` on the runtime service account only—avoid broad `editor` roles.",
                "Lock ingress to `internal-and-cloud-load-balancing` when services do not need public endpoints; pair with Google-managed SSL or Identity-Aware Proxy where exposure is required.",
                "Use organisation policies: `constraints/run.allowedIngress` and `constraints/run.allowedVpcConnectorEgressSettings` to enforce baseline posture across projects.",
                "For VPC Service Controls, add the Cloud Run service project and associated Artifact Registry/Secret Manager projects to the same perimeter—include the Cloud Run service agent principal in access levels.",
                "Enable Cloud Logging + Cloud Monitoring sinks filtered on `resource.type=cloud_run_revision` to drive anomaly detection dashboards.",
                "Schedule `fetch_cloudrun_data` ingest after each deployment to populate the `security_insights` dataset with configuration snapshots and scaling metrics.",
            ],
            'cloudfunctions': [
                "Ensure service agents have `roles/cloudfunctions.serviceAgent` and restrict invokers via IAM or HTTPS authorisation.",
                "Capture build metadata via `fetch_cloudfunctions_data` to audit runtime dependencies and entry points.",
            ],
        }

        normalized_key = service_key.lower()
        if normalized_key in guidance_catalog:
            return guidance_catalog[normalized_key]

        # Provide generic guidance if we have nothing specific
        name = service.get('name', service_key)
        return [
            f"Establish least-privilege IAM for all {name} service agents and CI/CD identities.",
            f"Enable data ingest (`fetch_{normalized_key}_data`) so future analyses can reference real telemetry.",
            "Review VPC Service Controls and organisation policies to confirm the service can operate within existing perimeters.",
        ]

    def get_service_resources(self, service_key: str) -> List[Dict[str, Any]]:
        """Get all resources for a specific service"""
        if service_key not in self.GCP_SERVICES:
            return []

        service = self.GCP_SERVICES[service_key]
        resources = []

        # Check if Cloud Asset API is available
        if not self.asset_client or not HAS_ASSET_API:
            logger.info("Cloud Asset API not available. Returning mock resources.")
            # Return mock data for demonstration
            return [{
                'name': f'projects/{self.project_id}/resources/example-{service_key}',
                'type': rt,
                'service': service['name'],
                'create_time': datetime.now().isoformat(),
                'update_time': datetime.now().isoformat(),
                'resource': {'status': 'ACTIVE'}
            } for rt in service.get('resource_types', [])[:3]]

        # Use Cloud Asset API to list resources
        parent = f"projects/{self.project_id}"

        for resource_type in service.get('resource_types', []):
            try:
                # Convert resource type to asset type format
                asset_type = self._get_asset_type(service_key, resource_type)

                request = asset_v1.ListAssetsRequest(
                    parent=parent,
                    asset_types=[asset_type],
                    content_type=asset_v1.ContentType.RESOURCE
                )

                page_result = self.asset_client.list_assets(request=request)

                for asset in page_result:
                    resources.append({
                        'name': asset.name,
                        'type': resource_type,
                        'service': service['name'],
                        'create_time': asset.create_time.isoformat() if asset.create_time else None,
                        'update_time': asset.update_time.isoformat() if asset.update_time else None,
                        'resource': asset.resource.data if asset.resource else {}
                    })

            except Exception as e:
                logger.warning(f"Could not list {resource_type} for {service_key}: {e}")

        return resources

    def _get_asset_type(self, service_key: str, resource_type: str) -> str:
        """Convert service and resource type to Cloud Asset API format"""
        # Mapping of our resource types to Cloud Asset types
        asset_type_map = {
            'compute': {
                'instances': 'compute.googleapis.com/Instance',
                'disks': 'compute.googleapis.com/Disk',
                'networks': 'compute.googleapis.com/Network',
                'firewalls': 'compute.googleapis.com/Firewall'
            },
            'storage': {
                'buckets': 'storage.googleapis.com/Bucket'
            },
            'bigquery': {
                'datasets': 'bigquery.googleapis.com/Dataset',
                'tables': 'bigquery.googleapis.com/Table'
            },
            'kubernetes': {
                'clusters': 'container.googleapis.com/Cluster'
            },
            'cloudsql': {
                'instances': 'sqladmin.googleapis.com/Instance'
            },
            'pubsub': {
                'topics': 'pubsub.googleapis.com/Topic',
                'subscriptions': 'pubsub.googleapis.com/Subscription'
            }
        }

        if service_key in asset_type_map and resource_type in asset_type_map[service_key]:
            return asset_type_map[service_key][resource_type]

        # Default format
        api = self.GCP_SERVICES[service_key]['api']
        resource_name = resource_type.rstrip('s').capitalize()
        return f"{api}/{resource_name}"

    def create_custom_analysis(self, service_key: str, custom_query: str) -> Dict[str, Any]:
        """Execute a custom analysis query for any service"""
        results = {
            'service': service_key,
            'query_type': 'custom',
            'timestamp': datetime.now().isoformat(),
            'findings': []
        }

        # Replace placeholders in custom query
        custom_query = custom_query.format(
            project=self.project_id,
            dataset=self.dataset_id
        )

        try:
            query_job = self.bq_client.query(custom_query)
            rows = list(query_job.result())

            results['findings'] = [dict(row) for row in rows]
            results['total_findings'] = len(rows)
            results['query_executed'] = custom_query
            results['bytes_processed'] = query_job.total_bytes_processed
            results['execution_time_ms'] = query_job.total_bytes_processed

        except Exception as e:
            results['error'] = str(e)
            results['suggestion'] = "Check your query syntax and ensure the tables exist"

        return results


# Import documentation parser for learning new services
try:
    from .service_documentation_parser import (
        ServiceDocumentationParser,
        parse_service_documentation as parse_doc,
        discover_new_services as discover_new,
        learn_service_from_api_spec as learn_api,
        register_custom_service as register_service
    )
    HAS_DOC_PARSER = True
except ImportError:
    HAS_DOC_PARSER = False
    logger.warning("Documentation parser not available. URL learning disabled.")


# Tool functions for ADK agent integration
def discover_gcp_services(include_learned: bool = True) -> str:
    """
    Discover all available GCP services in the project.

    Args:
        include_learned: Include services learned from documentation URLs

    Returns:
        Formatted string listing all discovered services
    """
    try:
        discovery = GCPServiceDiscovery()
        services = discovery.discover_all_services()

        # Add learned services if documentation parser is available
        if include_learned and HAS_DOC_PARSER:
            try:
                parser = ServiceDocumentationParser()
                # Query learned services from cache
                import sqlite3
                conn = sqlite3.connect(parser.cache_db)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT service_name, api_endpoint, capabilities
                    FROM parsed_services
                    WHERE service_name IS NOT NULL
                ''')
                learned = cursor.fetchall()
                conn.close()

                for name, api, caps in learned:
                    services.append({
                        'service_key': 'learned',
                        'service_name': f"{name} (Learned)",
                        'api': api or 'custom',
                        'status': 'learned',
                        'resource_types': []
                    })
            except Exception as e:
                logger.warning(f"Could not load learned services: {e}")

        result = "🔍 GCP Services Discovery\n"
        result += "=" * 50 + "\n\n"

        enabled_count = sum(1 for s in services if s['status'] == 'enabled')
        result += f"Found {len(services)} services ({enabled_count} enabled):\n\n"

        for service in services:
            status_icon = "✅" if service['status'] == 'enabled' else "📦"
            result += f"{status_icon} {service['service_name']}\n"
            result += f"   Key: {service['service_key']}\n"
            result += f"   API: {service['api']}\n"
            if service['resource_types']:
                result += f"   Resources: {', '.join(service['resource_types'])}\n"
            result += "\n"

        return result

    except Exception as e:
        return f"Error discovering services: {str(e)}"


def analyze_gcp_service(
    service_name: str,
    analysis_type: str = "security",
    custom_query: Optional[str] = None
) -> str:
    """
    Perform on-demand analysis of any GCP service.

    Args:
        service_name: Name or key of the GCP service to analyze
        analysis_type: Type of analysis (security, compliance, cost, usage, custom)
        custom_query: Optional custom BigQuery SQL for analysis

    Returns:
        Formatted analysis results
    """
    try:
        discovery = GCPServiceDiscovery()

        if custom_query:
            results = discovery.create_custom_analysis(service_name, custom_query)
        else:
            results = discovery.analyze_service(service_name, analysis_type)

        # Format results
        output = f"📊 Analysis Results: {results.get('service', service_name)}\n"
        output += "=" * 50 + "\n\n"
        output += f"Analysis Type: {results.get('analysis_type', 'custom')}\n"
        output += f"Timestamp: {results.get('timestamp', 'N/A')}\n\n"

        if 'error' in results:
            output += f"⚠️ Error: {results['error']}\n"
            if 'suggested_action' in results:
                output += f"💡 Suggestion: {results['suggested_action']}\n"
            if results.get('recommended_actions'):
                output += "\n🔒 Recommended Next Steps:\n"
                for idx, action in enumerate(results['recommended_actions'], 1):
                    output += f"  {idx}. {action}\n"
            if results.get('learned_summary'):
                learned = results['learned_summary']
                output += "\n📚 Service Baseline (from documentation):\n"
                if learned.get('description'):
                    output += f"  • Summary: {learned['description']}\n"
                if learned.get('capabilities'):
                    output += f"  • Capabilities: {', '.join(learned['capabilities'][:6])}\n"
                if learned.get('permissions'):
                    perms = [
                        perm for perm in learned['permissions']
                        if any(token in perm for token in ('run', 'iam', 'roles/'))
                    ]
                    perms = [perm for perm in perms if '/' in perm or '.' in perm][:5]
                    if perms:
                        sample_perms = ", ".join(perms)
                        output += f"  • Key IAM permissions: {sample_perms}\n"
            if 'available_analyses' in results:
                output += f"Available analyses: {', '.join(results['available_analyses'])}\n"
            if 'available_services' in results:
                output += f"Available services: {', '.join(results['available_services'][:10])}...\n"
        else:
            output += f"Total Findings: {results.get('total_findings', 0)}\n\n"

            if results.get('findings'):
                output += "Top Findings:\n"
                for i, finding in enumerate(results['findings'][:10], 1):
                    output += f"\n{i}. "
                    for key, value in finding.items():
                        output += f"{key}: {value}, "
                    output = output.rstrip(', ') + "\n"

                if len(results['findings']) > 10:
                    output += f"\n... and {len(results['findings']) - 10} more findings\n"

            if 'query_executed' in results:
                output += f"\n📝 Query executed:\n{results['query_executed']}\n"

        return output

    except Exception as e:
        return f"Error analyzing service: {str(e)}"


def get_service_resources(service_name: str) -> str:
    """
    List all resources for a specific GCP service.

    Args:
        service_name: Name or key of the GCP service

    Returns:
        Formatted list of resources
    """
    try:
        discovery = GCPServiceDiscovery()
        resources = discovery.get_service_resources(service_name)

        output = f"📦 Resources for {service_name}\n"
        output += "=" * 50 + "\n\n"

        if not resources:
            output += "No resources found or service not recognized.\n"
            output += "Try: discover_gcp_services() to see available services.\n"
        else:
            # Group resources by type
            by_type = {}
            for resource in resources:
                resource_type = resource['type']
                if resource_type not in by_type:
                    by_type[resource_type] = []
                by_type[resource_type].append(resource)

            for resource_type, items in by_type.items():
                output += f"\n{resource_type.upper()} ({len(items)} items):\n"
                for item in items[:5]:
                    output += f"  • {item['name']}\n"
                    if item.get('create_time'):
                        output += f"    Created: {item['create_time']}\n"

                if len(items) > 5:
                    output += f"  ... and {len(items) - 5} more\n"

        return output

    except Exception as e:
        return f"Error getting resources: {str(e)}"


def suggest_service_analysis(service_name: str) -> str:
    """
    Suggest relevant analyses for a GCP service.

    Args:
        service_name: Name or key of the GCP service

    Returns:
        Formatted suggestions for analysis
    """
    discovery = GCPServiceDiscovery()

    # Find the service
    service_key = None
    for key, details in discovery.GCP_SERVICES.items():
        if service_name.lower() in key.lower() or service_name.lower() in details['name'].lower():
            service_key = key
            break

    if not service_key:
        return f"Service '{service_name}' not found. Use discover_gcp_services() to see available services."

    service = discovery.GCP_SERVICES[service_key]

    output = f"💡 Analysis Suggestions for {service['name']}\n"
    output += "=" * 50 + "\n\n"

    output += f"Service API: {service['api']}\n"
    output += f"Resource Types: {', '.join(service.get('resource_types', []))}\n\n"

    output += "Available Analyses:\n"
    for analysis_type, query in service.get('analysis_queries', {}).items():
        output += f"\n📍 {analysis_type.upper()}:\n"
        output += f"   Purpose: Analyze {analysis_type} aspects of {service['name']}\n"
        output += f"   Command: analyze_gcp_service('{service_key}', '{analysis_type}')\n"

    output += "\n\nCustom Analysis:\n"
    output += "You can also create custom queries:\n"
    output += f"analyze_gcp_service('{service_key}', 'custom', 'YOUR SQL QUERY HERE')\n"

    return output


# New functions that use documentation parser
def learn_service_from_url(documentation_url: str) -> str:
    """
    Learn about a new GCP service by parsing its documentation URL.

    Args:
        documentation_url: URL to GCP service documentation

    Returns:
        Analysis of the learned service
    """
    if not HAS_DOC_PARSER:
        return "Documentation parser not available. Please install BeautifulSoup4: pip install beautifulsoup4"

    try:
        result = parse_doc(documentation_url)
        return result
    except Exception as e:
        return f"Error learning from URL: {str(e)}"


def discover_new_gcp_services(release_notes_url: Optional[str] = None) -> str:
    """
    Discover newly released GCP services from release notes.

    Args:
        release_notes_url: Optional URL to release notes page

    Returns:
        List of newly discovered services
    """
    if not HAS_DOC_PARSER:
        return "Documentation parser not available. Please install BeautifulSoup4: pip install beautifulsoup4"

    try:
        result = discover_new(release_notes_url)
        return result
    except Exception as e:
        return f"Error discovering new services: {str(e)}"


def register_new_service(
    service_name: str,
    api_endpoint: str,
    documentation_url: str,
    description: str = ""
) -> str:
    """
    Register a new GCP service manually for analysis.

    Args:
        service_name: Name of the new service
        api_endpoint: API endpoint (e.g., newservice.googleapis.com)
        documentation_url: URL to service documentation
        description: Brief description of the service

    Returns:
        Registration status
    """
    if not HAS_DOC_PARSER:
        return "Documentation parser not available. Please install BeautifulSoup4: pip install beautifulsoup4"

    try:
        # Register the service
        result = register_service(
            service_name=service_name,
            api_endpoint=api_endpoint,
            documentation_url=documentation_url,
            resource_types=[],
            capabilities=[]
        )

        # Also add to GCPServiceDiscovery catalog for immediate use
        discovery = GCPServiceDiscovery()
        service_key = api_endpoint.replace('.googleapis.com', '').replace('-', '')

        # Add to runtime catalog (won't persist between runs)
        discovery.GCP_SERVICES[service_key] = {
            'name': service_name,
            'api': api_endpoint,
            'description': description,
            'resource_types': [],
            'analysis_queries': {
                'overview': f"-- Custom query for {service_name}",
                'security': f"-- Security analysis for {service_name}",
            }
        }

        return result
    except Exception as e:
        return f"Error registering service: {str(e)}"


def learn_from_api_spec(api_spec_url: str) -> str:
    """
    Learn about a service from its API specification (OpenAPI, Proto, etc).

    Args:
        api_spec_url: URL to API specification

    Returns:
        Parsed API information
    """
    if not HAS_DOC_PARSER:
        return "Documentation parser not available. Please install BeautifulSoup4: pip install beautifulsoup4"

    try:
        result = learn_api(api_spec_url)
        return result
    except Exception as e:
        return f"Error learning from API spec: {str(e)}"


if __name__ == "__main__":
    # Test the service discovery
    print("Testing GCP Service Discovery...")

    # Discover services
    print(discover_gcp_services())

    # Analyze a specific service
    print(analyze_gcp_service("compute", "security"))

    # Get service resources
    print(get_service_resources("storage"))

    # Get suggestions
    print(suggest_service_analysis("bigquery"))
