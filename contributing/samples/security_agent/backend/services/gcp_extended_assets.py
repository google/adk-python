"""
Extended GCP Asset Discovery
Handles additional asset types: Cloud Functions, BigQuery, Pub/Sub, GKE
"""

import logging
from typing import List, Dict, Any
import os

logger = logging.getLogger(__name__)

class ExtendedAssetDiscovery:
    """Discover extended GCP asset types"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
    
    async def fetch_cloud_functions(self) -> List[Dict[str, Any]]:
        """Fetch Cloud Functions from GCP"""
        try:
            from google.cloud import functions_v1
            
            client = functions_v1.CloudFunctionsServiceClient()
            functions = []
            
            parent = f"projects/{self.project_id}/locations/-"
            request = functions_v1.ListFunctionsRequest(parent=parent)
            
            for function in client.list_functions(request=request):
                functions.append({
                    "name": function.name,
                    "asset_type": "cloudfunctions.googleapis.com/Function",
                    "status": function.status.name,
                    "entry_point": function.entry_point,
                    "runtime": function.runtime,
                    "trigger": function.event_trigger.event_type if function.event_trigger else "HTTP",
                    "available_memory": function.available_memory_mb,
                    "timeout": function.timeout.seconds if function.timeout else None,
                    "ingress_settings": function.ingress_settings.name,
                    "vpc_connector": function.vpc_connector,
                    "service_account": function.service_account_email,
                    "public_access": function.ingress_settings.name == "ALLOW_ALL"
                })
            
            return functions
            
        except Exception as e:
            error_str = str(e)
            if "403" in error_str or "Permission" in error_str or "not enabled" in error_str.lower():
                logger.debug(f"Cloud Functions API not available: {e}")
            else:
                logger.debug(f"Could not fetch Cloud Functions: {e}")
            return []
    
    async def fetch_bigquery_datasets(self) -> List[Dict[str, Any]]:
        """Fetch BigQuery datasets from GCP"""
        try:
            from google.cloud import bigquery
            
            client = bigquery.Client(project=self.project_id)
            datasets = []
            
            for dataset in client.list_datasets():
                dataset_ref = client.get_dataset(dataset.dataset_id)
                
                # Check if dataset has public access
                public_access = False
                if dataset_ref.access_entries:
                    for entry in dataset_ref.access_entries:
                        if entry.entity_id == "allUsers" or entry.entity_id == "allAuthenticatedUsers":
                            public_access = True
                            break
                
                datasets.append({
                    "name": dataset.dataset_id,
                    "asset_type": "bigquery.googleapis.com/Dataset",
                    "location": dataset_ref.location,
                    "created": dataset_ref.created.isoformat() if dataset_ref.created else None,
                    "modified": dataset_ref.modified.isoformat() if dataset_ref.modified else None,
                    "description": dataset_ref.description,
                    "default_encryption": dataset_ref.default_encryption_configuration.kms_key_name if dataset_ref.default_encryption_configuration else None,
                    "public_access": public_access,
                    "table_count": len(list(client.list_tables(dataset.dataset_id)))
                })
            
            return datasets
            
        except Exception as e:
            error_str = str(e)
            if "403" in error_str or "Permission" in error_str or "not enabled" in error_str.lower():
                logger.debug(f"BigQuery API not available: {e}")
            else:
                logger.debug(f"Could not fetch BigQuery datasets: {e}")
            return []
    
    async def fetch_pubsub_topics(self) -> List[Dict[str, Any]]:
        """Fetch Pub/Sub topics from GCP"""
        try:
            from google.cloud import pubsub_v1
            
            publisher = pubsub_v1.PublisherClient()
            topics = []
            
            project_path = f"projects/{self.project_id}"
            
            for topic in publisher.list_topics(request={"project": project_path}):
                # Get topic IAM policy to check for public access
                policy = publisher.get_iam_policy(request={"resource": topic.name})
                
                public_access = False
                for binding in policy.bindings:
                    if "allUsers" in binding.members or "allAuthenticatedUsers" in binding.members:
                        public_access = True
                        break
                
                topics.append({
                    "name": topic.name.split('/')[-1],
                    "asset_type": "pubsub.googleapis.com/Topic",
                    "full_name": topic.name,
                    "kms_key": topic.kms_key_name if topic.kms_key_name else None,
                    "message_retention": topic.message_retention_duration.seconds if topic.message_retention_duration else None,
                    "public_access": public_access,
                    "labels": dict(topic.labels) if topic.labels else {}
                })
            
            return topics
            
        except Exception as e:
            error_str = str(e)
            if "403" in error_str or "Permission" in error_str or "not enabled" in error_str.lower():
                logger.debug(f"Pub/Sub API not available: {e}")
            else:
                logger.debug(f"Could not fetch Pub/Sub topics: {e}")
            return []
    
    async def fetch_gke_clusters(self) -> List[Dict[str, Any]]:
        """Fetch GKE clusters from GCP"""
        try:
            from google.cloud import container_v1
            
            client = container_v1.ClusterManagerClient()
            clusters = []
            
            parent = f"projects/{self.project_id}/locations/-"
            
            for cluster in client.list_clusters(parent=parent).clusters:
                # Analyze security settings
                security_issues = []
                
                if not cluster.private_cluster_config or not cluster.private_cluster_config.enable_private_nodes:
                    security_issues.append("Public nodes enabled")
                    
                if not cluster.network_policy or not cluster.network_policy.enabled:
                    security_issues.append("Network policy disabled")
                    
                if not cluster.binary_authorization or not cluster.binary_authorization.enabled:
                    security_issues.append("Binary authorization disabled")
                    
                if cluster.legacy_abac and cluster.legacy_abac.enabled:
                    security_issues.append("Legacy ABAC enabled")
                
                clusters.append({
                    "name": cluster.name,
                    "asset_type": "container.googleapis.com/Cluster",
                    "location": cluster.location,
                    "status": cluster.status.name,
                    "version": cluster.current_master_version,
                    "node_count": cluster.current_node_count,
                    "node_version": cluster.current_node_version,
                    "network": cluster.network,
                    "private_cluster": cluster.private_cluster_config.enable_private_nodes if cluster.private_cluster_config else False,
                    "network_policy_enabled": cluster.network_policy.enabled if cluster.network_policy else False,
                    "binary_authorization": cluster.binary_authorization.enabled if cluster.binary_authorization else False,
                    "workload_identity": cluster.workload_identity_config.workload_pool if cluster.workload_identity_config else None,
                    "security_issues": security_issues,
                    "high_risk": len(security_issues) > 2
                })
            
            return clusters
            
        except Exception as e:
            error_str = str(e)
            if "403" in error_str or "Permission" in error_str or "not enabled" in error_str.lower():
                logger.debug(f"GKE API not available: {e}")
            else:
                logger.debug(f"Could not fetch GKE clusters: {e}")
            return []
    
    async def fetch_cloud_run_services(self) -> List[Dict[str, Any]]:
        """Fetch Cloud Run services from GCP"""
        try:
            from google.cloud import run_v2
            
            client = run_v2.ServicesClient()
            services = []
            
            parent = f"projects/{self.project_id}/locations/-"
            request = run_v2.ListServicesRequest(parent=parent)
            
            for service in client.list_services(request=request):
                # Check if service allows unauthenticated access
                public_access = False
                if service.ingress == run_v2.IngressTraffic.INGRESS_TRAFFIC_ALL:
                    # Check IAM bindings for allUsers
                    public_access = True  # Simplified - would need IAM check
                
                services.append({
                    "name": service.name.split('/')[-1],
                    "asset_type": "run.googleapis.com/Service",
                    "uri": service.uri if hasattr(service, 'uri') else "",
                    "generation": service.generation if hasattr(service, 'generation') else 0,
                    "ingress": service.ingress.name if hasattr(service, 'ingress') and hasattr(service.ingress, 'name') else "UNKNOWN",
                    "launch_stage": service.launch_stage.name if hasattr(service, 'launch_stage') and hasattr(service.launch_stage, 'name') else "GA",
                    "public_access": public_access,
                    "service_account": service.template.service_account if hasattr(service, 'template') and service.template and hasattr(service.template, 'service_account') else None,
                    "max_instances": service.template.scaling.max_instance_count if hasattr(service, 'template') and service.template and hasattr(service.template, 'scaling') and service.template.scaling and hasattr(service.template.scaling, 'max_instance_count') else None
                })
            
            return services
            
        except Exception as e:
            # Log at debug level when service is not enabled
            error_str = str(e)
            if "403" in error_str or "Permission" in error_str or "not enabled" in error_str.lower():
                logger.debug(f"Cloud Run API not available: {e}")
            else:
                logger.debug(f"Could not fetch Cloud Run services: {e}")
            return []
    
    async def fetch_app_engine_services(self) -> List[Dict[str, Any]]:
        """Fetch App Engine services from GCP"""
        try:
            from google.cloud import appengine_admin_v1
            
            client = appengine_admin_v1.ServicesClient()
            services = []
            
            parent = f"apps/{self.project_id}"
            request = appengine_admin_v1.ListServicesRequest(parent=parent)
            
            for service in client.list_services(request=request):
                services.append({
                    "name": service.name,
                    "asset_type": "appengine.googleapis.com/Service",
                    "id": service.id,
                    "split": dict(service.split.allocations) if service.split else {}
                })
            
            return services
            
        except Exception as e:
            logger.warning(f"Could not fetch App Engine services: {e}")
            return []