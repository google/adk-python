"""
GCP Project Snapshot Service
Caches project-wide metrics and resource counts in JSON format to avoid repeated API calls.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from google.cloud import asset_v1
from google.cloud import recommender_v1
from google.cloud import storage
from google.cloud import compute_v1
from google.cloud import functions_v1
from google.cloud import monitoring_v3

logger = logging.getLogger(__name__)

class GCPProjectSnapshot:
    """Service for caching GCP project metrics and resource counts."""
    
    def __init__(self, project_id: str, cache_duration_hours: int = 6):
        self.project_id = project_id
        self.cache_duration_hours = cache_duration_hours
        self.cache_dir = f"cache/snapshots/{project_id}"
        self.snapshot_file = f"{self.cache_dir}/project_snapshot.json"
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize clients (lazy loaded)
        self._asset_client = None
        self._recommender_client = None
        self._storage_client = None
        self._compute_client = None
        self._functions_client = None
        self._monitoring_client = None
    
    @property
    def asset_client(self):
        if not self._asset_client:
            self._asset_client = asset_v1.AssetServiceClient()
        return self._asset_client
    
    @property
    def recommender_client(self):
        if not self._recommender_client:
            self._recommender_client = recommender_v1.RecommenderClient()
        return self._recommender_client
    
    @property
    def storage_client(self):
        if not self._storage_client:
            self._storage_client = storage.Client(project=self.project_id)
        return self._storage_client
    
    @property
    def compute_client(self):
        if not self._compute_client:
            self._compute_client = compute_v1.InstancesClient()
        return self._compute_client
    
    @property
    def functions_client(self):
        if not self._functions_client:
            self._functions_client = functions_v1.CloudFunctionsServiceClient()
        return self._functions_client
    
    @property
    def monitoring_client(self):
        if not self._monitoring_client:
            self._monitoring_client = monitoring_v3.MetricServiceClient()
        return self._monitoring_client
    
    def get_snapshot(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get cached project snapshot or create new one if expired."""
        if not force_refresh and self._is_cache_valid():
            logger.info(f"📊 Using cached snapshot for project {self.project_id}")
            return self._load_snapshot()
        
        logger.info(f"🔄 Creating fresh snapshot for project {self.project_id}")
        return self._create_snapshot()
    
    def _is_cache_valid(self) -> bool:
        """Check if cached snapshot is still valid."""
        if not os.path.exists(self.snapshot_file):
            return False
        
        file_age = time.time() - os.path.getmtime(self.snapshot_file)
        max_age = self.cache_duration_hours * 3600  # Convert to seconds
        return file_age < max_age
    
    def _load_snapshot(self) -> Dict[str, Any]:
        """Load snapshot from cache file."""
        try:
            with open(self.snapshot_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading snapshot: {e}")
            return self._create_snapshot()
    
    def _create_snapshot(self) -> Dict[str, Any]:
        """Create fresh project snapshot."""
        snapshot = {
            "project_id": self.project_id,
            "timestamp": datetime.now().isoformat(),
            "cache_duration_hours": self.cache_duration_hours,
            "resources": {},
            "recommendations": {},
            "security_insights": {},
            "performance_metrics": {}
        }
        
        try:
            # Get resource counts
            snapshot["resources"] = self._get_resource_counts()
            
            # Get recommendations
            snapshot["recommendations"] = self._get_recommendations()
            
            # Get security insights
            snapshot["security_insights"] = self._get_security_insights()
            
            # Get performance metrics
            snapshot["performance_metrics"] = self._get_performance_metrics()
            
            # Save to cache
            self._save_snapshot(snapshot)
            
            logger.info(f"✅ Created snapshot with {len(snapshot['resources'])} resource types")
            return snapshot
            
        except Exception as e:
            logger.error(f"Error creating snapshot: {e}")
            # Return minimal snapshot on error
            snapshot["error"] = str(e)
            return snapshot
    
    def _get_resource_counts(self) -> Dict[str, int]:
        """Get counts of different resource types."""
        resource_counts = {}
        
        try:
            # Asset inventory aggregation
            request = asset_v1.SearchAllResourcesRequest(
                scope=f"projects/{self.project_id}",
                page_size=1000
            )
            
            resources = list(self.asset_client.search_all_resources(request=request))
            
            # Count by resource type
            for resource in resources:
                asset_type = resource.asset_type.split('/')[-1]
                resource_counts[asset_type] = resource_counts.get(asset_type, 0) + 1
            
            # Specific resource counts
            resource_counts.update({
                "total_assets": len(resources),
                "compute_instances": self._count_compute_instances(),
                "storage_buckets": self._count_storage_buckets(),
                "cloud_functions": self._count_cloud_functions()
            })
            
        except Exception as e:
            logger.error(f"Error getting resource counts: {e}")
            resource_counts["error"] = str(e)
        
        return resource_counts
    
    def _count_compute_instances(self) -> int:
        """Count compute instances."""
        try:
            instances = list(self.compute_client.aggregated_list(project=self.project_id))
            return sum(len(zone_instances.instances) for _, zone_instances in instances if zone_instances.instances)
        except Exception as e:
            logger.warning(f"Could not count compute instances: {e}")
            return 0
    
    def _count_storage_buckets(self) -> int:
        """Count storage buckets."""
        try:
            buckets = list(self.storage_client.list_buckets())
            return len(buckets)
        except Exception as e:
            logger.warning(f"Could not count storage buckets: {e}")
            return 0
    
    def _count_cloud_functions(self) -> int:
        """Count cloud functions."""
        try:
            parent = f"projects/{self.project_id}/locations/-"
            functions = list(self.functions_client.list_functions(parent=parent))
            return len(functions)
        except Exception as e:
            logger.warning(f"Could not count cloud functions: {e}")
            return 0
    
    def _get_recommendations(self) -> Dict[str, Any]:
        """Get recommendations from Recommender API."""
        recommendations = {
            "security": [],
            "cost": [],
            "performance": [],
            "reliability": []
        }
        
        try:
            # Define recommender types to fetch
            recommender_types = [
                "google.compute.instance.MachineTypeRecommender",
                "google.compute.disk.IdleResourceRecommender", 
                "google.cloudsql.instance.OutOfDiskRecommender",
                "google.storage.bucket.LifecycleConfigRecommender"
            ]
            
            for recommender_type in recommender_types:
                try:
                    parent = f"projects/{self.project_id}/locations/global/recommenders/{recommender_type}"
                    request = recommender_v1.ListRecommendationsRequest(parent=parent)
                    
                    recs = list(self.recommender_client.list_recommendations(request=request))
                    
                    # Categorize recommendations
                    category = "performance"  # Default
                    if "security" in recommender_type.lower():
                        category = "security"
                    elif "cost" in recommender_type.lower() or "idle" in recommender_type.lower():
                        category = "cost"
                    elif "reliability" in recommender_type.lower():
                        category = "reliability"
                    
                    for rec in recs:
                        recommendations[category].append({
                            "recommender": recommender_type.split('.')[-1],
                            "priority": rec.priority.name if rec.priority else "MEDIUM",
                            "description": rec.description,
                            "state": rec.state_info.state.name if rec.state_info else "ACTIVE"
                        })
                        
                except Exception as e:
                    logger.warning(f"Could not fetch recommendations for {recommender_type}: {e}")
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            recommendations["error"] = str(e)
        
        return recommendations
    
    def _get_security_insights(self) -> Dict[str, Any]:
        """Get security-related insights."""
        insights = {
            "iam_policies": 0,
            "service_accounts": 0,
            "encryption_keys": 0,
            "security_findings": []
        }
        
        try:
            # Use asset inventory to get security-related resources
            request = asset_v1.SearchAllResourcesRequest(
                scope=f"projects/{self.project_id}",
                asset_types=[
                    "iam.googleapis.com/ServiceAccount",
                    "cloudkms.googleapis.com/CryptoKey",
                    "compute.googleapis.com/Firewall"
                ]
            )
            
            resources = list(self.asset_client.search_all_resources(request=request))
            
            for resource in resources:
                if "ServiceAccount" in resource.asset_type:
                    insights["service_accounts"] += 1
                elif "CryptoKey" in resource.asset_type:
                    insights["encryption_keys"] += 1
            
        except Exception as e:
            logger.error(f"Error getting security insights: {e}")
            insights["error"] = str(e)
        
        return insights
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance-related metrics."""
        metrics = {
            "cpu_utilization": [],
            "memory_utilization": [],
            "disk_utilization": [],
            "network_throughput": []
        }
        
        try:
            # Get recent metrics from Cloud Monitoring
            project_name = f"projects/{self.project_id}"
            
            # Define time range (last 24 hours)
            now = time.time()
            interval = monitoring_v3.TimeInterval({
                "end_time": {"seconds": int(now)},
                "start_time": {"seconds": int(now - 86400)}  # 24 hours ago
            })
            
            # Sample metric types to fetch
            metric_types = [
                "compute.googleapis.com/instance/cpu/utilization",
                "compute.googleapis.com/instance/memory/utilization"
            ]
            
            for metric_type in metric_types:
                try:
                    request = monitoring_v3.ListTimeSeriesRequest(
                        name=project_name,
                        filter=f'metric.type="{metric_type}"',
                        interval=interval,
                        view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL
                    )
                    
                    results = list(self.monitoring_client.list_time_series(request=request))
                    
                    if "cpu" in metric_type:
                        metrics["cpu_utilization"] = len(results)
                    elif "memory" in metric_type:
                        metrics["memory_utilization"] = len(results)
                        
                except Exception as e:
                    logger.warning(f"Could not fetch metric {metric_type}: {e}")
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            metrics["error"] = str(e)
        
        return metrics
    
    def _save_snapshot(self, snapshot: Dict[str, Any]):
        """Save snapshot to cache file."""
        try:
            with open(self.snapshot_file, 'w') as f:
                json.dump(snapshot, f, indent=2, default=str)
            logger.info(f"💾 Saved snapshot to {self.snapshot_file}")
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")
    
    def get_resource_count(self, resource_type: str) -> int:
        """Get count of specific resource type from snapshot."""
        snapshot = self.get_snapshot()
        return snapshot.get("resources", {}).get(resource_type, 0)
    
    def get_recommendations_by_category(self, category: str) -> list:
        """Get recommendations by category from snapshot."""
        snapshot = self.get_snapshot()
        return snapshot.get("recommendations", {}).get(category, [])
    
    def refresh_snapshot(self) -> Dict[str, Any]:
        """Force refresh the snapshot."""
        return self.get_snapshot(force_refresh=True)