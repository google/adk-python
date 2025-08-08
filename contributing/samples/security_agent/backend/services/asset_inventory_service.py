"""
Real GCP Asset Inventory Service using Google Cloud Asset API
"""
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from google.cloud import asset_v1
from google.cloud import compute_v1
from google.cloud import storage
from google.cloud import monitoring_v3
from google.oauth2 import service_account
import logging

logger = logging.getLogger(__name__)

class GCPAssetInventoryService:
    """Service for retrieving real GCP asset inventory data."""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.project_resource = f"projects/{project_id}"
        
        # Set up authentication
        credentials_path = os.path.join(
            os.path.dirname(__file__), 
            "..", 
            "config", 
            "secrets", 
            f"{project_id}-52fed2a2dac3.json"
        )
        
        if os.path.exists(credentials_path):
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            self.credentials = credentials
            logger.info(f"Loaded service account credentials for {project_id}")
        else:
            logger.warning(f"Service account file not found: {credentials_path}")
            self.credentials = None
        
        # Initialize clients
        try:
            if self.credentials:
                self.asset_client = asset_v1.AssetServiceClient(credentials=self.credentials)
                self.compute_client = compute_v1.InstancesClient(credentials=self.credentials)
                self.storage_client = storage.Client(credentials=self.credentials)
                self.monitoring_client = monitoring_v3.MetricServiceClient(credentials=self.credentials)
                logger.info("Successfully initialized GCP service clients")
            else:
                # Fall back to default credentials
                self.asset_client = asset_v1.AssetServiceClient()
                self.compute_client = compute_v1.InstancesClient()
                self.storage_client = storage.Client()
                self.monitoring_client = monitoring_v3.MetricServiceClient()
                logger.info("Using default GCP credentials")
        except Exception as e:
            logger.error(f"Failed to initialize GCP clients: {e}")
            raise
    
    async def get_complete_asset_inventory(self) -> Dict[str, Any]:
        """Get complete asset inventory for the project."""
        try:
            logger.info(f"Fetching complete asset inventory for {self.project_id}")
            
            # Get all assets using Asset Inventory API
            assets = await self._get_all_assets()
            
            # Parse and categorize assets
            inventory = {
                "compute": await self._parse_compute_assets(assets),
                "storage": await self._parse_storage_assets(assets),
                "networking": await self._parse_networking_assets(assets),
                "security": await self._parse_security_assets(assets),
                "cost_analysis": await self._get_cost_analysis(),
                "total_assets": len(assets),
                "last_updated": datetime.utcnow().isoformat(),
                "project_id": self.project_id
            }
            
            logger.info(f"Successfully retrieved {len(assets)} assets for {self.project_id}")
            return inventory
            
        except Exception as e:
            logger.error(f"Error getting asset inventory: {e}")
            return self._get_fallback_inventory()
    
    async def _get_all_assets(self) -> List[Any]:
        """Get all assets using the Asset Inventory API."""
        try:
            # Request all asset types
            request = asset_v1.ListAssetsRequest(
                parent=self.project_resource,
                content_type=asset_v1.ContentType.RESOURCE,
                # Get assets with full resource data
                asset_types=[
                    "compute.googleapis.com/Instance",
                    "compute.googleapis.com/Disk", 
                    "storage.googleapis.com/Bucket",
                    "sqladmin.googleapis.com/Instance",
                    "compute.googleapis.com/Network",
                    "compute.googleapis.com/Firewall",
                    "iam.googleapis.com/ServiceAccount",
                    "appengine.googleapis.com/Application",
                    "redis.googleapis.com/Instance",
                    "container.googleapis.com/Cluster"
                ]
            )
            
            # List all assets
            assets = []
            page_result = self.asset_client.list_assets(request=request)
            
            for asset in page_result:
                assets.append(asset)
                
            return assets
            
        except Exception as e:
            logger.error(f"Error fetching assets from Asset API: {e}")
            return []
    
    async def _parse_compute_assets(self, assets: List[Any]) -> Dict[str, Any]:
        """Parse compute-related assets."""
        compute_data = {
            "vm_instances": {"count": 0, "running": 0, "stopped": 0, "details": []},
            "instance_groups": {"count": 0, "details": []},
            "load_balancers": {"count": 0, "details": []},
            "app_engine": {"count": 0, "details": []}
        }
        
        for asset in assets:
            if asset.asset_type == "compute.googleapis.com/Instance":
                compute_data["vm_instances"]["count"] += 1
                
                # Get instance details
                instance_data = {
                    "name": asset.resource.data.get("name", "unknown"),
                    "zone": asset.resource.data.get("zone", "").split("/")[-1] if asset.resource.data.get("zone") else "unknown",
                    "machine_type": asset.resource.data.get("machineType", "").split("/")[-1] if asset.resource.data.get("machineType") else "unknown",
                    "status": asset.resource.data.get("status", "unknown"),
                    "creation_timestamp": asset.resource.data.get("creationTimestamp", ""),
                    "network_interfaces": len(asset.resource.data.get("networkInterfaces", []))
                }
                
                if instance_data["status"] == "RUNNING":
                    compute_data["vm_instances"]["running"] += 1
                else:
                    compute_data["vm_instances"]["stopped"] += 1
                
                compute_data["vm_instances"]["details"].append(instance_data)
                
            elif asset.asset_type == "appengine.googleapis.com/Application":
                compute_data["app_engine"]["count"] += 1
                compute_data["app_engine"]["details"].append({
                    "id": asset.resource.data.get("id", "unknown"),
                    "serving_status": asset.resource.data.get("servingStatus", "unknown"),
                    "location_id": asset.resource.data.get("locationId", "unknown")
                })
        
        return compute_data
    
    async def _parse_storage_assets(self, assets: List[Any]) -> Dict[str, Any]:
        """Parse storage-related assets."""
        storage_data = {
            "cloud_storage": {"count": 0, "total_size_gb": 0, "details": []},
            "persistent_disks": {"count": 0, "total_size_gb": 0, "details": []},
            "cloud_sql": {"count": 0, "details": []}
        }
        
        for asset in assets:
            if asset.asset_type == "storage.googleapis.com/Bucket":
                storage_data["cloud_storage"]["count"] += 1
                
                bucket_data = {
                    "name": asset.resource.data.get("name", "unknown"),
                    "location": asset.resource.data.get("location", "unknown"),
                    "storage_class": asset.resource.data.get("storageClass", "unknown"),
                    "creation_time": asset.resource.data.get("timeCreated", ""),
                    "versioning_enabled": asset.resource.data.get("versioning", {}).get("enabled", False),
                    "lifecycle_rules": len(asset.resource.data.get("lifecycle", {}).get("rule", []))
                }
                
                storage_data["cloud_storage"]["details"].append(bucket_data)
                
            elif asset.asset_type == "compute.googleapis.com/Disk":
                storage_data["persistent_disks"]["count"] += 1
                
                size_gb = int(asset.resource.data.get("sizeGb", 0))
                storage_data["persistent_disks"]["total_size_gb"] += size_gb
                
                disk_data = {
                    "name": asset.resource.data.get("name", "unknown"),
                    "zone": asset.resource.data.get("zone", "").split("/")[-1] if asset.resource.data.get("zone") else "unknown",
                    "size_gb": size_gb,
                    "type": asset.resource.data.get("type", "").split("/")[-1] if asset.resource.data.get("type") else "unknown",
                    "status": asset.resource.data.get("status", "unknown"),
                    "in_use": bool(asset.resource.data.get("users"))
                }
                
                storage_data["persistent_disks"]["details"].append(disk_data)
                
            elif asset.asset_type == "sqladmin.googleapis.com/Instance":
                storage_data["cloud_sql"]["count"] += 1
                
                sql_data = {
                    "name": asset.resource.data.get("name", "unknown"),
                    "database_version": asset.resource.data.get("databaseVersion", "unknown"),
                    "tier": asset.resource.data.get("settings", {}).get("tier", "unknown"),
                    "state": asset.resource.data.get("state", "unknown"),
                    "region": asset.resource.data.get("region", "unknown"),
                    "ip_addresses": len(asset.resource.data.get("ipAddresses", []))
                }
                
                storage_data["cloud_sql"]["details"].append(sql_data)
        
        return storage_data
    
    async def _parse_networking_assets(self, assets: List[Any]) -> Dict[str, Any]:
        """Parse networking-related assets."""
        networking_data = {
            "vpc_networks": {"count": 0, "details": []},
            "firewall_rules": {"count": 0, "needs_review": 0, "details": []},
            "nat_gateways": {"count": 0, "details": []},
            "vpn_tunnels": {"count": 0, "details": []}
        }
        
        for asset in assets:
            if asset.asset_type == "compute.googleapis.com/Network":
                networking_data["vpc_networks"]["count"] += 1
                
                network_data = {
                    "name": asset.resource.data.get("name", "unknown"),
                    "auto_create_subnetworks": asset.resource.data.get("autoCreateSubnetworks", False),
                    "routing_mode": asset.resource.data.get("routingConfig", {}).get("routingMode", "unknown"),
                    "creation_timestamp": asset.resource.data.get("creationTimestamp", "")
                }
                
                networking_data["vpc_networks"]["details"].append(network_data)
                
            elif asset.asset_type == "compute.googleapis.com/Firewall":
                networking_data["firewall_rules"]["count"] += 1
                
                # Check if firewall rule needs review (overly permissive)
                source_ranges = asset.resource.data.get("sourceRanges", [])
                needs_review = "0.0.0.0/0" in source_ranges
                
                if needs_review:
                    networking_data["firewall_rules"]["needs_review"] += 1
                
                firewall_data = {
                    "name": asset.resource.data.get("name", "unknown"),
                    "direction": asset.resource.data.get("direction", "unknown"),
                    "source_ranges": source_ranges,
                    "allowed_ports": [rule.get("ports", []) for rule in asset.resource.data.get("allowed", [])],
                    "target_tags": asset.resource.data.get("targetTags", []),
                    "needs_review": needs_review
                }
                
                networking_data["firewall_rules"]["details"].append(firewall_data)
        
        return networking_data
    
    async def _parse_security_assets(self, assets: List[Any]) -> Dict[str, Any]:
        """Parse security-related assets."""
        security_data = {
            "service_accounts": {"count": 0, "details": []},
            "iam_policies": {"count": 0, "role_bindings": 0},
            "kms_keys": {"count": 0, "details": []},
            "secrets": {"count": 0, "details": []}
        }
        
        for asset in assets:
            if asset.asset_type == "iam.googleapis.com/ServiceAccount":
                security_data["service_accounts"]["count"] += 1
                
                sa_data = {
                    "email": asset.resource.data.get("email", "unknown"),
                    "display_name": asset.resource.data.get("displayName", ""),
                    "disabled": asset.resource.data.get("disabled", False),
                    "unique_id": asset.resource.data.get("uniqueId", ""),
                    "oauth2_client_id": asset.resource.data.get("oauth2ClientId", "")
                }
                
                security_data["service_accounts"]["details"].append(sa_data)
        
        return security_data
    
    async def _get_cost_analysis(self) -> Dict[str, Any]:
        """Get cost analysis (simplified version)."""
        # This would typically integrate with Cloud Billing API
        # For now, provide estimated costs based on asset counts
        return {
            "monthly_spend": "Contact Billing API for accurate data",
            "top_cost_drivers": {
                "compute": "Analysis pending",
                "storage": "Analysis pending",
                "networking": "Analysis pending"
            },
            "optimization_potential": "To be determined after analysis",
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def _get_fallback_inventory(self) -> Dict[str, Any]:
        """Return fallback inventory when API calls fail."""
        return {
            "error": True,
            "message": "Unable to retrieve real asset inventory - using fallback data",
            "compute": {
                "vm_instances": {"count": 0, "running": 0, "stopped": 0, "details": []},
                "instance_groups": {"count": 0, "details": []},
                "load_balancers": {"count": 0, "details": []},
                "app_engine": {"count": 0, "details": []}
            },
            "storage": {
                "cloud_storage": {"count": 0, "total_size_gb": 0, "details": []},
                "persistent_disks": {"count": 0, "total_size_gb": 0, "details": []},
                "cloud_sql": {"count": 0, "details": []}
            },
            "networking": {
                "vpc_networks": {"count": 0, "details": []},
                "firewall_rules": {"count": 0, "needs_review": 0, "details": []},
                "nat_gateways": {"count": 0, "details": []},
                "vpn_tunnels": {"count": 0, "details": []}
            },
            "security": {
                "service_accounts": {"count": 0, "details": []},
                "iam_policies": {"count": 0, "role_bindings": 0},
                "kms_keys": {"count": 0, "details": []},
                "secrets": {"count": 0, "details": []}
            },
            "cost_analysis": {
                "monthly_spend": 0,
                "top_cost_drivers": {"compute": 0, "storage": 0, "networking": 0},
                "optimization_potential": "0%"
            },
            "total_assets": 0,
            "last_updated": datetime.utcnow().isoformat(),
            "project_id": self.project_id
        }