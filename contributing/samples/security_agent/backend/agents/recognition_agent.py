"""
Recognition Agent - RADAR Phase 1

Discovers and inventories all cloud resources.
The "eyes" of the RADAR system.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class RecognitionAgent:
    """
    RADAR Phase 1: Recognize - Discovery and Inventory
    
    This agent discovers what exists in your environment.
    It's read-only and focused on cataloging resources.
    
    Key responsibilities:
    - Complete resource discovery
    - Service account inventory
    - Resource relationship mapping
    - Anomaly detection (test/temp resources in production)
    """
    
    def __init__(self, project_id: str):
        """Initialize Recognition Agent for resource discovery."""
        self.project_id = project_id
        self.name = "RecognitionAgent"
        self.description = "Discovers and inventories cloud resources"
        logger.info(f"👁️ Recognition Agent initialized for project {project_id}")
    
    async def discover_all_resources(self) -> Dict[str, Any]:
        """Perform complete resource discovery."""
        logger.info(f"👁️ Starting resource discovery for {self.project_id}")
        
        discovery_results = {
            "timestamp": datetime.now().isoformat(),
            "project_id": self.project_id,
            "resources": {},
            "summary": {},
            "anomalies": []
        }
        
        # Import tools we need
        from backend.api.asset_inventory import list_assets, AssetListRequest
        from backend.api.iam import list_service_accounts, ServiceAccountListRequest
        
        # Discover compute/storage/network resources
        try:
            asset_request = AssetListRequest(project_id=self.project_id)
            assets_result = await list_assets(asset_request)
        except Exception as e:
            logger.error(f"Failed to list assets: {e}")
            assets_result = {"success": False, "error": str(e)}
        
        if assets_result.get("success"):
            assets = assets_result.get("assets", [])
            discovery_results["resources"]["all_assets"] = assets
            discovery_results["summary"]["total_resources"] = len(assets)
            
            # Categorize by type
            asset_types = {}
            for asset in assets:
                asset_type = asset.get("asset_type", "unknown")
                if asset_type not in asset_types:
                    asset_types[asset_type] = 0
                asset_types[asset_type] += 1
            discovery_results["summary"]["by_type"] = asset_types
            
            # Flag unusual resources
            self._detect_resource_anomalies(assets, discovery_results["anomalies"])
        
        # Discover service accounts
        try:
            sa_request = ServiceAccountListRequest(project_id=self.project_id)
            sa_result = await list_service_accounts(sa_request)
        except Exception as e:
            logger.error(f"Failed to list service accounts: {e}")
            sa_result = {"success": False, "error": str(e)}
        
        if sa_result.get("success"):
            service_accounts = sa_result.get("service_accounts", [])
            discovery_results["resources"]["service_accounts"] = service_accounts
            discovery_results["summary"]["service_accounts"] = len(service_accounts)
            
            # Flag potential anomalies in service accounts
            self._detect_service_account_anomalies(service_accounts, discovery_results["anomalies"])
        
        discovery_results["success"] = True
        discovery_results["phase"] = "recognition"
        return discovery_results
    
    def _detect_resource_anomalies(self, assets: List[Dict], anomalies: List[Dict]):
        """Detect anomalies in discovered resources."""
        for asset in assets:
            name = asset.get("name", "")
            asset_type = asset.get("asset_type", "")
            
            # Check for test/temp resources
            if any(indicator in name.lower() for indicator in ["test", "temp", "demo", "sample"]):
                anomalies.append({
                    "type": "resource",
                    "resource": name,
                    "asset_type": asset_type,
                    "reason": "Test/temporary resource in production environment"
                })
            
            # Check for default names that might indicate misconfiguration
            if "default" in name.lower() and "network" not in asset_type.lower():
                anomalies.append({
                    "type": "resource",
                    "resource": name,
                    "asset_type": asset_type,
                    "reason": "Default naming might indicate misconfiguration"
                })
    
    def _detect_service_account_anomalies(self, service_accounts: List[Dict], anomalies: List[Dict]):
        """Detect anomalies in service accounts."""
        for sa in service_accounts:
            email = sa.get("email", "")
            display_name = sa.get("display_name", "")
            
            # Check for test/temp service accounts
            if any(indicator in email.lower() for indicator in ["test", "temp", "demo", "dev"]):
                anomalies.append({
                    "type": "service_account",
                    "resource": email,
                    "reason": "Test/temporary service account in production"
                })
            
            # Check for disabled accounts (might indicate unused resources)
            if sa.get("disabled", False):
                anomalies.append({
                    "type": "service_account",
                    "resource": email,
                    "reason": "Disabled service account still exists"
                })
            
            # Check for generic names
            if display_name and "service account" in display_name.lower() and len(display_name) < 20:
                anomalies.append({
                    "type": "service_account",
                    "resource": email,
                    "reason": "Generic service account name lacks description"
                })
    
    async def search_specific_resources(self, query: str) -> Dict[str, Any]:
        """Search for specific resources based on query."""
        logger.info(f"🔍 Searching for resources matching: {query}")
        
        from backend.api.asset_inventory import search_assets, AssetSearchRequest
        
        try:
            search_request = AssetSearchRequest(
                scope=f"projects/{self.project_id}",
                query=query
            )
            search_result = await search_assets(search_request)
            
            if search_result.get("success"):
                search_result["phase"] = "recognition"
                search_result["search_query"] = query
            
            return search_result
        except Exception as e:
            logger.error(f"Failed to search assets: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_resource_inventory_summary(self) -> Dict[str, Any]:
        """Get a high-level summary of resource inventory."""
        discovery = await self.discover_all_resources()
        
        if discovery.get("success"):
            summary = discovery.get("summary", {})
            
            # Create executive summary
            return {
                "success": True,
                "phase": "recognition",
                "summary": {
                    "total_resources": summary.get("total_resources", 0),
                    "service_accounts": summary.get("service_accounts", 0),
                    "resource_types": len(summary.get("by_type", {})),
                    "anomalies_detected": len(discovery.get("anomalies", [])),
                    "top_resource_types": self._get_top_resource_types(summary.get("by_type", {}))
                },
                "health_indicators": {
                    "has_anomalies": len(discovery.get("anomalies", [])) > 0,
                    "resource_diversity": len(summary.get("by_type", {})) > 5,
                    "service_account_ratio": summary.get("service_accounts", 0) / max(summary.get("total_resources", 1), 1)
                }
            }
        
        return discovery
    
    def _get_top_resource_types(self, by_type: Dict[str, int], limit: int = 5) -> List[Dict]:
        """Get top resource types by count."""
        sorted_types = sorted(by_type.items(), key=lambda x: x[1], reverse=True)
        return [
            {"type": t, "count": c}
            for t, c in sorted_types[:limit]
        ]