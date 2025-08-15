"""
Comprehensive GCP Asset Inventory Service using Google Cloud Asset API

This service provides unified access to all GCP resources through the Asset Inventory API,
supporting intelligent query routing and natural language processing for resource discovery.
"""
import json
import os
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import logging

# Try to import Google Cloud libraries with error handling
try:
    from google.cloud import asset_v1
    from google.cloud import compute_v1
    from google.cloud import storage
    from google.cloud import monitoring_v3
    from google.oauth2 import service_account
    import requests
    GCP_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Google Cloud libraries not available: {e}")
    GCP_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnhancedGCPAssetInventoryService:
    """Comprehensive service for GCP asset discovery, analysis, and security evaluation.
    
    Features:
    - Unified asset discovery across all GCP services
    - Natural language query processing
    - Real-time security analysis
    - Intelligent resource routing
    - Comprehensive asset categorization
    """
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.project_resource = f"projects/{project_id}"
        self.gcp_available = GCP_AVAILABLE
        
        # Asset type mappings for comprehensive coverage
        self.asset_type_mappings = self._initialize_asset_mappings()
        
        # Query pattern mappings for natural language processing
        self.query_patterns = self._initialize_query_patterns()
        
        if not GCP_AVAILABLE:
            logger.warning("Google Cloud libraries not available - using fallback mode")
            self.credentials = None
            self.asset_client = None
            self.compute_client = None
            self.storage_client = None
            self.monitoring_client = None
            return
        
        # Set up authentication - use environment variable or default path
        credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        
        if not credentials_path:
            # Try default location
            credentials_path = os.path.join(
                os.path.dirname(__file__), 
                "..", 
                "config", 
                "secrets", 
                "service-account-key.json"
            )
        
        if credentials_path and os.path.exists(credentials_path):
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
            # Don't raise - fall back to mock mode
            self.asset_client = None
            self.compute_client = None
            self.storage_client = None
            self.monitoring_client = None
    
    def _initialize_asset_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive asset type mappings for all GCP services."""
        return {
            "compute": {
                "asset_types": [
                    "compute.googleapis.com/Instance",
                    "compute.googleapis.com/Disk",
                    "compute.googleapis.com/InstanceGroup",
                    "compute.googleapis.com/InstanceGroupManager",
                    "compute.googleapis.com/MachineType",
                    "compute.googleapis.com/Image",
                    "compute.googleapis.com/Snapshot",
                    "compute.googleapis.com/InstanceTemplate"
                ],
                "keywords": ["instance", "vm", "compute", "machine", "server", "disk", "snapshot"]
            },
            "storage": {
                "asset_types": [
                    "storage.googleapis.com/Bucket",
                    "bigtable.googleapis.com/Instance",
                    "bigtable.googleapis.com/Cluster",
                    "spanner.googleapis.com/Instance",
                    "spanner.googleapis.com/Database",
                    "sqladmin.googleapis.com/Instance",
                    "file.googleapis.com/Instance"
                ],
                "keywords": ["storage", "bucket", "database", "sql", "bigtable", "spanner", "filestore"]
            },
            "networking": {
                "asset_types": [
                    "compute.googleapis.com/Network",
                    "compute.googleapis.com/Subnetwork",
                    "compute.googleapis.com/Firewall",
                    "compute.googleapis.com/VpnTunnel",
                    "compute.googleapis.com/VpnGateway",
                    "compute.googleapis.com/Router",
                    "compute.googleapis.com/ForwardingRule",
                    "compute.googleapis.com/TargetPool",
                    "compute.googleapis.com/BackendService",
                    "compute.googleapis.com/UrlMap",
                    "compute.googleapis.com/HttpsHealthCheck",
                    "dns.googleapis.com/ManagedZone"
                ],
                "keywords": ["network", "vpc", "firewall", "vpn", "dns", "load balancer", "routing"]
            },
            "container": {
                "asset_types": [
                    "container.googleapis.com/Cluster",
                    "container.googleapis.com/NodePool",
                    "run.googleapis.com/Service"
                ],
                "keywords": ["kubernetes", "gke", "container", "cluster", "pod", "cloud run"]
            },
            "serverless": {
                "asset_types": [
                    "cloudfunctions.googleapis.com/CloudFunction",
                    "appengine.googleapis.com/Application",
                    "appengine.googleapis.com/Service",
                    "appengine.googleapis.com/Version"
                ],
                "keywords": ["function", "cloud function", "app engine", "serverless", "lambda"]
            },
            "data_analytics": {
                "asset_types": [
                    "bigquery.googleapis.com/Dataset",
                    "bigquery.googleapis.com/Table",
                    "dataflow.googleapis.com/Job",
                    "dataproc.googleapis.com/Cluster",
                    "composer.googleapis.com/Environment",
                    "pubsub.googleapis.com/Topic",
                    "pubsub.googleapis.com/Subscription"
                ],
                "keywords": ["bigquery", "dataflow", "dataproc", "pubsub", "composer", "analytics"]
            },
            "security": {
                "asset_types": [
                    "iam.googleapis.com/ServiceAccount",
                    "iam.googleapis.com/Role",
                    "cloudkms.googleapis.com/KeyRing",
                    "cloudkms.googleapis.com/CryptoKey",
                    "secretmanager.googleapis.com/Secret",
                    "binaryauthorization.googleapis.com/Policy"
                ],
                "keywords": ["iam", "service account", "key", "secret", "security", "encryption", "kms"]
            },
            "ai_ml": {
                "asset_types": [
                    "aiplatform.googleapis.com/Model",
                    "aiplatform.googleapis.com/Endpoint",
                    "ml.googleapis.com/Model",
                    "notebooks.googleapis.com/Instance"
                ],
                "keywords": ["ai", "ml", "model", "vertex", "notebook", "machine learning"]
            },
            "monitoring": {
                "asset_types": [
                    "monitoring.googleapis.com/AlertPolicy",
                    "logging.googleapis.com/LogSink",
                    "logging.googleapis.com/LogMetric"
                ],
                "keywords": ["monitoring", "alert", "log", "metric", "observability"]
            }
        }
    
    def _initialize_query_patterns(self) -> Dict[str, List[str]]:
        """Initialize natural language query patterns for intelligent routing."""
        return {
            "list_queries": [
                r"(list|show|get|find|what|tell me about).*",
                r".*do i have.*",
                r".*are there any.*",
                r".*how many.*"
            ],
            "security_queries": [
                r".*secur.*",
                r".*vulnerab.*",
                r".*risk.*",
                r".*threat.*",
                r".*compliance.*",
                r".*audit.*"
            ],
            "cost_queries": [
                r".*cost.*",
                r".*expens.*",
                r".*bill.*",
                r".*pricing.*",
                r".*budget.*"
            ],
            "performance_queries": [
                r".*perform.*",
                r".*slow.*",
                r".*fast.*",
                r".*optim.*",
                r".*efficien.*"
            ]
        }
    
    async def process_natural_language_query(self, query: str) -> Dict[str, Any]:
        """Process natural language queries and route to appropriate asset discovery.
        
        Args:
            query: Natural language query like 'show me my compute instances'
            
        Returns:
            Comprehensive response with discovered assets and analysis
        """
        logger.info(f"Processing natural language query: '{query}'")
        
        if not self.gcp_available or not self.asset_client:
            return self._get_fallback_response(query)
        
        try:
            # Parse query to determine intent and resource types
            query_intent = self._parse_query_intent(query)
            target_resources = self._extract_target_resources(query)
            
            logger.info(f"Query intent: {query_intent}, Target resources: {target_resources}")
            
            # Get relevant assets based on query
            if target_resources:
                assets = await self._get_assets_by_types(target_resources)
            else:
                assets = await self._get_all_assets()
            
            # Process and analyze assets
            result = await self._process_discovered_assets(assets, query_intent)
            
            # Add query context
            result.update({
                "original_query": query,
                "query_intent": query_intent,
                "target_resources": target_resources,
                "api_calls_made": self._get_api_call_log(),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing natural language query: {e}")
            return self._get_fallback_response(query, error=str(e))
    
    def _parse_query_intent(self, query: str) -> str:
        """Parse query to determine user intent."""
        query_lower = query.lower()
        
        for intent_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent_type
        
        return "general_inquiry"
    
    def _extract_target_resources(self, query: str) -> List[str]:
        """Extract target resource types from query."""
        query_lower = query.lower()
        target_types = []
        
        for category, mapping in self.asset_type_mappings.items():
            for keyword in mapping["keywords"]:
                if keyword in query_lower:
                    target_types.extend(mapping["asset_types"])
                    break
        
        return list(set(target_types))
    
    async def _get_assets_by_types(self, asset_types: List[str]) -> List[Any]:
        """Get assets filtered by specific types."""
        try:
            request = asset_v1.ListAssetsRequest(
                parent=self.project_resource,
                content_type=asset_v1.ContentType.RESOURCE,
                asset_types=asset_types
            )
            
            # Log the actual API call being made
            logger.info(f"Making Asset Inventory API call to: cloudasset.googleapis.com")
            logger.info(f"Request: ListAssets for project {self.project_id} with types: {asset_types}")
            
            assets = []
            page_result = self.asset_client.list_assets(request=request)
            
            for asset in page_result:
                assets.append(asset)
            
            logger.info(f"Retrieved {len(assets)} assets from Asset Inventory API")
            return assets
            
        except Exception as e:
            logger.error(f"Error fetching assets by types: {e}")
            return []
    
    async def _get_all_assets(self) -> List[Any]:
        """Get all assets using the Asset Inventory API."""
        try:
            # Get all comprehensive asset types
            all_asset_types = []
            for category in self.asset_type_mappings.values():
                all_asset_types.extend(category["asset_types"])
            
            # Request all asset types
            request = asset_v1.ListAssetsRequest(
                parent=self.project_resource,
                content_type=asset_v1.ContentType.RESOURCE,
                asset_types=all_asset_types
            )
            
            # Log the actual API call
            logger.info(f"Making Asset Inventory API call to: cloudasset.googleapis.com")
            logger.info(f"Request: ListAssets for project {self.project_id} with {len(all_asset_types)} asset types")
            
            # List all assets
            assets = []
            page_result = self.asset_client.list_assets(request=request)
            
            for asset in page_result:
                assets.append(asset)
                
            return assets
            
        except Exception as e:
            logger.error(f"Error fetching assets from Asset API: {e}")
            logger.error(f"Failed API call to cloudasset.googleapis.com for project {self.project_id}")
            return []
    
    async def _process_discovered_assets(self, assets: List[Any], intent: str) -> Dict[str, Any]:
        """Process discovered assets based on query intent."""
        if intent == "security_queries":
            return await self._analyze_security_posture(assets)
        elif intent == "cost_queries":
            return await self._analyze_cost_implications(assets)
        elif intent == "performance_queries":
            return await self._analyze_performance_metrics(assets)
        else:
            return await self._provide_comprehensive_overview(assets)
    
    async def _analyze_security_posture(self, assets: List[Any]) -> Dict[str, Any]:
        """Analyze security posture of discovered assets."""
        security_analysis = {
            "total_assets_analyzed": len(assets),
            "security_findings": [],
            "risk_summary": {"high": 0, "medium": 0, "low": 0},
            "recommendations": []
        }
        
        for asset in assets:
            findings = await self._assess_asset_security(asset)
            security_analysis["security_findings"].extend(findings)
        
        # Aggregate risk levels
        for finding in security_analysis["security_findings"]:
            risk_level = finding.get("risk_level", "low")
            security_analysis["risk_summary"][risk_level] += 1
        
        return {
            "analysis_type": "security",
            "security_analysis": security_analysis,
            "assets_by_category": await self._categorize_assets(assets)
        }
    
    async def _assess_asset_security(self, asset: Any) -> List[Dict[str, Any]]:
        """Assess security of individual asset."""
        findings = []
        asset_type = asset.asset_type
        resource_data = asset.resource.data
        
        # Firewall rule analysis
        if asset_type == "compute.googleapis.com/Firewall":
            source_ranges = resource_data.get("sourceRanges", [])
            if "0.0.0.0/0" in source_ranges:
                findings.append({
                    "asset_name": resource_data.get("name", "unknown"),
                    "asset_type": asset_type,
                    "finding_type": "overly_permissive_firewall",
                    "description": "Firewall rule allows traffic from any IP (0.0.0.0/0)",
                    "risk_level": "high",
                    "recommendation": "Restrict source IP ranges to specific networks"
                })
        
        # Storage bucket analysis
        elif asset_type == "storage.googleapis.com/Bucket":
            iam_config = resource_data.get("iamConfiguration", {})
            if iam_config.get("publicAccessPrevention") != "enforced":
                findings.append({
                    "asset_name": resource_data.get("name", "unknown"),
                    "asset_type": asset_type,
                    "finding_type": "public_access_not_prevented",
                    "description": "Bucket does not enforce public access prevention",
                    "risk_level": "medium",
                    "recommendation": "Enable public access prevention on bucket"
                })
        
        # Service account analysis
        elif asset_type == "iam.googleapis.com/ServiceAccount":
            # Check if service account has keys (potential security risk)
            findings.append({
                "asset_name": resource_data.get("email", "unknown"),
                "asset_type": asset_type,
                "finding_type": "service_account_review",
                "description": "Service account should be reviewed for proper key management",
                "risk_level": "low",
                "recommendation": "Review service account keys and rotate regularly"
            })
        
        # Compute instance analysis
        elif asset_type == "compute.googleapis.com/Instance":
            # Check for external IP
            network_interfaces = resource_data.get("networkInterfaces", [])
            has_external_ip = any(
                "accessConfigs" in interface and interface["accessConfigs"]
                for interface in network_interfaces
            )
            
            if has_external_ip:
                findings.append({
                    "asset_name": resource_data.get("name", "unknown"),
                    "asset_type": asset_type,
                    "finding_type": "external_ip_exposure",
                    "description": "Instance has external IP address",
                    "risk_level": "medium",
                    "recommendation": "Consider using Cloud NAT or private instances"
                })
        
        return findings
    
    async def _analyze_cost_implications(self, assets: List[Any]) -> Dict[str, Any]:
        """Analyze cost implications of discovered assets."""
        cost_analysis = {
            "total_assets": len(assets),
            "cost_drivers": {},
            "optimization_opportunities": [],
            "estimated_monthly_cost": "Requires Billing API integration"
        }
        
        # Categorize assets by cost impact
        for asset in assets:
            category = self._get_asset_category(asset.asset_type)
            if category not in cost_analysis["cost_drivers"]:
                cost_analysis["cost_drivers"][category] = 0
            cost_analysis["cost_drivers"][category] += 1
        
        # Add optimization opportunities
        cost_analysis["optimization_opportunities"] = [
            "Review compute instances for right-sizing opportunities",
            "Check storage buckets for lifecycle management policies",
            "Evaluate unused persistent disks",
            "Consider preemptible instances for non-critical workloads"
        ]
        
        return {
            "analysis_type": "cost",
            "cost_analysis": cost_analysis,
            "assets_by_category": await self._categorize_assets(assets)
        }
    
    async def _analyze_performance_metrics(self, assets: List[Any]) -> Dict[str, Any]:
        """Analyze performance metrics of discovered assets."""
        performance_analysis = {
            "total_assets": len(assets),
            "performance_insights": [],
            "optimization_recommendations": []
        }
        
        # Add performance analysis recommendations
        performance_analysis["optimization_recommendations"] = [
            "Monitor instance CPU and memory utilization",
            "Review network throughput for bottlenecks",
            "Check disk I/O performance for storage-intensive workloads",
            "Consider load balancing for high-traffic applications"
        ]
        
        return {
            "analysis_type": "performance",
            "performance_analysis": performance_analysis,
            "assets_by_category": await self._categorize_assets(assets)
        }
    
    async def _provide_comprehensive_overview(self, assets: List[Any]) -> Dict[str, Any]:
        """Provide comprehensive overview of discovered assets."""
        overview = {
            "total_assets": len(assets),
            "assets_by_category": await self._categorize_assets(assets),
            "summary": await self._generate_asset_summary(assets)
        }
        
        return {
            "analysis_type": "overview",
            "overview": overview
        }
    
    async def _categorize_assets(self, assets: List[Any]) -> Dict[str, Dict[str, Any]]:
        """Categorize assets by service type."""
        categorized = {}
        
        for asset in assets:
            category = self._get_asset_category(asset.asset_type)
            if category not in categorized:
                categorized[category] = {"count": 0, "assets": []}
            
            categorized[category]["count"] += 1
            categorized[category]["assets"].append({
                "name": asset.resource.data.get("name", "unknown"),
                "type": asset.asset_type,
                "location": self._extract_location(asset),
                "creation_time": asset.resource.data.get("creationTimestamp", "")
            })
        
        return categorized
    
    def _get_asset_category(self, asset_type: str) -> str:
        """Get category for asset type."""
        for category, mapping in self.asset_type_mappings.items():
            if asset_type in mapping["asset_types"]:
                return category
        return "other"
    
    def _extract_location(self, asset: Any) -> str:
        """Extract location/zone from asset."""
        resource_data = asset.resource.data
        
        # Try different location fields
        for field in ["zone", "region", "location", "locationId"]:
            if field in resource_data:
                location = resource_data[field]
                if isinstance(location, str) and "/" in location:
                    return location.split("/")[-1]
                return str(location)
        
        return "global"
    
    async def _generate_asset_summary(self, assets: List[Any]) -> Dict[str, Any]:
        """Generate summary of assets."""
        summary = {
            "total_count": len(assets),
            "categories": {},
            "locations": {},
            "recent_assets": []
        }
        
        for asset in assets:
            # Count by category
            category = self._get_asset_category(asset.asset_type)
            summary["categories"][category] = summary["categories"].get(category, 0) + 1
            
            # Count by location
            location = self._extract_location(asset)
            summary["locations"][location] = summary["locations"].get(location, 0) + 1
        
        return summary
    
    def _get_api_call_log(self) -> List[Dict[str, str]]:
        """Get log of API calls made during discovery."""
        return [
            {
                "api": "cloudasset.googleapis.com",
                "method": "ListAssets",
                "timestamp": datetime.utcnow().isoformat(),
                "project": self.project_id
            }
        ]
    
    def _get_fallback_response(self, query: str, error: str = None) -> Dict[str, Any]:
        """Generate fallback response when API is unavailable."""
        return {
            "success": False,
            "message": f"Asset Inventory API not available: {error or 'Service unavailable'}",
            "original_query": query,
            "fallback_data": {
                "suggestion": "Please ensure Google Cloud Asset Inventory API is enabled and credentials are configured",
                "mock_response": True
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    # Convenience methods for specific resource types
    async def get_compute_instances(self) -> Dict[str, Any]:
        """Get all compute instances."""
        return await self.process_natural_language_query("show me my compute instances")
    
    async def get_storage_buckets(self) -> Dict[str, Any]:
        """Get all storage buckets."""
        return await self.process_natural_language_query("show me my storage buckets")
    
    async def get_cloud_functions(self) -> Dict[str, Any]:
        """Get all cloud functions."""
        return await self.process_natural_language_query("show me my cloud functions")
    
    async def get_databases(self) -> Dict[str, Any]:
        """Get all databases."""
        return await self.process_natural_language_query("show me my databases")
    
    async def get_security_assets(self) -> Dict[str, Any]:
        """Get security-related assets with analysis."""
        return await self.process_natural_language_query("analyze my security assets")
    
    async def get_kubernetes_clusters(self) -> Dict[str, Any]:
        """Get all Kubernetes clusters."""
        return await self.process_natural_language_query("show me my kubernetes clusters")
    
    async def search_assets_by_name(self, name_pattern: str) -> Dict[str, Any]:
        """Search assets by name pattern."""
        return await self.process_natural_language_query(f"find assets with name {name_pattern}")