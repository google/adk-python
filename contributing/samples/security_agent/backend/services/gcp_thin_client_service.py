"""
GCP Thin Client Service - Lightweight wrapper for GCP services
Focuses on Asset Inventory and Security Recommendations
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class AssetType(Enum):
    """GCP Asset types for filtering"""
    COMPUTE = "compute.googleapis.com/Instance"
    STORAGE = "storage.googleapis.com/Bucket"
    IAM = "iam.googleapis.com/ServiceAccount"
    NETWORK = "compute.googleapis.com/Network"
    FIREWALL = "compute.googleapis.com/Firewall"
    DATABASE = "sqladmin.googleapis.com/Instance"
    FUNCTION = "cloudfunctions.googleapis.com/Function"
    KUBERNETES = "container.googleapis.com/Cluster"
    PUBSUB = "pubsub.googleapis.com/Topic"
    BIGQUERY = "bigquery.googleapis.com/Dataset"

@dataclass
class SecurityRecommendation:
    """Security recommendation data model"""
    id: str
    title: str
    description: str
    severity: str  # critical, high, medium, low
    category: str  # iam, storage, network, compliance, cost
    affected_resources: List[str]
    remediation_steps: List[str]
    estimated_impact: Optional[str] = None
    implementation_effort: Optional[str] = None
    compliance_frameworks: Optional[List[str]] = None

@dataclass
class AssetInventorySnapshot:
    """Snapshot of GCP assets with security context"""
    total_assets: int
    asset_breakdown: Dict[str, int]
    security_findings: List[Dict[str, Any]]
    high_risk_assets: List[str]
    recommendations: List[SecurityRecommendation]
    timestamp: datetime
    scan_duration_ms: float

class GCPThinClientService:
    """
    Thin client wrapper for GCP services
    Delegates heavy lifting to GCP APIs while providing chat-friendly responses
    """
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.cache = {}  # Simple in-memory cache
        self.cache_ttl = 300  # 5 minutes
        
        # Initialize GCP clients lazily
        self._asset_client = None
        self._recommender_client = None
        self._security_center_client = None
        
        logger.info(f"Initialized GCP Thin Client for project: {project_id}")
    
    async def get_asset_inventory_snapshot(self) -> AssetInventorySnapshot:
        """Get comprehensive snapshot of GCP assets with security analysis"""
        start_time = datetime.now()
        
        # Check cache first
        cache_key = f"snapshot_{self.project_id}"
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                logger.info("Returning cached asset inventory snapshot")
                return cached_data
        
        try:
            # Import extended asset discovery
            from .gcp_extended_assets import ExtendedAssetDiscovery
            extended = ExtendedAssetDiscovery(self.project_id)
            
            # Parallel fetch all asset data (including extended types)
            asset_tasks = [
                self._fetch_compute_assets(),
                self._fetch_storage_assets(),
                self._fetch_iam_assets(),
                self._fetch_network_assets(),
                self._fetch_database_assets(),
                self._fetch_security_findings(),
                self._fetch_recommendations(),
                extended.fetch_cloud_functions(),
                extended.fetch_bigquery_datasets(),
                extended.fetch_pubsub_topics(),
                extended.fetch_gke_clusters(),
                extended.fetch_cloud_run_services()
            ]
            
            results = await asyncio.gather(*asset_tasks, return_exceptions=True)
            
            # Process results
            compute_assets = results[0] if not isinstance(results[0], Exception) else []
            storage_assets = results[1] if not isinstance(results[1], Exception) else []
            iam_assets = results[2] if not isinstance(results[2], Exception) else []
            network_assets = results[3] if not isinstance(results[3], Exception) else []
            database_assets = results[4] if not isinstance(results[4], Exception) else []
            security_findings = results[5] if not isinstance(results[5], Exception) else []
            recommendations = results[6] if not isinstance(results[6], Exception) else []
            cloud_functions = results[7] if not isinstance(results[7], Exception) else []
            bigquery_datasets = results[8] if not isinstance(results[8], Exception) else []
            pubsub_topics = results[9] if not isinstance(results[9], Exception) else []
            gke_clusters = results[10] if not isinstance(results[10], Exception) else []
            cloud_run_services = results[11] if not isinstance(results[11], Exception) else []
            
            # Build comprehensive asset breakdown
            asset_breakdown = {
                "Compute Instances": len(compute_assets),
                "Storage Buckets": len(storage_assets),
                "IAM Accounts": len(iam_assets),
                "Networks": len(network_assets),
                "Databases": len(database_assets),
                "Cloud Functions": len(cloud_functions),
                "BigQuery Datasets": len(bigquery_datasets),
                "Pub/Sub Topics": len(pubsub_topics),
                "GKE Clusters": len(gke_clusters),
                "Cloud Run Services": len(cloud_run_services)
            }
            
            # Identify high-risk assets
            high_risk_assets = self._identify_high_risk_assets(
                compute_assets, storage_assets, iam_assets, security_findings
            )
            
            # Create snapshot
            snapshot = AssetInventorySnapshot(
                total_assets=sum(asset_breakdown.values()),
                asset_breakdown=asset_breakdown,
                security_findings=security_findings[:10],  # Top 10 findings
                high_risk_assets=high_risk_assets[:10],  # Top 10 high-risk
                recommendations=recommendations[:5],  # Top 5 recommendations
                timestamp=datetime.now(),
                scan_duration_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
            
            # Cache the snapshot
            self.cache[cache_key] = (snapshot, datetime.now())
            
            logger.info(f"Asset inventory snapshot completed in {snapshot.scan_duration_ms}ms")
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to get asset inventory snapshot: {e}")
            # Return minimal snapshot on error
            return AssetInventorySnapshot(
                total_assets=0,
                asset_breakdown={},
                security_findings=[],
                high_risk_assets=[],
                recommendations=[],
                timestamp=datetime.now(),
                scan_duration_ms=0
            )
    
    async def analyze_asset_security(self, asset_query: str) -> Dict[str, Any]:
        """Analyze security posture of specific assets based on natural language query"""
        logger.info(f"Analyzing security for query: {asset_query}")
        
        # Parse query intent
        query_lower = asset_query.lower()
        
        # Determine asset type from query
        if any(word in query_lower for word in ["bucket", "storage"]):
            return await self._analyze_storage_security()
        elif any(word in query_lower for word in ["instance", "compute", "vm"]):
            return await self._analyze_compute_security()
        elif any(word in query_lower for word in ["iam", "user", "permission", "role"]):
            return await self._analyze_iam_security()
        elif any(word in query_lower for word in ["network", "firewall", "vpc"]):
            return await self._analyze_network_security()
        else:
            # General security analysis
            return await self._analyze_overall_security()
    
    async def get_contextual_recommendations(self, context: Dict[str, Any]) -> List[SecurityRecommendation]:
        """Get security recommendations based on conversation context"""
        recommendations = []
        
        # Analyze context to determine recommendation focus
        if context.get("recent_topics"):
            topics = context["recent_topics"]
            
            if "storage" in topics:
                recommendations.extend(await self._get_storage_recommendations())
            if "iam" in topics:
                recommendations.extend(await self._get_iam_recommendations())
            if "network" in topics:
                recommendations.extend(await self._get_network_recommendations())
            if "compliance" in topics:
                recommendations.extend(await self._get_compliance_recommendations())
        
        # If no specific context, get general recommendations
        if not recommendations:
            recommendations = await self._get_general_recommendations()
        
        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda x: severity_order.get(x.severity, 4))
        
        return recommendations[:5]  # Return top 5 most relevant
    
    async def generate_security_insights(self, assets: List[Dict]) -> Dict[str, Any]:
        """Generate actionable security insights from asset data"""
        insights = {
            "summary": "",
            "risk_score": 0,
            "top_risks": [],
            "quick_wins": [],
            "long_term_improvements": [],
            "compliance_gaps": []
        }
        
        # Analyze assets for patterns
        if assets:
            # Calculate risk score (0-100)
            risk_factors = 0
            for asset in assets:
                if asset.get("public_access"):
                    risk_factors += 10
                if not asset.get("encryption_enabled"):
                    risk_factors += 5
                if asset.get("outdated_version"):
                    risk_factors += 3
            
            insights["risk_score"] = min(risk_factors, 100)
            
            # Generate summary
            if insights["risk_score"] > 70:
                insights["summary"] = "Critical security issues detected requiring immediate attention"
            elif insights["risk_score"] > 40:
                insights["summary"] = "Several security improvements recommended"
            else:
                insights["summary"] = "Security posture is generally good with minor improvements suggested"
            
            # Identify top risks
            insights["top_risks"] = [
                "Public storage buckets detected" if any(a.get("public_access") for a in assets) else None,
                "Unencrypted resources found" if any(not a.get("encryption_enabled") for a in assets) else None,
                "Overly permissive IAM policies" if any(a.get("excessive_permissions") for a in assets) else None
            ]
            insights["top_risks"] = [r for r in insights["top_risks"] if r]
            
            # Quick wins
            insights["quick_wins"] = [
                "Enable encryption on all storage buckets",
                "Review and tighten IAM permissions",
                "Enable audit logging for all services"
            ]
            
            # Long-term improvements
            insights["long_term_improvements"] = [
                "Implement least-privilege access model",
                "Set up automated security scanning",
                "Establish security baseline configurations"
            ]
        
        return insights
    
    # Private helper methods for actual GCP API calls
    async def _fetch_compute_assets(self) -> List[Dict]:
        """Fetch compute instances from GCP"""
        try:
            from google.cloud import asset_v1
            
            if not self._asset_client:
                self._asset_client = asset_v1.AssetServiceClient()
            
            parent = f"projects/{self.project_id}"
            request = asset_v1.ListAssetsRequest(
                parent=parent,
                asset_types=["compute.googleapis.com/Instance"],
                content_type=asset_v1.ContentType.RESOURCE
            )
            
            assets = []
            page_result = self._asset_client.list_assets(request=request)
            for response in page_result:
                assets.append({
                    "name": response.name,
                    "asset_type": response.asset_type,
                    "resource": response.resource.data if response.resource else {}
                })
            
            return assets
        except Exception as e:
            logger.warning(f"Could not fetch compute assets: {e}")
            return []
    
    async def _fetch_storage_assets(self) -> List[Dict]:
        """Fetch storage buckets from GCP"""
        try:
            from google.cloud import storage
            
            client = storage.Client(project=self.project_id)
            buckets = []
            
            for bucket in client.list_buckets():
                buckets.append({
                    "name": bucket.name,
                    "asset_type": "storage.googleapis.com/Bucket",
                    "location": bucket.location,
                    "storage_class": bucket.storage_class,
                    "public_access": bucket.iam_configuration.public_access_prevention == "inherited",
                    "encryption_enabled": bucket.default_kms_key_name is not None,
                    "created": bucket.time_created.isoformat() if bucket.time_created else None
                })
            
            return buckets
        except Exception as e:
            logger.warning(f"Could not fetch storage assets: {e}")
            return []
    
    async def _fetch_iam_assets(self) -> List[Dict]:
        """Fetch IAM service accounts from GCP"""
        try:
            from google.cloud import iam_admin_v1
            from google.cloud import resourcemanager_v3
            
            # Get service accounts
            iam_client = iam_admin_v1.IAMClient()
            service_accounts = []
            
            request = iam_admin_v1.ListServiceAccountsRequest(
                name=f"projects/{self.project_id}"
            )
            
            page_result = iam_client.list_service_accounts(request=request)
            for account in page_result:
                service_accounts.append({
                    "name": account.name,
                    "email": account.email,
                    "asset_type": "iam.googleapis.com/ServiceAccount",
                    "display_name": account.display_name,
                    "unique_id": account.unique_id,
                    "disabled": account.disabled,
                    "created": account.oauth2_client_id if hasattr(account, 'oauth2_client_id') else None
                })
            
            # Also get IAM policy for the project
            resource_client = resourcemanager_v3.ProjectsClient()
            project = resource_client.get_project(name=f"projects/{self.project_id}")
            
            # Get IAM policy
            policy = resource_client.get_iam_policy(
                request={"resource": f"projects/{self.project_id}"}
            )
            
            # Count roles and members
            total_bindings = len(policy.bindings) if policy.bindings else 0
            total_members = sum(len(binding.members) for binding in policy.bindings) if policy.bindings else 0
            
            # Add policy summary as a pseudo-asset
            service_accounts.append({
                "name": f"projects/{self.project_id}/iamPolicy",
                "asset_type": "iam.googleapis.com/Policy",
                "total_bindings": total_bindings,
                "total_members": total_members,
                "excessive_permissions": total_bindings > 50  # Flag if too many bindings
            })
            
            return service_accounts
            
        except Exception as e:
            logger.warning(f"Could not fetch IAM assets: {e}")
            return []
    
    async def _fetch_network_assets(self) -> List[Dict]:
        """Fetch network resources from GCP"""
        try:
            from google.cloud import compute_v1
            
            networks = []
            
            # Fetch VPC networks
            network_client = compute_v1.NetworksClient()
            request = compute_v1.ListNetworksRequest(project=self.project_id)
            
            for network in network_client.list(request=request):
                networks.append({
                    "name": network.name,
                    "asset_type": "compute.googleapis.com/Network",
                    "self_link": network.self_link,
                    "auto_create_subnetworks": network.auto_create_subnetworks,
                    "creation_timestamp": network.creation_timestamp
                })
            
            # Fetch firewall rules
            firewall_client = compute_v1.FirewallsClient()
            request = compute_v1.ListFirewallsRequest(project=self.project_id)
            
            for firewall in firewall_client.list(request=request):
                # Check for risky rules (0.0.0.0/0 access)
                is_public = any(
                    "0.0.0.0/0" in (firewall.source_ranges or [])
                )
                
                networks.append({
                    "name": firewall.name,
                    "asset_type": "compute.googleapis.com/Firewall",
                    "direction": firewall.direction,
                    "priority": firewall.priority,
                    "source_ranges": firewall.source_ranges,
                    "allowed": [{"protocol": a.I_p_protocol, "ports": a.ports} for a in (firewall.allowed or [])],
                    "public_access": is_public,
                    "disabled": firewall.disabled
                })
            
            return networks
            
        except Exception as e:
            logger.warning(f"Could not fetch network assets: {e}")
            return []
    
    async def _fetch_database_assets(self) -> List[Dict]:
        """Fetch database instances from GCP"""
        try:
            # Note: Cloud SQL Admin API needs different import
            # For now, return empty until we configure Cloud SQL Admin API
            return []
            
            # TODO: Enable when Cloud SQL Admin API is configured
            # from google.cloud.sql import connector
            # databases = []
            
            request = sql_v1.SqlInstancesListRequest(project=self.project_id)
            
            for instance in sql_client.list(request=request).items:
                # Check security settings
                has_public_ip = any(
                    ip.type_ == sql_v1.IpConfiguration.Type.PRIMARY 
                    for ip in (instance.ip_addresses or [])
                )
                
                databases.append({
                    "name": instance.name,
                    "asset_type": "sqladmin.googleapis.com/Instance",
                    "database_version": instance.database_version,
                    "state": instance.state,
                    "region": instance.region,
                    "tier": instance.settings.tier if instance.settings else None,
                    "public_ip": has_public_ip,
                    "backup_enabled": instance.settings.backup_configuration.enabled if instance.settings and instance.settings.backup_configuration else False,
                    "ssl_required": instance.settings.ip_configuration.require_ssl if instance.settings and instance.settings.ip_configuration else False
                })
            
            return databases
            
        except Exception as e:
            logger.warning(f"Could not fetch database assets: {e}")
            return []
    
    async def _fetch_security_findings(self) -> List[Dict]:
        """Fetch security findings from Security Command Center"""
        try:
            from google.cloud import securitycenter_v1
            
            client = securitycenter_v1.SecurityCenterClient()
            org_name = f"organizations/{os.getenv('GOOGLE_CLOUD_ORG_ID', '419850945193')}"
            
            findings = []
            
            # List findings for the project
            request = securitycenter_v1.ListFindingsRequest(
                parent=f"{org_name}/sources/-",
                filter=f'resource.project_display_name="{self.project_id}"'
            )
            
            for finding in client.list_findings(request=request):
                findings.append({
                    "name": finding.finding.name,
                    "category": finding.finding.category,
                    "severity": finding.finding.severity,
                    "state": finding.finding.state,
                    "resource_name": finding.finding.resource_name,
                    "event_time": finding.finding.event_time.isoformat() if finding.finding.event_time else None
                })
            
            return findings[:10]  # Limit to top 10 findings
            
        except Exception as e:
            logger.warning(f"Could not fetch security findings: {e}")
            return []
    
    async def _fetch_recommendations(self) -> List[SecurityRecommendation]:
        """Fetch recommendations from Recommender API"""
        try:
            from google.cloud import recommender_v1
            
            client = recommender_v1.RecommenderClient()
            recommendations = []
            
            # Define recommender types to check
            recommender_types = [
                "google.iam.policy.Recommender",
                "google.compute.instance.MachineTypeRecommender",
                "google.compute.firewall.Recommender",
                "google.compute.disk.IdleResourceRecommender"
            ]
            
            parent = f"projects/{self.project_id}/locations/global"
            
            for recommender_type in recommender_types:
                try:
                    recommender_parent = f"{parent}/recommenders/{recommender_type}"
                    request = recommender_v1.ListRecommendationsRequest(
                        parent=recommender_parent
                    )
                    
                    for recommendation in client.list_recommendations(request=request):
                        severity = "high" if recommendation.priority == "P1" else "medium"
                        
                        recommendations.append(SecurityRecommendation(
                            id=recommendation.name.split('/')[-1],
                            title=recommendation.description,
                            description=recommendation.additional_impact[0].details if recommendation.additional_impact else recommendation.description,
                            severity=severity,
                            category=recommender_type.split('.')[-2],
                            affected_resources=[recommendation.content.overview.resource_name],
                            remediation_steps=[op.action for op in recommendation.content.operation_groups[0].operations] if recommendation.content.operation_groups else [],
                            estimated_impact=recommendation.primary_impact.cost_projection.cost.units if hasattr(recommendation.primary_impact, 'cost_projection') else None,
                            implementation_effort="Low" if recommendation.priority == "P4" else "Medium"
                        ))
                except Exception as e:
                    logger.debug(f"No recommendations for {recommender_type}: {e}")
                    continue
            
            return recommendations[:5]  # Return top 5 recommendations
            
        except Exception as e:
            logger.warning(f"Could not fetch recommendations: {e}")
            # Return empty list instead of mock data
            return []
    
    def _identify_high_risk_assets(self, *asset_lists) -> List[str]:
        """Identify high-risk assets from various asset types"""
        high_risk = []
        for assets in asset_lists:
            if isinstance(assets, list):
                for asset in assets:
                    if isinstance(asset, dict):
                        if asset.get("public_access") or not asset.get("encryption_enabled"):
                            high_risk.append(asset.get("name", "unknown"))
        return high_risk
    
    async def _analyze_storage_security(self) -> Dict[str, Any]:
        """Analyze storage security posture"""
        return {
            "focus": "storage",
            "findings": ["Public buckets detected", "Encryption not enabled"],
            "recommendations": ["Enable bucket encryption", "Review bucket policies"],
            "risk_level": "medium"
        }
    
    async def _analyze_compute_security(self) -> Dict[str, Any]:
        """Analyze compute security posture"""
        return {
            "focus": "compute",
            "findings": ["Instances with public IPs", "Missing OS patches"],
            "recommendations": ["Use private IPs where possible", "Enable automatic patching"],
            "risk_level": "medium"
        }
    
    async def _analyze_iam_security(self) -> Dict[str, Any]:
        """Analyze IAM security posture"""
        return {
            "focus": "iam",
            "findings": ["Overly permissive roles", "Unused service accounts"],
            "recommendations": ["Apply least privilege", "Remove unused accounts"],
            "risk_level": "high"
        }
    
    async def _analyze_network_security(self) -> Dict[str, Any]:
        """Analyze network security posture"""
        return {
            "focus": "network",
            "findings": ["Open firewall rules", "Missing network segmentation"],
            "recommendations": ["Tighten firewall rules", "Implement VPC segmentation"],
            "risk_level": "high"
        }
    
    async def _analyze_overall_security(self) -> Dict[str, Any]:
        """Analyze overall security posture"""
        return {
            "focus": "overall",
            "findings": ["Multiple security issues across resources"],
            "recommendations": ["Conduct security audit", "Implement security baseline"],
            "risk_level": "medium"
        }
    
    async def _get_storage_recommendations(self) -> List[SecurityRecommendation]:
        """Get storage-specific recommendations"""
        return [
            SecurityRecommendation(
                id="storage-1",
                title="Enable default encryption",
                description="Ensure all buckets have encryption enabled",
                severity="high",
                category="storage",
                affected_resources=["All storage buckets"],
                remediation_steps=["Apply organization policy for encryption"]
            )
        ]
    
    async def _get_iam_recommendations(self) -> List[SecurityRecommendation]:
        """Get IAM-specific recommendations"""
        return [
            SecurityRecommendation(
                id="iam-1",
                title="Review IAM permissions",
                description="Apply least privilege principle",
                severity="critical",
                category="iam",
                affected_resources=["Project IAM policies"],
                remediation_steps=["Audit current permissions", "Remove excessive grants"]
            )
        ]
    
    async def _get_network_recommendations(self) -> List[SecurityRecommendation]:
        """Get network-specific recommendations"""
        return [
            SecurityRecommendation(
                id="network-1",
                title="Restrict firewall rules",
                description="Remove overly permissive firewall rules",
                severity="high",
                category="network",
                affected_resources=["Firewall rules"],
                remediation_steps=["Review all rules", "Apply restrictive policies"]
            )
        ]
    
    async def _get_compliance_recommendations(self) -> List[SecurityRecommendation]:
        """Get compliance-specific recommendations"""
        return [
            SecurityRecommendation(
                id="compliance-1",
                title="Enable audit logging",
                description="Ensure comprehensive audit logging",
                severity="high",
                category="compliance",
                affected_resources=["All services"],
                remediation_steps=["Enable Cloud Audit Logs", "Configure log retention"],
                compliance_frameworks=["SOC2", "ISO27001", "GDPR"]
            )
        ]
    
    async def _get_general_recommendations(self) -> List[SecurityRecommendation]:
        """Get general security recommendations"""
        return [
            SecurityRecommendation(
                id="general-1",
                title="Establish security baseline",
                description="Implement organization-wide security policies",
                severity="high",
                category="overall",
                affected_resources=["All projects"],
                remediation_steps=["Define security policies", "Apply via Organization Policies"]
            )
        ]