"""
GCP Resource Management Client
==============================

Client for integrating with GCP Resource Manager and other resource-related APIs
for advanced asset discovery, policy management, and resource optimization.
"""

import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import os

try:
    from google.cloud import resourcemanager_v3
    from google.cloud import asset_v1
    from google.cloud import recommender_v1
    from google.cloud.resourcemanager_v3 import types as rm_types
    from google.cloud.asset_v1 import types as asset_types
    from google.cloud.recommender_v1 import types as rec_types
    GCLOUD_AVAILABLE = True
except ImportError:
    GCLOUD_AVAILABLE = False
    # Create mock types for when library is not available
    class MockRMTypes:
        class GetProjectRequest:
            def __init__(self, **kwargs):
                pass
        class GetIamPolicyRequest:
            def __init__(self, **kwargs):
                pass
    
    class MockAssetTypes:
        class SearchAllResourcesRequest:
            def __init__(self, **kwargs):
                pass
        class ListAssetsRequest:
            def __init__(self, **kwargs):
                pass
        class ContentType:
            RESOURCE = "RESOURCE"
    
    class MockRecTypes:
        class ListRecommendationsRequest:
            def __init__(self, **kwargs):
                pass
    
    rm_types = MockRMTypes() if not GCLOUD_AVAILABLE else rm_types
    asset_types = MockAssetTypes() if not GCLOUD_AVAILABLE else asset_types
    rec_types = MockRecTypes() if not GCLOUD_AVAILABLE else rec_types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GCPResourceClient:
    """GCP Resource Management client for advanced resource operations"""
    
    def __init__(self, project_id: str, organization_id: Optional[str] = None):
        """
        Initialize GCP Resource client
        
        Args:
            project_id: GCP project ID
            organization_id: GCP organization ID
        """
        self.project_id = project_id
        self.organization_id = organization_id
        
        if not GCLOUD_AVAILABLE:
            logger.warning("Google Cloud Resource Management libraries not available")
            self.resource_client = None
            self.asset_client = None
            self.recommender_client = None
            return
        
        try:
            # Initialize resource management clients
            self.resource_client = resourcemanager_v3.ProjectsClient()
            self.asset_client = asset_v1.AssetServiceClient()
            self.recommender_client = recommender_v1.RecommenderClient()
            
            # Set up resource names
            self.project_name = f"projects/{project_id}"
            self.org_name = f"organizations/{organization_id}" if organization_id else None
            
            logger.info(f"GCP Resource client initialized for project: {project_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize GCP Resource client: {e}")
            self.resource_client = None
            self.asset_client = None
            self.recommender_client = None
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to GCP Resource Management APIs"""
        if not self.resource_client:
            return {
                "connected": False,
                "error": "Google Cloud Resource Management libraries not available",
                "message": "Install required packages: google-cloud-resource-manager, google-cloud-asset, google-cloud-recommender"
            }
        
        try:
            # Test by getting project info
            request = rm_types.GetProjectRequest(name=self.project_name)
            project = self.resource_client.get_project(request=request)
            
            return {
                "connected": True,
                "project_id": self.project_id,
                "project_name": project.display_name,
                "project_state": project.state.name if project.state else None,
                "organization_id": self.organization_id,
                "message": "Connection successful"
            }
            
        except Exception as e:
            logger.error(f"GCP Resource connection test failed: {e}")
            return {
                "connected": False,
                "error": str(e),
                "message": "Connection test failed"
            }
    
    async def get_project_hierarchy(self) -> Dict[str, Any]:
        """Get project hierarchy information"""
        if not self.resource_client:
            return {
                "success": False,
                "error": "GCP Resource client not available"
            }
        
        try:
            # Get project details
            request = rm_types.GetProjectRequest(name=self.project_name)
            project = self.resource_client.get_project(request=request)
            
            hierarchy = {
                "project": {
                    "name": project.name,
                    "project_id": project.project_id,
                    "display_name": project.display_name,
                    "state": project.state.name if project.state else None,
                    "create_time": project.create_time.isoformat() if project.create_time else None,
                    "update_time": project.update_time.isoformat() if project.update_time else None,
                    "delete_time": project.delete_time.isoformat() if project.delete_time else None,
                    "etag": project.etag,
                    "parent": project.parent
                }
            }
            
            # Get parent information if available
            if project.parent:
                try:
                    if project.parent.startswith("folders/"):
                        # Handle folder parent
                        hierarchy["parent_type"] = "folder"
                        hierarchy["parent_id"] = project.parent.split("/")[1]
                    elif project.parent.startswith("organizations/"):
                        # Handle organization parent
                        hierarchy["parent_type"] = "organization"
                        hierarchy["parent_id"] = project.parent.split("/")[1]
                except Exception as e:
                    logger.warning(f"Could not get parent details: {e}")
                    hierarchy["parent_error"] = str(e)
            
            # Get project IAM policy if possible
            try:
                iam_policy = self.resource_client.get_iam_policy(
                    request=rm_types.GetIamPolicyRequest(resource=self.project_name)
                )
                
                bindings = []
                for binding in iam_policy.bindings:
                    bindings.append({
                        "role": binding.role,
                        "members": list(binding.members),
                        "condition": {
                            "title": binding.condition.title,
                            "description": binding.condition.description,
                            "expression": binding.condition.expression
                        } if binding.condition else None
                    })
                
                hierarchy["iam_policy"] = {
                    "version": iam_policy.version,
                    "bindings_count": len(bindings),
                    "bindings": bindings[:10]  # Limit to first 10 for readability
                }
                
            except Exception as e:
                logger.warning(f"Could not get IAM policy: {e}")
                hierarchy["iam_policy"] = {"error": str(e)}
            
            return {
                "success": True,
                "hierarchy": hierarchy
            }
            
        except Exception as e:
            logger.error(f"Get project hierarchy failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def search_assets(self, asset_types: Optional[List[str]] = None, 
                          query: Optional[str] = None) -> Dict[str, Any]:
        """Search for assets using Cloud Asset Inventory"""
        if not self.asset_client:
            return {
                "success": False,
                "error": "GCP Asset client not available"
            }
        
        try:
            # Use organization scope if available, otherwise project scope
            scope = self.org_name or self.project_name
            
            request = asset_types.SearchAllResourcesRequest(
                scope=scope,
                query=query or "",
                asset_types=asset_types or [],
                page_size=100
            )
            
            response = self.asset_client.search_all_resources(request=request)
            
            resources = []
            for resource in response:
                resource_data = {
                    "name": resource.name,
                    "asset_type": resource.asset_type,
                    "project": resource.project,
                    "display_name": resource.display_name,
                    "description": resource.description,
                    "location": resource.location,
                    "labels": dict(resource.labels) if resource.labels else {},
                    "network_tags": list(resource.network_tags) if resource.network_tags else [],
                    "kms_key": resource.kms_key if resource.kms_key else None,
                    "create_time": resource.create_time.isoformat() if resource.create_time else None,
                    "update_time": resource.update_time.isoformat() if resource.update_time else None,
                    "state": resource.state,
                    "parent_full_resource_name": resource.parent_full_resource_name,
                    "parent_asset_type": resource.parent_asset_type
                }
                
                # Additional data extraction based on asset type
                if hasattr(resource, 'additional_attributes') and resource.additional_attributes:
                    resource_data["additional_attributes"] = dict(resource.additional_attributes)
                
                resources.append(resource_data)
            
            # Group by asset type for summary
            by_type = {}
            for resource in resources:
                asset_type = resource["asset_type"]
                by_type[asset_type] = by_type.get(asset_type, 0) + 1
            
            return {
                "success": True,
                "scope": scope,
                "total_resources": len(resources),
                "query": query,
                "asset_types_requested": asset_types,
                "resources_by_type": by_type,
                "resources": resources
            }
            
        except Exception as e:
            logger.error(f"Asset search failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_asset_inventory(self, asset_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get comprehensive asset inventory"""
        if not self.asset_client:
            return {
                "success": False,
                "error": "GCP Asset client not available"
            }
        
        try:
            # Use project scope for inventory
            parent = self.project_name
            
            # Build content type for detailed information
            content_type = asset_types.ContentType.RESOURCE
            
            request = asset_types.ListAssetsRequest(
                parent=parent,
                content_type=content_type,
                asset_types=asset_types or [],
                page_size=100
            )
            
            response = self.asset_client.list_assets(request=request)
            
            assets = []
            resource_summary = {}
            
            for asset in response:
                asset_data = {
                    "name": asset.name,
                    "asset_type": asset.asset_type,
                    "update_time": asset.update_time.isoformat() if asset.update_time else None,
                    "ancestors": list(asset.ancestors) if asset.ancestors else []
                }
                
                # Add resource data if available
                if asset.resource:
                    resource_data = {
                        "version": asset.resource.version,
                        "discovery_document_uri": asset.resource.discovery_document_uri,
                        "discovery_name": asset.resource.discovery_name,
                        "resource_url": asset.resource.resource_url,
                        "parent": asset.resource.parent,
                        "location": asset.resource.location
                    }
                    
                    # Extract resource data (simplified JSON representation)
                    if asset.resource.data:
                        try:
                            # Convert protobuf to dict for easier handling
                            import json
                            from google.protobuf.json_format import MessageToDict
                            resource_data["data"] = MessageToDict(asset.resource.data)
                        except Exception as e:
                            logger.debug(f"Could not convert resource data: {e}")
                            resource_data["data"] = "Available but not parsed"
                    
                    asset_data["resource"] = resource_data
                
                # Add IAM policy if available
                if asset.iam_policy:
                    policy_data = {
                        "version": asset.iam_policy.version,
                        "etag": asset.iam_policy.etag.decode() if asset.iam_policy.etag else None,
                        "bindings_count": len(asset.iam_policy.bindings)
                    }
                    
                    # Include first few bindings for overview
                    bindings = []
                    for binding in asset.iam_policy.bindings[:5]:  # First 5 bindings
                        bindings.append({
                            "role": binding.role,
                            "members_count": len(binding.members),
                            "has_condition": bool(binding.condition)
                        })
                    
                    policy_data["sample_bindings"] = bindings
                    asset_data["iam_policy"] = policy_data
                
                assets.append(asset_data)
                
                # Update summary
                asset_type = asset.asset_type
                resource_summary[asset_type] = resource_summary.get(asset_type, 0) + 1
            
            return {
                "success": True,
                "parent": parent,
                "total_assets": len(assets),
                "asset_types_found": len(resource_summary),
                "resource_summary": resource_summary,
                "assets": assets,
                "inventory_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Asset inventory failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_recommendations(self, recommender_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get recommendations for resource optimization"""
        if not self.recommender_client:
            return {
                "success": False,
                "error": "GCP Recommender client not available"
            }
        
        try:
            # Common recommender types
            default_recommenders = [
                "google.compute.instance.MachineTypeRecommender",
                "google.compute.disk.IdleResourceRecommender",
                "google.compute.address.IdleResourceRecommender",
                "google.iam.policy.Recommender",
                "google.cloudsql.instance.PerformanceRecommender",
                "google.gke.cluster.NodePoolRecommender"
            ]
            
            target_recommenders = recommender_types or default_recommenders
            all_recommendations = []
            recommender_summary = {}
            
            # Get recommendations for each recommender type
            for recommender_type in target_recommenders:
                try:
                    parent = f"projects/{self.project_id}/locations/global/recommenders/{recommender_type}"
                    
                    request = rec_types.ListRecommendationsRequest(
                        parent=parent,
                        page_size=50
                    )
                    
                    response = self.recommender_client.list_recommendations(request=request)
                    
                    recommendations = []
                    for recommendation in response:
                        rec_data = {
                            "name": recommendation.name,
                            "description": recommendation.description,
                            "recommender_subtype": recommendation.recommender_subtype,
                            "last_refresh_time": recommendation.last_refresh_time.isoformat() if recommendation.last_refresh_time else None,
                            "priority": recommendation.priority.name if recommendation.priority else None,
                            "state": recommendation.state.state.name if recommendation.state else None
                        }
                        
                        # Add primary impact information
                        if recommendation.primary_impact:
                            impact = recommendation.primary_impact
                            rec_data["primary_impact"] = {
                                "category": impact.category.name if impact.category else None,
                                "cost_projection": {
                                    "cost_units": impact.cost_projection.cost.units if impact.cost_projection and impact.cost_projection.cost else None,
                                    "currency_code": impact.cost_projection.cost.currency_code if impact.cost_projection and impact.cost_projection.cost else None
                                } if impact.cost_projection else None
                            }
                        
                        # Add associated insights count
                        rec_data["associated_insights_count"] = len(recommendation.associated_insights)
                        
                        recommendations.append(rec_data)
                        all_recommendations.append(rec_data)
                    
                    recommender_summary[recommender_type] = {
                        "total_recommendations": len(recommendations),
                        "recommendations": recommendations[:5]  # Top 5 for summary
                    }
                    
                except Exception as e:
                    logger.debug(f"Could not get recommendations for {recommender_type}: {e}")
                    recommender_summary[recommender_type] = {
                        "error": str(e),
                        "total_recommendations": 0
                    }
            
            # Analyze recommendations
            high_priority = len([r for r in all_recommendations if r.get("priority") == "HIGH"])
            medium_priority = len([r for r in all_recommendations if r.get("priority") == "MEDIUM"])
            cost_saving_recs = len([r for r in all_recommendations 
                                  if r.get("primary_impact", {}).get("category") == "COST"])
            
            return {
                "success": True,
                "project_id": self.project_id,
                "recommender_types_checked": len(target_recommenders),
                "total_recommendations": len(all_recommendations),
                "high_priority_count": high_priority,
                "medium_priority_count": medium_priority,
                "cost_saving_recommendations": cost_saving_recs,
                "recommendations_by_type": recommender_summary,
                "top_recommendations": sorted(
                    all_recommendations,
                    key=lambda x: 1 if x.get("priority") == "HIGH" else 2 if x.get("priority") == "MEDIUM" else 3
                )[:10],
                "analysis_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Get recommendations failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def analyze_resource_utilization(self) -> Dict[str, Any]:
        """Analyze resource utilization patterns"""
        try:
            # Get asset inventory for analysis
            inventory_result = await self.get_asset_inventory()
            if not inventory_result["success"]:
                return inventory_result
            
            # Get recommendations for optimization insights
            recommendations_result = await self.get_recommendations()
            if not recommendations_result["success"]:
                recommendations_result = {"recommendations_by_type": {}}
            
            assets = inventory_result["assets"]
            resource_summary = inventory_result["resource_summary"]
            
            # Analyze resource distribution
            compute_resources = 0
            storage_resources = 0
            network_resources = 0
            database_resources = 0
            other_resources = 0
            
            for asset_type, count in resource_summary.items():
                if "compute" in asset_type.lower() or "instance" in asset_type.lower():
                    compute_resources += count
                elif "storage" in asset_type.lower() or "disk" in asset_type.lower() or "bucket" in asset_type.lower():
                    storage_resources += count
                elif "network" in asset_type.lower() or "firewall" in asset_type.lower() or "subnet" in asset_type.lower():
                    network_resources += count
                elif "sql" in asset_type.lower() or "database" in asset_type.lower() or "redis" in asset_type.lower():
                    database_resources += count
                else:
                    other_resources += count
            
            # Identify potential issues from recommendations
            utilization_issues = []
            if recommendations_result.get("recommendations_by_type"):
                for recommender, data in recommendations_result["recommendations_by_type"].items():
                    if data.get("total_recommendations", 0) > 0:
                        if "IdleResource" in recommender:
                            utilization_issues.append(f"Idle resources detected in {recommender}")
                        elif "MachineType" in recommender:
                            utilization_issues.append(f"Machine type optimization opportunities in {recommender}")
                        elif "Performance" in recommender:
                            utilization_issues.append(f"Performance optimization needed in {recommender}")
            
            # Calculate utilization score (simplified)
            total_resources = sum(resource_summary.values())
            total_recommendations = recommendations_result.get("total_recommendations", 0)
            
            # Higher recommendations relative to resources suggests lower utilization
            utilization_score = max(0, 100 - (total_recommendations / total_resources * 100)) if total_resources > 0 else 100
            
            return {
                "success": True,
                "project_id": self.project_id,
                "total_resources": total_resources,
                "resource_distribution": {
                    "compute": compute_resources,
                    "storage": storage_resources,
                    "network": network_resources,
                    "database": database_resources,
                    "other": other_resources
                },
                "utilization_score": round(utilization_score, 2),
                "optimization_opportunities": total_recommendations,
                "high_priority_optimizations": recommendations_result.get("high_priority_count", 0),
                "cost_saving_opportunities": recommendations_result.get("cost_saving_recommendations", 0),
                "utilization_issues": utilization_issues,
                "resource_efficiency": "HIGH" if utilization_score > 80 else "MEDIUM" if utilization_score > 60 else "LOW",
                "recommendations": self._generate_utilization_recommendations(
                    utilization_score, total_recommendations, utilization_issues
                ),
                "analysis_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Resource utilization analysis failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_resource_tags_analysis(self) -> Dict[str, Any]:
        """Analyze resource tagging and labeling patterns"""
        try:
            # Search for resources to analyze their labels
            search_result = await self.search_assets()
            if not search_result["success"]:
                return search_result
            
            resources = search_result["resources"]
            
            # Analyze labeling patterns
            labeled_resources = 0
            unlabeled_resources = 0
            all_labels = {}
            label_coverage = {}
            
            for resource in resources:
                labels = resource.get("labels", {})
                
                if labels:
                    labeled_resources += 1
                    for key, value in labels.items():
                        if key not in all_labels:
                            all_labels[key] = {}
                        if value not in all_labels[key]:
                            all_labels[key][value] = 0
                        all_labels[key][value] += 1
                        
                        # Track label coverage by asset type
                        asset_type = resource["asset_type"]
                        if key not in label_coverage:
                            label_coverage[key] = {}
                        if asset_type not in label_coverage[key]:
                            label_coverage[key][asset_type] = 0
                        label_coverage[key][asset_type] += 1
                else:
                    unlabeled_resources += 1
            
            total_resources = len(resources)
            labeling_percentage = (labeled_resources / total_resources * 100) if total_resources > 0 else 0
            
            # Identify common label patterns
            common_labels = sorted(all_labels.items(), key=lambda x: len(x[1]), reverse=True)[:10]
            
            # Generate labeling recommendations
            labeling_recommendations = []
            if labeling_percentage < 50:
                labeling_recommendations.append("Low labeling coverage detected. Implement consistent tagging strategy.")
            
            if unlabeled_resources > 10:
                labeling_recommendations.append(f"{unlabeled_resources} resources without labels. Consider bulk labeling.")
            
            # Check for standard labels
            standard_labels = ["environment", "owner", "project", "team", "cost-center", "application"]
            missing_standard_labels = []
            for std_label in standard_labels:
                if std_label not in all_labels:
                    missing_standard_labels.append(std_label)
            
            if missing_standard_labels:
                labeling_recommendations.append(f"Consider adding standard labels: {', '.join(missing_standard_labels)}")
            
            return {
                "success": True,
                "project_id": self.project_id,
                "total_resources_analyzed": total_resources,
                "labeled_resources": labeled_resources,
                "unlabeled_resources": unlabeled_resources,
                "labeling_percentage": round(labeling_percentage, 2),
                "unique_label_keys": len(all_labels),
                "common_labels": common_labels,
                "label_coverage_by_type": label_coverage,
                "labeling_quality": "GOOD" if labeling_percentage > 80 else "FAIR" if labeling_percentage > 50 else "POOR",
                "recommendations": labeling_recommendations,
                "analysis_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Resource tags analysis failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_utilization_recommendations(self, score: float, total_recs: int, 
                                           issues: List[str]) -> List[str]:
        """Generate utilization improvement recommendations"""
        recommendations = []
        
        if score < 60:
            recommendations.append("Low resource utilization detected. Review and optimize resource allocation.")
        
        if total_recs > 20:
            recommendations.append("High number of optimization recommendations. Prioritize high-impact changes.")
        
        if any("Idle" in issue for issue in issues):
            recommendations.append("Idle resources found. Consider downsizing or deleting unused resources.")
        
        if any("MachineType" in issue for issue in issues):
            recommendations.append("Machine type optimization available. Right-size instances for workload requirements.")
        
        if any("Performance" in issue for issue in issues):
            recommendations.append("Performance optimization opportunities identified. Review and implement suggested changes.")
        
        recommendations.append("Enable detailed monitoring and alerting for resource utilization metrics.")
        recommendations.append("Implement automated scaling policies where appropriate.")
        
        return recommendations
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get GCP Resource Management statistics"""
        try:
            # Get various resource metrics
            hierarchy_result = await self.get_project_hierarchy()
            inventory_result = await self.get_asset_inventory()
            recommendations_result = await self.get_recommendations()
            utilization_result = await self.analyze_resource_utilization()
            
            statistics = {
                "success": True,
                "project_id": self.project_id,
                "organization_id": self.organization_id,
                "total_assets": inventory_result.get("total_assets", 0) if inventory_result["success"] else 0,
                "asset_types": inventory_result.get("asset_types_found", 0) if inventory_result["success"] else 0,
                "total_recommendations": recommendations_result.get("total_recommendations", 0) if recommendations_result["success"] else 0,
                "high_priority_optimizations": recommendations_result.get("high_priority_count", 0) if recommendations_result["success"] else 0,
                "utilization_score": utilization_result.get("utilization_score", 0) if utilization_result["success"] else 0,
                "resource_efficiency": utilization_result.get("resource_efficiency", "UNKNOWN") if utilization_result["success"] else "UNKNOWN",
                "project_state": hierarchy_result.get("hierarchy", {}).get("project", {}).get("state") if hierarchy_result["success"] else "UNKNOWN",
                "analysis_time": datetime.now().isoformat()
            }
            
            return statistics
            
        except Exception as e:
            logger.error(f"GCP Resource statistics failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Example usage and testing
async def test_gcp_resource_client():
    """Test GCP Resource client functionality"""
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "test-project")
    org_id = os.getenv("GOOGLE_CLOUD_ORGANIZATION", None)
    
    client = GCPResourceClient(
        project_id=project_id,
        organization_id=org_id
    )
    
    # Test connection
    connection = await client.test_connection()
    print(f"Connection test: {connection}")
    
    if connection["connected"]:
        # Get project hierarchy
        hierarchy = await client.get_project_hierarchy()
        print(f"Project hierarchy: {hierarchy}")
        
        # Search assets
        assets = await client.search_assets(query="state:ACTIVE")
        print(f"Asset search: {assets}")
        
        # Get recommendations
        recommendations = await client.get_recommendations()
        print(f"Recommendations: {recommendations}")
        
        # Analyze resource utilization
        utilization = await client.analyze_resource_utilization()
        print(f"Resource utilization: {utilization}")
        
        # Get statistics
        stats = await client.get_statistics()
        print(f"Resource statistics: {stats}")


if __name__ == "__main__":
    asyncio.run(test_gcp_resource_client())