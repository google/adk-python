"""
VPC Service Controls API Client
===============================

Client for integrating with VPC Service Controls for perimeter management,
access policy configuration, and dry-run testing in Phase 2 features.
"""

import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import os

try:
    from google.cloud import accesscontextmanager_v1
    from google.cloud.accesscontextmanager_v1 import types
    from google.auth import default
    from google.api_core import retry
    GCLOUD_AVAILABLE = True
except ImportError:
    GCLOUD_AVAILABLE = False
    # Create mock types for when library is not available
    class MockTypes:
        class ListAccessPoliciesRequest:
            def __init__(self, **kwargs):
                pass
        class ListServicePerimetersRequest:
            def __init__(self, **kwargs):
                pass
        class GetServicePerimeterRequest:
            def __init__(self, **kwargs):
                pass
        class ListAccessLevelsRequest:
            def __init__(self, **kwargs):
                pass
        class SearchAllResourcesRequest:
            def __init__(self, **kwargs):
                pass
    types = MockTypes() if not GCLOUD_AVAILABLE else types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VPCServiceControlsClient:
    """VPC Service Controls client for perimeter and policy management"""
    
    def __init__(self, organization_id: str, project_id: Optional[str] = None):
        """
        Initialize VPC Service Controls client
        
        Args:
            organization_id: GCP organization ID (required for VPC-SC)
            project_id: GCP project ID
        """
        self.organization_id = organization_id
        self.project_id = project_id
        
        if not GCLOUD_AVAILABLE:
            logger.warning("Google Cloud Access Context Manager library not available")
            self.client = None
            return
        
        try:
            # Initialize the Access Context Manager client
            self.client = accesscontextmanager_v1.AccessContextManagerClient()
            
            # Set up resource names
            self.org_name = f"organizations/{organization_id}"
            self.access_policy_name = None  # Will be determined dynamically
            
            logger.info(f"VPC Service Controls client initialized for org: {organization_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize VPC Service Controls client: {e}")
            self.client = None
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to VPC Service Controls API"""
        if not self.client:
            return {
                "connected": False,
                "error": "Google Cloud Access Context Manager library not available",
                "message": "Install google-cloud-access-context-manager package"
            }
        
        try:
            # Test by listing access policies
            request = types.ListAccessPoliciesRequest(
                parent=self.org_name,
                page_size=1
            )
            
            response = self.client.list_access_policies(request=request)
            access_policies = [policy for policy in response]
            
            if access_policies:
                self.access_policy_name = access_policies[0].name
            
            return {
                "connected": True,
                "organization_id": self.organization_id,
                "project_id": self.project_id,
                "access_policies_found": len(access_policies),
                "message": "Connection successful"
            }
            
        except Exception as e:
            logger.error(f"VPC-SC connection test failed: {e}")
            return {
                "connected": False,
                "error": str(e),
                "message": "Connection test failed"
            }
    
    async def get_access_policies(self) -> Dict[str, Any]:
        """Get organization access policies"""
        if not self.client:
            return {
                "success": False,
                "error": "VPC Service Controls client not available"
            }
        
        try:
            request = types.ListAccessPoliciesRequest(parent=self.org_name)
            response = self.client.list_access_policies(request=request)
            
            policies = []
            for policy in response:
                policies.append({
                    "name": policy.name,
                    "title": policy.title,
                    "created_time": policy.create_time.isoformat() if policy.create_time else None,
                    "updated_time": policy.update_time.isoformat() if policy.update_time else None,
                    "scopes": list(policy.scopes) if policy.scopes else []
                })
                
                # Set the first policy as default if not set
                if not self.access_policy_name:
                    self.access_policy_name = policy.name
            
            return {
                "success": True,
                "total_policies": len(policies),
                "policies": policies
            }
            
        except Exception as e:
            logger.error(f"Get access policies failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def list_service_perimeters(self, policy_name: Optional[str] = None) -> Dict[str, Any]:
        """List service perimeters in an access policy"""
        if not self.client:
            return {
                "success": False,
                "error": "VPC Service Controls client not available"
            }
        
        try:
            # Use provided policy or default
            target_policy = policy_name or self.access_policy_name
            if not target_policy:
                # Try to get the first available policy
                policies_result = await self.get_access_policies()
                if policies_result["success"] and policies_result["policies"]:
                    target_policy = policies_result["policies"][0]["name"]
                else:
                    return {
                        "success": False,
                        "error": "No access policy found"
                    }
            
            request = types.ListServicePerimetersRequest(parent=target_policy)
            response = self.client.list_service_perimeters(request=request)
            
            perimeters = []
            for perimeter in response:
                perimeter_data = {
                    "name": perimeter.name,
                    "title": perimeter.title,
                    "description": perimeter.description,
                    "created_time": perimeter.create_time.isoformat() if perimeter.create_time else None,
                    "updated_time": perimeter.update_time.isoformat() if perimeter.update_time else None,
                    "perimeter_type": perimeter.perimeter_type.name if perimeter.perimeter_type else None,
                    "use_explicit_dry_run_spec": perimeter.use_explicit_dry_run_spec
                }
                
                # Add status information
                if perimeter.status:
                    perimeter_data["status"] = {
                        "resources": list(perimeter.status.resources),
                        "access_levels": list(perimeter.status.access_levels),
                        "restricted_services": list(perimeter.status.restricted_services),
                        "vpc_accessible_services": {
                            "enable_restriction": perimeter.status.vpc_accessible_services.enable_restriction,
                            "allowed_services": list(perimeter.status.vpc_accessible_services.allowed_services) if perimeter.status.vpc_accessible_services else []
                        } if perimeter.status.vpc_accessible_services else None,
                        "ingress_policies": len(perimeter.status.ingress_policies) if perimeter.status.ingress_policies else 0,
                        "egress_policies": len(perimeter.status.egress_policies) if perimeter.status.egress_policies else 0
                    }
                
                # Add dry run specification if available
                if perimeter.spec:
                    perimeter_data["dry_run_spec"] = {
                        "resources": list(perimeter.spec.resources),
                        "access_levels": list(perimeter.spec.access_levels),
                        "restricted_services": list(perimeter.spec.restricted_services),
                        "ingress_policies": len(perimeter.spec.ingress_policies) if perimeter.spec.ingress_policies else 0,
                        "egress_policies": len(perimeter.spec.egress_policies) if perimeter.spec.egress_policies else 0
                    }
                
                perimeters.append(perimeter_data)
            
            return {
                "success": True,
                "policy_name": target_policy,
                "total_perimeters": len(perimeters),
                "perimeters": perimeters
            }
            
        except Exception as e:
            logger.error(f"List service perimeters failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_service_perimeter(self, perimeter_name: str) -> Dict[str, Any]:
        """Get detailed information about a service perimeter"""
        if not self.client:
            return {
                "success": False,
                "error": "VPC Service Controls client not available"
            }
        
        try:
            request = types.GetServicePerimeterRequest(name=perimeter_name)
            perimeter = self.client.get_service_perimeter(request=request)
            
            perimeter_data = {
                "name": perimeter.name,
                "title": perimeter.title,
                "description": perimeter.description,
                "created_time": perimeter.create_time.isoformat() if perimeter.create_time else None,
                "updated_time": perimeter.update_time.isoformat() if perimeter.update_time else None,
                "perimeter_type": perimeter.perimeter_type.name if perimeter.perimeter_type else None,
                "use_explicit_dry_run_spec": perimeter.use_explicit_dry_run_spec
            }
            
            # Add detailed status
            if perimeter.status:
                perimeter_data["status"] = self._extract_perimeter_config(perimeter.status)
            
            # Add dry run spec
            if perimeter.spec:
                perimeter_data["dry_run_spec"] = self._extract_perimeter_config(perimeter.spec)
            
            return {
                "success": True,
                "perimeter": perimeter_data
            }
            
        except Exception as e:
            logger.error(f"Get service perimeter failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def test_dry_run_violations(self, perimeter_name: str) -> Dict[str, Any]:
        """Test for dry run violations in a service perimeter"""
        if not self.client:
            return {
                "success": False,
                "error": "VPC Service Controls client not available"
            }
        
        try:
            # Get perimeter details
            perimeter_result = await self.get_service_perimeter(perimeter_name)
            if not perimeter_result["success"]:
                return perimeter_result
            
            perimeter_data = perimeter_result["perimeter"]
            
            # Analyze dry run configuration
            violations = []
            
            if perimeter_data.get("use_explicit_dry_run_spec") and perimeter_data.get("dry_run_spec"):
                dry_run_spec = perimeter_data["dry_run_spec"]
                status = perimeter_data.get("status", {})
                
                # Compare dry run spec with current status
                violations.extend(self._analyze_resource_differences(
                    current=status.get("resources", []),
                    proposed=dry_run_spec.get("resources", []),
                    category="resources"
                ))
                
                violations.extend(self._analyze_service_differences(
                    current=status.get("restricted_services", []),
                    proposed=dry_run_spec.get("restricted_services", []),
                    category="restricted_services"
                ))
                
                violations.extend(self._analyze_access_level_differences(
                    current=status.get("access_levels", []),
                    proposed=dry_run_spec.get("access_levels", []),
                    category="access_levels"
                ))
            
            # Simulate dry run impact analysis
            impact_analysis = await self._simulate_dry_run_impact(perimeter_name, violations)
            
            return {
                "success": True,
                "perimeter_name": perimeter_name,
                "dry_run_enabled": perimeter_data.get("use_explicit_dry_run_spec", False),
                "total_violations": len(violations),
                "violations": violations,
                "impact_analysis": impact_analysis,
                "analysis_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Dry run violation test failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def list_access_levels(self, policy_name: Optional[str] = None) -> Dict[str, Any]:
        """List access levels in an access policy"""
        if not self.client:
            return {
                "success": False,
                "error": "VPC Service Controls client not available"
            }
        
        try:
            target_policy = policy_name or self.access_policy_name
            if not target_policy:
                policies_result = await self.get_access_policies()
                if policies_result["success"] and policies_result["policies"]:
                    target_policy = policies_result["policies"][0]["name"]
                else:
                    return {
                        "success": False,
                        "error": "No access policy found"
                    }
            
            request = types.ListAccessLevelsRequest(parent=target_policy)
            response = self.client.list_access_levels(request=request)
            
            access_levels = []
            for level in response:
                level_data = {
                    "name": level.name,
                    "title": level.title,
                    "description": level.description,
                    "created_time": level.create_time.isoformat() if level.create_time else None,
                    "updated_time": level.update_time.isoformat() if level.update_time else None
                }
                
                # Add basic conditions info
                if level.basic:
                    conditions = []
                    for condition in level.basic.conditions:
                        condition_data = {
                            "ip_subnetworks": list(condition.ip_subnetworks) if condition.ip_subnetworks else [],
                            "required_access_levels": list(condition.required_access_levels) if condition.required_access_levels else [],
                            "members": list(condition.members) if condition.members else [],
                            "negate": condition.negate,
                            "device_policy": bool(condition.device_policy) if condition.device_policy else False,
                            "regions": list(condition.regions) if condition.regions else []
                        }
                        conditions.append(condition_data)
                    
                    level_data["basic_conditions"] = conditions
                    level_data["combining_function"] = level.basic.combining_function.name if level.basic.combining_function else None
                
                access_levels.append(level_data)
            
            return {
                "success": True,
                "policy_name": target_policy,
                "total_access_levels": len(access_levels),
                "access_levels": access_levels
            }
            
        except Exception as e:
            logger.error(f"List access levels failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def analyze_perimeter_coverage(self, policy_name: Optional[str] = None) -> Dict[str, Any]:
        """Analyze service perimeter coverage and gaps"""
        try:
            # Get all perimeters
            perimeters_result = await self.list_service_perimeters(policy_name)
            if not perimeters_result["success"]:
                return perimeters_result
            
            perimeters = perimeters_result["perimeters"]
            
            # Analyze coverage
            all_resources = set()
            all_services = set()
            perimeter_analysis = []
            
            for perimeter in perimeters:
                status = perimeter.get("status", {})
                resources = status.get("resources", [])
                services = status.get("restricted_services", [])
                
                all_resources.update(resources)
                all_services.update(services)
                
                perimeter_analysis.append({
                    "name": perimeter["name"],
                    "title": perimeter["title"],
                    "resource_count": len(resources),
                    "service_count": len(services),
                    "has_dry_run": bool(perimeter.get("dry_run_spec")),
                    "ingress_policies": perimeter.get("status", {}).get("ingress_policies", 0),
                    "egress_policies": perimeter.get("status", {}).get("egress_policies", 0)
                })
            
            # Coverage analysis
            coverage_analysis = {
                "total_perimeters": len(perimeters),
                "total_protected_resources": len(all_resources),
                "total_restricted_services": len(all_services),
                "perimeter_breakdown": perimeter_analysis,
                "common_services": self._get_common_restricted_services(),
                "recommendations": self._generate_coverage_recommendations(perimeters)
            }
            
            return {
                "success": True,
                "policy_name": perimeters_result["policy_name"],
                "coverage_analysis": coverage_analysis,
                "analysis_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Perimeter coverage analysis failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extract_perimeter_config(self, config) -> Dict[str, Any]:
        """Extract perimeter configuration details"""
        config_data = {
            "resources": list(config.resources) if config.resources else [],
            "access_levels": list(config.access_levels) if config.access_levels else [],
            "restricted_services": list(config.restricted_services) if config.restricted_services else []
        }
        
        # VPC accessible services
        if config.vpc_accessible_services:
            config_data["vpc_accessible_services"] = {
                "enable_restriction": config.vpc_accessible_services.enable_restriction,
                "allowed_services": list(config.vpc_accessible_services.allowed_services)
            }
        
        # Ingress and egress policies
        config_data["ingress_policies"] = []
        if config.ingress_policies:
            for policy in config.ingress_policies:
                policy_data = {
                    "ingress_from": {
                        "sources": [],
                        "identities": list(policy.ingress_from.identities) if policy.ingress_from and policy.ingress_from.identities else [],
                        "identity_type": policy.ingress_from.identity_type.name if policy.ingress_from and policy.ingress_from.identity_type else None
                    } if policy.ingress_from else None,
                    "ingress_to": {
                        "operations": len(policy.ingress_to.operations) if policy.ingress_to and policy.ingress_to.operations else 0,
                        "resources": list(policy.ingress_to.resources) if policy.ingress_to and policy.ingress_to.resources else []
                    } if policy.ingress_to else None
                }
                config_data["ingress_policies"].append(policy_data)
        
        config_data["egress_policies"] = []
        if config.egress_policies:
            for policy in config.egress_policies:
                policy_data = {
                    "egress_from": {
                        "identities": list(policy.egress_from.identities) if policy.egress_from and policy.egress_from.identities else [],
                        "identity_type": policy.egress_from.identity_type.name if policy.egress_from and policy.egress_from.identity_type else None
                    } if policy.egress_from else None,
                    "egress_to": {
                        "operations": len(policy.egress_to.operations) if policy.egress_to and policy.egress_to.operations else 0,
                        "resources": list(policy.egress_to.resources) if policy.egress_to and policy.egress_to.resources else [],
                        "external_resources": list(policy.egress_to.external_resources) if policy.egress_to and policy.egress_to.external_resources else []
                    } if policy.egress_to else None
                }
                config_data["egress_policies"].append(policy_data)
        
        return config_data
    
    def _analyze_resource_differences(self, current: List[str], proposed: List[str], category: str) -> List[Dict[str, Any]]:
        """Analyze differences between current and proposed configurations"""
        violations = []
        
        current_set = set(current)
        proposed_set = set(proposed)
        
        # Resources to be added
        to_add = proposed_set - current_set
        for resource in to_add:
            violations.append({
                "violation_type": "RESOURCE_ADDITION",
                "category": category,
                "resource": resource,
                "impact": "MEDIUM",
                "description": f"Resource {resource} will be added to perimeter",
                "severity": "MEDIUM"
            })
        
        # Resources to be removed
        to_remove = current_set - proposed_set
        for resource in to_remove:
            violations.append({
                "violation_type": "RESOURCE_REMOVAL",
                "category": category,
                "resource": resource,
                "impact": "HIGH",
                "description": f"Resource {resource} will be removed from perimeter",
                "severity": "HIGH"
            })
        
        return violations
    
    def _analyze_service_differences(self, current: List[str], proposed: List[str], category: str) -> List[Dict[str, Any]]:
        """Analyze differences in restricted services"""
        violations = []
        
        current_set = set(current)
        proposed_set = set(proposed)
        
        # Services to be restricted
        to_restrict = proposed_set - current_set
        for service in to_restrict:
            violations.append({
                "violation_type": "SERVICE_RESTRICTION_ADDED",
                "category": category,
                "service": service,
                "impact": "HIGH",
                "description": f"Service {service} will be restricted",
                "severity": "HIGH"
            })
        
        # Services to be unrestricted
        to_unrestrict = current_set - proposed_set
        for service in to_unrestrict:
            violations.append({
                "violation_type": "SERVICE_RESTRICTION_REMOVED",
                "category": category,
                "service": service,
                "impact": "MEDIUM",
                "description": f"Service {service} restriction will be removed",
                "severity": "MEDIUM"
            })
        
        return violations
    
    def _analyze_access_level_differences(self, current: List[str], proposed: List[str], category: str) -> List[Dict[str, Any]]:
        """Analyze differences in access levels"""
        violations = []
        
        current_set = set(current)
        proposed_set = set(proposed)
        
        # Access levels to be added
        to_add = proposed_set - current_set
        for level in to_add:
            violations.append({
                "violation_type": "ACCESS_LEVEL_ADDED",
                "category": category,
                "access_level": level,
                "impact": "MEDIUM",
                "description": f"Access level {level} will be added",
                "severity": "MEDIUM"
            })
        
        # Access levels to be removed
        to_remove = current_set - proposed_set
        for level in to_remove:
            violations.append({
                "violation_type": "ACCESS_LEVEL_REMOVED",
                "category": category,
                "access_level": level,
                "impact": "HIGH",
                "description": f"Access level {level} will be removed",
                "severity": "HIGH"
            })
        
        return violations
    
    async def _simulate_dry_run_impact(self, perimeter_name: str, violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simulate the impact of dry run changes"""
        # Count violations by severity and type
        high_impact = len([v for v in violations if v.get("severity") == "HIGH"])
        medium_impact = len([v for v in violations if v.get("severity") == "MEDIUM"])
        low_impact = len([v for v in violations if v.get("severity") == "LOW"])
        
        # Estimate affected services and resources
        affected_services = set()
        affected_resources = set()
        
        for violation in violations:
            if violation.get("service"):
                affected_services.add(violation["service"])
            if violation.get("resource"):
                affected_resources.add(violation["resource"])
        
        return {
            "total_violations": len(violations),
            "high_impact_violations": high_impact,
            "medium_impact_violations": medium_impact,
            "low_impact_violations": low_impact,
            "affected_services": list(affected_services),
            "affected_resources": list(affected_resources),
            "estimated_downtime_risk": "HIGH" if high_impact > 5 else "MEDIUM" if high_impact > 0 else "LOW",
            "recommended_action": self._get_impact_recommendation(high_impact, medium_impact),
            "testing_required": high_impact > 0 or medium_impact > 3
        }
    
    def _get_common_restricted_services(self) -> List[str]:
        """Get list of commonly restricted GCP services"""
        return [
            "storage.googleapis.com",
            "bigquery.googleapis.com",
            "compute.googleapis.com",
            "container.googleapis.com",
            "cloudsql.googleapis.com",
            "dataflow.googleapis.com",
            "pubsub.googleapis.com",
            "logging.googleapis.com",
            "monitoring.googleapis.com",
            "cloudkms.googleapis.com"
        ]
    
    def _generate_coverage_recommendations(self, perimeters: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for perimeter coverage"""
        recommendations = []
        
        perimeters_without_dryrun = [p for p in perimeters if not p.get("dry_run_spec")]
        if perimeters_without_dryrun:
            recommendations.append(f"Enable dry-run mode for {len(perimeters_without_dryrun)} perimeters to test changes safely")
        
        small_perimeters = [p for p in perimeters if p.get("status", {}).get("resource_count", 0) < 5]
        if len(small_perimeters) > 3:
            recommendations.append("Consider consolidating small perimeters for easier management")
        
        perimeters_without_ingress = [p for p in perimeters if p.get("status", {}).get("ingress_policies", 0) == 0]
        if perimeters_without_ingress:
            recommendations.append("Review perimeters without ingress policies for potential access gaps")
        
        perimeters_without_egress = [p for p in perimeters if p.get("status", {}).get("egress_policies", 0) == 0]
        if perimeters_without_egress:
            recommendations.append("Consider adding explicit egress policies for better security control")
        
        return recommendations
    
    def _get_impact_recommendation(self, high_impact: int, medium_impact: int) -> str:
        """Get recommendation based on impact analysis"""
        if high_impact > 5:
            return "CAUTION: High number of high-impact violations. Extensive testing recommended before applying changes."
        elif high_impact > 0:
            return "WARNING: High-impact violations detected. Review and test carefully before applying."
        elif medium_impact > 10:
            return "REVIEW: Multiple medium-impact violations. Consider phased rollout."
        else:
            return "LOW RISK: Changes appear to have minimal impact. Proceed with normal testing."
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get VPC Service Controls statistics"""
        try:
            # Get perimeters data
            perimeters_result = await self.list_service_perimeters()
            access_levels_result = await self.list_access_levels()
            
            if not perimeters_result["success"]:
                return perimeters_result
            
            perimeters = perimeters_result["perimeters"]
            access_levels = access_levels_result.get("access_levels", []) if access_levels_result["success"] else []
            
            # Calculate statistics
            total_perimeters = len(perimeters)
            perimeters_with_dryrun = len([p for p in perimeters if p.get("dry_run_spec")])
            total_protected_resources = sum(len(p.get("status", {}).get("resources", [])) for p in perimeters)
            total_restricted_services = len(set().union(*[p.get("status", {}).get("restricted_services", []) for p in perimeters]))
            
            return {
                "success": True,
                "total_perimeters": total_perimeters,
                "perimeters_with_dry_run": perimeters_with_dryrun,
                "dry_run_adoption_rate": (perimeters_with_dryrun / total_perimeters * 100) if total_perimeters > 0 else 0,
                "total_access_levels": len(access_levels),
                "total_protected_resources": total_protected_resources,
                "total_restricted_services": total_restricted_services,
                "organization_id": self.organization_id,
                "analysis_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"VPC-SC statistics failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Example usage and testing
async def test_vpc_sc_client():
    """Test VPC Service Controls client functionality"""
    org_id = os.getenv("GOOGLE_CLOUD_ORGANIZATION", "123456789")
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "test-project")
    
    client = VPCServiceControlsClient(
        organization_id=org_id,
        project_id=project_id
    )
    
    # Test connection
    connection = await client.test_connection()
    print(f"Connection test: {connection}")
    
    if connection["connected"]:
        # Get access policies
        policies = await client.get_access_policies()
        print(f"Access policies: {policies}")
        
        # List service perimeters
        perimeters = await client.list_service_perimeters()
        print(f"Service perimeters: {perimeters}")
        
        # If we have perimeters, test dry run analysis
        if perimeters["success"] and perimeters["perimeters"]:
            first_perimeter = perimeters["perimeters"][0]["name"]
            violations = await client.test_dry_run_violations(first_perimeter)
            print(f"Dry run violations: {violations}")
        
        # Get statistics
        stats = await client.get_statistics()
        print(f"VPC-SC statistics: {stats}")


if __name__ == "__main__":
    asyncio.run(test_vpc_sc_client())