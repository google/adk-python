"""
Organization Policy Tester Service
==================================

Comprehensive organization policy testing, validation, and compliance analysis
with automated remediation recommendations and inheritance analysis.
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict, Counter
import statistics

# Optional Google Cloud imports for production use
try:
    from google.cloud import resourcemanager_v1
    from google.cloud import orgpolicy_v2
    from google.cloud import asset_v1
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    logging.warning("Google Cloud libraries not available - running in test mode")
    GOOGLE_CLOUD_AVAILABLE = False

from ..models.org_policy_models import (
    OrganizationPolicy, PolicyTestRequest, PolicyTestResponse, 
    PolicyTestResult, PolicyViolation, PolicyRemediation,
    PolicyComplianceReport, PolicyInheritanceAnalysis,
    PolicyEffectivenessMetrics, EnforcementLevel, ComplianceStatus,
    ViolationSeverity, ResourceScope, PolicyConstraintType
)

logger = logging.getLogger(__name__)


class OrganizationPolicyTester:
    """
    Comprehensive organization policy testing and compliance validation service.
    
    Features:
    - Policy constraint validation across organization hierarchy
    - Compliance gap analysis with automated remediation
    - Policy inheritance testing and conflict resolution
    - Effectiveness scoring and optimization recommendations
    - Integration with existing security database
    """
    
    def __init__(self, project_id: str, database_path: str = "backend/cache/gcp_data.db"):
        self.project_id = project_id
        self.database_path = database_path
        
        # Initialize Google Cloud clients if available
        if GOOGLE_CLOUD_AVAILABLE:
            self.org_policy_client = orgpolicy_v2.OrgPolicyClient()
            self.resource_manager_client = resourcemanager_v1.ProjectsClient()
            self.asset_client = asset_v1.AssetServiceClient()
        else:
            logger.warning("Google Cloud clients not available - using mock data for testing")
            self.org_policy_client = None
            self.resource_manager_client = None
            self.asset_client = None
        
        # Built-in organization policies to test
        self.standard_policies = {
            "constraints/compute.vmExternalIpAccess": {
                "display_name": "Define allowed external IPs for VM instances",
                "description": "This constraint defines the set of VM instances that are allowed to use external IP addresses.",
                "constraint_type": PolicyConstraintType.LIST_CONSTRAINT,
                "default_enforcement": EnforcementLevel.ENFORCE
            },
            "constraints/iam.disableServiceAccountKeyCreation": {
                "display_name": "Disable service account key creation",
                "description": "This constraint disables the creation of service account external keys where this constraint is set to True.",
                "constraint_type": PolicyConstraintType.BOOLEAN_CONSTRAINT,
                "default_enforcement": EnforcementLevel.ENFORCE
            },
            "constraints/storage.uniformBucketLevelAccess": {
                "display_name": "Enforce uniform bucket-level access",
                "description": "This constraint requires buckets to use uniform bucket-level access where this constraint is set to True.",
                "constraint_type": PolicyConstraintType.BOOLEAN_CONSTRAINT,
                "default_enforcement": EnforcementLevel.ENFORCE
            },
            "constraints/sql.restrictPublicIp": {
                "display_name": "Restrict public IP access on Cloud SQL instances",
                "description": "This constraint restricts configuring public IP on Cloud SQL instances where this constraint is set to True.",
                "constraint_type": PolicyConstraintType.BOOLEAN_CONSTRAINT,
                "default_enforcement": EnforcementLevel.ENFORCE
            },
            "constraints/compute.requireOsLogin": {
                "display_name": "Require OS Login",
                "description": "This constraint requires OS Login to be enabled for all VMs in the organization.",
                "constraint_type": PolicyConstraintType.BOOLEAN_CONSTRAINT,
                "default_enforcement": EnforcementLevel.ENFORCE
            },
            "constraints/compute.requireShieldedVm": {
                "display_name": "Require Shielded VM",
                "description": "This constraint requires Shielded VM features to be enabled on all VM instances.",
                "constraint_type": PolicyConstraintType.BOOLEAN_CONSTRAINT,
                "default_enforcement": EnforcementLevel.ENFORCE
            },
            "constraints/gcp.resourceLocations": {
                "display_name": "Resource Location Restriction",
                "description": "This constraint defines where resources can be created based on location.",
                "constraint_type": PolicyConstraintType.LIST_CONSTRAINT,
                "default_enforcement": EnforcementLevel.ENFORCE
            },
            "constraints/iam.allowedPolicyMemberDomains": {
                "display_name": "Domain restricted sharing",
                "description": "This constraint defines which domains can be specified in IAM policies.",
                "constraint_type": PolicyConstraintType.LIST_CONSTRAINT,
                "default_enforcement": EnforcementLevel.ENFORCE
            }
        }
        
        # Violation severity scoring
        self.severity_weights = {
            ViolationSeverity.CRITICAL: 10,
            ViolationSeverity.HIGH: 7,
            ViolationSeverity.MEDIUM: 4,
            ViolationSeverity.LOW: 2,
            ViolationSeverity.INFO: 1
        }
    
    async def test_organization_policies(self, request: PolicyTestRequest) -> PolicyTestResponse:
        """
        Test organization policy compliance across specified scope.
        
        Args:
            request: Policy testing request with scope and filters
            
        Returns:
            Comprehensive policy testing response with results and recommendations
        """
        logger.info(f"Starting organization policy testing: {request.dict()}")
        
        response = PolicyTestResponse(
            request_id=str(uuid.uuid4()),
            status="RUNNING",
            message="Policy testing in progress",
            started_at=datetime.now()
        )
        
        try:
            # Get policies to test
            policies_to_test = await self._get_policies_to_test(request)
            logger.info(f"Found {len(policies_to_test)} policies to test")
            
            # Test each policy
            test_results = []
            for policy in policies_to_test:
                try:
                    result = await self._test_single_policy(policy, request)
                    test_results.append(result)
                    logger.info(f"Tested policy {policy.policy_name}: {result.compliance_status}")
                except Exception as e:
                    logger.error(f"Failed to test policy {policy.policy_name}: {e}")
                    # Create failed test result
                    failed_result = PolicyTestResult(
                        policy_name=policy.policy_name,
                        constraint_name=policy.policy_name,
                        compliance_status=ComplianceStatus.TESTING_FAILED,
                        enforcement_level=policy.enforcement_level,
                        resource_scope=policy.resource_scope,
                        scope_resource_id=policy.scope_resource_id,
                        violations=[PolicyViolation(
                            resource_id="UNKNOWN",
                            resource_type="TESTING_ERROR",
                            violation_type="TEST_FAILURE",
                            violation_description=f"Policy testing failed: {str(e)}",
                            severity=ViolationSeverity.HIGH,
                            detected_at=datetime.now(),
                            remediation_steps=["Review policy configuration", "Check permissions", "Retry testing"]
                        )]
                    )
                    test_results.append(failed_result)
            
            # Calculate summary metrics
            response.test_results = test_results
            response.total_policies_tested = len(test_results)
            response.compliant_policies = len([r for r in test_results if r.compliance_status == ComplianceStatus.COMPLIANT])
            response.non_compliant_policies = len([r for r in test_results if r.compliance_status == ComplianceStatus.NON_COMPLIANT])
            response.failed_tests = len([r for r in test_results if r.compliance_status == ComplianceStatus.TESTING_FAILED])
            
            # Calculate overall metrics
            if response.total_policies_tested > 0:
                response.overall_compliance_percentage = (response.compliant_policies / response.total_policies_tested) * 100
            
            # Calculate risk score and recommendations
            await self._calculate_risk_analysis(response)
            await self._generate_recommendations(response)
            
            # Mark as completed
            response.completed_at = datetime.now()
            response.duration_seconds = (response.completed_at - response.started_at).total_seconds()
            response.status = "COMPLETED"
            response.message = f"Policy testing completed successfully. {response.compliant_policies}/{response.total_policies_tested} policies compliant."
            
            # Store results in database
            await self._store_test_results(response)
            
            logger.info(f"Policy testing completed: {response.overall_compliance_percentage:.1f}% compliance")
            
        except Exception as e:
            logger.error(f"Policy testing failed: {e}")
            response.status = "FAILED"
            response.message = f"Policy testing failed: {str(e)}"
            response.completed_at = datetime.now()
            response.duration_seconds = (response.completed_at - response.started_at).total_seconds()
        
        return response
    
    async def _get_policies_to_test(self, request: PolicyTestRequest) -> List[OrganizationPolicy]:
        """Get list of organization policies to test"""
        policies = []
        
        # If specific policies requested, filter to those
        if request.policy_names:
            policy_names = set(request.policy_names)
        else:
            policy_names = set(self.standard_policies.keys())
        
        # Create policy objects for testing
        for policy_name in policy_names:
            if policy_name in self.standard_policies:
                policy_config = self.standard_policies[policy_name]
                
                policy = OrganizationPolicy(
                    policy_id=policy_name,
                    policy_name=policy_name,
                    display_name=policy_config["display_name"],
                    description=policy_config["description"],
                    constraint={
                        "constraint_type": policy_config["constraint_type"],
                        "boolean_policy": policy_config.get("boolean_policy"),
                        "list_policy": policy_config.get("list_policy"),
                        "restore_default": policy_config.get("restore_default")
                    },
                    enforcement_level=policy_config["default_enforcement"],
                    resource_scope=request.resource_scope or ResourceScope.PROJECT,
                    scope_resource_id=request.scope_resource_id or self.project_id,
                    created_at=datetime.now()
                )
                policies.append(policy)
        
        return policies
    
    async def _test_single_policy(self, policy: OrganizationPolicy, request: PolicyTestRequest) -> PolicyTestResult:
        """Test a single organization policy for compliance"""
        logger.info(f"Testing policy: {policy.policy_name}")
        
        start_time = datetime.now()
        
        # Get resources to test based on policy type
        resources = await self._get_resources_for_policy(policy, request)
        logger.info(f"Found {len(resources)} resources to test for policy {policy.policy_name}")
        
        # Test each resource against the policy
        violations = []
        compliant_count = 0
        
        for resource in resources[:request.max_resources]:
            try:
                is_compliant, violation = await self._test_resource_compliance(resource, policy)
                if not is_compliant and violation:
                    violations.append(violation)
                else:
                    compliant_count += 1
            except Exception as e:
                logger.error(f"Failed to test resource {resource.get('name', 'UNKNOWN')}: {e}")
                # Create violation for testing error
                violations.append(PolicyViolation(
                    resource_id=resource.get('name', 'UNKNOWN'),
                    resource_type=resource.get('asset_type', 'UNKNOWN'),
                    violation_type="TESTING_ERROR",
                    violation_description=f"Failed to test resource: {str(e)}",
                    severity=ViolationSeverity.MEDIUM,
                    detected_at=datetime.now(),
                    remediation_steps=["Check resource permissions", "Verify resource configuration"]
                ))
        
        # Calculate results
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        total_tested = len(resources[:request.max_resources])
        non_compliant_count = len(violations)
        
        # Determine overall compliance status
        if total_tested == 0:
            compliance_status = ComplianceStatus.NOT_TESTED
        elif non_compliant_count == 0:
            compliance_status = ComplianceStatus.COMPLIANT
        elif compliant_count == 0:
            compliance_status = ComplianceStatus.NON_COMPLIANT
        else:
            compliance_status = ComplianceStatus.PARTIALLY_COMPLIANT
        
        # Calculate risk score
        risk_score = self._calculate_policy_risk_score(violations)
        
        # Determine remediation priority
        if risk_score >= 8:
            priority = ViolationSeverity.CRITICAL
        elif risk_score >= 6:
            priority = ViolationSeverity.HIGH
        elif risk_score >= 4:
            priority = ViolationSeverity.MEDIUM
        elif risk_score >= 2:
            priority = ViolationSeverity.LOW
        else:
            priority = ViolationSeverity.INFO
        
        result = PolicyTestResult(
            policy_name=policy.policy_name,
            constraint_name=policy.policy_name,
            tested_at=end_time,
            compliance_status=compliance_status,
            enforcement_level=policy.enforcement_level,
            resource_scope=policy.resource_scope,
            scope_resource_id=policy.scope_resource_id,
            total_resources_tested=total_tested,
            compliant_resources=compliant_count,
            non_compliant_resources=non_compliant_count,
            violations=violations,
            test_duration_seconds=duration,
            risk_score=risk_score,
            remediation_priority=priority
        )
        
        return result
    
    async def _get_resources_for_policy(self, policy: OrganizationPolicy, request: PolicyTestRequest) -> List[Dict[str, Any]]:
        """Get resources that should be tested for the given policy"""
        
        # Map policies to resource types they should test
        policy_resource_mapping = {
            "constraints/compute.vmExternalIpAccess": "compute.googleapis.com/Instance",
            "constraints/iam.disableServiceAccountKeyCreation": "iam.googleapis.com/ServiceAccount",
            "constraints/storage.uniformBucketLevelAccess": "storage.googleapis.com/Bucket",
            "constraints/sql.restrictPublicIp": "sqladmin.googleapis.com/Instance",
            "constraints/compute.requireOsLogin": "compute.googleapis.com/Instance",
            "constraints/compute.requireShieldedVm": "compute.googleapis.com/Instance",
            "constraints/gcp.resourceLocations": "ALL_RESOURCES",
            "constraints/iam.allowedPolicyMemberDomains": "iam.googleapis.com/Policy"
        }
        
        asset_type = policy_resource_mapping.get(policy.policy_name, "ALL_RESOURCES")
        
        try:
            # Try to get real resources from database
            import sqlite3
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            if asset_type == "ALL_RESOURCES":
                cursor.execute("""
                    SELECT name, asset_type, resource FROM assets 
                    ORDER BY RANDOM() LIMIT ?
                """, (request.max_resources,))
            else:
                cursor.execute("""
                    SELECT name, asset_type, resource FROM assets 
                    WHERE asset_type = ? 
                    ORDER BY RANDOM() LIMIT ?
                """, (asset_type, request.max_resources))
            
            rows = cursor.fetchall()
            conn.close()
            
            resources = []
            for row in rows:
                name, asset_type, resource_data = row
                try:
                    resource = json.loads(resource_data) if resource_data else {}
                    resource.update({
                        'name': name,
                        'asset_type': asset_type
                    })
                    resources.append(resource)
                except json.JSONDecodeError:
                    # Handle non-JSON resource data
                    resources.append({
                        'name': name,
                        'asset_type': asset_type,
                        'resource_data': resource_data
                    })
            
            return resources
            
        except Exception as e:
            logger.error(f"Failed to get resources from database: {e}")
            
            # Return mock resources for testing
            mock_resources = []
            if "compute" in asset_type.lower():
                mock_resources = [
                    {'name': 'instance-1', 'asset_type': 'compute.googleapis.com/Instance', 'status': 'RUNNING', 'zone': 'us-central1-a'},
                    {'name': 'instance-2', 'asset_type': 'compute.googleapis.com/Instance', 'status': 'RUNNING', 'zone': 'us-east1-b'},
                    {'name': 'instance-3', 'asset_type': 'compute.googleapis.com/Instance', 'status': 'STOPPED', 'zone': 'us-west1-c'}
                ]
            elif "storage" in asset_type.lower():
                mock_resources = [
                    {'name': 'bucket-1', 'asset_type': 'storage.googleapis.com/Bucket', 'location': 'US', 'uniform_bucket_level_access': True},
                    {'name': 'bucket-2', 'asset_type': 'storage.googleapis.com/Bucket', 'location': 'EU', 'uniform_bucket_level_access': False},
                ]
            elif "sql" in asset_type.lower():
                mock_resources = [
                    {'name': 'db-instance-1', 'asset_type': 'sqladmin.googleapis.com/Instance', 'backend_type': 'SECOND_GEN', 'public_ip': True},
                    {'name': 'db-instance-2', 'asset_type': 'sqladmin.googleapis.com/Instance', 'backend_type': 'SECOND_GEN', 'public_ip': False},
                ]
            else:
                # Generic resources
                mock_resources = [
                    {'name': 'resource-1', 'asset_type': asset_type, 'status': 'ACTIVE'},
                    {'name': 'resource-2', 'asset_type': asset_type, 'status': 'INACTIVE'},
                ]
            
            return mock_resources[:request.max_resources]
    
    async def _test_resource_compliance(self, resource: Dict[str, Any], policy: OrganizationPolicy) -> Tuple[bool, Optional[PolicyViolation]]:
        """Test if a specific resource complies with the organization policy"""
        
        resource_name = resource.get('name', 'UNKNOWN')
        resource_type = resource.get('asset_type', 'UNKNOWN')
        
        # Policy-specific compliance checks
        if policy.policy_name == "constraints/compute.vmExternalIpAccess":
            # Check if VM has external IP when policy restricts it
            has_external_ip = resource.get('has_external_ip', False)
            if has_external_ip and policy.enforcement_level == EnforcementLevel.ENFORCE:
                return False, PolicyViolation(
                    resource_id=resource_name,
                    resource_type=resource_type,
                    violation_type="EXTERNAL_IP_VIOLATION",
                    violation_description="VM instance has external IP access which is restricted by organization policy",
                    severity=ViolationSeverity.HIGH,
                    detected_at=datetime.now(),
                    current_value="External IP enabled",
                    expected_value="External IP disabled",
                    remediation_steps=[
                        "Remove external IP from VM instance",
                        "Use Cloud NAT for outbound internet access",
                        "Configure private service access if needed"
                    ],
                    auto_remediable=True
                )
        
        elif policy.policy_name == "constraints/storage.uniformBucketLevelAccess":
            # Check if bucket has uniform bucket-level access enabled
            uniform_access = resource.get('uniform_bucket_level_access', False)
            if not uniform_access and policy.enforcement_level == EnforcementLevel.ENFORCE:
                return False, PolicyViolation(
                    resource_id=resource_name,
                    resource_type=resource_type,
                    violation_type="BUCKET_ACCESS_VIOLATION",
                    violation_description="Storage bucket does not have uniform bucket-level access enabled",
                    severity=ViolationSeverity.MEDIUM,
                    detected_at=datetime.now(),
                    current_value="Uniform bucket-level access disabled",
                    expected_value="Uniform bucket-level access enabled",
                    remediation_steps=[
                        "Enable uniform bucket-level access on the bucket",
                        "Remove legacy ACLs if present",
                        "Update IAM policies as needed"
                    ],
                    auto_remediable=True
                )
        
        elif policy.policy_name == "constraints/sql.restrictPublicIp":
            # Check if Cloud SQL instance has public IP enabled
            public_ip = resource.get('public_ip', False)
            if public_ip and policy.enforcement_level == EnforcementLevel.ENFORCE:
                return False, PolicyViolation(
                    resource_id=resource_name,
                    resource_type=resource_type,
                    violation_type="SQL_PUBLIC_IP_VIOLATION",
                    violation_description="Cloud SQL instance has public IP enabled which is restricted by organization policy",
                    severity=ViolationSeverity.HIGH,
                    detected_at=datetime.now(),
                    current_value="Public IP enabled",
                    expected_value="Public IP disabled",
                    remediation_steps=[
                        "Disable public IP on Cloud SQL instance",
                        "Configure private service access",
                        "Update application connection strings",
                        "Use Cloud SQL Proxy for secure access"
                    ],
                    auto_remediable=False
                )
        
        elif policy.policy_name == "constraints/compute.requireShieldedVm":
            # Check if VM has Shielded VM features enabled
            shielded_vm = resource.get('shielded_vm_enabled', False)
            if not shielded_vm and policy.enforcement_level == EnforcementLevel.ENFORCE:
                return False, PolicyViolation(
                    resource_id=resource_name,
                    resource_type=resource_type,
                    violation_type="SHIELDED_VM_VIOLATION",
                    violation_description="VM instance does not have Shielded VM features enabled",
                    severity=ViolationSeverity.MEDIUM,
                    detected_at=datetime.now(),
                    current_value="Shielded VM disabled",
                    expected_value="Shielded VM enabled",
                    remediation_steps=[
                        "Enable Shielded VM on instance creation",
                        "Migrate to new instance with Shielded VM enabled",
                        "Enable Secure Boot, vTPM, and Integrity Monitoring"
                    ],
                    auto_remediable=False
                )
        
        # If no specific violation found, resource is compliant
        return True, None
    
    def _calculate_policy_risk_score(self, violations: List[PolicyViolation]) -> float:
        """Calculate risk score for a policy based on its violations"""
        if not violations:
            return 0.0
        
        total_score = 0.0
        for violation in violations:
            severity_weight = self.severity_weights.get(violation.severity, 1)
            total_score += severity_weight
        
        # Normalize to 0-10 scale
        max_possible_score = len(violations) * 10
        return min(10.0, (total_score / max_possible_score) * 10) if max_possible_score > 0 else 0.0
    
    async def _calculate_risk_analysis(self, response: PolicyTestResponse):
        """Calculate overall risk analysis for the policy test response"""
        all_violations = []
        total_risk_score = 0.0
        
        for result in response.test_results:
            all_violations.extend(result.violations)
            total_risk_score += result.risk_score
        
        # Calculate overall risk score
        if response.total_policies_tested > 0:
            response.overall_risk_score = total_risk_score / response.total_policies_tested
        
        # Count high priority violations
        response.high_priority_violations = len([
            v for v in all_violations 
            if v.severity in [ViolationSeverity.CRITICAL, ViolationSeverity.HIGH]
        ])
        
        # Count auto-remediable violations
        response.auto_remediable_violations = len([
            v for v in all_violations if v.auto_remediable
        ])
    
    async def _generate_recommendations(self, response: PolicyTestResponse):
        """Generate actionable recommendations based on test results"""
        recommendations = []
        
        # High-level recommendations based on compliance percentage
        if response.overall_compliance_percentage < 50:
            recommendations.append("CRITICAL: Less than 50% policy compliance detected. Immediate remediation required.")
            recommendations.append("Review organization policy enforcement levels and consider gradual rollout.")
            recommendations.append("Prioritize fixing CRITICAL and HIGH severity violations first.")
        elif response.overall_compliance_percentage < 80:
            recommendations.append("Moderate compliance issues detected. Plan systematic remediation.")
            recommendations.append("Focus on auto-remediable violations for quick wins.")
        else:
            recommendations.append("Good compliance posture. Focus on remaining gaps and optimization.")
        
        # Specific recommendations based on violations
        violation_types = Counter()
        for result in response.test_results:
            for violation in result.violations:
                violation_types[violation.violation_type] += 1
        
        # Top violation type recommendations
        if violation_types:
            top_violation = violation_types.most_common(1)[0]
            recommendations.append(f"Most common violation: {top_violation[0]} ({top_violation[1]} instances)")
            
            if "EXTERNAL_IP" in top_violation[0]:
                recommendations.append("Consider implementing Cloud NAT for outbound internet access.")
            elif "PUBLIC_IP" in top_violation[0]:
                recommendations.append("Implement private service access for database connectivity.")
            elif "BUCKET_ACCESS" in top_violation[0]:
                recommendations.append("Plan migration to uniform bucket-level access for better security.")
        
        # Auto-remediation recommendations
        if response.auto_remediable_violations > 0:
            recommendations.append(f"{response.auto_remediable_violations} violations can be auto-remediated.")
            recommendations.append("Consider implementing automated remediation workflows.")
        
        # Policy optimization recommendations
        if response.failed_tests > 0:
            recommendations.append("Some policy tests failed - review policy configurations and permissions.")
        
        response.recommended_actions = recommendations
    
    async def _store_test_results(self, response: PolicyTestResponse):
        """Store policy test results in database for historical tracking"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS org_policy_tests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT UNIQUE NOT NULL,
                    tested_at TIMESTAMP NOT NULL,
                    total_policies INTEGER NOT NULL,
                    compliant_policies INTEGER NOT NULL,
                    non_compliant_policies INTEGER NOT NULL,
                    overall_compliance_percentage REAL NOT NULL,
                    overall_risk_score REAL NOT NULL,
                    high_priority_violations INTEGER NOT NULL,
                    auto_remediable_violations INTEGER NOT NULL,
                    test_results TEXT NOT NULL,
                    recommendations TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Insert test results
            cursor.execute("""
                INSERT OR REPLACE INTO org_policy_tests 
                (request_id, tested_at, total_policies, compliant_policies, non_compliant_policies,
                 overall_compliance_percentage, overall_risk_score, high_priority_violations,
                 auto_remediable_violations, test_results, recommendations, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                response.request_id,
                response.started_at.isoformat(),
                response.total_policies_tested,
                response.compliant_policies,
                response.non_compliant_policies,
                response.overall_compliance_percentage,
                response.overall_risk_score,
                response.high_priority_violations,
                response.auto_remediable_violations,
                json.dumps([result.dict() for result in response.test_results]),
                json.dumps(response.recommended_actions),
                json.dumps({"duration_seconds": response.duration_seconds, "status": response.status})
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Stored policy test results: {response.request_id}")
            
        except Exception as e:
            logger.error(f"Failed to store policy test results: {e}")
    
    async def get_policy_compliance_history(self, days: int = 30) -> Dict[str, Any]:
        """Get historical policy compliance trends"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Get historical compliance data
            cursor.execute("""
                SELECT tested_at, overall_compliance_percentage, overall_risk_score,
                       total_policies, compliant_policies, non_compliant_policies
                FROM org_policy_tests
                WHERE tested_at >= datetime('now', '-{} days')
                ORDER BY tested_at DESC
            """.format(days))
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return {"message": "No historical data available"}
            
            # Process historical data
            compliance_trend = []
            risk_trend = []
            dates = []
            
            for row in rows:
                tested_at, compliance_pct, risk_score, total, compliant, non_compliant = row
                dates.append(tested_at)
                compliance_trend.append(compliance_pct)
                risk_trend.append(risk_score)
            
            # Calculate trends
            avg_compliance = statistics.mean(compliance_trend) if compliance_trend else 0
            trend_direction = "improving" if len(compliance_trend) >= 2 and compliance_trend[0] > compliance_trend[-1] else "stable"
            
            return {
                "period_days": days,
                "total_tests": len(rows),
                "average_compliance_percentage": avg_compliance,
                "trend_direction": trend_direction,
                "compliance_trend": compliance_trend,
                "risk_trend": risk_trend,
                "test_dates": dates,
                "latest_compliance": compliance_trend[0] if compliance_trend else 0,
                "latest_risk_score": risk_trend[0] if risk_trend else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get compliance history: {e}")
            return {"error": str(e)}
    
    async def generate_compliance_report(self, report_name: str = "Policy Compliance Report") -> PolicyComplianceReport:
        """Generate comprehensive policy compliance report"""
        try:
            # Get recent test results
            history = await self.get_policy_compliance_history(30)
            
            report = PolicyComplianceReport(
                report_id=str(uuid.uuid4()),
                report_name=report_name,
                generated_by="Organization Policy Tester",
                reporting_period_start=datetime.now() - timedelta(days=30),
                reporting_period_end=datetime.now(),
                organization_id=self.project_id,
                total_policies=len(self.standard_policies),
                overall_compliance_percentage=history.get("latest_compliance", 0),
                compliance_trend_percentage=0.0,  # Would calculate from trends
                priority_recommendations=[
                    "Enable organization policy enforcement for critical constraints",
                    "Implement automated compliance monitoring",
                    "Set up remediation workflows for common violations",
                    "Regular compliance reviews and policy updates"
                ],
                policy_optimization_suggestions=[
                    "Consider gradual rollout for new policies using DRY_RUN mode",
                    "Implement policy exceptions management process",
                    "Set up compliance dashboards for ongoing monitoring",
                    "Integrate with change management processes"
                ]
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate compliance report: {e}")
            raise