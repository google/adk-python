"""
Google Cloud Organization Policy API thin client wrapper.

This module provides a clean interface to Organization Policy for Day 2 operations.
Focuses on policy enforcement, compliance, and governance across the organization.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
import json

try:
    from google.cloud import orgpolicy_v2
    from google.api_core import exceptions
    from google.type import expr_pb2
    ORGPOLICY_AVAILABLE = True
except ImportError:
    ORGPOLICY_AVAILABLE = False
    orgpolicy_v2 = None

logger = logging.getLogger(__name__)

# ============================================
# Pydantic Models for Type Safety
# ============================================

class PolicyConstraintRequest(BaseModel):
    """Request model for managing policy constraints."""
    parent: str = Field(
        ...,
        description="Parent resource (e.g., organizations/12345, folders/67890, projects/my-project)"
    )
    constraint: str = Field(
        ...,
        description="Constraint name (e.g., compute.disableSerialPortAccess)"
    )

class PolicyRule(BaseModel):
    """Model for a policy rule."""
    values: Optional[Dict[str, Any]] = Field(
        None,
        description="List rule with allowed/denied values"
    )
    allow_all: Optional[bool] = Field(
        None,
        description="Allow all values"
    )
    deny_all: Optional[bool] = Field(
        None,
        description="Deny all values"
    )
    enforce: Optional[bool] = Field(
        None,
        description="Boolean constraint enforcement"
    )
    condition: Optional[Dict[str, Any]] = Field(
        None,
        description="CEL condition for the rule"
    )

class PolicyRequest(BaseModel):
    """Request model for creating/updating organization policies."""
    parent: str
    constraint: str
    rules: List[PolicyRule]
    inherit_from_parent: Optional[bool] = Field(
        False,
        description="Whether to inherit policy from parent"
    )
    reset: Optional[bool] = Field(
        False,
        description="Whether to reset policy to default"
    )
    etag: Optional[str] = Field(
        None,
        description="ETag for concurrency control"
    )

class CustomConstraintRequest(BaseModel):
    """Request model for custom constraints."""
    parent: str = Field(
        ...,
        description="Organization resource (e.g., organizations/12345)"
    )
    constraint_id: str
    display_name: str
    description: Optional[str] = None
    resource_types: List[str] = Field(
        ...,
        description="Resource types this constraint applies to"
    )
    method_types: List[str] = Field(
        ...,
        description="Method types (CREATE, UPDATE, DELETE)"
    )
    condition: str = Field(
        ...,
        description="CEL expression defining the constraint"
    )
    action_type: Optional[str] = Field(
        "DENY",
        description="Action to take (ALLOW or DENY)"
    )

class PolicyAnalysisRequest(BaseModel):
    """Request model for policy analysis."""
    scope: str = Field(
        ...,
        description="Scope for analysis (organization, folder, or project)"
    )
    constraint_filter: Optional[str] = None
    include_inherited: Optional[bool] = True
    check_compliance: Optional[bool] = True

# ============================================
# Core Organization Policy Functions
# ============================================

async def list_constraints(parent: str) -> Dict[str, Any]:
    """
    List all available constraints for an organization.
    
    Essential for understanding available governance controls.
    """
    if not ORGPOLICY_AVAILABLE:
        return {
            "success": False,
            "error": "Google Cloud Organization Policy library not available"
        }
    
    try:
        client = orgpolicy_v2.OrgPolicyClient()
        
        # List constraints
        constraints = []
        request = orgpolicy_v2.ListConstraintsRequest(parent=parent)
        
        for constraint in client.list_constraints(request=request):
            constraint_info = {
                "name": constraint.name,
                "display_name": constraint.display_name,
                "description": constraint.description,
                "constraint_default": str(constraint.constraint_default),
                "supports_dry_run": constraint.supports_dry_run
            }
            
            # Add constraint type info
            if constraint.list_constraint:
                constraint_info["type"] = "list"
                constraint_info["supports_in_operator"] = constraint.list_constraint.supports_in
                constraint_info["supports_under_operator"] = constraint.list_constraint.supports_under
            elif constraint.boolean_constraint:
                constraint_info["type"] = "boolean"
            else:
                constraint_info["type"] = "unknown"
            
            constraints.append(constraint_info)
        
        # Categorize constraints
        categories = _categorize_constraints(constraints)
        
        return {
            "success": True,
            "parent": parent,
            "count": len(constraints),
            "constraints": constraints,
            "categories": categories
        }
        
    except Exception as e:
        logger.error(f"Failed to list constraints: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def get_policy(request: PolicyConstraintRequest) -> Dict[str, Any]:
    """
    Get the organization policy for a specific constraint.
    
    Shows current policy configuration and inheritance.
    """
    if not ORGPOLICY_AVAILABLE:
        return {
            "success": False,
            "error": "Google Cloud Organization Policy library not available"
        }
    
    try:
        client = orgpolicy_v2.OrgPolicyClient()
        
        # Build policy name
        policy_name = f"{request.parent}/policies/{request.constraint}"
        
        # Get the policy
        policy = client.get_policy(request={"name": policy_name})
        
        # Process policy details
        policy_info = {
            "name": policy.name,
            "parent": request.parent,
            "constraint": request.constraint,
            "inherit_from_parent": policy.spec.inherit_from_parent if policy.spec else False,
            "reset": policy.spec.reset if policy.spec else False,
            "etag": policy.spec.etag if policy.spec else None,
            "update_time": policy.spec.update_time.isoformat() if policy.spec and policy.spec.update_time else None,
            "rules": []
        }
        
        # Process rules
        if policy.spec and policy.spec.rules:
            for rule in policy.spec.rules:
                rule_info = _process_policy_rule(rule)
                policy_info["rules"].append(rule_info)
        
        # Check effective policy
        effective = await get_effective_policy(request)
        if effective.get("success"):
            policy_info["effective_policy"] = effective.get("policy")
        
        return {
            "success": True,
            "policy": policy_info
        }
        
    except exceptions.NotFound:
        return {
            "success": True,
            "policy": None,
            "message": f"No policy set for constraint {request.constraint}"
        }
    except Exception as e:
        logger.error(f"Failed to get policy: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def get_effective_policy(request: PolicyConstraintRequest) -> Dict[str, Any]:
    """
    Get the effective organization policy including inheritance.
    
    Shows what policy is actually applied after inheritance.
    """
    if not ORGPOLICY_AVAILABLE:
        return {
            "success": False,
            "error": "Google Cloud Organization Policy library not available"
        }
    
    try:
        client = orgpolicy_v2.OrgPolicyClient()
        
        # Build policy name
        policy_name = f"{request.parent}/policies/{request.constraint}"
        
        # Get effective policy
        effective_policy = client.get_effective_policy(request={"name": policy_name})
        
        # Process effective policy details
        policy_info = {
            "constraint": request.constraint,
            "rules": []
        }
        
        # Process rules
        if effective_policy.spec and effective_policy.spec.rules:
            for rule in effective_policy.spec.rules:
                rule_info = _process_policy_rule(rule)
                policy_info["rules"].append(rule_info)
        
        return {
            "success": True,
            "parent": request.parent,
            "policy": policy_info
        }
        
    except Exception as e:
        logger.error(f"Failed to get effective policy: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def create_or_update_policy(request: PolicyRequest) -> Dict[str, Any]:
    """
    Create or update an organization policy.
    
    Critical for enforcing governance and compliance.
    """
    if not ORGPOLICY_AVAILABLE:
        return {
            "success": False,
            "error": "Google Cloud Organization Policy library not available"
        }
    
    try:
        client = orgpolicy_v2.OrgPolicyClient()
        
        # Build policy
        policy = orgpolicy_v2.Policy()
        policy.name = f"{request.parent}/policies/{request.constraint}"
        
        # Build policy spec
        spec = orgpolicy_v2.PolicySpec()
        spec.inherit_from_parent = request.inherit_from_parent
        spec.reset = request.reset
        
        if request.etag:
            spec.etag = request.etag
        
        # Add rules
        for rule_request in request.rules:
            rule = orgpolicy_v2.PolicySpec.PolicyRule()
            
            # Set rule type
            if rule_request.values:
                rule.values = orgpolicy_v2.PolicySpec.PolicyRule.StringValues(**rule_request.values)
            elif rule_request.allow_all is not None:
                rule.allow_all = rule_request.allow_all
            elif rule_request.deny_all is not None:
                rule.deny_all = rule_request.deny_all
            elif rule_request.enforce is not None:
                rule.enforce = rule_request.enforce
            
            # Add condition if present
            if rule_request.condition:
                rule.condition = expr_pb2.Expr(**rule_request.condition)
            
            spec.rules.append(rule)
        
        policy.spec = spec
        
        # Create or update the policy
        updated_policy = client.update_policy(request={"policy": policy})
        
        return {
            "success": True,
            "policy": {
                "name": updated_policy.name,
                "constraint": request.constraint,
                "parent": request.parent,
                "update_time": updated_policy.spec.update_time.isoformat() if updated_policy.spec.update_time else None
            },
            "message": f"Policy for {request.constraint} updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create/update policy: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def delete_policy(request: PolicyConstraintRequest) -> Dict[str, Any]:
    """
    Delete an organization policy.
    
    Removes policy enforcement for a constraint.
    """
    if not ORGPOLICY_AVAILABLE:
        return {
            "success": False,
            "error": "Google Cloud Organization Policy library not available"
        }
    
    try:
        client = orgpolicy_v2.OrgPolicyClient()
        
        # Build policy name
        policy_name = f"{request.parent}/policies/{request.constraint}"
        
        # Delete the policy
        client.delete_policy(request={"name": policy_name})
        
        return {
            "success": True,
            "message": f"Policy for {request.constraint} deleted successfully",
            "parent": request.parent,
            "constraint": request.constraint
        }
        
    except Exception as e:
        logger.error(f"Failed to delete policy: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def create_custom_constraint(request: CustomConstraintRequest) -> Dict[str, Any]:
    """
    Create a custom constraint for organization-specific policies.
    
    Allows enforcement of custom governance rules.
    """
    if not ORGPOLICY_AVAILABLE:
        return {
            "success": False,
            "error": "Google Cloud Organization Policy library not available"
        }
    
    try:
        client = orgpolicy_v2.OrgPolicyClient()
        
        # Build custom constraint
        constraint = orgpolicy_v2.CustomConstraint()
        constraint.name = f"{request.parent}/customConstraints/{request.constraint_id}"
        constraint.display_name = request.display_name
        
        if request.description:
            constraint.description = request.description
        
        constraint.resource_types.extend(request.resource_types)
        constraint.method_types.extend([
            orgpolicy_v2.CustomConstraint.MethodType[mt] 
            for mt in request.method_types
        ])
        constraint.condition = request.condition
        
        # Set action type
        if request.action_type == "ALLOW":
            constraint.action_type = orgpolicy_v2.CustomConstraint.ActionType.ALLOW
        else:
            constraint.action_type = orgpolicy_v2.CustomConstraint.ActionType.DENY
        
        # Create the custom constraint
        created = client.create_custom_constraint(
            request={
                "parent": request.parent,
                "custom_constraint": constraint
            }
        )
        
        return {
            "success": True,
            "constraint": {
                "name": created.name,
                "display_name": created.display_name,
                "resource_types": list(created.resource_types),
                "method_types": [str(mt) for mt in created.method_types],
                "condition": created.condition
            },
            "message": f"Custom constraint {request.constraint_id} created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create custom constraint: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def list_custom_constraints(parent: str) -> Dict[str, Any]:
    """List all custom constraints in the organization."""
    if not ORGPOLICY_AVAILABLE:
        return {
            "success": False,
            "error": "Google Cloud Organization Policy library not available"
        }
    
    try:
        client = orgpolicy_v2.OrgPolicyClient()
        
        constraints = []
        request = orgpolicy_v2.ListCustomConstraintsRequest(parent=parent)
        
        for constraint in client.list_custom_constraints(request=request):
            constraints.append({
                "name": constraint.name,
                "display_name": constraint.display_name,
                "description": constraint.description,
                "resource_types": list(constraint.resource_types),
                "method_types": [str(mt) for mt in constraint.method_types],
                "condition": constraint.condition,
                "action_type": str(constraint.action_type)
            })
        
        return {
            "success": True,
            "parent": parent,
            "count": len(constraints),
            "custom_constraints": constraints
        }
        
    except Exception as e:
        logger.error(f"Failed to list custom constraints: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def analyze_org_policies(request: PolicyAnalysisRequest) -> Dict[str, Any]:
    """
    Analyze organization policies for compliance and coverage.
    
    Provides insights into policy effectiveness and gaps.
    """
    try:
        # Get all policies for the scope
        policies_result = await list_policies(request.scope)
        
        if not policies_result.get("success"):
            return policies_result
        
        policies = policies_result.get("policies", [])
        
        # Analyze policies
        analysis = {
            "total_policies": len(policies),
            "policy_coverage": {},
            "compliance_status": {},
            "recommendations": [],
            "policy_gaps": []
        }
        
        # Categorize policies
        categories = {
            "security": [],
            "compliance": [],
            "cost": [],
            "operational": []
        }
        
        for policy in policies:
            constraint = policy.get("constraint", "")
            
            # Categorize by constraint name
            if any(word in constraint.lower() for word in ["serial", "ssh", "oslogin", "firewall"]):
                categories["security"].append(constraint)
            elif any(word in constraint.lower() for word in ["audit", "log", "retention"]):
                categories["compliance"].append(constraint)
            elif any(word in constraint.lower() for word in ["vm", "machine", "quota"]):
                categories["cost"].append(constraint)
            else:
                categories["operational"].append(constraint)
        
        analysis["policy_coverage"] = {
            "security": len(categories["security"]),
            "compliance": len(categories["compliance"]),
            "cost": len(categories["cost"]),
            "operational": len(categories["operational"])
        }
        
        # Check for essential policies
        essential_policies = [
            "compute.disableSerialPortAccess",
            "compute.requireOsLogin",
            "compute.requireShieldedVm",
            "storage.uniformBucketLevelAccess",
            "iam.disableServiceAccountKeyCreation",
            "sql.restrictPublicIp"
        ]
        
        applied_policies = [p.get("constraint") for p in policies]
        
        for essential in essential_policies:
            if essential not in applied_policies:
                analysis["policy_gaps"].append({
                    "constraint": essential,
                    "severity": "HIGH",
                    "recommendation": f"Consider applying {essential} for improved security"
                })
        
        # Generate recommendations
        if len(categories["security"]) < 5:
            analysis["recommendations"].append(
                "Low security policy coverage. Review and apply additional security constraints."
            )
        
        if len(categories["compliance"]) < 3:
            analysis["recommendations"].append(
                "Consider adding audit and logging policies for compliance requirements."
            )
        
        # Calculate compliance score
        compliance_score = min(100, (len(policies) / len(essential_policies)) * 100)
        analysis["compliance_status"] = {
            "score": compliance_score,
            "rating": "GOOD" if compliance_score > 70 else "NEEDS_IMPROVEMENT" if compliance_score > 40 else "POOR"
        }
        
        return {
            "success": True,
            "scope": request.scope,
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze org policies: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def list_policies(parent: str) -> Dict[str, Any]:
    """List all policies applied to a resource."""
    if not ORGPOLICY_AVAILABLE:
        return {
            "success": False,
            "error": "Google Cloud Organization Policy library not available"
        }
    
    try:
        client = orgpolicy_v2.OrgPolicyClient()
        
        policies = []
        request = orgpolicy_v2.ListPoliciesRequest(parent=parent)
        
        for policy in client.list_policies(request=request):
            policy_info = {
                "name": policy.name,
                "constraint": policy.name.split("/policies/")[-1],
                "inherit_from_parent": policy.spec.inherit_from_parent if policy.spec else False,
                "reset": policy.spec.reset if policy.spec else False,
                "rules_count": len(policy.spec.rules) if policy.spec and policy.spec.rules else 0
            }
            policies.append(policy_info)
        
        return {
            "success": True,
            "parent": parent,
            "count": len(policies),
            "policies": policies
        }
        
    except Exception as e:
        logger.error(f"Failed to list policies: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# ============================================
# Helper Functions
# ============================================

def _process_policy_rule(rule) -> Dict[str, Any]:
    """Process a policy rule into a standardized format."""
    rule_info = {}
    
    # Check rule type
    if rule.values:
        rule_info["type"] = "list"
        rule_info["allowed_values"] = list(rule.values.allowed_values)
        rule_info["denied_values"] = list(rule.values.denied_values)
    elif hasattr(rule, 'allow_all') and rule.allow_all:
        rule_info["type"] = "allow_all"
        rule_info["allow_all"] = True
    elif hasattr(rule, 'deny_all') and rule.deny_all:
        rule_info["type"] = "deny_all"
        rule_info["deny_all"] = True
    elif hasattr(rule, 'enforce'):
        rule_info["type"] = "boolean"
        rule_info["enforce"] = rule.enforce
    
    # Add condition if present
    if rule.condition:
        rule_info["condition"] = {
            "expression": rule.condition.expression,
            "title": rule.condition.title,
            "description": rule.condition.description,
            "location": rule.condition.location
        }
    
    return rule_info

def _categorize_constraints(constraints: List[Dict]) -> Dict[str, List[str]]:
    """Categorize constraints by type."""
    categories = {
        "compute": [],
        "storage": [],
        "iam": [],
        "network": [],
        "sql": [],
        "resource_manager": [],
        "other": []
    }
    
    for constraint in constraints:
        name = constraint.get("name", "")
        
        if name.startswith("constraints/compute."):
            categories["compute"].append(name)
        elif name.startswith("constraints/storage."):
            categories["storage"].append(name)
        elif name.startswith("constraints/iam."):
            categories["iam"].append(name)
        elif name.startswith("constraints/compute.") and "network" in name.lower():
            categories["network"].append(name)
        elif name.startswith("constraints/sql."):
            categories["sql"].append(name)
        elif name.startswith("constraints/resourcemanager."):
            categories["resource_manager"].append(name)
        else:
            categories["other"].append(name)
    
    return categories