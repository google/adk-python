"""IAM related API endpoints."""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

from .service import IAMPolicyAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/project/{project_id}/analyze-user/{user_email}")
async def analyze_user_iam_permissions(project_id: str, user_email: str) -> Dict[str, Any]:
    """Analyze a user's IAM permissions against security best practices."""
    try:
        analyzer = IAMPolicyAnalyzer()
        result = analyzer.analyze_user_permissions(project_id, user_email)
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing user IAM permissions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/project/{project_id}/analyze-all-users")
async def analyze_all_users_iam_permissions(project_id: str) -> Dict[str, Any]:
    """Analyze all users' IAM permissions in a project against security best practices."""
    try:
        analyzer = IAMPolicyAnalyzer()
        result = analyzer.analyze_all_users(project_id)
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing all users IAM permissions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/project/{project_id}/policy")
async def get_project_iam_policy(project_id: str) -> Dict[str, Any]:
    """Get the full IAM policy for a project."""
    try:
        analyzer = IAMPolicyAnalyzer()
        iam_policy = analyzer._get_project_iam_policy(project_id)
        
        return {
            "success": True,
            "project_id": project_id,
            "iam_policy": iam_policy
        }
        
    except Exception as e:
        logger.error(f"Error getting IAM policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/testing/scenarios")
async def get_iam_testing_scenarios() -> Dict[str, Any]:
    """Get predefined IAM testing scenarios for the security bot."""
    scenarios = [
        {
            "id": "overprivileged_users",
            "title": "Detect Overprivileged Users",
            "description": "Find users with high-risk roles like Owner, Editor, or broad Admin access",
            "prompt": "Analyze the project for users with overprivileged access. Focus on finding users with roles/owner, roles/editor, or any *Admin roles. Provide specific recommendations for each user.",
            "category": "high_priority",
            "complexity": "medium"
        },
        {
            "id": "service_account_risks",
            "title": "Service Account Security Review",
            "description": "Identify users who can impersonate service accounts or create tokens",
            "prompt": "Check for users with dangerous service account permissions like roles/iam.serviceAccountTokenCreator, roles/iam.serviceAccountUser, or roles/iam.serviceAccountActor. Explain the security risks and suggest mitigations.",
            "category": "high_priority",
            "complexity": "high"
        },
        {
            "id": "primitive_roles_audit",
            "title": "Primitive Roles Audit",
            "description": "Scan for usage of primitive roles (Owner/Editor/Viewer) in production",
            "prompt": "Perform an audit for primitive roles usage. List all users with roles/owner, roles/editor, or roles/viewer and recommend custom roles with minimal permissions instead.",
            "category": "medium_priority",
            "complexity": "low"
        },
        {
            "id": "compliance_check",
            "title": "SOC2/ISO27001 Compliance Check",
            "description": "Validate IAM configuration against compliance frameworks",
            "prompt": "Evaluate the current IAM configuration against SOC2 and ISO27001 requirements. Check for proper access control policies, regular access reviews, and principle of least privilege implementation.",
            "category": "compliance",
            "complexity": "high"
        },
        {
            "id": "external_users_review",
            "title": "External Users Access Review",
            "description": "Review access granted to external users and contractors",
            "prompt": "Identify external users (non-company email domains) and review their access levels. Recommend temporary access patterns and regular review cycles for external contributors.",
            "category": "medium_priority",
            "complexity": "medium"
        },
        {
            "id": "unused_permissions_analysis",
            "title": "Unused Permissions Analysis",
            "description": "Find users with permissions they haven't used recently",
            "prompt": "Analyze user permissions and identify potentially unused access rights. Suggest a process for removing stale permissions and implementing just-in-time access.",
            "category": "optimization",
            "complexity": "high"
        }
    ]
    
    return {
        "success": True,
        "scenarios": scenarios,
        "total_scenarios": len(scenarios)
    }


@router.post("/testing/run-scenario/{scenario_id}")
async def run_iam_testing_scenario(scenario_id: str, project_id: str) -> Dict[str, Any]:
    """Run a specific IAM testing scenario on a project."""
    try:
        # Get the scenario details
        scenarios_response = await get_iam_testing_scenarios()
        scenarios = scenarios_response["scenarios"]
        
        scenario = next((s for s in scenarios if s["id"] == scenario_id), None)
        if not scenario:
            raise HTTPException(status_code=404, detail=f"Scenario {scenario_id} not found")
        
        # Initialize analyzer
        analyzer = IAMPolicyAnalyzer()
        
        # Run comprehensive analysis
        all_users_analysis = analyzer.analyze_all_users(project_id)
        
        if not all_users_analysis["success"]:
            return all_users_analysis
        
        # Filter results based on scenario
        filtered_results = _filter_results_by_scenario(
            scenario_id, 
            all_users_analysis,
            scenario
        )
        
        return {
            "success": True,
            "scenario": scenario,
            "project_id": project_id,
            "results": filtered_results,
            "bot_prompt": scenario["prompt"],
            "analysis_summary": _generate_scenario_summary(scenario_id, filtered_results)
        }
        
    except Exception as e:
        logger.error(f"Error running IAM scenario {scenario_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _filter_results_by_scenario(scenario_id: str, analysis: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Filter analysis results based on the specific scenario."""
    user_analyses = analysis.get("user_analyses", {})
    
    if scenario_id == "overprivileged_users":
        return {
            "high_risk_users": [
                {"user": user, "analysis": data} 
                for user, data in user_analyses.items()
                if data.get("success") and (
                    data.get("security_analysis", {}).get("high_risk_roles") or
                    data.get("security_analysis", {}).get("overprivileged_roles")
                )
            ]
        }
    
    elif scenario_id == "service_account_risks":
        return {
            "risky_users": [
                {"user": user, "analysis": data}
                for user, data in user_analyses.items()
                if data.get("success") and data.get("security_analysis", {}).get("service_account_risks")
            ]
        }
    
    elif scenario_id == "primitive_roles_audit":
        primitive_roles = ["roles/owner", "roles/editor", "roles/viewer"]
        return {
            "users_with_primitive_roles": [
                {"user": user, "roles": data.get("roles", []), "analysis": data}
                for user, data in user_analyses.items()
                if data.get("success") and any(role in primitive_roles for role in data.get("roles", []))
            ]
        }
    
    elif scenario_id == "compliance_check":
        return {
            "compliance_violations": [
                {"user": user, "violations": data.get("security_analysis", {}).get("violations", {}), "analysis": data}
                for user, data in user_analyses.items()
                if data.get("success") and data.get("security_analysis", {}).get("violations")
            ]
        }
    
    elif scenario_id == "external_users_review":
        company_domains = ["google.com", "googlemail.com"]  # Add your company domains
        return {
            "external_users": [
                {"user": user, "analysis": data}
                for user, data in user_analyses.items()
                if data.get("success") and not any(domain in user for domain in company_domains)
            ]
        }
    
    else:  # Default: return all data
        return {"all_users": user_analyses}


def _generate_scenario_summary(scenario_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a summary for the scenario results."""
    if scenario_id == "overprivileged_users":
        high_risk_count = len(results.get("high_risk_users", []))
        return {
            "total_high_risk_users": high_risk_count,
            "severity": "HIGH" if high_risk_count > 0 else "LOW",
            "recommendation": "Review and remediate overprivileged access immediately" if high_risk_count > 0 else "No overprivileged users found"
        }
    
    elif scenario_id == "service_account_risks":
        risky_count = len(results.get("risky_users", []))
        return {
            "users_with_sa_risks": risky_count,
            "severity": "CRITICAL" if risky_count > 0 else "LOW",
            "recommendation": "Restrict service account impersonation permissions" if risky_count > 0 else "Service account security looks good"
        }
    
    elif scenario_id == "primitive_roles_audit":
        primitive_users = len(results.get("users_with_primitive_roles", []))
        return {
            "users_with_primitive_roles": primitive_users,
            "severity": "MEDIUM" if primitive_users > 0 else "LOW",
            "recommendation": "Replace primitive roles with custom roles" if primitive_users > 0 else "No primitive roles found"
        }
    
    return {"message": "Analysis completed successfully"}