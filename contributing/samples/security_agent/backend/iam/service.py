"""IAM Policy Analyzer Service for security best practices validation."""

import logging
from typing import Dict, List, Any, Optional
from google.auth import default
from google.cloud import resourcemanager_v3


logger = logging.getLogger(__name__)

class IAMPolicyAnalyzer:
    """Analyze IAM policies against security best practices."""
    
    def __init__(self):
        self.security_rules = self._load_security_rules()
    
    def _load_security_rules(self) -> Dict[str, Any]:
        """Load security best practices rules."""
        return {
            "high_risk_roles": [
                "roles/owner",
                "roles/editor", 
                "roles/iam.securityAdmin",
                "roles/iam.serviceAccountAdmin",
                "roles/resourcemanager.organizationAdmin",
                "roles/billing.admin"
            ],
            "overprivileged_patterns": [
                "roles/*Admin",
                "roles/*.admin",
                "roles/owner",
                "roles/editor"
            ],
            "service_account_risks": [
                "roles/iam.serviceAccountTokenCreator",
                "roles/iam.serviceAccountUser",
                "roles/iam.serviceAccountActor"
            ],
            "best_practices": {
                "principle_of_least_privilege": {
                    "description": "Users should have minimum permissions necessary",
                    "violations": ["roles/owner", "roles/editor", "*Admin"]
                },
                "no_primitive_roles": {
                    "description": "Avoid primitive roles (Owner, Editor, Viewer) in production",
                    "violations": ["roles/owner", "roles/editor"]
                },
                "service_account_security": {
                    "description": "Service account permissions should be tightly controlled",
                    "violations": ["roles/iam.serviceAccountTokenCreator"]
                },
                "regular_access_review": {
                    "description": "User access should be reviewed regularly",
                    "check": "manual_review_required"
                }
            },
            "compliance_frameworks": {
                "SOC2": {
                    "requirements": [
                        "Access control policies",
                        "Regular access reviews", 
                        "Principle of least privilege"
                    ]
                },
                "ISO27001": {
                    "requirements": [
                        "Access management policy",
                        "Privileged access management",
                        "Access rights review"
                    ]
                }
            }
        }
    
    def analyze_user_permissions(self, project_id: str, user_email: str) -> Dict[str, Any]:
        """Analyze a user's IAM permissions against security best practices."""
        try:
            # Get IAM policy for the project
            iam_policy = self._get_project_iam_policy(project_id)
            
            # Extract user's roles
            user_roles = self._extract_user_roles(iam_policy, user_email)
            
            # Analyze against security rules
            analysis = self._analyze_roles_against_rules(user_roles)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(user_roles, analysis)
            
            return {
                "success": True,
                "user_email": user_email,
                "project_id": project_id,
                "roles": user_roles,
                "security_analysis": analysis,
                "recommendations": recommendations,
                "risk_score": self._calculate_risk_score(analysis)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing user permissions: {e}")
            return {
                "success": False,
                "error": str(e),
                "user_email": user_email,
                "project_id": project_id
            }
    
    def _get_project_iam_policy(self, project_id: str) -> Dict[str, Any]:
        """Get IAM policy for a project using the Resource Manager API client library."""
        try:
            client = resourcemanager_v3.ProjectsClient()
            policy = client.get_iam_policy(project=f"projects/{project_id}")
            
            # Convert policy to a dictionary for consistency with existing code
            return self._policy_to_dict(policy)
            
        except Exception as e:
            logger.error(f"Error getting IAM policy for project {project_id}: {e}")
            raise Exception(f"Failed to get IAM policy: {e}")

    def _policy_to_dict(self, policy) -> Dict[str, Any]:
        """Converts a Policy object to a dictionary."""
        policy_dict = {
            "version": policy.version,
            "bindings": []
        }
        for binding in policy.bindings:
            policy_dict["bindings"].append({
                "role": binding.role,
                "members": list(binding.members)
            })
        if policy.etag:
            policy_dict["etag"] = policy.etag.decode('utf-8')
        return policy_dict
    
    def _extract_user_roles(self, iam_policy: Dict[str, Any], user_email: str) -> List[str]:
        """Extract roles assigned to a specific user."""
        user_roles = []
        user_member = f"user:{user_email}"
        
        for binding in iam_policy.get("bindings", []):
            if user_member in binding.get("members", []):
                user_roles.append(binding.get("role"))
        
        return user_roles
    
    def _analyze_roles_against_rules(self, roles: List[str]) -> Dict[str, Any]:
        """Analyze roles against security best practices."""
        analysis = {
            "high_risk_roles": [],
            "overprivileged_roles": [],
            "service_account_risks": [],
            "violations": {},
            "compliance_issues": {}
        }
        
        for role in roles:
            # Check for high-risk roles
            if role in self.security_rules["high_risk_roles"]:
                analysis["high_risk_roles"].append({
                    "role": role,
                    "reason": "High-privilege role with broad access"
                })
            
            # Check for overprivileged patterns
            for pattern in self.security_rules["overprivileged_patterns"]:
                if pattern.endswith("*") and role.startswith(pattern[:-1]):
                    analysis["overprivileged_roles"].append({
                        "role": role,
                        "pattern": pattern,
                        "reason": "Matches overprivileged pattern"
                    })
                elif role == pattern:
                    analysis["overprivileged_roles"].append({
                        "role": role,
                        "pattern": pattern,
                        "reason": "Exact match for overprivileged role"
                    })
            
            # Check service account risks
            if role in self.security_rules["service_account_risks"]:
                analysis["service_account_risks"].append({
                    "role": role,
                    "reason": "Can impersonate or act as service accounts"
                })
        
        # Check best practices violations
        for practice, config in self.security_rules["best_practices"].items():
            violations = []
            for violation_pattern in config.get("violations", []):
                for role in roles:
                    if violation_pattern.endswith("*") and role.startswith(violation_pattern[:-1]):
                        violations.append(role)
                    elif role == violation_pattern:
                        violations.append(role)
            
            if violations:
                analysis["violations"][practice] = {
                    "description": config["description"],
                    "violated_roles": violations
                }
        
        return analysis
    
    def _generate_recommendations(self, roles: List[str], analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate security recommendations based on analysis."""
        recommendations = []
        
        # High-risk role recommendations
        if analysis["high_risk_roles"]:
            recommendations.append({
                "type": "high_priority",
                "title": "Remove High-Risk Roles",
                "description": "User has high-privilege roles that should be avoided",
                "affected_roles": [r["role"] for r in analysis["high_risk_roles"]],
                "action": "Replace with least-privilege custom roles"
            })
        
        # Overprivileged recommendations
        if analysis["overprivileged_roles"]:
            recommendations.append({
                "type": "medium_priority", 
                "title": "Reduce Overprivileged Access",
                "description": "User has roles with broader permissions than typically needed",
                "affected_roles": [r["role"] for r in analysis["overprivileged_roles"]],
                "action": "Review actual usage and create custom roles with minimal permissions"
            })
        
        # Service account security
        if analysis["service_account_risks"]:
            recommendations.append({
                "type": "high_priority",
                "title": "Review Service Account Permissions", 
                "description": "User can impersonate service accounts, which poses security risks",
                "affected_roles": [r["role"] for r in analysis["service_account_risks"]],
                "action": "Restrict service account impersonation to specific accounts only"
            })
        
        # Best practices violations
        for practice, violation in analysis["violations"].items():
            recommendations.append({
                "type": "medium_priority",
                "title": f"Best Practice: {practice.replace('_', ' ').title()}",
                "description": violation["description"],
                "affected_roles": violation["violated_roles"],
                "action": "Follow principle of least privilege and use custom roles"
            })
        
        # Add general recommendations
        if not recommendations:
            recommendations.append({
                "type": "info",
                "title": "Security Review",
                "description": "User permissions appear to follow security best practices",
                "action": "Continue regular access reviews"
            })
        
        return recommendations
    
    def _calculate_risk_score(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk score based on analysis."""
        score = 0
        max_score = 100
        
        # High-risk roles (30 points each)
        score += len(analysis["high_risk_roles"]) * 30
        
        # Overprivileged roles (15 points each)  
        score += len(analysis["overprivileged_roles"]) * 15
        
        # Service account risks (25 points each)
        score += len(analysis["service_account_risks"]) * 25
        
        # Cap at max score
        score = min(score, max_score)
        
        # Determine risk level
        if score >= 70:
            risk_level = "HIGH"
        elif score >= 40:
            risk_level = "MEDIUM"
        elif score >= 20:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        return {
            "score": score,
            "max_score": max_score,
            "risk_level": risk_level,
            "description": self._get_risk_description(risk_level)
        }
    
    def _get_risk_description(self, risk_level: str) -> str:
        """Get description for risk level."""
        descriptions = {
            "HIGH": "User has multiple high-privilege roles that pose significant security risks",
            "MEDIUM": "User has some overprivileged access that should be reviewed",
            "LOW": "User has minor privilege issues that should be addressed",
            "MINIMAL": "User permissions generally follow security best practices"
        }
        return descriptions.get(risk_level, "Unknown risk level")
    
    def analyze_all_users(self, project_id: str) -> Dict[str, Any]:
        """Analyze all users in a project."""
        try:
            # Get IAM policy
            iam_policy = self._get_project_iam_policy(project_id)
            
            # Extract all users
            users = set()
            for binding in iam_policy.get("bindings", []):
                for member in binding.get("members", []):
                    if member.startswith("user:"):
                        users.add(member[5:])  # Remove 'user:' prefix
            
            # Analyze each user
            user_analyses = {}
            for user in users:
                user_analyses[user] = self.analyze_user_permissions(project_id, user)
            
            # Generate summary
            summary = self._generate_project_summary(user_analyses)
            
            return {
                "success": True,
                "project_id": project_id,
                "total_users": len(users),
                "user_analyses": user_analyses,
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Error analyzing all users: {e}")
            return {
                "success": False,
                "error": str(e),
                "project_id": project_id
            }
    
    def _generate_project_summary(self, user_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Generate project-wide security summary."""
        high_risk_users = []
        medium_risk_users = []
        total_violations = 0
        
        for user, analysis in user_analyses.items():
            if not analysis.get("success"):
                continue
                
            risk_level = analysis.get("risk_score", {}).get("risk_level", "MINIMAL")
            if risk_level == "HIGH":
                high_risk_users.append(user)
            elif risk_level == "MEDIUM":
                medium_risk_users.append(user)
            
            total_violations += len(analysis.get("security_analysis", {}).get("violations", {}))
        
        return {
            "high_risk_users": high_risk_users,
            "medium_risk_users": medium_risk_users,
            "total_violations": total_violations,
            "users_needing_review": len(high_risk_users) + len(medium_risk_users),
            "overall_risk": "HIGH" if high_risk_users else "MEDIUM" if medium_risk_users else "LOW"
        }