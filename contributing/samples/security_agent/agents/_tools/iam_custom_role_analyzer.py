"""
Custom IAM Role Analyzer
Matches custom roles to built-in roles and identifies permission gaps
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from google.cloud import iam_admin_v1
from google.cloud import bigquery
import json
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

class CustomRoleAnalyzer:
    """Analyzes custom IAM roles and matches them to built-in roles"""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.iam_client = iam_admin_v1.IAMClient()
        self.bq_client = bigquery.Client(project=project_id)
        self.dataset_id = "iam_analysis"
        self.table_id = "custom_role_analysis"

    def analyze_custom_role(self, custom_role_name: str) -> Dict:
        """
        Analyze a custom IAM role and find best matching built-in roles.

        Args:
            custom_role_name: Name of the custom role to analyze

        Returns:
            Analysis report with matches and recommendations
        """
        logger.info(f"Analyzing custom role: {custom_role_name}")

        # Fetch custom role details
        custom_role = self._get_custom_role(custom_role_name)
        if not custom_role:
            return {"error": f"Custom role {custom_role_name} not found"}

        custom_permissions = set(custom_role.included_permissions)

        # Get all built-in roles
        builtin_roles = self._get_builtin_roles()

        # Find best matches
        matches = self._find_best_matches(custom_permissions, builtin_roles)

        # Generate analysis
        analysis = {
            "custom_role": {
                "name": custom_role_name,
                "title": custom_role.title,
                "description": custom_role.description,
                "permission_count": len(custom_permissions),
                "permissions": sorted(list(custom_permissions))
            },
            "best_matches": matches[:5],
            "recommendations": self._generate_recommendations(
                custom_permissions, matches[0] if matches else None
            ),
            "security_assessment": self._assess_security_risks(custom_permissions),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }

        # Store in BigQuery
        self._store_analysis(analysis)

        return analysis

    def _get_custom_role(self, role_name: str) -> Optional[iam_admin_v1.Role]:
        """Fetch custom role from GCP"""
        try:
            if not role_name.startswith("projects/"):
                role_name = f"projects/{self.project_id}/roles/{role_name}"

            request = iam_admin_v1.GetRoleRequest(name=role_name)
            return self.iam_client.get_role(request=request)
        except Exception as e:
            logger.error(f"Failed to fetch custom role {role_name}: {e}")
            return None

    def _get_builtin_roles(self) -> List[iam_admin_v1.Role]:
        """Fetch all available built-in roles"""
        roles = []
        try:
            request = iam_admin_v1.ListRolesRequest(
                view=iam_admin_v1.RoleView.FULL
            )

            for role in self.iam_client.list_roles(request=request):
                if not role.name.startswith("projects/"):  # Only built-in roles
                    roles.append(role)

        except Exception as e:
            logger.error(f"Failed to fetch built-in roles: {e}")

        logger.info(f"Fetched {len(roles)} built-in roles")
        return roles

    def _find_best_matches(
        self,
        custom_permissions: Set[str],
        builtin_roles: List[iam_admin_v1.Role]
    ) -> List[Dict]:
        """Find built-in roles that best match the custom permissions"""
        matches = []

        for role in builtin_roles:
            if not hasattr(role, 'included_permissions'):
                continue

            builtin_permissions = set(role.included_permissions)

            # Calculate match metrics
            overlap = custom_permissions & builtin_permissions
            extra_in_custom = custom_permissions - builtin_permissions
            missing_from_custom = builtin_permissions - custom_permissions

            if len(overlap) == 0:
                continue

            # Calculate similarity score (Jaccard index)
            union = custom_permissions | builtin_permissions
            similarity = len(overlap) / len(union) if union else 0

            # Calculate coverage (what % of custom permissions are covered)
            coverage = len(overlap) / len(custom_permissions) if custom_permissions else 0

            matches.append({
                "role": role.name,
                "title": role.title,
                "description": role.description[:200] if role.description else "",
                "similarity_score": round(similarity * 100, 2),
                "coverage_score": round(coverage * 100, 2),
                "overlapping_permissions": sorted(list(overlap)),
                "overlap_count": len(overlap),
                "extra_permissions": sorted(list(extra_in_custom)),
                "extra_count": len(extra_in_custom),
                "missing_permissions": sorted(list(missing_from_custom)),
                "missing_count": len(missing_from_custom)
            })

        # Sort by similarity score descending
        matches.sort(key=lambda x: (x["similarity_score"], x["coverage_score"]), reverse=True)

        return matches

    def _generate_recommendations(
        self,
        custom_permissions: Set[str],
        best_match: Optional[Dict]
    ) -> Dict:
        """Generate recommendations based on analysis"""
        recommendations = {
            "summary": "",
            "actions": [],
            "alternative_approach": "",
            "risk_level": "low"
        }

        if not best_match:
            recommendations["summary"] = "No similar built-in roles found"
            recommendations["actions"].append(
                "Review if this custom role is necessary"
            )
            recommendations["risk_level"] = "high"
            return recommendations

        similarity = best_match["similarity_score"]
        coverage = best_match["coverage_score"]

        # High similarity - suggest using built-in role
        if similarity > 80:
            recommendations["summary"] = f"Consider using built-in role: {best_match['title']}"
            recommendations["actions"].append(
                f"Replace custom role with {best_match['role']}"
            )
            if best_match["extra_count"] > 0:
                recommendations["actions"].append(
                    f"Add {best_match['extra_count']} additional permissions if needed"
                )
            recommendations["risk_level"] = "low"

        # Medium similarity - suggest combination
        elif similarity > 50:
            recommendations["summary"] = "Consider using built-in role with modifications"
            recommendations["actions"].append(
                f"Use {best_match['role']} as base"
            )
            recommendations["actions"].append(
                "Create minimal custom role for additional permissions"
            )
            recommendations["alternative_approach"] = "Use multiple built-in roles instead"
            recommendations["risk_level"] = "medium"

        # Low similarity - keep custom but optimize
        else:
            recommendations["summary"] = "Custom role appears necessary"
            recommendations["actions"].append(
                "Review permissions for least privilege"
            )
            recommendations["actions"].append(
                "Document why custom role is required"
            )
            recommendations["risk_level"] = "low" if len(custom_permissions) < 50 else "medium"

        # Check for dangerous permissions
        dangerous_perms = self._check_dangerous_permissions(custom_permissions)
        if dangerous_perms:
            recommendations["actions"].append(
                f"Review dangerous permissions: {', '.join(dangerous_perms[:3])}"
            )
            recommendations["risk_level"] = "high"

        return recommendations

    def _assess_security_risks(self, permissions: Set[str]) -> Dict:
        """Assess security risks of the permission set"""
        risks = {
            "risk_score": 0,
            "risk_level": "low",
            "findings": [],
            "dangerous_permissions": []
        }

        # Define dangerous permissions
        dangerous_patterns = {
            "iam.serviceAccountKeys.create": "Can create service account keys",
            "iam.serviceAccounts.actAs": "Can impersonate service accounts",
            "resourcemanager.projects.setIamPolicy": "Can modify project IAM",
            "iam.roles.create": "Can create new roles",
            "iam.roles.delete": "Can delete roles",
            "compute.instances.setMetadata": "Can modify instance metadata",
            "storage.buckets.setIamPolicy": "Can modify bucket IAM",
            "owner": "Has owner-level access",
            "editor": "Has editor-level access"
        }

        # Check for dangerous permissions
        for perm in permissions:
            for pattern, description in dangerous_patterns.items():
                if pattern in perm.lower():
                    risks["dangerous_permissions"].append(perm)
                    risks["findings"].append(f"{perm}: {description}")
                    risks["risk_score"] += 20

        # Assess based on permission count
        if len(permissions) > 100:
            risks["findings"].append(f"Excessive permissions: {len(permissions)}")
            risks["risk_score"] += 30
        elif len(permissions) > 50:
            risks["findings"].append(f"Many permissions: {len(permissions)}")
            risks["risk_score"] += 15

        # Check for wildcards
        wildcard_perms = [p for p in permissions if '*' in p]
        if wildcard_perms:
            risks["findings"].append(f"Contains wildcard permissions: {len(wildcard_perms)}")
            risks["risk_score"] += 25

        # Set risk level based on score
        if risks["risk_score"] >= 70:
            risks["risk_level"] = "high"
        elif risks["risk_score"] >= 40:
            risks["risk_level"] = "medium"
        else:
            risks["risk_level"] = "low"

        return risks

    def _check_dangerous_permissions(self, permissions: Set[str]) -> List[str]:
        """Check for particularly dangerous permissions"""
        dangerous = []
        danger_keywords = [
            "setIamPolicy", "actAs", "create", "delete",
            "admin", "owner", "editor", "impersonate"
        ]

        for perm in permissions:
            if any(keyword in perm for keyword in danger_keywords):
                dangerous.append(perm)

        return dangerous

    def _store_analysis(self, analysis: Dict):
        """Store analysis results in BigQuery"""
        try:
            # Ensure dataset exists
            dataset_ref = f"{self.project_id}.{self.dataset_id}"
            try:
                self.bq_client.get_dataset(dataset_ref)
            except:
                dataset = bigquery.Dataset(dataset_ref)
                dataset.location = "US"
                self.bq_client.create_dataset(dataset, timeout=30)

            # Prepare row for insertion
            row = {
                "analysis_id": hashlib.md5(
                    f"{analysis['custom_role']['name']}-{analysis['analysis_timestamp']}".encode()
                ).hexdigest(),
                "custom_role_name": analysis["custom_role"]["name"],
                "custom_role_title": analysis["custom_role"]["title"],
                "permission_count": analysis["custom_role"]["permission_count"],
                "best_match_role": analysis["best_matches"][0]["role"] if analysis["best_matches"] else None,
                "best_match_similarity": analysis["best_matches"][0]["similarity_score"] if analysis["best_matches"] else 0,
                "recommendations": json.dumps(analysis["recommendations"]),
                "security_assessment": json.dumps(analysis["security_assessment"]),
                "full_analysis": json.dumps(analysis),
                "analysis_timestamp": analysis["analysis_timestamp"]
            }

            # Insert into BigQuery
            table_ref = f"{dataset_ref}.{self.table_id}"
            errors = self.bq_client.insert_rows_json(
                table_ref, [row], ignore_unknown_values=True
            )

            if errors:
                logger.error(f"Failed to store analysis: {errors}")
            else:
                logger.info(f"Analysis stored in BigQuery: {row['analysis_id']}")

        except Exception as e:
            logger.error(f"Failed to store analysis in BigQuery: {e}")


def analyze_all_custom_roles(project_id: str) -> List[Dict]:
    """
    Analyze all custom roles in a project.

    Args:
        project_id: GCP project ID

    Returns:
        List of analysis results
    """
    analyzer = CustomRoleAnalyzer(project_id)
    results = []

    try:
        # List all custom roles in project
        iam_client = iam_admin_v1.IAMClient()
        request = iam_admin_v1.ListRolesRequest(
            parent=f"projects/{project_id}",
            view=iam_admin_v1.RoleView.BASIC
        )

        for role in iam_client.list_roles(request=request):
            if role.name.startswith(f"projects/{project_id}/roles/"):
                logger.info(f"Analyzing role: {role.name}")
                analysis = analyzer.analyze_custom_role(role.name)
                results.append(analysis)

    except Exception as e:
        logger.error(f"Failed to analyze custom roles: {e}")

    return results


# For ADK tool integration
def analyze_custom_role_tool(role_name: str, project_id: Optional[str] = None) -> Dict:
    """
    ADK tool function for analyzing custom roles.

    Args:
        role_name: Name of the custom role
        project_id: Optional project ID (uses default if not provided)

    Returns:
        Analysis results
    """
    import os
    if not project_id:
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")

    analyzer = CustomRoleAnalyzer(project_id)
    return analyzer.analyze_custom_role(role_name)