"""
Custom IAM roles fetcher module
"""

from typing import List, Dict, Any
from datetime import datetime
from google.cloud import bigquery
import json

from .base import BaseFetcher
from shared import get_authenticated_client, Config


class CustomRolesFetcher(BaseFetcher):
    """Fetcher for custom IAM roles"""

    @property
    def table_name(self) -> str:
        return 'iam_custom_roles'

    @property
    def schema(self) -> List[bigquery.SchemaField]:
        return [
            bigquery.SchemaField("role_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("role_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("title", "STRING"),
            bigquery.SchemaField("description", "STRING"),
            bigquery.SchemaField("stage", "STRING"),
            bigquery.SchemaField("deleted", "BOOLEAN"),
            bigquery.SchemaField("included_permissions", "JSON"),
            bigquery.SchemaField("permission_count", "INTEGER"),
            bigquery.SchemaField("high_risk_permissions", "INTEGER"),
            bigquery.SchemaField("risk_level", "STRING"),
            bigquery.SchemaField("similar_predefined_roles", "JSON"),
            bigquery.SchemaField("created_time", "TIMESTAMP"),
            bigquery.SchemaField("project_id", "STRING"),
            bigquery.SchemaField("ingestion_time", "TIMESTAMP"),
            bigquery.SchemaField("last_refreshed", "TIMESTAMP")
        ]

    def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch custom roles from IAM"""
        roles = []

        try:
            from google.cloud import iam_admin_v1

            iam_client = get_authenticated_client('iam')

            # List custom roles
            parent = f"projects/{Config.PROJECT_ID}"
            list_request = iam_admin_v1.ListRolesRequest(
                parent=parent,
                view=iam_admin_v1.RoleView.FULL,
                show_deleted=False
            )

            for role in iam_client.list_roles(request=list_request):
                roles.append(self._convert_role_to_dict(role))

        except Exception as e:
            self.logger.warning(f"Failed to fetch custom roles: {e}")
            # Return sample data if enabled
            if Config.ENABLE_SAMPLE_DATA:
                roles = self.get_sample_data()

        return roles

    def _convert_role_to_dict(self, role) -> Dict[str, Any]:
        """Convert a role object to dictionary"""
        role_id = role.name.split("/")[-1]
        permissions_list = list(role.included_permissions) if role.included_permissions else []

        # Analyze permissions for risk
        high_risk_count = sum(1 for p in permissions_list if self._is_high_risk_permission(p))
        risk_level = self._calculate_risk_level(permissions_list, high_risk_count)

        # Find similar predefined roles
        similar_roles = self._find_similar_predefined_roles(permissions_list)

        return {
            "role_id": role_id,
            "role_name": role.name,
            "title": role.title or "",
            "description": role.description or "",
            "stage": role.stage.name if role.stage else "ALPHA",
            "deleted": role.deleted,
            "included_permissions": json.dumps(permissions_list),
            "permission_count": len(permissions_list),
            "high_risk_permissions": high_risk_count,
            "risk_level": risk_level,
            "similar_predefined_roles": json.dumps(similar_roles),
            "created_time": None,
            "last_refreshed": datetime.utcnow().isoformat()
        }

    def _is_high_risk_permission(self, permission: str) -> bool:
        """Check if a permission is high risk"""
        high_risk_patterns = [
            "delete", "setiampolicy", "admin", "actas",
            "create", "update", "impersonate"
        ]
        perm = permission.lower()
        return any(pattern in perm for pattern in high_risk_patterns)

    def _calculate_risk_level(self, permissions: List[str], high_risk_count: int) -> str:
        """Calculate overall risk level for a role"""
        if high_risk_count == 0:
            return "LOW"
        elif high_risk_count <= 2:
            return "MEDIUM"
        elif high_risk_count <= 5:
            return "HIGH"
        else:
            return "CRITICAL"

    def _find_similar_predefined_roles(self, permissions: List[str]) -> List[Dict[str, Any]]:
        """Find predefined roles similar to the custom role"""
        role_patterns = {
            "roles/viewer": ["get", "list"],
            "roles/editor": ["get", "list", "create", "update", "delete"],
            "roles/owner": ["get", "list", "create", "update", "delete", "setIamPolicy"],
            "roles/storage.admin": ["storage", "bucket", "object"],
            "roles/compute.admin": ["compute", "instance", "disk", "network"],
            "roles/iam.securityReviewer": ["iam", "list", "get"],
            "roles/bigquery.dataViewer": ["bigquery", "get", "list", "data"],
            "roles/bigquery.dataEditor": ["bigquery", "create", "update", "delete"]
        }

        similar_roles = []
        for role, patterns in role_patterns.items():
            match_count = sum(1 for p in patterns if any(p in perm.lower() for perm in permissions))
            if match_count > 0:
                similarity = (match_count / len(patterns)) * 100
                similar_roles.append({
                    "role": role,
                    "similarity_percentage": round(similarity, 2)
                })

        return sorted(similar_roles, key=lambda x: x["similarity_percentage"], reverse=True)[:3]

    def get_sample_data(self) -> List[Dict[str, Any]]:
        """Get sample custom roles data"""
        return [
            {
                "role_id": "CustomDataAnalyst",
                "role_name": f"projects/{Config.PROJECT_ID}/roles/CustomDataAnalyst",
                "title": "Custom Data Analyst",
                "description": "Custom role for data analysts with BigQuery access",
                "stage": "GA",
                "deleted": False,
                "included_permissions": json.dumps([
                    "bigquery.datasets.get",
                    "bigquery.tables.get",
                    "bigquery.tables.list",
                    "bigquery.jobs.create"
                ]),
                "permission_count": 4,
                "high_risk_permissions": 1,
                "risk_level": "MEDIUM",
                "similar_predefined_roles": json.dumps([
                    {"role": "roles/bigquery.dataViewer", "similarity_percentage": 75.0}
                ]),
                "created_time": None,
                "last_refreshed": datetime.utcnow().isoformat()
            }
        ]
