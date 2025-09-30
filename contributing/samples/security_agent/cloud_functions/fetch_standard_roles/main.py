#!/usr/bin/env python3
"""
Cloud Function to fetch and analyze predefined GCP IAM roles.
- Lists all available predefined roles in the project
- Extracts permissions for each role
- Categories roles by service
- Identifies high-privilege roles
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any
import functions_framework


def categorize_role(role_name: str) -> Dict[str, Any]:
    """Categorize a role based on its name"""
    # Extract service from role name
    parts = role_name.replace("roles/", "").split(".")

    if len(parts) == 1:
        # Primitive role (owner, editor, viewer)
        return {
            "category": "PRIMITIVE",
            "service": "core",
            "resource_type": "project"
        }

    service = parts[0]
    resource_type = parts[1] if len(parts) > 1 else "unknown"

    # Determine category
    category = "SERVICE_SPECIFIC"
    if "admin" in role_name.lower():
        category = "ADMIN"
    elif "viewer" in role_name.lower() or "reader" in role_name.lower():
        category = "READ_ONLY"
    elif "editor" in role_name.lower() or "writer" in role_name.lower():
        category = "WRITE"

    return {
        "category": category,
        "service": service,
        "resource_type": resource_type
    }


def analyze_permissions(permissions: List[str]) -> Dict[str, Any]:
    """Analyze permissions to determine capabilities and risk"""
    if not permissions:
        return {
            "total_permissions": 0,
            "high_risk_count": 0,
            "services_accessed": [],
            "capabilities": []
        }

    high_risk_verbs = ["delete", "setIamPolicy", "create", "update", "admin", "actAs"]
    services = set()
    capabilities = set()
    high_risk_count = 0

    for permission in permissions:
        parts = permission.split(".")
        if len(parts) > 0:
            services.add(parts[0])

        # Determine capability
        verb = parts[-1] if len(parts) > 0 else ""
        if "get" in verb or "list" in verb:
            capabilities.add("READ")
        if "create" in verb or "insert" in verb:
            capabilities.add("CREATE")
        if "update" in verb or "patch" in verb:
            capabilities.add("UPDATE")
        if "delete" in verb or "remove" in verb:
            capabilities.add("DELETE")
        if "setIamPolicy" in verb:
            capabilities.add("IAM_ADMIN")

        # Check for high risk
        if any(risk in permission.lower() for risk in high_risk_verbs):
            high_risk_count += 1

    return {
        "total_permissions": len(permissions),
        "high_risk_count": high_risk_count,
        "services_accessed": sorted(list(services)),
        "capabilities": sorted(list(capabilities))
    }


@functions_framework.http
def fetch_standard_roles(request):
    """Main Cloud Function to fetch predefined roles"""
    from google.cloud import iam_admin_v1
    from google.cloud import bigquery

    project_id = os.environ.get('PROJECT_ID', 'mgm-digitalconcierge')
    dataset_id = os.environ.get('BQ_DATASET_ID', 'security_insights')

    try:
        # Initialize clients
        iam_client = iam_admin_v1.IAMClient()
        bq_client = bigquery.Client()

        print(f"Fetching predefined roles for project: {project_id}")

        # Ensure dataset exists
        dataset_ref = bq_client.dataset(dataset_id)
        try:
            bq_client.get_dataset(dataset_ref)
        except:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"
            bq_client.create_dataset(dataset)
            print(f"Created dataset {dataset_id}")

        # Define table schema
        schema = [
            bigquery.SchemaField("role_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("title", "STRING"),
            bigquery.SchemaField("description", "STRING"),
            bigquery.SchemaField("stage", "STRING"),
            bigquery.SchemaField("category", "STRING"),  # PRIMITIVE, ADMIN, READ_ONLY, WRITE, SERVICE_SPECIFIC
            bigquery.SchemaField("service", "STRING"),
            bigquery.SchemaField("resource_type", "STRING"),
            bigquery.SchemaField("included_permissions", "JSON"),
            bigquery.SchemaField("permission_count", "INTEGER"),
            bigquery.SchemaField("high_risk_permissions", "INTEGER"),
            bigquery.SchemaField("services_accessed", "JSON"),
            bigquery.SchemaField("capabilities", "JSON"),
            bigquery.SchemaField("is_admin", "BOOLEAN"),
            bigquery.SchemaField("is_primitive", "BOOLEAN"),
            bigquery.SchemaField("project_id", "STRING"),
            bigquery.SchemaField("last_refreshed", "TIMESTAMP")
        ]

        # Create or update table
        table_id = f"{project_id}.{dataset_id}.standard_roles"
        table = bigquery.Table(table_id, schema=schema)
        table = bq_client.create_table(table, exists_ok=True)

        # Fetch predefined roles
        standard_roles_data = []

        # Query for all available roles (predefined and curated)
        list_request = iam_admin_v1.QueryGrantableRolesRequest(
            full_resource_name=f"//cloudresourcemanager.googleapis.com/projects/{project_id}"
        )

        # Get grantable roles
        roles_response = iam_client.query_grantable_roles(request=list_request)

        # Also get all predefined roles
        list_roles_request = iam_admin_v1.ListRolesRequest(
            view=iam_admin_v1.RoleView.FULL
        )

        all_roles = iam_client.list_roles(request=list_roles_request)

        # Process predefined roles
        processed_roles = set()

        for role in all_roles:
            # Skip custom roles (they contain 'projects/' or 'organizations/' in the name)
            if "projects/" in role.name or "organizations/" in role.name:
                continue

            if role.name in processed_roles:
                continue

            processed_roles.add(role.name)

            # Categorize role
            role_info = categorize_role(role.name)

            # Get permissions list
            permissions_list = list(role.included_permissions) if role.included_permissions else []

            # Analyze permissions
            permission_analysis = analyze_permissions(permissions_list)

            # Determine if admin role
            is_admin = (
                role_info["category"] == "ADMIN" or
                role.name in ["roles/owner", "roles/editor"] or
                "admin" in role.name.lower()
            )

            # Determine if primitive role
            is_primitive = role.name in ["roles/owner", "roles/editor", "roles/viewer"]

            record = {
                "role_name": role.name,
                "title": role.title or "",
                "description": role.description or "",
                "stage": role.stage.name if role.stage else "GA",
                "category": role_info["category"],
                "service": role_info["service"],
                "resource_type": role_info["resource_type"],
                "included_permissions": json.dumps(permissions_list),
                "permission_count": permission_analysis["total_permissions"],
                "high_risk_permissions": permission_analysis["high_risk_count"],
                "services_accessed": json.dumps(permission_analysis["services_accessed"]),
                "capabilities": json.dumps(permission_analysis["capabilities"]),
                "is_admin": is_admin,
                "is_primitive": is_primitive,
                "project_id": project_id,
                "last_refreshed": datetime.utcnow().isoformat()
            }
            standard_roles_data.append(record)

        # Load data to BigQuery
        if standard_roles_data:
            errors = bq_client.insert_rows_json(table, standard_roles_data)
            if errors:
                print(f"BigQuery insert errors: {errors}")
                return {"status": "error", "message": str(errors)}, 500

        # Get statistics
        admin_roles = sum(1 for r in standard_roles_data if r["is_admin"])
        primitive_roles = sum(1 for r in standard_roles_data if r["is_primitive"])
        read_only_roles = sum(1 for r in standard_roles_data if r["category"] == "READ_ONLY")

        # Count by service
        service_counts = {}
        for role in standard_roles_data:
            service = role["service"]
            service_counts[service] = service_counts.get(service, 0) + 1

        top_services = sorted(service_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        result = {
            "status": "success",
            "total_roles": len(standard_roles_data),
            "admin_roles": admin_roles,
            "primitive_roles": primitive_roles,
            "read_only_roles": read_only_roles,
            "top_services": dict(top_services),
            "table_updated": table_id,
            "timestamp": datetime.utcnow().isoformat()
        }

        print(f"Successfully processed {len(standard_roles_data)} predefined roles")
        return result, 200

    except Exception as e:
        print(f"Error: {e}")
        return {"status": "error", "message": str(e)}, 500