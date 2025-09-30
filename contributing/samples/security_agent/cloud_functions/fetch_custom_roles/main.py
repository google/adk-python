#!/usr/bin/env python3
"""
Cloud Function to fetch and analyze custom IAM roles.
- Lists all custom roles in the project
- Maps permissions to risk levels
- Suggests similar predefined roles
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any
import functions_framework


def analyze_permission(permission: str) -> Dict[str, Any]:
    """Analyze a permission to determine risk level"""
    parts = permission.split(".")
    service = parts[0] if len(parts) > 0 else "unknown"
    resource_type = parts[1] if len(parts) > 1 else "unknown"
    verb = parts[-1] if len(parts) > 0 else "unknown"
    
    # Risk assessment
    high_risk_patterns = ["delete", "setIamPolicy", "admin", "actAs", "create", "update"]
    medium_risk_patterns = ["write", "modify", "edit"]
    
    risk_level = "LOW"
    if any(pattern in permission.lower() for pattern in high_risk_patterns):
        risk_level = "HIGH"
    elif any(pattern in permission.lower() for pattern in medium_risk_patterns):
        risk_level = "MEDIUM"
    
    return {
        "permission": permission,
        "service": service,
        "resource_type": resource_type,
        "verb": verb,
        "risk_level": risk_level
    }


def find_similar_predefined_roles(permissions: List[str]) -> List[Dict[str, Any]]:
    """Find predefined roles similar to the custom role"""
    # Common predefined role patterns
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


@functions_framework.http
def fetch_custom_roles(request):
    """Main Cloud Function to fetch custom roles"""
    from google.cloud import iam_admin_v1
    from google.cloud import bigquery
    
    project_id = os.environ.get('PROJECT_ID', 'mgm-digitalconcierge')
    dataset_id = os.environ.get('BQ_DATASET_ID', 'security_insights')
    
    try:
        # Initialize clients
        iam_client = iam_admin_v1.IAMClient()
        bq_client = bigquery.Client()
        
        print(f"Fetching custom roles for project: {project_id}")
        
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
            bigquery.SchemaField("role_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("role_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("title", "STRING"),
            bigquery.SchemaField("description", "STRING"),
            bigquery.SchemaField("stage", "STRING"),
            bigquery.SchemaField("deleted", "BOOLEAN"),
            bigquery.SchemaField("included_permissions", "JSON"),
            bigquery.SchemaField("permission_count", "INTEGER"),
            bigquery.SchemaField("high_risk_permissions", "INTEGER"),
            bigquery.SchemaField("similar_predefined_roles", "JSON"),
            bigquery.SchemaField("created_time", "TIMESTAMP"),
            bigquery.SchemaField("project_id", "STRING"),
            bigquery.SchemaField("last_refreshed", "TIMESTAMP")
        ]
        
        # Create or update table
        table_id = f"{project_id}.{dataset_id}.custom_roles"
        table = bigquery.Table(table_id, schema=schema)
        table = bq_client.create_table(table, exists_ok=True)
        
        # Fetch custom roles
        custom_roles_data = []
        parent = f"projects/{project_id}"
        list_request = iam_admin_v1.ListRolesRequest(
            parent=parent,
            view=iam_admin_v1.RoleView.FULL,
            show_deleted=False
        )
        
        roles = iam_client.list_roles(request=list_request)
        
        for role in roles:
            role_id = role.name.split("/")[-1]
            permissions_list = list(role.included_permissions) if role.included_permissions else []
            
            # Analyze permissions
            high_risk_count = sum(1 for p in permissions_list 
                                 if analyze_permission(p)["risk_level"] == "HIGH")
            
            # Find similar predefined roles
            similar_roles = find_similar_predefined_roles(permissions_list)
            
            record = {
                "role_id": role_id,
                "role_name": role.name,
                "title": role.title or "",
                "description": role.description or "",
                "stage": role.stage.name if role.stage else "ALPHA",
                "deleted": role.deleted,
                "included_permissions": json.dumps(permissions_list),
                "permission_count": len(permissions_list),
                "high_risk_permissions": high_risk_count,
                "similar_predefined_roles": json.dumps(similar_roles),
                "created_time": None,  # Add if available
                "project_id": project_id,
                "last_refreshed": datetime.utcnow().isoformat()
            }
            custom_roles_data.append(record)
        
        # Load data to BigQuery
        if custom_roles_data:
            errors = bq_client.insert_rows_json(table, custom_roles_data)
            if errors:
                print(f"BigQuery insert errors: {errors}")
                return {"status": "error", "message": str(errors)}, 500
        
        result = {
            "status": "success",
            "custom_roles_count": len(custom_roles_data),
            "high_risk_roles": sum(1 for r in custom_roles_data if r["high_risk_permissions"] > 0),
            "table_updated": table_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        print(f"Successfully processed {len(custom_roles_data)} custom roles")
        return result, 200
        
    except Exception as e:
        print(f"Error: {e}")
        return {"status": "error", "message": str(e)}, 500