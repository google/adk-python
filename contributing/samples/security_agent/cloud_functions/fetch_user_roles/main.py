#!/usr/bin/env python3
"""
Cloud Function to fetch IAM bindings for human users.
Tracks user email addresses and their assigned roles.
"""

import os
import json
from datetime import datetime
import functions_framework


@functions_framework.http
def fetch_user_roles(request):
    """Main Cloud Function to fetch user IAM bindings"""
    from google.cloud import resourcemanager_v3
    from google.cloud import bigquery

    project_id = os.environ.get('PROJECT_ID', 'mgm-digitalconcierge')
    dataset_id = os.environ.get('BQ_DATASET_ID', 'security_insights')

    try:
        # Initialize clients
        rm_client = resourcemanager_v3.ProjectsClient()
        bq_client = bigquery.Client()

        print(f"Fetching user IAM bindings for project: {project_id}")

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
            bigquery.SchemaField("user_email", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("role", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("role_type", "STRING"),  # PRIMITIVE, PREDEFINED, CUSTOM
            bigquery.SchemaField("is_admin", "BOOLEAN"),
            bigquery.SchemaField("is_owner", "BOOLEAN"),
            bigquery.SchemaField("is_external", "BOOLEAN"),
            bigquery.SchemaField("domain", "STRING"),
            bigquery.SchemaField("condition_expression", "STRING"),
            bigquery.SchemaField("condition_title", "STRING"),
            bigquery.SchemaField("project_id", "STRING"),
            bigquery.SchemaField("last_refreshed", "TIMESTAMP")
        ]

        # Create or update table
        table_id = f"{project_id}.{dataset_id}.user_roles"
        table = bigquery.Table(table_id, schema=schema)
        table = bq_client.create_table(table, exists_ok=True)

        # Get IAM policy
        resource = f"projects/{project_id}"
        policy = rm_client.get_iam_policy(
            request={"resource": resource}
        )

        user_roles_data = []
        project_domain = f"{project_id}.iam.gserviceaccount.com"

        for binding in policy.bindings:
            role = binding.role

            # Determine role type
            role_type = "CUSTOM"
            if role.startswith("roles/"):
                if "/" not in role[6:]:  # No additional slashes after "roles/"
                    role_type = "PRIMITIVE"
                else:
                    role_type = "PREDEFINED"

            # Check if admin or owner
            is_admin = "admin" in role.lower() or role == "roles/editor"
            is_owner = role == "roles/owner"

            for member in binding.members:
                # Only process user accounts (not service accounts or groups)
                if member.startswith("user:"):
                    email = member.replace("user:", "")
                    domain = email.split("@")[1] if "@" in email else ""
                    is_external = not email.endswith(f"@{project_domain}")

                    record = {
                        "user_email": email,
                        "role": role,
                        "role_type": role_type,
                        "is_admin": is_admin,
                        "is_owner": is_owner,
                        "is_external": is_external,
                        "domain": domain,
                        "condition_expression": binding.condition.expression if binding.condition else None,
                        "condition_title": binding.condition.title if binding.condition else None,
                        "project_id": project_id,
                        "last_refreshed": datetime.utcnow().isoformat()
                    }
                    user_roles_data.append(record)

        # Load data to BigQuery
        if user_roles_data:
            errors = bq_client.insert_rows_json(table, user_roles_data)
            if errors:
                print(f"BigQuery insert errors: {errors}")
                return {"status": "error", "message": str(errors)}, 500

        # Get statistics
        unique_users = len(set(r["user_email"] for r in user_roles_data))
        admin_users = len(set(r["user_email"] for r in user_roles_data if r["is_admin"]))
        owner_users = len(set(r["user_email"] for r in user_roles_data if r["is_owner"]))
        external_users = len(set(r["user_email"] for r in user_roles_data if r["is_external"]))

        result = {
            "status": "success",
            "total_bindings": len(user_roles_data),
            "unique_users": unique_users,
            "admin_users": admin_users,
            "owner_users": owner_users,
            "external_users": external_users,
            "table_updated": table_id,
            "timestamp": datetime.utcnow().isoformat()
        }

        print(f"Successfully processed {len(user_roles_data)} user role bindings")
        return result, 200

    except Exception as e:
        print(f"Error: {e}")
        return {"status": "error", "message": str(e)}, 500