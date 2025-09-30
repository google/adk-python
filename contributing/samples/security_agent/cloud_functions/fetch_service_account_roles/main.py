#!/usr/bin/env python3
"""
Cloud Function to fetch IAM bindings for service accounts.
Tracks service account details, their assigned roles, and keys.
"""

import os
import json
from datetime import datetime
import functions_framework


@functions_framework.http
def fetch_service_account_roles(request):
    """Main Cloud Function to fetch service account IAM bindings"""
    from google.cloud import resourcemanager_v3
    from google.cloud import iam_admin_v1
    from google.cloud import bigquery

    project_id = os.environ.get('PROJECT_ID', 'mgm-digitalconcierge')
    dataset_id = os.environ.get('BQ_DATASET_ID', 'security_insights')

    try:
        # Initialize clients
        rm_client = resourcemanager_v3.ProjectsClient()
        iam_client = iam_admin_v1.IAMClient()
        bq_client = bigquery.Client()

        print(f"Fetching service account IAM bindings for project: {project_id}")

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
            bigquery.SchemaField("service_account_email", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("service_account_id", "STRING"),
            bigquery.SchemaField("display_name", "STRING"),
            bigquery.SchemaField("role", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("role_type", "STRING"),  # PRIMITIVE, PREDEFINED, CUSTOM
            bigquery.SchemaField("is_admin", "BOOLEAN"),
            bigquery.SchemaField("is_google_managed", "BOOLEAN"),
            bigquery.SchemaField("is_user_managed", "BOOLEAN"),
            bigquery.SchemaField("has_keys", "BOOLEAN"),
            bigquery.SchemaField("key_count", "INTEGER"),
            bigquery.SchemaField("disabled", "BOOLEAN"),
            bigquery.SchemaField("condition_expression", "STRING"),
            bigquery.SchemaField("condition_title", "STRING"),
            bigquery.SchemaField("project_id", "STRING"),
            bigquery.SchemaField("last_refreshed", "TIMESTAMP")
        ]

        # Create or update table
        table_id = f"{project_id}.{dataset_id}.service_account_roles"
        table = bigquery.Table(table_id, schema=schema)
        table = bq_client.create_table(table, exists_ok=True)

        # Get IAM policy
        resource = f"projects/{project_id}"
        policy = rm_client.get_iam_policy(
            request={"resource": resource}
        )

        # Get all service accounts in the project
        service_accounts_info = {}
        try:
            sa_request = iam_admin_v1.ListServiceAccountsRequest(
                name=f"projects/{project_id}",
                page_size=100
            )
            sa_response = iam_client.list_service_accounts(request=sa_request)

            for sa in sa_response:
                # Get service account details
                sa_email = sa.email
                service_accounts_info[sa_email] = {
                    "id": sa.unique_id,
                    "display_name": sa.display_name or "",
                    "disabled": sa.disabled
                }

                # Get service account keys
                keys_request = iam_admin_v1.ListServiceAccountKeysRequest(
                    name=sa.name,
                    key_types=["USER_MANAGED"]
                )
                try:
                    keys = iam_client.list_service_account_keys(request=keys_request)
                    service_accounts_info[sa_email]["key_count"] = len(list(keys))
                    service_accounts_info[sa_email]["has_keys"] = len(list(keys)) > 0
                except:
                    service_accounts_info[sa_email]["key_count"] = 0
                    service_accounts_info[sa_email]["has_keys"] = False

        except Exception as e:
            print(f"Warning: Could not fetch service account details: {e}")

        sa_roles_data = []
        google_managed_domains = [
            "gserviceaccount.com",
            "cloudservices.gserviceaccount.com",
            "appspot.gserviceaccount.com",
            "cloudbuild.gserviceaccount.com",
            "firebase.com"
        ]

        for binding in policy.bindings:
            role = binding.role

            # Determine role type
            role_type = "CUSTOM"
            if role.startswith("roles/"):
                if "/" not in role[6:]:
                    role_type = "PRIMITIVE"
                else:
                    role_type = "PREDEFINED"

            # Check if admin
            is_admin = "admin" in role.lower() or role in ["roles/owner", "roles/editor"]

            for member in binding.members:
                # Only process service accounts
                if member.startswith("serviceAccount:"):
                    email = member.replace("serviceAccount:", "")

                    # Check if Google-managed
                    is_google_managed = any(domain in email for domain in google_managed_domains)

                    # Get additional info if available
                    sa_info = service_accounts_info.get(email, {})

                    record = {
                        "service_account_email": email,
                        "service_account_id": sa_info.get("id", ""),
                        "display_name": sa_info.get("display_name", ""),
                        "role": role,
                        "role_type": role_type,
                        "is_admin": is_admin,
                        "is_google_managed": is_google_managed,
                        "is_user_managed": not is_google_managed,
                        "has_keys": sa_info.get("has_keys", False),
                        "key_count": sa_info.get("key_count", 0),
                        "disabled": sa_info.get("disabled", False),
                        "condition_expression": binding.condition.expression if binding.condition else None,
                        "condition_title": binding.condition.title if binding.condition else None,
                        "project_id": project_id,
                        "last_refreshed": datetime.utcnow().isoformat()
                    }
                    sa_roles_data.append(record)

        # Load data to BigQuery
        if sa_roles_data:
            errors = bq_client.insert_rows_json(table, sa_roles_data)
            if errors:
                print(f"BigQuery insert errors: {errors}")
                return {"status": "error", "message": str(errors)}, 500

        # Get statistics
        unique_sas = len(set(r["service_account_email"] for r in sa_roles_data))
        admin_sas = len(set(r["service_account_email"] for r in sa_roles_data if r["is_admin"]))
        google_managed = len(set(r["service_account_email"] for r in sa_roles_data if r["is_google_managed"]))
        with_keys = len(set(r["service_account_email"] for r in sa_roles_data if r["has_keys"]))

        result = {
            "status": "success",
            "total_bindings": len(sa_roles_data),
            "unique_service_accounts": unique_sas,
            "admin_service_accounts": admin_sas,
            "google_managed": google_managed,
            "user_managed": unique_sas - google_managed,
            "with_keys": with_keys,
            "table_updated": table_id,
            "timestamp": datetime.utcnow().isoformat()
        }

        print(f"Successfully processed {len(sa_roles_data)} service account role bindings")
        return result, 200

    except Exception as e:
        print(f"Error: {e}")
        return {"status": "error", "message": str(e)}, 500