#!/usr/bin/env python3
"""
Enhanced Cloud Function to fetch comprehensive IAM data including:
- All IAM bindings at project level
- Custom roles with their permissions
- Predefined roles used in the project
- Service accounts and their keys
- Role permission mappings for analysis
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Set, Optional
import functions_framework

from google.cloud import bigquery

try:
    from google.protobuf.json_format import MessageToDict
except ImportError:  # pragma: no cover - optional dependency
    MessageToDict = None

try:
    from google.protobuf.message import Message  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Message = None  # type: ignore

# Lazy load heavy dependencies to avoid timeout during cold start
_iam_client: Optional[Any] = None
_rm_client: Optional[Any] = None
_bq_client: Optional[Any] = None


def _to_serializable(value: Any) -> Any:
    """Convert protobuf or complex objects into JSON-serializable data."""

    if value is None:
        return None

    if MessageToDict and Message is not None:
        try:
            if isinstance(value, Message):
                return MessageToDict(value, preserving_proto_field_name=True)
        except Exception:
            pass

    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(item) for item in value]

    if hasattr(value, "__dict__"):
        return {
            key: _to_serializable(val)
            for key, val in value.__dict__.items()
            if not key.startswith("_")
        }

    return value


def safe_json_dump(value: Any) -> str:
    """Serialize complex values to JSON, handling protobuf descriptors safely."""

    return json.dumps(_to_serializable(value), default=str)


def serialize_condition(condition: Any) -> Dict[str, Any]:
    """Convert IAM binding condition objects into a plain dictionary."""

    if not condition:
        return {}

    converted = _to_serializable(condition)
    if isinstance(converted, dict):
        return converted

    return {
        "title": getattr(condition, "title", "") or "",
        "description": getattr(condition, "description", "") or "",
        "expression": getattr(condition, "expression", "") or "",
        "location": getattr(condition, "location", "") or "",
    }


def serialize_service_account(sa: Any) -> Dict[str, Any]:
    """Normalize service account protobufs into dictionaries."""

    if not sa:
        return {}

    converted = _to_serializable(sa)
    if isinstance(converted, dict):
        return converted

    return {
        "name": getattr(sa, "name", ""),
        "email": getattr(sa, "email", ""),
        "unique_id": getattr(sa, "unique_id", ""),
        "disabled": getattr(sa, "disabled", False),
        "description": getattr(sa, "description", "") or "",
        "display_name": getattr(sa, "display_name", "") or "",
    }

def get_iam_client():
    """Lazy load IAM client"""
    global _iam_client
    if _iam_client is None:
        try:
            from google.cloud import iam_admin_v1
            _iam_client = iam_admin_v1.IAMClient()
        except Exception as e:
            print(f"Warning: Failed to initialize IAM client: {e}")
            from google.cloud import iam
            _iam_client = iam.IAMClient()
    return _iam_client

def get_resource_manager_client():
    """Lazy load Resource Manager client"""
    global _rm_client
    if _rm_client is None:
        try:
            from google.cloud import resourcemanager_v3
            _rm_client = resourcemanager_v3.ProjectsClient()
        except ImportError:
            # Fallback to older import path
            from google.cloud import resource_manager
            _rm_client = resource_manager.ProjectsClient()
    return _rm_client

def get_bigquery_client():
    """Lazy load BigQuery client"""
    global _bq_client
    if _bq_client is None:
        try:
            from google.cloud import bigquery
            _bq_client = bigquery.Client()
        except Exception as e:
            print(f"Warning: Failed to initialize BigQuery client: {e}")
            raise
    return _bq_client


def analyze_permission(permission: str) -> Dict[str, Any]:
    """
    Analyze a permission string to extract metadata
    """
    parts = permission.split(".")

    # Determine service
    service = parts[0] if len(parts) > 0 else "unknown"

    # Determine resource type
    resource_type = parts[1] if len(parts) > 1 else "unknown"

    # Determine verb
    verb = parts[-1] if len(parts) > 0 else "unknown"

    # Determine if it's a data access permission
    data_access_verbs = ["get", "list", "read", "download", "export"]
    is_data_access = any(v in verb.lower() for v in data_access_verbs)

    # Determine if it's an admin permission
    admin_indicators = ["admin", "create", "delete", "update", "setIamPolicy", "manage"]
    is_admin = any(ind in permission.lower() for ind in admin_indicators)

    # Determine risk level
    high_risk_patterns = ["delete", "setIamPolicy", "admin", "actAs", "create", "update"]
    medium_risk_patterns = ["write", "modify", "edit"]

    risk_level = "LOW"
    if any(pattern in permission.lower() for pattern in high_risk_patterns):
        risk_level = "HIGH"
    elif any(pattern in permission.lower() for pattern in medium_risk_patterns):
        risk_level = "MEDIUM"

    return {
        "service": service,
        "resource_type": resource_type,
        "verb": verb,
        "is_data_access": is_data_access,
        "is_admin": is_admin,
        "risk_level": risk_level
    }


def analyze_similar_roles(permissions: List[str]) -> List[Dict[str, Any]]:
    """
    Analyze which predefined roles are most similar to a custom role
    based on permission overlap
    """
    # Common role permission patterns (simplified for matching)
    role_patterns = {
        "roles/viewer": ["get", "list", "read"],
        "roles/editor": ["get", "list", "create", "update", "delete"],
        "roles/owner": ["get", "list", "create", "update", "delete", "setIamPolicy"],
        "roles/compute.viewer": ["compute.*.get", "compute.*.list"],
        "roles/compute.admin": ["compute.*"],
        "roles/storage.objectViewer": ["storage.objects.get", "storage.objects.list"],
        "roles/storage.admin": ["storage.*"],
        "roles/bigquery.dataViewer": ["bigquery.*.get", "bigquery.datasets.get"],
        "roles/bigquery.admin": ["bigquery.*"],
        "roles/iam.securityAdmin": ["iam.*", "resourcemanager.projects.getIamPolicy"],
        "roles/iam.serviceAccountAdmin": ["iam.serviceAccounts.*"],
        "roles/container.admin": ["container.*"],
        "roles/cloudsql.admin": ["cloudsql.*"],
        "roles/monitoring.viewer": ["monitoring.*.get", "monitoring.*.list"]
    }

    similar_roles = []
    permission_set = set(permissions)

    for role, patterns in role_patterns.items():
        match_count = 0
        for pattern in patterns:
            if "*" in pattern:
                # Wildcard pattern
                prefix = pattern.replace("*", "")
                match_count += sum(1 for p in permission_set if p.startswith(prefix))
            else:
                # Check for partial matches in permissions
                for perm in permission_set:
                    if pattern in perm or perm.endswith(pattern):
                        match_count += 1
                        break

        if match_count > 0:
            similarity = (match_count / len(permissions)) * 100 if permissions else 0
            similar_roles.append({
                "role": role,
                "similarity_percentage": round(similarity, 2),
                "matched_permissions": match_count
            })

    # Sort by similarity
    similar_roles.sort(key=lambda x: x["similarity_percentage"], reverse=True)

    return similar_roles[:5]  # Return top 5 similar roles


def analyze_role_permissions(role_name: str, included_permissions: List[str] = None) -> Dict[str, Any]:
    """Enhanced analysis of role criticality including permission analysis"""
    # High-risk roles
    critical_roles = {
        'roles/owner': 'CRITICAL',
        'roles/editor': 'HIGH',
        'roles/iam.securityAdmin': 'CRITICAL',
        'roles/compute.admin': 'HIGH',
        'roles/storage.admin': 'HIGH',
        'roles/bigquery.admin': 'HIGH',
        'roles/container.admin': 'HIGH',
        'roles/iam.serviceAccountAdmin': 'HIGH',
        'roles/iam.serviceAccountKeyAdmin': 'HIGH',
        'roles/resourcemanager.organizationAdmin': 'CRITICAL'
    }

    risk_level = critical_roles.get(role_name, 'MEDIUM')
    if 'viewer' in role_name.lower() or 'reader' in role_name.lower():
        risk_level = 'LOW'

    # For custom roles, analyze based on permissions
    if role_name.startswith('projects/') or role_name.startswith('organizations/'):
        if included_permissions:
            # Check for high-risk permissions
            high_risk_perms = sum(1 for p in included_permissions if
                                 'setIamPolicy' in p or 'delete' in p or 'admin' in p.lower())
            if high_risk_perms > 5:
                risk_level = 'CRITICAL'
            elif high_risk_perms > 2:
                risk_level = 'HIGH'

    return {
        'role_name': role_name,
        'risk_level': risk_level,
        'is_primitive': role_name in ['roles/owner', 'roles/editor', 'roles/viewer'],
        'is_custom': role_name.startswith('projects/') or role_name.startswith('organizations/')
    }


@functions_framework.http
def fetch_iam_accounts(request):
    """
    Enhanced Cloud Function that fetches comprehensive IAM data including:
    - All IAM bindings at project level
    - Custom roles with their permissions
    - Predefined roles used in the project
    - Service accounts and their keys
    - Role permission mappings for analysis

    Args:
        request: HTTP request object

    Returns:
        JSON response with complete IAM analysis
    """

    # Initialize clients (lazy loading)
    iam_client = get_iam_client()
    rm_client = get_resource_manager_client()
    bq_client = get_bigquery_client()

    # Get configuration from environment
    project_id = os.environ.get('PROJECT_ID', 'mgm-digitalconcierge')
    dataset_id = os.environ.get('BQ_DATASET_ID', 'security_insights')

    print(f"Starting comprehensive IAM data fetch for project: {project_id}")

    try:
        # Get project IAM policy
        resource = f"projects/{project_id}"
        iam_policy = rm_client.get_iam_policy(
            request={"resource": resource}
        )

        # Process IAM bindings
        iam_accounts_data = []
        custom_roles_data = []
        role_permissions_data = []
        processed_members = set()
        processed_custom_roles = set()

        for binding in iam_policy.bindings:
            role = binding.role
            role_analysis = analyze_role_permissions(role)

            for member in binding.members:
                # Create unique identifier
                member_id = f"{member}_{role}"

                # Parse member type and identity
                member_parts = member.split(':')
                member_type = member_parts[0] if len(member_parts) > 0 else 'unknown'
                member_identity = member_parts[1] if len(member_parts) > 1 else member

                # Determine if it's a service account
                is_service_account = (
                    member_type == 'serviceAccount' or
                    member_identity.endswith('.iam.gserviceaccount.com')
                )

                # Check for Google-managed service accounts
                is_google_managed = (
                    member_identity.endswith('@cloudservices.gserviceaccount.com') or
                    member_identity.endswith('@developer.gserviceaccount.com') or
                    member_identity.startswith('service-') or
                    '-compute@developer.gserviceaccount.com' in member_identity
                )

                account_record = {
                    'account_id': member_id,
                    'member': member,
                    'member_type': member_type,
                    'email': member_identity if member_type in ['user', 'serviceAccount'] else None,
                    'role': role,
                    'role_type': role_analysis['is_primitive'] and 'primitive' or 'predefined',
                    'risk_level': role_analysis['risk_level'],
                    'is_service_account': is_service_account,
                    'is_google_managed': is_google_managed,
                    'is_external': not member_identity.endswith(f'@{project_id}.iam.gserviceaccount.com'),
                    'has_admin_privileges': 'admin' in role.lower() or role in ['roles/owner', 'roles/editor'],
                    'project_id': project_id,
                    'conditions': safe_json_dump(serialize_condition(binding.condition)),
                    'last_refreshed': datetime.utcnow().isoformat(),
                    'refresh_job': 'scheduled_6h'
                }

                iam_accounts_data.append(account_record)
                processed_members.add(member)

        # Fetch service accounts details
        try:
            service_accounts = iam_client.list_service_accounts(
                request={"name": f"projects/{project_id}"}
            )

            for sa in service_accounts:
                sa_email = sa.email
                sa_member = f"serviceAccount:{sa_email}"

                # Check if we already have this service account from bindings
                if sa_member not in processed_members:
                    # Add service account even if it has no roles
                    account_record = {
                        'account_id': f"{sa_member}_no_role",
                        'member': sa_member,
                        'member_type': 'serviceAccount',
                        'email': sa_email,
                        'role': 'NO_ROLE_ASSIGNED',
                        'role_type': 'none',
                        'risk_level': 'INFO',
                        'is_service_account': True,
                        'is_google_managed': False,
                        'is_external': False,
                        'has_admin_privileges': False,
                        'project_id': project_id,
                        'service_account_details': safe_json_dump(serialize_service_account(sa)),
                        'conditions': safe_json_dump({}),
                        'last_refreshed': datetime.utcnow().isoformat(),
                        'refresh_job': 'scheduled_6h'
                    }
                    iam_accounts_data.append(account_record)
                else:
                    # Update existing record with service account details
                    for record in iam_accounts_data:
                        if record['email'] == sa_email:
                            record['service_account_details'] = safe_json_dump(serialize_service_account(sa))

        except Exception as e:
            print(f"Warning: Could not fetch service account details: {e}")

        # Fetch all custom roles in the project
        print("Fetching custom roles...")
        try:
            parent = f"projects/{project_id}"
            # Import iam_admin_v1 locally for request types
            from google.cloud import iam_admin_v1
            list_roles_request = iam_admin_v1.ListRolesRequest(
                parent=parent,
                view=iam_admin_v1.RoleView.FULL,
                show_deleted=False
            )

            roles_iterator = iam_client.list_roles(request=list_roles_request)

            for custom_role in roles_iterator:
                role_id = custom_role.name.split("/")[-1]

                # Skip if already processed
                if custom_role.name in processed_custom_roles:
                    continue

                processed_custom_roles.add(custom_role.name)

                # Get permissions list
                permissions_list = list(custom_role.included_permissions) if custom_role.included_permissions else []

                # Analyze similar predefined roles
                similar_roles = analyze_similar_roles(permissions_list)

                # Create custom role record
                custom_role_data = {
                    'role_id': role_id,
                    'name': custom_role.name,
                    'title': custom_role.title if hasattr(custom_role, 'title') else "",
                    'description': custom_role.description if hasattr(custom_role, 'description') else "",
                    'included_permissions': safe_json_dump(permissions_list),
                    'stage': custom_role.stage.name if hasattr(custom_role, 'stage') else "GA",
                    'deleted': custom_role.deleted if hasattr(custom_role, 'deleted') else False,
                    'etag': custom_role.etag if hasattr(custom_role, 'etag') else "",
                    'permission_count': len(permissions_list),
                    'similar_predefined_roles': safe_json_dump(similar_roles),
                    'risk_analysis': safe_json_dump(analyze_role_permissions(custom_role.name, permissions_list)),
                    'last_refreshed': datetime.utcnow().isoformat(),
                    'project_id': project_id
                }
                custom_roles_data.append(custom_role_data)

                # Analyze each permission for the role_permissions table
                for permission in permissions_list:
                    perm_analysis = analyze_permission(permission)
                    permission_record = {
                        'mapping_id': f"{custom_role.name}_{permission}".replace("/", "_").replace(".", "_"),
                        'role_name': custom_role.name,
                        'role_type': 'CUSTOM',
                        'permission': permission,
                        'service': perm_analysis['service'],
                        'resource_type': perm_analysis['resource_type'],
                        'verb': perm_analysis['verb'],
                        'is_data_access': perm_analysis['is_data_access'],
                        'is_admin': perm_analysis['is_admin'],
                        'risk_level': perm_analysis['risk_level'],
                        'project_id': project_id,
                        'last_refreshed': datetime.utcnow().isoformat()
                    }
                    role_permissions_data.append(permission_record)

        except Exception as e:
            print(f"Warning: Could not fetch custom roles: {e}")

        # Also analyze permissions for common predefined roles
        print("Analyzing predefined roles...")
        common_predefined_roles = [
            "roles/owner",
            "roles/editor",
            "roles/viewer",
            "roles/compute.admin",
            "roles/storage.admin",
            "roles/iam.securityAdmin",
            "roles/bigquery.admin"
        ]

        for role_name in common_predefined_roles:
            try:
                role = iam_client.get_role(name=role_name)
                # Only sample first 50 permissions to avoid huge data
                sample_permissions = list(role.included_permissions)[:50] if role.included_permissions else []

                for permission in sample_permissions:
                    perm_analysis = analyze_permission(permission)
                    permission_record = {
                        'mapping_id': f"{role_name}_{permission}".replace("/", "_").replace(".", "_"),
                        'role_name': role_name,
                        'role_type': 'PREDEFINED',
                        'permission': permission,
                        'service': perm_analysis['service'],
                        'resource_type': perm_analysis['resource_type'],
                        'verb': perm_analysis['verb'],
                        'is_data_access': perm_analysis['is_data_access'],
                        'is_admin': perm_analysis['is_admin'],
                        'risk_level': perm_analysis['risk_level'],
                        'project_id': project_id,
                        'last_refreshed': datetime.utcnow().isoformat()
                    }
                    role_permissions_data.append(permission_record)
            except:
                continue

        # Load data to BigQuery
        # 1. Load IAM accounts data
        if iam_accounts_data:
            table_id = f"{project_id}.{dataset_id}.iam_accounts"

            # Define schema for iam_accounts
            schema = [
                bigquery.SchemaField("account_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("member", "STRING"),
                bigquery.SchemaField("member_type", "STRING"),
                bigquery.SchemaField("email", "STRING"),
                bigquery.SchemaField("role", "STRING"),
                bigquery.SchemaField("role_type", "STRING"),
                bigquery.SchemaField("risk_level", "STRING"),
                bigquery.SchemaField("is_service_account", "BOOLEAN"),
                bigquery.SchemaField("is_google_managed", "BOOLEAN"),
                bigquery.SchemaField("is_external", "BOOLEAN"),
                bigquery.SchemaField("has_admin_privileges", "BOOLEAN"),
                bigquery.SchemaField("project_id", "STRING"),
                bigquery.SchemaField("service_account_details", "JSON"),
                bigquery.SchemaField("conditions", "JSON"),
                bigquery.SchemaField("last_refreshed", "TIMESTAMP"),
                bigquery.SchemaField("refresh_job", "STRING"),
            ]

            # Configure load job
            job_config = bigquery.LoadJobConfig(
                schema=schema,
                write_disposition="WRITE_TRUNCATE",
                create_disposition="CREATE_IF_NEEDED",
            )

            # Load data
            job = bq_client.load_table_from_json(
                iam_accounts_data,
                table_id,
                job_config=job_config
            )
            job.result()

            print(f"Successfully loaded {len(iam_accounts_data)} IAM accounts to BigQuery")

        # 2. Load custom roles data
        if custom_roles_data:
            custom_roles_table_id = f"{project_id}.{dataset_id}.custom_roles"

            # Define schema for custom_roles
            custom_roles_schema = [
                bigquery.SchemaField("role_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("name", "STRING"),
                bigquery.SchemaField("title", "STRING"),
                bigquery.SchemaField("description", "STRING"),
                bigquery.SchemaField("included_permissions", "JSON"),
                bigquery.SchemaField("stage", "STRING"),
                bigquery.SchemaField("deleted", "BOOLEAN"),
                bigquery.SchemaField("etag", "STRING"),
                bigquery.SchemaField("permission_count", "INTEGER"),
                bigquery.SchemaField("similar_predefined_roles", "JSON"),
                bigquery.SchemaField("risk_analysis", "JSON"),
                bigquery.SchemaField("last_refreshed", "TIMESTAMP"),
                bigquery.SchemaField("project_id", "STRING")
            ]

            # Configure and load custom roles
            custom_roles_job_config = bigquery.LoadJobConfig(
                schema=custom_roles_schema,
                write_disposition="WRITE_TRUNCATE",
                create_disposition="CREATE_IF_NEEDED",
            )

            custom_roles_job = bq_client.load_table_from_json(
                custom_roles_data,
                custom_roles_table_id,
                job_config=custom_roles_job_config
            )
            custom_roles_job.result()

            print(f"Successfully loaded {len(custom_roles_data)} custom roles to BigQuery")

        # 3. Load role permissions mapping data
        if role_permissions_data:
            role_permissions_table_id = f"{project_id}.{dataset_id}.role_permissions"

            # Define schema for role_permissions
            role_permissions_schema = [
                bigquery.SchemaField("mapping_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("role_name", "STRING"),
                bigquery.SchemaField("role_type", "STRING"),  # CUSTOM or PREDEFINED
                bigquery.SchemaField("permission", "STRING"),
                bigquery.SchemaField("service", "STRING"),
                bigquery.SchemaField("resource_type", "STRING"),
                bigquery.SchemaField("verb", "STRING"),
                bigquery.SchemaField("is_data_access", "BOOLEAN"),
                bigquery.SchemaField("is_admin", "BOOLEAN"),
                bigquery.SchemaField("risk_level", "STRING"),
                bigquery.SchemaField("project_id", "STRING"),
                bigquery.SchemaField("last_refreshed", "TIMESTAMP")
            ]

            # Configure and load role permissions
            role_permissions_job_config = bigquery.LoadJobConfig(
                schema=role_permissions_schema,
                write_disposition="WRITE_TRUNCATE",
                create_disposition="CREATE_IF_NEEDED",
            )

            role_permissions_job = bq_client.load_table_from_json(
                role_permissions_data,
                role_permissions_table_id,
                job_config=role_permissions_job_config
            )
            role_permissions_job.result()

            print(f"Successfully loaded {len(role_permissions_data)} role-permission mappings to BigQuery")

        # Create aggregated views and statistics
        create_iam_stats_view(bq_client, project_id, dataset_id)
        create_custom_role_analysis_view(bq_client, project_id, dataset_id)
        create_permission_risk_view(bq_client, project_id, dataset_id)

        # Log refresh metadata
        metadata_table_id = f"{project_id}.{dataset_id}.refresh_metadata"
        metadata_record = [{
            'table_name': 'iam_accounts',
            'refresh_time': datetime.utcnow().isoformat(),
            'record_count': len(iam_accounts_data),
            'status': 'success',
            'refresh_type': 'scheduled',
            'details': safe_json_dump({
                'total_accounts': len(iam_accounts_data),
                'service_accounts': sum(1 for a in iam_accounts_data if a['is_service_account']),
                'admin_accounts': sum(1 for a in iam_accounts_data if a['has_admin_privileges']),
                'external_accounts': sum(1 for a in iam_accounts_data if a['is_external']),
                'custom_roles': len(custom_roles_data),
                'permission_mappings': len(role_permissions_data)
            }),
            'error_message': None
        }]

        try:
            metadata_job = bq_client.load_table_from_json(
                metadata_record,
                metadata_table_id,
                job_config=bigquery.LoadJobConfig(
                    write_disposition="WRITE_APPEND",
                    schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION]
                )
            )
            metadata_job.result()
        except Exception as e:
            print(f"Warning: Could not update refresh metadata: {e}")

        # Return comprehensive statistics
        return {
            'status': 'success',
            'tables_updated': {
                'iam_accounts': len(iam_accounts_data),
                'custom_roles': len(custom_roles_data),
                'role_permissions': len(role_permissions_data)
            },
            'statistics': {
                'iam_accounts': {
                    'total': len(iam_accounts_data),
                    'service_accounts': sum(1 for a in iam_accounts_data if a['is_service_account']),
                    'admin_accounts': sum(1 for a in iam_accounts_data if a['has_admin_privileges']),
                    'high_risk': sum(1 for a in iam_accounts_data if a['risk_level'] in ['HIGH', 'CRITICAL']),
                    'external': sum(1 for a in iam_accounts_data if a['is_external'])
                },
                'custom_roles': {
                    'total': len(custom_roles_data),
                    'with_high_risk_permissions': sum(
                        1 for r in custom_roles_data
                        if json.loads(r['risk_analysis'])['risk_level'] in ['HIGH', 'CRITICAL']
                    ) if custom_roles_data else 0
                },
                'permissions': {
                    'total_mappings': len(role_permissions_data),
                    'high_risk_permissions': sum(
                        1 for p in role_permissions_data
                        if p['risk_level'] == 'HIGH'
                    ),
                    'admin_permissions': sum(
                        1 for p in role_permissions_data
                        if p['is_admin']
                    )
                }
            },
            'timestamp': datetime.utcnow().isoformat()
        }

    except Exception as e:
        error_msg = f"Error in fetch_iam_accounts: {str(e)}"
        print(error_msg)

        # Log error to metadata
        try:
            metadata_table_id = f"{project_id}.{dataset_id}.refresh_metadata"
            error_record = [{
                'table_name': 'iam_accounts',
                'refresh_time': datetime.utcnow().isoformat(),
                'record_count': 0,
                'status': 'failed',
                'refresh_type': 'scheduled',
                'error_message': str(e)[:1000]
            }]

            bq_client.load_table_from_json(
                error_record,
                metadata_table_id,
                job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
            ).result()
        except:
            pass

        return {
            'status': 'error',
            'error': error_msg,
            'timestamp': datetime.utcnow().isoformat()
        }, 500


def create_iam_stats_view(bq_client, project_id, dataset_id):
    """Create or update a view for IAM statistics"""
    view_id = f"{project_id}.{dataset_id}.iam_stats_view"

    view_query = f"""
    CREATE OR REPLACE VIEW `{view_id}` AS
    SELECT
        COUNT(DISTINCT email) as total_accounts,
        COUNTIF(is_service_account) as service_accounts,
        COUNTIF(NOT is_service_account) as human_accounts,
        COUNTIF(has_admin_privileges) as admin_accounts,
        COUNTIF(risk_level = 'CRITICAL') as critical_risk_accounts,
        COUNTIF(risk_level = 'HIGH') as high_risk_accounts,
        COUNTIF(is_external) as external_accounts,
        COUNTIF(is_google_managed) as google_managed_accounts,
        MAX(last_refreshed) as last_updated
    FROM `{project_id}.{dataset_id}.iam_accounts`
    """

    try:
        bq_client.query(view_query).result()
        print(f"Created/updated IAM stats view: {view_id}")
    except Exception as e:
        print(f"Warning: Could not create stats view: {e}")


def create_custom_role_analysis_view(bq_client, project_id, dataset_id):
    """Create a view for custom role analysis and mapping to predefined roles"""
    view_id = f"{project_id}.{dataset_id}.custom_role_analysis_view"

    view_query = f"""
    CREATE OR REPLACE VIEW `{view_id}` AS
    WITH role_similarity AS (
        SELECT
            cr.role_id,
            cr.name as custom_role_name,
            cr.title,
            cr.description,
            cr.permission_count,
            JSON_EXTRACT_SCALAR(sr.similar_role, '$.role') as similar_predefined_role,
            CAST(JSON_EXTRACT_SCALAR(sr.similar_role, '$.similarity_percentage') AS FLOAT64) as similarity_percentage,
            CAST(JSON_EXTRACT_SCALAR(sr.similar_role, '$.matched_permissions') AS INT64) as matched_permissions,
            JSON_EXTRACT_SCALAR(cr.risk_analysis, '$.risk_level') as risk_level
        FROM `{project_id}.{dataset_id}.custom_roles` cr,
        UNNEST(JSON_EXTRACT_ARRAY(cr.similar_predefined_roles)) as sr
    )
    SELECT
        custom_role_name,
        title,
        description,
        permission_count,
        risk_level,
        ARRAY_AGG(
            STRUCT(
                similar_predefined_role,
                similarity_percentage,
                matched_permissions
            )
            ORDER BY similarity_percentage DESC
            LIMIT 3
        ) as top_similar_roles,
        MAX(similarity_percentage) as max_similarity_percentage,
        STRING_AGG(similar_predefined_role, ', ' ORDER BY similarity_percentage DESC LIMIT 3) as suggested_replacements
    FROM role_similarity
    GROUP BY custom_role_name, title, description, permission_count, risk_level
    ORDER BY permission_count DESC
    """

    try:
        bq_client.query(view_query).result()
        print(f"Created/updated custom role analysis view: {view_id}")
    except Exception as e:
        print(f"Warning: Could not create custom role analysis view: {e}")


def create_permission_risk_view(bq_client, project_id, dataset_id):
    """Create a view for permission risk analysis"""
    view_id = f"{project_id}.{dataset_id}.permission_risk_view"

    view_query = f"""
    CREATE OR REPLACE VIEW `{view_id}` AS
    WITH permission_stats AS (
        SELECT
            permission,
            service,
            resource_type,
            verb,
            risk_level,
            COUNT(DISTINCT role_name) as role_count,
            COUNTIF(role_type = 'CUSTOM') as custom_role_count,
            COUNTIF(role_type = 'PREDEFINED') as predefined_role_count,
            COUNTIF(is_admin) as admin_permission_count,
            COUNTIF(is_data_access) as data_access_count
        FROM `{project_id}.{dataset_id}.role_permissions`
        GROUP BY permission, service, resource_type, verb, risk_level
    )
    SELECT
        service,
        COUNT(DISTINCT permission) as total_permissions,
        COUNTIF(risk_level = 'HIGH') as high_risk_permissions,
        COUNTIF(risk_level = 'MEDIUM') as medium_risk_permissions,
        COUNTIF(risk_level = 'LOW') as low_risk_permissions,
        SUM(custom_role_count) as used_in_custom_roles,
        SUM(predefined_role_count) as used_in_predefined_roles,
        ARRAY_AGG(
            STRUCT(permission, risk_level, role_count)
            ORDER BY
                CASE risk_level
                    WHEN 'HIGH' THEN 1
                    WHEN 'MEDIUM' THEN 2
                    WHEN 'LOW' THEN 3
                END,
                role_count DESC
            LIMIT 10
        ) as top_permissions
    FROM permission_stats
    GROUP BY service
    ORDER BY high_risk_permissions DESC, total_permissions DESC
    """

    try:
        bq_client.query(view_query).result()
        print(f"Created/updated permission risk view: {view_id}")
    except Exception as e:
        print(f"Warning: Could not create permission risk view: {e}")


# For local testing
if __name__ == "__main__":
    class MockRequest:
        def __init__(self):
            self.json = {}

    result = fetch_iam_accounts(MockRequest())
    print(safe_json_dump(result))
