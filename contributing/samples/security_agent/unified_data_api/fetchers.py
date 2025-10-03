#!/usr/bin/env python3
"""
Data fetchers for GCP resources
Each fetcher returns typed Pydantic models
"""

from typing import List, Optional
from google.cloud import compute_v1, iam_admin_v1, storage, securitycenter
import logging

from .models import (
    IAMAccount, CustomRole, ServiceAccountRole,
    ComputeInstance, FirewallRule, Network,
    StorageBucket, SecurityFinding, SecurityFeed,
    ReleaseNote, ConfluencePage,
    AccountType, Severity
)

logger = logging.getLogger(__name__)


class IAMFetcher:
    """Fetch IAM-related resources"""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self._iam_client = None
        self._rm_client = None

    def _get_iam_client(self):
        """Lazy load IAM client"""
        if self._iam_client is None:
            self._iam_client = iam_admin_v1.IAMClient()
        return self._iam_client

    def _get_resource_manager_client(self):
        """Lazy load Resource Manager client"""
        if self._rm_client is None:
            from google.cloud import resourcemanager_v3
            self._rm_client = resourcemanager_v3.ProjectsClient()
        return self._rm_client

    def fetch_iam_accounts(self) -> List[IAMAccount]:
        """
        Fetch all IAM accounts with roles

        Returns typed IAMAccount objects from project IAM policy bindings
        """
        logger.info(f"Fetching IAM accounts for project {self.project_id}...")

        try:
            rm_client = self._get_resource_manager_client()
            iam_client = self._get_iam_client()

            # Get project IAM policy
            resource = f"projects/{self.project_id}"
            iam_policy = rm_client.get_iam_policy(request={"resource": resource})

            accounts = []
            processed_members = set()

            # Process IAM bindings
            for binding in iam_policy.bindings:
                role = binding.role

                # Determine if primitive role
                is_primitive = role in ["roles/owner", "roles/editor", "roles/viewer"]

                for member in binding.members:
                    # Parse member type and identity
                    member_parts = member.split(':')
                    member_type = member_parts[0] if len(member_parts) > 0 else 'unknown'
                    member_identity = member_parts[1] if len(member_parts) > 1 else member

                    # Map to AccountType enum
                    if member_type == 'serviceAccount':
                        acct_type = AccountType.SERVICE_ACCOUNT
                    elif member_type == 'user':
                        acct_type = AccountType.USER
                    elif member_type == 'group':
                        acct_type = AccountType.GROUP
                    else:
                        acct_type = AccountType.DOMAIN

                    # Create typed IAMAccount
                    account = IAMAccount(
                        email=member_identity if member_type in ['user', 'serviceAccount'] else f"{member_type}:{member_identity}",
                        account_type=acct_type,
                        role=role,
                        project_id=self.project_id,
                        is_primitive_role=is_primitive,
                        resource_name=member,
                        labels={}
                    )

                    accounts.append(account)
                    processed_members.add(member)

            # Fetch service accounts without roles
            try:
                service_accounts = iam_client.list_service_accounts(
                    request={"name": f"projects/{self.project_id}"}
                )

                for sa in service_accounts:
                    sa_email = sa.email
                    sa_member = f"serviceAccount:{sa_email}"

                    if sa_member not in processed_members:
                        # Service account with no roles
                        account = IAMAccount(
                            email=sa_email,
                            account_type=AccountType.SERVICE_ACCOUNT,
                            role="NO_ROLE_ASSIGNED",
                            project_id=self.project_id,
                            is_primitive_role=False,
                            resource_name=sa.name,
                            labels={}
                        )
                        accounts.append(account)

            except Exception as e:
                logger.warning(f"Could not fetch service account details: {e}")

            logger.info(f"Fetched {len(accounts)} IAM accounts")
            return accounts

        except Exception as e:
            logger.error(f"Failed to fetch IAM accounts: {e}")
            return []

    def fetch_custom_roles(self) -> List[CustomRole]:
        """
        Fetch custom IAM roles

        Returns typed CustomRole objects with permissions
        """
        logger.info(f"Fetching custom roles for project {self.project_id}...")

        try:
            iam_client = self._get_iam_client()
            parent = f"projects/{self.project_id}"

            list_roles_request = iam_admin_v1.ListRolesRequest(
                parent=parent,
                view=iam_admin_v1.RoleView.FULL,
                show_deleted=False
            )

            roles_iterator = iam_client.list_roles(request=list_roles_request)

            custom_roles = []
            for role in roles_iterator:
                role_id = role.name.split("/")[-1]
                permissions_list = list(role.included_permissions) if role.included_permissions else []

                custom_role = CustomRole(
                    role_id=role_id,
                    role_name=role.name,
                    title=role.title if hasattr(role, 'title') else "",
                    description=role.description if hasattr(role, 'description') else None,
                    permissions=permissions_list,
                    project_id=self.project_id,
                    deleted=role.deleted if hasattr(role, 'deleted') else False,
                    stage=role.stage.name if hasattr(role, 'stage') else "GA"
                )
                custom_roles.append(custom_role)

            logger.info(f"Fetched {len(custom_roles)} custom roles")
            return custom_roles

        except Exception as e:
            logger.error(f"Failed to fetch custom roles: {e}")
            return []

    def fetch_service_account_roles(self) -> List[ServiceAccountRole]:
        """
        Fetch service account roles

        Returns typed ServiceAccountRole objects with role assignments
        """
        logger.info(f"Fetching service account roles for project {self.project_id}...")

        try:
            iam_client = self._get_iam_client()

            service_accounts = iam_client.list_service_accounts(
                request={"name": f"projects/{self.project_id}"}
            )

            sa_roles = []
            for sa in service_accounts:
                # Get roles assigned to this service account
                # This would require checking IAM policy bindings
                # For now, create basic record
                sa_role = ServiceAccountRole(
                    service_account_email=sa.email,
                    project_id=self.project_id,
                    roles=[],  # Would need to query IAM policy
                    keys_count=0,  # Would need to list keys
                    enabled=not (sa.disabled if hasattr(sa, 'disabled') else False)
                )
                sa_roles.append(sa_role)

            logger.info(f"Fetched {len(sa_roles)} service accounts")
            return sa_roles

        except Exception as e:
            logger.error(f"Failed to fetch service account roles: {e}")
            return []


class ComputeFetcher:
    """Fetch Compute Engine resources"""

    def __init__(self, project_id: str):
        self.project_id = project_id

    def fetch_compute_instances(self, zones: Optional[List[str]] = None) -> List[ComputeInstance]:
        """Fetch compute instances"""
        # TODO: Implement using existing cloud_functions/fetch_compute_instances/main.py logic
        logger.info("Fetching compute instances...")
        return []


class NetworkFetcher:
    """Fetch Network resources"""

    def __init__(self, project_id: str):
        self.project_id = project_id

    def fetch_firewall_rules(self) -> List[FirewallRule]:
        """Fetch VPC firewall rules"""
        # TODO: Implement using existing cloud_functions/fetch_firewall_rules/main.py logic
        logger.info("Fetching firewall rules...")
        return []

    def fetch_networks(self) -> List[Network]:
        """Fetch VPC networks"""
        logger.info("Fetching VPC networks...")
        return []


class StorageFetcher:
    """Fetch Cloud Storage resources"""

    def __init__(self, project_id: str):
        self.project_id = project_id

    def fetch_storage_buckets(self) -> List[StorageBucket]:
        """Fetch storage buckets"""
        # TODO: Implement using existing cloud_functions/fetch_storage_buckets/main.py logic
        logger.info("Fetching storage buckets...")
        return []


class SecurityFetcher:
    """Fetch security-related resources"""

    def __init__(self, project_id: str):
        self.project_id = project_id

    def fetch_security_findings(self, min_severity: Optional[str] = None) -> List[SecurityFinding]:
        """Fetch Security Command Center findings"""
        # TODO: Implement using existing cloud_functions/fetch_security_findings/main.py logic
        logger.info("Fetching security findings...")
        return []


class FeedFetcher:
    """Fetch external feeds and documentation"""

    def __init__(self):
        pass

    def fetch_security_feeds(self) -> List[SecurityFeed]:
        """Fetch external security feeds"""
        # TODO: Implement using existing cloud_functions/fetch_security_feeds/main.py logic
        logger.info("Fetching security feeds...")
        return []

    def fetch_release_notes(self) -> List[ReleaseNote]:
        """Fetch GCP release notes"""
        # TODO: Implement using existing cloud_functions/fetch_gcp_release_notes/main.py logic
        logger.info("Fetching GCP release notes...")
        return []

    def fetch_confluence_pages(self, space_key: Optional[str] = None) -> List[ConfluencePage]:
        """Fetch Confluence pages"""
        # TODO: Implement using existing cloud_functions/confluence_sync/main.py logic
        logger.info("Fetching Confluence pages...")
        return []
