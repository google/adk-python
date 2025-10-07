#!/usr/bin/env python3
"""
Data fetchers for GCP resources
Each fetcher returns typed Pydantic models
"""

from typing import List, Optional
import logging

# Lazy load GCP clients to avoid import errors
# from google.cloud import compute_v1, iam_admin_v1, storage, securitycenter

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
            from google.cloud.iam_admin_v1 import IAMClient
            self._iam_client = IAMClient()
        return self._iam_client

    def _get_resource_manager_client(self):
        """Lazy load Resource Manager client"""
        if self._rm_client is None:
            from google.cloud.resourcemanager_v3 import ProjectsClient
            self._rm_client = ProjectsClient()
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
            from google.cloud.iam_admin_v1 import ListRolesRequest, RoleView
            iam_client = self._get_iam_client()
            parent = f"projects/{self.project_id}"

            list_roles_request = ListRolesRequest(
                parent=parent,
                view=RoleView.FULL,
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
        """
        Fetch GCP release notes from RSS feeds

        Returns typed ReleaseNote objects from multiple GCP RSS feeds
        """
        import hashlib
        import re
        import requests
        import feedparser
        from dateutil import parser as date_parser

        logger.info("Fetching GCP release notes from RSS feeds...")

        # GCP Release Notes RSS Feeds
        rss_feeds = [
            {
                'url': 'https://cloud.google.com/feeds/gcp-release-notes.xml',
                'source': 'gcp_general',
                'name': 'Google Cloud Platform General Release Notes'
            },
            {
                'url': 'https://cloud.google.com/feeds/compute-release-notes.xml',
                'source': 'gcp_compute',
                'name': 'Google Compute Engine Release Notes'
            },
            {
                'url': 'https://cloud.google.com/feeds/gke-release-notes.xml',
                'source': 'gcp_gke',
                'name': 'Google Kubernetes Engine Release Notes'
            },
            {
                'url': 'https://cloud.google.com/feeds/iam-release-notes.xml',
                'source': 'gcp_iam',
                'name': 'Google Cloud IAM Release Notes'
            },
            {
                'url': 'https://cloud.google.com/feeds/security-command-center-release-notes.xml',
                'source': 'gcp_scc',
                'name': 'Security Command Center Release Notes'
            }
        ]

        all_releases = []

        try:
            for feed_config in rss_feeds:
                try:
                    logger.debug(f"Fetching feed: {feed_config['name']}")

                    # Fetch RSS feed
                    response = requests.get(feed_config['url'], timeout=30)
                    response.raise_for_status()

                    # Parse RSS feed
                    feed = feedparser.parse(response.content)

                    for entry in feed.entries:
                        # Create unique ID for deduplication
                        entry_id = hashlib.md5(
                            f"{entry.link}_{entry.title}".encode()
                        ).hexdigest()

                        # Parse publish date
                        published_date = None
                        if hasattr(entry, 'published'):
                            try:
                                published_date = date_parser.parse(entry.published)
                            except:
                                pass

                        # Extract and clean description
                        description = ""
                        if hasattr(entry, 'description'):
                            description = re.sub('<[^<]+?>', '', entry.description).strip()
                        elif hasattr(entry, 'summary'):
                            description = re.sub('<[^<]+?>', '', entry.summary).strip()

                        # Extract security keywords
                        security_keywords = self._extract_security_keywords(f"{entry.title} {description}")

                        # Calculate security relevance score
                        security_score = self._calculate_security_score(entry.title, description, security_keywords)

                        # Categorize the service
                        service_category = self._categorize_service(entry.title, description)

                        # Create typed ReleaseNote
                        release_note = ReleaseNote(
                            entry_id=entry_id,
                            title=entry.title,
                            description=description[:4000],  # Limit description length
                            link=entry.link,
                            source_feed=feed_config['source'],
                            feed_name=feed_config['name'],
                            published_date=published_date,
                            service_category=service_category,
                            security_keywords=security_keywords,
                            security_score=security_score,
                            is_security_related=security_score >= 3,
                            refresh_job='api_fetch'
                        )

                        all_releases.append(release_note)

                except requests.exceptions.RequestException as e:
                    logger.warning(f"Error fetching feed {feed_config['url']}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing feed {feed_config['name']}: {e}")
                    continue

            logger.info(f"Fetched {len(all_releases)} release notes from {len(rss_feeds)} RSS feeds")
            return all_releases

        except Exception as e:
            logger.error(f"Failed to fetch release notes: {e}")
            return []

    def _extract_security_keywords(self, text: str) -> List[str]:
        """Extract security-related keywords from text"""
        if not text:
            return []

        text_lower = text.lower()
        security_keywords = [
            'security', 'vulnerability', 'cve', 'patch', 'fix', 'bug fix',
            'authentication', 'authorization', 'encryption', 'ssl', 'tls',
            'firewall', 'iam', 'access control', 'privilege', 'permission',
            'audit', 'compliance', 'gdpr', 'sox', 'hipaa', 'pci',
            'threat', 'attack', 'malware', 'intrusion', 'breach',
            'certificate', 'key management', 'oauth', 'saml',
            'network security', 'data protection', 'privacy'
        ]

        found_keywords = []
        for keyword in security_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)

        return list(set(found_keywords))

    def _categorize_service(self, title: str, description: str) -> str:
        """Categorize the service based on title and description"""
        content = f"{title} {description}".lower()

        categories = {
            'compute': ['compute engine', 'gke', 'kubernetes', 'vm', 'instance', 'container'],
            'storage': ['cloud storage', 'persistent disk', 'filestore', 'backup'],
            'networking': ['vpc', 'cloud nat', 'load balancer', 'cdn', 'dns', 'firewall'],
            'database': ['cloud sql', 'firestore', 'bigtable', 'spanner', 'redis'],
            'security': ['iam', 'security command center', 'kms', 'secret manager', 'certificate'],
            'ai_ml': ['ai platform', 'automl', 'vertex ai', 'tensorflow', 'machine learning'],
            'analytics': ['bigquery', 'dataflow', 'dataproc', 'pub/sub', 'analytics'],
            'serverless': ['cloud functions', 'cloud run', 'app engine', 'workflows'],
            'monitoring': ['cloud monitoring', 'logging', 'trace', 'profiler', 'error reporting'],
            'data': ['dataflow', 'dataproc', 'dataprep', 'data fusion', 'data catalog']
        }

        for category, keywords in categories.items():
            if any(keyword in content for keyword in keywords):
                return category

        return 'other'

    def _calculate_security_score(self, title: str, description: str, keywords: List[str]) -> int:
        """Calculate a security relevance score (0-10)"""
        score = 0
        title_lower = title.lower()
        desc_lower = description.lower() if description else ""

        # High priority security terms
        high_priority = ['cve', 'vulnerability', 'security fix', 'patch', 'critical']
        medium_priority = ['authentication', 'authorization', 'encryption', 'firewall', 'iam']
        low_priority = ['compliance', 'audit', 'privacy', 'access control']

        # Score based on keywords
        for keyword in keywords:
            if any(hp in keyword for hp in high_priority):
                score += 3
            elif any(mp in keyword for mp in medium_priority):
                score += 2
            elif any(lp in keyword for lp in low_priority):
                score += 1

        # Additional scoring for title/description context
        if any(term in title_lower for term in high_priority):
            score += 2
        if any(term in desc_lower for term in high_priority):
            score += 2

        return min(score, 10)  # Cap at 10

    def fetch_confluence_pages(self, space_key: Optional[str] = None) -> List[ConfluencePage]:
        """Fetch Confluence pages"""
        # TODO: Implement using existing cloud_functions/confluence_sync/main.py logic
        logger.info("Fetching Confluence pages...")
        return []
