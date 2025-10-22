"""
Fetcher modules for unified Cloud Functions
"""

from .security_findings import SecurityFindingsFetcher
from .custom_roles import CustomRolesFetcher
from .compute_instances import ComputeInstancesFetcher
from .firewall_rules import FirewallRulesFetcher
from .storage_buckets import StorageBucketsFetcher
from .iam_accounts import IAMAccountsFetcher
from .service_account_roles import ServiceAccountRolesFetcher
from .standard_roles import StandardRolesFetcher
from .user_roles import UserRolesFetcher

# Registry of all fetchers
FETCHERS_REGISTRY = {
    'security_findings': SecurityFindingsFetcher,
    'custom_roles': CustomRolesFetcher,
    'compute_instances': ComputeInstancesFetcher,
    'firewall_rules': FirewallRulesFetcher,
    'storage_buckets': StorageBucketsFetcher,
    'iam_accounts': IAMAccountsFetcher,
    'service_account_roles': ServiceAccountRolesFetcher,
    'standard_roles': StandardRolesFetcher,
    'user_roles': UserRolesFetcher
}

__all__ = [
    'SecurityFindingsFetcher',
    'CustomRolesFetcher',
    'ComputeInstancesFetcher',
    'FirewallRulesFetcher',
    'StorageBucketsFetcher',
    'IAMAccountsFetcher',
    'ServiceAccountRolesFetcher',
    'StandardRolesFetcher',
    'UserRolesFetcher',
    'FETCHERS_REGISTRY'
]