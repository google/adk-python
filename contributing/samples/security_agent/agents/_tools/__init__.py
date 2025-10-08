"""
BigQuery Security Agent Tools
Comprehensive tool suite for security analysis, BigQuery operations, and GCP service management
"""

# Security Analysis Tools
from .security_tools import (
    get_security_insights_summary,
    query_security_insights,
    get_security_statistics,
    get_resources_by_severity,
    get_recent_findings,
    export_findings_to_csv
)

from .typed_security_tools import (
    get_primitive_role_accounts,
    get_old_service_account_keys,
    get_open_firewall_rules,
    get_ssh_accessible_resources,
    get_public_storage_buckets,
    get_unencrypted_buckets,
    get_critical_security_findings,
    get_high_severity_findings_by_resource,
    analyze_iam_security_posture,
    analyze_network_security_posture
)

# IAM Analysis Tools
from .iam_custom_role_analyzer import (
    analyze_all_custom_roles,
    analyze_custom_role_tool
)

# BigQuery Tools
from .bigquery_tools import (
    hello_world,
    list_datasets,
    list_tables,
    get_table_schema,
    run_query,
    analyze_query_cost,
    get_table_sample
)

# Exploration Tools
from .exploration_tools import (
    explore_all_tables_and_views,
    analyze_table_or_view
)

# Confluence Documentation Tools
from .confluence_tools import (
    search_confluence_documentation,
    get_confluence_document,
    analyze_confluence_coverage,
    get_confluence_statistics,
    refresh_confluence_cache
)

# Feed Tools (Security Threat & GCP Release Notes)
from .feed_tools import (
    query_gcp_release_notes,
    query_security_threat_feeds,
    get_feed_statistics,
    search_feeds_by_keyword
)

# Service Discovery Tools
from .service_discovery import (
    discover_gcp_services,
    analyze_gcp_service,
    get_service_resources,
    suggest_service_analysis,
    learn_service_from_url,
    discover_new_gcp_services,
    register_new_service,
    learn_from_api_spec
)

# Service Documentation Parser
from .service_documentation_parser import (
    parse_service_documentation,
    discover_new_services,
    learn_service_from_api_spec as learn_service_from_api_spec_parser,
    register_custom_service
)

# Service Onboarding
from .service_onboarding import (
    onboard_service
)

# MSA/Release Analyzer
from .msa_analyzer import (
    analyze_releases,
    analyze_gcp_releases
)

__all__ = [
    # Security Analysis Tools (6)
    'get_security_insights_summary',
    'query_security_insights',
    'get_security_statistics',
    'get_resources_by_severity',
    'get_recent_findings',
    'export_findings_to_csv',

    # Typed Security Tools (10)
    'get_primitive_role_accounts',
    'get_old_service_account_keys',
    'get_open_firewall_rules',
    'get_ssh_accessible_resources',
    'get_public_storage_buckets',
    'get_unencrypted_buckets',
    'get_critical_security_findings',
    'get_high_severity_findings_by_resource',
    'analyze_iam_security_posture',
    'analyze_network_security_posture',

    # IAM Analysis Tools (2)
    'analyze_all_custom_roles',
    'analyze_custom_role_tool',

    # BigQuery Tools (7)
    'hello_world',
    'list_datasets',
    'list_tables',
    'get_table_schema',
    'run_query',
    'analyze_query_cost',
    'get_table_sample',

    # Exploration Tools (2)
    'explore_all_tables_and_views',
    'analyze_table_or_view',

    # Confluence Tools (5)
    'search_confluence_documentation',
    'get_confluence_document',
    'analyze_confluence_coverage',
    'get_confluence_statistics',
    'refresh_confluence_cache',

    # Feed Tools (4)
    'query_gcp_release_notes',
    'query_security_threat_feeds',
    'get_feed_statistics',
    'search_feeds_by_keyword',

    # Service Discovery Tools (8)
    'discover_gcp_services',
    'analyze_gcp_service',
    'get_service_resources',
    'suggest_service_analysis',
    'learn_service_from_url',
    'discover_new_gcp_services',
    'register_new_service',
    'learn_from_api_spec',

    # Service Documentation Parser (4)
    'parse_service_documentation',
    'discover_new_services',
    'learn_service_from_api_spec_parser',
    'register_custom_service',

    # Service Onboarding (1)
    'onboard_service',

    # MSA/Release Analyzer (2)
    'analyze_releases',
    'analyze_gcp_releases',
]
