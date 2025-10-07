# Security Agent Tools Documentation

Complete reference for all 32 tools registered in the ADK Security Agent.

## Table of Contents

1. [Quick Tool Selector](#-quick-tool-selector)
2. [Quick Reference Matrix](#quick-reference-matrix)
3. [Security-Focused Tools](#security-focused-tools) (3 tools)
4. [Data Exploration Tools](#data-exploration-tools) (2 tools)
5. [RSS Feed Analysis Tools](#rss-feed-analysis-tools) (4 tools)
6. [Confluence Documentation Tools](#confluence-documentation-tools) (5 tools)
7. [GCP Service Discovery Tools](#gcp-service-discovery-tools) (8 tools)
8. [Service Evaluation Tools](#service-evaluation-tools) (2 tools)
9. [Standard BigQuery Tools](#standard-bigquery-tools) (7 tools)
10. [Common Pitfalls](#-common-pitfalls-and-best-practices)
11. [Recommended Tool Chains](#-recommended-tool-chains)
12. [Troubleshooting](#-troubleshooting)
13. [Usage Examples](#usage-examples)
14. [Version History](#-version-history)

---

## 🎯 Quick Tool Selector

**I want to...**
- 📊 Get security overview → `get_security_insights_summary()`
- 🔍 Find specific findings → `query_security_insights()`
- 📈 See distribution stats → `get_security_statistics()`
- 🆕 Evaluate new service → `evaluate_new_service()`
- ✅ Check compliance → `check_service_compliance()`
- 🔎 Discover services → `discover_gcp_services()`
- 📚 Search docs → `search_confluence_documentation()`
- 🚨 Monitor threats → `query_security_threat_feeds()`
- 📰 Track GCP changes → `analyze_gcp_releases()`
- 💾 Run custom SQL → `run_query()`

## Quick Reference Matrix

| Tool Name | Speed | BigQuery Cost | Use Case |
|-----------|-------|---------------|----------|
| get_security_insights_summary | ⚡ Fast (<2s) | ~$0.01 | Quick overview of security findings |
| query_security_insights | ⚡ Fast (2-5s) | ~$0.02-0.05 | Filter specific findings |
| get_security_statistics | ⚡ Fast (2-5s) | ~$0.03 | Distribution analysis |
| evaluate_new_service | 🐢 Slow (5-10s) | $0 (No BQ) | Full security assessment |
| check_service_compliance | 🏃 Medium (2-5s) | ~$0.05-0.10 | Compliance validation |
| discover_gcp_services | 🏃 Medium (3-7s) | $0 (API only) | Service catalog |
| analyze_gcp_service | 🏃 Medium (3-8s) | ~$0.05 | Service-specific analysis |
| analyze_gcp_releases | 🐌 Very Slow (30-60s) | ~$0.10 | Release impact assessment |
| search_confluence_documentation | ⚡ Fast (1-3s) | $0 (Cache) | Policy documentation |
| query_security_threat_feeds | 🏃 Medium (5-10s) | $0 (RSS) | CVE and threat monitoring |
| run_query | ⚡-🐌 Varies | Varies ($0.01-$10+) | Custom SQL analysis |
| analyze_query_cost | ⚡ Fast (<1s) | $0 (Dry run) | Cost estimation |

**Legend**: ⚡ Fast (<2s) | 🏃 Medium (2-5s) | 🐢 Slow (5-10s) | 🐌 Very Slow (>10s)

---

## Security-Focused Tools

### 1. get_security_insights_summary

**Performance**: ⚡ Fast (<2s) | **Cost**: ~$0.01 per query

**Purpose**: Summarize the primary security findings table with structured metrics.

**Source**: `agents/_tools/security_tools.py`

**Parameters**: None

**Returns**:
- `StructuredToolResponse` containing:
  - `summary`: Formatted text with table statistics and metrics
  - `data`: Dictionary with dataset info, table details, and metrics
  - `metadata`: Query used to generate the summary

**Description**: Provides a high-level overview of the security_insights dataset including total records, unique categories, severity levels, resource types, and date range of findings.

**Example Queries**:
- "What's the overall status of our security findings?"
- "Give me a summary of the security insights dataset"
- "How many security findings do we have?"

**Response Example**:
```
📊 Security Insights Summary (security_insights.security_findings):
   Table Size: 1,234 rows, 512,345 bytes
   Total Records: 1,234
   Unique Categories: 8
   Severity Levels: 4
   Resource Types: 12
   Date Range: 2025-01-15 to 2025-10-07
```

**Related Tools**: `query_security_insights()`, `get_security_statistics()`

---

### 2. query_security_insights

**Purpose**: Query the security findings table with optional filtering.

**Source**: `agents/_tools/security_tools.py`

**Parameters**:
- `query_filter` (str, optional): SQL WHERE clause to filter results (e.g., "severity='HIGH'")
- `limit` (int, optional): Maximum number of results to return (default: MAX_RESULTS)

**Returns**:
- `StructuredToolResponse` containing:
  - `summary`: Formatted text with query results
  - `data`: Dictionary with dataset info, row count, columns, and records
  - `metadata`: Executed SQL query

**Description**: Queries the security_insights table with custom filters and returns matching findings with full details.

**Example Queries**:
- "Show me all high severity security findings"
- "Find security issues from the last week WHERE created_at > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)"
- "List all security findings for BigQuery resources"

---

### 3. get_security_statistics

**Purpose**: Provide aggregated statistics from the security findings table.

**Source**: `agents/_tools/security_tools.py`

**Parameters**:
- `group_by` (str, optional): Field to group statistics by (default: "severity")
  - Valid values: severity, category, resource_type, status, region

**Returns**:
- `StructuredToolResponse` containing:
  - `summary`: Formatted text with distribution statistics
  - `data`: Dictionary with group_by field, total records, and distribution details
  - `metadata`: Query used for aggregation

**Description**: Generates aggregated statistics showing the distribution of security findings by a specified field.

**Example Queries**:
- "What's the breakdown of security findings by severity?"
- "Show me security statistics grouped by resource type"
- "Analyze security findings by region"

---

## Data Exploration Tools

### 4. explore_all_tables_and_views

**Purpose**: List all tables AND views in a dataset, distinguishing between them.

**Source**: `agents/_tools/exploration_tools.py`

**Parameters**:
- `dataset_id` (str, optional): Dataset ID to explore (default: DEFAULT_DATASET)

**Returns**: String with formatted list of tables, views, and external tables with metadata

**Description**: Provides a comprehensive overview of all objects (tables, views, external tables) in a BigQuery dataset with row counts, sizes, and descriptions.

**Example Queries**:
- "What tables and views exist in the security_insights dataset?"
- "Explore the dataset and show me all available objects"
- "List all tables in the dataset"

---

### 5. analyze_table_or_view

**Purpose**: Analyze a table or view to understand its structure and content.

**Source**: `agents/_tools/exploration_tools.py`

**Parameters**:
- `dataset_id` (str, required): Dataset ID containing the object
- `object_id` (str, required): Table or view ID to analyze

**Returns**: String with detailed analysis including schema, sample data, and metadata

**Description**: Provides in-depth analysis of a specific table or view including schema details, data samples, and statistics.

**Example Queries**:
- "Analyze the security_findings table"
- "Show me details about the gcp_release_notes view"
- "What's the structure of the security_threat_feeds table?"

---

## RSS Feed Analysis Tools

### 6. query_gcp_release_notes

**Purpose**: Query Google Cloud Platform release notes from RSS feeds.

**Source**: `agents/_tools/feed_tools.py`

**Parameters**:
- `days_back` (int, optional): Number of days to search (default: 30)
- `security_only` (bool, optional): Only return security-related notes (default: False)
- `service_category` (str, optional): Filter by service category
- `min_security_score` (int, optional): Minimum security score 0-10 (default: 0)

**Returns**: Formatted string with release notes matching criteria

**Description**: Searches GCP release notes feed for recent updates, with filtering options for security relevance and service categories.

**Example Queries**:
- "What are the latest GCP release notes from the last 7 days?"
- "Show me security-related release notes for compute services"
- "Find GCP updates with high security scores"

---

### 7. query_security_threat_feeds

**Purpose**: Query security threat feeds (CVE, advisories, threat intelligence).

**Source**: `agents/_tools/feed_tools.py`

**Parameters**:
- `days_back` (int, optional): Number of days to search (default: 7)
- `severity` (str, optional): Filter by severity (critical, high, medium, low)
- `threat_type` (str, optional): Filter by threat type (vulnerability, malware, phishing)
- `min_cvss_score` (float, optional): Minimum CVSS score 0.0-10.0 (default: 0.0)
- `cloud_related_only` (bool, optional): Only cloud-related threats (default: False)
- `immediate_action_only` (bool, optional): Only threats requiring immediate action (default: False)

**Returns**: Formatted string with security threats matching criteria

**Description**: Searches security threat feeds for vulnerabilities, CVEs, and security advisories with comprehensive filtering options.

**Example Queries**:
- "Show me critical security threats from the last 7 days"
- "Find CVEs affecting cloud services with CVSS score above 8"
- "List all security vulnerabilities requiring immediate action"

---

### 8. get_feed_statistics

**Purpose**: Get statistics about the RSS feed data.

**Source**: `agents/_tools/feed_tools.py`

**Parameters**: None

**Returns**: Formatted string with feed statistics and freshness

**Description**: Provides comprehensive statistics about both GCP release notes and security threat feeds including counts, recency, and severity breakdowns.

**Example Queries**:
- "What's the status of our RSS feeds?"
- "Show me statistics for security feeds"
- "How fresh is our feed data?"

---

### 9. search_feeds_by_keyword

**Purpose**: Search across all RSS feeds by keyword.

**Source**: `agents/_tools/feed_tools.py`

**Parameters**:
- `keyword` (str, required): Keyword to search for in titles and descriptions
- `days_back` (int, optional): Number of days to search (default: 30)
- `include_release_notes` (bool, optional): Include GCP release notes (default: True)
- `include_security_feeds` (bool, optional): Include security threat feeds (default: True)

**Returns**: Formatted string with search results

**Description**: Performs keyword search across both release notes and security feeds with flexible filtering options.

**Example Queries**:
- "Search for 'encryption' in all feeds"
- "Find mentions of 'BigQuery' in release notes from the last 14 days"
- "Look for security threats mentioning 'authentication'"

---

## Confluence Documentation Tools

### 10. search_confluence_documentation

**Purpose**: Search Confluence documentation for security policies and procedures.

**Source**: `agents/_tools/confluence_tools.py`

**Parameters**:
- `query` (str, required): Search query string (supports CQL syntax)
- `spaces` (List[str], optional): List of Confluence spaces to search (default: configured spaces)
- `limit` (int, optional): Maximum number of results (default: 10)
- `use_cache` (bool, optional): Whether to check cache first (default: True)

**Returns**: Dictionary containing search results with document metadata

**Description**: Searches Confluence spaces for documentation using intelligent caching to minimize API calls. Supports both live API queries and cached fallback.

**Example Queries**:
- "Search Confluence for GCP security policies"
- "Find IAM best practices in the security space"
- "Look for network security documentation"

---

### 11. get_confluence_document

**Purpose**: Retrieve a specific Confluence document by ID.

**Source**: `agents/_tools/confluence_tools.py`

**Parameters**:
- `document_id` (str, required): Confluence document/page ID
- `use_cache` (bool, optional): Whether to check cache first (default: True)
- `include_content` (bool, optional): Whether to include full content (default: True)

**Returns**: Dictionary containing document details

**Description**: Fetches a specific Confluence page with full metadata and content, using cache when available.

**Example Queries**:
- "Get Confluence document 123456789"
- "Retrieve the security policy page"
- "Show me the content of document ID XYZ"

---

### 12. analyze_confluence_coverage

**Purpose**: Analyze documentation coverage for specified security topics.

**Source**: `agents/_tools/confluence_tools.py`

**Parameters**:
- `topics` (List[str], required): List of topics to check coverage for
- `spaces` (List[str], optional): Confluence spaces to analyze (default: configured spaces)

**Returns**: Dictionary containing coverage analysis and recommendations

**Description**: Evaluates whether specified topics are adequately documented in Confluence and provides recommendations for gaps.

**Example Queries**:
- "Check if we have documentation for IAM, encryption, and network security"
- "Analyze coverage for GCP security topics"
- "What security topics are missing documentation?"

---

### 13. get_confluence_statistics

**Purpose**: Get statistics about cached Confluence documentation.

**Source**: `agents/_tools/confluence_tools.py`

**Parameters**: None

**Returns**: Dictionary containing cache statistics and configuration

**Description**: Provides detailed statistics about the Confluence cache including document counts, space breakdown, cache age, and configuration settings.

**Example Queries**:
- "What's the status of our Confluence cache?"
- "How many documents are cached?"
- "When was the cache last updated?"

---

### 14. refresh_confluence_cache

**Purpose**: Refresh the Confluence cache by fetching latest documents.

**Source**: `agents/_tools/confluence_tools.py`

**Parameters**:
- `spaces` (List[str], optional): Specific spaces to refresh (default: all configured)
- `force` (bool, optional): Force refresh even if cache is fresh (default: False)

**Returns**: Dictionary containing refresh status and statistics

**Description**: Manually triggers a cache refresh to fetch the latest Confluence documents from specified spaces.

**Example Queries**:
- "Refresh the Confluence cache"
- "Update cached documentation for the SEC space"
- "Force a full cache refresh"

---

## GCP Service Discovery Tools

### 15. discover_gcp_services

**Purpose**: Discover all available GCP services in the project.

**Source**: `agents/_tools/service_discovery.py`

**Parameters**:
- `include_learned` (bool, optional): Include services learned from documentation URLs (default: True)

**Returns**: Formatted string listing all discovered services with status and resource types

**Description**: Uses Cloud Asset API (if available) to discover enabled GCP services, falling back to catalog of known services.

**Example Queries**:
- "What GCP services are available?"
- "List all enabled GCP services"
- "Show me the service catalog"

---

### 16. analyze_gcp_service

**Purpose**: Perform on-demand analysis of any GCP service.

**Source**: `agents/_tools/service_discovery.py`

**Parameters**:
- `service_name` (str, required): Name or key of the GCP service to analyze
- `analysis_type` (str, optional): Type of analysis (default: "security")
  - Options: security, compliance, cost, usage, custom
- `custom_query` (str, optional): Optional custom BigQuery SQL for analysis

**Returns**: Formatted analysis results with findings and recommendations

**Description**: Executes predefined or custom security analysis queries against BigQuery data for any GCP service.

**Example Queries**:
- "Analyze security of Cloud Storage"
- "Run compliance analysis on BigQuery"
- "Perform cost analysis for Compute Engine"

---

### 17. get_service_resources

**Purpose**: List all resources for a specific GCP service.

**Source**: `agents/_tools/service_discovery.py`

**Parameters**:
- `service_name` (str, required): Name or key of the GCP service

**Returns**: Formatted list of resources grouped by type

**Description**: Lists all resources (instances, buckets, clusters, etc.) for a given GCP service using Cloud Asset API.

**Example Queries**:
- "What Cloud Storage resources exist?"
- "List all Compute Engine instances"
- "Show me BigQuery datasets and tables"

---

### 18. suggest_service_analysis

**Purpose**: Suggest relevant analyses for a GCP service.

**Source**: `agents/_tools/service_discovery.py`

**Parameters**:
- `service_name` (str, required): Name or key of the GCP service

**Returns**: Formatted suggestions with available analysis types and commands

**Description**: Provides recommendations for security, compliance, and operational analyses available for a specific service.

**Example Queries**:
- "What analyses are available for Cloud SQL?"
- "Suggest security checks for Kubernetes Engine"
- "What can I analyze about Pub/Sub?"

---

### 19. learn_service_from_url

**Purpose**: Learn about a new GCP service by parsing its documentation URL.

**Source**: `agents/_tools/service_discovery.py`

**Parameters**:
- `documentation_url` (str, required): URL to GCP service documentation

**Returns**: Analysis of the learned service

**Description**: Parses GCP service documentation from a URL to automatically learn about new or unfamiliar services.

**Example Queries**:
- "Learn about this service from https://cloud.google.com/service/docs"
- "Parse documentation for a new GCP service"
- "Add a service from its documentation URL"

---

### 20. discover_new_gcp_services

**Purpose**: Discover newly released GCP services from release notes.

**Source**: `agents/_tools/service_discovery.py`

**Parameters**:
- `release_notes_url` (str, optional): Optional URL to release notes page

**Returns**: List of newly discovered services

**Description**: Scans GCP release notes to identify and learn about newly announced services.

**Example Queries**:
- "Check for new GCP services"
- "What services were recently announced?"
- "Discover new GCP offerings"

---

### 21. register_new_service

**Purpose**: Register a new GCP service manually for analysis.

**Source**: `agents/_tools/service_discovery.py`

**Parameters**:
- `service_name` (str, required): Name of the new service
- `api_endpoint` (str, required): API endpoint (e.g., newservice.googleapis.com)
- `documentation_url` (str, required): URL to service documentation
- `description` (str, optional): Brief description of the service

**Returns**: Registration status

**Description**: Manually registers a new GCP service into the discovery system for future analysis.

**Example Queries**:
- "Register the new Cloud XYZ service"
- "Add a custom service to the catalog"
- "Register newservice.googleapis.com"

---

### 22. learn_from_api_spec

**Purpose**: Learn about a service from its API specification.

**Source**: `agents/_tools/service_discovery.py`

**Parameters**:
- `api_spec_url` (str, required): URL to API specification (OpenAPI, Proto, etc.)

**Returns**: Parsed API information

**Description**: Parses OpenAPI, Protocol Buffer, or other API specifications to understand service capabilities.

**Example Queries**:
- "Parse this OpenAPI spec"
- "Learn from the service API specification"
- "Analyze the API at this URL"

---

### 23. analyze_gcp_releases

**Performance**: 🐌 Very Slow (30-60s) | **Cost**: ~$0.10 (BigQuery + RSS parsing)

**Purpose**: Multi-Service Analyzer (MSA) for GCP release notes impact assessment across security, billing, and compliance domains.

**Source**: `agents/_tools/msa_analyzer.py`

**Parameters**:
- `days_back` (int, optional): Number of days to look back for release notes (default: 7)

**Returns**:
- Comprehensive `Dict[str, Any]` impact analysis report containing:
  - `analysis_id`: Unique identifier for this analysis
  - `timestamp`: Analysis execution time
  - `summary`: High-level metrics (total changes, affected services, risk score, critical issues)
  - `breakdown_by_category`: Count of changes by category (security, billing, deprecation, compliance, etc.)
  - `security_impact`: Detailed security impact analysis including:
    - `risk_level`: Overall security risk (low/medium/high)
    - `critical_updates`: List of critical security updates requiring immediate action
    - `authentication_changes`: IAM and auth-related changes
    - `encryption_changes`: Encryption and TLS modifications
    - `permission_changes`: Role and policy updates
    - `vulnerabilities_fixed`: CVEs and security patches
    - `action_required`: List of required security actions
  - `billing_impact`: Cost impact analysis including:
    - `estimated_impact`: Overall billing impact (increase/decrease/neutral)
    - `pricing_changes`: Price increases or decreases by service
    - `free_tier_changes`: Modifications to free tier limits
    - `new_charges`: New billing items introduced
    - `cost_optimization_opportunities`: Potential savings identified
  - `compliance_impact`: Regulatory compliance analysis including:
    - `impact_level`: Compliance risk level (low/medium/high)
    - `new_certifications`: New compliance certifications achieved
    - `lost_certifications`: Deprecated compliance status
    - `regulation_changes`: Regulatory requirement updates
    - `data_residency_changes`: Data location changes
    - `audit_requirements`: New audit obligations
  - `recommendations`: Prioritized actionable recommendations by priority level
  - `affected_services`: List of GCP services impacted by changes
  - `action_items`: Categorized by urgency (immediate, within 7 days, 30 days, 90 days)
  - `release_notes_analyzed`: Top 10 release notes for reference

**Description**:
The Multi-Service Analyzer (MSA) is a sophisticated tool that monitors GCP's official release notes RSS feed and performs comprehensive impact assessment across three critical domains:

1. **Security Impact Analysis**: Identifies critical updates, authentication changes, encryption modifications, permission changes, and vulnerability fixes. Assigns risk scores and generates security-specific action items.

2. **Billing Impact Analysis**: Detects pricing changes, free tier modifications, new charges, and cost optimization opportunities. Estimates overall financial impact (increase/decrease/neutral).

3. **Compliance Impact Analysis**: Tracks certification changes, regulatory updates, data residency requirements, and audit obligations. Critical for regulated industries (HIPAA, PCI-DSS, SOX, GDPR).

The tool intelligently filters release notes to only analyze services currently active in your environment, reducing noise and focusing on relevant changes. It uses natural language processing to categorize changes, assess severity, and generate prioritized recommendations with specific deadlines.

**Example Queries**:
- "Analyze GCP releases from the last 7 days"
- "What's the security impact of recent GCP changes?"
- "Check for billing changes in recent releases"
- "Are there any compliance-related updates I need to know about?"
- "Show me critical updates requiring immediate action"

**Response Example**:
```
📊 ANALYSIS SUMMARY
   Analysis ID: abc123def456
   Changes Analyzed: 42
   Services Affected: 8
   Overall Risk: MEDIUM (Score: 45)
   Critical Issues: 2

🔒 SECURITY IMPACT
   Risk Level: MEDIUM
   ⚠️  Critical Updates: 2
   🔐 Auth Changes: 3
   ✅ Vulnerabilities Fixed: 5

💰 BILLING IMPACT
   Estimated Impact: INCREASE
   💵 Pricing Changes: 4
   💡 Optimization Opportunities: 2

📋 COMPLIANCE IMPACT
   Impact Level: LOW
   ✅ New Certifications: 1

🎯 TOP RECOMMENDATIONS
   1. 🔴 [CRITICAL] Apply critical security update for Cloud Storage
      Deadline: immediate
   2. 🟠 [HIGH] Review budget impact for BigQuery
      Deadline: 7 days
   3. 🟡 [MEDIUM] Test authentication and permissions
      Deadline: 14 days
```

**Best Practices**:
- Run weekly to stay current with GCP changes
- Review critical updates immediately
- Schedule dedicated time for impact analysis
- Document compliance-related changes for audit trails
- Use with `search_confluence_documentation()` to check internal policies

**Related Tools**:
- `query_gcp_release_notes()` - For broader release note queries
- `query_security_threat_feeds()` - For CVE monitoring
- `search_feeds_by_keyword()` - For targeted searches
- `discover_gcp_services()` - To update active services list

**Performance Notes**:
- First run (cold start): 45-60 seconds due to RSS parsing and web scraping
- Subsequent runs (warm cache): 30-40 seconds
- Processes 50-100 release notes per execution
- Stores results in BigQuery for historical analysis
- Uses GCS for caching processed release note IDs

---

## Service Evaluation Tools

### 24. evaluate_new_service

**Performance**: 🐢 Slow (5-10s) | **Cost**: $0 (No BigQuery unless check_current_compliance=True)

**Purpose**: Evaluate a new GCP service for security, compliance, and risk through comprehensive multi-domain assessment.

**Source**: `agents/_tools/service_evaluation/evaluator.py`

**Parameters**:
- `service_name` (str, required): Name of the GCP service (e.g., "Cloud Storage", "BigQuery")
- `service_type` (str, required): Service type category (storage, compute, database, networking, analytics, ml)
- `service_profile` (str, optional): JSON string with detailed service profile for advanced risk assessment
- `use_case` (str, optional): Description of how the service will be used
- `data_classification` (str, optional): Data sensitivity level (public, internal, confidential, restricted)
- `check_current_compliance` (bool, optional): Check actual environment compliance via BigQuery (default: False, adds ~$0.05 cost)
- `return_format` (str, optional): Output format - 'object', 'dict', or 'summary' (default: 'object')

**Returns**:
- `ServiceEvaluationResult` object containing:
  - `risk_assessment`: Complete risk analysis with scores, severity, and mitigation priorities
  - `applicable_controls`: List of security controls mapped to service type
  - `controls_by_category`: Controls grouped by category (IAM, encryption, network, logging, etc.)
  - `controls_by_severity`: Count of controls by severity level (critical/high/medium/low)
  - `enforcement_options`: Available enforcement methods (Org Policies, Terraform, IAM, etc.)
  - `enforcement_by_method`: Enforcement options grouped by implementation method
  - `approval_workflow`: Required approval steps, approvers, artifacts, and SLA
  - `compliance_report`: (If check_current_compliance=True) Current compliance status from BigQuery
  - `compliance_gaps`: (If enabled) List of compliance violations to remediate
  - `recommendations`: Actionable security recommendations prioritized by risk
  - `next_steps`: Ordered implementation steps with timelines
  - `summary`: High-level metrics and scores

**Description**:
The Service Evaluator performs a comprehensive multi-layer assessment of new GCP services before deployment:

1. **Risk Assessment Engine**: Evaluates 8 risk factors including data sensitivity, network exposure, authentication strength, encryption status, compliance requirements, third-party integrations, data volume, and operational maturity. Produces weighted risk score (0-100) and risk level (low/medium/high/critical).

2. **Security Controls Inventory**: Maps 50+ security controls to the service type across 7 categories: IAM, encryption, network security, logging & monitoring, data protection, compliance, and operational security. Each control includes severity, compliance mappings (CIS, NIST, PCI-DSS, HIPAA, SOX, GDPR), and validation queries.

3. **Enforcement Analyzer**: Provides 3-5 enforcement options per control across multiple methods: Organization Policies (preventive, auto-enforced), Terraform (IaC, version-controlled), Cloud Functions (reactive, automated), Security Command Center (detective, monitoring), and IAM Policies (access control). Each option includes complexity, automation level, cost estimate, and maintenance effort.

4. **Approval Workflow Engine**: Determines required approvals based on risk level, including technical review, security review, compliance review, and executive approval for high-risk services. Provides SLA timelines, required artifacts (architecture diagrams, threat models, compliance checklists), and escalation criteria.

5. **Compliance Checker** (Optional): Queries BigQuery to validate actual environment state against security controls. Identifies gaps between policy (what SHOULD be) and reality (what IS). Returns compliance score, violation count by severity, and specific remediation steps for each gap.

**Example Queries**:
- "Evaluate Cloud Storage for confidential data usage"
- "Assess security requirements for a new BigQuery project with check_current_compliance=True"
- "What controls are needed for Cloud SQL with HIPAA data?"
- "Evaluate Compute Engine for internal applications with public internet exposure"

**Response Example** (summary format):
```
═══════════════════════════════════════════════════════════════
  GCP SERVICE SECURITY EVALUATION REPORT
═══════════════════════════════════════════════════════════════

SERVICE: Cloud Storage (storage)
EVALUATED: 2025-10-07T14:32:15Z

RISK ASSESSMENT
───────────────────────────────────────────────────────────────
Risk Level:  HIGH
Risk Score:  72/100

SECURITY CONTROLS
───────────────────────────────────────────────────────────────
Total Controls:     23
  - Critical:       5
  - High:           8

ENFORCEMENT
───────────────────────────────────────────────────────────────
Total Options:      18
Fully Automated:    12

APPROVAL WORKFLOW
───────────────────────────────────────────────────────────────
Approvals Required: 3
Timeline:           7 days

COMPLIANCE STATUS (Current Environment)
───────────────────────────────────────────────────────────────
Compliance Score:   67%
Violations Found:   8 (3 critical, 5 high)
Gaps to Remediate:  8

KEY RECOMMENDATIONS
───────────────────────────────────────────────────────────────
⚠️ HIGH RISK SERVICE: Risk score 72/100. Prioritize security controls.
🔒 Implement 5 critical security controls before deployment.
✅ 12 controls can be fully automated using Org Policies and Terraform.
🔐 High data sensitivity detected. Implement CMEK encryption and DLP.

NEXT STEPS
───────────────────────────────────────────────────────────────
1. Review this complete evaluation report with your team
2. Prioritize implementation of critical and high-severity controls
3. Prepare required approval artifacts (architecture diagram, threat model, compliance checklist)
4. Run automated pre-flight checks (8 checks available)
5. Submit for approval (estimated timeline: 7 days)
```

**Best Practices**:
- Always specify `service_type` accurately for correct control mapping
- Enable `check_current_compliance=True` to get actual environment state
- Use `data_classification='confidential'` or `'restricted'` for sensitive data
- Review all compliance_gaps before deployment
- Implement automated enforcement (Org Policies) for critical controls first

**Related Tools**:
- `check_service_compliance()` - Validate compliance after evaluation
- `suggest_service_analysis()` - Get service-specific analysis recommendations
- `discover_gcp_services()` - Browse available service types
- `search_confluence_documentation()` - Find related security policies

---

### 25. check_service_compliance

**Performance**: 🏃 Medium (2-5s) | **Cost**: ~$0.05-0.10 per service type

**Purpose**: Validate security controls against actual BigQuery environment state - the "glue layer" bridging policy with reality.

**Source**: `agents/_tools/service_evaluation/compliance_checker.py`

**Parameters**:
- `service_type` (str, required): Service type to validate (storage, compute, bigquery, database, networking, analytics)
- `detailed` (bool, optional): Include full violation details with resource names and remediation steps (default: False)

**Returns**:
- JSON string containing `ComplianceReport`:
  - `service_type`: Service type checked
  - `total_controls_checked`: Number of controls validated
  - `controls_passed`: Count of compliant controls
  - `controls_failed`: Count of non-compliant controls
  - `compliance_score`: Percentage score (0-100)
  - `total_violations`: Total violation count across all controls
  - `violations_by_severity`: Breakdown of violations (critical/high/medium/low)
  - `control_statuses`: (If detailed=True) Per-control status with violations
  - `summary`: Human-readable compliance summary

**Description**:
The Compliance Checker is the critical "glue layer" that connects security policy (what SHOULD be enforced) with environment reality (what IS currently deployed). It executes validation queries against BigQuery tables containing actual GCP resource configurations to identify gaps between desired and actual state.

**How It Works**:
1. Maps service_type to applicable security controls
2. Retrieves validation queries for each control from the Security Controls Inventory
3. Executes queries against BigQuery tables (storage_buckets, compute_instances, etc.)
4. Compares results against compliance criteria
5. Generates compliance report with violations, severity, and remediation guidance

**Example Use Cases**:
- **Pre-Deployment Validation**: Check compliance before deploying new services
- **Continuous Monitoring**: Schedule daily/weekly compliance checks
- **Audit Preparation**: Generate compliance reports for auditors
- **Remediation Tracking**: Monitor progress on fixing violations
- **Drift Detection**: Identify when configurations deviate from policy

**Example Queries**:
- "Check compliance for storage services"
- "What security violations exist for compute resources with detailed=True?"
- "Validate BigQuery security controls and show me the gaps"
- "Run compliance check for database services"

**Response Example** (detailed=False):
```json
{
  "service_type": "storage",
  "total_controls_checked": 23,
  "controls_passed": 15,
  "controls_failed": 8,
  "compliance_score": 65.2,
  "total_violations": 12,
  "violations_by_severity": {
    "critical": 3,
    "high": 5,
    "medium": 3,
    "low": 1
  },
  "summary": "Storage compliance: 65.2% (15/23 controls passed). 8 controls failed with 12 violations (3 critical, 5 high)."
}
```

**Response Example** (detailed=True):
```json
{
  "service_type": "storage",
  "compliance_score": 65.2,
  "control_statuses": [
    {
      "control_id": "STR-001",
      "control_name": "Prevent public bucket access",
      "status": "FAIL",
      "severity": "critical",
      "violations": [
        {
          "resource_name": "gs://public-bucket-001",
          "resource_type": "storage_bucket",
          "violation_details": "Bucket allows public access via 'allUsers' IAM binding",
          "severity": "critical",
          "remediation": "Remove 'allUsers' from bucket IAM policy: gcloud storage buckets remove-iam-policy-binding gs://public-bucket-001 --member=allUsers --role=roles/storage.objectViewer"
        },
        {
          "resource_name": "gs://public-bucket-002",
          "resource_type": "storage_bucket",
          "violation_details": "Bucket configured with uniform bucket-level access disabled",
          "severity": "high",
          "remediation": "Enable uniform bucket-level access: gcloud storage buckets update gs://public-bucket-002 --uniform-bucket-level-access"
        }
      ]
    }
  ]
}
```

**Best Practices**:
- Run with `detailed=False` first for quick overview
- Use `detailed=True` when remediating violations
- Schedule regular compliance checks (daily for critical, weekly for others)
- Track compliance_score trends over time
- Prioritize critical and high severity violations first
- Store compliance reports in BigQuery for historical analysis

**Related Tools**:
- `evaluate_new_service()` - Includes compliance checking as part of evaluation
- `query_security_insights()` - Query existing security findings
- `get_security_statistics()` - Analyze security finding patterns
- `run_query()` - Execute custom compliance validation queries

---

## Standard BigQuery Tools

### 26. hello_world

**Purpose**: Execute a Hello World query in BigQuery.

**Source**: `agents/_tools/bigquery_tools.py`

**Parameters**: None

**Returns**: String with greeting, timestamp, project ID, and tags

**Description**: Simple test query to verify BigQuery connectivity and agent functionality.

**Example Queries**:
- "Test BigQuery connection"
- "Hello world"
- "Are you working?"

---

### 27. list_datasets

**Purpose**: List all BigQuery datasets in the project.

**Source**: `agents/_tools/bigquery_tools.py`

**Parameters**: None

**Returns**: String with dataset list including descriptions, locations, and creation dates

**Description**: Enumerates all BigQuery datasets in the current project with metadata highlighting the default dataset.

**Example Queries**:
- "What datasets exist?"
- "List all BigQuery datasets"
- "Show me available datasets"

---

### 28. list_tables

**Purpose**: List all tables in a specific BigQuery dataset.

**Source**: `agents/_tools/bigquery_tools.py`

**Parameters**:
- `dataset_id` (str, optional): Dataset ID (default: configured default dataset)

**Returns**: String with table list including types, row counts, and sizes

**Description**: Lists all tables in a dataset with detailed metadata including table types and sizes.

**Example Queries**:
- "What tables are in security_insights?"
- "List all tables in the dataset"
- "Show me table sizes"

---

### 29. get_table_schema

**Purpose**: Get the schema of a specific BigQuery table.

**Source**: `agents/_tools/bigquery_tools.py`

**Parameters**:
- `dataset_id` (str, optional): Dataset ID (default: configured default)
- `table_id` (str, optional): Table ID (default: configured default)

**Returns**: String with complete schema including field names, types, modes, and descriptions

**Description**: Displays the complete schema definition for a BigQuery table.

**Example Queries**:
- "What's the schema of security_findings?"
- "Show me table structure"
- "What columns exist in the table?"

---

### 30. run_query

**Purpose**: Execute any BigQuery SQL query.

**Source**: `agents/_tools/bigquery_tools.py`

**Parameters**:
- `query` (str, required): BigQuery SQL query to execute

**Returns**: String with query results, column names, and execution statistics

**Description**: Executes arbitrary SQL queries against BigQuery with results limited to prevent overwhelming output.

**Example Queries**:
- "Run this query: SELECT * FROM security_findings WHERE severity='HIGH'"
- "Execute a custom SQL query"
- "Query the database for specific data"

---

### 31. analyze_query_cost

**Purpose**: Analyze the estimated cost and data processed for a query without running it.

**Source**: `agents/_tools/bigquery_tools.py`

**Parameters**:
- `query` (str, required): BigQuery SQL query to analyze

**Returns**: String with bytes processed, estimated cost, and validation status

**Description**: Performs a dry run of the query to estimate costs before execution.

**Example Queries**:
- "How much will this query cost?"
- "Estimate cost for SELECT * FROM large_table"
- "Check query cost before running"

---

### 32. get_table_sample

**Purpose**: Get a sample of rows from a table.

**Source**: `agents/_tools/bigquery_tools.py`

**Parameters**:
- `dataset_id` (str, optional): Dataset ID (default: configured default)
- `table_id` (str, optional): Table ID (default: configured default)
- `limit` (int, optional): Number of rows to return (default: SAMPLE_LIMIT)

**Returns**: Query results with sample rows

**Description**: Retrieves a limited sample of rows from a table for quick data preview.

**Example Queries**:
- "Show me sample data from security_findings"
- "Get 10 rows from the table"
- "Preview the table data"

---

## ⚠️ Common Pitfalls and Best Practices

### Security Tools
- ❌ **Don't**: Use `run_query()` for simple security queries
- ✅ **Do**: Use `query_security_insights()` - it's optimized and cached
- ❌ **Don't**: Query without filters on large tables
- ✅ **Do**: Always add WHERE clauses for date ranges and severity filters
- ❌ **Don't**: Ignore the StructuredToolResponse format
- ✅ **Do**: Parse the `data` and `metadata` fields for programmatic access

### Service Evaluation
- ❌ **Don't**: Call `evaluate_new_service()` without specifying `service_type`
- ✅ **Do**: Always provide service_type for accurate control mapping
- ❌ **Don't**: Skip `check_current_compliance=True` parameter
- ✅ **Do**: Enable it to get actual environment state from BigQuery
- ❌ **Don't**: Ignore the compliance_gaps in the response
- ✅ **Do**: Review and remediate all compliance violations before deployment

### Service Discovery
- ❌ **Don't**: Assume all GCP services are pre-configured
- ✅ **Do**: Use `learn_service_from_url()` for new or unfamiliar services
- ❌ **Don't**: Run `analyze_gcp_service()` without checking table existence
- ✅ **Do**: Run `discover_gcp_services()` first to see what's available

### BigQuery Tools
- ❌ **Don't**: Use `run_query()` without checking cost first for large queries
- ✅ **Do**: Call `analyze_query_cost()` for queries scanning >1GB
- ❌ **Don't**: Query production tables directly in testing environments
- ✅ **Do**: Use `get_table_sample()` for data preview and testing
- ❌ **Don't**: Forget to add LIMIT clauses on unbounded queries
- ✅ **Do**: Always use LIMIT 1000 or appropriate row limits

### RSS Feed Analysis
- ❌ **Don't**: Run `analyze_gcp_releases()` every hour (it's slow and expensive)
- ✅ **Do**: Schedule it weekly or when critical changes are expected
- ❌ **Don't**: Ignore the `days_back` parameter - defaults to 7 days
- ✅ **Do**: Adjust `days_back` based on your analysis needs (7-30 days typical)

### Confluence Integration
- ❌ **Don't**: Call Confluence API without cache (`use_cache=False`)
- ✅ **Do**: Always enable cache to minimize API calls and respect rate limits
- ❌ **Don't**: Search without specifying spaces
- ✅ **Do**: Limit searches to relevant spaces (SEC, POLICY, GCP) for faster results
- ❌ **Don't**: Expect real-time updates if cache is stale
- ✅ **Do**: Call `refresh_confluence_cache()` periodically to keep data fresh

---

## 🔗 Recommended Tool Chains

### Chain 1: New Service Onboarding Workflow
**Use Case**: Evaluating and onboarding a new GCP service to your environment

**Steps**:
```python
# 1. Learn about the service from documentation
learn_service_from_url('https://cloud.google.com/service-name/docs')

# 2. Perform comprehensive security evaluation
evaluate_new_service(
    service_name='Service Name',
    service_type='compute',  # or storage, database, etc.
    data_classification='confidential',
    use_case='Production workload processing',
    check_current_compliance=True  # Check actual environment state
)

# 3. Validate against security controls
check_service_compliance(
    service_type='compute',
    detailed=True  # Get full violation details
)

# 4. Get specific analysis recommendations
suggest_service_analysis('Service Name')

# 5. Search for relevant policies in Confluence
search_confluence_documentation(
    query='compute security policies',
    spaces=['SEC', 'POLICY']
)
```

**Expected Timeline**: 15-20 minutes for complete evaluation

---

### Chain 2: Security Investigation and Incident Response
**Use Case**: Investigating security findings and determining scope of issues

**Steps**:
```python
# 1. Get high-level overview
get_security_insights_summary()

# 2. Filter to critical/high severity incidents
query_security_insights(
    query_filter="severity IN ('CRITICAL', 'HIGH') AND created_at > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)",
    limit=100
)

# 3. Analyze patterns and distributions
get_security_statistics(group_by='resource_type')
get_security_statistics(group_by='category')

# 4. Deep dive with custom analysis
run_query("""
    SELECT
        resource_type,
        category,
        COUNT(*) as finding_count,
        STRING_AGG(DISTINCT resource_name, ', ') as affected_resources
    FROM `security_insights.security_findings`
    WHERE severity = 'CRITICAL'
    AND created_at > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
    GROUP BY resource_type, category
    ORDER BY finding_count DESC
""")

# 5. Check for related vulnerabilities
query_security_threat_feeds(
    severity='critical',
    cloud_related_only=True,
    days_back=7
)
```

**Expected Timeline**: 5-10 minutes for comprehensive investigation

---

### Chain 3: Compliance Audit Preparation
**Use Case**: Preparing for security audit or compliance review

**Steps**:
```python
# 1. Discover all active GCP services
discover_gcp_services(include_learned=True)

# 2. Analyze each critical service
analyze_gcp_service('storage', 'security')
analyze_gcp_service('compute', 'compliance')
analyze_gcp_service('bigquery', 'security')

# 3. Check compliance for each service type
check_service_compliance('storage', detailed=True)
check_service_compliance('compute', detailed=True)
check_service_compliance('database', detailed=True)

# 4. Verify documentation coverage
analyze_confluence_coverage(
    topics=['IAM', 'Encryption', 'Network Security', 'Data Classification', 'Incident Response']
)

# 5. Search for relevant policies
search_confluence_documentation(
    query='compliance AND (HIPAA OR PCI OR SOX)',
    spaces=['SEC', 'POLICY', 'COMPLIANCE']
)

# 6. Generate compliance report
run_query("""
    SELECT
        service_type,
        COUNT(*) as total_resources,
        COUNTIF(encryption_enabled) as encrypted_count,
        COUNTIF(public_access) as public_access_count,
        COUNTIF(iam_compliant) as iam_compliant_count
    FROM `security_insights.resource_inventory`
    GROUP BY service_type
    ORDER BY total_resources DESC
""")
```

**Expected Timeline**: 20-30 minutes for comprehensive audit

---

### Chain 4: Weekly Security Posture Review
**Use Case**: Regular weekly security review and monitoring

**Steps**:
```python
# 1. Analyze recent GCP service changes
analyze_gcp_releases(days_back=7)

# 2. Check for new security threats
query_security_threat_feeds(
    severity='high',
    cloud_related_only=True,
    immediate_action_only=True,
    days_back=7
)

# 3. Review security findings trends
get_security_statistics(group_by='severity')
get_security_statistics(group_by='category')

# 4. Check for new security findings
query_security_insights(
    query_filter="created_at > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)",
    limit=50
)

# 5. Search for updates to security policies
search_confluence_documentation(
    query='modified > now()-7d AND (security OR policy)',
    spaces=['SEC', 'POLICY']
)
```

**Expected Timeline**: 10-15 minutes for weekly review

---

### Chain 5: Service Migration Planning
**Use Case**: Planning migration from one service to another

**Steps**:
```python
# 1. Evaluate current service
analyze_gcp_service('old-service', 'security')
check_service_compliance('old-service-type', detailed=True)

# 2. Evaluate target service
evaluate_new_service(
    service_name='New Service',
    service_type='new-service-type',
    check_current_compliance=True
)

# 3. Compare security controls
run_query("""
    SELECT
        control_id,
        control_name,
        old_service_status,
        new_service_status
    FROM `security_insights.control_comparison`
    WHERE migration_id = 'old-to-new-migration'
""")

# 4. Check for migration guidance
search_confluence_documentation(
    query='migration AND old-service AND new-service',
    spaces=['ENGINEERING', 'SEC']
)

# 5. Review recent changes to both services
search_feeds_by_keyword(
    keyword='old-service OR new-service',
    days_back=90
)
```

**Expected Timeline**: 25-35 minutes for migration analysis

---

## 🔧 Troubleshooting

### Error: "No module named 'bs4'"
**Cause**: BeautifulSoup not installed in ADK environment

**Solution**:
```bash
# Install required dependencies in ADK virtualenv
/Users/stuartgano/.local/pipx/venvs/google-adk/bin/python3.13 -m pip install beautifulsoup4 lxml feedparser

# Or install in your local environment
pip install beautifulsoup4 lxml feedparser
```

**Affected Tools**: `learn_service_from_url()`, `discover_new_gcp_services()`, `analyze_gcp_releases()`

---

### Error: "Connection refused" on port 8000
**Cause**: ADK backend not running

**Solution**:
```bash
# Start ADK web interface
adk web

# Or specify custom port
adk web --port 8080
```

**Verification**:
```bash
# Check if backend is running
curl http://localhost:8000/health
```

---

### Error: "Type annotation error for function"
**Cause**: ADK requires `Optional[str]` not `str = None` for nullable parameters

**Solution**:
```python
# ❌ Wrong
def my_function(param: str = None):
    pass

# ✅ Correct
from typing import Optional

def my_function(param: Optional[str] = None):
    pass
```

---

### Error: "Table not found: security_insights.security_findings"
**Cause**: Security insights table hasn't been created or populated

**Solution**:
```bash
# Run data collection to populate tables
python scripts/collect_security_data.py

# Or create sample data for testing
python scripts/create_sample_data.py
```

**Prevention**: Run `explore_all_tables_and_views()` to verify table existence before querying

---

### Error: "Confluence API rate limit exceeded"
**Cause**: Too many API calls to Confluence without caching

**Solution**:
```python
# Always use cache
search_confluence_documentation(
    query='your search',
    use_cache=True  # Enable cache (default)
)

# Refresh cache during off-peak hours
refresh_confluence_cache(force=False)  # Respects cache TTL
```

**Rate Limits**: Confluence allows 100 requests/minute. Cache reduces calls by 95%+.

---

### Slow Response Times (>30 seconds)
**Cause**: Large table queries without limits or filters

**Solution**:
```python
# ❌ Slow query
run_query("SELECT * FROM large_table")

# ✅ Fast query with filters and limit
run_query("""
    SELECT * FROM large_table
    WHERE created_at > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
    AND severity = 'HIGH'
    LIMIT 1000
""")

# Check cost before running
analyze_query_cost("SELECT * FROM large_table")
```

**Performance Tips**:
- Always use WHERE clauses with date ranges
- Add LIMIT clauses to unbounded queries
- Use partitioned tables when possible
- Check query cost for queries >1GB scan

---

### Error: "Service 'X' not found in service catalog"
**Cause**: Service not in pre-defined catalog

**Solution**:
```python
# Option 1: Learn from documentation URL
learn_service_from_url('https://cloud.google.com/service-x/docs')

# Option 2: Manually register the service
register_new_service(
    service_name='Service X',
    api_endpoint='servicex.googleapis.com',
    documentation_url='https://cloud.google.com/service-x/docs',
    description='Description of Service X'
)

# Option 3: Check for new services
discover_new_gcp_services()
```

---

### Warning: "BigQuery not available for MSA"
**Cause**: BigQuery client initialization failed

**Solution**:
```bash
# Set required environment variables
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Verify credentials
gcloud auth application-default print-access-token

# Test BigQuery connection
bq ls
```

---

### Error: "Compliance check failed: Table does not exist"
**Cause**: Compliance validation tables not created

**Solution**:
```bash
# Create required compliance tables
python scripts/setup_compliance_tables.py

# Populate with current resource state
python scripts/collect_compliance_data.py
```

**Required Tables**:
- `security_insights.storage_buckets`
- `security_insights.compute_instances`
- `security_insights.bigquery_datasets`
- `security_insights.cloudsql_instances`

---

### Best Practices for Troubleshooting

1. **Enable Debug Logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **Check Tool Availability**:
```python
# Verify all tools are registered
from agents.security_agent import list_tools
print(list_tools())
```

3. **Test Individual Components**:
```python
# Test BigQuery connectivity
from agents._tools.bigquery_tools import hello_world
print(hello_world())
```

4. **Review ADK Logs**:
```bash
# Check ADK logs for errors
tail -f ~/.adk/logs/agent.log
```

5. **Validate Environment Variables**:
```bash
# Check all required variables are set
env | grep GOOGLE
```

---

## Usage Examples

### Security Analysis Workflow

```python
# 1. Get overview of security findings
get_security_insights_summary()

# 2. Analyze specific severity level
query_security_insights(query_filter="severity='CRITICAL'", limit=20)

# 3. Check distribution by resource type
get_security_statistics(group_by="resource_type")

# 4. Check compliance status
check_service_compliance(service_type="storage", detailed=True)
```

### Service Discovery and Evaluation

```python
# 1. Discover available services
discover_gcp_services(include_learned=True)

# 2. Analyze specific service
analyze_gcp_service(service_name="Cloud Storage", analysis_type="security")

# 3. Evaluate for new use case
evaluate_new_service(
    service_name="Cloud SQL",
    service_type="database",
    data_classification="confidential",
    check_current_compliance=True
)

# 4. Get compliance gaps
check_service_compliance(service_type="database", detailed=True)
```

### Release Notes Impact Assessment

```python
# 1. Analyze recent releases
analyze_gcp_releases(days_back=7)

# 2. Search for specific changes
search_feeds_by_keyword(keyword="encryption", days_back=30)

# 3. Check security threats
query_security_threat_feeds(
    severity="critical",
    cloud_related_only=True,
    days_back=7
)
```

### Documentation and Knowledge Search

```python
# 1. Search Confluence for policies
search_confluence_documentation(
    query="GCP security policies",
    spaces=["SEC", "POLICY"]
)

# 2. Check documentation coverage
analyze_confluence_coverage(
    topics=["IAM", "Encryption", "Network Security"]
)

# 3. Get specific document
get_confluence_document(document_id="123456789")
```

### Custom BigQuery Analysis

```python
# 1. Explore dataset
explore_all_tables_and_views(dataset_id="security_insights")

# 2. Analyze specific table
analyze_table_or_view(
    dataset_id="security_insights",
    object_id="security_findings"
)

# 3. Run custom analysis
run_query("""
    SELECT severity, COUNT(*) as count
    FROM security_insights.security_findings
    WHERE created_at > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
    GROUP BY severity
    ORDER BY count DESC
""")

# 4. Check query cost first
analyze_query_cost("SELECT * FROM large_table")
```

---

## Tool Categories Summary

| Category | Tool Count | Primary Use Case | Avg Performance | Avg Cost |
|----------|-----------|------------------|-----------------|----------|
| Security-Focused | 3 | Direct security findings analysis | ⚡ Fast (1-5s) | ~$0.02/query |
| Data Exploration | 2 | Dataset and table discovery | ⚡ Fast (2-3s) | ~$0.01/query |
| RSS Feed Analysis | 4 | Release notes and threat monitoring | 🏃 Medium (5-15s) | $0 (RSS feeds) |
| Confluence Documentation | 5 | Policy and procedure documentation | ⚡ Fast (1-3s) | $0 (Cached) |
| GCP Service Discovery | 8 | Service catalog and dynamic analysis | 🏃 Medium (3-10s) | ~$0.05/query |
| Service Evaluation | 2 | Security assessment and compliance | 🏃-🐢 Medium-Slow (5-10s) | ~$0.05-0.10/query |
| Standard BigQuery | 7 | General database operations | ⚡-🐌 Varies (1-60s) | Varies ($0-$10+) |
| **Total** | **32** | **Comprehensive security intelligence** | **Avg: 🏃 5-8s** | **Avg: $0.03/query** |

**Performance Legend**: ⚡ Fast (<2s) | 🏃 Medium (2-10s) | 🐢 Slow (10-30s) | 🐌 Very Slow (>30s)

---

## Integration Notes

All tools are registered as `FunctionTool` objects in the ADK agent and can be invoked through natural language queries. The agent automatically selects appropriate tools based on user intent.

Tools are organized in the following files:
- `agents/_tools/security_tools.py` - Security-focused BigQuery tools
- `agents/_tools/exploration_tools.py` - Data exploration utilities
- `agents/_tools/bigquery_tools.py` - Standard BigQuery operations
- `agents/_tools/feed_tools.py` - RSS feed analysis
- `agents/_tools/confluence_tools.py` - Confluence integration
- `agents/_tools/service_discovery.py` - GCP service discovery
- `agents/_tools/msa_analyzer.py` - Multi-service analyzer
- `agents/_tools/service_evaluation/evaluator.py` - Service evaluation framework
- `agents/_tools/service_evaluation/compliance_checker.py` - Compliance validation

---

## 📋 Version History

### v1.1 (October 2025)
**Enhanced Documentation Release**

**New Features**:
- 🎯 Added Quick Tool Selector with intent-based navigation
- 📊 Added Quick Reference Matrix with performance and cost metrics
- ⚠️ Added comprehensive Common Pitfalls and Best Practices section
- 🔗 Added 5 Recommended Tool Chains for common workflows
- 🔧 Added extensive Troubleshooting section with 10+ common errors
- 📈 Enhanced all tool descriptions with performance indicators
- 🔍 Added detailed documentation for tool #23 (analyze_gcp_releases)
- 💡 Added response examples for major tools
- 🔄 Added cross-references between related tools

**Improvements**:
- Enhanced Tool Categories Summary table with performance and cost data
- Added "Related Tools" sections to promote tool discovery
- Improved navigation with expanded Table of Contents
- Added performance notes and best practices throughout
- Enhanced troubleshooting guidance with specific solutions

**Documentation Stats**:
- Total sections: 14 (up from 8)
- Total tools documented: 32
- Example code snippets: 50+
- Troubleshooting entries: 10+
- Tool chains: 5

---

### v1.0 (October 2025)
**Initial Release**

**Features**:
- 32 tools across 7 categories
- Comprehensive tool documentation
- Basic usage examples
- Integration notes

**Tool Categories**:
- Security-Focused Tools (3)
- Data Exploration Tools (2)
- RSS Feed Analysis Tools (4)
- Confluence Documentation Tools (5)
- GCP Service Discovery Tools (8)
- Service Evaluation Tools (2)
- Standard BigQuery Tools (7)

**Highlights**:
- Multi-Service Analyzer (MSA) for release impact assessment
- Service evaluation framework with compliance checking
- Dynamic GCP service discovery
- Confluence integration with intelligent caching
- Security threat feed monitoring

---

*Last Updated: October 2025 - Generated from security_agent v1.1*

---

## Additional Resources

### Official Documentation
- [Google ADK Documentation](https://cloud.google.com/adk)
- [BigQuery Documentation](https://cloud.google.com/bigquery/docs)
- [GCP Security Best Practices](https://cloud.google.com/security/best-practices)
- [Confluence API Documentation](https://developer.atlassian.com/cloud/confluence/)

### Internal Resources
- Security Agent GitHub Repository: `ADK/contributing/samples/security_agent/`
- Tool Source Code: `agents/_tools/`
- Configuration Files: `config/`
- Example Scripts: `scripts/`

### Support
- **Issues**: Report bugs or request features via GitHub Issues
- **Questions**: Contact the security team or file a support ticket
- **Contributions**: Submit pull requests for tool enhancements

### Quick Links
- [ADK Quick Start Guide](../README.md)
- [Security Agent Setup](../SETUP.md)
- [Confluence Tool Documentation](../docs/CONFLUENCE.md)
- [Service Evaluation Framework](../docs/SERVICE_EVALUATION.md)

---

**Need Help?**

If you encounter issues not covered in this documentation:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Review the [Common Pitfalls](#-common-pitfalls-and-best-practices)
3. Consult the [Tool Chains](#-recommended-tool-chains) for workflow examples
4. Contact the security team for assistance
