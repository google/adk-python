# Security Agent Tools Documentation

Complete reference for all 32 tools registered in the ADK Security Agent.

## Table of Contents

1. [Security-Focused Tools](#security-focused-tools) (3 tools)
2. [Data Exploration Tools](#data-exploration-tools) (2 tools)
3. [RSS Feed Analysis Tools](#rss-feed-analysis-tools) (4 tools)
4. [Confluence Documentation Tools](#confluence-documentation-tools) (5 tools)
5. [GCP Service Discovery Tools](#gcp-service-discovery-tools) (8 tools)
6. [Service Evaluation Tools](#service-evaluation-tools) (2 tools)
7. [Standard BigQuery Tools](#standard-bigquery-tools) (7 tools)
8. [Usage Examples](#usage-examples)

---

## Security-Focused Tools

### 1. get_security_insights_summary

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

**Purpose**: Multi-Service Analyzer for release notes impact assessment.

**Source**: `agents/_tools/msa_analyzer.py`

**Parameters**:
- `days_back` (int, optional): Number of days to analyze (default: 7)

**Returns**: Comprehensive impact analysis report with security, billing, and compliance impacts

**Description**: Analyzes GCP release notes for impacts on security, billing, and compliance across all active services. Provides risk scoring and actionable recommendations.

**Example Queries**:
- "Analyze GCP releases from the last 7 days"
- "What's the security impact of recent GCP changes?"
- "Check for billing changes in recent releases"

---

## Service Evaluation Tools

### 24. evaluate_new_service

**Purpose**: Evaluate a new GCP service for security, compliance, and risk.

**Source**: `agents/_tools/service_evaluation/evaluator.py`

**Parameters**:
- `service_name` (str, required): Name of the GCP service
- `service_type` (str, required): Service type (storage, compute, database, etc.)
- `service_profile` (str, optional): JSON string with service profile details
- `use_case` (str, optional): Use case description
- `data_classification` (str, optional): Data classification level
- `check_current_compliance` (bool, optional): Check current compliance against BigQuery (default: False)
- `return_format` (str, optional): Output format - 'object', 'dict', or 'summary' (default: 'object')

**Returns**: ServiceEvaluationResult with complete security assessment

**Description**: Performs comprehensive evaluation including risk assessment, security controls mapping, enforcement options, approval workflow determination, and compliance checking.

**Example Queries**:
- "Evaluate Cloud Storage for confidential data usage"
- "Assess security requirements for a new BigQuery project"
- "What controls are needed for Cloud SQL with HIPAA data?"

---

### 25. check_service_compliance

**Purpose**: Validate security controls against actual BigQuery environment state.

**Source**: `agents/_tools/service_evaluation/compliance_checker.py`

**Parameters**:
- `service_type` (str, required): Service type (storage, compute, bigquery, etc.)
- `detailed` (bool, optional): Include detailed violation information (default: False)

**Returns**: JSON string with compliance report

**Description**: The "glue layer" that bridges policy (what SHOULD be) with reality (what IS) by executing validation queries against BigQuery to find compliance gaps.

**Example Queries**:
- "Check compliance for storage services"
- "What security violations exist for compute resources?"
- "Validate BigQuery security controls"

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

| Category | Tool Count | Primary Use Case |
|----------|-----------|------------------|
| Security-Focused | 3 | Direct security findings analysis |
| Data Exploration | 2 | Dataset and table discovery |
| RSS Feed Analysis | 4 | Release notes and threat monitoring |
| Confluence Documentation | 5 | Policy and procedure documentation |
| GCP Service Discovery | 8 | Service catalog and dynamic analysis |
| Service Evaluation | 2 | Security assessment and compliance |
| Standard BigQuery | 7 | General database operations |
| **Total** | **32** | **Comprehensive security intelligence** |

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

*Last Updated: Generated from security_agent v1.0*
