# Security Agent Tools Directory

This directory contains all 32 tools registered with the ADK Security Agent. Each tool is a Python function that performs a specific security analysis, data retrieval, or monitoring task.

## Overview

**Total Tools**: 32 specialized security analysis tools

**Organization**: Tools are grouped by functionality into separate Python modules

**Registration**: All tools are registered in `agents/agent.py` as `FunctionTool` objects

**Documentation**: Complete tool reference available in `docs/TOOLS.md`

## Tool Categories

### 1. Security-Focused Tools (3 tools)
**File**: `security_tools.py`

Direct analysis of security findings from the primary security dataset.

```python
get_security_insights_summary()      # Overview of security findings
query_security_insights(filter)      # Custom filtered queries
get_security_statistics(group_by)    # Aggregated statistics
```

**Use Cases**:
- Quick security overview
- Filtered security queries
- Distribution analysis by severity, category, resource type

### 2. Standard BigQuery Tools (7 tools)
**File**: `bigquery_tools.py`

General-purpose BigQuery operations for any dataset or table.

```python
hello_world()                        # Connection test
list_datasets()                      # Show all datasets
list_tables(dataset_id)              # Tables in dataset
get_table_schema(dataset_id, table_id)  # Table structure
run_query(sql)                       # Execute any SQL
analyze_query_cost(sql)              # Cost estimation
get_table_sample(dataset_id, table_id)  # Preview data
```

**Use Cases**:
- Dataset exploration
- Schema inspection
- Custom SQL queries
- Cost analysis

### 3. Data Exploration Tools (2 tools)
**File**: `exploration_tools.py`

Enhanced dataset and table discovery with detailed metadata.

```python
explore_all_tables_and_views(dataset_id)     # List all objects
analyze_table_or_view(dataset_id, object_id) # Deep dive analysis
```

**Use Cases**:
- Understanding dataset structure
- Discovering available data
- Analyzing table contents

### 4. RSS Feed Analysis Tools (4 tools)
**File**: `feed_tools.py`

Monitor GCP release notes and security threat feeds.

```python
query_gcp_release_notes(days_back, security_only)  # GCP updates
query_security_threat_feeds(severity, cvss_score)  # CVEs and threats
get_feed_statistics()                              # Feed metrics
search_feeds_by_keyword(keyword)                   # Keyword search
```

**Use Cases**:
- Track GCP platform changes
- Monitor security vulnerabilities
- CVE analysis
- Threat intelligence

### 5. Confluence Documentation Tools (5 tools)
**File**: `confluence_tools.py`

Search and retrieve documentation from Atlassian Confluence.

```python
search_confluence_documentation(query, spaces)  # Search docs
get_confluence_document(document_id)           # Retrieve page
analyze_confluence_coverage(topics)            # Gap analysis
get_confluence_statistics()                    # Cache stats
refresh_confluence_cache(spaces, force)        # Update cache
```

**Use Cases**:
- Find security policies
- Retrieve compliance documentation
- Analyze documentation coverage
- Policy and procedure lookup

### 6. GCP Service Discovery Tools (8 tools)
**File**: `service_discovery.py`

Discover, analyze, and learn about GCP services dynamically.

```python
discover_gcp_services()                  # List enabled services
analyze_gcp_service(name, analysis_type) # On-demand analysis
get_service_resources(service_name)      # Enumerate resources
suggest_service_analysis(user_query)     # AI-powered suggestions

# Learning new services
learn_service_from_url(documentation_url)  # Parse docs
discover_new_gcp_services(release_url)     # Find new releases
register_new_service(name, endpoint, url)  # Manual registration
learn_from_api_spec(api_spec_url)         # Parse API specs
```

**Use Cases**:
- Service catalog management
- Security analysis for any GCP service
- Learning about new services
- Dynamic service onboarding

### 7. Service Evaluation Tools (2 tools)
**Directory**: `service_evaluation/`

**Files**: `evaluator.py`, `compliance_checker.py`

Comprehensive security assessment and compliance validation framework.

```python
evaluate_new_service(name, type, profile)     # Security assessment
check_service_compliance(service_type)        # Compliance validation
```

**Use Cases**:
- New service security evaluation
- Compliance checking (PCI, HIPAA, SOC2)
- Risk assessment
- Control mapping

### 8. Multi-Service Analyzer (1 tool)
**File**: `msa_analyzer.py`

Analyze GCP release notes for security, billing, and compliance impacts.

```python
analyze_gcp_releases(days_back)  # Release notes impact analysis
```

**Use Cases**:
- Monitor GCP platform changes
- Assess security impacts
- Track billing changes
- Compliance impact analysis

## File Organization

```
agents/_tools/
├── __init__.py                  # Tool exports for agent registration
├── base.py                      # Shared configuration and utilities
├── security_tools.py            # Security-focused tools (3)
├── bigquery_tools.py            # Standard BigQuery tools (7)
├── exploration_tools.py         # Dataset exploration (2)
├── feed_tools.py                # RSS feed analysis (4)
├── confluence_tools.py          # Confluence integration (5)
├── service_discovery.py         # GCP service discovery (8)
├── msa_analyzer.py              # Multi-Service Analyzer (1)
├── service_evaluation/          # Service evaluation framework (2)
│   ├── __init__.py
│   ├── evaluator.py            # Service security assessment
│   └── compliance_checker.py   # Compliance validation
├── iam_custom_role_analyzer.py  # IAM custom role analysis (utility)
├── service_documentation_parser.py  # Documentation parsing (utility)
└── service_onboarding.py        # Service onboarding (utility)
```

## How Tools Are Registered

Tools are registered in `agents/agent.py`:

```python
from google.adk.tools import FunctionTool

# Import tools
from ._tools import (
    get_security_insights_summary,
    query_security_insights,
    # ... all other tools
)

# Wrap as FunctionTool objects
tools = [
    FunctionTool(get_security_insights_summary),
    FunctionTool(query_security_insights),
    FunctionTool(get_security_statistics),
    # ... all 32 tools
]

# Create agent with tools
root_agent = LlmAgent(
    name="security_bigquery_agent",
    model="gemini-2.5-flash",
    instruction=instruction,
    tools=tools
)
```

## How to Add New Tools

### Step 1: Create Tool Function

Choose the appropriate file based on tool category, or create a new file if starting a new category.

**Example**: Adding a network security analysis tool to `security_tools.py`

```python
# File: agents/_tools/security_tools.py

from typing import Optional
from .base import get_bq_client, PROJECT_ID, DEFAULT_DATASET

def analyze_network_security(
    min_risk_score: int = 50,
    include_vpcs: bool = True,
    region: Optional[str] = None
) -> str:
    """
    Analyze network security configurations including firewalls and VPCs.

    This tool examines firewall rules, VPC settings, and network policies
    to identify potential security risks in network configurations.

    Args:
        min_risk_score: Minimum risk score to include (0-100). Default: 50
        include_vpcs: Include VPC configuration analysis. Default: True
        region: Filter by GCP region (optional)

    Returns:
        Formatted analysis with findings and recommendations
    """
    client = get_bq_client()

    # Build query
    query = f"""
        SELECT
            firewall_rule_name,
            direction,
            source_ranges,
            allowed_ports,
            risk_score,
            risk_factors,
            vpc_network
        FROM `{PROJECT_ID}.{DEFAULT_DATASET}.firewall_rules`
        WHERE risk_score >= {min_risk_score}
    """

    if region:
        query += f" AND region = '{region}'"

    query += " ORDER BY risk_score DESC LIMIT 100"

    # Execute query
    results = client.query(query).result()

    # Format output
    output = [f"Network Security Analysis (Risk >= {min_risk_score})\n"]
    output.append("=" * 70 + "\n")

    for row in results:
        output.append(f"\n🔥 {row.firewall_rule_name}")
        output.append(f"   VPC: {row.vpc_network}")
        output.append(f"   Risk Score: {row.risk_score}/100")
        output.append(f"   Direction: {row.direction}")

        if row.risk_factors:
            output.append(f"   ⚠️  Risks: {', '.join(row.risk_factors)}")

    return "\n".join(output)
```

### Step 2: Add Type Annotations (Required)

ADK requires proper type annotations for all parameters:

```python
# ✅ CORRECT - All parameters have types
def my_tool(
    param1: str,                    # Required string
    param2: int = 10,               # Optional int with default
    param3: Optional[bool] = None   # Explicitly optional
) -> str:                           # Return type
    pass

# ❌ WRONG - Missing type annotations
def my_tool(param1, param2=10):
    pass

# ❌ WRONG - Missing return type
def my_tool(param1: str) -> None:  # Should return str, not None
    pass
```

### Step 3: Write Comprehensive Docstring

The docstring becomes the tool description that the agent uses to understand when to call your tool:

```python
def my_tool(param: str) -> str:
    """
    One-line summary of what the tool does.

    Detailed description explaining the purpose, use cases, and behavior
    of the tool. This helps the agent understand when to call this tool
    versus other similar tools.

    Args:
        param: Description of parameter including valid values and defaults

    Returns:
        Description of what the tool returns and format
    """
    pass
```

### Step 4: Register in agent.py

Add your tool to `agents/agent.py`:

```python
# Import your new tool
from ._tools.security_tools import (
    get_security_insights_summary,
    query_security_insights,
    get_security_statistics,
    analyze_network_security,  # NEW TOOL
)

# Add to tools list
tools = [
    # ... existing tools ...
    FunctionTool(analyze_network_security),  # NEW TOOL
]
```

### Step 5: Export from __init__.py

Add your tool to `agents/_tools/__init__.py`:

```python
from .security_tools import (
    get_security_insights_summary,
    query_security_insights,
    get_security_statistics,
    analyze_network_security,  # NEW TOOL
)

__all__ = [
    # ... existing tools ...
    "analyze_network_security",  # NEW TOOL
]
```

### Step 6: Test Your Tool

```bash
# Restart ADK backend
# Terminal 1: Ctrl+C, then
adk web

# Test via Python
python3 -c "
from agents._tools.security_tools import analyze_network_security
result = analyze_network_security(min_risk_score=75)
print(result)
"

# Test via web UI
# Navigate to http://localhost:5001
# Query: "Analyze network security with risk score above 75"
```

### Step 7: Document in TOOLS.md

Add comprehensive documentation to `docs/TOOLS.md`:

```markdown
### XX. analyze_network_security

**Purpose**: Analyze network security configurations including firewalls and VPCs.

**Source**: `agents/_tools/security_tools.py`

**Parameters**:
- `min_risk_score` (int, optional): Minimum risk score 0-100 (default: 50)
- `include_vpcs` (bool, optional): Include VPC analysis (default: True)
- `region` (str, optional): Filter by GCP region

**Returns**: Formatted string with network security findings

**Description**: Examines firewall rules, VPC settings, and network policies...

**Example Queries**:
- "Analyze network security"
- "Show me high-risk firewall rules"
- "Check network security for us-central1"
```

## Type Annotation Requirements

### Basic Types

```python
from typing import Optional, List, Dict, Any

def example_tool(
    string_param: str,              # Required string
    int_param: int,                 # Required integer
    bool_param: bool,               # Required boolean
    float_param: float,             # Required float
    optional_str: Optional[str] = None,      # Optional string
    list_param: List[str] = [],              # List of strings
    dict_param: Dict[str, Any] = {},         # Dictionary
    default_int: int = 10                    # With default value
) -> str:                           # Return type
    """Tool description."""
    pass
```

### Complex Types

```python
from typing import Optional, List, Dict, Union, Literal

def advanced_tool(
    # Union types (multiple valid types)
    union_param: Union[str, int],

    # Literal types (specific values only)
    severity: Literal["low", "medium", "high", "critical"],

    # Nested structures
    nested_dict: Dict[str, List[int]],

    # Optional complex types
    optional_list: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Advanced tool with complex types."""
    pass
```

### StructuredToolResponse

For tools that return structured data:

```python
from .base import StructuredToolResponse

def structured_tool() -> StructuredToolResponse:
    """Tool that returns structured data."""
    return StructuredToolResponse(
        summary="Human-readable summary",
        data={
            "key": "value",
            "metrics": {"count": 10}
        },
        metadata={"query": "SELECT ..."}
    )
```

## Common Patterns

### Pattern 1: BigQuery Query Tool

```python
from .base import get_bq_client, PROJECT_ID, DEFAULT_DATASET

def query_data(filter_value: str, limit: int = 100) -> str:
    """Query data with filter."""
    client = get_bq_client()

    query = f"""
        SELECT * FROM `{PROJECT_ID}.{DEFAULT_DATASET}.table_name`
        WHERE column = '{filter_value}'
        LIMIT {limit}
    """

    results = client.query(query).result()

    output = []
    for row in results:
        output.append(f"{row.column}: {row.value}")

    return "\n".join(output)
```

### Pattern 2: Data Analysis Tool

```python
def analyze_data(
    metric: str = "severity",
    include_details: bool = False
) -> str:
    """Analyze data by metric."""
    # Fetch data
    data = fetch_from_bigquery()

    # Perform analysis
    analysis = {}
    for item in data:
        key = item.get(metric)
        analysis[key] = analysis.get(key, 0) + 1

    # Format output
    output = [f"Analysis by {metric}:\n"]
    for key, count in sorted(analysis.items()):
        output.append(f"{key}: {count}")

    return "\n".join(output)
```

### Pattern 3: External API Tool

```python
import requests
from typing import Optional

def fetch_external_data(
    service_name: str,
    api_key: Optional[str] = None
) -> str:
    """Fetch data from external API."""
    # Build request
    url = f"https://api.example.com/{service_name}"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    # Make request with error handling
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return f"Error fetching data: {str(e)}"

    # Process and format response
    return format_response(data)
```

## Best Practices

### 1. Single Responsibility
Each tool should do one thing well:
```python
# ✅ GOOD - Single purpose
def get_firewall_rules() -> str:
    """Get all firewall rules."""
    pass

def analyze_firewall_risks() -> str:
    """Analyze firewall rule risks."""
    pass

# ❌ BAD - Multiple responsibilities
def get_and_analyze_firewalls() -> str:
    """Get firewall rules and analyze them."""
    pass
```

### 2. Clear Naming
Use descriptive, action-oriented names:
```python
# ✅ GOOD
get_security_findings()
analyze_iam_permissions()
query_release_notes()

# ❌ BAD
security()
iam()
notes()
```

### 3. Comprehensive Docstrings
Include purpose, parameters, returns, and examples:
```python
def example_tool(param: str) -> str:
    """
    One-line summary.

    Detailed description with context and use cases.
    Explain when to use this tool vs alternatives.

    Args:
        param: What this parameter controls and valid values

    Returns:
        What the tool returns and in what format

    Examples:
        >>> example_tool("value")
        'Result...'
    """
    pass
```

### 4. Error Handling
Handle errors gracefully and return useful messages:
```python
def safe_tool(param: str) -> str:
    """Tool with error handling."""
    try:
        result = perform_operation(param)
        return result
    except ValueError as e:
        return f"Invalid parameter: {str(e)}"
    except Exception as e:
        return f"Error performing operation: {str(e)}"
```

### 5. Configurable Defaults
Use sensible defaults that can be overridden:
```python
def flexible_tool(
    required_param: str,
    limit: int = 100,           # Reasonable default
    include_details: bool = True # Sensible default
) -> str:
    """Tool with good defaults."""
    pass
```

## Testing Tools

### Unit Tests

```python
# File: tests/test_my_tool.py

import pytest
from agents._tools.security_tools import analyze_network_security

def test_analyze_network_security():
    """Test basic functionality."""
    result = analyze_network_security(min_risk_score=50)
    assert result is not None
    assert "Network Security Analysis" in result

def test_analyze_network_security_high_threshold():
    """Test with high risk threshold."""
    result = analyze_network_security(min_risk_score=90)
    assert result is not None

def test_analyze_network_security_region_filter():
    """Test region filtering."""
    result = analyze_network_security(region="us-central1")
    assert "us-central1" in result or "No results" in result
```

### Integration Tests

```python
def test_tool_integration_with_agent():
    """Test tool works with agent."""
    from agents.agent import root_agent

    # Verify tool is registered
    tool_names = [t.function.__name__ for t in root_agent.tools]
    assert "analyze_network_security" in tool_names

    # Test tool execution
    result = analyze_network_security(min_risk_score=75)
    assert result is not None
```

## Debugging Tools

### Enable Logging

```python
import logging

# At top of tool file
logger = logging.getLogger(__name__)

def my_tool(param: str) -> str:
    """Tool with logging."""
    logger.info(f"Tool called with param: {param}")

    try:
        result = perform_operation(param)
        logger.info("Operation succeeded")
        return result
    except Exception as e:
        logger.error(f"Error in tool: {str(e)}")
        raise
```

### Test Tool Directly

```python
# Quick test without agent
python3 -c "
from agents._tools.security_tools import my_tool
result = my_tool('test')
print(result)
"
```

### Check Tool Registration

```python
# Verify tool is registered
python3 -c "
from agents.agent import root_agent
tools = [t.function.__name__ for t in root_agent.tools]
print('Total tools:', len(tools))
print('Tools:', tools)
"
```

## Additional Resources

- **Detailed Tool Reference**: See `docs/TOOLS.md` for complete documentation of all 32 tools
- **Agent Instructions**: See `docs/agent_instructions.md` for agent behavior and tool selection
- **Developer Guide**: See `INSTRUCTIONS.md` for development setup and workflows
- **Base Utilities**: See `base.py` for shared configuration and helper functions

---

**Questions or Issues?**

Check the troubleshooting section in `INSTRUCTIONS.md` or review existing tools as examples.
