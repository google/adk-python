# BigQueryToolset

The `BigQueryToolset` provides a collection of tools for an agent to interact
with Google BigQuery. It allows agents to inspect metadata, execute SQL queries,
and perform advanced data analysis like forecasting or anomaly detection.

## Introduction

The `BigQueryToolset` simplifies the process of exposing BigQuery capabilities
to an agent by aggregating multiple specialized tools into a single unit. It
manages the instantiation of tools for dataset exploration, table management,
and query execution, ensuring they share consistent authentication and
configuration settings.

Developers use this toolset to build agents that can answer questions about
structured data, generate insights from large datasets, or automate routine
database administrative tasks. The toolset depends on
`BigQueryCredentialsConfig` for managing authentication and
`BigQueryToolConfig` for defining runtime behavior, such as query limits and
write permissions.

## Get started

The following example demonstrates how to initialize the `BigQueryToolset` with
default credentials and provide it to an agent.

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.integrations.bigquery.bigquery_credentials import BigQueryCredentialsConfig
from google.adk.integrations.bigquery.bigquery_toolset import BigQueryToolset

# Initialize the toolset with default credentials and settings.
credentials_config = BigQueryCredentialsConfig()
bigquery_toolset = BigQueryToolset(credentials_config=credentials_config)

# The toolset is passed directly to the agent tools list.
agent = LlmAgent(
    name="bigquery_explorer",
    instruction="Help the user explore their BigQuery datasets.",
    tools=[bigquery_toolset],
)
```

## How it works

The `BigQueryToolset` aggregates several specialized BigQuery tools into a
single object that an agent can consume. When the agent calls `get_tools`, the
toolset instantiates `GoogleTool` objects for functions that handle metadata
retrieval, SQL execution, and data insights.

The toolset uses the `BigQueryCredentialsConfig` to authorize these requests and
the `BigQueryToolConfig` to govern how the `execute_sql` tool behaves. If a
`tool_filter` is provided, the toolset compares the name of each discovered tool
against the filter to determine if it should be included in the final list
returned to the agent.

## Configuration options

The toolset introduces the following configuration options in its constructor.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `tool_filter` | `Optional[Union[ToolPredicate, List[str]]]` | `None` | Filters which tools from the set are available to the agent. |
| `credentials_config` | `Optional[BigQueryCredentialsConfig]` | `None` | The authentication configuration for BigQuery API calls. |
| `bigquery_tool_config` | `Optional[BigQueryToolConfig]` | `None` | Settings for query execution, such as write modes and row limits. |

The `tool_filter` accepts either a list of specific tool names to include or a
`ToolPredicate` callable for dynamic filtering based on the execution context.
Providing a filter is useful when you want to restrict an agent to read-only
metadata tools.

The `credentials_config` manages how the toolset authenticates with Google Cloud
Platform. If it is not provided, the tools attempt to use environment-specific
defaults.

The `bigquery_tool_config` controls the operational limits of the tools. For
example, it defines the maximum number of rows a query can return and whether
the agent is allowed to perform write operations. If this is omitted, the
toolset uses a default `BigQueryToolConfig` instance.

## Advanced applications

You can restrict the agent to a specific subset of BigQuery capabilities by
providing a list of tool names to the `tool_filter` parameter. This is helpful
when an agent only needs to perform metadata lookups without the ability to
execute arbitrary SQL.

```python
from google.adk.integrations.bigquery.bigquery_toolset import BigQueryToolset

# Only expose metadata tools to the agent.
metadata_only_filter = [
    "list_dataset_ids",
    "get_dataset_info",
    "list_table_ids",
    "get_table_info",
]

toolset = BigQueryToolset(
    tool_filter=metadata_only_filter
)
```

## Limitations

The toolset is limited to the specific operations defined in its internal tool
modules, such as metadata inspection and SQL execution. It does not support
every BigQuery API feature, such as managing IAM policies or creating
reservation slots.

## Related samples

- [bigquery_agent](../../../../../contributing/samples/a2a/a2a_auth/remote_a2a/bigquery_agent/agent.py) - An agent that manages user data on BigQuery using OAuth2.
- [bigquery](../../../../../contributing/samples/integrations/bigquery/agent.py) - A data science agent that answers questions and executes SQL queries.
- [google_api](../../../../../contributing/samples/integrations/google_api/agent.py) - A sample demonstrating general Google API tool integration.
