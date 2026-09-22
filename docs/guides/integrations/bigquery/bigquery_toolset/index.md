# BigQueryToolset

`BigQueryToolset` gives an agent eleven tools for exploring and querying
BigQuery: listing datasets and tables, reading metadata, running SQL,
forecasting, and asking natural-language questions about the data. Whether the
agent can write anything is decided by one setting, `write_mode`, which defaults
to refusing every statement that is not a `SELECT`.

To try it you need a Google Cloud project with BigQuery enabled and credentials
that can reach it, which for local work usually means Application Default
Credentials. BigQuery charges for the data a query scans, so put a ceiling on
that with `maximum_bytes_billed` before you let an agent write its own SQL. One
of the eleven tools, `ask_data_insights`, needs the Conversational Analytics
API turned on separately and does not work until you do that.

## Introduction

Most data agents need the same handful of capabilities: find out what tables
exist, look at their schemas, and run a query. Writing those as function tools
means writing a BigQuery client, credential handling, result truncation, and
some way of stopping the model from dropping a table. `BigQueryToolset` is that
work already done.

You construct it with a credentials configuration and, optionally, a tool
configuration, then put it in an agent's `tools` list like any other toolset.
`get_tools` returns the eleven tools, each wrapped as a `GoogleTool` that
resolves credentials at call time.

The one decision you cannot postpone is `write_mode`. It is not a convenience
setting. It is the control that separates an agent that can only read from one
that can modify permanent tables, and it is enforced per statement, inside the
`execute_sql` tool, by dry-running the query against BigQuery before executing
it. Its default is the safe one, so a toolset you have not configured cannot
write anything at all.

## The import paths

Two of the four names you need are exported from the package and two are not:

```python
# Available from the package.
from google.adk.integrations.bigquery import BigQueryCredentialsConfig
from google.adk.integrations.bigquery import BigQueryToolset

# Only available from the submodule.
from google.adk.integrations.bigquery.config import BigQueryToolConfig
from google.adk.integrations.bigquery.config import WriteMode
```

If you import `BigQueryToolConfig` or `WriteMode` from
`google.adk.integrations.bigquery`, you get an `ImportError`, because neither
name is exported from the package. Import them from
`google.adk.integrations.bigquery.config`, as the sample agent does.

## Get started

This example builds a read-only analyst agent. `write_mode` is left at its
default, so the toolset runs `SELECT` statements and refuses everything
else.

```python
from google.adk.agents import LlmAgent
from google.adk.integrations.bigquery import BigQueryCredentialsConfig
from google.adk.integrations.bigquery import BigQueryToolset
from google.adk.integrations.bigquery.config import BigQueryToolConfig
import google.auth

# Application Default Credentials. See
# https://cloud.google.com/docs/authentication/provide-credentials-adc
application_default_credentials, _ = google.auth.default()

bigquery_toolset = BigQueryToolset(
    credentials_config=BigQueryCredentialsConfig(
        credentials=application_default_credentials
    ),
    bigquery_tool_config=BigQueryToolConfig(
        application_name="my_analyst_agent",
        max_query_result_rows=50,
    ),
)

root_agent = LlmAgent(
    name="bigquery_agent",
    description="Answers questions about data in BigQuery.",
    instruction=(
        "You answer questions about BigQuery data. Discover the available"
        " datasets and tables before writing any SQL, and explain what a query"
        " does before running it."
    ),
    tools=[bigquery_toolset],
)
```

## The tools

`get_tools` returns eleven tools, all wrapped as `GoogleTool`. They fall into
four groups. If your agent needs only one or two of those groups, `tool_filter`
is how you drop the rest.

### Discovery and metadata

Reach for these when the agent has to find out what exists before it can write
any SQL. None of them can write, so `write_mode` never comes into it for this
group.

*   `list_dataset_ids` lists the datasets in a project.
*   `get_dataset_info` reads a dataset's metadata.
*   `list_table_ids` lists the tables in a dataset.
*   `get_table_info` reads a table's schema and metadata.
*   `search_catalog` searches the Dataplex catalog for BigQuery entries.

### Query execution

These two run a statement and report on the run. `execute_sql` is the only tool
`write_mode` gates, so it is the one to think about before you widen that
setting.

*   `execute_sql` runs, or dry-runs, a SQL statement.
*   `get_job_info` reads a job's metadata.

### Statistical analysis

Reach for these when the question is statistical rather than a lookup. The last
two build a temporary BigQuery ML model before they query it, so neither works
under the default `write_mode`.

*   `forecast` does time-series forecasting through `AI.FORECAST`.
*   `analyze_contribution` runs contribution analysis: it builds a temporary
    model, then `ML.GET_INSIGHTS`.
*   `detect_anomalies` runs anomaly detection: it builds a temporary model,
    then `ML.DETECT_ANOMALIES`.

### Natural-language answers

One tool answers a question in natural language rather than returning rows, and
it needs setup the other ten do not.

*   `ask_data_insights` sends the question through the Conversational Analytics
    API. If you have not enabled and configured that API in your project, the
    tool fails at call time while the other ten keep working.

## write_mode

`WriteMode` has three values, and they are not three points on a convenience
spectrum. They are three different answers to the question "what may this agent
destroy".

| Value | Permits | Refuses |
| :--- | :--- | :--- |
| `WriteMode.BLOCKED` *(default)* | Statements whose type is `SELECT`. | Everything else, including inserts, updates, and DDL. |
| `WriteMode.PROTECTED` | `SELECT`, plus writes whose destination is the anonymous dataset of a BigQuery session. | Writes to a permanent dataset. |
| `WriteMode.ALLOWED` | Every statement the credentials permit. | Nothing. |

**How the refusal works.** Before executing anything, `execute_sql` submits the
statement to BigQuery as a dry run and inspects the reported `statement_type`.
Under `BLOCKED`, anything other than `SELECT` returns
`{"status": "ERROR", "error_details": "Read-only mode only supports SELECT statements."}`.
That is a value returned to the model, not a raised exception, so the model sees
the refusal and usually tries to rephrase. Your application code never sees
an error.

**What `PROTECTED` actually protects.** On the first statement, the toolset
opens a BigQuery
[session](https://cloud.google.com/bigquery/docs/sessions-intro) and stores its
id and its anonymous dataset id in the tool context's state, so every later
statement in the same conversation runs in that session. A temporary table
created there is real and queryable, and BigQuery discards it when the session
ends. Before executing, the dry run is inspected again: a statement that is not
a `SELECT` and whose reported destination is a dataset *other than* the
session's anonymous dataset is refused. So the guarantee is "writes land in
scratch space, permanent tables are untouched", and it rests on BigQuery
reporting a destination for the statement.

**`ALLOWED` means allowed.** The only remaining limit is what the credentials
can do in IAM. If the service account can drop a production table, so can the
agent. Where you need a write-capable agent, the durable control is a
least-privilege service account, with `write_mode` as the second layer rather
than the only one.

Two tools do not follow the setting. `analyze_contribution` and
`detect_anomalies` both have to create a temporary BigQuery ML model, so under
`BLOCKED` they refuse outright and return an error to the model, and under
`ALLOWED` they downgrade themselves to `PROTECTED` for the statements they issue
so that the model creation and the query share one session. In other words,
those two tools are unusable in a strictly read-only agent, and that is by
design rather than a bug.

## Constructor options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `tool_filter` | `ToolPredicate \| list[str] \| None` | `None` | Which of the eleven tools to expose. All keyword-only. |
| `credentials_config` | `BigQueryCredentialsConfig \| None` | `None` | How the tools authenticate. |
| `bigquery_tool_config` | `BigQueryToolConfig \| None` | `None` | Behavior of the tools. A default `BigQueryToolConfig()` is built when omitted. |

**`tool_filter`** accepts either a list of tool names or a callable taking
`(tool, readonly_context)` and returning a bool. A name in the list that matches
nothing is silently ignored, so a typo removes a tool rather than raising. An
empty list exposes nothing.

**`credentials_config`** is optional. If you omit it, ADK does no credential
management at all: each tool is handed `credentials=None`, and the BigQuery
client falls back to whatever Application Default Credentials the process has.
That is fine for local development and wrong for anything multi-tenant, where
you want each end user's own identity.

## BigQueryToolConfig options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `write_mode` | `WriteMode` | `WriteMode.BLOCKED` | What `execute_sql` may run. See above. |
| `max_query_result_rows` | `int` | `50` | Row cap on query results. |
| `maximum_bytes_billed` | `int \| None` | `None` | Cost ceiling per query, in bytes. |
| `compute_project_id` | `str \| None` | `None` | Restricts query execution to one project. |
| `location` | `str \| None` | `None` | BigQuery location for data and compute. |
| `application_name` | `str \| None` | `None` | Identifies your agent in user agents and job labels. |
| `job_labels` | `dict[str, str] \| None` | `None` | Labels added to every job the tools run. |

If you misspell an option, you get a Pydantic validation error at construction
rather than a silently ignored field, because the model forbids unknown fields.
Three of the fields validate their values and raise rather than coercing:

*   `maximum_bytes_billed` below `10485760` raises. BigQuery's on-demand pricing
    has a 10 MB floor per query and per referenced table, so a smaller ceiling
    could never be met.
*   `application_name` containing a space raises, because the value goes into a
    user-agent string and a BigQuery job label.
*   `job_labels` raises for more than twenty entries, for an empty key, or for
    any key beginning with `adk-bigquery-`, which is reserved. ADK adds
    `adk-bigquery-tool` to every job itself, and `adk-bigquery-application-name`
    when you set `application_name`.

**`max_query_result_rows`** is passed to BigQuery as `max_results`, so it caps
what comes back rather than what the query scans. Raising it raises the number
of rows that end up in the model's context, which is usually what you are paying
for. It also caps the results of `ask_data_insights`.

**`compute_project_id`** is a guardrail worth setting alongside `write_mode`.
The model supplies `project_id` as a tool argument, and when
`compute_project_id` is set, a request naming any other project is refused with
an error rather than executed. It constrains where compute happens, not which
data can be read.

**`maximum_bytes_billed`** applies only to the final execution. The dry runs the
toolset uses to classify a statement do not consume it, and neither do they cost
anything.

## Credentials

`BigQueryCredentialsConfig` extends the shared Google credentials
configuration and supports three mutually exclusive ways of authenticating:

*   `credentials=` takes an existing `google.auth` credentials object, such as
    Application Default Credentials or a loaded service-account key. Every end
    user's requests then run as that one identity.
*   `client_id=` and `client_secret=` together start an interactive OAuth flow,
    so each end user consents and the tools act as them. That flow produces the
    pause-for-consent handshake described in
    [the tool auth guide](../../../auth/tool_auth/index.md).
*   `external_access_token_key=` names the key under which the surrounding
    platform has already placed an access token in tool context state.

Setting `credentials` together with any of the others is rejected. When you do
not set `scopes` explicitly, BigQuery's config fills in two:
`https://www.googleapis.com/auth/bigquery` and
`https://www.googleapis.com/auth/dataplex.read-write`. The second is what
`search_catalog` needs, and it is broader than read-only; narrow `scopes`
yourself if your agent never searches the catalog.

## Advanced applications

Three adjustments come up once a read-only agent works: narrowing the tool
list, varying that list by user, and giving the agent scratch space it can
write to.

### Cut the toolset down

*   **Problem solved**: eleven tool declarations is a lot of context for an agent
    that only needs to run one query, and every tool you expose is a tool the
    model can misuse.
*   **Implementation**: pass the names you want.

```python
toolset = BigQueryToolset(
    credentials_config=credentials_config,
    tool_filter=["list_table_ids", "get_table_info", "execute_sql"],
)
```

### Filter by context

*   **Problem solved**: the same agent should offer fewer tools to some users
    than to others.
*   **Implementation**: pass a predicate instead of a list. It receives the tool
    and the read-only context, and runs on every `get_tools` call, so it sees
    current session state.

```python
def only_metadata_for_guests(tool, readonly_context=None) -> bool:
  if readonly_context and readonly_context.state.get("role") == "analyst":
    return True
  return tool.name != "execute_sql"


toolset = BigQueryToolset(
    credentials_config=credentials_config,
    tool_filter=only_metadata_for_guests,
)
```

### Let the agent build scratch tables

*   **Problem solved**: a multi-step analysis needs somewhere to put intermediate
    results, but must not touch permanent tables.
*   **Implementation**: `WriteMode.PROTECTED`. The session and its anonymous
    dataset are created on the first statement and reused for the conversation.

```python
tool_config = BigQueryToolConfig(
    write_mode=WriteMode.PROTECTED,
    compute_project_id="my-project",
    maximum_bytes_billed=10 * 1024 * 1024 * 1024,
)
```

## Limitations

*   **Refusals are returned, not raised.** A blocked statement produces an error
    dictionary that goes back to the model. Nothing propagates to your code, and
    nothing appears in your logs unless you add a callback that inspects tool
    results.
*   **`write_mode` gates one tool.** It is enforced inside `execute_sql`, which
    is also the path `forecast`, `analyze_contribution`, and `detect_anomalies`
    run through. The metadata tools and `search_catalog` do not consult it,
    because they cannot write.
*   **`analyze_contribution` and `detect_anomalies` do not work under
    `BLOCKED`.** Both need to create a temporary model, so a strictly read-only
    agent effectively has nine tools.
*   **Every `execute_sql` call costs at least one extra BigQuery job.** The dry
    run used to classify the statement is a separate request, and `PROTECTED`
    adds another to open the session. Dry runs are not billed, but they do add
    latency.
*   **`application_name` and `job_labels` are for tracking only.** They travel
    with the BigQuery job so you can attribute usage and cost. The source says
    explicitly that neither should be used for security-sensitive decisions, so
    do not build an access rule that reads them back.
*   **The credentials configuration is provisional.** Its base class documents
    itself as not for production use and possibly deprecated later.
*   **`ask_data_insights` needs extra setup.** Without the Conversational
    Analytics API configured, that tool fails at call time while the other ten
    work.

## Related samples

*   [BigQuery tools](../../../../../contributing/samples/integrations/bigquery/agent.py)
    runs one toolset against every credential path in turn: Application Default
    Credentials, OAuth, a service-account key and an external token, with
    `write_mode=WriteMode.ALLOWED`.
*   [BigQuery over MCP](../../../../../contributing/samples/integrations/bigquery_mcp/agent.py)
    reaches BigQuery through an MCP server instead of this toolset.

## Related guides

*   [AuthConfig and authenticated tools](../../../auth/tool_auth/index.md)
    explains what happens when `client_id` and `client_secret` send a user
    through OAuth.
