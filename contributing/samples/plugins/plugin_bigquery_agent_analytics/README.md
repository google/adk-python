# BigQuery Agent Analytics Plugin

`BigQueryAgentAnalyticsPlugin` writes agent events to a BigQuery table through
the Write API, so runs can be analysed with SQL. It is registered on the
`App`, so it records every agent, model and tool call without touching agent
code.

**Constructor arguments:**

- `project_id`, `dataset_id`: where the events go. The dataset must already
  exist; the plugin creates the table and, by default, a `v_<event_type>` view
  per event type.
- `table_id`: table name, `agent_events` by default.
- `config`: a `BigQueryLoggerConfig` for batching, retries, event allow and
  deny lists, content truncation and schema upgrades.
- `location`: BigQuery location, `US` by default.
- `credentials`: Application Default Credentials when unset.

## Before you run it

1. Install the extra: `pip install "google-adk[bigquery-analytics]"`.
1. Create the dataset, for example
   `bq --location=US mk --dataset PROJECT:agent_analytics`.
1. Grant the account running the agent BigQuery Data Editor
   (`roles/bigquery.dataEditor`) on the dataset and BigQuery Job User
   (`roles/bigquery.jobUser`) on the project.

## Sample

The agent looks up delivery status for two orders, so the run produces model
calls and tool calls worth reading back.

```bash
export GOOGLE_CLOUD_PROJECT=your-project
export BIGQUERY_ANALYTICS_DATASET=agent_analytics
python contributing/samples/plugins/plugin_bigquery_agent_analytics/main.py
```

Output:

```
user: What is the status of order A1?
agent: Order A1 is currently in transit.

user: And order B2?
agent: Order B2 is currently in transit.

Events written to your-project.agent_analytics.agent_events
```

Reading the events back:

```sql
SELECT event_type, COUNT(*) AS events
FROM `your-project.agent_analytics.agent_events`
GROUP BY event_type ORDER BY events DESC
```

```
LLM_REQUEST                        4
LLM_RESPONSE                       4
USER_MESSAGE_RECEIVED              2
INVOCATION_STARTING                2
AGENT_STARTING                     2
WORKFLOW_NODE_FIRST_OBSERVED       2
TOOL_STARTING                      2
TOOL_COMPLETED                     2
AGENT_RESPONSE                     2
AGENT_COMPLETED                    2
WORKFLOW_NODE_OBSERVATION_SUMMARY  2
INVOCATION_COMPLETED               2
```

The table is partitioned by day on `timestamp`, so cost stays predictable as
runs accumulate.

## Writing and shutting down

With the default `BigQueryLoggerConfig` (`flush_on_run_end=True`), the writer
runs on the caller's event loop and each run waits for its rows to be written
before it finishes. `await plugin.close()` releases the writer; the sample gets
that by closing the runner with `async with InMemoryRunner(...)`.

The plugin fails open: if it cannot write, for example because the dataset or a
permission is missing, it logs an error and lets the agent keep running. The
sample checks `plugin.get_drop_stats()` after closing and prints the non-zero
counts instead of the success line when any row was dropped or degraded.
