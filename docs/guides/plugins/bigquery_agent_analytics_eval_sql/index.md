# BigQuery Agent Analytics eval queries

`build_eval_session_query_sql` and `build_eval_turn_query_sql` build the SQL
that turns the events `BigQueryAgentAnalyticsPlugin` logs into one evaluation
row per session or per turn. An evaluation job runs the SQL itself, so it can
keep BigQuery from scanning the whole events table, which a query on the
plugin's eval views cannot do.

## Introduction

`BigQueryAgentAnalyticsPlugin` writes one row per agent event to a BigQuery
table. Agent evaluation needs a different shape: one row per session or per
turn that holds the whole exchange as a trace, next to flags that say whether
the row is fit to judge. With `create_eval_views=True` in its
`BigQueryLoggerConfig`, the plugin creates two views of that shape,
`{view_prefix}_eval_session` and `{view_prefix}_eval_turn`. Every column of
those views is aggregated over the events of a session or a turn, so BigQuery
cannot apply a filter on a view column to the table's `timestamp` partitions,
and every query of a view scans the whole table.

The query builders return the SQL of those views with two conditions added.
`raw_where_clause` narrows the events that the query reads, so that BigQuery
skips the daily `timestamp` partitions that the condition rules out.
`where_clause` picks the sessions or turns to return after they are rebuilt. A
scheduled evaluation job uses both: a lookback that bounds the scan, and a time
window that selects the rows that the job has not judged yet.

The builders need only the Python standard library. You can import them from
`google.adk.plugins` without the `bigquery-analytics` extra, which installs
`pyarrow` and `google-cloud-bigquery-storage` for the plugin's writer. Running
the SQL needs a BigQuery client, such as the `google-cloud-bigquery` package.

## Get started

The following job reads the sessions that ended on the previous UTC day and are
ready for evaluation. It also reads the week of events before that day, so that
a session that started before the window still comes out whole.

```python
import datetime

from google.adk.plugins import build_eval_session_query_sql
from google.cloud import bigquery

sql = build_eval_session_query_sql(
    "my-project.agent_analytics.agent_events",
    raw_where_clause="timestamp >= @lookback_start",
    where_clause=(
        "session_end_timestamp >= @window_start"
        " AND session_end_timestamp < @window_end"
        " AND eval_ready"
    ),
)

window_end = datetime.datetime.now(datetime.timezone.utc).replace(
    hour=0, minute=0, second=0, microsecond=0
)
window_start = window_end - datetime.timedelta(days=1)
lookback_start = window_start - datetime.timedelta(days=7)

job_config = bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ScalarQueryParameter(
            "lookback_start", "TIMESTAMP", lookback_start
        ),
        bigquery.ScalarQueryParameter(
            "window_start", "TIMESTAMP", window_start
        ),
        bigquery.ScalarQueryParameter("window_end", "TIMESTAMP", window_end),
    ]
)
for row in bigquery.Client().query(sql, job_config=job_config).result():
  print(row.session_id, row.turn_count, row.eval_ready)
```

Each row is one session: its `trace`, the counts and flags beside it, and
`eval_ready`. Schedule the job some time after midnight UTC, so that the last
turns of the day are finished and written when it runs. The section on choosing
between the two conditions explains why each bound sits where it does.

## How it works

Each builder returns one `WITH ... SELECT` query that rebuilds sessions or turns
from the events table and returns the rows of the matching eval view:

| Builder | One row per | Same rows as |
| :--- | :--- | :--- |
| `build_eval_session_query_sql` | Session, with every turn in `trace`. | The `{view_prefix}_eval_session` view. |
| `build_eval_turn_query_sql` | Turn, keyed by `session_id` and `invocation_id`, with only that turn in `trace`. | The `{view_prefix}_eval_turn` view. |

The query reads the rows of the events table that pass `raw_where_clause` and
rebuilds each session or turn from them: its events in order, the configuration
of its agents, and its counts and flags. It then filters the rebuilt rows with
`where_clause`. The order matters to a caller. `where_clause` sees whole rows,
so it cannot remove an event from a row that it keeps. `raw_where_clause` sees
single events, so it decides which events a row is built from, and a condition
that drops an event of a session changes that session's row.

The query declares no query parameters of its own: your job binds every `@name`
parameter that the two conditions use. The query also ends without a
semicolon, so the job can append `ORDER BY` or `LIMIT` to it or read it as a
subquery.

### Output columns

A session row has the following columns, grouped by what they describe:

| Columns | What they hold |
| :--- | :--- |
| `session_id`, `user_id`, `root_agent` | The session, its user, and its root agent. |
| `session_start_timestamp`, `session_end_timestamp`, `turn_count` | When the session ran, from its first event to its last, and how many turns it has. |
| `tool_call_count`, `tool_error_count`, `llm_error_count`, `agent_error_count`, `has_error` | The tool calls and the errors that the session logged. |
| `has_tool_loop`, `max_consecutive_identical_tool_calls`, `distinct_tools_used` | Repeated tool calls and the tools used. `has_tool_loop` is TRUE when a turn makes the same tool call, with the same arguments, three or more times in a row. |
| `total_prompt_tokens`, `total_completion_tokens`, `total_tokens`, `total_latency_ms` | Token and latency totals. |
| `has_truncated_content`, `eval_ready` | Whether the row is fit to judge. |
| `trace` | The conversation, as JSON. |

A turn row has the same counts, flags, and `trace`, for its one turn. It adds
`invocation_id` and `turn_index`, the turn's place in its session counted from
0, and has `turn_start_timestamp`, `turn_end_timestamp`, and `turn_latency_ms`
in place of the session's timestamps, turn count, and total latency.

### The trace column

`trace` holds the conversation as JSON in the
`google.cloud.aiplatform.v1beta1.AgentData` format, for agent evaluation. Its
`agents` object maps each agent id to the configuration that the agent logged:
its instruction, description, sub-agents, and function declarations. Its
`turns` array holds each turn, numbered from 0 in start order, with a
`turn_index`, a `turn_id`, and the turn's `events` in order, each with an
`author`, a `content`, and an `event_time`. In a turn row, `turns` holds only
that turn, numbered 0 as single-turn metrics expect, while the `turn_index`
column keeps its place in the session.

### When a row is ready to judge

A turn is ready when it has user text, has finished with an
`INVOCATION_COMPLETED` event and an `AGENT_RESPONSE` event that has text, and
its trace has events and no truncated payload. A session is ready when every
turn is ready, with one exception: a turn that paused for a tool or for human
input, and the turn whose function response resumes it, are ready together,
because the paused turn has no reply of its own and the resuming turn has no
user text.

`has_truncated_content` is TRUE when an event in the trace was logged with part
of its payload cut off, and such a row is never ready. The flag does not tell
you whether truncation is the only reason that a row is not ready, because no
output column holds the other conditions.

## Configuration options

The builders take the events table and three keyword-only options:

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `full_table_id` | `str` | *Required* | The events table, as `project.dataset.table`. |
| `raw_where_clause` | `str` | `""` | A condition on the events table that narrows the events that the query reads. |
| `where_clause` | `str` | `""` | A condition on the output columns that picks the rows to return. |
| `denied_columns` | `Collection[str] \| None` | `()` | The payload columns that the table leaves out. |

`full_table_id` is the table that the plugin writes. The builders quote it in
backticks, so it has to name a dataset and cannot contain a backtick, a
backslash, a control character, or a line break. The builders raise
`ValueError` for any of those.

`raw_where_clause` and `where_clause` are SQL that the builders insert as
written, inside parentheses of their own. Bind every value as a query
parameter, such as `@window_start`, and never splice untrusted text into them.
Each condition may start with `WHERE` and end with semicolons, which the
builders drop. A condition with no SQL outside its comments, or with a block
comment, string literal, or quoted identifier that is not closed, raises
`ValueError`. An empty condition adds no filter, so with both conditions empty
the query returns every row of the view.

`denied_columns` takes the `payload_column_denylist` of the plugin's
`BigQueryLoggerConfig`, in any collection. A table that the plugin writes
without `content_parts`, `attributes`, or `latency_ms` has no such column, so
the query reads an empty array or a NULL JSON value in its place. `content`
cannot be denied, because the trace is built from it, and neither can an
identity column such as `is_truncated`. The builders raise `ValueError` for
either. `None`, the default of `payload_column_denylist`, denies nothing.

### Choosing between `raw_where_clause` and `where_clause`

The two conditions answer different questions. `raw_where_clause` decides which
events the query sees, which is what lets BigQuery skip partitions.
`where_clause` decides which of the rebuilt rows come back, and it cannot make a
row incomplete.

Give `raw_where_clause` three properties:

- **A lower bound on `timestamp` before the first event of every session that
  you want to read.** BigQuery skips the daily partitions before that bound,
  which is where the savings come from. A session whose first event is older
  than the bound comes out partial: it loses its earlier turns and the agent
  configuration that they logged, and nothing in the output marks it as
  partial. A wider lookback scans more bytes and cuts short fewer sessions.
- **No upper bound earlier than the time that the job runs.** A bound at the
  end of your window cuts off every session that is still running at that
  moment. The cut session looks finished: its end timestamp falls inside the
  window and it can pass `eval_ready`, so the job judges part of a
  conversation, then judges it again, whole, in a later window. Leave the upper
  bound out, or set it to the time that the job runs or later.
- **No filter on a column that changes within a session, such as `agent` or
  `event_type`.** Such a filter removes some of the events of every session and
  leaves traces with holes.

Put the time window in `where_clause`, on a timestamp of the rebuilt row,
together with `eval_ready`: `session_end_timestamp` for sessions, and
`turn_end_timestamp` or `turn_start_timestamp` for turns. Run the job only
after the end of the window plus the time that the plugin takes to make its
writes readable, which depends on its batching and retry settings, so that the
last events of the window are in the table when the query runs.

A job that runs too early can keep a session or turn from ever being evaluated,
because `timestamp` holds the time that the plugin logs an event rather than the
time that the event becomes readable. A session or turn that ends in the window,
but whose final `INVOCATION_COMPLETED` event is not readable yet when the job
runs, is not `eval_ready`, so the job skips it. Once that event is readable, the
row still ends inside the window, so the `where_clause` of every later window
leaves it out. No later job reads such a turn, and a later job reads such a
session only if its user returns to it.

A session can come back. Filtering on `session_end_timestamp` reads a session in
the window where its last turn ended, as of the moment that the job runs, so a
user who returns to the session later moves its end into a later window, where
the job reads it again with the new turns. Deduplicate on `session_id` when each
session should be judged once.

For turns, the timestamp that you filter on decides what happens to a turn that
is still running when the job runs. Filtering on `turn_end_timestamp` reads a
turn in the window where it finishes, so a later job reads that turn once it
finishes. Filtering on `turn_start_timestamp` reads a turn in the window where
it starts, and a turn that is still running then is not ready, so no later job
reads it.

## Advanced applications

These applications build on the same two conditions.

### Read the turns of a window

The following query returns the turns that finished during the window, for
metrics that judge one exchange at a time. Its lookback still has to reach the
first event of each session, because a turn keeps its `turn_index` and the
system prompt that its session logged first.

```python
from google.adk.plugins import build_eval_turn_query_sql

sql = build_eval_turn_query_sql(
    "my-project.agent_analytics.agent_events",
    raw_where_clause="timestamp >= @lookback_start",
    where_clause=(
        "turn_end_timestamp >= @window_start"
        " AND turn_end_timestamp < @window_end"
        " AND eval_ready"
    ),
)
```

### Sort, page, or aggregate the rows

Because the query ends without a semicolon, a job can append clauses to it or
wrap it in a subquery. The following snippet builds one query that sorts the
ready sessions by end time and returns the first 100, and one that counts the
ready sessions of each root agent.

```python
from google.adk.plugins import build_eval_session_query_sql

sql = build_eval_session_query_sql(
    "my-project.agent_analytics.agent_events",
    raw_where_clause="timestamp >= @lookback_start",
    where_clause="eval_ready",
)
page_sql = f"{sql}\nORDER BY session_end_timestamp DESC\nLIMIT 100"
count_sql = (
    f"SELECT root_agent, COUNT(*) AS sessions FROM ({sql}) GROUP BY root_agent"
)
```

### Read chosen sessions

To evaluate a known set of sessions again, add their ids to
`raw_where_clause`. `session_id` is the same for every event of a session, so
the condition keeps each chosen session whole while every step of the query
processes fewer rows. BigQuery still scans every partition after
`@lookback_start`, because the default clustering of the plugin's table does
not include `session_id`.

```python
from google.adk.plugins import build_eval_session_query_sql

sql = build_eval_session_query_sql(
    "my-project.agent_analytics.agent_events",
    raw_where_clause=(
        "timestamp >= @lookback_start AND session_id IN UNNEST(@session_ids)"
    ),
)
```

The job binds `@session_ids` as an array parameter, with
`bigquery.ArrayQueryParameter("session_ids", "STRING", session_ids)`.

### Run the SQL without Python

The SQL comes from versioned templates in
[`bigquery_agent_analytics_sql/`](../../../../src/google/adk/plugins/bigquery_agent_analytics_sql/),
the one copy of that SQL. A service that cannot import Python, such as a server
in another language, can fill the same templates. The header of each file
states its placeholders and the rules for each value, and the `_v1` in a file
name versions that contract: a change to the placeholders or to the output
columns comes as a new file.

## Limitations

The builders leave the following to the caller:

- **Nothing marks a partial row.** A `raw_where_clause` that drops some of a
  session's events yields a row that looks complete.
- **Partition pruning needs a partitioned table.** The lookback saves bytes on
  a table partitioned on `timestamp`, as the plugin creates it. On a table
  without that partitioning, the condition still narrows the rows, but
  BigQuery reads the whole table.
- **The conditions are checked, not parsed.** The builders reject a condition
  with no SQL outside its comments, and a block comment, string literal, or
  quoted identifier that is not closed. Any other mistake in a condition
  surfaces when BigQuery runs the query.
- **The builders do not check what the table holds.** A deployment that leaves
  tool or model events out of the table gets traces of the events that it
  logged.
- **A resumed session is read again.** Filtering on `session_end_timestamp`
  returns a session once per window in which it ends, as
  [Choosing between `raw_where_clause` and `where_clause`](#choosing-between-raw_where_clause-and-where_clause)
  describes.
