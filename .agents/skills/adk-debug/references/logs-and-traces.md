# Logs and traces

## Where the logs actually go

This differs per command and is the most common reason "I turned on `-v` and
saw nothing".

Command | Destination
--- | ---
`adk run` | `{tempdir}/agents_log/agent.{timestamp}.log`, i.e. `/tmp/agents_log/...` on Linux, with an `agent.latest.log` symlink. It clears the root logger's handlers, so **nothing** is logged to the terminal.
`adk web`, `adk api_server`, `adk eval`, everything else | stderr, via `logging.basicConfig`. No log file is created.

```bash
adk run -v {agent_dir} "{query}"
tail -F /tmp/agents_log/agent.latest.log
```

For `adk web`, tee stderr into a file so both you and the user can read it:

```bash
adk web -v {agents_dir} 2>&1 | tee {readable_path}/adk_web.log
```

`-v` is a shortcut for `--log_level DEBUG`; the levels are `DEBUG`, `INFO`,
`WARNING`, `ERROR`, `CRITICAL`. ADK's own records go to the `google_adk`
logger, so filter with `grep google_adk` or raise only that logger in-process.
Setup lives in `src/google/adk/cli/utils/logs.py`.

## Trace endpoints

Only `adk web` registers these. `adk api_server` runs the production-safe
`ApiServer`, which has no `/dev/...` routes, so trace lookups there 404.

```bash
# Every span for a session
curl -s http://localhost:8000/dev/apps/{app_name}/debug/trace/session/{session_id} \
  | python3 -m json.tool

# The single trace recorded against one event id
curl -s http://localhost:8000/dev/apps/{app_name}/debug/trace/{event_id} \
  | python3 -m json.tool
```

The session response is a list of spans, each with `name`, `span_id`,
`trace_id`, `parent_span_id`, `start_time`, `end_time`, and `attributes`.
Spans are kept in memory by the running server, so restarting it loses them.

Span name | What it covers
--- | ---
`call_llm` | One model call, including the `before_model` / `after_model` callbacks.
`execute_tool (merged)` | A batch of tool calls dispatched from one model response.
`generate_content {model}` | The underlying GenAI SDK call, when the OTel GenAI instrumentation is active.

## Reading what the model actually received

Pull the `call_llm` spans and decode `gcp.vertex.agent.llm_request` — it is a
JSON *string* holding `contents`, `config` (tools, `response_schema`,
`response_mime_type`, `system_instruction`), and `model`. This is the ground
truth for "why did the model do that": compare it against what you believe the
agent is configured to send.

Attribute | Meaning
--- | ---
`gcp.vertex.agent.llm_request` | Full request as a JSON string.
`gcp.vertex.agent.llm_response` | Full response as a JSON string.
`gcp.vertex.agent.tool_call_args` / `.tool_response` | Tool arguments and result.
`gcp.vertex.agent.event_id` | Correlates the span with an event in the session.
`gcp.vertex.agent.invocation_id` / `.session_id` | Correlates spans across one turn.
`gen_ai.request.model` | Model name as sent.
`gen_ai.usage.input_tokens` / `.output_tokens` | Token counts — check these before blaming a prompt for being ignored.
`gen_ai.response.finish_reasons` | List of lowercased reasons, e.g. `["max_tokens"]` for a truncated answer or `["safety"]` for a filtered one.

If the content-bearing attributes come back as `"{}"`, content capture is off —
see the env vars below, not a bug in the agent.

## Environment variables

Variable | Effect
--- | ---
`ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS` | Whether prompts and responses are written onto the legacy `gcp.vertex.agent.*` span attributes. Defaults to `true`; set `false` to strip content.
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | OTel-spec content capture for the GenAI semantic-convention spans and log records.
`ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN` | Pins the telemetry schema to `1` (default off Agent Engine) or `2`. Under `2` the `invocation` span becomes `invoke_workflow` and `call_llm` goes away, so check this first if the span names above are missing.
`GOOGLE_CLOUD_PROJECT` | Required by `adk web --trace_to_cloud`; without it the server logs a warning and exports nothing.

`--trace_to_cloud` only exports traces. `--otel_to_cloud` is the newer flag and
covers Cloud Trace plus Cloud Logging; `adk deploy agent_engine` already warns
that `--trace_to_cloud` is being replaced by it.

Source: `src/google/adk/telemetry/tracing.py`,
`src/google/adk/telemetry/context.py`,
`src/google/adk/telemetry/_schema_version.py`.
