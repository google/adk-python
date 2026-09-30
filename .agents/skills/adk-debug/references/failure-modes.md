# ADK failure modes

Match the symptom first; each entry names the mechanism and a check you can
actually run.

## The agent emits raw JSON instead of calling tools

`output_schema` puts the model into controlled generation: it sets
`config.response_schema` and `config.response_mime_type =
"application/json"` on the request, and a model in JSON mode returns JSON, not
tool calls.

Check the `call_llm` span's `gcp.vertex.agent.llm_request` for
`response_mime_type` (`references/logs-and-traces.md`). ADK only
applies the schema when the agent has no tools, or when the model can do
schemas and tools together — so the opposite symptom, a schema that seems to be
ignored, means you landed on the other branch.

Source: `src/google/adk/flows/llm_flows/basic.py`,
`src/google/adk/utils/output_schema_utils.py`.

## `ValueError` when constructing the agent

Three messages come from the same validator, all meaning "you set this on
`generate_content_config` instead of on the agent":

- `All tools must be set via LlmAgent.tools.`
- `System instruction must be set via LlmAgent.instruction.`
- `Response schema must be set via LlmAgent.output_schema.`

Source: `LlmAgent.validate_generate_content_config` in
`src/google/adk/agents/llm_agent.py`.

## `LlmCallsLimitExceededError: Max number of llm calls limit of N exceeded`

`run_config.max_llm_calls` was hit. Treat the limit as a loop detector before
raising it — dump the events and look for the same tool being called with the
same arguments turn after turn. Source:
`src/google/adk/agents/invocation_context.py`.

## A tool "fails" but the agent carries on

Tool failures are converted into a function response carrying the error, so the
model sees a result and keeps going. Look for a `functionResponse` whose payload
has an `error` key. `FunctionTool` produces the same shape for two non-exception
cases: missing mandatory arguments, and a confirmation-required tool that was
not confirmed or was rejected.

To intervene, register `on_tool_error_callback` on a plugin or
`on_tool_error_callbacks` on the agent. Source:
`src/google/adk/flows/llm_flows/functions.py`,
`src/google/adk/tools/function_tool.py`.

## `adk web` does not list the agent, or returns 404

```bash
curl -s http://localhost:8000/list-apps | python3 -m json.tool
```

The loader accepts four layouts under `{agents_dir}`, checking for a top-level
`app` before `root_agent`:

```text
{name}/agent.py          # defines root_agent (or app)
{name}.py                # defines root_agent (or app)
{name}/__init__.py       # defines root_agent (or app) in the package
{name}/root_agent.yaml   # config-defined agent
```

`__init__.py` does not need `from . import agent` — the loader imports the
`agent` submodule itself. Pointing `adk web` at a directory that itself contains
`agent.py` or `root_agent.yaml` runs that single agent instead of treating the
directory as a collection. Source:
`src/google/adk/cli/utils/agent_loader.py`.

## A sub-agent cannot see the parent conversation

Events carry a `branch` (`agent_1.agent_2.agent_3`), and the content builder
drops events that do not belong to the current agent's branch — that isolation
is deliberate, so peers do not read each other's history. Delegated task agents
are isolated further by `isolation_scope`.

There is no flag to switch it off. Put whatever the sub-agent needs into the
delegation input; the sub-agent's `description` is what steers the parent into
including it. Source: `_is_event_belongs_to_branch` in
`src/google/adk/flows/llm_flows/contents.py`.

## The whole agent stalls while one tool runs

A synchronous tool function is awaited inline on the event loop, so anything
blocking inside it — a `requests` call, `time.sleep`, a large file read —
freezes the entire run, not just that tool.

Make the tool `async`. In live mode only, you can instead hand tools to a thread
pool:

```python
from google.adk.agents.run_config import RunConfig, ToolThreadPoolConfig

run_config = RunConfig(tool_thread_pool_config=ToolThreadPoolConfig())  # 4 workers
```

Source: `FunctionTool._invoke_callable` in
`src/google/adk/tools/function_tool.py`,
`_call_tool_in_thread_pool` in `src/google/adk/flows/llm_flows/functions.py`.

## The run stops early and nothing looks wrong

`adk run` exits 2 when an event carries `longRunningToolIds`: a
human-in-the-loop tool is waiting for an answer. See
`references/cli-run.md` for how to resume.

## The answer is cut off, empty, or blocked

Read `gen_ai.response.finish_reasons` on the `call_llm` span rather than
inferring from the text — `max_tokens` means raise `max_output_tokens`,
`safety` and `recitation` mean the model refused. See
`references/logs-and-traces.md`.
