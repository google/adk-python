# Governing an ADK agent with agent-hooks

This sample shows how to govern an ADK agent with
[agent-hooks](https://github.com/responsibleai/agent-hooks), a framework-neutral
_control_ contract for AI agent systems. You register one or more
**interceptors** (policy engines, content filters, egress guards, ...) once, and
[`AgentHooksPlugin`](../../../src/google/adk/plugins/_agent_hooks_plugin.py)
enforces their verdicts at every governed point in the ADK lifecycle.

## What it demonstrates

The agent is a customer-support assistant with two tools: `lookup_account` and
`delete_account`. A single [`ToolGovernanceInterceptor`](governance.py) applies
two policies:

- **deny** — `delete_account` is destructive, so the interceptor blocks the tool
  call before it runs. The model receives a policy error and tells the user it
  cannot perform the action.
- **transform** — `lookup_account` returns an `email` and an `api_key`. The
  interceptor redacts those fields _before the model or the transcript sees
  them_.

Every decision is recorded as an auditable `InterceptionRecord`; `main.py`
prints the trail at the end.

## Interception-point mapping

| ADK plugin callback           | agent-hooks point |
| ----------------------------- | ----------------- |
| `on_user_message_callback`    | `agent_startup` once, then `input` |
| `before_run_callback`         | startup fallback / deny enforcement |
| `before_model_callback`       | `pre_model_call`  |
| `after_model_callback`        | `post_model_call` |
| `on_model_error_callback`     | `post_model_call` |
| `before_tool_callback`        | `pre_tool_call`   |
| `after_tool_callback`         | `post_tool_call`  |
| `on_tool_error_callback`      | `post_tool_call`  |
| `on_event_callback` (final)   | `output`          |
| `on_run_complete_callback`    | `agent_shutdown` on success |
| `on_run_error_callback`       | `agent_shutdown` on error |
| `on_run_cancelled_callback`   | `agent_shutdown` on cancellation |

A model or tool call that **errors** is routed to `on_model_error_callback` /
`on_tool_error_callback`, so its `post_*` record is still emitted and paired with
the `pre_*` (the errored result is discarded and the error propagates).

The plugin emits `agent_startup` lazily before the first `input`. This preserves
the Agent Hooks ordering contract even though ADK invokes its user-message seam
before `before_run_callback`.

### Coverage and limitations

The mapping is faithful to what ADK's seams can express, with these deliberate
gaps:

- **`pre_model_call` transform is not applied** — rebuilding a provider-native
  request from wire messages is not round-trip safe, so a `transform` there is
  enforced as a fail-closed `deny` rather than silently applied.
- **Sub-agents do not emit their own `agent_startup`** — the run-level seam
  fires once per invocation, so a nested/transferred agent is governed at the
  model, tool, input, and output points but not with a separate startup record.
- **A `deny` at `output` cannot retract already-streamed tokens** — it replaces
  the final response event, but any partial content already yielded to the
  caller in a streaming run has left the process.
- **The approval resolver is not exposed by this adapter** — the test suite
  runs 32 packaged CTK vectors and maintains exhaustive, categorized skips for
  the 14 resolver-dependent vectors plus the post-tool custom-reason vector.


## Prerequisites

1. Install ADK with the optional `agent-hooks` extra, plus LiteLLM for the local
   model:

   ```bash
   pip install "google-adk[agent-hooks]" litellm
   ```

2. Install [Ollama](https://ollama.com/) and pull a tool-capable model:

   ```bash
   ollama pull qwen2.5
   ```

   The example runs against a real local model, so tool-calling behavior is not
   scripted. Any tool-capable Ollama model works; edit the `LiteLlm(model=...)`
   line in [`agent.py`](agent.py) to change it.

## Run

```bash
python -m contributing.samples.agent_hooks.main
```

Expected shape of the output:

- For "look up account 42", the agent calls `lookup_account` and summarizes the
  result — with `email` and `api_key` already redacted.
- For "delete account 42", the `delete_account` call is denied and the agent
  explains it cannot delete the account.
- The audit trail lists every interception point and its verdict, including the
  `pre_tool_call -> deny` and `post_tool_call -> transform` decisions.

## Enforcement semantics

`AgentHooksPlugin` **fails closed**: a `deny` blocks the guarded action, a
`transform` rewrites the guarded value, and any engine error, malformed verdict,
or interceptor timeout becomes a fail-closed block — it never fails open. Set
`mode="evaluate_only"` on the plugin to record decisions without enforcing them.

Interceptors must use `async def intercept(...)` when the timeout is enabled.
A synchronous interceptor can run only with `timeout=None`, which explicitly
accepts that Python cannot preempt a blocking in-process call. Multimodal ADK
content is preserved as structured JSON for policy inspection; unsupported or
over-deep values are blocked rather than silently truncated.

Agent Hooks reserves its governed plugin callbacks by default because ADK stops
the callback chain on the first replacement. Overlapping plugins are rejected
at registration time; `allow_unsafe_plugin_composition=True` is an explicit
opt-out for cooperative deployments that accept the bypass risk.

Every decision is written to payload-free structured logs. A custom
`record_sink` can persist the complete `InterceptionRecord`; sink failures block
in enforce mode by default. `audit_failure_mode="log"` keeps the policy result
instead, making lossy audit delivery an explicit choice.

Model-facing refusal content and tool errors use the fixed reason
`policy_denied`; custom policy reasons and messages stay in audit records and
operator metadata so policy internals cannot be reflected into the model.

## Trust model

agent-hooks is a _cooperative_ control contract, **not** a security boundary:
interceptors run in-process with full data access and the interception points do
not guarantee complete mediation. See the agent-hooks
[`SECURITY.md`](https://github.com/responsibleai/agent-hooks/blob/main/SECURITY.md)
before relying on it for isolation.
