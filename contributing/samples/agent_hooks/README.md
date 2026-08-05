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
| `before_run_callback`         | `agent_startup`   |
| `on_user_message_callback`    | `input`           |
| `before_model_callback`       | `pre_model_call`  |
| `after_model_callback`        | `post_model_call` |
| `before_tool_callback`        | `pre_tool_call`   |
| `after_tool_callback`         | `post_tool_call`  |
| `on_event_callback` (final)   | `output`          |
| `after_run_callback`          | `agent_shutdown`  |

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

## Trust model

agent-hooks is a _cooperative_ control contract, **not** a security boundary:
interceptors run in-process with full data access and the interception points do
not guarantee complete mediation. See the agent-hooks
[`SECURITY.md`](https://github.com/responsibleai/agent-hooks/blob/main/SECURITY.md)
before relying on it for isolation.
