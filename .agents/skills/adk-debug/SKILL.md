---
name: adk-debug
description: >-
  Diagnoses misbehaving ADK agents by inspecting sessions, events, tool calls,
  and the exact request that reached the model. Covers the `adk run` CLI and the
  `adk web` dev server with its session, trace, and debug HTTP endpoints. Use
  when an agent returns the wrong answer, ignores a tool or swallows a tool
  error, hangs, loops, emits raw JSON instead of calling tools, is not
  discovered by `adk web`, when a sub-agent cannot see the parent conversation,
  or when you need the LLM request/response, token counts, or logs for a run.
  Don't use for how ADK is designed internally (use `adk-architecture`), for
  building a new agent or workflow (use `adk-agent-builder`), for environment or
  dependency setup failures (use `adk-setup`), or for lint and style nits (use
  `adk-style`).
---

# Debugging ADK agents

Two entry points. Default to `adk run`: one process, no server, and `--jsonl`
output that pipes straight into `grep` or `python3`. Switch to `adk web` when
you need the browser UI, a persisted session you can click through, or the
trace endpoints that expose the exact LLM request.

## First moves

1. Reproduce headlessly: `adk run --jsonl {agent_dir} "{query}"`. Without
   `--jsonl`, `adk run` prints only text parts — tool calls and tool errors are
   invisible.
2. Read the log file. `adk run` writes to `/tmp/agents_log/agent.latest.log` and
   nothing to the terminal; `adk web` does the opposite. See
   [logs-and-traces.md](references/logs-and-traces.md).
3. Match the symptom in [failure-modes.md](references/failure-modes.md) before
   reading source — most reports are one of a handful of known shapes.
4. If the text is fine but the routing is not, dump the events and read
   `author`, `branch`, `nodeInfo.path`, and `actions` —
   [event-flow.md](references/event-flow.md).
5. If the model itself misbehaved, read what it actually received from the
   `call_llm` span rather than guessing from the agent definition —
   [logs-and-traces.md](references/logs-and-traces.md).

## References

- [cli-run.md](references/cli-run.md) — `adk run` flags, the JSONL event shape,
  multi-turn and human-in-the-loop resume, exit codes, driving a `Runner` from
  Python.
- [web-api.md](references/web-api.md) — starting `adk web`, listing and reading
  sessions over HTTP, posting test messages to `/run_sse`.
- [logs-and-traces.md](references/logs-and-traces.md) — log levels and where
  each command writes them, the trace endpoints, span attributes, and the env
  vars that control whether prompts appear in spans.
- [failure-modes.md](references/failure-modes.md) — ADK-specific symptoms with
  the cause and a concrete check for each.
- [event-flow.md](references/event-flow.md) — how an invocation becomes events,
  callback order, the event fields that matter, and where each stage lives in
  the source.

## Ground rules

- Leave sessions in place when you finish. The user may still want to open them
  in the web UI, and `adk web` has no undelete.
- Delete any throwaway agent you created for a repro, unless the user asked to
  keep it.
- Reach for a unit test in `tests/unittests/` when the bug is inside one
  component, and for a sample under `contributing/samples/` (see
  `adk-sample-creator`) when it only reproduces with runner, agent, and workflow
  wired together.
