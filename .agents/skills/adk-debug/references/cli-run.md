# Debugging with `adk run`

`adk run {agent_dir}` with a trailing query argument runs one turn and exits;
without a query it drops into an interactive prompt. Prefer the query form —
it needs no human in the loop and composes with shell tooling.

```bash
adk run --jsonl {agent_dir} "{query}"
adk run --jsonl --in_memory {agent_dir} "{query}"    # no persisted session
```

## Flags

Flag | Default | Why you'd use it
--- | --- | ---
`--jsonl` | off | One JSON object per event on stdout. Without it only text parts are printed, so tool calls, tool errors, and actions are invisible.
`--in_memory` | off | Skip the local session store, so repeated runs cannot contaminate each other.
`--session_id {id}` | new session | In query mode, reuse that session (creating it if absent) — this is how you carry state across separate `adk run` invocations. In interactive mode it only names the file `--save_session` writes.
`--state '{json}'` | none | Seed session state for a run that only misbehaves with particular state.
`--replay {file.json}` | none | Replay a saved state + query list into a fresh session. Mutually exclusive with a query argument.
`--resume {file.json}` | none | Reopen a session saved by `--save_session` and keep interacting (interactive mode only).
`--timeout 30s` | none | Bound a hanging turn instead of waiting forever.
`-v` / `--log_level DEBUG` | `INFO` | Raise log verbosity. Output goes to the log file, not the terminal — see `references/logs-and-traces.md`.
`--default_llm_model {model}` | none | Override the model for agents that do not set one, e.g. to test whether the model is the problem.

Full list: `adk run --help`.

## JSONL event shape

Each line is `Event.model_dump(mode='json', by_alias=True, exclude_none=True)`,
so keys are camelCase (`invocationId`, `functionCall`, `longRunningToolIds`),
with `session_id` and `node_path` injected and `author` first. Empty `actions`
entries are dropped, so an absent `actions` key means "no actions", not
"unknown".

In `--jsonl` mode stdout is pure JSONL; the human-readable session banner only
prints when `--jsonl` is off, and goes to stderr either way. So this is safe:

```bash
adk run --jsonl {agent_dir} "{query}" 2>/dev/null > /tmp/events.jsonl
head -1 /tmp/events.jsonl | python3 -m json.tool     # inspect the real shape
```

Read one event before writing a parser — the schema changes. Then filter on
whatever you actually saw, for example every tool call:

```python
import json

for line in open("/tmp/events.jsonl"):
  event = json.loads(line)
  for part in (event.get("content") or {}).get("parts", []):
    if "functionCall" in part:
      print(event["author"], part["functionCall"]["name"], part["functionCall"].get("args"))
```

## Exit codes

Code | Meaning
--- | ---
`0` | The turn completed.
`1` | Error — bad `--state` JSON, both a query and `--replay`, no query and no stdin, timeout, or an exception during the run.
`2` | Paused. The run emitted an event with `longRunningToolIds`, i.e. a human-in-the-loop tool is waiting.

On exit 2 the run prints the session id. Resume by re-running with that
`--session_id` and the answer as the query — ADK maps the query onto the
pending `adk_request_confirmation` / `adk_request_input` function response
automatically, so do not try to hand-craft a `FunctionResponse`. For a
confirmation, a plain `yes`/`no` works; pass a JSON object to supply a custom
payload.

## Driving a Runner from Python

Use this when you need to assert on events rather than eyeball them. Two
things that bite:

- `new_message` must be a `types.Content`, not a string.
- `Runner` takes keyword arguments only, and `auto_create_session` defaults to
  `False`, so the session must exist before you run.

```python
import asyncio

from google.adk import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types

agent = Agent(name="test", model="gemini-2.5-flash", instruction="...")
runner = InMemoryRunner(agent=agent, app_name="test")


async def main():
  session = await runner.session_service.create_session(
      app_name="test", user_id="u"
  )
  async for event in runner.run_async(
      user_id="u",
      session_id=session.id,
      new_message=types.Content(role="user", parts=[types.Part(text="hello")]),
  ):
    print(event.author, event.content)
    if event.actions.transfer_to_agent:
      print("  -> transfer to", event.actions.transfer_to_agent)
    if event.output is not None:
      print("  -> output:", event.output)


asyncio.run(main())
```

`InMemorySessionService.create_session_sync` still exists but logs a
deprecation warning; use the async `create_session`.

To print events the way the CLI does, without reimplementing the formatting:

```python
from google.adk.utils._debug_output import print_event

print_event(event)                # text parts only
print_event(event, verbose=True)  # plus tool calls, tool results, code, blobs
```

`verbose` is keyword-only. Source: `src/google/adk/utils/_debug_output.py`.
