---
name: adk-sample-creator
description: >-
  Creates a new sample agent in the ADK Python repository — the sample
  directory, its `agent.py`, and its `README.md` — following the conventions
  the existing samples already use. Use when the user wants to add a sample or
  example demonstrating a feature or agent pattern (dynamic nodes,
  fan-out/fan-in, a standalone tool-using agent), asks where a new sample
  belongs under `contributing/samples/`, or wants an existing sample's README
  brought up to the standard structure. Don't use for building a real working
  agent for the user's own project (use `adk-agent-builder`), or for checking
  whether the Python blocks in a Markdown file run (use `adk-verify-snippets`).
---

# ADK Sample Creator

Creates samples under `contributing/samples/`. These are deliberately minimal
agents that each exercise one or two features — distinct from the `adk-samples`
repository, which hosts full end-to-end applications.

Read the `adk-style` skill first for ADK 2.0 conventions if you have not
already.

## 1. Pick the category directory

Almost every sample lives at
`contributing/samples/{category}/{sample_name}/`. List the categories and
confirm with the user which one the sample belongs in before creating
anything — a workflow sample landing outside `workflows/` is the usual mistake.

```bash
ls contributing/samples/
```

Categories include `workflows`, `patterns`, `core`, `multi_agent`, `tools`,
`models`, `live`, `mcp`, `a2a`, `evaluation`, and `plugins`. A handful of
samples nest one level further when a single feature needs several variants, as
`plugins/plugin_reflect_tool_retry/basic/` does.

Name the sample directory in `snake_case` after the feature it demonstrates:
`dynamic_nodes`, `fan_out_fan_in`, `streaming_tool_events`.

Do not add an `_agent` suffix, and do not repeat the category as a prefix —
every sample is an agent, and the category is already in the path. Many existing
directories still carry both; do not copy them.

## 2. Write `agent.py`

Contents of a sample directory:

| File | Required | Purpose |
| --- | --- | --- |
| `agent.py` | yes | The agent or workflow. Must expose `root_agent`. |
| `README.md` | yes | See [readme-template.md](references/readme-template.md). |
| `__init__.py` | sometimes | Present when the sample is imported as a package. |
| `tests/*.json` | no | Recorded sessions used as eval sets. |

Use absolute imports so the file can be run and imported directly.

Do not set `model=` on `Agent` instances. Samples inherit the
system-configured model, which keeps them working when the default model
changes; hardcoding `model="gemini-2.5-flash"` pins the sample to a model that
will be retired. Set it only when the user explicitly asks for a specific model.

Then pick one of the two shapes.

### Pattern A — Workflow, for multi-step graphs

Use when the sample needs multiple nodes, routing, or parallel execution.

```python
from google.adk import Agent
from google.adk import Context
from google.adk import Event
from google.adk import Workflow
from google.adk.workflow import JoinNode
from google.adk.workflow import node
```

Import `Workflow` from `google.adk`, not from a private
`google.adk.workflow._*` module.

```python
my_agent = Agent(name="my_agent", instruction="...")


@node()
async def my_node(node_input: str) -> str:
  return "result"


root_agent = Workflow(
    name="root_agent",
    edges=[("START", my_node)],
)
```

A plain function can be used as a node directly in `edges`; reach for the
`@node(...)` decorator when you need one of its options, such as
`rerun_on_resume=True` for a node that calls `ctx.run_node`.

### Pattern B — Standalone agent, for single-agent or simple tool use

Use when there is no graph and the agent drives its own loop.

```python
from google.adk import Agent
from google.adk.tools import google_search

root_agent = Agent(
    name="standalone_assistant",
    instruction="You are a helpful assistant.",
    description="An assistant that can help with queries.",
    tools=[google_search],
)
```

## 3. Write `README.md`

Follow [readme-template.md](references/readme-template.md) — section order,
prompt formatting, the Mermaid topology rules, and the relative link depth for
`docs/guides/`.

## Worked examples

Read these two before writing a new Pattern A sample — one dynamic graph, one
static one.

-   `contributing/samples/workflows/dynamic_nodes/agent.py` — a Python node
    driving a `while` loop with `ctx.run_node`, so the number of agent calls is
    decided at runtime rather than by the edges.

    ```python
    @node(rerun_on_resume=True)
    async def orchestrate(ctx: Context, node_input: str) -> str:
      yield Event(state={"topic": node_input})

      while True:
        headline = await ctx.run_node(generate_headline)
        # ...
    ```

-   `contributing/samples/workflows/fan_out_fan_in/agent.py` — three functions
    run in parallel from `START`, collected by a `JoinNode`, then aggregated.

    ```python
    join_node = JoinNode(name="join_for_results")

    root_agent = Workflow(
        name="root_agent",
        edges=[(
            "START",
            (make_uppercase, count_characters, reverse_string),
            join_node,
            aggregate,
        )],
    )
    ```
