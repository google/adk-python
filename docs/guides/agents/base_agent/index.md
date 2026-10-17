# BaseAgent

`BaseAgent` is the abstract base class for all agent implementations in ADK. It defines the execution contract for turn-based processing, sub-agent hierarchies, event generation, and lifecycle callbacks across standalone agents and workflow nodes.

## Introduction

An application often requires deterministic control flow, custom container orchestrators, or non-LLM logic that still participates in session history, event streaming, and sub-agent hierarchies. Standard LLM agents provide model-driven reasoning, but custom architectures require a predictable execution interface that integrates directly with `Runner` and `Workflow`.

`BaseAgent` provides this foundation by extending `BaseNode` with agent-specific lifecycle hooks and sub-agent coordination. Concrete classes such as `LlmAgent` build upon `BaseAgent`, while custom implementations subclass it directly to produce specialized orchestration behavior without modifying runtime infrastructure.

## Get started

Subclass `BaseAgent` and implement the abstract `_run_async_impl` method. This method receives an `InvocationContext` and yields `Event` instances as execution proceeds.

The example below creates a deterministic agent that echoes user input back as a model response, wraps it in an `App`, and executes it through a `Runner`.

```python
import asyncio
from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents import InvocationContext
from google.adk.apps import App
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


class EchoAgent(BaseAgent):
  """A deterministic agent that formats input into a greeting response."""

  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    user_text = ""
    if ctx.user_content and ctx.user_content.parts:
      for part in ctx.user_content.parts:
        if part.text:
          user_text = part.text
          break

    response_text = f"Echo: {user_text}"
    yield Event(
        author=self.name,
        branch=ctx.branch,
        invocation_id=ctx.invocation_id,
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=response_text)],
        ),
    )


async def main() -> None:
  agent = EchoAgent(name="echo_agent", description="Echoes user input.")
  app = App(name="echo_app", root_agent=agent)
  session_service = InMemorySessionService()
  runner = Runner(app=app, session_service=session_service)

  session = await session_service.create_session(
      app_name="echo_app", user_id="user_1"
  )
  message = types.Content(
      role="user",
      parts=[types.Part.from_text(text="Hello ADK")],
  )

  async for event in runner.run_async(
      user_id="user_1",
      session_id=session.id,
      new_message=message,
  ):
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.text:
          print(part.text)


if __name__ == "__main__":
  asyncio.run(main())
```

Every agent requires a valid Python identifier as its `name`. The runner passes invocation metadata in `InvocationContext`, which supplies the active session, execution branch, and incoming user message.

## How it works

When a caller invokes `run_async` on an agent, `BaseAgent` executes an orchestration pipeline around `_run_async_impl`.

First, `BaseAgent` executes any configured `before_agent_callback` handlers. If a before callback returns a truthy `types.Content`, `BaseAgent` skips `_run_async_impl` entirely, emits the returned content as an event, and terminates the turn early. This mechanism allows intercepting runs for cached responses, policy checks, or safety blocks.

When no callback intercepts execution, `BaseAgent` invokes `_run_async_impl` and yields each generated `Event` downstream to the runner or parent node.

After `_run_async_impl` finishes, `BaseAgent` invokes any configured `after_agent_callback` handlers. If an after callback returns a truthy `types.Content`, `BaseAgent` yields an additional `Event` containing that content before completing.

Because `BaseAgent` subclasses `BaseNode`, any agent instance functions as a workflow node. While `BaseAgent` implements `_run_impl` to run within a workflow, `BaseAgent` itself ignores `node_input`; derived classes like `LlmAgent` override `_run_impl` to bind upstream node output into user content.

Sub-agents declare parent-child relationships through the `sub_agents` list. During initialization, `BaseAgent` sets `parent_agent` references on all children, establishing the agent tree hierarchy used for branch isolation and turn routing.

## Configuration options

The table below lists configuration options introduced by `BaseAgent`.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `name` | `str` | Required | Unique identifier for the agent within the agent tree. |
| `description` | `str` | `""` | Summary of agent capabilities, used by parent agents and routers for delegation. |
| `sub_agents` | `list[BaseAgent]` | `[]` | Child agents managed directly by this agent. |
| `before_agent_callback` | `BeforeAgentCallback \| None` | `None` | Callback or list of callbacks executed before the agent runs. |
| `after_agent_callback` | `AfterAgentCallback \| None` | `None` | Callback or list of callbacks executed after the agent runs. |

The `name` field must be a valid Python identifier and cannot equal `"user"`, which ADK reserves for end-user input.

The `description` field provides routing information. In multi-agent systems, parent agents and models inspect descriptions to determine delegation targets.

The `sub_agents` list defines child participants. An agent instance can have at most one parent agent; reusing an agent instance across multiple parents or separate trees raises a `ValueError`.

The `before_agent_callback` option accepts a callable or list of callables receiving `Context`. Returning `None` continues normal execution, while returning a `types.Content` object bypasses the agent run.

The `after_agent_callback` option accepts a callable or list of callables receiving `Context`. Returning `types.Content` appends a trailing event to the session history.

## Advanced applications

### Custom router agent

A custom agent can inspect user input and dynamically delegate execution to specific sub-agents based on deterministic rules rather than model reasoning.

The example below inspects incoming text keywords and delegates to the appropriate child agent.

```python
from collections.abc import AsyncGenerator
from contextlib import aclosing

from google.adk.agents import BaseAgent
from google.adk.agents import InvocationContext
from google.adk.events import Event


class KeywordRouterAgent(BaseAgent):
  """Routes requests to sub-agents based on keyword matching."""

  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    text = ""
    if ctx.user_content and ctx.user_content.parts:
      for part in ctx.user_content.parts:
        if part.text:
          text = part.text.lower()
          break

    target_name = "general_agent"
    if "billing" in text:
      target_name = "billing_agent"
    elif "support" in text:
      target_name = "support_agent"

    target_agent = self.find_agent(target_name)
    if target_agent is None:
      raise ValueError(f"Unknown routing target: {target_name}")

    async with aclosing(target_agent.run_async(ctx)) as agen:
      async for event in agen:
        yield event
```

A custom agent delegates by resolving the target through `find_agent` and yielding every event the selected sub-agent produces. Yielding the child events keeps the sub-agent's output in the session history and in the runner's event stream.

Setting `transfer_to_agent` on `EventActions` does not achieve this routing. The runner selects the agent for the next turn from the event author and the pending function call, so a bare `transfer_to_agent` action on a custom agent event leaves the same agent active. `LlmAgent` honors `transfer_to_agent` only on an event that also carries a `transfer_to_agent` function response, which the built-in transfer tool emits.

### Guardrails with before callbacks

Before callbacks enforce authorization or content filtering before invoking heavier agent logic.

The example below blocks requests missing an authorization header in session state.

```python
from google.adk.agents import Context
from google.genai import types


def check_auth_callback(ctx: Context) -> types.Content | None:
  """Verifies authorization token existence before running the agent."""
  token = ctx.state.get("auth_token")
  if not token:
    return types.Content(
        role="model",
        parts=[types.Part.from_text(text="Access denied: missing auth token.")],
    )
  return None
```

Returning `types.Content` halts agent execution immediately and delivers the error message to the caller.

## Limitations

Sub-agent uniqueness is enforced per instance. An agent instance can have at most one parent across its entire lifecycle, meaning reusing the same instance under multiple parents or across separate agent trees raises a `ValueError`. When identical behavior is needed in multiple places, instantiate distinct agents with separate names.

The name `"user"` is reserved by ADK runtime internals and raises a Pydantic `ValidationError` during initialization.

The `_run_async_impl` method must be an asynchronous generator. Methods that return plain values rather than yielding events cause runtime failures during execution streaming.

## Related samples

- [Agent in Workflow](../../../../contributing/samples/workflows/agent_in_workflow/agent.py) - Demonstrates embedding agents as workflow nodes.
- [Multi-Agent Transfer](../../../../contributing/samples/multi_agent/three_layer_transfer/agent.py) - Demonstrates multi-agent coordination across parent and sub-agent hierarchies.
