# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The canonical agent and workflow graphs, and the runs that drive them.

The one-tool agent every other scenario is a variation on, the workflows it is
wrapped in (nested workflow, nested agents, agent-tool), and the ``run_*``
drivers that run one to completion and hand back the events it emitted.
"""

from __future__ import annotations

from contextlib import aclosing

from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.events.event import Event
from google.adk.models.base_llm import BaseLlm
from google.adk.runners import InMemoryRunner
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.workflow._base_node import START
from google.adk.workflow._workflow import Workflow
from google.genai.types import Content
from google.genai.types import Part

from ....testing_utils import TestInMemoryRunner
from .conversation import AGENT_DESCRIPTION
from .conversation import AGENT_NAME
from .conversation import AGENT_TOOL_WORKFLOW_NAME
from .conversation import BASE_INSTRUCTION
from .conversation import DELEGATE_AGENT_DESCRIPTION
from .conversation import DELEGATE_AGENT_NAME
from .conversation import DELEGATING_AGENT_DESCRIPTION
from .conversation import DELEGATING_AGENT_NAME
from .conversation import NESTED_AGENT_DESCRIPTION
from .conversation import NESTED_AGENT_NAME
from .conversation import NESTED_WORKFLOW_NAME
from .conversation import NODE_APP_NAME
from .conversation import NODE_RESULT
from .conversation import NODE_USER_ID
from .conversation import SPECIALIST_AGENT_DESCRIPTION
from .conversation import SPECIALIST_AGENT_NAME
from .conversation import TOOL_RESULT_PREFIX
from .conversation import USER_PROMPT
from .conversation import WORKFLOW_NAME


def build_test_agent(
    model: BaseLlm, *, tool_exception: Exception | None = None
) -> Agent:
  """Builds the canonical 1-tool, 2-LLM-turn agent around ``model``.

  ``model`` comes from ``inference_under_test``, which pairs it with the
  matching instrumentation. With ``tool_exception`` the tool raises it
  instead of returning, exercising the tool-failure telemetry path.
  """

  def some_tool(arg1: str) -> str:
    """A sample tool."""
    if tool_exception is not None:
      raise tool_exception

    return f"{TOOL_RESULT_PREFIX}{arg1}"

  return Agent(
      name=AGENT_NAME,
      description=AGENT_DESCRIPTION,
      instruction=BASE_INSTRUCTION,
      model=model,
      tools=[FunctionTool(some_tool)],
  )


def build_multi_agent_test_agent(model: BaseLlm) -> Agent:
  """Builds the canonical two-agent turn: the root hands off to a specialist.

  One model call each, billing the same two usages the single-agent scenario
  spends over its two calls. The turn totals therefore match that scenario's
  exactly, and only the per-agent split tells the two recordings apart --
  which is the point: it is where an agent's spend is booked that a turn-grain
  metric has to get right.
  """
  specialist = Agent(
      name=SPECIALIST_AGENT_NAME,
      description=SPECIALIST_AGENT_DESCRIPTION,
      instruction=BASE_INSTRUCTION,
      model=model,
  )
  return Agent(
      name=AGENT_NAME,
      description=AGENT_DESCRIPTION,
      instruction=BASE_INSTRUCTION,
      model=model,
      sub_agents=[specialist],
  )


def build_test_runner(
    model: BaseLlm, *, tool_exception: Exception | None = None
) -> TestInMemoryRunner:
  """Builds a runner around the canonical agent (no workflow wrapper)."""
  return TestInMemoryRunner(
      node=build_test_agent(model, tool_exception=tool_exception)
  )


def build_multi_agent_test_runner(model: BaseLlm) -> TestInMemoryRunner:
  """Builds a runner around the two-agent handoff."""
  return TestInMemoryRunner(node=build_multi_agent_test_agent(model))


def build_test_workflow(
    model: BaseLlm, *, tool_exception: Exception | None = None
) -> Workflow:
  """Builds the canonical Workflow: a nested workflow feeding the agent.

  The nested workflow's node is a plain function, which spends nothing, so the
  nested workflow earns a `gen_ai.workflow.nested` span and duration but no
  token datapoint.
  """
  test_agent = build_test_agent(model, tool_exception=tool_exception)

  async def some_node(ctx, node_input):
    return NODE_RESULT

  # Trivial workflow to test o11y of nested workflows
  nested_workflow = Workflow(
      name=NESTED_WORKFLOW_NAME,
      edges=[(START, some_node)],
  )

  return Workflow(
      name=WORKFLOW_NAME,
      edges=[(START, nested_workflow, test_agent)],
  )


def build_nested_agents_test_workflow(model: BaseLlm) -> Workflow:
  """Builds the canonical Workflow with an agent as the nested node.

  A nested workflow has to run an agent for the token grain to have anything
  to report.
  """
  test_agent = build_test_agent(model)

  nested_workflow = Workflow(
      name=NESTED_WORKFLOW_NAME,
      edges=[(START, build_nested_test_agent(model))],
  )

  return Workflow(
      name=WORKFLOW_NAME,
      edges=[(START, nested_workflow, test_agent)],
  )


def build_nested_test_agent(model: BaseLlm) -> Agent:
  """Builds the single-turn agent the nested workflow runs, when it runs one."""
  return Agent(
      name=NESTED_AGENT_NAME,
      description=NESTED_AGENT_DESCRIPTION,
      instruction=BASE_INSTRUCTION,
      model=model,
  )


def build_agent_tool_test_workflow(model: BaseLlm) -> Workflow:
  """Builds the graph whose agent calls an ``AgentTool``, starting a Runner.

  An ``AgentTool`` runs the agent it wraps on a Runner of its own, nested
  inside the turn that called the tool. What that Runner spends therefore
  belongs to the calling turn, and not to a turn of its own.
  """
  delegate = Agent(
      name=DELEGATE_AGENT_NAME,
      description=DELEGATE_AGENT_DESCRIPTION,
      instruction=BASE_INSTRUCTION,
      model=model,
  )
  delegating_agent = Agent(
      name=DELEGATING_AGENT_NAME,
      description=DELEGATING_AGENT_DESCRIPTION,
      instruction=BASE_INSTRUCTION,
      model=model,
      tools=[AgentTool(agent=delegate)],
  )
  return Workflow(
      name=AGENT_TOOL_WORKFLOW_NAME,
      edges=[(START, delegating_agent)],
  )


async def _run_workflow(
    workflow: Workflow, event_sink: list[Event] | None
) -> list[Event]:
  """Runs a workflow to completion, draining the event stream."""
  runner = InMemoryRunner(app_name=NODE_APP_NAME, node=workflow)
  session = await runner.session_service.create_session(
      app_name=NODE_APP_NAME, user_id=NODE_USER_ID
  )
  content = Content(parts=[Part.from_text(text=USER_PROMPT)], role="user")

  collected_events: list[Event] = event_sink if event_sink is not None else []

  async with aclosing(
      runner.run_async(
          user_id=NODE_USER_ID,
          session_id=session.id,
          new_message=content,
      )
  ) as agen:
    async for event in agen:
      collected_events.append(event)

  return collected_events


async def run_node_scenario(
    model: BaseLlm,
    *,
    tool_exception: Exception | None = None,
    event_sink: list[Event] | None = None,
) -> list[Event]:
  """Runs the workflow scenario to completion, draining the event stream.

  If ``event_sink`` is provided, collected events are appended to it as they
  are drained. This lets callers inspect the events that were emitted before
  an exception propagates (e.g. when ``tool_exception`` is set).
  """
  return await _run_workflow(
      build_test_workflow(model, tool_exception=tool_exception), event_sink
  )


async def run_agent_tool_scenario(
    model: BaseLlm, *, event_sink: list[Event] | None = None
) -> list[Event]:
  """Runs the agent-tool workflow scenario to completion."""
  return await _run_workflow(build_agent_tool_test_workflow(model), event_sink)


async def run_nested_agents_scenario(
    model: BaseLlm, *, event_sink: list[Event] | None = None
) -> list[Event]:
  """Runs the nested-agent workflow scenario to completion."""
  return await _run_workflow(
      build_nested_agents_test_workflow(model), event_sink
  )


async def run_streaming_agent_scenario(
    runner: TestInMemoryRunner, *, event_sink: list[Event] | None = None
) -> list[Event]:
  """Runs the canonical agent with SSE streaming turned on.

  Drives the runner directly rather than through the shared helper, which
  offers no way to ask for a streaming run.
  """
  collected_events: list[Event] = event_sink if event_sink is not None else []
  session = await runner.session_service.create_session(
      app_name="InMemoryRunner", user_id="test_user"
  )
  agen = runner.run_async(
      user_id=session.user_id,
      session_id=session.id,
      new_message=Content(
          parts=[Part.from_text(text=USER_PROMPT)], role="user"
      ),
      run_config=RunConfig(streaming_mode=StreamingMode.SSE),
  )

  async with aclosing(agen):
    async for event in agen:
      collected_events.append(event)

  return collected_events


async def run_agent_scenario(
    runner: TestInMemoryRunner, *, event_sink: list[Event] | None = None
) -> list[Event]:
  """Runs the non-node scenario to completion, draining the event stream.

  Collects like ``run_node_scenario``: every scenario reports the events it
  emitted, and an ``event_sink`` keeps the ones that came before an
  exception.
  """
  collected_events: list[Event] = event_sink if event_sink is not None else []

  async with aclosing(
      runner.run_async_with_new_session_agen(
          Content(parts=[Part.from_text(text=USER_PROMPT)], role="user")
      )
  ) as agen:
    async for event in agen:
      collected_events.append(event)

  return collected_events
