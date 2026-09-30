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

"""Unit tests for _node_runner_utils helper module."""

from __future__ import annotations

from contextlib import aclosing
from typing import Any
from typing import AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.context import Context
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.events.event import Event
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow import _node_runner_utils
from google.adk.workflow._base_node import BaseNode
from google.genai import types
from opentelemetry import context as otel_context
import pytest


def _fc_part(name: str, id_: str) -> types.Part:
  return types.Part(
      function_call=types.FunctionCall(name=name, id=id_, args={})
  )


def _fr_part(name: str, id_: str) -> types.Part:
  return types.Part(
      function_response=types.FunctionResponse(name=name, id=id_, response={})
  )


class _SimpleTestNode(BaseNode):

  def __init__(self, name: str = "simple_node"):
    super().__init__(name=name)

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Event, None]:
    yield Event(
        author=self.name,
        content=types.Content(
            role="model", parts=[types.Part(text="node response")]
        ),
    )


class _MockLlmAgent(LlmAgent):
  """Mock LLM agent for unit testing."""

  def __init__(self, name: str):
    super().__init__(name=name, model="gemini-1.5-pro", sub_agents=[])

  async def _run_async_impl(
      self, invocation_context: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(
        invocation_id=invocation_context.invocation_id,
        author=self.name,
        content=types.Content(
            role="model", parts=[types.Part(text="Test LLM response")]
        ),
    )


class _FailingTestNode(BaseNode):

  def __init__(self, name: str = "failing_node"):
    super().__init__(name=name)

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Event, None]:
    raise RuntimeError("node boom")
    yield  # pylint: disable=unreachable


async def test_run_node_async_executes_node_and_yields_events():
  """Executing a node via run_node_async streams its emitted events."""
  session_service = InMemorySessionService()
  node = _SimpleTestNode("test_node")
  app = App(name="test_app", root_agent=BaseAgent(name="root_agent"))
  runner = Runner(app=app, session_service=session_service)

  session = await session_service.create_session(
      app_name="test_app", user_id="user_1", session_id="session_1"
  )

  events = []
  async for event in _node_runner_utils.run_node_async(
      runner,
      user_id="user_1",
      session_id="session_1",
      node=node,
      session=session,
      new_message=types.Content(
          role="user", parts=[types.Part.from_text(text="hello")]
      ),
  ):
    events.append(event)

  assert len(events) == 1
  assert events[0].author == "test_node"
  assert events[0].content.parts[0].text == "node response"


async def test_run_node_async_halts_on_early_exit_from_plugin():
  """Returning Content from before_run_callback halts execution early."""
  session_service = InMemorySessionService()
  node = _SimpleTestNode("test_node")

  class EarlyExitPlugin(BasePlugin):

    def __init__(self):
      super().__init__(name="early_exit")

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> types.Content | None:
      return types.Content(
          role="model", parts=[types.Part(text="blocked by safety")]
      )

  app = App(
      name="test_app",
      root_agent=BaseAgent(name="root_agent"),
      plugins=[EarlyExitPlugin()],
  )
  runner = Runner(app=app, session_service=session_service)

  session = await session_service.create_session(
      app_name="test_app", user_id="user_1", session_id="session_1"
  )

  events = []
  async for event in _node_runner_utils.run_node_async(
      runner,
      user_id="user_1",
      session_id="session_1",
      node=node,
      session=session,
  ):
    events.append(event)

  assert len(events) == 1
  assert events[0].author == "model"
  assert events[0].content.parts[0].text == "blocked by safety"


async def test_run_node_async_notifies_plugins_on_failure():
  """An unhandled error in node execution notifies error plugins before re-raising."""
  session_service = InMemorySessionService()
  node = _FailingTestNode("failing_node")

  error_notified = []

  class ErrorTrackingPlugin(BasePlugin):

    def __init__(self):
      super().__init__(name="error_tracker")

    async def on_run_error_callback(
        self, *, invocation_context: InvocationContext, error: Exception
    ) -> None:
      error_notified.append(error)

  app = App(
      name="test_app",
      root_agent=BaseAgent(name="root_agent"),
      plugins=[ErrorTrackingPlugin()],
  )
  runner = Runner(app=app, session_service=session_service)

  session = await session_service.create_session(
      app_name="test_app", user_id="user_1", session_id="session_1"
  )

  with pytest.raises(RuntimeError, match="node boom"):
    async for _ in _node_runner_utils.run_node_async(
        runner,
        user_id="user_1",
        session_id="session_1",
        node=node,
        session=session,
    ):
      pass

  assert len(error_notified) == 1
  assert str(error_notified[0]) == "node boom"


async def test_run_node_async_does_not_leak_context():
  """Caller OpenTelemetry context is preserved during run_node_async iteration."""

  class _TestEchoNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield "echo"

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="test_app",
      node=_TestEchoNode(name="test_node"),
      session_service=session_service,
      artifact_service=InMemoryArtifactService(),
      auto_create_session=True,
  )

  test_key = otel_context.create_key("test_key_run_node_async")
  token = otel_context.attach(
      otel_context.set_value(test_key, "caller_val_run_node_async")
  )
  caller_ctx = otel_context.get_current()
  try:
    events = []
    async with aclosing(
        _node_runner_utils.run_node_async(
            runner,
            user_id="user_1",
            session_id="session_1",
            new_message=types.Content(
                role="user", parts=[types.Part(text="hello")]
            ),
            yield_user_message=True,
        )
    ) as agen:
      async for event in agen:
        assert otel_context.get_current() == caller_ctx
        events.append(event)
    assert otel_context.get_current() == caller_ctx
    assert len(events) == 2
  finally:
    otel_context.detach(token)


async def test_run_node_async_prefers_response_owner_over_supplied_invocation_id():
  """A caller-supplied invocation id is reconciled against the response."""
  session_service = InMemorySessionService()
  node_agent = _MockLlmAgent("solo")
  runner = Runner(
      app=App(name="test_app", root_agent=node_agent),
      session_service=session_service,
  )
  session = await session_service.create_session(
      app_name="test_app", user_id="u", session_id="s"
  )
  await session_service.append_event(
      session,
      Event(
          invocation_id="inv_real",
          author="solo",
          content=types.Content(parts=[_fc_part("t", "fc-1")]),
      ),
  )

  used: dict[str, str] = {}
  original = runner._new_invocation_context

  def _capture(*args, **kwargs):
    ctx = original(*args, **kwargs)
    used.setdefault("invocation_id", ctx.invocation_id)
    return ctx

  runner._new_invocation_context = _capture

  async for _ in _node_runner_utils.run_node_async(
      runner,
      user_id="u",
      session_id="s",
      invocation_id="inv_wrong",
      new_message=types.Content(role="user", parts=[_fr_part("t", "fc-1")]),
      node=node_agent,
  ):
    pass

  assert used["invocation_id"] == "inv_real"


async def test_run_node_async_rejects_responses_straddling_two_invocations():
  """A supplied id does not allow resuming responses from multiple invocations."""
  session_service = InMemorySessionService()
  node_agent = _MockLlmAgent("solo")
  runner = Runner(
      app=App(name="test_app", root_agent=node_agent),
      session_service=session_service,
  )
  session = await session_service.create_session(
      app_name="test_app", user_id="u", session_id="s"
  )
  for invocation_id, call_id in (("inv_a", "fc-1"), ("inv_b", "fc-2")):
    await session_service.append_event(
        session,
        Event(
            invocation_id=invocation_id,
            author="solo",
            content=types.Content(parts=[_fc_part("t", call_id)]),
        ),
    )

  with pytest.raises(ValueError, match="multiple"):
    async for _ in _node_runner_utils.run_node_async(
        runner,
        user_id="u",
        session_id="s",
        invocation_id="inv_wrong",
        new_message=types.Content(
            role="user", parts=[_fr_part("t", "fc-1"), _fr_part("t", "fc-2")]
        ),
        node=node_agent,
    ):
      pass
