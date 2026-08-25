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

import asyncio
from contextlib import aclosing
import importlib
import logging
from pathlib import Path
import sys
import textwrap
from typing import AsyncGenerator
from typing import Optional
from unittest import mock
from unittest.mock import AsyncMock
from unittest.mock import create_autospec
from unittest.mock import patch

from google.adk import runners
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm.task._finish_task_tool import FINISH_TASK_ERROR_RESULT
from google.adk.agents.llm.task._finish_task_tool import FINISH_TASK_SUCCESS_RESULT
from google.adk.agents.llm.task._finish_task_tool import FINISH_TASK_TOOL_NAME
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.agents.run_config import RunConfig
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.cli.utils.agent_loader import AgentLoader
from google.adk.errors.session_not_found_error import SessionNotFoundError
from google.adk.events.event import Event
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.runners import Runner
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.base_session_service import GetSessionConfig
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.tools.base_toolset import BaseToolset
from google.genai import types
import pytest

from tests.unittests import testing_utils

TEST_APP_ID = "test_app"
TEST_USER_ID = "test_user"
TEST_SESSION_ID = "test_session"


class MockAgent(BaseAgent):
  """Mock agent for unit testing."""

  def __init__(
      self,
      name: str,
      parent_agent: Optional[BaseAgent] = None,
  ):
    super().__init__(name=name, sub_agents=[])
    # BaseAgent doesn't have disallow_transfer_to_parent field
    # This is intentional as we want to test non-LLM agents
    if parent_agent:
      self.parent_agent = parent_agent

  async def _run_async_impl(
      self, invocation_context: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(
        invocation_id=invocation_context.invocation_id,
        author=self.name,
        content=types.Content(
            role="model", parts=[types.Part(text="Test response")]
        ),
    )


class MockLiveAgent(BaseAgent):
  """Mock live agent for unit testing."""

  def __init__(self, name: str):
    super().__init__(name=name, sub_agents=[])

  async def _run_live_impl(
      self, invocation_context: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(
        invocation_id=invocation_context.invocation_id,
        author=self.name,
        content=types.Content(
            role="model", parts=[types.Part(text="live hello")]
        ),
    )


class MockLlmAgent(LlmAgent):
  """Mock LLM agent for unit testing."""

  def __init__(
      self,
      name: str,
      disallow_transfer_to_parent: bool = False,
      parent_agent: Optional[BaseAgent] = None,
  ):
    # Use a string model instead of mock
    super().__init__(name=name, model="gemini-1.5-pro", sub_agents=[])
    self.disallow_transfer_to_parent = disallow_transfer_to_parent
    self.parent_agent = parent_agent

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


class MockAgentWithMetadata(BaseAgent):
  """Mock agent that returns event-level custom metadata."""

  def __init__(self, name: str):
    super().__init__(name=name, sub_agents=[])

  async def _run_async_impl(
      self, invocation_context: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(
        invocation_id=invocation_context.invocation_id,
        author=self.name,
        content=types.Content(
            role="model", parts=[types.Part(text="Test response")]
        ),
        custom_metadata={"event_key": "event_value"},
    )


class MockPlugin(BasePlugin):
  """Mock plugin for unit testing."""

  ON_USER_CALLBACK_MSG = (
      "Modified user message ON_USER_CALLBACK_MSG from MockPlugin"
  )
  ON_EVENT_CALLBACK_MSG = "Modified event ON_EVENT_CALLBACK_MSG from MockPlugin"
  ON_EVENT_CALLBACK_METADATA = {"plugin_key": "plugin_value"}

  def __init__(self):
    super().__init__(name="mock_plugin")
    self.enable_user_message_callback = False
    self.enable_event_callback = False
    self.user_content_seen_in_before_run_callback = None

  async def on_user_message_callback(
      self,
      *,
      invocation_context: InvocationContext,
      user_message: types.Content,
  ) -> Optional[types.Content]:
    if not self.enable_user_message_callback:
      return None
    return types.Content(
        role="model",
        parts=[types.Part(text=self.ON_USER_CALLBACK_MSG)],
    )

  async def before_run_callback(
      self,
      *,
      invocation_context: InvocationContext,
  ) -> None:
    self.user_content_seen_in_before_run_callback = (
        invocation_context.user_content
    )

  async def on_event_callback(
      self, *, invocation_context: InvocationContext, event: Event
  ) -> Optional[Event]:
    if not self.enable_event_callback:
      return None
    return Event(
        invocation_id="",
        author="",
        content=types.Content(
            parts=[
                types.Part(
                    text=self.ON_EVENT_CALLBACK_MSG,
                )
            ],
            role=event.content.role,
        ),
        custom_metadata=self.ON_EVENT_CALLBACK_METADATA,
    )


class TestRunnerFindAgentToRun:
  """Tests for Runner._find_agent_to_run method."""

  def setup_method(self):
    """Set up test fixtures."""
    self.session_service = InMemorySessionService()
    self.artifact_service = InMemoryArtifactService()

    # Create test agents
    self.root_agent = MockLlmAgent("root_agent")
    self.sub_agent1 = MockLlmAgent("sub_agent1", parent_agent=self.root_agent)
    self.sub_agent2 = MockLlmAgent("sub_agent2", parent_agent=self.root_agent)
    self.non_transferable_agent = MockLlmAgent(
        "non_transferable",
        disallow_transfer_to_parent=True,
        parent_agent=self.root_agent,
    )

    self.root_agent.sub_agents = [
        self.sub_agent1,
        self.sub_agent2,
        self.non_transferable_agent,
    ]

    self.runner = Runner(
        app_name="test_app",
        agent=self.root_agent,
        session_service=self.session_service,
        artifact_service=self.artifact_service,
    )

  def test_find_agent_to_run_with_function_response_scenario(self):
    """Test finding agent when last event is function response."""
    # Create a function call from sub_agent1
    function_call = types.FunctionCall(id="func_123", name="test_func", args={})
    function_response = types.FunctionResponse(
        id="func_123", name="test_func", response={}
    )

    call_event = Event(
        invocation_id="inv1",
        author="sub_agent1",
        content=types.Content(
            role="model", parts=[types.Part(function_call=function_call)]
        ),
    )

    response_event = Event(
        invocation_id="inv2",
        author="user",
        content=types.Content(
            role="user", parts=[types.Part(function_response=function_response)]
        ),
    )

    session = Session(
        id="test_session",
        user_id="test_user",
        app_name="test_app",
        events=[call_event, response_event],
    )

    result = self.runner._find_agent_to_run(session, self.root_agent)
    assert result == self.sub_agent1

  def test_find_agent_to_run_returns_root_agent_when_no_events(self):
    """Test that root agent is returned when session has no non-user events."""
    session = Session(
        id="test_session",
        user_id="test_user",
        app_name="test_app",
        events=[
            Event(
                invocation_id="inv1",
                author="user",
                content=types.Content(
                    role="user", parts=[types.Part(text="Hello")]
                ),
            )
        ],
    )

    result = self.runner._find_agent_to_run(session, self.root_agent)
    assert result == self.root_agent

  def test_find_agent_to_run_returns_root_agent_when_found_in_events(self):
    """Test that root agent is returned when it's found in session events."""
    session = Session(
        id="test_session",
        user_id="test_user",
        app_name="test_app",
        events=[
            Event(
                invocation_id="inv1",
                author="root_agent",
                content=types.Content(
                    role="model", parts=[types.Part(text="Root response")]
                ),
            )
        ],
    )

    result = self.runner._find_agent_to_run(session, self.root_agent)
    assert result == self.root_agent

  def test_find_agent_to_run_returns_transferable_sub_agent(self):
    """Test that transferable sub agent is returned when found."""
    session = Session(
        id="test_session",
        user_id="test_user",
        app_name="test_app",
        events=[
            Event(
                invocation_id="inv1",
                author="sub_agent1",
                content=types.Content(
                    role="model", parts=[types.Part(text="Sub agent response")]
                ),
            )
        ],
    )

    result = self.runner._find_agent_to_run(session, self.root_agent)
    assert result == self.sub_agent1

  def test_find_agent_to_run_skips_non_transferable_agent(self):
    """Test that non-transferable agent is skipped and root agent is returned."""
    session = Session(
        id="test_session",
        user_id="test_user",
        app_name="test_app",
        events=[
            Event(
                invocation_id="inv1",
                author="non_transferable",
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="Non-transferable response")],
                ),
            )
        ],
    )

    result = self.runner._find_agent_to_run(session, self.root_agent)
    assert result == self.root_agent

  def test_find_agent_to_run_skips_unknown_agent(self):
    """Test that unknown agent is skipped and root agent is returned."""
    session = Session(
        id="test_session",
        user_id="test_user",
        app_name="test_app",
        events=[
            Event(
                invocation_id="inv1",
                author="unknown_agent",
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="Unknown agent response")],
                ),
            ),
            Event(
                invocation_id="inv2",
                author="root_agent",
                content=types.Content(
                    role="model", parts=[types.Part(text="Root response")]
                ),
            ),
        ],
    )

    result = self.runner._find_agent_to_run(session, self.root_agent)
    assert result == self.root_agent

  def test_find_agent_to_run_function_response_takes_precedence(self):
    """Test that function response scenario takes precedence over other logic."""
    # Create a function call from sub_agent2
    function_call = types.FunctionCall(id="func_456", name="test_func", args={})
    function_response = types.FunctionResponse(
        id="func_456", name="test_func", response={}
    )

    call_event = Event(
        invocation_id="inv1",
        author="sub_agent2",
        content=types.Content(
            role="model", parts=[types.Part(function_call=function_call)]
        ),
    )

    # Add another event from root_agent
    root_event = Event(
        invocation_id="inv2",
        author="root_agent",
        content=types.Content(
            role="model", parts=[types.Part(text="Root response")]
        ),
    )

    response_event = Event(
        invocation_id="inv3",
        author="user",
        content=types.Content(
            role="user", parts=[types.Part(function_response=function_response)]
        ),
    )

    session = Session(
        id="test_session",
        user_id="test_user",
        app_name="test_app",
        events=[call_event, root_event, response_event],
    )

    # Function-response routing only applies when resumability is enabled.
    self.runner.resumability_config = ResumabilityConfig(is_resumable=True)

    # Should return sub_agent2 due to function response, not root_agent
    result = self.runner._find_agent_to_run(session, self.root_agent)
    assert result == self.sub_agent2

  def test_find_agent_to_run_skips_function_response_when_not_resumable(self):
    """Test that function response scenario is skipped when not resumable."""
    function_call = types.FunctionCall(id="func_456", name="test_func", args={})
    function_response = types.FunctionResponse(
        id="func_456", name="test_func", response={}
    )

    call_event = Event(
        invocation_id="inv1",
        author="non_transferable",
        content=types.Content(
            role="model", parts=[types.Part(function_call=function_call)]
        ),
    )

    response_event = Event(
        invocation_id="inv2",
        author="user",
        content=types.Content(
            role="user", parts=[types.Part(function_response=function_response)]
        ),
    )

    session = Session(
        id="test_session",
        user_id="test_user",
        app_name="test_app",
        events=[call_event, response_event],
    )

    self.runner.resumability_config = ResumabilityConfig(is_resumable=False)

    result = self.runner._find_agent_to_run(session, self.root_agent)
    assert result == self.root_agent

  def test_find_agent_to_run_uses_function_response_when_resumable(self):
    """Test that function response scenario is used when resumable."""
    function_call = types.FunctionCall(id="func_456", name="test_func", args={})
    function_response = types.FunctionResponse(
        id="func_456", name="test_func", response={}
    )

    call_event = Event(
        invocation_id="inv1",
        author="non_transferable",
        content=types.Content(
            role="model", parts=[types.Part(function_call=function_call)]
        ),
    )

    response_event = Event(
        invocation_id="inv2",
        author="user",
        content=types.Content(
            role="user", parts=[types.Part(function_response=function_response)]
        ),
    )

    session = Session(
        id="test_session",
        user_id="test_user",
        app_name="test_app",
        events=[call_event, response_event],
    )

    self.runner.resumability_config = ResumabilityConfig(is_resumable=True)

    result = self.runner._find_agent_to_run(session, self.root_agent)
    assert result == self.non_transferable_agent

  def test_find_agent_to_run_resumable_unknown_function_call_author_falls_back(
      self,
  ):
    """Resumable routing must not return None for an unknown call author.

    When the matching function-call event is authored by something that is not
    an agent in the current hierarchy (e.g. "user", or a stale/foreign agent
    name carried over from a previous turn/session), `find_agent` returns None.
    Previously this None propagated to `build_node`, raising a confusing
    "Invalid node type: <class 'NoneType'>" error. We now fall through to the
    root agent instead.
    """
    function_call = types.FunctionCall(id="func_456", name="test_func", args={})
    function_response = types.FunctionResponse(
        id="func_456", name="test_func", response={}
    )

    # The function call is authored by "user", which is not an agent name.
    call_event = Event(
        invocation_id="inv1",
        author="user",
        content=types.Content(
            role="model", parts=[types.Part(function_call=function_call)]
        ),
    )

    response_event = Event(
        invocation_id="inv2",
        author="user",
        content=types.Content(
            role="user", parts=[types.Part(function_response=function_response)]
        ),
    )

    session = Session(
        id="test_session",
        user_id="test_user",
        app_name="test_app",
        events=[call_event, response_event],
    )

    self.runner.resumability_config = ResumabilityConfig(is_resumable=True)

    result = self.runner._find_agent_to_run(session, self.root_agent)
    assert result == self.root_agent

  def test_find_agent_to_run_resumable_stale_function_call_author_falls_back(
      self,
  ):
    """Resumable routing falls back to root for a stale/foreign call author."""
    function_call = types.FunctionCall(id="func_789", name="test_func", args={})
    function_response = types.FunctionResponse(
        id="func_789", name="test_func", response={}
    )

    # The function call is authored by an agent that is not in the hierarchy.
    call_event = Event(
        invocation_id="inv1",
        author="agent_from_a_previous_session",
        content=types.Content(
            role="model", parts=[types.Part(function_call=function_call)]
        ),
    )

    response_event = Event(
        invocation_id="inv2",
        author="user",
        content=types.Content(
            role="user", parts=[types.Part(function_response=function_response)]
        ),
    )

    session = Session(
        id="test_session",
        user_id="test_user",
        app_name="test_app",
        events=[call_event, response_event],
    )

    self.runner.resumability_config = ResumabilityConfig(is_resumable=True)

    result = self.runner._find_agent_to_run(session, self.root_agent)
    assert result == self.root_agent

  def test_is_transferable_across_agent_tree_with_llm_agent(self):
    """Test _is_transferable_across_agent_tree with LLM agent."""
    result = self.runner._is_transferable_across_agent_tree(self.sub_agent1)
    assert result is True

  def test_is_transferable_across_agent_tree_with_non_transferable_agent(self):
    """Test _is_transferable_across_agent_tree with non-transferable agent."""
    result = self.runner._is_transferable_across_agent_tree(
        self.non_transferable_agent
    )
    assert result is False

  def test_is_transferable_across_agent_tree_with_non_llm_agent(self):
    """Test _is_transferable_across_agent_tree with non-LLM agent."""
    non_llm_agent = MockAgent("non_llm_agent")
    # MockAgent inherits from BaseAgent, not LlmAgent, so it should return False
    result = self.runner._is_transferable_across_agent_tree(non_llm_agent)
    assert result is False


@pytest.mark.asyncio
async def test_session_not_found_message_includes_alignment_hint():

  class RunnerWithMismatch(Runner):

    def _infer_agent_origin(
        self, agent: BaseAgent
    ) -> tuple[Optional[str], Optional[Path]]:
      del agent
      return "expected_app", Path("/workspace/agents/expected_app")

  session_service = InMemorySessionService()
  runner = RunnerWithMismatch(
      app_name="configured_app",
      agent=MockLlmAgent("root_agent"),
      session_service=session_service,
      artifact_service=InMemoryArtifactService(),
  )

  agen = runner.run_async(
      user_id="user",
      session_id="missing",
      new_message=types.Content(role="user", parts=[]),
  )

  with pytest.raises(SessionNotFoundError) as excinfo:
    await agen.__anext__()

  await agen.aclose()

  message = str(excinfo.value)
  assert "Session not found" in message
  assert "configured_app" in message
  assert "expected_app" in message
  assert "Ensure the runner app_name matches" in message


@pytest.mark.asyncio
async def test_session_auto_creation():

  class RunnerWithMismatch(Runner):

    def _infer_agent_origin(
        self, agent: BaseAgent
    ) -> tuple[Optional[str], Optional[Path]]:
      del agent
      return "expected_app", Path("/workspace/agents/expected_app")

  session_service = InMemorySessionService()
  runner = RunnerWithMismatch(
      app_name="expected_app",
      agent=MockLlmAgent("test_agent"),
      session_service=session_service,
      artifact_service=InMemoryArtifactService(),
      auto_create_session=True,
  )

  agen = runner.run_async(
      user_id="user",
      session_id="missing",
      new_message=types.Content(role="user", parts=[types.Part(text="hi")]),
  )

  event = await agen.__anext__()
  await agen.aclose()

  # Verify that session_id="missing" doesn't error out - session is auto-created
  assert event.author == "test_agent"
  assert event.content.parts[0].text == "Test LLM response"


@pytest.mark.asyncio
async def test_rewind_auto_create_session_on_missing_session():
  """When auto_create_session=True, rewind should create session if missing.

  The newly created session won't contain the target invocation, so
  `rewind_async` should raise an Invocation ID not found error (rather than
  a session not found error), demonstrating auto-creation occurred.
  """
  session_service = InMemorySessionService()
  runner = Runner(
      app_name="auto_create_app",
      agent=MockLlmAgent("agent_for_rewind"),
      session_service=session_service,
      artifact_service=InMemoryArtifactService(),
      auto_create_session=True,
  )

  with pytest.raises(ValueError, match=r"Invocation ID not found: inv_missing"):
    await runner.rewind_async(
        user_id="user",
        session_id="missing",
        rewind_before_invocation_id="inv_missing",
    )

  # Verify the session actually exists now due to auto-creation.
  session = await session_service.get_session(
      app_name="auto_create_app", user_id="user", session_id="missing"
  )
  assert session is not None
  assert session.app_name == "auto_create_app"


@pytest.mark.asyncio
async def test_run_live_auto_create_session():
  """run_live should auto-create session when missing and yield events."""
  session_service = InMemorySessionService()
  artifact_service = InMemoryArtifactService()
  runner = Runner(
      app_name="live_app",
      agent=MockLiveAgent("live_agent"),
      session_service=session_service,
      artifact_service=artifact_service,
      auto_create_session=True,
  )

  # An empty LiveRequestQueue is sufficient for our mock agent.
  from google.adk.agents.live_request_queue import LiveRequestQueue

  live_queue = LiveRequestQueue()

  agen = runner.run_live(
      user_id="user",
      session_id="missing",
      live_request_queue=live_queue,
  )

  event = await agen.__anext__()
  await agen.aclose()

  assert event.author == "live_agent"
  assert event.content.parts[0].text == "live hello"

  # Session should have been created automatically.
  session = await session_service.get_session(
      app_name="live_app", user_id="user", session_id="missing"
  )
  assert session is not None


def test_run_passes_state_delta():
  """run should forward state_delta down to run_async."""
  import asyncio

  session_service = InMemorySessionService()
  runner = Runner(
      app_name=TEST_APP_ID,
      agent=MockAgent("test_agent"),
      session_service=session_service,
      artifact_service=InMemoryArtifactService(),
      auto_create_session=True,
  )

  state_delta = {"test_key": "test_value"}

  events = list(
      runner.run(
          user_id=TEST_USER_ID,
          session_id=TEST_SESSION_ID,
          new_message=types.Content(
              role="user", parts=[types.Part(text="hello")]
          ),
          state_delta=state_delta,
      )
  )

  assert len(events) >= 1

  session = asyncio.run(
      session_service.get_session(
          app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
      )
  )
  session_events = session.events

  user_event = next(e for e in session_events if e.author == "user")
  assert user_event.actions.state_delta == state_delta


def test_run_reraises_agent_error():
  """run should re-raise an error the agent raised on the worker thread."""

  class FailingAgent(BaseAgent):

    async def _run_async_impl(
        self, invocation_context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
      raise ValueError("agent failed")
      yield  # pragma: no cover

  runner = Runner(
      app_name=TEST_APP_ID,
      agent=FailingAgent(name="failing_agent"),
      session_service=InMemorySessionService(),
      artifact_service=InMemoryArtifactService(),
      auto_create_session=True,
  )

  with pytest.raises(ValueError, match="agent failed"):
    list(
        runner.run(
            user_id=TEST_USER_ID,
            session_id=TEST_SESSION_ID,
            new_message=types.Content(
                role="user", parts=[types.Part(text="hello")]
            ),
        )
    )


def test_run_yields_events_before_reraising_agent_error():
  """run should deliver the events produced before the failure."""

  class FailsAfterOneEventAgent(BaseAgent):

    async def _run_async_impl(
        self, invocation_context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
      yield Event(
          invocation_id=invocation_context.invocation_id,
          author=self.name,
          content=types.Content(role="model", parts=[types.Part(text="hi")]),
      )
      raise ValueError("agent failed late")

  runner = Runner(
      app_name=TEST_APP_ID,
      agent=FailsAfterOneEventAgent(name="failing_agent"),
      session_service=InMemorySessionService(),
      artifact_service=InMemoryArtifactService(),
      auto_create_session=True,
  )

  events = []
  with pytest.raises(ValueError, match="agent failed late"):
    for event in runner.run(
        user_id=TEST_USER_ID,
        session_id=TEST_SESSION_ID,
        new_message=types.Content(
            role="user", parts=[types.Part(text="hello")]
        ),
    ):
      events.append(event)

  assert [e.author for e in events] == ["failing_agent"]


def test_run_reports_agent_cancellation_as_runtime_error():
  """A cancellation is reported without being re-raised as a CancelledError.

  Re-raising it on the calling thread would read as the caller having been
  cancelled, which an enclosing event loop absorbs silently.
  """

  class CancelledAgent(BaseAgent):

    async def _run_async_impl(
        self, invocation_context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
      raise asyncio.CancelledError("agent cancelled")
      yield  # pragma: no cover

  runner = Runner(
      app_name=TEST_APP_ID,
      agent=CancelledAgent(name="cancelled_agent"),
      session_service=InMemorySessionService(),
      artifact_service=InMemoryArtifactService(),
      auto_create_session=True,
  )

  with pytest.raises(RuntimeError, match="CancelledError") as exc_info:
    list(
        runner.run(
            user_id=TEST_USER_ID,
            session_id=TEST_SESSION_ID,
            new_message=types.Content(
                role="user", parts=[types.Part(text="hello")]
            ),
        )
    )

  assert isinstance(exc_info.value.__cause__, asyncio.CancelledError)


@pytest.mark.asyncio
async def test_run_async_propagates_invocation_id():
  """run_async should propagate invocation_id to the invocation context and events."""

  session_service = InMemorySessionService()
  runner = Runner(
      app_name=TEST_APP_ID,
      agent=MockAgent("test_agent"),
      session_service=session_service,
      artifact_service=InMemoryArtifactService(),
      auto_create_session=True,
  )

  custom_invocation_id = "my_custom_invocation_id"

  agen = runner.run_async(
      user_id=TEST_USER_ID,
      session_id=TEST_SESSION_ID,
      new_message=types.Content(role="user", parts=[types.Part(text="hello")]),
      invocation_id=custom_invocation_id,
  )

  events = []
  async with aclosing(agen) as a:
    async for event in a:
      events.append(event)

  assert len(events) >= 1
  # Verify yielded events have the custom invocation ID
  for event in events:
    assert event.invocation_id == custom_invocation_id

  # Verify the session has the custom invocation ID in its events
  session = await session_service.get_session(
      app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )
  assert session is not None
  assert len(session.events) == 2
  for event in session.events:
    assert event.invocation_id == custom_invocation_id


@pytest.mark.asyncio
async def test_run_live_persists_event_callback_modifications():
  """run_live should persist the same event it streams after callback changes."""
  session_service = InMemorySessionService()
  artifact_service = InMemoryArtifactService()
  plugin = MockPlugin()
  plugin.enable_event_callback = True
  runner = Runner(
      app_name="live_app",
      agent=MockLiveAgent("live_agent"),
      session_service=session_service,
      artifact_service=artifact_service,
      plugins=[plugin],
  )
  await session_service.create_session(
      app_name="live_app", user_id="user", session_id="live_session"
  )

  from google.adk.agents.live_request_queue import LiveRequestQueue

  live_queue = LiveRequestQueue()
  agen = runner.run_live(
      user_id="user",
      session_id="live_session",
      live_request_queue=live_queue,
  )

  streamed_event = await agen.__anext__()
  await agen.aclose()

  session = await session_service.get_session(
      app_name="live_app", user_id="user", session_id="live_session"
  )
  persisted_event = session.events[0]

  assert streamed_event.author == "live_agent"
  assert streamed_event.invocation_id
  assert streamed_event.content.parts[0].text == (
      MockPlugin.ON_EVENT_CALLBACK_MSG
  )
  assert streamed_event.custom_metadata == MockPlugin.ON_EVENT_CALLBACK_METADATA

  assert persisted_event.id == streamed_event.id
  assert persisted_event.timestamp == streamed_event.timestamp
  assert persisted_event.author == streamed_event.author
  assert persisted_event.invocation_id == streamed_event.invocation_id
  assert persisted_event.content.parts[0].text == (
      MockPlugin.ON_EVENT_CALLBACK_MSG
  )
  assert (
      persisted_event.custom_metadata == MockPlugin.ON_EVENT_CALLBACK_METADATA
  )


@pytest.mark.asyncio
async def test_runner_allows_nested_agent_directories(tmp_path, monkeypatch):
  project_root = tmp_path / "workspace"
  agent_dir = project_root / "agents" / "examples" / "hello_world"
  agent_dir.mkdir(parents=True)
  # Make package structure importable.
  for pkg_dir in [
      project_root / "agents",
      project_root / "agents" / "examples",
      agent_dir,
  ]:
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
  # Extra directories that previously confused origin inference, e.g. virtualenv.
  (project_root / "agents" / ".venv").mkdir()

  agent_source = textwrap.dedent("""\
      from google.adk.events.event import Event
      from google.adk.agents.base_agent import BaseAgent
      from google.genai import types


      class SimpleAgent(BaseAgent):

        def __init__(self):
          super().__init__(name='simplest_agent', sub_agents=[])

        async def _run_async_impl(self, invocation_context):
          yield Event(
              invocation_id=invocation_context.invocation_id,
              author=self.name,
              content=types.Content(
                  role='model',
                  parts=[types.Part(text='hello from nested')],
              ),
          )


      root_agent = SimpleAgent()
      """)
  (agent_dir / "agent.py").write_text(agent_source, encoding="utf-8")

  monkeypatch.chdir(project_root)
  loader = AgentLoader(agents_dir="agents/examples")
  loaded_agent = loader.load_agent("hello_world")

  assert isinstance(loaded_agent, BaseAgent)
  session_service = InMemorySessionService()
  artifact_service = InMemoryArtifactService()
  runner = Runner(
      app_name="hello_world",
      agent=loaded_agent,
      session_service=session_service,
      artifact_service=artifact_service,
  )
  assert runner._app_name_alignment_hint is None

  session = await session_service.create_session(
      app_name="hello_world",
      user_id="user",
  )
  agen = runner.run_async(
      user_id=session.user_id,
      session_id=session.id,
      new_message=types.Content(
          role="user",
          parts=[types.Part(text="hi")],
      ),
  )
  event = await agen.__anext__()
  await agen.aclose()

  assert event.author == "simplest_agent"
  assert event.content
  assert event.content.parts
  assert event.content.parts[0].text == "hello from nested"


@pytest.mark.asyncio
async def test_run_config_custom_metadata_propagates_to_events():
  session_service = InMemorySessionService()
  runner = Runner(
      app_name=TEST_APP_ID,
      agent=MockAgentWithMetadata("metadata_agent"),
      session_service=session_service,
      artifact_service=InMemoryArtifactService(),
  )
  await session_service.create_session(
      app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )

  run_config = RunConfig(custom_metadata={"request_id": "req-1"})
  events = [
      event
      async for event in runner.run_async(
          user_id=TEST_USER_ID,
          session_id=TEST_SESSION_ID,
          new_message=types.Content(role="user", parts=[types.Part(text="hi")]),
          run_config=run_config,
      )
  ]

  assert events[0].custom_metadata is not None
  assert events[0].custom_metadata["request_id"] == "req-1"
  assert events[0].custom_metadata["event_key"] == "event_value"

  session = await session_service.get_session(
      app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )
  user_event = next(event for event in session.events if event.author == "user")
  assert user_event.custom_metadata == {"request_id": "req-1"}


@pytest.mark.asyncio
async def test_run_config_custom_metadata_stamps_user_event_in_chat_mode():
  """LlmAgent chat path stamps the user event with run-level custom_metadata."""
  session_service = InMemorySessionService()

  def _before_agent_callback(callback_context) -> types.Content:
    del callback_context  # Unused; short-circuits the model call.
    return types.Content(role="model", parts=[types.Part(text="hi back")])

  agent = LlmAgent(
      name="chat_agent", before_agent_callback=_before_agent_callback
  )
  runner = Runner(
      app_name=TEST_APP_ID, agent=agent, session_service=session_service
  )
  await session_service.create_session(
      app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )

  run_config = RunConfig(custom_metadata={"turn_id": "t-1"})
  async for _ in runner.run_async(
      user_id=TEST_USER_ID,
      session_id=TEST_SESSION_ID,
      new_message=types.Content(role="user", parts=[types.Part(text="hi")]),
      run_config=run_config,
  ):
    pass

  session = await session_service.get_session(
      app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )
  user_event = next(event for event in session.events if event.author == "user")
  assert user_event.custom_metadata == {"turn_id": "t-1"}


@pytest.mark.asyncio
async def test_runner_root_task_mode_promotes_finish_task_output():
  """Root LlmAgent(mode='task') promotes the finish_task output onto an event."""
  session_service = InMemorySessionService()
  agent = LlmAgent(
      name="task_agent",
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(
                  name="finish_task", args={"result": "the answer"}
              )
          ]
      ),
      mode="task",
  )
  runner = Runner(
      app_name=TEST_APP_ID, agent=agent, session_service=session_service
  )
  await session_service.create_session(
      app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )

  events = []
  async for event in runner.run_async(
      user_id=TEST_USER_ID,
      session_id=TEST_SESSION_ID,
      new_message=types.Content(
          role="user", parts=[types.Part(text="do task")]
      ),
  ):
    events.append(event)

  outputs = [e.output for e in events if e.output is not None]
  assert outputs, f"no event carried .output; events={events}"
  assert any(
      isinstance(o, dict) and o.get("result") == "the answer" for o in outputs
  ), f"finish_task output not promoted onto event.output; got {outputs}"


@pytest.mark.asyncio
async def test_runner_root_task_mode_unwraps_primitive_output():
  """Root LlmAgent(mode='task') unwraps primitive output schemas on promotion."""
  session_service = InMemorySessionService()
  agent = LlmAgent(
      name="task_agent",
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(
                  name="finish_task", args={"result": 42}
              )
          ]
      ),
      mode="task",
      output_schema=int,
  )
  runner = Runner(
      app_name=TEST_APP_ID, agent=agent, session_service=session_service
  )
  await session_service.create_session(
      app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )

  events = []
  async for event in runner.run_async(
      user_id=TEST_USER_ID,
      session_id=TEST_SESSION_ID,
      new_message=types.Content(
          role="user", parts=[types.Part(text="do task")]
      ),
  ):
    events.append(event)

  outputs = [e.output for e in events if e.output is not None]
  assert outputs == [42], f"finish_task output was not unwrapped; got {outputs}"


@pytest.mark.asyncio
async def test_runner_root_task_mode_writes_output_key_to_session_state():
  """Root LlmAgent(mode='task') with output_key writes result to session state."""
  session_service = InMemorySessionService()
  agent = LlmAgent(
      name="task_agent",
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(
                  name="finish_task", args={"result": "key_value"}
              )
          ]
      ),
      mode="task",
      output_key="my_result_key",
  )
  runner = Runner(
      app_name=TEST_APP_ID, agent=agent, session_service=session_service
  )
  await session_service.create_session(
      app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )

  events = []
  async for event in runner.run_async(
      user_id=TEST_USER_ID,
      session_id=TEST_SESSION_ID,
      new_message=types.Content(
          role="user", parts=[types.Part(text="do task")]
      ),
  ):
    events.append(event)

  session = await session_service.get_session(
      app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )
  assert session.state.get("my_result_key") == {"result": "key_value"}


@pytest.mark.asyncio
async def test_runner_raises_on_root_llm_agent_with_single_turn_mode():
  """Runner raises ValueError if root LlmAgent runs with mode='single_turn'."""
  session_service = InMemorySessionService()
  agent = LlmAgent(name="single_turn_agent", mode="single_turn")
  runner = Runner(
      app_name=TEST_APP_ID, agent=agent, session_service=session_service
  )
  await session_service.create_session(
      app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )

  with pytest.raises(
      ValueError,
      match=(
          "LlmAgent as root agent must have mode='chat' or 'task', but got"
          " mode='single_turn'."
      ),
  ):
    async for _ in runner.run_async(
        user_id=TEST_USER_ID,
        session_id=TEST_SESSION_ID,
        new_message=types.Content(
            role="user", parts=[types.Part(text="do task")]
        ),
    ):
      pass


@pytest.mark.asyncio
async def test_chat_mode_fetches_session_once_per_turn():
  """Root LlmAgent chat path reuses the prologue fetch inside the node run."""
  session_service = InMemorySessionService()

  def _before_agent_callback(callback_context) -> types.Content:
    del callback_context  # Unused; short-circuits the model call.
    return types.Content(role="model", parts=[types.Part(text="hi back")])

  agent = LlmAgent(
      name="chat_agent", before_agent_callback=_before_agent_callback
  )
  runner = Runner(
      app_name=TEST_APP_ID, agent=agent, session_service=session_service
  )
  original_get_session = session_service.get_session
  await session_service.create_session(
      app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )

  spy = AsyncMock(wraps=session_service.get_session)
  session_service.get_session = spy

  async for _ in runner.run_async(
      user_id=TEST_USER_ID,
      session_id=TEST_SESSION_ID,
      new_message=types.Content(role="user", parts=[types.Part(text="hi")]),
  ):
    pass

  assert spy.call_count == 1

  # Correctness: the user message is still persisted despite the single fetch.
  session = await original_get_session(
      app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )
  assert any(event.author == "user" for event in session.events)


@pytest.mark.asyncio
async def test_chat_mode_honors_get_session_config():
  """Root LlmAgent chat path threads get_session_config into the fetch."""
  session_service = InMemorySessionService()

  def _before_agent_callback(callback_context) -> types.Content:
    del callback_context  # Unused; short-circuits the model call.
    return types.Content(role="model", parts=[types.Part(text="hi back")])

  agent = LlmAgent(
      name="chat_agent", before_agent_callback=_before_agent_callback
  )
  runner = Runner(
      app_name=TEST_APP_ID, agent=agent, session_service=session_service
  )
  session = await session_service.create_session(
      app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )
  for i in range(3):
    await session_service.append_event(
        session=session,
        event=Event(
            invocation_id=f"seed-{i}",
            author="user",
            content=types.Content(
                role="user", parts=[types.Part(text=f"seed-{i}")]
            ),
        ),
    )

  seen_configs = []
  seen_event_counts = []
  original_get_session = session_service.get_session

  async def _spy_get_session(*args, **kwargs):
    fetched = await original_get_session(*args, **kwargs)
    seen_configs.append(kwargs.get("config"))
    seen_event_counts.append(None if fetched is None else len(fetched.events))
    return fetched

  session_service.get_session = _spy_get_session

  run_config = RunConfig(
      get_session_config=GetSessionConfig(num_recent_events=1)
  )
  async for _ in runner.run_async(
      user_id=TEST_USER_ID,
      session_id=TEST_SESSION_ID,
      new_message=types.Content(role="user", parts=[types.Part(text="hi")]),
      run_config=run_config,
  ):
    pass

  assert seen_configs
  assert all(
      config == GetSessionConfig(num_recent_events=1) for config in seen_configs
  )
  # num_recent_events=1 bounds the fetched history to the single latest event.
  assert all(count == 1 for count in seen_event_counts)


class TestRunnerWithPlugins:
  """Tests for Runner with plugins."""

  def setup_method(self):
    self.plugin = MockPlugin()
    self.session_service = InMemorySessionService()
    self.artifact_service = InMemoryArtifactService()
    self.root_agent = MockLlmAgent("root_agent")
    self.runner = Runner(
        app_name="test_app",
        agent=MockLlmAgent("test_agent"),
        session_service=self.session_service,
        artifact_service=self.artifact_service,
        plugins=[self.plugin],
    )

  async def run_test(self, original_user_input="Hello") -> list[Event]:
    """Prepares the test by creating a session and running the runner."""
    await self.session_service.create_session(
        app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
    )
    events = []
    async for event in self.runner.run_async(
        user_id=TEST_USER_ID,
        session_id=TEST_SESSION_ID,
        new_message=types.Content(
            role="user", parts=[types.Part(text=original_user_input)]
        ),
    ):
      events.append(event)
    return events

  @pytest.mark.asyncio
  async def test_runner_is_initialized_with_plugins(self):
    """Test that the runner is initialized with plugins."""
    await self.run_test()

    assert self.runner.plugin_manager is not None

  @pytest.mark.asyncio
  async def test_runner_modifies_user_message_before_execution(self):
    """Test that the runner modifies the user message before execution."""
    original_user_input = "original_input"
    self.plugin.enable_user_message_callback = True

    await self.run_test(original_user_input=original_user_input)
    session = await self.session_service.get_session(
        app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
    )
    generated_event = session.events[0]
    modified_user_message = generated_event.content.parts[0].text

    assert modified_user_message == MockPlugin.ON_USER_CALLBACK_MSG
    assert self.plugin.user_content_seen_in_before_run_callback is not None
    assert (
        self.plugin.user_content_seen_in_before_run_callback.parts[0].text
        == MockPlugin.ON_USER_CALLBACK_MSG
    )

  @pytest.mark.asyncio
  async def test_runner_modifies_event_after_execution(self):
    """Test that the runner modifies the event after execution."""
    self.plugin.enable_event_callback = True

    events = await self.run_test()
    generated_event = events[0]
    modified_event_message = generated_event.content.parts[0].text

    assert modified_event_message == MockPlugin.ON_EVENT_CALLBACK_MSG

  @pytest.mark.asyncio
  async def test_runner_persists_event_callback_modifications(self):
    """Event callback output should be persisted, not only streamed."""
    self.plugin.enable_event_callback = True

    events = await self.run_test()
    streamed_event = events[0]

    session = await self.session_service.get_session(
        app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
    )
    persisted_event = session.events[1]

    assert streamed_event.author == "test_agent"
    assert streamed_event.invocation_id
    assert streamed_event.content.parts[0].text == (
        MockPlugin.ON_EVENT_CALLBACK_MSG
    )
    assert (
        streamed_event.custom_metadata == MockPlugin.ON_EVENT_CALLBACK_METADATA
    )

    assert persisted_event.id == streamed_event.id
    assert persisted_event.timestamp == streamed_event.timestamp
    assert persisted_event.author == streamed_event.author
    assert persisted_event.invocation_id == streamed_event.invocation_id
    assert persisted_event.content.parts[0].text == (
        MockPlugin.ON_EVENT_CALLBACK_MSG
    )
    assert (
        persisted_event.custom_metadata == MockPlugin.ON_EVENT_CALLBACK_METADATA
    )

  @pytest.mark.asyncio
  async def test_runner_close_calls_plugin_close(self):
    """Test that runner.close() calls plugin manager close."""
    # Mock the plugin manager's close method
    self.runner.plugin_manager.close = AsyncMock()

    await self.runner.close()

    self.runner.plugin_manager.close.assert_awaited_once()

  @pytest.mark.asyncio
  async def test_runner_close_does_not_cancel_toolset_cleanup(self):
    """Caller cancellation should not cancel an in-flight toolset close."""

    class SlowCloseToolset(BaseToolset):

      def __init__(self):
        super().__init__()
        self.close_started = asyncio.Event()
        self.close_finished = asyncio.Event()
        self.close_cancelled = False

      async def get_tools(self, readonly_context=None):
        del readonly_context
        return []

      async def close(self) -> None:
        self.close_started.set()
        try:
          await asyncio.sleep(0.05)
          self.close_finished.set()
        except asyncio.CancelledError:
          self.close_cancelled = True
          raise

    toolset = SlowCloseToolset()
    runner = Runner(
        app_name="test_app",
        agent=LlmAgent(
            name="test_agent", model="gemini-1.5-pro", tools=[toolset]
        ),
        session_service=self.session_service,
        artifact_service=self.artifact_service,
    )

    close_task = asyncio.create_task(runner.close())
    await toolset.close_started.wait()
    close_task.cancel()

    with pytest.raises(asyncio.CancelledError):
      await close_task

    assert close_task.cancelled() is True
    assert toolset.close_cancelled is False
    assert toolset.close_finished.is_set()

  @pytest.mark.asyncio
  async def test_runner_passes_plugin_close_timeout(self):
    """Test that runner passes plugin_close_timeout to PluginManager."""
    runner = Runner(
        app_name="test_app",
        agent=MockLlmAgent("test_agent"),
        session_service=self.session_service,
        artifact_service=self.artifact_service,
        plugins=[self.plugin],
        plugin_close_timeout=10.0,
    )
    assert runner.plugin_manager._close_timeout == 10.0

  @pytest.mark.filterwarnings(
      "ignore:The `plugins` argument is deprecated:DeprecationWarning"
  )
  def test_runner_init_raises_error_with_app_and_agent(self):
    """Test that ValueError is raised when app and agent are provided."""
    with pytest.raises(
        ValueError,
        match="Only one of app, agent, or node may be provided.",
    ):
      Runner(
          app=App(name="test_app", root_agent=self.root_agent),
          agent=self.root_agent,
          session_service=self.session_service,
          artifact_service=self.artifact_service,
      )

  @pytest.mark.filterwarnings(
      "ignore:The `plugins` argument is deprecated:DeprecationWarning"
  )
  def test_runner_init_allows_app_name_override_with_app(self):
    """Test that app_name can override app.name when both are provided."""
    app = App(name="test_app", root_agent=self.root_agent)
    runner = Runner(
        app=app,
        app_name="override_name",
        session_service=self.session_service,
        artifact_service=self.artifact_service,
    )
    assert runner.app_name == "override_name"
    assert runner.agent == self.root_agent
    assert runner.app == app

  def test_runner_init_raises_error_without_app_and_app_name(self):
    """Test ValueError is raised when app is not provided and app_name is missing."""
    with pytest.raises(
        ValueError,
        match=(
            "app_name is required when agent is provided|One of app, agent, or"
            " node must be provided"
        ),
    ):
      Runner(
          agent=self.root_agent,
          session_service=self.session_service,
          artifact_service=self.artifact_service,
      )

  def test_runner_init_raises_error_without_app_and_agent(self):
    """Test ValueError is raised when app is not provided and agent is missing."""
    with pytest.raises(
        ValueError,
        match=(
            "app_name is required when agent is provided|One of app, agent, or"
            " node must be provided"
        ),
    ):
      Runner(
          app_name="test_app",
          session_service=self.session_service,
          artifact_service=self.artifact_service,
      )


class TestRunnerCacheConfig:
  """Tests for Runner cache config extraction and handling."""

  def setup_method(self):
    """Set up test fixtures."""
    self.session_service = InMemorySessionService()
    self.artifact_service = InMemoryArtifactService()
    self.root_agent = MockLlmAgent("root_agent")

  def test_runner_extracts_cache_config_from_app(self):
    """Test that Runner extracts cache config from App."""
    cache_config = ContextCacheConfig(
        cache_intervals=15, ttl_seconds=3600, min_tokens=1024
    )

    app = App(
        name="test_app",
        root_agent=self.root_agent,
        context_cache_config=cache_config,
    )

    runner = Runner(
        app=app,
        session_service=self.session_service,
        artifact_service=self.artifact_service,
    )

    assert runner.context_cache_config == cache_config
    assert runner.context_cache_config.cache_intervals == 15
    assert runner.context_cache_config.ttl_seconds == 3600
    assert runner.context_cache_config.min_tokens == 1024

  def test_runner_with_app_without_cache_config(self):
    """Test Runner with App that has no cache config."""
    app = App(
        name="test_app", root_agent=self.root_agent, context_cache_config=None
    )

    runner = Runner(
        app=app,
        session_service=self.session_service,
        artifact_service=self.artifact_service,
    )

    assert runner.context_cache_config is None

  def test_runner_without_app_has_no_cache_config(self):
    """Test Runner created without App has no cache config."""
    runner = Runner(
        app_name="test_app",
        agent=self.root_agent,
        session_service=self.session_service,
        artifact_service=self.artifact_service,
    )

    assert runner.context_cache_config is None

  def test_runner_cache_config_passed_to_invocation_context(self):
    """Test that cache config is passed to InvocationContext."""
    cache_config = ContextCacheConfig(
        cache_intervals=20, ttl_seconds=7200, min_tokens=2048
    )

    app = App(
        name="test_app",
        root_agent=self.root_agent,
        context_cache_config=cache_config,
    )

    runner = Runner(
        app=app,
        session_service=self.session_service,
        artifact_service=self.artifact_service,
    )

    # Create a mock session
    mock_session = Session(
        id=TEST_SESSION_ID,
        app_name=TEST_APP_ID,
        user_id=TEST_USER_ID,
        events=[],
    )

    # Create invocation context using runner's method
    invocation_context = runner._new_invocation_context(mock_session)

    assert invocation_context.context_cache_config == cache_config
    assert invocation_context.context_cache_config.cache_intervals == 20

  def test_runner_validate_params_return_order(self):
    """Test that _validate_runner_params returns values in correct order."""
    cache_config = ContextCacheConfig(cache_intervals=25)

    app = App(
        name="order_test_app",
        root_agent=self.root_agent,
        context_cache_config=cache_config,
        resumability_config=ResumabilityConfig(is_resumable=True),
    )

    runner = Runner(
        app=app,
        session_service=self.session_service,
        artifact_service=self.artifact_service,
    )

    # Test the validation method directly
    app_name, agent, context_cache_config, resumability_config, plugins = (
        runner._validate_runner_params(app, None, None, None)
    )

    assert app_name == "order_test_app"
    assert agent == self.root_agent
    assert context_cache_config == cache_config
    assert context_cache_config.cache_intervals == 25
    assert resumability_config == app.resumability_config
    assert plugins == []

  def test_runner_validate_params_without_app(self):
    """Test _validate_runner_params without App returns None for cache config."""
    runner = Runner(
        app_name="test_app",
        agent=self.root_agent,
        session_service=self.session_service,
        artifact_service=self.artifact_service,
    )

    app_name, agent, context_cache_config, resumability_config, plugins = (
        runner._validate_runner_params(None, "test_app", self.root_agent, None)
    )

    assert app_name == "test_app"
    assert agent == self.root_agent
    assert context_cache_config is None
    assert resumability_config is None
    assert plugins is None

  def test_runner_app_name_and_agent_extracted_correctly(self):
    """Test that app_name and agent are correctly extracted from App."""
    cache_config = ContextCacheConfig()

    app = App(
        name="extracted_app",
        root_agent=self.root_agent,
        context_cache_config=cache_config,
    )

    runner = Runner(
        app=app,
        session_service=self.session_service,
        artifact_service=self.artifact_service,
    )

    assert runner.app_name == "extracted_app"
    assert runner.agent == self.root_agent
    assert runner.context_cache_config == cache_config

  def test_runner_realistic_cache_config_scenario(self):
    """Test realistic scenario with production-like cache config."""
    # Production cache config
    production_cache_config = ContextCacheConfig(
        cache_intervals=30,
        ttl_seconds=14400,
        min_tokens=4096,  # 4 hours
    )

    app = App(
        name="production_app",
        root_agent=self.root_agent,
        context_cache_config=production_cache_config,
    )

    runner = Runner(
        app=app,
        session_service=self.session_service,
        artifact_service=self.artifact_service,
    )

    # Verify all settings are preserved
    assert runner.context_cache_config.cache_intervals == 30
    assert runner.context_cache_config.ttl_seconds == 14400
    assert runner.context_cache_config.ttl_string == "14400s"
    assert runner.context_cache_config.min_tokens == 4096

    # Verify string representation
    expected_str = (
        "ContextCacheConfig(cache_intervals=30, ttl=14400s, min_tokens=4096, "
        "create_http_options=None)"
    )
    assert str(runner.context_cache_config) == expected_str


class TestRunnerUncachedTransferWarning:
  """Tests for the warning about agent transfer without a context cache."""

  def setup_method(self):
    """Set up test fixtures."""
    self.session_service = InMemorySessionService()
    runners._UNCACHED_TRANSFER_APPS.clear()

  def teardown_method(self):
    runners._UNCACHED_TRANSFER_APPS.clear()

  def _multi_agent(self) -> LlmAgent:
    return LlmAgent(
        name="root_agent",
        model="gemini-1.5-pro",
        sub_agents=[MockLlmAgent("sub_agent")],
    )

  def _warnings(self, caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "context_cache_config" in record.getMessage()
    ]

  def test_warns_for_multi_agent_app_without_cache_config(self, caplog):
    """Transfer is possible and no cache is configured, so warn."""
    app = App(name="multi_agent_app", root_agent=self._multi_agent())

    with caplog.at_level(logging.WARNING):
      Runner(app=app, session_service=self.session_service)

    messages = self._warnings(caplog)
    assert len(messages) == 1
    assert "multi_agent_app" in messages[0]

  def test_no_warning_when_cache_config_present(self, caplog):
    """An app that configures a context cache is not warned."""
    app = App(
        name="cached_app",
        root_agent=self._multi_agent(),
        context_cache_config=ContextCacheConfig(),
    )

    with caplog.at_level(logging.WARNING):
      Runner(app=app, session_service=self.session_service)

    assert not self._warnings(caplog)

  def test_no_warning_without_transfer_targets(self, caplog):
    """A single-agent app cannot transfer, so nothing is lost."""
    app = App(name="single_agent_app", root_agent=MockLlmAgent("root_agent"))

    with caplog.at_level(logging.WARNING):
      Runner(app=app, session_service=self.session_service)

    assert not self._warnings(caplog)

  def test_warns_only_once_per_app(self, caplog):
    """Rebuilding the runner for the same app does not warn again."""
    app = App(name="multi_agent_app", root_agent=self._multi_agent())

    with caplog.at_level(logging.WARNING):
      for _ in range(3):
        Runner(app=app, session_service=self.session_service)

    assert len(self._warnings(caplog)) == 1


class TestRunnerResolveApp:
  """Tests for Runner._resolve_app and node support."""

  def setup_method(self):
    self.session_service = InMemorySessionService()
    self.artifact_service = InMemoryArtifactService()
    self.root_agent = MockLlmAgent("root_agent")

  def test_resolve_app_with_agent_wraps_in_app(self):
    """Test that a bare agent is wrapped into an App."""
    runner = Runner(
        app_name="test_app",
        agent=self.root_agent,
        session_service=self.session_service,
        artifact_service=self.artifact_service,
    )
    assert runner.app is not None
    assert runner.app.root_agent is self.root_agent
    assert runner.app_name == "test_app"
    assert runner.agent is self.root_agent

  def test_resolve_app_with_node_wraps_in_app(self):
    """Test that a bare node is wrapped into an App."""
    from google.adk.workflow._base_node import BaseNode

    node = BaseNode(name="test_node")
    runner = Runner(
        node=node,
        session_service=self.session_service,
        artifact_service=self.artifact_service,
    )
    assert runner.app is not None
    assert runner.app.root_agent is node
    assert runner.app_name == "test_node"
    assert runner.agent is node

  def test_resolve_app_with_node_and_app_name(self):
    """Test that app_name overrides node.name."""
    from google.adk.workflow._base_node import BaseNode

    node = BaseNode(name="node_name")
    runner = Runner(
        app_name="custom_name",
        node=node,
        session_service=self.session_service,
        artifact_service=self.artifact_service,
    )
    assert runner.app_name == "custom_name"

  def test_resolve_app_rejects_app_and_agent(self):
    """Test that providing both app and agent raises."""
    app = App(name="test_app", root_agent=self.root_agent)
    with pytest.raises(
        ValueError,
        match=(
            r"Only one of app, agent, or node may be provided, but got:"
            r" app=App, agent=MockLlmAgent\. Pass exactly one to Runner\(\)\."
        ),
    ):
      Runner(
          app=app,
          agent=self.root_agent,
          session_service=self.session_service,
      )

  def test_resolve_app_rejects_app_and_node(self):
    """Test that providing both app and node raises."""
    from google.adk.workflow._base_node import BaseNode

    app = App(name="test_app", root_agent=self.root_agent)
    node = BaseNode(name="test_node")
    with pytest.raises(
        ValueError,
        match=(
            r"Only one of app, agent, or node may be provided, but got:"
            r" app=App, node=BaseNode\. Pass exactly one to Runner\(\)\."
        ),
    ):
      Runner(
          app=app,
          node=node,
          session_service=self.session_service,
      )

  def test_resolve_app_rejects_agent_and_node(self):
    """Test that providing both agent and node raises."""
    from google.adk.workflow._base_node import BaseNode

    node = BaseNode(name="test_node")
    with pytest.raises(
        ValueError,
        match=(
            r"Only one of app, agent, or node may be provided, but got:"
            r" agent=MockLlmAgent, node=BaseNode\. Pass exactly one to"
            r" Runner\(\)\."
        ),
    ):
      Runner(
          app_name="test_app",
          agent=self.root_agent,
          node=node,
          session_service=self.session_service,
      )

  def test_resolve_app_rejects_none(self):
    """Test that providing no app, agent, or node raises."""
    with pytest.raises(
        ValueError,
        match=(
            r"One of app, agent, or node must be provided\. Got none\. Pass"
            r" exactly one to Runner\(\)\."
        ),
    ):
      Runner(
          app_name="test_app",
          session_service=self.session_service,
      )

  def test_resolve_app_extracts_node_from_app(self):
    """Test that Runner extracts node from App into agent field."""
    from google.adk.workflow._base_node import BaseNode

    node = BaseNode(name="test_node")
    app = App(name="test_app", root_agent=node)
    runner = Runner(
        app=app,
        session_service=self.session_service,
        artifact_service=self.artifact_service,
    )
    assert runner.agent is node
    assert runner.app_name == "test_app"
    assert runner.context_cache_config is None
    assert runner.resumability_config is None


class TestRunnerShouldAppendEvent:
  """Tests for Runner._should_append_event method."""

  def setup_method(self):
    """Set up test fixtures."""
    self.session_service = InMemorySessionService()
    self.artifact_service = InMemoryArtifactService()
    self.root_agent = MockLlmAgent("root_agent")
    self.runner = Runner(
        app_name="test_app",
        agent=self.root_agent,
        session_service=self.session_service,
        artifact_service=self.artifact_service,
    )

  def test_should_append_event_finished_input_transcription(self):
    event = Event(
        invocation_id="inv1",
        author="user",
        input_transcription=types.Transcription(text="hello", finished=True),
    )
    assert self.runner._should_append_event(event, is_live_call=True) is True

  def test_should_append_event_unfinished_input_transcription(self):
    event = Event(
        invocation_id="inv1",
        author="user",
        input_transcription=types.Transcription(text="hello", finished=False),
    )
    assert self.runner._should_append_event(event, is_live_call=True) is True

  def test_should_append_event_finished_output_transcription(self):
    event = Event(
        invocation_id="inv1",
        author="model",
        output_transcription=types.Transcription(text="world", finished=True),
    )
    assert self.runner._should_append_event(event, is_live_call=True) is True

  def test_should_append_event_unfinished_output_transcription(self):
    event = Event(
        invocation_id="inv1",
        author="model",
        output_transcription=types.Transcription(text="world", finished=False),
    )
    assert self.runner._should_append_event(event, is_live_call=True) is True

  def test_should_not_append_event_live_model_audio(self):
    event = Event(
        invocation_id="inv1",
        author="model",
        content=types.Content(
            parts=[
                types.Part(
                    inline_data=types.Blob(data=b"123", mime_type="audio/pcm")
                )
            ]
        ),
    )
    assert self.runner._should_append_event(event, is_live_call=True) is False

  def test_should_append_event_non_live_model_audio(self):
    event = Event(
        invocation_id="inv1",
        author="model",
        content=types.Content(
            parts=[
                types.Part(
                    inline_data=types.Blob(data=b"123", mime_type="audio/pcm")
                )
            ]
        ),
    )
    assert self.runner._should_append_event(event, is_live_call=False) is True

  def test_should_append_event_other_event(self):
    event = Event(
        invocation_id="inv1",
        author="model",
        content=types.Content(parts=[types.Part(text="text")]),
    )
    assert self.runner._should_append_event(event, is_live_call=True) is True

  def test_should_not_append_event_live_model_video(self):
    event = Event(
        invocation_id="inv1",
        author="model",
        content=types.Content(
            parts=[
                types.Part(
                    inline_data=types.Blob(data=b"123", mime_type="video/mp4")
                )
            ]
        ),
    )
    assert self.runner._should_append_event(event, is_live_call=True) is False

  def test_should_append_event_non_live_model_video(self):
    event = Event(
        invocation_id="inv1",
        author="model",
        content=types.Content(
            parts=[
                types.Part(
                    inline_data=types.Blob(data=b"123", mime_type="video/mp4")
                )
            ]
        ),
    )
    assert self.runner._should_append_event(event, is_live_call=False) is True


@pytest.fixture
def user_agent_module(tmp_path, monkeypatch):
  """Fixture that creates a temporary user agent module for testing.

  Yields a callable that creates an agent module with the given name and
  returns the loaded agent.
  """
  created_modules = []
  original_path = None

  def _create_agent(agent_dir_name: str):
    nonlocal original_path
    agent_dir = tmp_path / "agents" / agent_dir_name
    agent_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "agents" / "__init__.py").write_text("", encoding="utf-8")
    (agent_dir / "__init__.py").write_text("", encoding="utf-8")

    agent_source = f"""\
from google.adk.agents.llm_agent import LlmAgent

class MyAgent(LlmAgent):
    pass

root_agent = MyAgent(name="{agent_dir_name}", model="gemini-2.5-flash")
"""
    (agent_dir / "agent.py").write_text(agent_source, encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    if original_path is None:
      original_path = str(tmp_path)
      sys.path.insert(0, original_path)

    module_name = f"agents.{agent_dir_name}.agent"
    module = importlib.import_module(module_name)
    created_modules.append(module_name)
    return module.root_agent

  yield _create_agent

  # Cleanup
  if original_path and original_path in sys.path:
    sys.path.remove(original_path)
  for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("agents"):
      del sys.modules[mod_name]


class TestRunnerInferAgentOrigin:
  """Tests for Runner._infer_agent_origin method."""

  def setup_method(self):
    """Set up test fixtures."""
    self.session_service = InMemorySessionService()
    self.artifact_service = InMemoryArtifactService()

  def test_infer_agent_origin_uses_adk_metadata_when_available(self):
    """Test that _infer_agent_origin uses _adk_origin_* metadata when set."""
    agent = MockLlmAgent("test_agent")
    # Simulate metadata set by AgentLoader
    agent._adk_origin_app_name = "my_app"
    agent._adk_origin_path = Path("/workspace/agents/my_app")

    runner = Runner(
        app_name="my_app",
        agent=agent,
        session_service=self.session_service,
        artifact_service=self.artifact_service,
    )

    origin_name, origin_path = runner._infer_agent_origin(agent)
    assert origin_name == "my_app"
    assert origin_path == Path("/workspace/agents/my_app")

  def test_infer_agent_origin_no_false_positive_for_direct_llm_agent(self):
    """Test that using LlmAgent directly doesn't trigger mismatch warning.

    Regression test: users who instantiate LlmAgent directly and run from a
    directory that is a parent of the ADK installation were getting false
    positive 'App name mismatch' warnings.

    This also verifies that _infer_agent_origin returns None for ADK internal
    modules (google.adk.*).
    """
    agent = LlmAgent(
        name="my_custom_agent",
        model="gemini-2.5-flash",
    )

    runner = Runner(
        app_name="my_custom_agent",
        agent=agent,
        session_service=self.session_service,
        artifact_service=self.artifact_service,
    )

    # Should return None for ADK internal modules
    origin_name, _ = runner._infer_agent_origin(agent)
    assert origin_name is None
    # No mismatch warning should be generated
    assert runner._app_name_alignment_hint is None

  def test_infer_agent_origin_with_subclassed_agent_in_user_code(
      self, user_agent_module
  ):
    """Test that subclassed agents in user code still trigger origin inference."""
    agent = user_agent_module("my_agent")

    runner = Runner(
        app_name="my_agent",
        agent=agent,
        session_service=self.session_service,
        artifact_service=self.artifact_service,
    )

    # Should infer origin correctly from user's code
    origin_name, origin_path = runner._infer_agent_origin(agent)
    assert origin_name == "my_agent"
    assert runner._app_name_alignment_hint is None

  def test_infer_agent_origin_detects_mismatch_for_user_agent(
      self, user_agent_module
  ):
    """Test that mismatched app_name is detected for user-defined agents."""
    agent = user_agent_module("actual_name")

    runner = Runner(
        app_name="wrong_name",  # Intentionally wrong
        agent=agent,
        session_service=self.session_service,
        artifact_service=self.artifact_service,
    )

    # Should detect the mismatch
    assert runner._app_name_alignment_hint is not None
    assert "wrong_name" in runner._app_name_alignment_hint
    assert "actual_name" in runner._app_name_alignment_hint


@pytest.mark.asyncio
async def test_run_async_passes_get_session_config():
  """run_async should forward RunConfig.get_session_config to get_session."""
  from google.adk.sessions.base_session_service import GetSessionConfig

  session_service = InMemorySessionService()

  # Pre-create a session with multiple events.
  session = await session_service.create_session(
      app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )
  for i in range(10):
    await session_service.append_event(
        session=session,
        event=Event(
            invocation_id=f"inv_{i}",
            author="user",
            content=types.Content(
                role="user", parts=[types.Part(text=f"message {i}")]
            ),
        ),
    )

  runner = Runner(
      app_name=TEST_APP_ID,
      agent=MockAgent("test_agent"),
      session_service=session_service,
      artifact_service=InMemoryArtifactService(),
  )

  # Run with num_recent_events=3 to only load recent events.
  config = RunConfig(
      get_session_config=GetSessionConfig(num_recent_events=3),
  )

  events = []
  async for event in runner.run_async(
      user_id=TEST_USER_ID,
      session_id=TEST_SESSION_ID,
      new_message=types.Content(role="user", parts=[types.Part(text="hello")]),
      run_config=config,
  ):
    events.append(event)

  # Agent should still produce output (session was found).
  assert len(events) >= 1
  assert events[0].author == "test_agent"


@pytest.mark.asyncio
async def test_run_async_teardown_on_aclose():
  """Closing run_async generator using aclose() should abort and cancel the running agent task."""
  import asyncio

  session_service = InMemorySessionService()
  artifact_service = InMemoryArtifactService()

  was_cancelled = {"value": False}

  class CancellingAgent(BaseAgent):

    def __init__(self, name: str):
      super().__init__(name=name, sub_agents=[])

    async def _run_async_impl(
        self, invocation_context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
      try:
        yield Event(
            invocation_id=invocation_context.invocation_id,
            author=self.name,
            content=types.Content(
                role="model", parts=[types.Part(text="First response")]
            ),
        )
        # Block simulating slow ongoing task
        await asyncio.sleep(5.0)
        yield Event(
            invocation_id=invocation_context.invocation_id,
            author=self.name,
            content=types.Content(
                role="model", parts=[types.Part(text="Second response")]
            ),
        )
      except (asyncio.CancelledError, GeneratorExit):
        was_cancelled["value"] = True
        raise

  runner = Runner(
      app_name=TEST_APP_ID,
      agent=CancellingAgent("cancel_agent"),
      session_service=session_service,
      artifact_service=artifact_service,
      auto_create_session=True,
  )

  # Given a run session
  agen = runner.run_async(
      user_id=TEST_USER_ID,
      session_id=TEST_SESSION_ID,
      new_message=types.Content(role="user", parts=[types.Part(text="hello")]),
  )

  # When the client reads the first event and then calls aclose()
  event = await agen.__anext__()
  assert event.content.parts[0].text == "First response"

  await agen.aclose()

  # Then the running agent was immediately aborted and cancelled
  assert was_cancelled["value"] is True


@pytest.mark.asyncio
async def test_run_live_passes_get_session_config():
  """run_live should forward RunConfig.get_session_config to get_session."""
  from google.adk.agents.live_request_queue import LiveRequestQueue
  from google.adk.sessions.base_session_service import GetSessionConfig

  session_service = InMemorySessionService()

  # Pre-create session.
  await session_service.create_session(
      app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )

  runner = Runner(
      app_name=TEST_APP_ID,
      agent=MockLiveAgent("live_agent"),
      session_service=session_service,
      artifact_service=InMemoryArtifactService(),
  )

  config = RunConfig(
      get_session_config=GetSessionConfig(num_recent_events=5),
  )

  live_queue = LiveRequestQueue()
  agen = runner.run_live(
      user_id=TEST_USER_ID,
      session_id=TEST_SESSION_ID,
      live_request_queue=live_queue,
      run_config=config,
  )

  event = await agen.__anext__()
  await agen.aclose()

  assert event.author == "live_agent"
  assert event.content.parts[0].text == "live hello"


@pytest.mark.asyncio
async def test_rewind_async_passes_get_session_config():
  """rewind_async should forward RunConfig.get_session_config to get_session."""
  from google.adk.sessions.base_session_service import GetSessionConfig

  session_service = InMemorySessionService()

  runner = Runner(
      app_name=TEST_APP_ID,
      agent=MockAgent("test_agent"),
      session_service=session_service,
      artifact_service=InMemoryArtifactService(),
      auto_create_session=True,
  )

  config = RunConfig(
      get_session_config=GetSessionConfig(num_recent_events=5),
  )

  # rewind_async on a fresh session will raise because the invocation_id
  # doesn't exist, but it demonstrates that the config path works.
  with pytest.raises(ValueError, match=r"Invocation ID not found"):
    await runner.rewind_async(
        user_id=TEST_USER_ID,
        session_id="new_session",
        rewind_before_invocation_id="inv_missing",
        run_config=config,
    )


@pytest.mark.asyncio
async def test_run_debug_passes_get_session_config():
  """run_debug should forward RunConfig.get_session_config to get_session."""
  from google.adk.sessions.base_session_service import GetSessionConfig

  session_service = InMemorySessionService()

  runner = Runner(
      app_name=TEST_APP_ID,
      agent=MockAgent("test_agent"),
      session_service=session_service,
      artifact_service=InMemoryArtifactService(),
  )

  config = RunConfig(
      get_session_config=GetSessionConfig(num_recent_events=5),
  )

  events = await runner.run_debug(
      "hello",
      run_config=config,
      quiet=True,
  )

  assert len(events) >= 1
  assert events[0].author == "test_agent"


@pytest.mark.asyncio
async def test_get_session_config_limits_events():
  """Verify that num_recent_events actually limits loaded events."""
  from google.adk.sessions.base_session_service import GetSessionConfig

  session_service = InMemorySessionService()

  # Create session and add events.
  session = await session_service.create_session(
      app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )
  for i in range(10):
    await session_service.append_event(
        session=session,
        event=Event(
            invocation_id=f"inv_{i}",
            author="user",
            content=types.Content(
                role="user", parts=[types.Part(text=f"message {i}")]
            ),
        ),
    )

  # Without config: should load all events.
  full_session = await session_service.get_session(
      app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )
  assert len(full_session.events) == 10

  # With config: should limit events.
  limited_session = await session_service.get_session(
      app_name=TEST_APP_ID,
      user_id=TEST_USER_ID,
      session_id=TEST_SESSION_ID,
      config=GetSessionConfig(num_recent_events=3),
  )
  assert len(limited_session.events) == 3


@pytest.mark.asyncio
async def test_run_async_rejects_user_function_call():
  """Verify that runner rejects user-authored messages with function calls."""
  session_service = InMemorySessionService()
  runner = Runner(
      app_name=TEST_APP_ID,
      agent=MockAgent("test_agent"),
      session_service=session_service,
      artifact_service=InMemoryArtifactService(),
      auto_create_session=True,
  )

  malicious_message = types.Content(
      role="user",
      parts=[
          types.Part(
              function_call=types.FunctionCall(
                  name="some_tool",
                  args={"key": "value"},
              )
          )
      ],
  )

  agen = runner.run_async(
      user_id=TEST_USER_ID,
      session_id=TEST_SESSION_ID,
      new_message=malicious_message,
  )

  with pytest.raises(ValueError, match="cannot contain function calls"):
    async with aclosing(agen) as a:
      async for _ in a:
        pass


def test_runner_agent_is_a_class_attribute():
  """``agent`` must stay in ``dir(Runner)`` for callers that mock a Runner."""
  assert "agent" in dir(Runner)
  assert Runner.agent is None
  assert create_autospec(Runner).agent is not None


@pytest.mark.asyncio
async def test_runner_delegation_finds_active_task_scope_on_non_terminal_error():
  """find_active_task_scope ignores non-terminal errors; remains on active task."""
  session_service = InMemorySessionService()
  agent = LlmAgent(name="task_agent", mode="task")
  runner = Runner(
      app_name=TEST_APP_ID, agent=agent, session_service=session_service
  )
  await session_service.create_session(
      app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )

  # Simulate non-terminal error (validation failure)
  events = [
      Event(
          author="task_agent",
          invocation_id="inv-1",
          isolation_scope="scope-1",
          content=types.Content(
              parts=[
                  types.Part.from_function_response(
                      name=FINISH_TASK_TOOL_NAME,
                      response={"result": "Validation failed; retry"},
                  )
              ]
          ),
      )
  ]
  session = await session_service.get_session(
      app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )
  session.events.extend(events)
  assert runners._find_active_task_scope(session) == ("scope-1", "inv-1")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result", [FINISH_TASK_SUCCESS_RESULT, FINISH_TASK_ERROR_RESULT]
)
async def test_runner_delegation_closes_active_task_scope_on_terminal_results(
    result,
):
  """find_active_task_scope returns None if the task scope has finished with a terminal result."""
  session_service = InMemorySessionService()
  agent = LlmAgent(name="task_agent", mode="task")
  runner = Runner(
      app_name=TEST_APP_ID, agent=agent, session_service=session_service
  )
  await session_service.create_session(
      app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )

  # Simulate terminal result
  events = [
      Event(
          author="task_agent",
          invocation_id="inv-1",
          isolation_scope="scope-1",
          content=types.Content(
              parts=[
                  types.Part.from_function_response(
                      name=FINISH_TASK_TOOL_NAME,
                      response={"result": result},
                  )
              ]
          ),
      )
  ]
  session = await session_service.get_session(
      app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )
  session.events.extend(events)
  assert runners._find_active_task_scope(session) is None


@pytest.mark.asyncio
async def test_runner_picks_coordinator_when_has_remote_a2a_task_subagent():
  """Runner runs coordinator, not sub-agent, when RemoteA2aAgent is in task mode."""
  session_service = InMemorySessionService()

  sub_agent = RemoteA2aAgent(
      name="remote_task_agent",
      agent_card="https://example.com/rpc",
      mode="task",
  )

  # Coordinator LlmAgent in chat mode
  coordinator = LlmAgent(
      name="coordinator", mode="chat", sub_agents=[sub_agent]
  )
  sub_agent.parent_agent = coordinator

  runner = Runner(
      app_name=TEST_APP_ID, agent=coordinator, session_service=session_service
  )
  await session_service.create_session(
      app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )

  # Simulate some events so _find_agent_to_run would be called on resume.
  events = [
      Event(
          author="remote_task_agent",
          invocation_id="inv-1",
          isolation_scope="scope-1",
          content=types.Content(parts=[types.Part(text="task progress")]),
      )
  ]
  session = await session_service.get_session(
      app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )
  session.events.extend(events)

  # Mock _run_node_async to just yield a dummy event and return
  async def mock_run_node_async(*args, **kwargs):
    yield Event(
        author="system",
        content=types.Content(parts=[types.Part(text="dummy")]),
    )

  with patch.object(
      runner, "_run_node_async", side_effect=mock_run_node_async
  ) as mock_run_node:
    # Run with new message (resume-like)
    async for _ in runner.run_async(
        user_id=TEST_USER_ID,
        session_id=TEST_SESSION_ID,
        new_message=types.Content(parts=[types.Part(text="user reply")]),
    ):
      pass

    # Verify that _run_node_async was called with the coordinator (self.agent)
    # not the sub_agent.
    assert mock_run_node.call_count == 1
    called_node = mock_run_node.call_args[1].get("node")
    assert called_node == coordinator


@pytest.mark.asyncio
async def test_run_async_does_not_leak_context_base_node():
  """Caller OpenTelemetry context is preserved during run_async iteration for BaseNode."""
  from typing import Any

  from google.adk.agents.context import Context
  from google.adk.workflow._base_node import BaseNode
  from opentelemetry import context as otel_context

  class _TestEchoNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield "echo"

  session_service = InMemorySessionService()
  runner = Runner(
      app_name=TEST_APP_ID,
      node=_TestEchoNode(name="test_node"),
      session_service=session_service,
      artifact_service=InMemoryArtifactService(),
      auto_create_session=True,
  )

  test_key = otel_context.create_key("test_key_node")
  token = otel_context.attach(
      otel_context.set_value(test_key, "caller_val_node")
  )
  caller_ctx = otel_context.get_current()
  try:
    events = []
    async with aclosing(
        runner.run_async(
            user_id=TEST_USER_ID,
            session_id=TEST_SESSION_ID,
            new_message=types.Content(
                role="user", parts=[types.Part(text="hello")]
            ),
        )
    ) as agen:
      async for event in agen:
        assert otel_context.get_current() == caller_ctx
        assert otel_context.get_value(test_key) == "caller_val_node"
        events.append(event)
    assert events
    assert otel_context.get_current() == caller_ctx
  finally:
    otel_context.detach(token)


@pytest.mark.asyncio
async def test_run_async_does_not_leak_context_base_agent():
  """Caller OpenTelemetry context is preserved during run_async iteration for BaseAgent."""
  from opentelemetry import context as otel_context

  session_service = InMemorySessionService()
  runner = Runner(
      app_name=TEST_APP_ID,
      agent=MockAgent("test_agent"),
      session_service=session_service,
      artifact_service=InMemoryArtifactService(),
      auto_create_session=True,
  )

  test_key = otel_context.create_key("test_key_agent")
  token = otel_context.attach(
      otel_context.set_value(test_key, "caller_val_agent")
  )
  caller_ctx = otel_context.get_current()
  try:
    events = []
    async with aclosing(
        runner.run_async(
            user_id=TEST_USER_ID,
            session_id=TEST_SESSION_ID,
            new_message=types.Content(
                role="user", parts=[types.Part(text="hello")]
            ),
        )
    ) as agen:
      async for event in agen:
        assert otel_context.get_current() == caller_ctx
        assert otel_context.get_value(test_key) == "caller_val_agent"
        events.append(event)
    assert events
    assert otel_context.get_current() == caller_ctx
  finally:
    otel_context.detach(token)


@pytest.mark.asyncio
async def test_run_node_async_does_not_leak_context():
  """Caller OpenTelemetry context is preserved during _run_node_async iteration."""
  from typing import Any

  from google.adk.agents.context import Context
  from google.adk.workflow._base_node import BaseNode
  from opentelemetry import context as otel_context

  class _TestEchoNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield "echo"

  session_service = InMemorySessionService()
  runner = Runner(
      app_name=TEST_APP_ID,
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
        runner._run_node_async(
            user_id=TEST_USER_ID,
            session_id=TEST_SESSION_ID,
            new_message=types.Content(
                role="user", parts=[types.Part(text="hello")]
            ),
            yield_user_message=True,
        )
    ) as agen:
      async for event in agen:
        assert otel_context.get_current() == caller_ctx
        assert otel_context.get_value(test_key) == "caller_val_run_node_async"
        events.append(event)
    assert events
    assert otel_context.get_current() == caller_ctx
  finally:
    otel_context.detach(token)


@pytest.mark.asyncio
async def test_run_async_does_not_leak_context_llm_agent():
  """Caller OpenTelemetry context is preserved during run_async iteration for LlmAgent."""
  from opentelemetry import context as otel_context

  session_service = InMemorySessionService()
  runner = Runner(
      app_name=TEST_APP_ID,
      agent=MockLlmAgent("test_llm_agent"),
      session_service=session_service,
      artifact_service=InMemoryArtifactService(),
      auto_create_session=True,
  )

  test_key = otel_context.create_key("test_key_llm")
  token = otel_context.attach(
      otel_context.set_value(test_key, "caller_val_llm")
  )
  caller_ctx = otel_context.get_current()
  try:
    events = []
    async with aclosing(
        runner.run_async(
            user_id=TEST_USER_ID,
            session_id=TEST_SESSION_ID,
            new_message=types.Content(
                role="user", parts=[types.Part(text="hello")]
            ),
        )
    ) as agen:
      async for event in agen:
        assert otel_context.get_current() == caller_ctx
        assert otel_context.get_value(test_key) == "caller_val_llm"
        events.append(event)
    assert events
    assert otel_context.get_current() == caller_ctx
  finally:
    otel_context.detach(token)


@pytest.mark.asyncio
async def test_run_live_does_not_leak_context():
  """Caller OpenTelemetry context is preserved during run_live iteration."""
  from google.adk.agents.live_request_queue import LiveRequestQueue
  from opentelemetry import context as otel_context

  session_service = InMemorySessionService()
  runner = Runner(
      app_name=TEST_APP_ID,
      agent=MockLiveAgent("test_live_agent"),
      session_service=session_service,
      artifact_service=InMemoryArtifactService(),
      auto_create_session=True,
  )

  live_queue = LiveRequestQueue()
  test_key = otel_context.create_key("test_key_live")
  token = otel_context.attach(
      otel_context.set_value(test_key, "caller_val_live")
  )
  caller_ctx = otel_context.get_current()
  try:
    events = []
    async with aclosing(
        runner.run_live(
            user_id=TEST_USER_ID,
            session_id=TEST_SESSION_ID,
            live_request_queue=live_queue,
        )
    ) as agen:
      async for event in agen:
        assert otel_context.get_current() == caller_ctx
        assert otel_context.get_value(test_key) == "caller_val_live"
        events.append(event)
    assert events
    assert otel_context.get_current() == caller_ctx
  finally:
    otel_context.detach(token)


@pytest.mark.asyncio
async def test_base_agent_run_async_does_not_leak_context():
  """Caller OpenTelemetry context is preserved during BaseAgent.run_async iteration."""
  from google.adk.plugins.plugin_manager import PluginManager
  from opentelemetry import context as otel_context

  agent = MockAgent("test_agent")
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )
  inv_ctx = InvocationContext(
      session=session,
      session_service=session_service,
      plugin_manager=PluginManager(),
      agent=agent,
      invocation_id="inv_test",
  )

  test_key = otel_context.create_key("test_key_base_agent_run_async")
  token = otel_context.attach(
      otel_context.set_value(test_key, "caller_val_base_agent_run_async")
  )
  caller_ctx = otel_context.get_current()
  try:
    events = []
    async with aclosing(agent.run_async(parent_context=inv_ctx)) as agen:
      async for event in agen:
        assert otel_context.get_current() == caller_ctx
        assert (
            otel_context.get_value(test_key)
            == "caller_val_base_agent_run_async"
        )
        events.append(event)
    assert events
    assert otel_context.get_current() == caller_ctx
  finally:
    otel_context.detach(token)


@pytest.mark.asyncio
async def test_base_agent_run_live_does_not_leak_context():
  """Caller OpenTelemetry context is preserved during BaseAgent.run_live iteration."""
  from google.adk.plugins.plugin_manager import PluginManager
  from opentelemetry import context as otel_context

  agent = MockLiveAgent("test_live_agent")
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name=TEST_APP_ID, user_id=TEST_USER_ID, session_id=TEST_SESSION_ID
  )
  inv_ctx = InvocationContext(
      session=session,
      session_service=session_service,
      plugin_manager=PluginManager(),
      agent=agent,
      invocation_id="inv_test_live",
  )

  test_key = otel_context.create_key("test_key_base_agent_run_live")
  token = otel_context.attach(
      otel_context.set_value(test_key, "caller_val_base_agent_run_live")
  )
  caller_ctx = otel_context.get_current()
  try:
    events = []
    async with aclosing(agent.run_live(parent_context=inv_ctx)) as agen:
      async for event in agen:
        assert otel_context.get_current() == caller_ctx
        assert (
            otel_context.get_value(test_key) == "caller_val_base_agent_run_live"
        )
        events.append(event)
    assert events
    assert otel_context.get_current() == caller_ctx
  finally:
    otel_context.detach(token)


@pytest.mark.asyncio
async def test_setup_context_for_new_invocation_restores_branch_for_subagent():
  """Tests that _setup_context_for_new_invocation restores ic.branch for non-root agent."""
  session_service = InMemorySessionService()
  root_agent = MockLlmAgent("coordinator")
  sub_agent = MockLlmAgent("worker", parent_agent=root_agent)
  root_agent.sub_agents = [sub_agent]
  app = App(name="test_app", root_agent=root_agent)
  runner = Runner(app=app, session_service=session_service)

  session = await session_service.create_session(
      app_name="test_app", user_id="user_1", session_id="session_1"
  )
  # A frame event (function call) authored by the sub-agent on its branch.
  event = Event(
      invocation_id="inv_1",
      author="worker",
      branch="worker@1",
      content=types.Content(
          parts=[
              types.Part(
                  function_call=types.FunctionCall(name="t", id="fc-1", args={})
              )
          ]
      ),
  )
  await session_service.append_event(session, event)

  ic = await runner._setup_context_for_new_invocation(
      session=session,
      new_message=types.Content(parts=[types.Part.from_text(text="new msg")]),
      run_config=RunConfig(),
      state_delta=None,
      invocation_id="inv_2",
  )
  assert ic.agent == sub_agent
  assert ic.branch == "worker@1"


@pytest.mark.asyncio
async def test_setup_context_restores_branch_for_resumed_subagent():
  """Tests that _setup_context_for_resumed_invocation restores ic.branch when resuming a subagent."""
  session_service = InMemorySessionService()
  root_agent = MockLlmAgent("coordinator")
  sub_agent = MockLlmAgent("worker", parent_agent=root_agent)
  root_agent.sub_agents = [sub_agent]
  app = App(
      name="test_app",
      root_agent=root_agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = Runner(app=app, session_service=session_service)

  session = await session_service.create_session(
      app_name="test_app", user_id="user_1", session_id="session_1"
  )
  event1 = Event(
      invocation_id="inv_1",
      author="user",
      content=types.Content(parts=[types.Part.from_text(text="start")]),
  )
  event2 = Event(
      invocation_id="inv_1",
      author="worker",
      branch="worker@1",
      content=types.Content(
          parts=[
              types.Part(
                  function_call=types.FunctionCall(name="t", id="fc-1", args={})
              )
          ]
      ),
  )
  await session_service.append_event(session, event1)
  await session_service.append_event(session, event2)

  ic = await runner._setup_context_for_resumed_invocation(
      session=session,
      new_message=types.Content(parts=[types.Part.from_text(text="continue")]),
      invocation_id="inv_1",
      run_config=RunConfig(),
      state_delta=None,
  )
  assert ic.agent == sub_agent
  assert ic.branch == "worker@1"


@pytest.mark.asyncio
async def test_run_node_restores_branch_on_function_response_resume():
  """Resuming a node with a function response restores its historical branch.

  The resume message is a function response whose originating call was authored
  on the node's branch, exercising both the history restore and the user-event
  branch assignment.
  """
  session_service = InMemorySessionService()
  root_agent = MockLlmAgent("coordinator")
  node_agent = MockLlmAgent("node_agent", parent_agent=root_agent)
  root_agent.sub_agents = [node_agent]
  app = App(name="test_app", root_agent=root_agent)
  runner = Runner(app=app, session_service=session_service)

  session = await session_service.create_session(
      app_name="test_app", user_id="user_1", session_id="session_1"
  )
  # The node's own frame event: a function call on its branch.
  event = Event(
      invocation_id="inv_1",
      author="node_agent",
      branch="node_agent@1",
      content=types.Content(
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      name="get_input", id="fc-1", args={}
                  )
              )
          ]
      ),
  )
  await session_service.append_event(session, event)

  resume_msg = types.Content(
      role="user",
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name="get_input", id="fc-1", response={}
              )
          )
      ],
  )

  events = []
  async for ev in runner._run_node_async(
      user_id="user_1",
      session_id="session_1",
      new_message=resume_msg,
      node=node_agent,
  ):
    events.append(ev)

  model_events = [e for e in events if e.author == "node_agent"]
  assert model_events
  assert model_events[0].branch == "node_agent@1"


@pytest.mark.asyncio
async def test_resumed_invocation_ignores_branch_from_other_invocation():
  """Invocation-scoping: a resumed sub-agent does not inherit a stale branch.

  The only branched ``worker`` event lives in a *different* invocation
  (``inv_0``). Scoping the branch scan to the resumed invocation (``inv_1``)
  must prevent that stale branch from being restored.
  """
  session_service = InMemorySessionService()
  root_agent = MockLlmAgent("coordinator")
  sub_agent = MockLlmAgent("worker", parent_agent=root_agent)
  root_agent.sub_agents = [sub_agent]
  app = App(
      name="test_app",
      root_agent=root_agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = Runner(app=app, session_service=session_service)

  session = await session_service.create_session(
      app_name="test_app", user_id="user_1", session_id="session_1"
  )
  # Stale frame event authored by "worker" in a previous invocation.
  stale = Event(
      invocation_id="inv_0",
      author="worker",
      branch="worker@stale",
      content=types.Content(
          parts=[
              types.Part(
                  function_call=types.FunctionCall(name="t", id="fc-0", args={})
              )
          ]
      ),
  )
  # Current invocation to resume: only a user message, no worker branch yet.
  user_evt = Event(
      invocation_id="inv_1",
      author="user",
      content=types.Content(parts=[types.Part.from_text(text="start")]),
  )
  await session_service.append_event(session, stale)
  await session_service.append_event(session, user_evt)

  ic = await runner._setup_context_for_resumed_invocation(
      session=session,
      new_message=types.Content(parts=[types.Part.from_text(text="continue")]),
      invocation_id="inv_1",
      run_config=RunConfig(),
      state_delta=None,
  )
  assert ic.agent == sub_agent
  # Must NOT have inherited the stale branch from inv_0.
  assert ic.branch != "worker@stale"
  assert ic.branch is None


@pytest.mark.asyncio
async def test_restore_prefers_agent_frame_over_tool_message_branch():
  """A tool message must not win over the agent's own frame branch.

  ``functions.py`` authors a tool's user-facing message under the agent's name
  and ``base_agent.py`` stamps it with the agent's node path, yet it carries the
  tool's branch. The restore must use the agent's own frame event instead.
  """
  from google.adk.events.event import NodeInfo

  session_service = InMemorySessionService()
  root_agent = MockLlmAgent("coordinator")
  sub_agent = MockLlmAgent("worker", parent_agent=root_agent)
  root_agent.sub_agents = [sub_agent]
  app = App(
      name="test_app",
      root_agent=root_agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = Runner(app=app, session_service=session_service)

  session = await session_service.create_session(
      app_name="test_app", user_id="user_1", session_id="session_1"
  )
  user_evt = Event(
      invocation_id="inv_1",
      author="user",
      content=types.Content(parts=[types.Part.from_text(text="start")]),
  )
  # The agent's own frame event, on the agent's branch.
  worker_frame = Event(
      invocation_id="inv_1",
      author="worker",
      branch="worker@1",
      content=types.Content(
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      name="do", id="fc-1", args={}
                  )
              )
          ]
      ),
  )
  # A tool message: agent name + agent node path, but the TOOL's branch.
  # Appended last so it is the most recent match in the reverse scan.
  tool_msg = Event(
      invocation_id="inv_1",
      author="worker",
      branch="do@fc-1",
      node_info=NodeInfo(path="coordinator/worker"),
      content=types.Content(parts=[types.Part.from_text(text="tool output")]),
  )
  await session_service.append_event(session, user_evt)
  await session_service.append_event(session, worker_frame)
  await session_service.append_event(session, tool_msg)

  ic = await runner._setup_context_for_resumed_invocation(
      session=session,
      new_message=types.Content(parts=[types.Part.from_text(text="continue")]),
      invocation_id="inv_1",
      run_config=RunConfig(),
      state_delta=None,
  )
  assert ic.agent == sub_agent
  # The tool's branch must not have won.
  assert ic.branch == "worker@1"


@pytest.mark.asyncio
async def test_restore_branch_for_non_resumable_subagent_text_turn():
  """A non-resumable sub-agent that ended on a text turn keeps its branch.

  Agent-state checkpoints are only emitted when the app is resumable, so a
  non-resumable sub-agent may leave nothing behind but a plain text event. That
  event still carries the agent's branch and must be eligible for restore.
  """
  session_service = InMemorySessionService()
  root_agent = MockLlmAgent("coordinator")
  sub_agent = MockLlmAgent("worker", parent_agent=root_agent)
  root_agent.sub_agents = [sub_agent]
  # Not resumable: no agent_state / end_of_agent events are ever emitted.
  app = App(name="test_app", root_agent=root_agent)
  runner = Runner(app=app, session_service=session_service)

  session = await session_service.create_session(
      app_name="test_app", user_id="user_1", session_id="session_1"
  )
  # The sub-agent's only trace is a plain text turn on its branch.
  await session_service.append_event(
      session,
      Event(
          invocation_id="inv_1",
          author="worker",
          branch="worker@1",
          content=types.Content(parts=[types.Part.from_text(text="done")]),
      ),
  )

  ic = await runner._setup_context_for_new_invocation(
      session=session,
      new_message=types.Content(parts=[types.Part.from_text(text="next")]),
      run_config=RunConfig(),
      state_delta=None,
      invocation_id="inv_2",
  )
  assert ic.agent == sub_agent
  assert ic.branch == "worker@1"


@pytest.mark.asyncio
async def test_run_live_restores_branch_for_non_root_agent():
  """run_live restores a non-root agent's branch from history."""

  class MockLiveLlmAgent(MockLlmAgent):
    """Transferable agent whose live turn reports the context branch."""

    async def _run_live_impl(
        self, invocation_context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
      yield Event(
          invocation_id=invocation_context.invocation_id,
          author=self.name,
          branch=invocation_context.branch,
          content=types.Content(
              role="model", parts=[types.Part(text="live hello")]
          ),
      )

  session_service = InMemorySessionService()
  root_agent = MockLiveLlmAgent("coordinator")
  sub_agent = MockLiveLlmAgent("worker", parent_agent=root_agent)
  root_agent.sub_agents = [sub_agent]
  runner = Runner(
      app_name="live_app",
      agent=root_agent,
      session_service=session_service,
  )

  session = await session_service.create_session(
      app_name="live_app", user_id="user_1", session_id="session_1"
  )
  await session_service.append_event(
      session,
      Event(
          invocation_id="inv_1",
          author="worker",
          branch="worker@1",
          content=types.Content(parts=[types.Part.from_text(text="earlier")]),
      ),
  )

  from google.adk.agents.live_request_queue import LiveRequestQueue

  live_queue = LiveRequestQueue()
  agen = runner.run_live(
      user_id="user_1",
      session_id="session_1",
      live_request_queue=live_queue,
  )
  event = await agen.__anext__()
  await agen.aclose()

  # The resumed sub-agent runs on its historical branch, not the root branch.
  assert event.author == "worker"
  assert event.branch == "worker@1"


@pytest.mark.asyncio
async def test_restore_keeps_sub_branch_keyed_by_function_call_id():
  """A sub-agent branch keyed by a function call id is restored, not skipped."""
  session_service = InMemorySessionService()
  root_agent = MockLlmAgent("coordinator")
  sub_agent = MockLlmAgent("worker", parent_agent=root_agent)
  root_agent.sub_agents = [sub_agent]
  app = App(
      name="test_app",
      root_agent=root_agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = Runner(app=app, session_service=session_service)
  session = await session_service.create_session(
      app_name="test_app", user_id="user_1", session_id="session_1"
  )
  await session_service.append_event(
      session,
      Event(
          invocation_id="inv_1",
          author="user",
          content=types.Content(parts=[types.Part.from_text(text="start")]),
      ),
  )
  # The sub-agent's own branch, keyed by a function call recorded in the
  # session -- the shape `AgentTool` produces.
  await session_service.append_event(
      session,
      Event(
          invocation_id="inv_1",
          author="worker",
          branch="coordinator.worker@fc-1",
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name="dig", id="fc-1", args={}
                      )
                  )
              ]
          ),
      ),
  )

  ic = await runner._setup_context_for_resumed_invocation(
      session=session,
      new_message=types.Content(parts=[types.Part.from_text(text="continue")]),
      invocation_id="inv_1",
      run_config=RunConfig(),
      state_delta=None,
  )
  assert ic.agent == sub_agent
  assert ic.branch == "coordinator.worker@fc-1"


@pytest.mark.asyncio
async def test_run_async_with_mock_session_service_does_not_corrupt_branch():
  """A mock session service returning a Mock from append_event does not corrupt ic.branch."""
  mock_session_service = mock.AsyncMock(spec=BaseSessionService)
  mock_session = Session(
      id="session_1", app_name="test_app", user_id="user_1", events=[], state={}
  )
  mock_session_service.create_session.return_value = mock_session
  mock_session_service.get_session.return_value = mock_session

  root_agent = MockLlmAgent("coordinator")
  runner = Runner(
      app_name="test_app",
      agent=root_agent,
      session_service=mock_session_service,
  )

  events = []
  async for event in runner.run_async(
      user_id="user_1",
      session_id="session_1",
      new_message=types.Content(parts=[types.Part.from_text(text="hello")]),
  ):
    events.append(event)

  assert len(events) == 1
  assert events[0].branch is None


@pytest.mark.asyncio
async def test_resume_finds_user_message_whose_text_is_not_the_first_part():
  """A multimodal user turn can be resumed even when text is not `parts[0]`.

  A user who attaches an image and then asks about it produces
  `[image, text]`. Matching only `parts[0].text` misses that message, and the
  resume path turns "not found" into a hard error.
  """
  session_service = InMemorySessionService()
  root_agent = MockLlmAgent("coordinator")
  app = App(
      name="test_app",
      root_agent=root_agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = Runner(app=app, session_service=session_service)
  session = await session_service.create_session(
      app_name="test_app", user_id="user_1", session_id="session_1"
  )
  await session_service.append_event(
      session,
      Event(
          invocation_id="inv_1",
          author="user",
          content=types.Content(
              role="user",
              parts=[
                  types.Part(
                      inline_data=types.Blob(
                          mime_type="image/png", data=b"\x89PNG"
                      )
                  ),
                  types.Part(text="what is in this picture?"),
              ],
          ),
      ),
  )

  ic = await runner._setup_context_for_resumed_invocation(
      session=session,
      new_message=None,
      invocation_id="inv_1",
      run_config=RunConfig(),
      state_delta=None,
  )

  assert ic.user_content is not None
  assert ic.user_content.parts[1].text == "what is in this picture?"


def test_find_user_message_for_invocation_requires_some_text():
  """A message with no text at all is still not treated as the user message."""
  session_service = InMemorySessionService()
  runner = Runner(
      app=App(name="test_app", root_agent=MockLlmAgent("coordinator")),
      session_service=session_service,
  )
  image_only = Event(
      invocation_id="inv_1",
      author="user",
      content=types.Content(
          role="user",
          parts=[
              types.Part(
                  inline_data=types.Blob(mime_type="image/png", data=b"\x89PNG")
              )
          ],
      ),
  )
  assert runner._find_user_message_for_invocation([image_only], "inv_1") is None


def _fc_part(name: str, call_id: str) -> types.Part:
  return types.Part(
      function_call=types.FunctionCall(name=name, id=call_id, args={})
  )


def _fr_part(name: str, call_id: str) -> types.Part:
  return types.Part(
      function_response=types.FunctionResponse(
          name=name, id=call_id, response={}
      )
  )


@pytest.mark.asyncio
async def test_resolve_invocation_id_rejects_responses_from_two_invocations():
  """Every response is checked, not just the first one.

  A message answering several calls at once must resolve to a single
  invocation. Inspecting only `function_responses[0]` would attribute the
  remaining responses to whichever invocation happened to come first.
  """
  session_service = InMemorySessionService()
  runner = Runner(
      app=App(name="test_app", root_agent=MockLlmAgent("root")),
      session_service=session_service,
  )
  session = await session_service.create_session(
      app_name="test_app", user_id="u", session_id="s"
  )
  for inv, call_id in (("inv_a", "fc-a"), ("inv_b", "fc-b")):
    await session_service.append_event(
        session,
        Event(
            invocation_id=inv,
            author="root",
            content=types.Content(parts=[_fc_part("t", call_id)]),
        ),
    )

  both = types.Content(
      role="user", parts=[_fr_part("t", "fc-a"), _fr_part("t", "fc-b")]
  )
  with pytest.raises(ValueError, match="resolve to multiple invocations"):
    runner._resolve_invocation_id(session, both, None)


@pytest.mark.asyncio
async def test_resolve_invocation_id_accepts_responses_from_one_invocation():
  """Several responses to calls from the same invocation still resolve."""
  session_service = InMemorySessionService()
  runner = Runner(
      app=App(name="test_app", root_agent=MockLlmAgent("root")),
      session_service=session_service,
  )
  session = await session_service.create_session(
      app_name="test_app", user_id="u", session_id="s"
  )
  await session_service.append_event(
      session,
      Event(
          invocation_id="inv_a",
          author="root",
          content=types.Content(
              parts=[_fc_part("t", "fc-a"), _fc_part("t", "fc-b")]
          ),
      ),
  )

  both = types.Content(
      role="user", parts=[_fr_part("t", "fc-a"), _fr_part("t", "fc-b")]
  )
  assert runner._resolve_invocation_id(session, both, None) == "inv_a"


if __name__ == "__main__":
  pytest.main([__file__])
