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

import importlib
from pathlib import Path
import sys
import textwrap
from typing import AsyncGenerator
from typing import Optional
from unittest.mock import AsyncMock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.cli.utils.agent_loader import AgentLoader
from google.adk.events.event import Event
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.tools.function_tool import FunctionTool
from google.genai import types
import pytest

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

  with pytest.raises(ValueError) as excinfo:
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


@pytest.mark.asyncio
async def test_run_live_detects_streaming_tools_with_canonical_tools():
  """run_live should detect streaming tools using canonical_tools and tool.name."""

  # Define streaming tools - one as raw function, one wrapped in FunctionTool
  async def raw_streaming_tool(
      input_stream: LiveRequestQueue,
  ) -> AsyncGenerator[str, None]:
    """A raw streaming tool function."""
    yield "test"

  async def wrapped_streaming_tool(
      input_stream: LiveRequestQueue,
  ) -> AsyncGenerator[str, None]:
    """A streaming tool wrapped in FunctionTool."""
    yield "test"

  def non_streaming_tool(param: str) -> str:
    """A regular non-streaming tool."""
    return param

  # Create a mock LlmAgent that yields an event and captures invocation context
  captured_context = {}

  class StreamingToolsAgent(LlmAgent):

    async def _run_live_impl(
        self, invocation_context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
      # Capture the active_streaming_tools for verification
      captured_context["active_streaming_tools"] = (
          invocation_context.active_streaming_tools
      )
      yield Event(
          invocation_id=invocation_context.invocation_id,
          author=self.name,
          content=types.Content(
              role="model", parts=[types.Part(text="streaming test")]
          ),
      )

  agent = StreamingToolsAgent(
      name="streaming_agent",
      model="gemini-2.0-flash",
      tools=[
          raw_streaming_tool,  # Raw function
          FunctionTool(wrapped_streaming_tool),  # Wrapped in FunctionTool
          non_streaming_tool,  # Non-streaming tool (should not be detected)
      ],
  )

  session_service = InMemorySessionService()
  artifact_service = InMemoryArtifactService()
  runner = Runner(
      app_name="streaming_test_app",
      agent=agent,
      session_service=session_service,
      artifact_service=artifact_service,
      auto_create_session=True,
  )

  live_queue = LiveRequestQueue()

  agen = runner.run_live(
      user_id="user",
      session_id="test_session",
      live_request_queue=live_queue,
  )

  event = await agen.__anext__()
  await agen.aclose()

  assert event.author == "streaming_agent"

  # Verify streaming tools were detected correctly
  active_tools = captured_context.get("active_streaming_tools", {})
  assert "raw_streaming_tool" in active_tools
  assert "wrapped_streaming_tool" in active_tools
  # Non-streaming tool should not be detected
  assert "non_streaming_tool" not in active_tools


@pytest.mark.asyncio
async def test_runner_allows_nested_agent_directories(tmp_path, monkeypatch):
  project_root = tmp_path / "workspace"
  agent_dir = project_root / "agents" / "examples" / "001_hello_world"
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
  loaded_agent = loader.load_agent("001_hello_world")

  assert isinstance(loaded_agent, BaseAgent)
  session_service = InMemorySessionService()
  artifact_service = InMemoryArtifactService()
  runner = Runner(
      app_name="001_hello_world",
      agent=loaded_agent,
      session_service=session_service,
      artifact_service=artifact_service,
  )
  assert runner._app_name_alignment_hint is None

  session = await session_service.create_session(
      app_name="001_hello_world",
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

    # Should return sub_agent2 due to function response, not root_agent
    result = self.runner._find_agent_to_run(session, self.root_agent)
    assert result == self.sub_agent2

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
  async def test_runner_close_calls_plugin_close(self):
    """Test that runner.close() calls plugin manager close."""
    # Mock the plugin manager's close method
    self.runner.plugin_manager.close = AsyncMock()

    await self.runner.close()

    self.runner.plugin_manager.close.assert_awaited_once()

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
        match="When app is provided, agent should not be provided.",
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
        match="Either app or both app_name and agent must be provided.",
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
        match="Either app or both app_name and agent must be provided.",
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
        cache_intervals=30, ttl_seconds=14400, min_tokens=4096  # 4 hours
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
        "ContextCacheConfig(cache_intervals=30, ttl=14400s, min_tokens=4096)"
    )
    assert str(runner.context_cache_config) == expected_str


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

root_agent = MyAgent(name="{agent_dir_name}", model="gemini-2.0-flash")
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

    Regression test for GitHub issue #3143: Users who instantiate LlmAgent
    directly and run from a directory that is a parent of the ADK installation
    were getting false positive 'App name mismatch' warnings.

    This also verifies that _infer_agent_origin returns None for ADK internal
    modules (google.adk.*).
    """
    agent = LlmAgent(
        name="my_custom_agent",
        model="gemini-2.0-flash",
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


if __name__ == "__main__":
  pytest.main([__file__])
