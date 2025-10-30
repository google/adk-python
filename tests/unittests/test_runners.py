# Copyright 2025 Google LLC
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

from typing import Optional

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.events.event import Event
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
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

  async def _run_async_impl(self, invocation_context):
    yield Event(
        invocation_id=invocation_context.invocation_id,
        author=self.name,
        content=types.Content(
            role="model", parts=[types.Part(text="Test response")]
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

  async def _run_async_impl(self, invocation_context):
    yield Event(
        invocation_id=invocation_context.invocation_id,
        author=self.name,
        content=types.Content(
            role="model", parts=[types.Part(text="Test LLM response")]
        ),
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

  @pytest.mark.asyncio
  async def test_runner_modifies_event_after_execution(self):
    """Test that the runner modifies the event after execution."""
    self.plugin.enable_event_callback = True

    events = await self.run_test()
    generated_event = events[0]
    modified_event_message = generated_event.content.parts[0].text

    assert modified_event_message == MockPlugin.ON_EVENT_CALLBACK_MSG

  def test_runner_init_raises_error_with_app_and_app_name_and_agent(self):
    """Test that ValueError is raised when app, app_name and agent are provided."""
    with pytest.raises(
        ValueError,
        match="When app is provided, app_name should not be provided.",
    ):
      Runner(
          app=App(name="test_app", root_agent=self.root_agent),
          app_name="test_app",
          agent=self.root_agent,
          session_service=self.session_service,
          artifact_service=self.artifact_service,
      )

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


if __name__ == "__main__":
  pytest.main([__file__])
