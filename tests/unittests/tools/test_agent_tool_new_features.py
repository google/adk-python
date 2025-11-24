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

"""Tests for new features added to AgentTool and DatabaseSessionService."""

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.plugins.plugin_manager import PluginManager
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from google.genai.types import Part
from pytest import mark

from .. import testing_utils

function_call_no_schema = Part.from_function_call(
    name="tool_agent", args={"request": "test1"}
)


@mark.asyncio
async def test_agent_tool_handles_dict_args():
  """Test that AgentTool handles dictionary arguments correctly (non-request key)."""

  mock_model = testing_utils.MockModel.create(
      responses=["response to dict arg"]
  )

  tool_agent = Agent(
      name="tool_agent",
      model=mock_model,
  )

  # Create invocation context
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name="test_app", user_id="test_user"
  )

  invocation_context = InvocationContext(
      invocation_id="test_invocation",
      agent=tool_agent,
      session=session,
      session_service=session_service,
      artifact_service=InMemoryArtifactService(),
      memory_service=InMemoryMemoryService(),
      plugin_manager=PluginManager(plugins=[]),
      run_config=RunConfig(),
  )

  tool_context = ToolContext(invocation_context=invocation_context)
  agent_tool = AgentTool(agent=tool_agent)

  # Test with dict argument that doesn't have 'request' key
  result = await agent_tool.run_async(
      args={"custom_key": "custom_value"}, tool_context=tool_context
  )

  assert result is not None
  assert "response to dict arg" in str(result)


@mark.asyncio
async def test_database_session_service_persists_usage_metadata():
  """Test that DatabaseSessionService correctly persists usage_metadata with flag_modified."""
  import os
  import tempfile

  from google.adk.sessions.database_session_service import DatabaseSessionService

  # Create temporary database
  temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
  temp_db.close()
  db_url = f"sqlite+aiosqlite:///{temp_db.name}"

  try:
    service = DatabaseSessionService(db_url)

    # Create session
    session = await service.create_session(
        app_name="test_app", user_id="user123"
    )

    # Create event with usage_metadata
    event = Event(
        id="evt1",
        invocation_id="inv1",
        author="model",
        actions=EventActions(),
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100,
            candidates_token_count=50,
            total_token_count=150,
        ),
    )

    # Persist event
    await service.append_event(session, event)

    # Retrieve session and verify usage_metadata was persisted
    retrieved_session = await service.get_session(
        app_name="test_app", user_id="user123", session_id=session.id
    )

    assert retrieved_session is not None
    assert len(retrieved_session.events) > 0

    # Find the event with usage_metadata
    found_usage_metadata = False
    for evt in retrieved_session.events:
      if evt.usage_metadata is not None:
        assert evt.usage_metadata.total_token_count == 150
        assert evt.usage_metadata.prompt_token_count == 100
        assert evt.usage_metadata.candidates_token_count == 50
        found_usage_metadata = True
        break

    assert found_usage_metadata, "usage_metadata was not persisted correctly"

  finally:
    # Cleanup
    if os.path.exists(temp_db.name):
      os.unlink(temp_db.name)


@mark.asyncio
async def test_database_session_service_persists_citation_metadata():
  """Test that DatabaseSessionService correctly persists citation_metadata."""
  import os
  import tempfile

  from google.adk.sessions.database_session_service import DatabaseSessionService

  # Create temporary database
  temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
  temp_db.close()
  db_url = f"sqlite+aiosqlite:///{temp_db.name}"

  try:
    service = DatabaseSessionService(db_url)

    # Create session
    session = await service.create_session(
        app_name="test_app", user_id="user123"
    )

    # Create event with citation_metadata
    event = Event(
        id="evt1",
        invocation_id="inv1",
        author="model",
        actions=EventActions(),
        citation_metadata=types.CitationMetadata(
            citations=[
                types.Citation(
                    start_index=0,
                    end_index=10,
                    uri="https://example.com",
                    title="Example Source",
                )
            ]
        ),
    )

    # Persist event
    await service.append_event(session, event)

    # Retrieve session and verify citation_metadata was persisted
    retrieved_session = await service.get_session(
        app_name="test_app", user_id="user123", session_id=session.id
    )

    assert retrieved_session is not None
    assert len(retrieved_session.events) > 0

    # Find the event with citation_metadata
    found_citation_metadata = False
    for evt in retrieved_session.events:
      if evt.citation_metadata is not None:
        assert len(evt.citation_metadata.citations) == 1
        assert evt.citation_metadata.citations[0].uri == "https://example.com"
        assert evt.citation_metadata.citations[0].title == "Example Source"
        found_citation_metadata = True
        break

    assert (
        found_citation_metadata
    ), "citation_metadata was not persisted correctly"

  finally:
    # Cleanup
    if os.path.exists(temp_db.name):
      os.unlink(temp_db.name)


# Made with Bob
