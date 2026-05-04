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

"""Tests for runner.rewind_async."""

from typing import Any
from typing import Optional
from typing import Union

from google.adk.agents.base_agent import BaseAgent
from google.adk.artifacts.base_artifact_service import ensure_part
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.events.event import Event
from google.adk.events.event import EventActions
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
import pytest


class _NoFileDataArtifactService(InMemoryArtifactService):
  """Artifact service that rejects file_data parts, like GCS/File services."""

  async def save_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      artifact: Union[types.Part, dict[str, Any]],
      session_id: Optional[str] = None,
      custom_metadata: Optional[dict[str, Any]] = None,
  ) -> int:
    artifact = ensure_part(artifact)
    if artifact.file_data:
      raise NotImplementedError(
          "Saving artifact with file_data is not supported."
      )
    return await super().save_artifact(
        app_name=app_name,
        user_id=user_id,
        filename=filename,
        artifact=artifact,
        session_id=session_id,
        custom_metadata=custom_metadata,
    )


class TestRunnerRewind:
  """Tests for runner.rewind_async."""

  runner: Runner

  def setup_method(self):
    """Set up test fixtures."""
    root_agent = BaseAgent(name="test_agent")
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    self.runner = Runner(
        app_name="test_app",
        agent=root_agent,
        session_service=session_service,
        artifact_service=artifact_service,
    )

  @pytest.mark.asyncio
  async def test_rewind_async_with_state_and_artifacts(self):
    """Tests rewind_async rewinds state and artifacts."""
    runner = self.runner
    user_id = "test_user"
    session_id = "test_session"

    # 1. Setup session and initial artifacts
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )

    # invocation1
    await runner.artifact_service.save_artifact(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id,
        filename="f1",
        artifact=types.Part.from_text(text="f1v0"),
    )
    event1 = Event(
        invocation_id="invocation1",
        author="agent",
        content=types.Content(parts=[types.Part.from_text(text="event1")]),
        actions=EventActions(
            state_delta={"k1": "v1"}, artifact_delta={"f1": 0}
        ),
    )
    await runner.session_service.append_event(session=session, event=event1)

    # invocation2
    await runner.artifact_service.save_artifact(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id,
        filename="f1",
        artifact=types.Part.from_text(text="f1v1"),
    )
    await runner.artifact_service.save_artifact(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id,
        filename="f2",
        artifact=types.Part.from_text(text="f2v0"),
    )
    event2 = Event(
        invocation_id="invocation2",
        author="agent",
        content=types.Content(parts=[types.Part.from_text(text="event2")]),
        actions=EventActions(
            state_delta={"k1": "v2", "k2": "v2"},
            artifact_delta={"f1": 1, "f2": 0},
        ),
    )
    await runner.session_service.append_event(session=session, event=event2)

    # invocation3
    event3 = Event(
        invocation_id="invocation3",
        author="agent",
        content=types.Content(parts=[types.Part.from_text(text="event3")]),
        actions=EventActions(state_delta={"k2": "v3"}),
    )
    await runner.session_service.append_event(session=session, event=event3)

    session = await runner.session_service.get_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )
    assert session.state == {"k1": "v2", "k2": "v3"}
    assert await runner.artifact_service.load_artifact(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id,
        filename="f1",
    ) == types.Part.from_text(text="f1v1")
    assert await runner.artifact_service.load_artifact(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id,
        filename="f2",
    ) == types.Part.from_text(text="f2v0")

    # 2. Rewind before invocation2
    await runner.rewind_async(
        user_id=user_id,
        session_id=session_id,
        rewind_before_invocation_id="invocation2",
    )

    # 3. Verify state and artifacts are rewound
    session = await runner.session_service.get_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )
    # After rewind before invocation2, only event1 state delta should apply.
    assert session.state["k1"] == "v1"
    assert not session.state["k2"]
    # f1 should be rewound to v0
    assert await runner.artifact_service.load_artifact(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id,
        filename="f1",
    ) == types.Part.from_text(text="f1v0")
    # f2 should not exist
    assert (
        await runner.artifact_service.load_artifact(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id,
            filename="f2",
        )
        is None
    )

  @pytest.mark.asyncio
  async def test_rewind_async_not_first_invocation(self):
    """Tests rewind_async rewinds state and artifacts to invocation2."""
    runner = self.runner
    user_id = "test_user"
    session_id = "test_session"

    # 1. Setup session and initial artifacts
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )
    # invocation1
    await runner.artifact_service.save_artifact(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id,
        filename="f1",
        artifact=types.Part.from_text(text="f1v0"),
    )
    event1 = Event(
        invocation_id="invocation1",
        author="agent",
        content=types.Content(parts=[types.Part.from_text(text="event1")]),
        actions=EventActions(
            state_delta={"k1": "v1"}, artifact_delta={"f1": 0}
        ),
    )
    await runner.session_service.append_event(session=session, event=event1)

    # invocation2
    await runner.artifact_service.save_artifact(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id,
        filename="f1",
        artifact=types.Part.from_text(text="f1v1"),
    )
    await runner.artifact_service.save_artifact(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id,
        filename="f2",
        artifact=types.Part.from_text(text="f2v0"),
    )
    event2 = Event(
        invocation_id="invocation2",
        author="agent",
        content=types.Content(parts=[types.Part.from_text(text="event2")]),
        actions=EventActions(
            state_delta={"k1": "v2", "k2": "v2"},
            artifact_delta={"f1": 1, "f2": 0},
        ),
    )
    await runner.session_service.append_event(session=session, event=event2)

    # invocation3
    event3 = Event(
        invocation_id="invocation3",
        author="agent",
        content=types.Content(parts=[types.Part.from_text(text="event3")]),
        actions=EventActions(state_delta={"k2": "v3"}),
    )
    await runner.session_service.append_event(session=session, event=event3)

    # 2. Rewind before invocation3
    await runner.rewind_async(
        user_id=user_id,
        session_id=session_id,
        rewind_before_invocation_id="invocation3",
    )

    # 3. Verify state and artifacts are rewound
    session = await runner.session_service.get_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )
    # After rewind before invocation3, event1 and event2 state deltas should apply.
    assert session.state == {"k1": "v2", "k2": "v2"}
    # f1 should be v1
    assert await runner.artifact_service.load_artifact(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id,
        filename="f1",
    ) == types.Part.from_text(text="f1v1")
    # f2 should be v0
    assert await runner.artifact_service.load_artifact(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id,
        filename="f2",
    ) == types.Part.from_text(text="f2v0")


class TestRunnerRewindNoFileData:
  """Tests that rewind works with artifact services that reject file_data."""

  @pytest.mark.asyncio
  async def test_rewind_uses_load_artifact_not_file_data(self):
    """Rewind must not construct file_data parts for artifact restoration.

    GCS and File artifact services reject file_data parts. The runner
    should use load_artifact to get inline_data instead.
    """
    root_agent = BaseAgent(name="test_agent")
    session_service = InMemorySessionService()
    artifact_service = _NoFileDataArtifactService()
    runner = Runner(
        app_name="test_app",
        agent=root_agent,
        session_service=session_service,
        artifact_service=artifact_service,
    )
    user_id = "test_user"
    session_id = "test_session"

    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )

    # invocation1: create artifact f1 v0
    await runner.artifact_service.save_artifact(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id,
        filename="f1",
        artifact=types.Part.from_text(text="f1v0"),
    )
    event1 = Event(
        invocation_id="invocation1",
        author="agent",
        content=types.Content(parts=[types.Part.from_text(text="e1")]),
        actions=EventActions(
            state_delta={"k1": "v1"}, artifact_delta={"f1": 0}
        ),
    )
    await runner.session_service.append_event(session=session, event=event1)

    # invocation2: update artifact f1 to v1
    await runner.artifact_service.save_artifact(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id,
        filename="f1",
        artifact=types.Part.from_text(text="f1v1"),
    )
    event2 = Event(
        invocation_id="invocation2",
        author="agent",
        content=types.Content(parts=[types.Part.from_text(text="e2")]),
        actions=EventActions(artifact_delta={"f1": 1}),
    )
    await runner.session_service.append_event(session=session, event=event2)

    session = await runner.session_service.get_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )

    # Rewind before invocation2 — this would raise NotImplementedError
    # with the old code that constructed file_data parts.
    await runner.rewind_async(
        user_id=user_id,
        session_id=session_id,
        rewind_before_invocation_id="invocation2",
    )

    # f1 should be restored to v0 content
    restored = await runner.artifact_service.load_artifact(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id,
        filename="f1",
    )
    assert restored == types.Part.from_text(text="f1v0")
