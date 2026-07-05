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

import json
import os
import tempfile
from types import SimpleNamespace

from google.adk.events.event import Event
from google.adk.memory.vertex_ai_rag_memory_service import _build_source_display_name
from google.adk.memory.vertex_ai_rag_memory_service import _SOURCE_DISPLAY_NAME_PREFIX
from google.adk.memory.vertex_ai_rag_memory_service import VertexAiRagMemoryService
from google.adk.sessions.session import Session
from google.genai import types
import pytest


def _rag_context(source_display_name: str, text: str) -> SimpleNamespace:
  return SimpleNamespace(
      source_display_name=source_display_name,
      text=json.dumps({"author": "user", "timestamp": 1, "text": text}),
  )


def _sample_session() -> Session:
  return Session(
      app_name="demo",
      user_id="alice",
      id="session-1",
      last_update_time=1,
      events=[
          Event(
              id="event-1",
              author="user",
              timestamp=1,
              content=types.Content(parts=[types.Part(text="hello")]),
          )
      ],
  )


def _track_temp_files(mocker) -> list[str]:
  """Records the paths of every NamedTemporaryFile created."""
  created_paths: list[str] = []
  real_named_temp_file = tempfile.NamedTemporaryFile

  def _spy(*args, **kwargs):
    temp_file = real_named_temp_file(*args, **kwargs)
    created_paths.append(temp_file.name)
    return temp_file

  mocker.patch(
      "google.adk.memory.vertex_ai_rag_memory_service.tempfile.NamedTemporaryFile",
      side_effect=_spy,
  )
  return created_paths


@pytest.mark.asyncio
async def test_add_session_removes_temp_file_when_upload_fails(mocker):
  """The temp file must not leak when rag.upload_file raises."""
  memory_service = VertexAiRagMemoryService(rag_corpus="unused")
  created_paths = _track_temp_files(mocker)
  fake_client = mocker.Mock()
  fake_client.rag.upload_file.side_effect = RuntimeError("upload boom")
  mocker.patch("agentplatform.Client", return_value=fake_client)

  with pytest.raises(RuntimeError, match="upload boom"):
    await memory_service.add_session_to_memory(_sample_session())

  assert created_paths, "expected a temp file to have been created"
  assert not os.path.exists(created_paths[0])


@pytest.mark.asyncio
async def test_add_session_does_not_create_temp_file_without_rag_resources(
    mocker,
):
  """Validation happens before the temp file is created, so nothing leaks."""
  memory_service = VertexAiRagMemoryService(rag_corpus="unused")
  memory_service._vertex_rag_store.rag_resources = None
  created_paths = _track_temp_files(mocker)

  with pytest.raises(ValueError, match="Rag resources must be set."):
    await memory_service.add_session_to_memory(_sample_session())

  assert created_paths == []


@pytest.mark.asyncio
async def test_search_memory_rejects_ambiguous_legacy_display_names(mocker):
  """Ensures dotted user IDs cannot match another user's legacy memory."""
  memory_service = VertexAiRagMemoryService(rag_corpus="unused")

  fake_client = mocker.Mock()
  fake_client.rag.retrieve_contexts.return_value = SimpleNamespace(
      contexts=SimpleNamespace(
          contexts=[
              _rag_context(
                  "demo.alice.smith.session_secret",
                  "SECRET_FROM_ALICE_SMITH",
              ),
              _rag_context(
                  _build_source_display_name("demo", "alice", "session_ok"),
                  "NORMAL_ALICE_MEMORY",
              ),
              _rag_context(
                  "demo.alice.legacy_session",
                  "LEGACY_ALICE_MEMORY",
              ),
              _rag_context("demo.bob.session_other", "BOB_MEMORY"),
          ]
      )
  )

  mocker.patch("agentplatform.Client", return_value=fake_client)

  response = await memory_service.search_memory(
      app_name="demo", user_id="alice", query="secret"
  )

  texts = [memory.content.parts[0].text for memory in response.memories]
  assert texts == ["NORMAL_ALICE_MEMORY", "LEGACY_ALICE_MEMORY"]


@pytest.mark.asyncio
async def test_add_and_search_memory_uses_unambiguous_display_names(mocker):
  memory_service = VertexAiRagMemoryService(rag_corpus="unused")

  fake_client = mocker.Mock()
  mocker.patch("agentplatform.Client", return_value=fake_client)

  await memory_service.add_session_to_memory(
      Session(
          app_name="demo.app",
          user_id="alice.smith",
          id="session.secret",
          last_update_time=1,
          events=[
              Event(
                  id="event-1",
                  author="user",
                  timestamp=1,
                  content=types.Content(
                      parts=[types.Part(text="sensitive memory")]
                  ),
              )
          ],
      )
  )

  display_name = fake_client.rag.upload_file.call_args.kwargs["display_name"]
  assert display_name.startswith(_SOURCE_DISPLAY_NAME_PREFIX)
  assert display_name != "demo.app.alice.smith.session.secret"

  fake_client.rag.retrieve_contexts.return_value = SimpleNamespace(
      contexts=SimpleNamespace(
          contexts=[_rag_context(display_name, "sensitive memory")]
      )
  )

  response = await memory_service.search_memory(
      app_name="demo.app", user_id="alice.smith", query="sensitive"
  )

  assert [memory.content.parts[0].text for memory in response.memories] == [
      "sensitive memory"
  ]
