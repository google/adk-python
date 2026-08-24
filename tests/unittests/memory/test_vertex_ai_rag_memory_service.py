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
import json
import os
import tempfile
import threading
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


def _session() -> Session:
  return Session(
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


@pytest.fixture(name="temp_dir")
def _temp_dir(tmp_path, monkeypatch):
  """Redirects NamedTemporaryFile so a leaked transcript is observable."""
  monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
  return tmp_path


@pytest.mark.asyncio
async def test_search_memory_rejects_ambiguous_legacy_display_names(mocker):
  """Ensures dotted user IDs cannot match another user's legacy memory."""
  memory_service = VertexAiRagMemoryService(rag_corpus="unused")
  fake_rag = SimpleNamespace(
      retrieval_query=mocker.Mock(
          return_value=SimpleNamespace(
              contexts=SimpleNamespace(
                  contexts=[
                      _rag_context(
                          "demo.alice.smith.session_secret",
                          "SECRET_FROM_ALICE_SMITH",
                      ),
                      _rag_context(
                          _build_source_display_name(
                              "demo", "alice", "session_ok"
                          ),
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
      )
  )
  mocker.patch("google.adk.dependencies.vertexai.rag", fake_rag)

  response = await memory_service.search_memory(
      app_name="demo", user_id="alice", query="secret"
  )

  texts = [memory.content.parts[0].text for memory in response.memories]
  assert texts == ["NORMAL_ALICE_MEMORY", "LEGACY_ALICE_MEMORY"]


@pytest.mark.asyncio
async def test_add_and_search_memory_uses_unambiguous_display_names(
    mocker, temp_dir
):
  memory_service = VertexAiRagMemoryService(rag_corpus="unused")
  upload_file = mocker.Mock()
  fake_rag = SimpleNamespace(upload_file=upload_file)
  mocker.patch("google.adk.dependencies.vertexai.rag", fake_rag)

  await memory_service.add_session_to_memory(_session())

  display_name = upload_file.call_args.kwargs["display_name"]
  assert display_name.startswith(_SOURCE_DISPLAY_NAME_PREFIX)
  assert display_name != "demo.app.alice.smith.session.secret"

  fake_rag.retrieval_query = mocker.Mock(
      return_value=SimpleNamespace(
          contexts=SimpleNamespace(
              contexts=[_rag_context(display_name, "sensitive memory")]
          )
      )
  )

  response = await memory_service.search_memory(
      app_name="demo.app", user_id="alice.smith", query="sensitive"
  )

  assert [memory.content.parts[0].text for memory in response.memories] == [
      "sensitive memory"
  ]
  assert not list(temp_dir.iterdir())


@pytest.mark.asyncio
async def test_add_session_cleans_temp_file_after_partial_upload_failure(
    mocker, temp_dir
):
  attempted_corpora = []

  def upload_file(*, corpus_name, **_kwargs):
    attempted_corpora.append(corpus_name)
    if corpus_name == "second":
      raise RuntimeError("upload failed")

  fake_rag = SimpleNamespace(upload_file=upload_file)
  mocker.patch("google.adk.dependencies.vertexai.rag", fake_rag)
  memory_service = VertexAiRagMemoryService(rag_corpus="first")
  memory_service._vertex_rag_store.rag_resources = [
      types.VertexRagStoreRagResource(rag_corpus="first"),
      types.VertexRagStoreRagResource(rag_corpus="second"),
      types.VertexRagStoreRagResource(rag_corpus="third"),
  ]

  with pytest.raises(RuntimeError, match="upload failed"):
    await memory_service.add_session_to_memory(_session())

  assert attempted_corpora == ["first", "second"]
  assert not list(temp_dir.iterdir())


@pytest.mark.asyncio
async def test_add_session_cleans_temp_file_when_cancelled(mocker, temp_dir):
  upload_started = threading.Event()
  allow_upload_to_finish = threading.Event()

  def upload_file(**_kwargs):
    upload_started.set()
    # Bounded so a regression that runs this on the event loop, where nothing
    # can set the event, fails the test instead of hanging the suite.
    allow_upload_to_finish.wait(timeout=30)

  fake_rag = SimpleNamespace(upload_file=upload_file)
  mocker.patch("google.adk.dependencies.vertexai.rag", fake_rag)
  memory_service = VertexAiRagMemoryService(rag_corpus="corpus")

  add_session = asyncio.create_task(
      memory_service.add_session_to_memory(_session())
  )
  try:
    await asyncio.to_thread(upload_started.wait)
    add_session.cancel()
    with pytest.raises(asyncio.CancelledError):
      await add_session
  finally:
    allow_upload_to_finish.set()

  assert not list(temp_dir.iterdir())


@pytest.mark.asyncio
async def test_failed_temp_file_removal_does_not_mask_the_upload_error(
    mocker, temp_dir
):
  """Cleanup runs in a finally, so it must not displace the real exception.

  A cancelled upload can leave the worker thread still holding the file, and
  on some platforms the remove then fails. Raising from the finally block
  would replace the exception already propagating.
  """

  def upload_file(**_kwargs):
    raise RuntimeError("upload failed")

  mocker.patch(
      "google.adk.dependencies.vertexai.rag",
      SimpleNamespace(upload_file=upload_file),
  )
  mocker.patch(
      "google.adk.memory.vertex_ai_rag_memory_service.os.remove",
      side_effect=PermissionError("file still in use"),
  )
  memory_service = VertexAiRagMemoryService(rag_corpus="corpus")

  with pytest.raises(RuntimeError, match="upload failed"):
    await memory_service.add_session_to_memory(_session())


@pytest.mark.asyncio
async def test_add_session_leaves_no_temp_file_when_corpus_missing(temp_dir):
  memory_service = VertexAiRagMemoryService(rag_corpus=None)

  with pytest.raises(ValueError, match="rag_corpus must be set"):
    await memory_service.add_session_to_memory(_session())

  assert not list(temp_dir.iterdir())


@pytest.mark.asyncio
async def test_add_session_uploads_off_the_event_loop(mocker, temp_dir):
  upload_thread_ids = []

  def upload_file(*, path, **_kwargs):
    upload_thread_ids.append(threading.get_ident())
    # The transcript is still on disk while the upload is in flight.
    assert os.path.exists(path)

  fake_rag = SimpleNamespace(upload_file=upload_file)
  mocker.patch("google.adk.dependencies.vertexai.rag", fake_rag)
  memory_service = VertexAiRagMemoryService(rag_corpus="corpus")

  await memory_service.add_session_to_memory(_session())

  assert upload_thread_ids == [upload_thread_ids[0]]
  assert threading.get_ident() not in upload_thread_ids
  assert not list(temp_dir.iterdir())


@pytest.mark.asyncio
async def test_search_memory_queries_off_the_event_loop(mocker):
  query_thread_ids = []

  def retrieval_query(**_kwargs):
    query_thread_ids.append(threading.get_ident())
    return SimpleNamespace(contexts=SimpleNamespace(contexts=[]))

  fake_rag = SimpleNamespace(retrieval_query=retrieval_query)
  mocker.patch("google.adk.dependencies.vertexai.rag", fake_rag)
  memory_service = VertexAiRagMemoryService(rag_corpus="corpus")

  await memory_service.search_memory(
      app_name="demo", user_id="alice", query="memory"
  )

  assert query_thread_ids
  assert threading.get_ident() not in query_thread_ids
