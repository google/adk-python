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
from types import SimpleNamespace

from google.adk.events.event import Event
from google.adk.memory.vertex_ai_rag_memory_service import _build_source_display_name
from google.adk.memory.vertex_ai_rag_memory_service import _SOURCE_DISPLAY_NAME_PREFIX
from google.adk.memory.vertex_ai_rag_memory_service import VertexAiRagMemoryService
from google.adk.sessions.session import Session
from google.genai import types
import pytest
from pytest_mock import MockerFixture


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


class _StallingClose:
  """An aclose() that parks mid-flight so a cancellation can race it."""

  def __init__(self):
    self.started = asyncio.Event()
    self.release = asyncio.Event()
    self.completed = False

  async def aclose(self) -> None:
    self.started.set()
    await self.release.wait()
    self.completed = True


async def _cancel_while_closing(task: asyncio.Task, close: _StallingClose):
  task.cancel()
  await close.started.wait()
  # Second cancellation, delivered while the close is still suspended.
  task.cancel()
  close.release.set()
  with pytest.raises(asyncio.CancelledError):
    await task
  await asyncio.sleep(0)


@pytest.fixture(name="temp_dir")
def _temp_dir(tmp_path, monkeypatch):
  """Redirects NamedTemporaryFile so a leaked transcript is observable."""
  monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
  return tmp_path


@pytest.mark.asyncio
@pytest.mark.parametrize("configured_top_k", [7, None])
async def test_search_memory_forwards_similarity_top_k(
    mocker: MockerFixture,
    configured_top_k: int | None,
) -> None:
  memory_service = VertexAiRagMemoryService(
      rag_corpus="unused",
      similarity_top_k=configured_top_k,
  )
  fake_client = mocker.Mock()
  fake_client.aclose = mocker.AsyncMock()
  fake_client.rag.retrieve_contexts = mocker.AsyncMock(
      return_value=SimpleNamespace(contexts=SimpleNamespace(contexts=[]))
  )
  mocker.patch(
      "agentplatform.Client", return_value=mocker.Mock(aio=fake_client)
  )

  await memory_service.search_memory(
      app_name="demo", user_id="alice", query="memory"
  )

  fake_client.rag.retrieve_contexts.assert_awaited_once()
  kwargs = fake_client.rag.retrieve_contexts.call_args.kwargs
  assert kwargs["query"].similarity_top_k == configured_top_k
  # retrieveContexts reads top-k from the query; sending it on the store
  # would put an undefined field on the request.
  assert kwargs["vertex_rag_store"].similarity_top_k is None


@pytest.mark.asyncio
async def test_search_memory_rejects_ambiguous_legacy_display_names(mocker):
  """Ensures dotted user IDs cannot match another user's legacy memory."""
  memory_service = VertexAiRagMemoryService(rag_corpus="unused")

  fake_client = mocker.Mock()
  fake_client.aclose = mocker.AsyncMock()
  fake_client.rag.retrieve_contexts = mocker.AsyncMock(
      return_value=SimpleNamespace(
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
  )

  mocker.patch(
      "agentplatform.Client", return_value=mocker.Mock(aio=fake_client)
  )

  response = await memory_service.search_memory(
      app_name="demo", user_id="alice", query="secret"
  )

  texts = [memory.content.parts[0].text for memory in response.memories]
  assert texts == ["NORMAL_ALICE_MEMORY", "LEGACY_ALICE_MEMORY"]
  fake_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_add_and_search_memory_uses_unambiguous_display_names(
    mocker, temp_dir
):
  memory_service = VertexAiRagMemoryService(rag_corpus="unused")

  fake_client = mocker.Mock()
  fake_client.aclose = mocker.AsyncMock()
  fake_client.rag.upload_file = mocker.AsyncMock()
  fake_client.rag.retrieve_contexts = mocker.AsyncMock()
  mocker.patch(
      "agentplatform.Client", return_value=mocker.Mock(aio=fake_client)
  )

  await memory_service.add_session_to_memory(_session())

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
  assert fake_client.aclose.await_count == 2
  assert not list(temp_dir.iterdir())


@pytest.mark.asyncio
async def test_add_session_upload_does_not_block_event_loop(mocker, temp_dir):
  upload_started = asyncio.Event()
  allow_upload_to_finish = asyncio.Event()
  uploaded_path: str | None = None

  async def upload_file(*, path: str, **_kwargs: object) -> None:
    nonlocal uploaded_path
    uploaded_path = path
    upload_started.set()
    await allow_upload_to_finish.wait()

  fake_client = mocker.Mock()
  fake_client.aclose = mocker.AsyncMock()
  fake_client.rag.upload_file = mocker.AsyncMock(side_effect=upload_file)
  mocker.patch(
      "agentplatform.Client", return_value=mocker.Mock(aio=fake_client)
  )
  memory_service = VertexAiRagMemoryService(rag_corpus="corpus")

  add_session = asyncio.create_task(
      memory_service.add_session_to_memory(_session())
  )
  await upload_started.wait()

  assert not add_session.done()
  assert uploaded_path and os.path.exists(uploaded_path)
  allow_upload_to_finish.set()
  await add_session
  assert not list(temp_dir.iterdir())


@pytest.mark.asyncio
async def test_add_session_cleans_temp_file_after_partial_upload_failure(
    mocker, temp_dir
):
  attempted_corpora: list[str] = []

  async def upload_file(*, corpus_name: str, **_kwargs: object) -> None:
    attempted_corpora.append(corpus_name)
    if corpus_name == "second":
      raise RuntimeError("upload failed")

  fake_client = mocker.Mock()
  fake_client.aclose = mocker.AsyncMock()
  fake_client.rag.upload_file = mocker.AsyncMock(side_effect=upload_file)
  mocker.patch(
      "agentplatform.Client", return_value=mocker.Mock(aio=fake_client)
  )
  memory_service = VertexAiRagMemoryService(rag_corpus="first")
  memory_service._vertex_rag_store.rag_resources = [  # pylint: disable=protected-access
      types.VertexRagStoreRagResource(rag_corpus="first"),
      types.VertexRagStoreRagResource(rag_corpus="second"),
      types.VertexRagStoreRagResource(rag_corpus="third"),
  ]

  with pytest.raises(RuntimeError, match="upload failed"):
    await memory_service.add_session_to_memory(_session())

  assert attempted_corpora == ["first", "second"]
  assert not list(temp_dir.iterdir())
  fake_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_add_session_cleans_temp_file_when_cancelled(mocker, temp_dir):
  upload_started = asyncio.Event()

  async def upload_file(**_kwargs: object) -> None:
    upload_started.set()
    await asyncio.Event().wait()

  fake_client = mocker.Mock()
  fake_client.aclose = mocker.AsyncMock()
  fake_client.rag.upload_file = mocker.AsyncMock(side_effect=upload_file)
  mocker.patch(
      "agentplatform.Client", return_value=mocker.Mock(aio=fake_client)
  )
  memory_service = VertexAiRagMemoryService(rag_corpus="corpus")
  add_session = asyncio.create_task(
      memory_service.add_session_to_memory(_session())
  )
  await upload_started.wait()

  add_session.cancel()
  with pytest.raises(asyncio.CancelledError):
    await add_session

  assert not list(temp_dir.iterdir())
  fake_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_add_session_cleans_temp_file_when_close_fails(mocker, temp_dir):
  fake_client = mocker.Mock()
  fake_client.aclose = mocker.AsyncMock(
      side_effect=RuntimeError("close failed")
  )
  fake_client.rag.upload_file = mocker.AsyncMock()
  mocker.patch(
      "agentplatform.Client", return_value=mocker.Mock(aio=fake_client)
  )
  memory_service = VertexAiRagMemoryService(rag_corpus="corpus")

  with pytest.raises(RuntimeError, match="close failed"):
    await memory_service.add_session_to_memory(_session())

  assert not list(temp_dir.iterdir())


@pytest.mark.asyncio
async def test_add_session_finishes_close_cancelled_mid_flight(
    mocker, temp_dir
):
  upload_started = asyncio.Event()
  close = _StallingClose()

  async def upload_file(**_kwargs: object) -> None:
    upload_started.set()
    await asyncio.Event().wait()

  fake_client = mocker.Mock()
  fake_client.aclose = close.aclose
  fake_client.rag.upload_file = upload_file
  mocker.patch(
      "agentplatform.Client", return_value=mocker.Mock(aio=fake_client)
  )
  memory_service = VertexAiRagMemoryService(rag_corpus="corpus")
  add_session = asyncio.create_task(
      memory_service.add_session_to_memory(_session())
  )
  await upload_started.wait()

  await _cancel_while_closing(add_session, close)

  assert close.completed
  assert not list(temp_dir.iterdir())


@pytest.mark.asyncio
async def test_search_memory_finishes_close_cancelled_mid_flight(mocker):
  retrieve_started = asyncio.Event()
  close = _StallingClose()

  async def retrieve_contexts(**_kwargs: object) -> None:
    retrieve_started.set()
    await asyncio.Event().wait()

  fake_client = mocker.Mock()
  fake_client.aclose = close.aclose
  fake_client.rag.retrieve_contexts = retrieve_contexts
  mocker.patch(
      "agentplatform.Client", return_value=mocker.Mock(aio=fake_client)
  )
  memory_service = VertexAiRagMemoryService(rag_corpus="corpus")
  search = asyncio.create_task(
      memory_service.search_memory(app_name="demo", user_id="alice", query="q")
  )
  await retrieve_started.wait()

  await _cancel_while_closing(search, close)

  assert close.completed


@pytest.mark.asyncio
async def test_add_session_leaves_no_temp_file_when_corpus_missing(temp_dir):
  memory_service = VertexAiRagMemoryService(rag_corpus=None)

  with pytest.raises(ValueError, match="rag_corpus must be set"):
    await memory_service.add_session_to_memory(_session())

  assert not list(temp_dir.iterdir())
