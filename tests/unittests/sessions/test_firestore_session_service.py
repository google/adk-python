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

"""Unit tests for FirestoreSessionService.

All Firestore interactions are mocked so no real Firestore connection is
needed.
"""

from __future__ import annotations

import copy
from typing import Any
from unittest import mock

from google.adk.errors.already_exists_error import AlreadyExistsError
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.base_session_service import GetSessionConfig
from google.genai import types
import pytest

# ---------------------------------------------------------------------------
# Helpers – lightweight in-memory Firestore mock
# ---------------------------------------------------------------------------


class _FakeDocSnapshot:
  """Mimics a Firestore DocumentSnapshot."""

  def __init__(self, doc_id: str, data: dict[str, Any] | None):
    self.id = doc_id
    self._data = data
    self.exists = data is not None
    self.reference = mock.AsyncMock()
    self.reference.delete = mock.AsyncMock()

  def to_dict(self) -> dict[str, Any]:
    return copy.deepcopy(self._data) if self._data else {}


class _FakeDocRef:
  """Mimics an async Firestore DocumentReference."""

  def __init__(self, doc_id: str, store: dict[str, dict[str, Any]]):
    self.id = doc_id
    self._store = store
    self._subcollections: dict[str, _FakeCollection] = {}

  async def get(self):
    data = self._store.get(self.id)
    return _FakeDocSnapshot(self.id, copy.deepcopy(data) if data else None)

  async def set(self, data, merge=False):
    if merge and self.id in self._store:
      self._store[self.id].update(data)
    else:
      self._store[self.id] = copy.deepcopy(data)

  async def update(self, data):
    if self.id in self._store:
      self._store[self.id].update(data)

  async def delete(self):
    self._store.pop(self.id, None)

  def collection(self, name: str) -> _FakeCollection:
    if name not in self._subcollections:
      self._subcollections[name] = _FakeCollection({})
    return self._subcollections[name]


class _FakeCollection:
  """Mimics an async Firestore CollectionReference backed by a dict."""

  def __init__(self, store: dict[str, dict[str, Any]] | None = None):
    self._store: dict[str, dict[str, Any]] = store if store is not None else {}
    # Persistent doc refs so subcollections survive across calls
    self._doc_refs: dict[str, _FakeDocRef] = {}

  def document(self, doc_id: str) -> _FakeDocRef:
    if doc_id not in self._doc_refs:
      self._doc_refs[doc_id] = _FakeDocRef(doc_id, self._store)
    return self._doc_refs[doc_id]

  def where(self, **kwargs):
    return _FakeQuery(self._store, kwargs.get("filter"))

  def order_by(self, field):
    return _FakeQuery(self._store, None, order_field=field)

  async def stream(self):
    for doc_id, data in list(self._store.items()):
      snapshot = _FakeDocSnapshot(doc_id, copy.deepcopy(data))
      snapshot.reference = self.document(doc_id)
      yield snapshot


class _FakeQuery:
  """Mimics an async Firestore query."""

  def __init__(self, store, filt=None, order_field=None):
    self._store = store
    self._filters: list = []
    if filt:
      self._filters.append(filt)
    self._order_field = order_field

  def where(self, **kwargs):
    new_q = _FakeQuery(self._store, order_field=self._order_field)
    new_q._filters = list(self._filters)
    filt = kwargs.get("filter")
    if filt:
      new_q._filters.append(filt)
    return new_q

  def order_by(self, field):
    self._order_field = field
    return self

  def _matches(self, data: dict) -> bool:
    for f in self._filters:
      field_path = f.field_path
      op = f.op_string
      val = f.value
      actual = data.get(field_path)
      if op == "==" and actual != val:
        return False
      if op == ">=" and (actual is None or actual < val):
        return False
    return True

  async def stream(self):
    items = [
        (doc_id, copy.deepcopy(data))
        for doc_id, data in self._store.items()
        if self._matches(data)
    ]
    if self._order_field:
      items.sort(key=lambda x: x[1].get(self._order_field, 0))
    for doc_id, data in items:
      yield _FakeDocSnapshot(doc_id, data)


class _FakeFieldFilter:

  def __init__(self, field_path, op_string, value):
    self.field_path = field_path
    self.op_string = op_string
    self.value = value


class _FakeFirestoreClient:
  """Mimics google.cloud.firestore_v1.AsyncClient."""

  def __init__(self):
    self._collections: dict[str, _FakeCollection] = {}

  def collection(self, name: str) -> _FakeCollection:
    if name not in self._collections:
      self._collections[name] = _FakeCollection({})
    return self._collections[name]

  @staticmethod
  def field_filter(field_path, op_string, value):
    return _FakeFieldFilter(field_path, op_string, value)

  def close(self):
    pass


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def firestore_service():
  """Creates a FirestoreSessionService with a mocked Firestore client."""
  fake_client = _FakeFirestoreClient()

  with mock.patch.dict(
      "sys.modules",
      {
          "google.cloud.firestore_v1": mock.MagicMock(
              AsyncClient=lambda **kwargs: fake_client
          ),
      },
  ):
    from google.adk.sessions.firestore_session_service import FirestoreSessionService

    service = FirestoreSessionService(project="test-project")
    # Replace the client with our fake
    service._db = fake_client
    return service


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

APP = "test_app"
USER = "test_user"


@pytest.mark.asyncio
async def test_create_session(firestore_service):
  session = await firestore_service.create_session(app_name=APP, user_id=USER)
  assert session.app_name == APP
  assert session.user_id == USER
  assert session.id  # auto-generated UUID
  assert session.last_update_time > 0


@pytest.mark.asyncio
async def test_create_session_with_custom_id(firestore_service):
  session = await firestore_service.create_session(
      app_name=APP, user_id=USER, session_id="custom-123"
  )
  assert session.id == "custom-123"


@pytest.mark.asyncio
async def test_create_session_duplicate_id_raises(firestore_service):
  await firestore_service.create_session(
      app_name=APP, user_id=USER, session_id="dup-id"
  )
  with pytest.raises(AlreadyExistsError):
    await firestore_service.create_session(
        app_name=APP, user_id=USER, session_id="dup-id"
    )


@pytest.mark.asyncio
async def test_create_session_with_state(firestore_service):
  state = {
      "app:theme": "dark",
      "user:lang": "en",
      "counter": 0,
  }
  session = await firestore_service.create_session(
      app_name=APP, user_id=USER, state=state
  )
  assert session.state.get("app:theme") == "dark"
  assert session.state.get("user:lang") == "en"
  assert session.state.get("counter") == 0


@pytest.mark.asyncio
async def test_get_session(firestore_service):
  created = await firestore_service.create_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  retrieved = await firestore_service.get_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  assert retrieved is not None
  assert retrieved.id == "s1"
  assert retrieved.app_name == APP


@pytest.mark.asyncio
async def test_get_session_nonexistent_returns_none(firestore_service):
  result = await firestore_service.get_session(
      app_name=APP, user_id=USER, session_id="nonexistent"
  )
  assert result is None


@pytest.mark.asyncio
async def test_get_session_wrong_user_returns_none(firestore_service):
  await firestore_service.create_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  result = await firestore_service.get_session(
      app_name=APP, user_id="other_user", session_id="s1"
  )
  assert result is None


@pytest.mark.asyncio
async def test_list_sessions(firestore_service):
  await firestore_service.create_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  await firestore_service.create_session(
      app_name=APP, user_id=USER, session_id="s2"
  )
  await firestore_service.create_session(
      app_name=APP, user_id="other", session_id="s3"
  )

  # List for specific user
  response = await firestore_service.list_sessions(app_name=APP, user_id=USER)
  assert len(response.sessions) == 2

  # List all sessions for app
  response_all = await firestore_service.list_sessions(app_name=APP)
  assert len(response_all.sessions) == 3


@pytest.mark.asyncio
async def test_delete_session(firestore_service):
  await firestore_service.create_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  await firestore_service.delete_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  result = await firestore_service.get_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  assert result is None


@pytest.mark.asyncio
async def test_delete_nonexistent_session_is_noop(firestore_service):
  # Should not raise
  await firestore_service.delete_session(
      app_name=APP, user_id=USER, session_id="nonexistent"
  )


@pytest.mark.asyncio
async def test_append_event(firestore_service):
  session = await firestore_service.create_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  event = Event(
      invocation_id="inv-1",
      author="user",
      content=types.Content(role="user", parts=[types.Part(text="Hello")]),
  )
  result = await firestore_service.append_event(session, event)
  assert result.id == event.id

  # Verify event is retrievable
  retrieved = await firestore_service.get_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  assert retrieved is not None
  assert len(retrieved.events) == 1


@pytest.mark.asyncio
async def test_append_event_with_state_delta(firestore_service):
  session = await firestore_service.create_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  event = Event(
      invocation_id="inv-1",
      author="agent",
      actions=EventActions(
          state_delta={"counter": 42, "app:global_key": "val"}
      ),
  )
  await firestore_service.append_event(session, event)

  retrieved = await firestore_service.get_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  assert retrieved is not None
  assert retrieved.state.get("counter") == 42
  assert retrieved.state.get("app:global_key") == "val"


@pytest.mark.asyncio
async def test_append_partial_event_skipped(firestore_service):
  session = await firestore_service.create_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  event = Event(
      invocation_id="inv-1",
      author="agent",
      partial=True,
      content=types.Content(role="model", parts=[types.Part(text="partial")]),
  )
  result = await firestore_service.append_event(session, event)
  assert result.partial is True

  retrieved = await firestore_service.get_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  assert retrieved is not None
  assert len(retrieved.events) == 0


@pytest.mark.asyncio
async def test_get_session_with_num_recent_events(firestore_service):
  session = await firestore_service.create_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  for i in range(5):
    event = Event(
        invocation_id=f"inv-{i}",
        author="user",
        content=types.Content(role="user", parts=[types.Part(text=f"msg-{i}")]),
    )
    await firestore_service.append_event(session, event)

  config = GetSessionConfig(num_recent_events=2)
  retrieved = await firestore_service.get_session(
      app_name=APP, user_id=USER, session_id="s1", config=config
  )
  assert retrieved is not None
  assert len(retrieved.events) == 2


@pytest.mark.asyncio
async def test_app_state_shared_across_sessions(firestore_service):
  state = {"app:shared": "value"}
  await firestore_service.create_session(
      app_name=APP, user_id=USER, session_id="s1", state=state
  )
  s2 = await firestore_service.create_session(
      app_name=APP, user_id=USER, session_id="s2"
  )
  assert s2.state.get("app:shared") == "value"


@pytest.mark.asyncio
async def test_user_state_shared_across_sessions(firestore_service):
  state = {"user:pref": "compact"}
  await firestore_service.create_session(
      app_name=APP, user_id=USER, session_id="s1", state=state
  )
  s2 = await firestore_service.create_session(
      app_name=APP, user_id=USER, session_id="s2"
  )
  assert s2.state.get("user:pref") == "compact"


@pytest.mark.asyncio
async def test_temp_state_not_persisted(firestore_service):
  session = await firestore_service.create_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  event = Event(
      invocation_id="inv-1",
      author="agent",
      actions=EventActions(
          state_delta={"temp:scratch": "tmp", "keep_me": "yes"}
      ),
  )
  await firestore_service.append_event(session, event)

  retrieved = await firestore_service.get_session(
      app_name=APP, user_id=USER, session_id="s1"
  )
  assert retrieved is not None
  assert "temp:scratch" not in retrieved.state
  assert retrieved.state.get("keep_me") == "yes"
