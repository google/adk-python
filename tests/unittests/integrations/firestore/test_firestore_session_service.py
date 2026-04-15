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

from __future__ import annotations

from unittest import mock

from google.adk.events.event import Event
from google.adk.integrations.firestore.firestore_session_service import FirestoreSessionService
import pytest


@pytest.fixture
def mock_firestore_client():
  client = mock.MagicMock()
  collection_ref = mock.MagicMock()
  doc_ref = mock.MagicMock()
  subcollection_ref = mock.MagicMock()
  subdoc_ref = mock.MagicMock()
  sessions_coll_ref = mock.MagicMock()
  sessions_doc_ref = mock.MagicMock()

  client.collection.return_value = collection_ref
  collection_ref.document.return_value = doc_ref
  doc_ref.collection.return_value = subcollection_ref
  subcollection_ref.document.return_value = subdoc_ref
  subdoc_ref.collection.return_value = sessions_coll_ref
  sessions_coll_ref.document.return_value = sessions_doc_ref

  doc_snapshot = mock.MagicMock()
  doc_snapshot.exists = False
  doc_snapshot.to_dict.return_value = {}

  subdoc_ref.get = mock.AsyncMock(return_value=doc_snapshot)
  sessions_doc_ref.get = mock.AsyncMock(return_value=doc_snapshot)
  doc_ref.get = mock.AsyncMock(return_value=doc_snapshot)

  sessions_doc_ref.set = mock.AsyncMock()
  sessions_doc_ref.delete = mock.AsyncMock()

  events_collection_ref = mock.MagicMock()
  sessions_doc_ref.collection.return_value = events_collection_ref
  events_collection_ref.order_by.return_value = events_collection_ref
  events_collection_ref.where.return_value = events_collection_ref
  events_collection_ref.limit_to_last.return_value = events_collection_ref
  events_collection_ref.get = mock.AsyncMock(return_value=[])

  sessions_coll_ref.get = mock.AsyncMock(return_value=[])
  sessions_coll_ref.where.return_value = sessions_coll_ref

  client.collection_group.return_value = collection_ref

  batch = mock.MagicMock()
  client.batch.return_value = batch
  batch.commit = mock.AsyncMock()

  return client


def test_init_missing_dependency():
  import builtins

  original_import = builtins.__import__

  def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "google.cloud" and "firestore" in fromlist:
      raise ImportError("Mocked import error")
    return original_import(name, globals, locals, fromlist, level)

  with mock.patch("builtins.__import__", side_effect=mock_import):
    with pytest.raises(ImportError, match="requires google-cloud-firestore"):
      FirestoreSessionService()


@pytest.mark.asyncio
async def test_create_session(mock_firestore_client):

  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"

  with mock.patch("google.cloud.firestore.async_transactional", lambda x: x):
    session = await service.create_session(app_name=app_name, user_id=user_id)

  assert session.app_name == app_name
  assert session.user_id == user_id
  assert session.id

  mock_firestore_client.collection.assert_any_call("adk-session")
  mock_firestore_client.collection.assert_any_call("app_states")
  mock_firestore_client.collection.assert_any_call("user_states")

  root_coll = mock_firestore_client.collection.return_value
  app_ref = root_coll.document.return_value
  users_coll = app_ref.collection.return_value
  user_ref = users_coll.document.return_value
  sessions_ref = user_ref.collection.return_value
  session_doc_ref = sessions_ref.document.return_value

  from google.cloud import firestore

  transaction = mock_firestore_client.transaction.return_value
  transaction.set.assert_called_once()
  args, kwargs = transaction.set.call_args
  assert args[0] == session_doc_ref
  assert args[1]["id"] == session.id
  assert args[1]["appName"] == app_name
  assert args[1]["userId"] == user_id
  assert args[1]["state"] == {}
  assert args[1]["createTime"] == firestore.SERVER_TIMESTAMP
  assert args[1]["updateTime"] == firestore.SERVER_TIMESTAMP


@pytest.mark.asyncio
async def test_get_session_not_found(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"
  session_id = "test_session"

  session = await service.get_session(
      app_name=app_name, user_id=user_id, session_id=session_id
  )

  assert session is None

  mock_firestore_client.collection.assert_called_with("adk-session")
  root_coll = mock_firestore_client.collection.return_value
  root_coll.document.assert_called_with(app_name)
  app_ref = root_coll.document.return_value
  app_ref.collection.assert_called_with("users")
  users_coll = app_ref.collection.return_value
  users_coll.document.assert_called_with(user_id)
  user_ref = users_coll.document.return_value
  user_ref.collection.assert_called_with("sessions")
  sessions_ref = user_ref.collection.return_value
  sessions_ref.document.assert_called_with(session_id)


@pytest.mark.asyncio
async def test_get_session_found(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"
  session_id = "test_session"

  root_coll = mock_firestore_client.collection.return_value
  app_ref = root_coll.document.return_value
  users_coll = app_ref.collection.return_value
  user_ref = users_coll.document.return_value
  sessions_ref = user_ref.collection.return_value
  sessions_doc_ref = sessions_ref.document.return_value

  session_snap = mock.MagicMock()
  session_snap.exists = True
  session_snap.to_dict.return_value = {
      "id": session_id,
      "appName": app_name,
      "userId": user_id,
      "state": {"key": "value"},
      "updateTime": 1234567890.0,
  }
  sessions_doc_ref.get.return_value = session_snap

  # Decouple app and user documents so they do not duplicate values
  app_state_coll = mock_firestore_client.collection.return_value
  app_doc_ref = app_state_coll.document.return_value
  app_snap = mock.MagicMock()
  app_snap.exists = False
  app_snap.to_dict.return_value = {}
  app_doc_ref.get.return_value = app_snap

  user_state_coll = mock_firestore_client.collection.return_value
  user_doc_ref = user_state_coll.document.return_value
  user_snap = mock.MagicMock()
  user_snap.exists = False
  user_snap.to_dict.return_value = {}
  user_doc_ref.get.return_value = user_snap

  events_collection_ref = (
      mock_firestore_client.collection.return_value.document.return_value.collection.return_value.document.return_value.collection.return_value.document.return_value.collection.return_value
  )
  event_doc = mock.MagicMock()
  event_doc.to_dict.return_value = {
      "event_data": {"invocation_id": "test_inv", "author": "user"}
  }
  events_collection_ref.get = mock.AsyncMock(return_value=[event_doc])

  session = await service.get_session(
      app_name=app_name, user_id=user_id, session_id=session_id
  )

  assert session is not None
  assert session.id == session_id
  assert session.state == {"key": "value"}
  assert len(session.events) == 1
  assert session.events[0].invocation_id == "test_inv"


@pytest.mark.asyncio
async def test_delete_session(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"
  session_id = "test_session"

  events_ref = (
      mock_firestore_client.collection.return_value.document.return_value.collection.return_value.document.return_value.collection.return_value.document.return_value.collection.return_value
  )
  event_doc = mock.AsyncMock()

  async def to_async_iter(iterable):
    for item in iterable:
      yield item

  events_ref.stream.return_value = to_async_iter([event_doc])

  await service.delete_session(
      app_name=app_name, user_id=user_id, session_id=session_id
  )

  events_ref.stream.assert_called_once()
  mock_firestore_client.batch.assert_called_once()
  batch = mock_firestore_client.batch.return_value
  batch.delete.assert_called_once_with(event_doc.reference)
  batch.commit.assert_called_once()

  session_doc_ref = (
      mock_firestore_client.collection.return_value.document.return_value.collection.return_value.document.return_value.collection.return_value.document.return_value
  )
  session_doc_ref.delete.assert_called_once()


@pytest.mark.asyncio
async def test_append_event(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"
  from google.adk.sessions.session import Session

  session = Session(id="test_session", app_name=app_name, user_id=user_id)
  event = Event(invocation_id="test_inv", author="user")

  session_doc_snapshot = mock.MagicMock()
  session_doc_snapshot.exists = True
  session_doc_snapshot.to_dict.return_value = {"revision": 0}

  root_coll = mock_firestore_client.collection.return_value
  app_ref = root_coll.document.return_value
  users_coll = app_ref.collection.return_value
  user_ref = users_coll.document.return_value
  sessions_ref = user_ref.collection.return_value
  session_doc_ref = sessions_ref.document.return_value
  session_doc_ref.get = mock.AsyncMock(return_value=session_doc_snapshot)

  with mock.patch("google.cloud.firestore.async_transactional", lambda x: x):
    await service.append_event(session, event)

  from google.cloud import firestore

  transaction = mock_firestore_client.transaction.return_value
  transaction.set.assert_called()  # Invoked for events appends
  transaction.update.assert_called_once()  # Invoked for session revisions

  args, kwargs = transaction.update.call_args
  assert args[1]["revision"] == 1
  assert args[1]["updateTime"] == firestore.SERVER_TIMESTAMP


@pytest.mark.asyncio
async def test_append_event_with_state_delta(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"
  from google.adk.sessions.session import Session

  session = Session(id="test_session", app_name=app_name, user_id=user_id)

  event = mock.MagicMock()
  event.partial = False
  event.id = "test_event_id"
  event.actions.state_delta = {
      "_app_my_key": "app_val",
      "_user_my_key": "user_val",
      "session_key": "session_val",
  }
  event.model_dump.return_value = {"id": "test_event_id", "author": "user"}

  service._update_app_state_transactional = mock.AsyncMock()
  service._update_user_state_transactional = mock.AsyncMock()

  session_doc_snapshot = mock.MagicMock()
  session_doc_snapshot.exists = True
  session_doc_snapshot.to_dict.return_value = {"revision": 0}

  root_coll = mock_firestore_client.collection.return_value
  app_ref = root_coll.document.return_value
  users_coll = app_ref.collection.return_value
  user_ref = users_coll.document.return_value
  sessions_ref = user_ref.collection.return_value
  session_doc_ref = sessions_ref.document.return_value
  session_doc_ref.get = mock.AsyncMock(return_value=session_doc_snapshot)

  with mock.patch("google.cloud.firestore.async_transactional", lambda x: x):
    await service.append_event(session, event)

  transaction = mock_firestore_client.transaction.return_value
  transaction.set.assert_called()

  assert session.state["session_key"] == "session_val"

  from google.cloud import firestore

  transaction.update.assert_called_once()
  args, kwargs = transaction.update.call_args
  # In modular Firestore configurations alignments, updating variables mock assertions core setups
  assert args[1]["state"] == session.state
  assert args[1]["updateTime"] == firestore.SERVER_TIMESTAMP


@pytest.mark.asyncio
async def test_append_event_with_temp_state(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"
  from google.adk.events.event import Event
  from google.adk.events.event import EventActions
  from google.adk.sessions.session import Session

  session = Session(id="test_session", app_name=app_name, user_id=user_id)

  event = Event(
      invocation_id="test_inv",
      author="user",
      actions=EventActions(
          state_delta={"temp:k1": "v1", "session_key": "session_val"}
      ),
  )

  session_doc_snapshot = mock.MagicMock()
  session_doc_snapshot.exists = True
  session_doc_snapshot.to_dict.return_value = {"revision": 0}

  root_coll = mock_firestore_client.collection.return_value
  app_ref = root_coll.document.return_value
  users_coll = app_ref.collection.return_value
  user_ref = users_coll.document.return_value
  sessions_ref = user_ref.collection.return_value
  session_doc_ref = sessions_ref.document.return_value
  session_doc_ref.get = mock.AsyncMock(return_value=session_doc_snapshot)

  with mock.patch("google.cloud.firestore.async_transactional", lambda x: x):
    await service.append_event(session, event)

  # 1. Verify it was applied in-memory
  assert session.state["temp:k1"] == "v1"
  assert session.state["session_key"] == "session_val"

  # 2. Verify it was trimmed before Firestore save
  transaction = mock_firestore_client.transaction.return_value
  transaction.set.assert_called()

  # Filter calls for the one that actually sets the event data
  event_set_calls = [
      call
      for call in transaction.set.call_args_list
      if len(call[0]) > 1
      and isinstance(call[0][1], dict)
      and "event_data" in call[0][1]
  ]
  assert len(event_set_calls) == 1
  event_data = event_set_calls[0][0][1]["event_data"]

  # Temporary keys should be deleted from delta before snapshot
  assert "temp:k1" not in event_data["actions"]["state_delta"]
  assert event_data["actions"]["state_delta"]["session_key"] == "session_val"


@pytest.mark.asyncio
async def test_list_sessions_with_user_id(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"

  session_doc = mock.MagicMock()
  session_doc.to_dict.return_value = {
      "id": "session1",
      "appName": app_name,
      "userId": user_id,
      "state": {"session_key": "session_val"},
  }

  app_state_coll = mock.MagicMock()
  user_state_coll = mock.MagicMock()
  sessions_coll = mock.MagicMock()

  def collection_side_effect(name):
    if name == service.app_state_collection:
      return app_state_coll
    elif name == service.user_state_collection:
      return user_state_coll
    elif name == service.root_collection:
      return sessions_coll
    return mock.MagicMock()

  mock_firestore_client.collection.side_effect = collection_side_effect

  app_doc = mock.MagicMock()
  app_doc.exists = True
  app_doc.to_dict.return_value = {"app_key": "app_val"}
  app_doc_ref = mock.MagicMock()
  app_state_coll.document.return_value = app_doc_ref
  app_doc_ref.get = mock.AsyncMock(return_value=app_doc)

  user_doc = mock.MagicMock()
  user_doc.exists = True
  user_doc.to_dict.return_value = {"user_key": "user_val"}
  user_app_doc = mock.MagicMock()
  user_state_coll.document.return_value = user_app_doc
  users_coll = mock.MagicMock()
  user_app_doc.collection.return_value = users_coll
  user_doc_ref = mock.MagicMock()
  users_coll.document.return_value = user_doc_ref
  user_doc_ref.get = mock.AsyncMock(return_value=user_doc)

  app_doc_in_root = mock.MagicMock()
  sessions_coll.document.return_value = app_doc_in_root
  users_coll = mock.MagicMock()
  app_doc_in_root.collection.return_value = users_coll
  user_doc_in_users = mock.MagicMock()
  users_coll.document.return_value = user_doc_in_users
  sessions_subcoll = mock.MagicMock()
  user_doc_in_users.collection.return_value = sessions_subcoll
  sessions_query = mock.MagicMock()
  sessions_subcoll.where.return_value = sessions_query
  sessions_query.get = mock.AsyncMock(return_value=[session_doc])

  response = await service.list_sessions(app_name=app_name, user_id=user_id)

  assert len(response.sessions) == 1
  session = response.sessions[0]
  assert session.id == "session1"
  assert session.state["session_key"] == "session_val"
  assert session.state["app:app_key"] == "app_val"
  assert session.state["user:user_key"] == "user_val"


@pytest.mark.asyncio
async def test_list_sessions_without_user_id(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"

  session_doc = mock.MagicMock()
  session_doc.to_dict.return_value = {
      "id": "session1",
      "appName": app_name,
      "userId": "user1",
      "state": {"session_key": "session_val"},
  }

  mock_firestore_client.collection_group.return_value.where.return_value.get = (
      mock.AsyncMock(return_value=[session_doc])
  )

  app_state_coll = mock.MagicMock()
  user_state_coll = mock.MagicMock()

  def collection_side_effect(name):
    if name == service.app_state_collection:
      return app_state_coll
    elif name == service.user_state_collection:
      return user_state_coll
    return mock.MagicMock()

  mock_firestore_client.collection.side_effect = collection_side_effect

  app_doc = mock.MagicMock()
  app_doc.exists = True
  app_doc.to_dict.return_value = {"app_key": "app_val"}
  app_doc_ref = mock.MagicMock()
  app_state_coll.document.return_value = app_doc_ref
  app_doc_ref.get = mock.AsyncMock(return_value=app_doc)

  user_doc = mock.MagicMock()
  user_doc.id = "user1"
  user_doc.to_dict.return_value = {"user_key": "user_val"}
  user_app_doc = mock.MagicMock()
  user_state_coll.document.return_value = user_app_doc
  users_coll = mock.MagicMock()
  user_app_doc.collection.return_value = users_coll
  users_coll.get = mock.AsyncMock(return_value=[user_doc])

  response = await service.list_sessions(app_name=app_name)

  assert len(response.sessions) == 1
  session = response.sessions[0]
  assert session.id == "session1"
  assert session.state["app:app_key"] == "app_val"
  assert session.state["user:user_key"] == "user_val"

  mock_firestore_client.collection_group.assert_called_once_with("sessions")
  mock_firestore_client.collection_group.return_value.where.assert_called_once_with(
      "appName", "==", app_name
  )


@pytest.mark.asyncio
async def test_list_sessions_filters_other_apps(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"

  session_doc = mock.MagicMock()
  session_doc.to_dict.return_value = {
      "id": "session1",
      "appName": app_name,
      "userId": "user1",
      "state": {"session_key": "session_val"},
  }

  mock_firestore_client.collection_group.return_value.where.return_value.get = (
      mock.AsyncMock(return_value=[session_doc])
  )

  app_state_coll = mock.MagicMock()
  user_state_coll = mock.MagicMock()

  def collection_side_effect(name):
    if name == service.app_state_collection:
      return app_state_coll
    elif name == service.user_state_collection:
      return user_state_coll
    return mock.MagicMock()

  mock_firestore_client.collection.side_effect = collection_side_effect

  app_doc = mock.MagicMock()
  app_doc.exists = True
  app_doc.to_dict.return_value = {"app_key": "app_val"}
  app_doc_ref = mock.MagicMock()
  app_state_coll.document.return_value = app_doc_ref
  app_doc_ref.get = mock.AsyncMock(return_value=app_doc)

  user_doc = mock.MagicMock()
  user_doc.id = "user1"
  user_doc.to_dict.return_value = {"user_key": "user_val"}
  user_app_doc = mock.MagicMock()
  user_state_coll.document.return_value = user_app_doc
  users_coll = mock.MagicMock()
  user_app_doc.collection.return_value = users_coll
  users_coll.get = mock.AsyncMock(return_value=[user_doc])

  response = await service.list_sessions(app_name=app_name)

  assert len(response.sessions) == 1
  assert response.sessions[0].id == "session1"
  assert response.sessions[0].app_name == app_name

  mock_firestore_client.collection_group.assert_called_once_with("sessions")
  mock_firestore_client.collection_group.return_value.where.assert_called_once_with(
      "appName", "==", app_name
  )


@pytest.mark.asyncio
async def test_create_session_already_exists(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"

  doc_snapshot = (
      mock_firestore_client.collection.return_value.document.return_value.collection.return_value.document.return_value.get.return_value
  )
  doc_snapshot.exists = True

  from google.adk.errors.already_exists_error import AlreadyExistsError

  with mock.patch("google.cloud.firestore.async_transactional", lambda x: x):
    with pytest.raises(AlreadyExistsError):
      await service.create_session(
          app_name=app_name, user_id=user_id, session_id="existing_id"
      )


@pytest.mark.asyncio
async def test_get_session_with_config(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"
  session_id = "test_session"

  doc_snapshot = (
      mock_firestore_client.collection.return_value.document.return_value.collection.return_value.document.return_value.get.return_value
  )
  doc_snapshot.exists = True
  doc_snapshot.to_dict.return_value = {
      "id": session_id,
      "appName": app_name,
      "userId": user_id,
  }

  events_collection_ref = (
      mock_firestore_client.collection.return_value.document.return_value.collection.return_value.document.return_value.collection.return_value.document.return_value.collection.return_value
  )

  from google.adk.sessions.base_session_service import GetSessionConfig

  config = GetSessionConfig(after_timestamp=1234567890.0, num_recent_events=5)

  await service.get_session(
      app_name=app_name, user_id=user_id, session_id=session_id, config=config
  )

  events_collection_ref.where.assert_called_once()
  events_collection_ref.limit_to_last.assert_called_once_with(5)


@pytest.mark.asyncio
async def test_delete_session_batching(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"
  session_id = "test_session"

  events_ref = (
      mock_firestore_client.collection.return_value.document.return_value.collection.return_value.document.return_value.collection.return_value.document.return_value.collection.return_value
  )

  dummy_docs = [mock.MagicMock() for _ in range(501)]

  async def to_async_iter(iterable):
    for item in iterable:
      yield item

  events_ref.stream.return_value = to_async_iter(dummy_docs)

  batch = mock_firestore_client.batch.return_value

  await service.delete_session(
      app_name=app_name, user_id=user_id, session_id=session_id
  )

  assert batch.commit.call_count == 2
  assert batch.delete.call_count == 501


@pytest.mark.asyncio
async def test_append_event_partial(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  from google.adk.sessions.session import Session

  session = Session(id="test_session", app_name="test_app", user_id="test_user")

  event = Event(invocation_id="test_inv", author="user", partial=True)

  result = await service.append_event(session, event)

  assert result == event
  mock_firestore_client.batch.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_get_session_empty_data(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"
  session_id = "test_session"

  doc_snapshot = (
      mock_firestore_client.collection.return_value.document.return_value.collection.return_value.document.return_value.get.return_value
  )
  doc_snapshot.exists = True
  doc_snapshot.to_dict.return_value = {}

  session = await service.get_session(
      app_name=app_name, user_id=user_id, session_id=session_id
  )

  assert session is None


@pytest.mark.asyncio
async def test_list_sessions_missing_states(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"

  session_doc = mock.MagicMock()
  session_doc.to_dict.return_value = {
      "id": "session1",
      "appName": app_name,
      "userId": user_id,
      "state": {"session_key": "session_val"},
  }

  app_state_coll = mock.MagicMock()
  user_state_coll = mock.MagicMock()
  sessions_coll = mock.MagicMock()

  def collection_side_effect(name):
    if name == service.app_state_collection:
      return app_state_coll
    elif name == service.user_state_collection:
      return user_state_coll
    elif name == service.root_collection:
      return sessions_coll
    return mock.MagicMock()

  mock_firestore_client.collection.side_effect = collection_side_effect

  app_doc = mock.MagicMock()
  app_doc.exists = False
  app_doc_ref = mock.MagicMock()
  app_state_coll.document.return_value = app_doc_ref
  app_doc_ref.get = mock.AsyncMock(return_value=app_doc)

  user_doc = mock.MagicMock()
  user_doc.exists = False
  user_app_doc = mock.MagicMock()
  user_state_coll.document.return_value = user_app_doc
  users_coll = mock.MagicMock()
  user_app_doc.collection.return_value = users_coll
  user_doc_ref = mock.MagicMock()
  users_coll.document.return_value = user_doc_ref
  user_doc_ref.get = mock.AsyncMock(return_value=user_doc)

  app_doc_in_root = mock.MagicMock()
  sessions_coll.document.return_value = app_doc_in_root
  users_coll = mock.MagicMock()
  app_doc_in_root.collection.return_value = users_coll
  user_doc_in_users = mock.MagicMock()
  users_coll.document.return_value = user_doc_in_users
  sessions_subcoll = mock.MagicMock()
  user_doc_in_users.collection.return_value = sessions_subcoll
  sessions_query = mock.MagicMock()
  sessions_subcoll.where.return_value = sessions_query
  sessions_query.get = mock.AsyncMock(return_value=[session_doc])

  response = await service.list_sessions(app_name=app_name, user_id=user_id)

  assert len(response.sessions) == 1
  session = response.sessions[0]
  assert session.id == "session1"
  assert session.state["session_key"] == "session_val"
  assert "_app_app_key" not in session.state
  assert "_user_user_key" not in session.state
