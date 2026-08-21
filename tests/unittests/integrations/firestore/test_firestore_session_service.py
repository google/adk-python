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

import contextlib
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import json
import os
import time
from unittest import mock

from google.adk.errors import StaleSessionError
from google.adk.errors.already_exists_error import AlreadyExistsError
from google.adk.errors.session_not_found_error import SessionNotFoundError
from google.adk.events.event import Event
from google.adk.events.event import EventActions
from google.adk.integrations.firestore.firestore_session_service import FirestoreSessionService
from google.adk.sessions.base_session_service import GetSessionConfig
from google.adk.sessions.session import Session
from google.cloud import firestore
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
  assert session._storage_update_marker == "0"

  mock_firestore_client.collection.assert_any_call("adk-session")
  mock_firestore_client.collection.assert_any_call("app_states")
  mock_firestore_client.collection.assert_any_call("user_states")

  root_coll = mock_firestore_client.collection.return_value
  app_ref = root_coll.document.return_value
  users_coll = app_ref.collection.return_value
  user_ref = users_coll.document.return_value
  sessions_ref = user_ref.collection.return_value
  session_doc_ref = sessions_ref.document.return_value

  transaction = mock_firestore_client.transaction.return_value
  transaction.set.assert_called_once()
  args, kwargs = transaction.set.call_args
  assert args[0] == session_doc_ref
  assert args[1]["id"] == session.id
  assert args[1]["appName"] == app_name
  assert args[1]["userId"] == user_id
  assert json.loads(args[1]["state"]) == {}
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
  assert session._storage_update_marker == "0"


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

  transaction = mock_firestore_client.transaction.return_value
  transaction.set.assert_called()  # Invoked for events appends
  transaction.update.assert_called_once()  # Invoked for session revisions

  args, kwargs = transaction.update.call_args
  assert args[1]["revision"] == 1
  assert args[1]["updateTime"] == firestore.SERVER_TIMESTAMP
  assert session.last_update_time == event.timestamp


@pytest.mark.asyncio
async def test_append_event_session_not_found(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  session = Session(id="test_session", app_name="test_app", user_id="test_user")
  event = Event(invocation_id="test_inv", author="user")

  session_doc_snapshot = mock.MagicMock()
  session_doc_snapshot.exists = False

  root_coll = mock_firestore_client.collection.return_value
  app_ref = root_coll.document.return_value
  users_coll = app_ref.collection.return_value
  user_ref = users_coll.document.return_value
  sessions_ref = user_ref.collection.return_value
  session_doc_ref = sessions_ref.document.return_value
  session_doc_ref.get = mock.AsyncMock(return_value=session_doc_snapshot)

  with mock.patch("google.cloud.firestore.async_transactional", lambda x: x):
    with pytest.raises(SessionNotFoundError):
      await service.append_event(session, event)


@pytest.mark.asyncio
async def test_append_event_rejects_stale_revision(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  session = Session(id="test_session", app_name="test_app", user_id="test_user")
  session._storage_update_marker = "0"
  event = Event(invocation_id="test_inv", author="user")

  session_doc_snapshot = mock.MagicMock()
  session_doc_snapshot.exists = True
  session_doc_snapshot.to_dict.return_value = {"revision": 1}
  session_doc_ref = (
      mock_firestore_client.collection.return_value.document.return_value.collection.return_value.document.return_value.collection.return_value.document.return_value
  )
  session_doc_ref.get = mock.AsyncMock(return_value=session_doc_snapshot)

  with mock.patch("google.cloud.firestore.async_transactional", lambda x: x):
    with pytest.raises(StaleSessionError, match="modified in storage"):
      await service.append_event(session, event)

  transaction = mock_firestore_client.transaction.return_value
  transaction.set.assert_not_called()
  transaction.update.assert_not_called()


@pytest.mark.asyncio
async def test_append_event_with_state_delta(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"
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

  transaction.update.assert_called_once()
  args, kwargs = transaction.update.call_args
  assert json.loads(args[1]["state"]) == session.state
  assert args[1]["updateTime"] == firestore.SERVER_TIMESTAMP


@pytest.mark.asyncio
async def test_append_event_repeated_non_serializable_state_delta(
    mock_firestore_client,
):
  """The raw delta the base class merges back must not break a later append."""
  service = FirestoreSessionService(client=mock_firestore_client)
  session = Session(id="test_session", app_name="test_app", user_id="test_user")

  session_doc_snapshot = mock.MagicMock()
  session_doc_snapshot.exists = True
  session_doc_snapshot.to_dict.side_effect = [{"revision": 0}, {"revision": 1}]

  root_coll = mock_firestore_client.collection.return_value
  app_ref = root_coll.document.return_value
  users_coll = app_ref.collection.return_value
  user_ref = users_coll.document.return_value
  sessions_ref = user_ref.collection.return_value
  session_doc_ref = sessions_ref.document.return_value
  session_doc_ref.get = mock.AsyncMock(return_value=session_doc_snapshot)

  # The second delta does not mention "callback", so the only copy of it left
  # for the second write is the raw one the base class merged into the session.
  deltas = [{"callback": lambda: 1, "ok": 2}, {"turn": 3}]

  with mock.patch("google.cloud.firestore.async_transactional", lambda x: x):
    for state_delta in deltas:
      await service.append_event(
          session,
          Event(
              invocation_id="test_inv",
              author="user",
              actions=EventActions(state_delta=state_delta),
          ),
      )

  transaction = mock_firestore_client.transaction.return_value
  assert transaction.update.call_count == 2
  persisted_state = json.loads(transaction.update.call_args[0][1]["state"])
  assert persisted_state["ok"] == 2
  assert persisted_state["turn"] == 3
  assert isinstance(persisted_state["callback"], str)


@pytest.mark.asyncio
async def test_append_event_keeps_app_and_user_state_native(
    mock_firestore_client,
):
  """App and user state are written natively, so a datetime stays a datetime."""
  service = FirestoreSessionService(client=mock_firestore_client)
  session = Session(id="test_session", app_name="test_app", user_id="test_user")
  when = datetime(2024, 6, 15, 10, 30, tzinfo=timezone.utc)

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
    await service.append_event(
        session,
        Event(
            invocation_id="test_inv",
            author="user",
            actions=EventActions(
                state_delta={
                    "app:started": when,
                    "user:seen": when,
                    "session_key": when,
                }
            ),
        ),
    )

  transaction = mock_firestore_client.transaction.return_value
  written = {
      call.args[0]: call.args[1] for call in transaction.set.call_args_list
  }
  assert written[app_ref]["started"] == when
  assert written[user_ref]["seen"] == when
  # The session bucket is json.dumps'd, so it is still coerced.
  persisted_state = json.loads(transaction.update.call_args[0][1]["state"])
  assert isinstance(persisted_state["session_key"], str)


@pytest.mark.asyncio
async def test_create_session_keeps_app_and_user_state_native(
    mock_firestore_client,
):
  """create_session writes app and user state without coercing it."""
  service = FirestoreSessionService(client=mock_firestore_client)
  when = datetime(2024, 6, 15, 10, 30, tzinfo=timezone.utc)

  with mock.patch("google.cloud.firestore.async_transactional", lambda x: x):
    await service.create_session(
        app_name="test_app",
        user_id="test_user",
        state={
            "app:started": when,
            "user:seen": when,
            "session_key": when,
        },
    )

  root_coll = mock_firestore_client.collection.return_value
  app_ref = root_coll.document.return_value
  users_coll = app_ref.collection.return_value
  user_ref = users_coll.document.return_value

  transaction = mock_firestore_client.transaction.return_value
  written = {
      call.args[0]: call.args[1] for call in transaction.set.call_args_list
  }
  assert written[app_ref]["started"] == when
  assert written[user_ref]["seen"] == when
  sessions_ref = user_ref.collection.return_value
  session_doc_ref = sessions_ref.document.return_value
  persisted_state = json.loads(written[session_doc_ref]["state"])
  assert isinstance(persisted_state["session_key"], str)


@pytest.mark.asyncio
async def test_append_event_with_temp_state(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"
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

  # 3. Verify temp keys are NOT written to session state in Firestore
  transaction.update.assert_called_once()
  update_args, _ = transaction.update.call_args
  persisted_state = update_args[1]["state"]
  assert (
      "temp:k1" not in persisted_state
  ), "temp: keys must not be persisted to Firestore session state"
  assert "session_key" in persisted_state


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
      "updateTime": 1234567890.0,
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
  assert session.last_update_time == 1234567890.0


@pytest.mark.asyncio
async def test_list_sessions_preserves_datetime_update_time(
    mock_firestore_client,
):
  """list_sessions converts datetime updateTime to last_update_time."""
  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"
  update_time = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)

  session_doc = mock.MagicMock()
  session_doc.to_dict.return_value = {
      "id": "session1",
      "appName": app_name,
      "userId": user_id,
      "state": {},
      "updateTime": update_time,
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
  app_doc.to_dict.return_value = {}
  app_doc_ref = mock.MagicMock()
  app_state_coll.document.return_value = app_doc_ref
  app_doc_ref.get = mock.AsyncMock(return_value=app_doc)

  user_doc = mock.MagicMock()
  user_doc.exists = False
  user_doc.to_dict.return_value = {}
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
  assert response.sessions[0].last_update_time == update_time.timestamp()


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
      "updateTime": 1234567890.0,
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
  user_doc.exists = True
  user_doc.to_dict.return_value = {"user_key": "user_val"}
  user_app_doc = mock.MagicMock()
  user_state_coll.document.return_value = user_app_doc
  users_coll = mock.MagicMock()
  user_app_doc.collection.return_value = users_coll
  user_doc_ref = mock.MagicMock()
  users_coll.document.return_value = user_doc_ref

  async def mock_get_all(refs):
    yield user_doc

  mock_firestore_client.get_all = mock_get_all

  response = await service.list_sessions(app_name=app_name)

  assert len(response.sessions) == 1
  session = response.sessions[0]
  assert session.id == "session1"
  assert session.state["app:app_key"] == "app_val"
  assert session.state["user:user_key"] == "user_val"
  assert session.last_update_time == 1234567890.0

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
  user_doc.exists = True
  user_doc.to_dict.return_value = {"user_key": "user_val"}
  user_app_doc = mock.MagicMock()
  user_state_coll.document.return_value = user_app_doc
  users_coll = mock.MagicMock()
  user_app_doc.collection.return_value = users_coll
  user_doc_ref = mock.MagicMock()
  users_coll.document.return_value = user_doc_ref

  async def mock_get_all(refs):
    yield user_doc

  mock_firestore_client.get_all = mock_get_all

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

  config = GetSessionConfig(after_timestamp=1234567890.0, num_recent_events=5)

  await service.get_session(
      app_name=app_name, user_id=user_id, session_id=session_id, config=config
  )

  events_collection_ref.where.assert_called_once()
  events_collection_ref.limit_to_last.assert_called_once_with(5)


@pytest.mark.asyncio
async def test_get_session_with_zero_recent_events(mock_firestore_client):
  """Requesting zero events returns none of them, not the whole transcript.

  Callers use a count of zero to check whether a session exists without
  paying for its history, so the events query must be skipped entirely.
  """
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
  stored_event_doc = mock.MagicMock()
  stored_event_doc.to_dict.return_value = {
      "event_data": (
          Event(invocation_id="inv", author="user").model_dump(mode="json")
      )
  }
  events_collection_ref.get = mock.AsyncMock(return_value=[stored_event_doc])

  session = await service.get_session(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      config=GetSessionConfig(num_recent_events=0),
  )

  assert session is not None
  assert session.events == []
  events_collection_ref.get.assert_not_called()
  events_collection_ref.limit_to_last.assert_not_called()


@contextlib.contextmanager
def _pinned_local_timezone(name: str):
  """Pins the process timezone for the duration of the block.

  ``time.tzset`` is POSIX-only, so on other platforms the block runs in the
  host zone instead. Restoring ``TZ`` without a second ``tzset`` would leave
  the C library pinned for the rest of the session, so both are undone.
  """
  if not hasattr(time, "tzset"):
    yield
    return
  previous = os.environ.get("TZ")
  os.environ["TZ"] = name
  time.tzset()
  try:
    yield
  finally:
    if previous is None:
      os.environ.pop("TZ", None)
    else:
      os.environ["TZ"] = previous
    time.tzset()


def _wire_epoch(value: datetime) -> float:
  """The epoch the Firestore client encodes for ``value``.

  The client reads a naive datetime as UTC rather than as local time, so a
  naive cursor lands on the wire shifted by the host's UTC offset.
  """
  if value.tzinfo is None:
    value = value.replace(tzinfo=timezone.utc)
  return value.timestamp()


@pytest.mark.asyncio
async def test_get_session_after_timestamp_cursor_is_utc_aware(
    mock_firestore_client,
):
  """The after_timestamp cursor must be an aware UTC datetime.

  Events are written with an aware UTC server timestamp, so a naive local
  cursor is compared against them shifted by the host's UTC offset: it
  replays events west of UTC and silently drops them east of it.
  """
  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"
  session_id = "test_session"
  after_timestamp = 1234567890.0

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

  # Pinned east of UTC so a naive cursor is skewed even on a UTC host.
  with _pinned_local_timezone("Asia/Tokyo"):
    await service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        config=GetSessionConfig(after_timestamp=after_timestamp),
    )

  events_collection_ref.where.assert_called_once()
  field, operator, cursor = events_collection_ref.where.call_args.args
  assert (field, operator) == ("timestamp", ">=")
  assert cursor.utcoffset() == timedelta(0), f"cursor is not UTC: {cursor!r}"
  assert _wire_epoch(cursor) == after_timestamp


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
