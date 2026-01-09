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
"""Tests for migration scripts."""

from __future__ import annotations

from datetime import datetime
from datetime import timezone

from google.adk.events.event_actions import EventActions
from google.adk.sessions.migration import _schema_check_utils
from google.adk.sessions.migration import migrate_from_sqlalchemy_pickle as mfsp
from google.adk.sessions.schemas import v0
from google.adk.sessions.schemas import v1
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def test_migrate_from_sqlalchemy_pickle(tmp_path):
  """Tests for migrate_from_sqlalchemy_pickle."""
  source_db_path = tmp_path / "source_pickle.db"
  dest_db_path = tmp_path / "dest_json.db"
  source_db_url = f"sqlite:///{source_db_path}"
  dest_db_url = f"sqlite:///{dest_db_path}"

  # Set up source DB with old pickle schema
  source_engine = create_engine(source_db_url)
  v0.Base.metadata.create_all(source_engine)
  SourceSession = sessionmaker(bind=source_engine)
  source_session = SourceSession()

  # Populate source data
  now = datetime.now(timezone.utc)
  app_state = v0.StorageAppState(
      app_name="app1", state={"akey": 1}, update_time=now
  )
  user_state = v0.StorageUserState(
      app_name="app1", user_id="user1", state={"ukey": 2}, update_time=now
  )
  session = v0.StorageSession(
      app_name="app1",
      user_id="user1",
      id="session1",
      state={"skey": 3},
      create_time=now,
      update_time=now,
  )
  event = v0.StorageEvent(
      id="event1",
      app_name="app1",
      user_id="user1",
      session_id="session1",
      invocation_id="invoke1",
      author="user",
      actions=EventActions(state_delta={"skey": 4}),
      timestamp=now,
  )
  source_session.add_all([app_state, user_state, session, event])
  source_session.commit()
  source_session.close()

  mfsp.migrate(source_db_url, dest_db_url)

  # Verify destination DB
  dest_engine = create_engine(dest_db_url)
  DestSession = sessionmaker(bind=dest_engine)
  dest_session = DestSession()

  metadata = dest_session.query(v1.StorageMetadata).first()
  assert metadata is not None
  assert metadata.key == _schema_check_utils.SCHEMA_VERSION_KEY
  assert metadata.value == _schema_check_utils.SCHEMA_VERSION_1_JSON

  app_state_res = dest_session.query(v1.StorageAppState).first()
  assert app_state_res is not None
  assert app_state_res.app_name == "app1"
  assert app_state_res.state == {"akey": 1}

  user_state_res = dest_session.query(v1.StorageUserState).first()
  assert user_state_res is not None
  assert user_state_res.user_id == "user1"
  assert user_state_res.state == {"ukey": 2}

  session_res = dest_session.query(v1.StorageSession).first()
  assert session_res is not None
  assert session_res.id == "session1"
  assert session_res.state == {"skey": 3}

  event_res = dest_session.query(v1.StorageEvent).first()
  assert event_res is not None
  assert event_res.id == "event1"
  assert "state_delta" in event_res.event_data["actions"]
  assert event_res.event_data["actions"]["state_delta"] == {"skey": 4}

  dest_session.close()
