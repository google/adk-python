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

"""Tests for the secret: session state scope."""

import json

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions._session_util import extract_state_delta
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.adk.sessions.state import State
from google.adk.utils.instructions_utils import _is_valid_state_name
import pytest

from .test_session_service import get_session_service
from .test_session_service import SessionServiceType

# ---------------------------------------------------------------------------
# Unit: extract_state_delta
# ---------------------------------------------------------------------------


class TestExtractStateDelta:

  def test_secret_keys_excluded_from_all_buckets(self):
    """secret: keys must not appear in app, user, or session buckets."""
    state = {
        'app:a': 1,
        'user:u': 2,
        'temp:t': 3,
        'secret:tok': 'abc',
        'normal': 4,
    }
    deltas = extract_state_delta(state)
    assert deltas['app'] == {'a': 1}
    assert deltas['user'] == {'u': 2}
    assert deltas['session'] == {'normal': 4}
    # secret and temp keys must not be in any bucket
    all_values = {}
    for bucket in deltas.values():
      all_values.update(bucket)
    assert 'secret:tok' not in all_values
    assert 'temp:t' not in all_values

  def test_only_secret_keys(self):
    """If all keys are secret:, all buckets should be empty."""
    state = {'secret:a': 1, 'secret:b': 2}
    deltas = extract_state_delta(state)
    assert deltas == {'app': {}, 'user': {}, 'session': {}}


# ---------------------------------------------------------------------------
# Unit: State.SECRET_PREFIX
# ---------------------------------------------------------------------------


class TestStatePrefix:

  def test_secret_prefix_exists(self):
    assert State.SECRET_PREFIX == 'secret:'


# ---------------------------------------------------------------------------
# Unit: _is_valid_state_name
# ---------------------------------------------------------------------------


class TestValidation:

  def test_secret_prefix_is_valid(self):
    assert _is_valid_state_name('secret:token') is True

  def test_secret_prefix_invalid_name(self):
    # Non-identifier after prefix
    assert _is_valid_state_name('secret:123abc') is False


# ---------------------------------------------------------------------------
# Unit: BaseSessionService cache helpers
# ---------------------------------------------------------------------------


class TestBaseSessionServiceHelpers:
  """Tests the lifecycle helpers on BaseSessionService."""

  def _make_service(self):
    """Create an InMemorySessionService (simplest concrete subclass)."""
    return InMemorySessionService()

  def _make_session(self, app='app', user='user', sid='s1'):
    return Session(
        app_name=app,
        user_id=user,
        id=sid,
        state={},
    )

  def _make_event(self, state_delta):
    return Event(
        invocation_id='inv',
        author='agent',
        actions=EventActions(state_delta=state_delta),
    )

  def test_lazy_cache_initialization(self):
    """Cache is lazily created on first access."""
    svc = self._make_service()
    cache = svc._secret_state_cache
    assert isinstance(cache, dict)
    assert len(cache) == 0
    # Same object on second access
    assert svc._secret_state_cache is cache

  def test_apply_and_trim_secret_state(self):
    svc = self._make_service()
    session = self._make_session()
    event = self._make_event({
        'secret:token': 'abc',
        'normal_key': 'val',
    })

    svc._apply_secret_state(session, event)
    svc._trim_secret_delta_state(event)

    # Secret key applied to session state
    assert session.state.get('secret:token') == 'abc'
    # Secret key removed from event delta
    assert 'secret:token' not in event.actions.state_delta
    # Normal key still in delta
    assert event.actions.state_delta.get('normal_key') == 'val'
    # Secret key in process cache
    cache_key = ('app', 'user', 's1')
    assert svc._secret_state_cache[cache_key]['secret:token'] == 'abc'

  def test_seed_and_restore_secret_state(self):
    svc = self._make_service()
    state = {
        'secret:cred': 'xyz',
        'app:foo': 'bar',
        'normal': 123,
    }
    cleaned = svc._seed_secret_state_on_create(
        app_name='app',
        user_id='user',
        session_id='s1',
        state=state,
    )
    # Returned state has secret keys stripped
    assert 'secret:cred' not in cleaned
    assert cleaned['app:foo'] == 'bar'
    assert cleaned['normal'] == 123

    # Restore into a session
    session = self._make_session()
    svc._restore_secret_state(session)
    assert session.state.get('secret:cred') == 'xyz'

  def test_seed_with_no_secret_keys(self):
    svc = self._make_service()
    state = {'normal': 1}
    result = svc._seed_secret_state_on_create(
        app_name='a', user_id='u', session_id='s', state=state
    )
    assert result == {'normal': 1}

  def test_seed_with_none_state(self):
    svc = self._make_service()
    result = svc._seed_secret_state_on_create(
        app_name='a', user_id='u', session_id='s', state=None
    )
    assert result is None

  def test_evict_secret_state(self):
    svc = self._make_service()
    svc._secret_state_cache[('a', 'u', 's')] = {'secret:x': 1}
    svc._evict_secret_state('a', 'u', 's')
    assert ('a', 'u', 's') not in svc._secret_state_cache

  def test_evict_nonexistent_key(self):
    """Evicting a non-existent key should not raise."""
    svc = self._make_service()
    svc._evict_secret_state('a', 'u', 's')  # no-op


# ---------------------------------------------------------------------------
# Integration: session service lifecycle (parametrized)
# ---------------------------------------------------------------------------

_SERVICE_TYPES = [
    SessionServiceType.IN_MEMORY,
    SessionServiceType.DATABASE,
    SessionServiceType.SQLITE,
]


@pytest.fixture(params=_SERVICE_TYPES)
async def session_service(request, tmp_path):
  service = get_session_service(request.param, tmp_path)
  yield service
  if isinstance(service, DatabaseSessionService):
    await service.close()


class TestSecretStateLifecycle:
  """Integration tests for secret: state across session lifecycle."""

  async def test_append_event_secret_survives_across_turns(
      self, session_service
  ):
    """Secret state set via append_event survives get_session."""
    session = await session_service.create_session(
        app_name='app', user_id='user'
    )
    event = Event(
        invocation_id='inv1',
        author='agent',
        actions=EventActions(
            state_delta={'secret:token': 'abc123', 'visible': 'yes'}
        ),
    )
    await session_service.append_event(session=session, event=event)

    # Secret is in in-memory session
    assert session.state.get('secret:token') == 'abc123'
    assert session.state.get('visible') == 'yes'

    # Secret key trimmed from event delta
    assert 'secret:token' not in event.actions.state_delta

    # Secret survives get_session (restored from cache)
    restored = await session_service.get_session(
        app_name='app',
        user_id='user',
        session_id=session.id,
    )
    assert restored.state.get('secret:token') == 'abc123'
    assert restored.state.get('visible') == 'yes'

  async def test_create_session_with_secret_state(self, session_service):
    """Secret keys in initial state are cached, not persisted."""
    session = await session_service.create_session(
        app_name='app',
        user_id='user',
        state={'secret:init_cred': 'init_val', 'normal': 'nval'},
    )
    # Secret available in returned session
    assert session.state.get('secret:init_cred') == 'init_val'
    assert session.state.get('normal') == 'nval'

    # Secret survives get_session
    restored = await session_service.get_session(
        app_name='app',
        user_id='user',
        session_id=session.id,
    )
    assert restored.state.get('secret:init_cred') == 'init_val'
    assert restored.state.get('normal') == 'nval'

  async def test_delete_session_evicts_secret_cache(self, session_service):
    """Deleting a session evicts its secret cache entry."""
    session = await session_service.create_session(
        app_name='app',
        user_id='user',
        state={'secret:key': 'val'},
    )
    sid = session.id

    await session_service.delete_session(
        app_name='app', user_id='user', session_id=sid
    )

    # Cache should be empty for this session
    cache_key = ('app', 'user', sid)
    assert cache_key not in session_service._secret_state_cache

  async def test_list_sessions_does_not_include_secret_state(
      self, session_service
  ):
    """list_sessions must NOT merge secret state."""
    session = await session_service.create_session(
        app_name='app',
        user_id='user',
        state={'secret:hidden': 'shhh', 'visible': 'ok'},
    )

    response = await session_service.list_sessions(
        app_name='app', user_id='user'
    )
    assert len(response.sessions) >= 1
    listed = next(s for s in response.sessions if s.id == session.id)
    # Secret should NOT be in listed session state
    assert listed.state.get('secret:hidden') is None
    # Normal state should be present
    assert listed.state.get('visible') == 'ok'

  async def test_secret_and_temp_independent(self, session_service):
    """secret: and temp: scopes work independently."""
    session = await session_service.create_session(
        app_name='app', user_id='user'
    )
    event = Event(
        invocation_id='inv1',
        author='agent',
        actions=EventActions(
            state_delta={
                'secret:s': 'secret_val',
                'temp:t': 'temp_val',
                'normal': 'n',
            }
        ),
    )
    await session_service.append_event(session=session, event=event)

    # Both available in-memory
    assert session.state.get('secret:s') == 'secret_val'
    assert session.state.get('temp:t') == 'temp_val'

    # Neither in event delta
    assert 'secret:s' not in event.actions.state_delta
    assert 'temp:t' not in event.actions.state_delta
    # Normal key persisted
    assert event.actions.state_delta.get('normal') == 'n'

    # After get_session: secret survives, temp does not
    restored = await session_service.get_session(
        app_name='app',
        user_id='user',
        session_id=session.id,
    )
    assert restored.state.get('secret:s') == 'secret_val'
    # temp is lost after get_session (invocation-scoped)
    assert restored.state.get('temp:t') is None


# ---------------------------------------------------------------------------
# BQAA redaction
# ---------------------------------------------------------------------------


class TestBQAARedaction:

  def test_secret_key_redacted(self):
    from google.adk.plugins.bigquery_agent_analytics_plugin import _recursive_smart_truncate

    obj = {'secret:token': 'my_secret', 'normal': 'visible'}
    result, _ = _recursive_smart_truncate(obj, max_len=-1)
    assert result['secret:token'] == '[REDACTED]'
    assert result['normal'] == 'visible'

  def test_json_blob_with_sensitive_keys_redacted(self):
    from google.adk.plugins.bigquery_agent_analytics_plugin import _recursive_smart_truncate

    cred_json = json.dumps({
        'access_token': 'ya29.xxx',
        'refresh_token': 'rt_xxx',
        'client_id': 'my_client',
    })
    obj = {'bigquery_token_cache': cred_json, 'normal': 'ok'}
    result, _ = _recursive_smart_truncate(obj, max_len=-1)
    assert result['bigquery_token_cache'] == '[REDACTED]'
    assert result['normal'] == 'ok'

  def test_json_blob_without_sensitive_keys_not_redacted(self):
    from google.adk.plugins.bigquery_agent_analytics_plugin import _recursive_smart_truncate

    safe_json = json.dumps({'query': 'SELECT 1', 'count': 42})
    obj = {'data': safe_json}
    result, _ = _recursive_smart_truncate(obj, max_len=-1)
    # Should not be redacted since no sensitive keys
    assert result['data'] == safe_json

  def test_non_json_string_not_redacted(self):
    from google.adk.plugins.bigquery_agent_analytics_plugin import _recursive_smart_truncate

    obj = {'note': 'this is a normal string'}
    result, _ = _recursive_smart_truncate(obj, max_len=-1)
    assert result['note'] == 'this is a normal string'

  def test_is_sensitive_json_string_helper(self):
    from google.adk.plugins.bigquery_agent_analytics_plugin import _is_sensitive_json_string

    assert _is_sensitive_json_string('not json') is False
    assert _is_sensitive_json_string('') is False
    assert _is_sensitive_json_string('{"safe_key": 1}') is False
    assert _is_sensitive_json_string('{"access_token": "ya29"}') is True
    assert _is_sensitive_json_string('{"Client_Secret": "xxx"}') is True
    # Array does not trigger
    assert _is_sensitive_json_string('[{"access_token": "ya29"}]') is False
