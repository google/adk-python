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

"""Tests for utilities in eval."""

import os
from pathlib import Path
from unittest import mock

from google.adk.cli.utils import evals
from google.adk.events.event import Event
from google.adk.sessions.session import Session
from google.genai import types
import pytest


@mock.patch.dict(os.environ, {'GOOGLE_CLOUD_PROJECT': 'test-project'})
@mock.patch(
    'google.adk.evaluation.gcs_eval_set_results_manager.GcsEvalSetResultsManager',
    autospec=True,
)
@mock.patch(
    'google.adk.evaluation.gcs_eval_sets_manager.GcsEvalSetsManager',
    autospec=True,
)
def test_create_gcs_eval_managers_from_uri_success(
    mock_gcs_eval_sets_manager, mock_gcs_eval_set_results_manager
):
  mock_gcs_eval_sets_manager.return_value = mock.MagicMock()
  mock_gcs_eval_set_results_manager.return_value = mock.MagicMock()

  managers = evals.create_gcs_eval_managers_from_uri('gs://test-bucket')

  assert managers is not None
  mock_gcs_eval_sets_manager.assert_called_once_with(
      bucket_name='test-bucket', project='test-project'
  )
  mock_gcs_eval_set_results_manager.assert_called_once_with(
      bucket_name='test-bucket', project='test-project'
  )
  assert managers.eval_sets_manager == mock_gcs_eval_sets_manager.return_value
  assert (
      managers.eval_set_results_manager
      == mock_gcs_eval_set_results_manager.return_value
  )


def test_create_gcs_eval_managers_from_uri_failure():
  with pytest.raises(ValueError):
    evals.create_gcs_eval_managers_from_uri('unsupported-uri')


def test_resolve_eval_storage_uri_prefers_explicit_over_env(monkeypatch):
  """An explicit URI wins over ADK_EVAL_STORAGE_URI and ADK_EVAL_STORAGE_DIR."""
  monkeypatch.setenv(evals.ADK_EVAL_STORAGE_URI_ENV, 'gs://from-env')
  monkeypatch.setenv(evals.ADK_EVAL_STORAGE_DIR_ENV, '/tmp/from-dir')

  assert evals.resolve_eval_storage_uri('gs://explicit') == 'gs://explicit'


def test_resolve_eval_storage_uri_uses_uri_env(monkeypatch):
  """ADK_EVAL_STORAGE_URI is used when no explicit URI is given."""
  monkeypatch.delenv(evals.ADK_EVAL_STORAGE_DIR_ENV, raising=False)
  monkeypatch.setenv(evals.ADK_EVAL_STORAGE_URI_ENV, 'file:///tmp/from-uri')

  assert evals.resolve_eval_storage_uri(None) == 'file:///tmp/from-uri'


def test_resolve_eval_storage_uri_uses_dir_env(monkeypatch, tmp_path):
  """ADK_EVAL_STORAGE_DIR becomes a file:// URI when no other URI is set."""
  monkeypatch.delenv(evals.ADK_EVAL_STORAGE_URI_ENV, raising=False)
  storage_dir = tmp_path / 'evals'
  monkeypatch.setenv(evals.ADK_EVAL_STORAGE_DIR_ENV, str(storage_dir))

  expected = Path(os.path.abspath(str(storage_dir))).as_uri()
  assert evals.resolve_eval_storage_uri(None) == expected


def test_resolve_eval_storage_uri_returns_none_without_overrides(monkeypatch):
  """No URI or env override means callers should use agents_dir."""
  monkeypatch.delenv(evals.ADK_EVAL_STORAGE_URI_ENV, raising=False)
  monkeypatch.delenv(evals.ADK_EVAL_STORAGE_DIR_ENV, raising=False)

  assert evals.resolve_eval_storage_uri(None) is None


def test_local_path_from_file_uri_decodes_posix_path():
  """file:///tmp/adk_evals maps to the local /tmp/adk_evals directory."""
  assert evals.local_path_from_file_uri('file:///tmp/adk_evals') == (
      '/tmp/adk_evals'
  )


def test_local_path_from_file_uri_rejects_non_file_scheme():
  with pytest.raises(ValueError, match='Unsupported evals storage URI'):
    evals.local_path_from_file_uri('gs://bucket')


def test_resolve_eval_storage_defaults_to_agents_dir(monkeypatch):
  """Without overrides, local evals stay in the agent directory."""
  monkeypatch.delenv(evals.ADK_EVAL_STORAGE_URI_ENV, raising=False)
  monkeypatch.delenv(evals.ADK_EVAL_STORAGE_DIR_ENV, raising=False)

  storage = evals.resolve_eval_storage(None, 'some/agents')

  assert storage.gcs_uri is None
  assert storage.local_dir == 'some/agents'


def test_resolve_eval_storage_file_uri_creates_directory(tmp_path):
  """file:// URIs store evals in that directory, creating it if needed."""
  storage_dir = tmp_path / 'nested' / 'evals'

  storage = evals.resolve_eval_storage(storage_dir.as_uri(), 'some/agents')

  assert storage.gcs_uri is None
  assert storage.local_dir == str(storage_dir)
  assert storage_dir.is_dir()


def test_resolve_eval_storage_dir_env_creates_directory(monkeypatch, tmp_path):
  """ADK_EVAL_STORAGE_DIR is created and used instead of agents_dir."""
  monkeypatch.delenv(evals.ADK_EVAL_STORAGE_URI_ENV, raising=False)
  storage_dir = tmp_path / 'from_env'
  monkeypatch.setenv(evals.ADK_EVAL_STORAGE_DIR_ENV, str(storage_dir))

  storage = evals.resolve_eval_storage(None, 'some/agents')

  assert storage.gcs_uri is None
  assert storage.local_dir == str(storage_dir)
  assert storage_dir.is_dir()


def test_resolve_eval_storage_dir_env_expands_user_home(monkeypatch, tmp_path):
  """~ in ADK_EVAL_STORAGE_DIR expands to the user home directory."""
  monkeypatch.delenv(evals.ADK_EVAL_STORAGE_URI_ENV, raising=False)
  monkeypatch.setenv('HOME', str(tmp_path))
  monkeypatch.setenv(evals.ADK_EVAL_STORAGE_DIR_ENV, '~/evals')

  storage = evals.resolve_eval_storage(None, 'some/agents')

  assert storage.local_dir == str(tmp_path / 'evals')
  assert (tmp_path / 'evals').is_dir()


def test_resolve_eval_storage_gcs_uri():
  storage = evals.resolve_eval_storage('gs://my-bucket', 'some/agents')

  assert storage.gcs_uri == 'gs://my-bucket'
  assert storage.local_dir == 'some/agents'


def test_resolve_eval_storage_rejects_unknown_scheme():
  with pytest.raises(ValueError, match='Unsupported evals storage URI'):
    evals.resolve_eval_storage('s3://bucket', 'some/agents')


def _event(author: str, text: str, invocation_id: str) -> Event:
  return Event(
      author=author,
      invocation_id=invocation_id,
      content=types.Content(
          role='user' if author == 'user' else 'model',
          parts=[types.Part(text=text)],
      ),
  )


def _session(events: list[Event]) -> Session:
  return Session(id='s1', app_name='app', user_id='u1', events=events)


def test_convert_session_to_eval_invocations_groups_events_by_invocation():
  session = _session([
      _event('user', 'first question', 'inv-1'),
      _event('agent', 'first answer', 'inv-1'),
      _event('user', 'second question', 'inv-2'),
      _event('agent', 'second answer', 'inv-2'),
  ])

  invocations = evals.convert_session_to_eval_invocations(session)

  assert [i.invocation_id for i in invocations] == ['inv-1', 'inv-2']
  assert [i.user_content.parts[0].text for i in invocations] == [
      'first question',
      'second question',
  ]
  assert [i.final_response.parts[0].text for i in invocations] == [
      'first answer',
      'second answer',
  ]


def test_convert_session_to_eval_invocations_handles_missing_history():
  """The CLI calls this before a session has any turns, and on no session."""
  assert evals.convert_session_to_eval_invocations(_session([])) == []
  assert evals.convert_session_to_eval_invocations(None) == []
