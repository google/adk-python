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
from types import SimpleNamespace
from unittest import mock

from google.adk.cli.utils import _eval_storage
from google.adk.cli.utils import evals
from google.adk.evaluation import local_eval_set_results_manager
from google.adk.evaluation.gcs_eval_set_results_manager import GcsEvalSetResultsManager
from google.adk.evaluation.gcs_eval_sets_manager import GcsEvalSetsManager
from google.adk.evaluation.local_eval_sets_manager import LocalEvalSetsManager
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
  mock_gcs_eval_sets_manager.return_value = mock.MagicMock(
      spec=GcsEvalSetsManager
  )
  mock_gcs_eval_set_results_manager.return_value = mock.MagicMock(
      spec=GcsEvalSetResultsManager
  )

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


@mock.patch.dict(os.environ, {'GOOGLE_CLOUD_PROJECT': 'test-project'})
@mock.patch(
    'google.adk.evaluation.gcs_eval_set_results_manager.GcsEvalSetResultsManager',
    autospec=True,
)
@mock.patch(
    'google.adk.evaluation.gcs_eval_sets_manager.GcsEvalSetsManager',
    autospec=True,
)
def test_create_gcs_eval_managers_from_uri_extracts_bucket_from_path(
    mock_gcs_eval_sets_manager, mock_gcs_eval_set_results_manager
):
  mock_gcs_eval_sets_manager.return_value = mock.MagicMock(
      spec=GcsEvalSetsManager
  )
  mock_gcs_eval_set_results_manager.return_value = mock.MagicMock(
      spec=GcsEvalSetResultsManager
  )

  evals.create_gcs_eval_managers_from_uri('gs://test-bucket/some/path')

  mock_gcs_eval_sets_manager.assert_called_once_with(
      bucket_name='test-bucket', project='test-project'
  )
  mock_gcs_eval_set_results_manager.assert_called_once_with(
      bucket_name='test-bucket', project='test-project'
  )


def test_create_gcs_eval_managers_from_uri_failure():
  with pytest.raises(ValueError):
    evals.create_gcs_eval_managers_from_uri('unsupported-uri')


def test_create_eval_managers_defaults_to_agents_dir(monkeypatch):
  """Without overrides, local evals stay in the agent directory."""
  monkeypatch.delenv(_eval_storage.ADK_EVAL_STORAGE_URI_ENV, raising=False)

  sets_manager, results_manager = _eval_storage.create_eval_managers(
      None, 'some/agents'
  )

  assert isinstance(sets_manager, LocalEvalSetsManager)
  assert isinstance(
      results_manager,
      local_eval_set_results_manager.LocalEvalSetResultsManager,
  )
  assert sets_manager._agents_dir == 'some/agents'
  assert results_manager._agents_dir == 'some/agents'


def test_create_eval_managers_prefers_explicit_uri_over_env(monkeypatch):
  """An explicit URI wins over ADK_EVAL_STORAGE_URI."""
  monkeypatch.setenv(_eval_storage.ADK_EVAL_STORAGE_URI_ENV, 'gs://from-env')
  expected_sets = mock.MagicMock()
  expected_results = mock.MagicMock()
  mock_create_gcs = mock.MagicMock(
      return_value=SimpleNamespace(
          eval_sets_manager=expected_sets,
          eval_set_results_manager=expected_results,
      )
  )
  monkeypatch.setattr(
      evals, 'create_gcs_eval_managers_from_uri', mock_create_gcs
  )

  sets_manager, results_manager = _eval_storage.create_eval_managers(
      'gs://explicit', 'some/agents'
  )

  assert sets_manager is expected_sets
  assert results_manager is expected_results
  mock_create_gcs.assert_called_once_with('gs://explicit')


def test_create_eval_managers_uses_uri_env(monkeypatch, tmp_path):
  """ADK_EVAL_STORAGE_URI is used when no explicit URI is given."""
  storage_dir = tmp_path / 'from_uri'
  monkeypatch.setenv(
      _eval_storage.ADK_EVAL_STORAGE_URI_ENV, storage_dir.as_uri()
  )

  sets_manager, results_manager = _eval_storage.create_eval_managers(
      None, 'some/agents'
  )

  assert isinstance(sets_manager, LocalEvalSetsManager)
  assert isinstance(
      results_manager,
      local_eval_set_results_manager.LocalEvalSetResultsManager,
  )
  assert sets_manager._agents_dir == str(storage_dir)
  assert results_manager._agents_dir == str(storage_dir)
  assert storage_dir.is_dir()


def test_create_eval_managers_file_uri_creates_directory(tmp_path):
  """file:// URIs store evals in that directory, creating it if needed."""
  storage_dir = tmp_path / 'nested' / 'evals'

  sets_manager, results_manager = _eval_storage.create_eval_managers(
      storage_dir.as_uri(), 'some/agents'
  )

  assert sets_manager._agents_dir == str(storage_dir)
  assert results_manager._agents_dir == str(storage_dir)
  assert storage_dir.is_dir()


def test_create_eval_managers_accepts_localhost_file_uri(tmp_path):
  """file://localhost/... resolves to the local directory."""
  storage_dir = tmp_path / 'adk_evals'

  sets_manager, results_manager = _eval_storage.create_eval_managers(
      f'file://localhost{storage_dir}', 'some/agents'
  )

  assert sets_manager._agents_dir == str(storage_dir)
  assert results_manager._agents_dir == str(storage_dir)
  assert storage_dir.is_dir()


def test_create_eval_managers_rejects_non_local_file_uri_host():
  """file:// URIs with a non-local host are rejected."""
  with pytest.raises(
      ValueError, match='file:// URIs must target the local host'
  ):
    _eval_storage.create_eval_managers(
        'file://remotehost/abs/path', 'some/agents'
    )


def test_create_eval_managers_rejects_relative_file_uri():
  """file:// URIs with a relative path are rejected."""
  with pytest.raises(
      ValueError, match='file:// URIs must use an absolute path'
  ):
    _eval_storage.create_eval_managers('file:relative/path', 'some/agents')


def test_create_eval_managers_gcs_uri(monkeypatch):
  expected_sets = mock.MagicMock()
  expected_results = mock.MagicMock()
  mock_create_gcs = mock.MagicMock(
      return_value=SimpleNamespace(
          eval_sets_manager=expected_sets,
          eval_set_results_manager=expected_results,
      )
  )
  monkeypatch.setattr(
      evals, 'create_gcs_eval_managers_from_uri', mock_create_gcs
  )

  sets_manager, results_manager = _eval_storage.create_eval_managers(
      'gs://my-bucket', 'some/agents'
  )

  assert sets_manager is expected_sets
  assert results_manager is expected_results
  mock_create_gcs.assert_called_once_with('gs://my-bucket')


def test_create_eval_managers_gcs_uri_missing_project_raises_value_error(
    monkeypatch,
):
  """A gs:// URI without GOOGLE_CLOUD_PROJECT raises ValueError naming it."""
  monkeypatch.setenv(_eval_storage.ADK_EVAL_STORAGE_URI_ENV, 'gs://my-bucket')
  monkeypatch.delenv('GOOGLE_CLOUD_PROJECT', raising=False)

  with pytest.raises(ValueError, match='GOOGLE_CLOUD_PROJECT'):
    _eval_storage.create_eval_managers(None, 'some/agents')


def test_create_eval_managers_rejects_unknown_scheme():
  with pytest.raises(ValueError, match='Unsupported evals storage URI'):
    _eval_storage.create_eval_managers('s3://bucket', 'some/agents')


def test_create_eval_managers_raises_when_local_dir_unwritable(
    monkeypatch, tmp_path
):
  """Unwritable local paths raise ValueError naming the URI and env var."""
  storage_dir = tmp_path / 'unwritable'

  def fail_makedirs(_path, exist_ok=False):
    raise PermissionError('read-only filesystem')

  monkeypatch.setattr(os, 'makedirs', fail_makedirs)

  with pytest.raises(
      ValueError, match='ADK_EVAL_STORAGE_URI.*read-only filesystem'
  ):
    _eval_storage.create_eval_managers(storage_dir.as_uri(), 'some/agents')


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
