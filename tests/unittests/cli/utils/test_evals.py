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
from unittest import mock

from google.adk.cli.utils import evals
from google.adk.evaluation.gcs_eval_set_results_manager import GcsEvalSetResultsManager
from google.adk.evaluation.gcs_eval_sets_manager import GcsEvalSetsManager
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


def test_create_gcs_eval_managers_from_uri_failure():
  with pytest.raises(ValueError):
    evals.create_gcs_eval_managers_from_uri('unsupported-uri')


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
