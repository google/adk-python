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

"""Tests for the ADK knowledge search tool's request wiring and error paths."""

from __future__ import annotations

from typing import Any
import uuid

from google.adk.cli.built_in_agents.tools import search_adk_knowledge as module
from google.adk.cli.built_in_agents.tools.search_adk_knowledge import post_request
from google.adk.cli.built_in_agents.tools.search_adk_knowledge import search_adk_knowledge
import pytest
import requests

_BASE = module.KNOWLEDGE_SERVICE_APP_URL
_APP = module.KNOWLEDGE_SERVICE_APP_NAME
_USER = module.KNOWLEDGE_SERVICE_APP_USER_NAME


class _RecordingPostRequest:
  """Stands in for post_request, replaying scripted results in order."""

  def __init__(self, results: list[Any]):
    self._results = list(results)
    self.calls: list[tuple[str, dict[str, Any]]] = []

  def __call__(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    self.calls.append((url, payload))
    result = self._results.pop(0)
    if isinstance(result, Exception):
      raise result
    return result


def test_search_adk_knowledge_runs_the_query_on_the_server_issued_session(
    monkeypatch,
):
  fake = _RecordingPostRequest(
      [{'id': 'server-session'}, {'events': [{'text': 'answer'}]}]
  )
  monkeypatch.setattr(module, 'post_request', fake)

  result = search_adk_knowledge('how do i define a sub agent')

  create_url, create_payload = fake.calls[0]
  prefix = f'{_BASE}/apps/{_APP}/users/{_USER}/sessions/'
  assert create_url.startswith(prefix)
  # A brand-new random session per call, so concurrent searches cannot collide.
  assert uuid.UUID(create_url[len(prefix) :]).version == 4
  assert create_payload == {}

  search_url, search_payload = fake.calls[1]
  assert search_url == f'{_BASE}/run'
  # The session id sent with the query is the one the server handed back, not
  # the locally generated uuid in the create URL.
  assert search_payload == {
      'app_name': _APP,
      'user_id': _USER,
      'session_id': 'server-session',
      'new_message': {
          'role': 'user',
          'parts': [{'text': 'how do i define a sub agent'}],
      },
  }
  assert result == {
      'status': 'success',
      'response': {'events': [{'text': 'answer'}]},
  }


def test_search_adk_knowledge_returns_an_error_when_session_creation_fails(
    monkeypatch,
):
  fake = _RecordingPostRequest([requests.exceptions.ConnectionError('boom')])
  monkeypatch.setattr(module, 'post_request', fake)

  result = search_adk_knowledge('anything')

  assert result == {
      'status': 'error',
      'error_message': 'Failed to create session: boom',
  }
  # The query is never attempted without a session.
  assert len(fake.calls) == 1


def test_search_adk_knowledge_returns_an_error_when_the_query_fails(
    monkeypatch,
):
  fake = _RecordingPostRequest(
      [{'id': 'server-session'}, requests.exceptions.Timeout('too slow')]
  )
  monkeypatch.setattr(module, 'post_request', fake)

  result = search_adk_knowledge('anything')

  assert result == {
      'status': 'error',
      'error_message': 'Failed to search ADK knowledge base: too slow',
  }


class _FakeResponse:

  def __init__(self, payload: Any, error: Exception | None = None):
    self._payload = payload
    self._error = error

  def raise_for_status(self) -> None:
    if self._error:
      raise self._error

  def json(self) -> Any:
    return self._payload


def test_post_request_posts_json_with_a_timeout_and_returns_the_body(
    monkeypatch,
):
  captured: dict[str, Any] = {}

  def fake_post(url, **kwargs):
    captured['url'] = url
    captured.update(kwargs)
    return _FakeResponse({'id': 'abc'})

  monkeypatch.setattr(requests, 'post', fake_post)

  assert post_request('https://example.invalid/x', {'k': 'v'}) == {'id': 'abc'}
  assert captured['url'] == 'https://example.invalid/x'
  # Sent as a JSON body (not form data), and never allowed to hang forever.
  assert captured['json'] == {'k': 'v'}
  assert captured['timeout'] == 60
  assert captured['headers']['Content-Type'] == 'application/json'


def test_post_request_raises_on_an_error_status(monkeypatch):
  error = requests.exceptions.HTTPError('503 Service Unavailable')
  monkeypatch.setattr(
      requests, 'post', lambda *a, **k: _FakeResponse(None, error=error)
  )

  # search_adk_knowledge relies on this to turn a bad status into its error
  # dict, so the status must not be swallowed here.
  with pytest.raises(requests.exceptions.HTTPError, match='503'):
    post_request('https://example.invalid/x', {})
