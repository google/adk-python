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

import asyncio

import pytest

# Skipped at collection, not per test: the module under test builds on the
# client extension seam, which is the one part of the SDK that 1.x does not
# have under any name, so this file cannot be imported there at all.
pytest.importorskip(
    'mcp.client.extension',
    reason='the MCP Tasks extension requires MCP SDK 2.x',
)

# pylint: disable=g-import-not-at-top
from google.adk.tools.mcp_tool._tasks import _CancelTaskRequest
from google.adk.tools.mcp_tool._tasks import _GetTaskRequest
from google.adk.tools.mcp_tool._tasks import _GetTaskResult
from google.adk.tools.mcp_tool._tasks import _poll_interval_seconds
from google.adk.tools.mcp_tool._tasks import _resolve_task
from google.adk.tools.mcp_tool._tasks import _TaskParams
from google.adk.tools.mcp_tool._tasks import _TaskResult
from google.adk.tools.mcp_tool._tasks import make_tasks_claim
from google.adk.tools.mcp_tool._tasks import TASKS_EXTENSION_ID
from mcp.client.extension import ClaimContext
from mcp.shared.exceptions import MCPError

# pylint: enable=g-import-not-at-top

_CREATED = '2026-08-21T10:30:00Z'


def _task_handle(**overrides) -> _TaskResult:
  """A `resultType: "task"` answer, working unless told otherwise."""
  fields = {
      'task_id': 'task-1',
      'status': 'working',
      'created_at': _CREATED,
      'last_updated_at': _CREATED,
      'ttl_ms': 60000,
      'poll_interval_ms': 1,
  }
  fields.update(overrides)
  return _TaskResult(**fields)


def _task_state(status: str, **overrides) -> _GetTaskResult:
  """One `tasks/get` answer."""
  fields = {
      'task_id': 'task-1',
      'status': status,
      'created_at': _CREATED,
      'last_updated_at': _CREATED,
      'ttl_ms': 60000,
      'poll_interval_ms': 1,
  }
  fields.update(overrides)
  return _GetTaskResult(**fields)


class _FakeSession:
  """A session that replays a scripted sequence of `tasks/*` answers."""

  def __init__(self, answers):
    self._answers = list(answers)
    self.requests = []

  async def send_request(self, request, result_type, **kwargs):
    del result_type, kwargs
    self.requests.append(request)
    if isinstance(request, _CancelTaskRequest):
      return None
    answer = self._answers.pop(0)
    if isinstance(answer, Exception):
      raise answer
    return answer

  @property
  def polls(self):
    return [r for r in self.requests if isinstance(r, _GetTaskRequest)]

  @property
  def cancels(self):
    return [r for r in self.requests if isinstance(r, _CancelTaskRequest)]


def _ctx(session) -> ClaimContext:
  return ClaimContext(
      session=session, tool_name='slow_tool', read_timeout_seconds=None
  )


class TestWireModels:
  """The models have to match SEP-2663 byte for byte on the wire."""

  def test_task_handle_round_trips_the_specified_shape(self):
    wire = {
        'resultType': 'task',
        'taskId': '786512e2-9e0d-44bd-8f29-789f320fe840',
        'status': 'working',
        'statusMessage': 'The operation is now in progress.',
        'createdAt': '2025-11-25T10:30:00Z',
        'lastUpdatedAt': '2025-11-25T10:40:00Z',
        'ttlMs': 60000,
        'pollIntervalMs': 5000,
    }

    parsed = _TaskResult.model_validate(wire)

    assert parsed.task_id == '786512e2-9e0d-44bd-8f29-789f320fe840'
    assert parsed.ttl_ms == 60000
    assert parsed.poll_interval_ms == 5000
    assert parsed.model_dump(by_alias=True, exclude_none=True) == wire

  def test_ttl_is_required_and_nullable(self):
    """The specification makes ttlMs required, but allows a null value."""
    assert _task_handle(ttl_ms=None).ttl_ms is None

    with pytest.raises(Exception):
      _TaskResult.model_validate({
          'resultType': 'task',
          'taskId': 't',
          'status': 'working',
          'createdAt': _CREATED,
          'lastUpdatedAt': _CREATED,
      })

  def test_task_requests_carry_the_task_id_as_their_routing_name(self):
    """`Mcp-Name` is built from name_param, and tasks/* must set it."""
    assert _GetTaskRequest.name_param == 'taskId'
    assert _CancelTaskRequest.name_param == 'taskId'

    request = _GetTaskRequest(params=_TaskParams(task_id='task-1'))

    assert request.model_dump(by_alias=True, exclude_none=True) == {
        'method': 'tasks/get',
        'params': {'taskId': 'task-1'},
    }

  def test_claim_is_registered_for_the_task_result_type(self):
    claim = make_tasks_claim()

    assert claim.result_type == 'task'
    assert claim.model is _TaskResult
    assert TASKS_EXTENSION_ID == 'io.modelcontextprotocol/tasks'


class TestPollInterval:
  """The server states a preference; it is honored, within reason."""

  def test_absent_interval_falls_back_to_a_default(self):
    assert _poll_interval_seconds(None) == 1.0

  def test_stated_interval_is_honored(self):
    assert _poll_interval_seconds(2500) == 2.5

  def test_zero_is_clamped_so_the_loop_cannot_spin(self):
    assert _poll_interval_seconds(0) == 0.1

  def test_an_absurd_interval_is_capped(self):
    assert _poll_interval_seconds(600000) == 30.0


class TestResolveTask:
  """The state machine that turns a task handle into a tool result."""

  @pytest.mark.asyncio
  async def test_polls_until_completion_and_returns_the_result(self):
    session = _FakeSession([
        _task_state('working'),
        _task_state(
            'completed',
            result={'content': [{'type': 'text', 'text': 'done'}]},
        ),
    ])

    result = await _resolve_task(_task_handle(), _ctx(session))

    assert not result.is_error
    assert result.content[0].text == 'done'
    assert len(session.polls) == 2

  @pytest.mark.asyncio
  async def test_a_handle_that_is_already_terminal_is_still_read(self):
    """The result only ever travels on tasks/get, never on the handle."""
    session = _FakeSession([
        _task_state('completed', result={'content': []}),
    ])

    result = await _resolve_task(
        _task_handle(status='completed'), _ctx(session)
    )

    assert not result.is_error
    assert len(session.polls) == 1

  @pytest.mark.asyncio
  async def test_a_failed_task_becomes_a_failed_tool_result(self):
    """A tool that fails reports isError; a task should not change that."""
    session = _FakeSession([
        _task_state(
            'failed', error={'code': -32000, 'message': 'device offline'}
        ),
    ])

    result = await _resolve_task(_task_handle(), _ctx(session))

    assert result.is_error
    assert 'device offline' in result.content[0].text
    assert '-32000' in result.content[0].text

  @pytest.mark.asyncio
  async def test_a_server_cancelled_task_becomes_a_failed_tool_result(self):
    session = _FakeSession([_task_state('cancelled')])

    result = await _resolve_task(_task_handle(), _ctx(session))

    assert result.is_error
    assert 'cancelled by the server' in result.content[0].text

  @pytest.mark.asyncio
  async def test_input_required_is_reported_and_the_task_released(self):
    """Nothing here can answer, so the server should stop holding the task."""
    session = _FakeSession([
        _task_state('input_required', input_requests={'q1': {}}),
    ])

    result = await _resolve_task(_task_handle(), _ctx(session))

    assert result.is_error
    assert 'requires additional input' in result.content[0].text
    assert len(session.cancels) == 1

  @pytest.mark.asyncio
  async def test_an_unreadable_task_becomes_a_failed_tool_result(self):
    """An expired handle answers with an error, and retrying cannot help."""
    session = _FakeSession([MCPError(code=-32602, message='unknown taskId')])

    result = await _resolve_task(_task_handle(), _ctx(session))

    assert result.is_error
    assert 'unknown taskId' in result.content[0].text
    assert len(session.polls) == 1

  @pytest.mark.asyncio
  async def test_a_completed_task_without_a_result_is_reported(self):
    session = _FakeSession([_task_state('completed')])

    result = await _resolve_task(_task_handle(), _ctx(session))

    assert result.is_error
    assert 'without a result' in result.content[0].text

  @pytest.mark.asyncio
  async def test_an_unusable_payload_does_not_escape_as_an_exception(self):
    """The caller has no way to guard a resolver, so nothing may leak out."""
    session = _FakeSession([
        _task_state('completed', result={'content': 'not a content list'}),
    ])

    result = await _resolve_task(_task_handle(), _ctx(session))

    assert result.is_error
    assert 'not a valid tool result' in result.content[0].text

  @pytest.mark.asyncio
  async def test_cancelling_the_caller_cancels_the_task_on_the_server(self):
    """The whole point of propagating cancellation into the resolver."""
    started = asyncio.Event()

    class _SlowSession(_FakeSession):

      async def send_request(self, request, result_type, **kwargs):
        if isinstance(request, _GetTaskRequest):
          started.set()
          await asyncio.sleep(30)
        return await super().send_request(request, result_type, **kwargs)

    session = _SlowSession([])
    task = asyncio.create_task(_resolve_task(_task_handle(), _ctx(session)))
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
      await task

    assert len(session.cancels) == 1
    assert session.cancels[0].params.task_id == 'task-1'

  @pytest.mark.asyncio
  async def test_the_interval_follows_the_latest_value_the_server_sends(self):
    """A server may change its mind between polls, and is obeyed."""
    slept = []
    session = _FakeSession([
        _task_state('working', poll_interval_ms=2000),
        _task_state('completed', result={'content': []}),
    ])

    real_sleep = asyncio.sleep

    async def _record(delay):
      slept.append(delay)
      await real_sleep(0)

    asyncio.sleep = _record
    try:
      await _resolve_task(_task_handle(poll_interval_ms=50), _ctx(session))
    finally:
      asyncio.sleep = real_sleep

    # First from the handle, then from what the first poll reported.
    assert slept == [0.1, 2.0]
