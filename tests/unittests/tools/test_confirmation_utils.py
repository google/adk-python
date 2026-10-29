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

"""Tests for the shared tool confirmation helpers.

Verifies that the gate holds back a call that needs confirmation, records the
confirmation request, and lets approved or unguarded calls through.
"""

import json
from typing import Any
from unittest.mock import MagicMock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.sessions.session import Session
from google.adk.tools._confirmation_utils import apply_confirmation_gate
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_confirmation import ConfirmationRequest
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.adk.tools.tool_context import ToolContext
import pytest

_FUNCTION_CALL_ID = 'call-1'


def _make_tool_context(
    tool_confirmation: ToolConfirmation | None = None,
) -> ToolContext:
  invocation_context = MagicMock(spec=InvocationContext)
  invocation_context._state_schema = None
  invocation_context.session = MagicMock(spec=Session)
  invocation_context.session.state = {}
  return ToolContext(
      invocation_context=invocation_context,
      function_call_id=_FUNCTION_CALL_ID,
      tool_confirmation=tool_confirmation,
  )


def _delete_file(path: str) -> dict[str, str]:
  return {'deleted': path}


async def test_tool_without_confirmation_proceeds():
  """A tool that does not require confirmation is let through untouched."""
  tool = FunctionTool(func=_delete_file)
  tool_context = _make_tool_context()

  response = await apply_confirmation_gate(tool, {'path': 'a'}, tool_context)

  assert response is None
  assert not tool_context.actions.requested_tool_confirmations
  assert not tool_context.actions.skip_summarization


async def test_unanswered_call_requests_confirmation():
  """A call that needs confirmation is held back and records the request."""
  tool = FunctionTool(func=_delete_file, require_confirmation=True)
  tool_context = _make_tool_context()

  response = await apply_confirmation_gate(tool, {'path': 'a'}, tool_context)

  assert response == {
      'error': 'This tool call requires confirmation, please approve or reject.'
  }
  requested = tool_context.actions.requested_tool_confirmations
  assert list(requested) == [_FUNCTION_CALL_ID]
  assert requested[_FUNCTION_CALL_ID].hint == (
      'Please approve or reject the tool call _delete_file() by responding'
      ' with a FunctionResponse with an expected ToolConfirmation payload.'
  )
  assert requested[_FUNCTION_CALL_ID].payload is None
  assert not tool_context.actions.skip_summarization


async def test_confirmed_call_proceeds():
  """An approved call is let through without a new request."""
  tool = FunctionTool(func=_delete_file, require_confirmation=True)
  tool_context = _make_tool_context(ToolConfirmation(confirmed=True))

  response = await apply_confirmation_gate(tool, {'path': 'a'}, tool_context)

  assert response is None
  assert not tool_context.actions.requested_tool_confirmations


async def test_rejected_call_is_answered_with_rejection():
  """A rejected call is answered with the rejection error."""
  tool = FunctionTool(func=_delete_file, require_confirmation=True)
  tool_context = _make_tool_context(ToolConfirmation(confirmed=False))

  response = await apply_confirmation_gate(tool, {'path': 'a'}, tool_context)

  assert response == {'error': 'This tool call is rejected.'}
  assert not tool_context.actions.requested_tool_confirmations


async def test_confirmation_predicate_sees_call_args():
  """The tool's confirmation predicate decides per call from its arguments."""
  tool = FunctionTool(
      func=_delete_file,
      require_confirmation=lambda path: path.startswith('/prod'),
  )

  dev_response = await apply_confirmation_gate(
      tool, {'path': '/dev/a'}, _make_tool_context()
  )
  prod_response = await apply_confirmation_gate(
      tool, {'path': '/prod/a'}, _make_tool_context()
  )

  assert dev_response is None
  assert prod_response == {
      'error': 'This tool call requires confirmation, please approve or reject.'
  }


async def test_non_bool_confirmation_answer_proceeds():
  """Only a True answer from check_require_confirmation holds the call."""

  class _TruthyTool(BaseTool):

    async def check_require_confirmation(
        self, args: dict[str, Any], tool_context: ToolContext
    ) -> Any:
      return 'yes'

  tool = _TruthyTool(name='truthy', description='Answers with a string.')
  tool_context = _make_tool_context()

  response = await apply_confirmation_gate(tool, {}, tool_context)

  assert response is None
  assert not tool_context.actions.requested_tool_confirmations


class _DescribingTool(BaseTool):
  """A gated tool that tells the user what they are approving."""

  def __init__(self, request: ConfirmationRequest):
    super().__init__(name='describing', description='Describes its call.')
    self._request = request
    self.hook_calls: list[tuple[dict[str, Any], ToolContext]] = []

  async def check_require_confirmation(
      self, args: dict[str, Any], tool_context: ToolContext
  ) -> bool:
    return True

  async def get_confirmation_request(
      self, args: dict[str, Any], tool_context: ToolContext
  ) -> ConfirmationRequest:
    self.hook_calls.append((args, tool_context))
    return self._request


async def test_base_tool_default_request_is_the_generic_hint():
  """A tool that does not override the hook asks with the generic hint."""
  tool = FunctionTool(func=_delete_file)

  request = await tool.get_confirmation_request(
      {'path': 'a'}, _make_tool_context()
  )

  assert request == ConfirmationRequest(
      hint=(
          'Please approve or reject the tool call _delete_file() by responding'
          ' with a FunctionResponse with an expected ToolConfirmation payload.'
      ),
      payload=None,
  )


async def test_tool_supplied_hint_and_payload_are_requested():
  """The gate asks with the hint and payload the tool supplies."""
  tool = _DescribingTool(
      ConfirmationRequest(
          hint='Delete a?', payload={'label': 'A', 'tags': ['x', 1, None]}
      )
  )
  tool_context = _make_tool_context()

  response = await apply_confirmation_gate(tool, {'path': 'a'}, tool_context)

  assert response == {
      'error': 'This tool call requires confirmation, please approve or reject.'
  }
  requested = tool_context.actions.requested_tool_confirmations[
      _FUNCTION_CALL_ID
  ]
  assert requested.hint == 'Delete a?'
  assert requested.payload == {'label': 'A', 'tags': ['x', 1, None]}
  assert requested.confirmed is False
  # The payload reaches the client as JSON.
  assert json.loads(requested.model_dump_json())['payload'] == {
      'label': 'A',
      'tags': ['x', 1, None],
  }


async def test_tool_supplied_payload_without_hint_keeps_payload():
  """A payload is sent even when the tool leaves the hint empty."""
  tool = _DescribingTool(ConfirmationRequest(payload={'k': 'v'}))
  tool_context = _make_tool_context()

  await apply_confirmation_gate(tool, {}, tool_context)

  requested = tool_context.actions.requested_tool_confirmations[
      _FUNCTION_CALL_ID
  ]
  assert requested.hint == ''
  assert requested.payload == {'k': 'v'}


async def test_hook_receives_call_args_and_context():
  """The hook is given the arguments of the call and its tool context."""
  tool = _DescribingTool(ConfirmationRequest(hint='h'))
  tool_context = _make_tool_context()
  args = {'path': '/prod/a', 'force': True}

  await apply_confirmation_gate(tool, args, tool_context)

  assert len(tool.hook_calls) == 1
  seen_args, seen_context = tool.hook_calls[0]
  assert seen_args == args
  assert seen_context is tool_context


async def test_hook_is_not_called_for_ungated_call():
  """A tool that does not need confirmation is never asked for a request."""

  class _UngatedTool(_DescribingTool):

    async def check_require_confirmation(
        self, args: dict[str, Any], tool_context: ToolContext
    ) -> bool:
      return False

  tool = _UngatedTool(ConfirmationRequest(hint='h'))
  tool_context = _make_tool_context()

  response = await apply_confirmation_gate(tool, {}, tool_context)

  assert response is None
  assert tool.hook_calls == []
  assert not tool_context.actions.requested_tool_confirmations


@pytest.mark.parametrize(
    ('confirmation', 'expected_response'),
    [
        (ToolConfirmation(confirmed=True), None),
        (
            ToolConfirmation(confirmed=False),
            {'error': 'This tool call is rejected.'},
        ),
    ],
    ids=['approved', 'rejected'],
)
async def test_hook_is_not_called_once_the_call_is_answered(
    confirmation: ToolConfirmation, expected_response: dict[str, str] | None
):
  """An answered call is approved or rejected without asking again."""
  tool = _DescribingTool(ConfirmationRequest(hint='h', payload={'k': 'v'}))
  tool_context = _make_tool_context(confirmation)

  response = await apply_confirmation_gate(tool, {}, tool_context)

  assert response == expected_response
  assert tool.hook_calls == []
  assert not tool_context.actions.requested_tool_confirmations


async def test_failing_hook_neither_requests_nor_lets_the_call_through():
  """A hook that raises propagates, so the call is not run or approved."""

  class _FailingTool(_DescribingTool):

    async def get_confirmation_request(
        self, args: dict[str, Any], tool_context: ToolContext
    ) -> ConfirmationRequest:
      raise RuntimeError('cannot describe the call')

  tool = _FailingTool(ConfirmationRequest())
  tool_context = _make_tool_context()

  with pytest.raises(RuntimeError, match='cannot describe the call'):
    await apply_confirmation_gate(tool, {}, tool_context)

  assert not tool_context.actions.requested_tool_confirmations
