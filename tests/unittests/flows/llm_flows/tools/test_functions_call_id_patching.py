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

"""Tests that patching the compat shim still controls generated call ids.

`cli/agent_test_runner.py` patches
`google.adk.flows.llm_flows.functions.generate_client_function_call_id` to make
golden files reproducible, and callers outside ADK patch the same name. Before
the tools subpackage was split out, every id in `functions.py` came from that
one module global, so a single patch covered all of them. These tests pin that
property down so the split cannot quietly take it away again.
"""

from __future__ import annotations

from unittest import mock

from fastapi.openapi.models import HTTPBearer
from google.adk.agents.llm_agent import Agent
from google.adk.auth.auth_tool import AuthConfig
from google.adk.events.event import Event
from google.adk.flows.llm_flows import functions
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.genai import types
import pytest

from .... import testing_utils


def _patch_id_generator(*ids: str):
  """Patches the shim name that external callers actually patch."""
  return mock.patch.object(
      functions,
      'generate_client_function_call_id',
      side_effect=list(ids),
  )


def test_populate_client_function_call_id_honors_shim_patch():
  event = Event(
      invocation_id='invocation-id',
      author='agent',
      content=types.Content(
          role='model',
          parts=[types.Part.from_function_call(name='some_tool', args={})],
      ),
  )

  with _patch_id_generator('fc-1'):
    functions.populate_client_function_call_id(event)

  assert event.get_function_calls()[0].id == 'fc-1'


@pytest.mark.asyncio
async def test_build_auth_request_event_honors_shim_patch():
  """Covers the adk_request_credential path used by EUC agent tests."""
  invocation_context = await testing_utils.create_invocation_context(
      agent=Agent(name='agent', model='gemini-2.5-flash')
  )

  with _patch_id_generator('fc-1'):
    event = functions.build_auth_request_event(
        invocation_context,
        {'original_call_id': AuthConfig(auth_scheme=HTTPBearer())},
    )

  assert [fc.id for fc in event.get_function_calls()] == ['fc-1']
  assert event.long_running_tool_ids == {'fc-1'}


@pytest.mark.asyncio
async def test_generate_request_confirmation_event_honors_shim_patch():
  """Covers the adk_request_confirmation path used by confirmation agent tests."""
  invocation_context = await testing_utils.create_invocation_context(
      agent=Agent(name='agent', model='gemini-2.5-flash')
  )
  original_call = types.Part.from_function_call(name='some_tool', args={})
  original_call.function_call.id = 'original_call_id'
  function_call_event = Event(
      invocation_id='invocation-id',
      author='agent',
      content=types.Content(role='model', parts=[original_call]),
  )
  function_response_event = Event(
      invocation_id='invocation-id',
      author='agent',
  )
  function_response_event.actions.requested_tool_confirmations = {
      'original_call_id': ToolConfirmation(hint='please confirm')
  }

  with _patch_id_generator('fc-1'):
    event = functions.generate_request_confirmation_event(
        invocation_context,
        function_call_event,
        function_response_event,
    )

  assert [fc.id for fc in event.get_function_calls()] == ['fc-1']
