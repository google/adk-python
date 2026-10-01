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

"""Tests for the conformance recordings plugin."""

from pathlib import Path
from unittest import mock

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.cli.plugins.recordings_plugin import RecordingsPlugin
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
import pytest

from ... import testing_utils

_AGENT_NAME = 'recording_test_agent'


async def _make_invocation_context(test_case_dir: Path) -> InvocationContext:
  invocation_context = await testing_utils.create_invocation_context(
      agent=LlmAgent(name=_AGENT_NAME)
  )
  invocation_context.session.state['_adk_recordings_config'] = {
      'dir': str(test_case_dir),
      'user_message_index': 0,
      'streaming_mode': 'none',
  }
  return invocation_context


async def test_after_run_omits_http_options_from_the_recording_file(tmp_path):
  """http_options must not reach the recording file.

  The plugin keeps the live LlmRequest and writes it once the run ends, so
  whatever the model layer put on config.http_options is still there. headers
  commonly holds an Authorization bearer token, and base_url and extra_body
  carry caller-supplied credentials too; the recording file is committed as a
  conformance fixture.
  """
  plugin = RecordingsPlugin()
  invocation_context = await testing_utils.create_invocation_context(
      testing_utils.create_test_agent(name='agent_a')
  )
  invocation_context.session.state['_adk_recordings_config'] = {
      'dir': str(tmp_path),
      'user_message_index': 0,
      'streaming_mode': 'none',
  }
  callback_context = CallbackContext(invocation_context)
  llm_request = LlmRequest(
      model='fake-model',
      contents=[
          types.Content(role='user', parts=[types.Part(text='roll a die')])
      ],
      config=types.GenerateContentConfig(
          temperature=0.5,
          http_options=types.HttpOptions(
              headers={'Authorization': 'Bearer test-bearer-token'},
              base_url='https://proxy.example/?sig=test-signature',
              extra_body={'api_key': 'test-extra-body-key'},
          ),
      ),
  )

  await plugin.before_run_callback(invocation_context=invocation_context)
  await plugin.before_model_callback(
      callback_context=callback_context, llm_request=llm_request
  )
  await plugin.after_model_callback(
      callback_context=callback_context,
      llm_response=LlmResponse(
          content=types.Content(
              role='model', parts=[types.Part(text='rolled a 4')]
          )
      ),
  )
  await plugin.after_run_callback(invocation_context=invocation_context)

  written = (tmp_path / 'generated-recordings.yaml').read_text(encoding='utf-8')
  assert 'http_options' not in written
  assert 'test-bearer-token' not in written
  assert 'test-signature' not in written
  assert 'test-extra-body-key' not in written
  # Confirm that only http_options was dropped from the config.
  assert 'fake-model' in written
  assert 'roll a die' in written
  assert 'rolled a 4' in written
  assert 'temperature: 0.5' in written


@pytest.mark.asyncio
async def test_recordings_are_saved_on_run_completion(tmp_path: Path):
  invocation_context = await _make_invocation_context(tmp_path)
  plugin = RecordingsPlugin()

  await plugin.before_run_callback(invocation_context=invocation_context)
  await plugin.after_run_callback(invocation_context=invocation_context)

  assert (tmp_path / 'generated-recordings.yaml').exists()
  assert not plugin._invocation_states


@pytest.mark.asyncio
async def test_failure_to_save_recordings_is_surfaced(tmp_path: Path):
  invocation_context = await _make_invocation_context(tmp_path)
  plugin = RecordingsPlugin()

  await plugin.before_run_callback(invocation_context=invocation_context)

  with mock.patch(
      'google.adk.cli.plugins.recordings_plugin.dump_pydantic_to_yaml',
      autospec=True,
      side_effect=OSError('disk is full'),
  ):
    with pytest.raises(OSError, match='disk is full'):
      await plugin.after_run_callback(invocation_context=invocation_context)

  # The per-invocation state is still cleaned up.
  assert not plugin._invocation_states


@pytest.mark.asyncio
async def test_unsupported_streaming_mode_is_surfaced(tmp_path: Path):
  invocation_context = await _make_invocation_context(tmp_path)
  plugin = RecordingsPlugin()

  await plugin.before_run_callback(invocation_context=invocation_context)
  plugin._streaming_mode = 'bidi'

  with pytest.raises(ValueError, match='Unsupported streaming mode'):
    await plugin.after_run_callback(invocation_context=invocation_context)
