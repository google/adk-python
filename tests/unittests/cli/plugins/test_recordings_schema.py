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

"""Tests for the recordings schema used by the record/replay plugins."""

from google.adk.cli.plugins.recordings_schema import LlmRecording
from google.adk.cli.plugins.recordings_schema import Recording
from google.adk.cli.plugins.recordings_schema import Recordings
from google.adk.cli.plugins.recordings_schema import ToolRecording
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.utils.yaml_utils import dump_pydantic_to_yaml
from google.genai import types
from pydantic import ValidationError
import pytest
import yaml


def _tool_recording() -> Recording:
  return Recording(
      user_message_index=0,
      agent_name='dice_agent',
      tool_recording=ToolRecording(
          tool_call=types.FunctionCall(
              id='fc-1', name='roll_die', args={'sides': 6}
          ),
          tool_response=types.FunctionResponse(
              id='fc-1', name='roll_die', response={'result': 4}
          ),
      ),
  )


def _llm_recording() -> Recording:
  return Recording(
      user_message_index=1,
      agent_name='dice_agent',
      llm_recording=LlmRecording(
          llm_request=LlmRequest(
              model='fake-model',
              contents=[
                  types.Content(
                      role='user', parts=[types.Part(text='roll a die')]
                  )
              ],
          ),
          llm_responses=[
              LlmResponse(
                  content=types.Content(
                      role='model', parts=[types.Part(text='rolled a 4')]
                  )
              )
          ],
      ),
  )


def test_recordings_round_trip_through_yaml_preserves_recordings(tmp_path):
  """A file written by the recorder must reload into an equal model.

  The recorder writes with dump_pydantic_to_yaml (which drops None and
  default-valued fields) and the replayer reads it back with
  Recordings.model_validate, so anything lost in that pass is silently lost
  from a replay run.
  """
  recordings = Recordings(recordings=[_tool_recording(), _llm_recording()])
  path = tmp_path / 'generated-recordings.yaml'

  dump_pydantic_to_yaml(recordings, path, sort_keys=False)
  reloaded = Recordings.model_validate(
      yaml.safe_load(path.read_text(encoding='utf-8'))
  )

  assert reloaded == recordings
  # Guard against a degenerate match of two empty models: the fields the
  # replayer actually reads must survive the round trip.
  tool_recording = reloaded.recordings[0].tool_recording
  assert tool_recording.tool_call.name == 'roll_die'
  assert tool_recording.tool_call.args == {'sides': 6}
  assert tool_recording.tool_response.response == {'result': 4}
  llm_recording = reloaded.recordings[1].llm_recording
  assert llm_recording.llm_request.model == 'fake-model'
  assert llm_recording.llm_responses[0].content.parts[0].text == 'rolled a 4'


@pytest.mark.parametrize(
    'model,payload',
    [
        (Recordings, {'recordings': []}),
        (Recording, {'user_message_index': 0, 'agent_name': 'a'}),
        (LlmRecording, {'llm_responses': []}),
        (ToolRecording, {}),
    ],
)
def test_recording_models_reject_unknown_fields(model, payload):
  """extra='forbid' turns a mistyped key into an error, not silent data loss."""
  # Control: the payload without the stray key is accepted.
  assert isinstance(model.model_validate(dict(payload)), model)

  with pytest.raises(ValidationError) as exc_info:
    model.model_validate({**payload, 'not_a_real_field': 1})

  assert 'not_a_real_field' in str(exc_info.value)


def test_recordings_rejects_unknown_field_nested_in_a_recording():
  """The whole file is rejected, not just the offending recording."""
  with pytest.raises(ValidationError) as exc_info:
    Recordings.model_validate({
        'recordings': [{
            'user_message_index': 0,
            'agent_name': 'a',
            # Plural typo of `tool_recording`.
            'tool_recordings': {'tool_call': {'name': 'roll_die'}},
        }]
    })

  assert 'tool_recordings' in str(exc_info.value)


def test_recording_requires_the_fields_replay_filters_on():
  """user_message_index and agent_name select which recording is replayed."""
  with pytest.raises(ValidationError) as exc_info:
    Recording.model_validate({'tool_recording': None})

  message = str(exc_info.value)
  assert 'user_message_index' in message
  assert 'agent_name' in message
