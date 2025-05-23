# Copyright 2025 Google LLC
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

from google.adk.agents import Agent
from google.adk.agents import LiveRequestQueue
from google.adk.models import LlmResponse
from google.genai import types
import pytest

from .. import testing_utils


def test_streaming():
  response1 = LlmResponse(
      turn_complete=True,
  )

  mock_model = testing_utils.MockModel.create([response1])

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[],
  )

  runner = testing_utils.InMemoryRunner(
      root_agent=root_agent, response_modalities=['AUDIO']
  )
  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      blob=types.Blob(data=b'\x00\xFF', mime_type='audio/pcm')
  )
  res_events = runner.run_live(live_request_queue)

  assert res_events is not None, 'Expected a list of events, got None.'
  assert (
      len(res_events) > 0
  ), 'Expected at least one response, but got an empty list.'
